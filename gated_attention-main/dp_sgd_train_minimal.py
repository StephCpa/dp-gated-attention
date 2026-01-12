#!/usr/bin/env python
import argparse
import math
import random
import types
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def _ensure_gated_attention_package():
    repo_dir = Path(__file__).resolve().parent
    pkg_name = "gated_attention"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(repo_dir)]
        sys.modules[pkg_name] = pkg


def _load_gated_attention_modules():
    _ensure_gated_attention_package()

    from gated_attention.configuration_qwen3 import Qwen3Config  # type: ignore
    from gated_attention.modeling_qwen3 import Qwen3ForCausalLM  # type: ignore

    return Qwen3Config, Qwen3ForCausalLM


def build_tokenizer(name_or_path, seq_len):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.model_max_length = seq_len
    return tokenizer


class TokenizedTextDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, text_field, seq_len):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.seq_len = seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        text = sample.get(self.text_field, "")
        if not isinstance(text, str) or not text.strip():
            text = "[EMPTY]"
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.seq_len,
            return_attention_mask=True,
        )
        input_ids = enc["input_ids"].squeeze(0)
        labels = input_ids.clone()
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            return input_ids, labels
        return input_ids, labels, attention_mask.squeeze(0)


def build_real_dataloader(
    args,
    tokenizer,
    *,
    dataset_split=None,
    batch_size=None,
    shuffle=None,
    max_samples=None,
):
    from datasets import load_dataset

    split = args.dataset_split if dataset_split is None else dataset_split
    batch_size = args.batch_size if batch_size is None else batch_size
    shuffle = args.shuffle if shuffle is None else shuffle
    max_samples = args.max_samples if max_samples is None else max_samples

    dataset = load_dataset(args.dataset, args.dataset_config, split=split)
    if max_samples and max_samples > 0:
        dataset = dataset.select(range(max_samples))
    if shuffle:
        dataset = dataset.shuffle(seed=args.seed)

    tokenized_dataset = TokenizedTextDataset(dataset, tokenizer, args.text_field, args.seq_len)

    def collate(examples):
        if len(examples[0]) == 2:
            input_ids, labels = zip(*examples)
            batch = {
                "input_ids": torch.stack(input_ids, dim=0),
                "labels": torch.stack(labels, dim=0),
            }
            return batch
        input_ids, labels, attention_mask = zip(*examples)
        return {
            "input_ids": torch.stack(input_ids, dim=0),
            "labels": torch.stack(labels, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
        }

    return DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
    )


def make_synthetic_batch(batch_size, seq_len, vocab_size, device):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "labels": labels}


def normalize_batch(batch):
    if isinstance(batch, dict):
        return batch
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            input_ids, labels = batch
            return {"input_ids": input_ids, "labels": labels}
        if len(batch) == 3:
            input_ids, labels, attention_mask = batch
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def batch_to_device(batch, device):
    batch = normalize_batch(batch)
    return {k: v.to(device) for k, v in batch.items()}


def evaluate(model, data_loader, device, max_steps):
    if data_loader is None:
        return None, None
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            if max_steps and idx >= max_steps:
                break
            batch = batch_to_device(batch, device)
            if batch["input_ids"].size(0) == 0:
                continue
            outputs = model(**batch, use_cache=False)
            losses.append(outputs.loss.item())
    if was_training:
        model.train()
    if not losses:
        return None, None
    avg_loss = sum(losses) / len(losses)
    try:
        ppl = math.exp(avg_loss)
    except OverflowError:
        ppl = float("inf")
    return avg_loss, ppl


def per_sample_grads(model, batch):
    params = [p for p in model.parameters() if p.requires_grad]
    grads = []
    batch_size = batch["input_ids"].size(0)
    for i in range(batch_size):
        model.zero_grad(set_to_none=True)
        single = {k: v[i : i + 1] for k, v in batch.items()}
        outputs = model(**single, use_cache=False)
        loss = outputs.loss
        g = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
        grads.append([gi.detach() for gi in g])
    return grads, params


def clip_and_aggregate(grads, clip_norm):
    clipped = []
    for g_list in grads:
        total_norm = torch.sqrt(sum((g.norm() ** 2) for g in g_list))
        scale = min(1.0, clip_norm / (total_norm + 1e-6))
        clipped.append([g * scale for g in g_list])

    agg = []
    for p_idx in range(len(clipped[0])):
        stacked = torch.stack([g[p_idx] for g in clipped], dim=0)
        agg.append(stacked.mean(dim=0))
    return agg


def dp_sgd_step(model, batch, clip_norm, noise_multiplier, lr):
    grads, params = per_sample_grads(model, batch)
    agg_grads = clip_and_aggregate(grads, clip_norm)

    with torch.no_grad():
        for p, g in zip(params, agg_grads):
            noise = torch.normal(
                mean=0.0,
                std=noise_multiplier * clip_norm,
                size=g.shape,
                device=g.device,
            )
            p -= lr * (g + noise)


def infinite_dataloader(data_loader):
    while True:
        for batch in data_loader:
            yield batch


def estimate_epsilon_rdp(steps, sample_rate, noise_multiplier, delta):
    from opacus.accountants import RDPAccountant

    accountant = RDPAccountant()
    for _ in range(steps):
        accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    return accountant.get_epsilon(delta)


def build_model(
    vocab_size,
    hidden_size,
    intermediate_size,
    num_layers,
    num_heads,
    head_dim,
    max_position_embeddings,
    gate_type,
):
    Qwen3Config, Qwen3ForCausalLM = _load_gated_attention_modules()

    config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        head_dim=head_dim,
        max_position_embeddings=max_position_embeddings,
        use_qk_norm=False,
        attention_dropout=0.0,
        headwise_attn_output_gate=gate_type == "headwise",
        elementwise_attn_output_gate=gate_type == "elementwise",
    )
    if not hasattr(config, "qkv_bias"):
        config.qkv_bias = False
    return Qwen3ForCausalLM(config)


def build_gpt2_model(args):
    from transformers import AutoConfig, AutoModelForCausalLM
    _ensure_gated_attention_package()
    from gated_attention.modeling_gpt2_gated import apply_gpt2_gated_attention

    if not args.hf_model:
        raise ValueError("GPT2 mode requires --hf-model pointing to a local or HF model name/path.")

    config = AutoConfig.from_pretrained(args.hf_model)
    config.headwise_attn_output_gate = args.gate_type == "headwise"
    config.elementwise_attn_output_gate = args.gate_type == "elementwise"

    model = AutoModelForCausalLM.from_pretrained(args.hf_model, config=config)
    model = apply_gpt2_gated_attention(model, gate_type=args.gate_type)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3-minimal", choices=["qwen3-minimal", "gpt2"])
    parser.add_argument("--hf-model", type=str, default=None)
    parser.add_argument("--gate-type", type=str, default="headwise", choices=["headwise", "elementwise", "none"])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--noise-multiplier", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=5)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-dataset-split", type=str, default="validation")
    parser.add_argument("--eval-max-samples", type=int, default=0)
    parser.add_argument("--use-opacus", action="store_true")
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--sample-rate", type=float, default=None)
    parser.add_argument("--no-dataset", action="store_true")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.no_dataset:
        tokenizer = None
        vocab_size = args.vocab_size
        data_loader = None
        eval_loader = None
    else:
        tokenizer = build_tokenizer(args.tokenizer, args.seq_len)
        vocab_size = len(tokenizer)
        data_loader = build_real_dataloader(args, tokenizer)
        eval_loader = None
        if args.eval_every and args.eval_every > 0:
            eval_batch_size = args.eval_batch_size or args.batch_size
            try:
                eval_loader = build_real_dataloader(
                    args,
                    tokenizer,
                    dataset_split=args.eval_dataset_split,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    max_samples=args.eval_max_samples,
                )
            except Exception as exc:
                print(f"warning: failed to build eval dataloader: {exc}")
                eval_loader = None

    if args.model == "gpt2":
        model = build_gpt2_model(args).to(args.device)
        vocab_size = getattr(model.config, "vocab_size", vocab_size)
    else:
        model = build_model(
            vocab_size=vocab_size,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            max_position_embeddings=args.seq_len,
            gate_type=args.gate_type,
        ).to(args.device)

    model.train()

    if args.use_opacus:
        from opacus import PrivacyEngine

        if data_loader is None:
            raise ValueError("Opacus mode requires a real dataset. Remove --no-dataset.")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        privacy_engine = PrivacyEngine(accountant="rdp")
        model, optimizer, data_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.clip_norm,
        )

        step = 0
        for batch in data_loader:
            batch = batch_to_device(batch, args.device)
            if batch["input_ids"].size(0) == 0:
                continue
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            if step % args.print_every == 0:
                print(f"step={step:03d} loss={loss.item():.4f}")
            if eval_loader is not None and args.eval_every and step % args.eval_every == 0:
                eval_loss, eval_ppl = evaluate(
                    model, eval_loader, args.device, args.eval_steps
                )
                if eval_loss is not None:
                    print(
                        f"eval_step={step:03d} eval_loss={eval_loss:.4f} eval_ppl={eval_ppl:.2f}"
                    )
            if step >= args.steps:
                break

        epsilon = privacy_engine.get_epsilon(delta=args.delta)
        print(f"epsilon_rdp(opacus)={epsilon:.3f} delta={args.delta}")
        return

    if data_loader is None:
        data_iter = None
    else:
        data_iter = infinite_dataloader(data_loader)

    if args.sample_rate is None:
        if data_loader is None:
            sample_rate = 1.0
        else:
            dataset_len = len(data_loader.dataset)
            if dataset_len <= 0:
                raise ValueError("dataset length is zero; provide --sample-rate.")
            sample_rate = min(1.0, args.batch_size / dataset_len)
    else:
        sample_rate = args.sample_rate

    for step in range(1, args.steps + 1):
        if data_iter is None:
            batch = make_synthetic_batch(args.batch_size, args.seq_len, vocab_size, args.device)
        else:
            batch = next(data_iter)
            batch = batch_to_device(batch, args.device)

        with torch.no_grad():
            loss = model(**batch, use_cache=False).loss.item()

        dp_sgd_step(
            model,
            batch=batch,
            clip_norm=args.clip_norm,
            noise_multiplier=args.noise_multiplier,
            lr=args.lr,
        )

        if step % args.print_every == 0:
            print(f"step={step:03d} loss={loss:.4f}")
        if eval_loader is not None and args.eval_every and step % args.eval_every == 0:
            eval_loss, eval_ppl = evaluate(
                model, eval_loader, args.device, args.eval_steps
            )
            if eval_loss is not None:
                print(
                    f"eval_step={step:03d} eval_loss={eval_loss:.4f} eval_ppl={eval_ppl:.2f}"
                )

    epsilon = estimate_epsilon_rdp(
        steps=args.steps,
        sample_rate=sample_rate,
        noise_multiplier=args.noise_multiplier,
        delta=args.delta,
    )
    print(f"epsilon_rdp(manual)={epsilon:.3f} delta={args.delta} sample_rate={sample_rate:.6f}")


if __name__ == "__main__":
    main()
