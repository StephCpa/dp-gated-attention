#!/usr/bin/env python
"""
validate_non_dp_sparsity.py
============================
Quick validation: does gated attention learn selectivity WITHOUT DP?

If gate_sparsity stays 0 even without DP noise, then Path C (transfer
frozen gates) is not viable — there's no selectivity to transfer.

If gate_sparsity > 0.1 emerges, Path C is worth pursuing.

Runs standard (non-DP) fine-tuning of gated GPT-2 on WikiText-2.
Expected runtime: ~30 min on a single GPU.

Usage
-----
CUDA_VISIBLE_DEVICES=5 python validate_non_dp_sparsity.py --steps 5000
"""

import argparse
import math
import random
import sys
import types
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def _ensure_pkg():
    repo_dir = Path(__file__).resolve().parent
    pkg_name = "gated_attention"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(repo_dir)]
        sys.modules[pkg_name] = pkg


def build_tokenizer(name, seq_len):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "[PAD]"
    tok.model_max_length = seq_len
    return tok


class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, seq_len):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx].get("text", "") or "[EMPTY]"
        enc = self.tokenizer(
            text, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.seq_len, return_attention_mask=True,
        )
        ids = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)
        labels = ids.clone().masked_fill(mask == 0, -100)
        return ids, labels, mask


def collate(examples):
    ids, labs, masks = zip(*examples)
    input_ids = torch.stack(ids)
    bsz, seq_len = input_ids.shape
    return {
        "input_ids": input_ids,
        "labels": torch.stack(labs),
        "attention_mask": torch.stack(masks),
        "position_ids": torch.arange(seq_len).unsqueeze(0).expand(bsz, -1),
    }


@torch.no_grad()
def measure_sparsity(model, batch, device, threshold=0.1):
    _ensure_pkg()
    from modeling_gpt2_gated_dp import model_gate_sparsity
    input_ids = batch["input_ids"][:1].to(device)
    hidden = model.transformer.wte(input_ids)
    pos = torch.arange(input_ids.size(1), device=device).unsqueeze(0)
    hidden = hidden + model.transformer.wpe(pos)
    return model_gate_sparsity(model, hidden, threshold=threshold)


def evaluate(model, loader, device, max_steps=50):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch, use_cache=False)
            losses.append(out.loss.item())
    model.train()
    if not losses:
        return None, None
    avg = sum(losses) / len(losses)
    return avg, math.exp(avg) if avg < 20 else float("inf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", default="gpt2")
    parser.add_argument("--gate-type", default="headwise")
    parser.add_argument("--init-gate-bias", type=float, default=0.5)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    _ensure_pkg()
    from transformers import AutoModelForCausalLM
    from modeling_gpt2_gated_dp import apply_dp_gated_attention, convert_conv1d_to_linear

    tokenizer = build_tokenizer(args.hf_model, args.seq_len)

    from datasets import load_dataset
    train_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    val_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

    train_loader = DataLoader(
        TokenizedDataset(train_ds, tokenizer, args.seq_len),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        TokenizedDataset(val_ds, tokenizer, args.seq_len),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate,
    )

    # Build model with gated attention (NO DP)
    model = AutoModelForCausalLM.from_pretrained(args.hf_model)
    model = apply_dp_gated_attention(
        model, gate_type=args.gate_type, init_gate_bias=args.init_gate_bias,
    )
    model = convert_conv1d_to_linear(model)
    model = model.to(args.device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("=" * 60)
    print("Non-DP Gated Attention Sparsity Validation")
    print("=" * 60)
    print(f"Model: {args.hf_model}  Gate: {args.gate_type}  Bias: {args.init_gate_bias}")
    print(f"NO DP — standard fine-tuning")
    print()

    step = 0
    data_iter = iter(train_loader)
    recent_losses = []

    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        batch = {k: v.to(args.device) for k, v in batch.items()}
        out = model(**batch, use_cache=False)
        loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step += 1
        recent_losses.append(loss.item())

        if step % 50 == 0:
            avg = sum(recent_losses[-50:]) / min(len(recent_losses), 50)
            print(f"step={step:05d}  train_loss={avg:.4f}")

        if step % args.eval_every == 0:
            eval_loss, eval_ppl = evaluate(model, val_loader, args.device)
            sparsity = measure_sparsity(model, batch, args.device)

            ppl_str = f"{eval_ppl:.2f}" if eval_ppl else "n/a"
            print(f"  >>> [eval] step={step:05d}  ppl={ppl_str}  "
                  f"gate_sparsity={sparsity:.4f}  <<<")

            if sparsity > 0.05:
                print(f"\n  *** SPARSITY DETECTED: {sparsity:.4f} > 0.05 ***")
                print(f"  *** Path C is viable — gate learns selectivity without DP ***\n")

    # Final eval
    eval_loss, eval_ppl = evaluate(model, val_loader, args.device)
    sparsity = measure_sparsity(model, batch, args.device)

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"  eval_ppl:       {eval_ppl:.2f}")
    print(f"  gate_sparsity:  {sparsity:.4f}")
    print()

    if sparsity > 0.05:
        print("VERDICT: Gate learned selectivity without DP.")
        print("         Path C (frozen gate transfer) is VIABLE.")
        print("         Next: freeze gates, DP fine-tune body, compare to baseline.")
    else:
        print("VERDICT: Gate did NOT learn selectivity even without DP.")
        print("         Path C is NOT viable for this model/task.")
        print("         Recommend: stop this line (Path A) or try larger model.")


if __name__ == "__main__":
    main()
