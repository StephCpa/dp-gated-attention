#!/usr/bin/env python
"""
dp_train_v2.py
==============
DP-SGD training script for GPT-2 with optional DP-aware gated attention.

Key differences from dp_sgd_train_minimal.py
---------------------------------------------
1. Uses DPGatedGPT2Attention (modeling_gpt2_gated_dp.py) with configurable
   init_gate_bias instead of the original GatedGPT2Attention.

2. ε-adaptive L1 gate regularisation
   At each step we add   λ(t) · gate_l1_loss   to the task loss, where
       λ(t) = gate_l1_lambda · (1 - t / total_steps) ** gate_l1_gamma
   This is a step-based proxy for the fraction of unused privacy budget:
   early in training (large noise) the penalty is strong, forcing the gate
   to remain sparse; late in training (noise has been "spent") the penalty
   relaxes so the gate can open up and maximise utility.
   λ = 0 disables regularisation (conditions A and B in the ablation).

3. Metrics CSV
   Every eval_every steps the script appends a row to a CSV file with:
   step, eval_loss, eval_ppl, gate_sparsity, eps_consumed, lambda_gate
   This allows post-hoc Pareto analysis without re-running experiments.

4. Single-seed, single-condition design
   This script trains one (condition, seed) pair.  The run_experiment_v2.py
   driver calls it in parallel across conditions and seeds.

Usage
-----
python dp_train_v2.py \
    --hf-model gpt2 \
    --tokenizer gpt2 \
    --dataset wikitext --dataset-config wikitext-2-raw-v1 \
    --seq-len 128 --batch-size 16 --steps 10000 \
    --noise-multiplier 0.516 --clip-norm 1.0 --lr 5e-5 \
    --gate-type headwise \
    --init-gate-bias -2.197 \
    --gate-l1-lambda 0.01 --gate-l1-gamma 1.0 \
    --delta 1e-5 --seed 7 \
    --metrics-csv results/condition_D_seed7.csv
"""

import argparse
import csv
import math
import os
import random
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Path setup so we can import from the same directory
# ---------------------------------------------------------------------------

def _ensure_pkg():
    repo_dir = Path(__file__).resolve().parent
    pkg_name = "gated_attention"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(repo_dir)]
        sys.modules[pkg_name] = pkg


# ---------------------------------------------------------------------------
# Dataset utilities (unchanged from v1)
# ---------------------------------------------------------------------------

def build_tokenizer(name_or_path, seq_len):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "[PAD]"
    tok.model_max_length = seq_len
    return tok


class TokenizedTextDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, text_field, seq_len):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.seq_len = seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx].get(self.text_field, "") or "[EMPTY]"
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
        mask = enc["attention_mask"].squeeze(0)
        labels = labels.masked_fill(mask == 0, -100)
        return input_ids, labels, mask


def build_dataloader(args, tokenizer, split, batch_size, shuffle, max_samples=0):
    from datasets import load_dataset
    ds = load_dataset(args.dataset, args.dataset_config, split=split)
    if max_samples and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))
    if shuffle:
        ds = ds.shuffle(seed=args.seed)

    def collate(examples):
        ids, labs, masks = zip(*examples)
        batch = {
            "input_ids": torch.stack(ids),
            "labels": torch.stack(labs),
            "attention_mask": torch.stack(masks),
        }
        return batch

    return DataLoader(
        TokenizedTextDataset(ds, tokenizer, args.text_field, args.seq_len),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
    )


def infinite_loader(loader):
    while True:
        yield from loader


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(args):
    _ensure_pkg()
    from transformers import AutoModelForCausalLM
    from modeling_gpt2_gated_dp import apply_dp_gated_attention

    model = AutoModelForCausalLM.from_pretrained(args.hf_model)
    model = apply_dp_gated_attention(
        model,
        gate_type=args.gate_type,
        init_gate_bias=args.init_gate_bias,
    )
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader, device, max_steps):
    if loader is None:
        return None, None
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_steps and i >= max_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch, use_cache=False)
            losses.append(out.loss.item())
    if was_training:
        model.train()
    if not losses:
        return None, None
    avg = sum(losses) / len(losses)
    try:
        ppl = math.exp(avg)
    except OverflowError:
        ppl = float("inf")
    return avg, ppl


# ---------------------------------------------------------------------------
# Gate sparsity measurement
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_sparsity(model, batch, device, threshold=0.1):
    """
    Measure average gate sparsity using the embedding output of the first
    batch element as proxy hidden states.

    We use the embedding output (before any attention layer) because:
    - It is available without a full forward pass
    - The gate projection is linear in hidden states, so this gives a
      valid lower-bound estimate of the operational sparsity
    """
    _ensure_pkg()
    from modeling_gpt2_gated_dp import model_gate_sparsity

    batch = {k: v.to(device) for k, v in batch.items()}
    # Get embedding output as sample hidden states
    with torch.no_grad():
        input_ids = batch["input_ids"][:1]  # single example
        hidden = model.transformer.wte(input_ids)  # (1, L, D)
        # Add positional embedding if available
        pos = torch.arange(input_ids.size(1), device=device).unsqueeze(0)
        hidden = hidden + model.transformer.wpe(pos)

    sparsity = model_gate_sparsity(model, hidden, threshold=threshold)
    return sparsity


# ---------------------------------------------------------------------------
# ε-adaptive L1 schedule
# ---------------------------------------------------------------------------

def gate_l1_lambda(step: int, total_steps: int, lam0: float, gamma: float) -> float:
    """
    λ(t) = λ₀ · (1 - t/T)^γ

    Starts at λ₀ when t=0 (high noise, strong sparsity pressure).
    Decays to 0 when t=T (budget spent, gate can open freely).
    """
    if total_steps <= 0 or lam0 <= 0.0:
        return 0.0
    frac = max(0.0, 1.0 - step / total_steps)
    return lam0 * (frac ** gamma)


# ---------------------------------------------------------------------------
# CSV metrics logging
# ---------------------------------------------------------------------------

class MetricsLogger:
    def __init__(self, path: str, condition: str, seed: int, eps_target: float):
        self.path = path
        self.condition = condition
        self.seed = seed
        self.eps_target = eps_target
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "condition", "seed", "eps_target",
                "step", "train_loss",
                "eval_loss", "eval_ppl",
                "gate_sparsity", "eps_consumed",
                "lambda_gate",
            ])

    def log(self, step, train_loss, eval_loss, eval_ppl,
            gate_sparsity, eps_consumed, lam):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.condition, self.seed, self.eps_target,
                step, f"{train_loss:.6f}",
                f"{eval_loss:.6f}" if eval_loss is not None else "",
                f"{eval_ppl:.4f}" if eval_ppl is not None else "",
                f"{gate_sparsity:.4f}",
                f"{eps_consumed:.4f}",
                f"{lam:.6f}",
            ])


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DP-SGD training v2")

    # Model
    parser.add_argument("--hf-model", required=True,
                        help="HuggingFace model name or local path (e.g. gpt2)")
    parser.add_argument("--gate-type", default="headwise",
                        choices=["headwise", "elementwise", "none"])
    parser.add_argument("--init-gate-bias", type=float, default=math.log(0.1 / 0.9),
                        help="Gate bias at init. -2.197 → sigmoid≈0.1 (DP-aware). "
                             "0.0 → sigmoid=0.5 (naive baseline).")

    # Data
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)

    # DP
    parser.add_argument("--noise-multiplier", type=float, required=True,
                        help="Opacus noise_multiplier σ. Use run_experiment_v2.py "
                             "to auto-search σ given a target ε.")
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--eps-target", type=float, default=None,
                        help="Target ε (informational only, used in CSV logging).")
    parser.add_argument("--poisson-sampling", action="store_true", default=True)
    parser.add_argument("--no-poisson-sampling",
                        action="store_false", dest="poisson_sampling")
    parser.add_argument("--grad-sample-mode", default="functorch",
                        choices=["hooks", "functorch"])
    parser.add_argument("--max-physical-batch-size", type=int, default=0)

    # Optimiser
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--steps", type=int, default=10000)

    # Gate L1 regularisation
    parser.add_argument("--gate-l1-lambda", type=float, default=0.0,
                        help="Initial L1 gate regularisation weight λ₀. "
                             "0 disables (conditions A, B, C). "
                             "Typical: 0.01–0.1 for condition D.")
    parser.add_argument("--gate-l1-gamma", type=float, default=1.0,
                        help="Decay exponent γ for λ(t) = λ₀·(1-t/T)^γ.")

    # Logging & eval
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--condition", type=str, default="unknown",
                        help="Condition label written to the CSV (e.g. 'D_dp_init_l1').")
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--sparsity-threshold", type=float, default=0.1)
    parser.add_argument("--metrics-csv", type=str, default="metrics.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tokenizer = build_tokenizer(args.tokenizer, args.seq_len)

    train_loader = build_dataloader(
        args, tokenizer, split=args.dataset_split,
        batch_size=args.batch_size, shuffle=True,
        max_samples=args.max_samples,
    )
    try:
        eval_loader = build_dataloader(
            args, tokenizer, split=args.eval_split,
            batch_size=args.batch_size, shuffle=False,
        )
    except Exception as e:
        print(f"Warning: could not build eval loader: {e}")
        eval_loader = None

    model = build_model(args).to(args.device)

    # Untie weights for Opacus compatibility (GPT-2 ties lm_head to wte)
    if hasattr(model, "lm_head") and hasattr(model, "transformer"):
        embed_weight = model.transformer.wte.weight
        if model.lm_head.weight is embed_weight:
            model.lm_head.weight = torch.nn.Parameter(
                embed_weight.detach().clone()
            )
            model.config.tie_word_embeddings = False

    # Convert Conv1D → nn.Linear for Opacus compatibility
    from modeling_gpt2_gated_dp import convert_conv1d_to_linear
    model = convert_conv1d_to_linear(model)

    # Fix remaining Opacus issues (unsupported layers, etc.)
    from opacus.validators import ModuleValidator
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)

    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Opacus PrivacyEngine
    from opacus import PrivacyEngine
    privacy_engine = PrivacyEngine(accountant="rdp")
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.clip_norm,
        poisson_sampling=args.poisson_sampling,
        grad_sample_mode=args.grad_sample_mode,
    )

    if args.max_physical_batch_size and args.max_physical_batch_size > 0:
        from opacus.utils.batch_memory_manager import BatchMemoryManager
        loader_ctx = BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=args.max_physical_batch_size,
            optimizer=optimizer,
        )
    else:
        loader_ctx = nullcontext(train_loader)

    logger = MetricsLogger(
        path=args.metrics_csv,
        condition=args.condition,
        seed=args.seed,
        eps_target=args.eps_target or 0.0,
    )

    step = 0
    recent_losses = []

    with loader_ctx as dp_loader:
        data_iter = infinite_loader(dp_loader)

        while step < args.steps:
            batch = next(data_iter)
            batch = {k: v.to(args.device) for k, v in batch.items()}
            if batch["input_ids"].size(0) == 0:
                continue

            # ---------------------------------------------------------------
            # Forward pass with optional gate L1 regularisation
            # ---------------------------------------------------------------
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss

            lam = gate_l1_lambda(step, args.steps, args.gate_l1_lambda, args.gate_l1_gamma)
            if lam > 0.0 and args.gate_type != "none":
                _ensure_pkg()
                from modeling_gpt2_gated_dp import model_gate_l1_loss
                # Adding L1 to task loss before backward is DP-safe because
                # L1 regularisation only depends on current parameters (not data).
                # Post-processing theorem guarantees DP is preserved.
                loss = loss + lam * model_gate_l1_loss(model)

            # ---------------------------------------------------------------
            # Backward + DP-SGD step (Opacus handles clip + noise)
            # ---------------------------------------------------------------
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            recent_losses.append(loss.item())

            # ---------------------------------------------------------------
            # Logging
            # ---------------------------------------------------------------
            if step % args.print_every == 0:
                avg_train = sum(recent_losses[-args.print_every:]) / min(len(recent_losses), args.print_every)
                eps_consumed = privacy_engine.get_epsilon(delta=args.delta)
                print(
                    f"step={step:06d}  loss={avg_train:.4f}  "
                    f"ε={eps_consumed:.3f}  λ={lam:.5f}"
                )

            if args.eval_every and step % args.eval_every == 0:
                eval_loss, eval_ppl = evaluate(
                    model, eval_loader, args.device, args.eval_steps
                )
                eps_consumed = privacy_engine.get_epsilon(delta=args.delta)

                # Gate sparsity: use current training batch as probe
                if args.gate_type != "none":
                    sparsity = measure_sparsity(
                        model, batch, args.device, threshold=args.sparsity_threshold
                    )
                else:
                    sparsity = 0.0

                avg_train = sum(recent_losses[-args.eval_every:]) / min(len(recent_losses), args.eval_every)

                logger.log(
                    step=step,
                    train_loss=avg_train,
                    eval_loss=eval_loss,
                    eval_ppl=eval_ppl,
                    gate_sparsity=sparsity,
                    eps_consumed=eps_consumed,
                    lam=lam,
                )

                ppl_str = f"{eval_ppl:.3f}" if eval_ppl is not None else "n/a"
                print(
                    f"  [eval] step={step:06d}  eval_ppl={ppl_str}  "
                    f"gate_sparsity={sparsity:.3f}  ε_consumed={eps_consumed:.3f}"
                )

    # Final epsilon
    eps_final = privacy_engine.get_epsilon(delta=args.delta)
    print(f"\nTraining complete. ε={eps_final:.4f}  δ={args.delta}")
    print(f"Metrics written to: {args.metrics_csv}")


if __name__ == "__main__":
    main()
