#!/usr/bin/env python
"""
dp_train_v3_hetero.py
=====================
Heterogeneous-noise DP-SGD training (Exp 2 of v3 framework).

Implements per-parameter-group noise multipliers for DP-SGD, motivated
by the refined theory:

    Δ₂(W_gate) / Δ₂(W_body) ≤ sup σ'(x) = 1/4

The gate's natural sensitivity is bounded by the sigmoid derivative,
so it can be trained with a smaller noise multiplier σ_g < σ_b while
keeping the joint privacy budget within target.

Implementation
--------------
We bypass Opacus's PrivacyEngine entirely. Instead we use GradSampleModule
to compute per-sample gradients, then manually:
  1. Compute global per-sample norm across ALL parameters (body + gate)
  2. Clip per-sample with global clip_norm C
  3. For each parameter, aggregate clipped grads
  4. Add noise: σ_b·C for body parameters, σ_g·C for the gate slice of c_attn

For c_attn (which contains both body Q/K/V rows and gate rows), noise is
applied per-slice: rows [embed:embed+gate_dim] get σ_g, others get σ_b.

Privacy accounting
------------------
We track two parallel RDP accountants (one per noise level) and combine
them via additive RDP composition:

    ε_total(α) = ε_body(α) + ε_gate(α)

This is the correct formula for two Gaussian mechanisms that share input
data but use independent noise (they form a single combined mechanism
whose outputs are jointly observable).

Usage
-----
python dp_train_v3_hetero.py \
    --hf-model gpt2 --tokenizer gpt2 \
    --noise-multiplier-body 0.5 --noise-multiplier-gate 0.25 \
    --clip-norm 1.0 --lr 5e-5 --steps 10000 \
    --gate-type headwise --init-gate-bias 0.5 \
    --metrics-csv results/v3_hetero/k0.5_eps3_seed7.csv
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
# Dataset utilities
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
            text, return_tensors="pt", padding="max_length", truncation=True,
            max_length=self.seq_len, return_attention_mask=True,
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
        input_ids = torch.stack(ids)
        bsz, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(bsz, -1)
        return {
            "input_ids": input_ids,
            "labels": torch.stack(labs),
            "attention_mask": torch.stack(masks),
            "position_ids": position_ids,
        }

    return DataLoader(
        TokenizedTextDataset(ds, tokenizer, args.text_field, args.seq_len),
        batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
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
    from modeling_gpt2_gated_dp import (
        apply_dp_gated_attention,
        convert_conv1d_to_linear,
    )

    model = AutoModelForCausalLM.from_pretrained(args.hf_model)
    model = apply_dp_gated_attention(
        model, gate_type=args.gate_type, init_gate_bias=args.init_gate_bias,
    )
    return model


# ---------------------------------------------------------------------------
# Locate gate slices in c_attn weight/bias for per-slice noise
# ---------------------------------------------------------------------------

def collect_gate_slices(model):
    """
    Walk the model and return a dict mapping id(param) -> (start, end)
    indicating the row range in the parameter that corresponds to gate.

    After Conv1D → nn.Linear conversion, c_attn has shape:
        weight: (3*embed + gate_dim, embed)  [out_features, in_features]
        bias:   (3*embed + gate_dim,)
    The gate slice occupies rows [embed : embed + gate_dim] in dim 0.
    """
    _ensure_pkg()
    from modeling_gpt2_gated_dp import DPGatedGPT2Attention

    gate_slices = {}
    for module in model.modules():
        if isinstance(module, DPGatedGPT2Attention) and module._fused_gate:
            q_end, gate_end = module._gate_slice()
            # After Conv1D → Linear, weight is (out, in) so the gate rows
            # are in dim 0. For Conv1D (pre-conversion) it would be dim 1.
            # We assume conversion has been done.
            gate_slices[id(module.c_attn.weight)] = (q_end, gate_end)
            if module.c_attn.bias is not None:
                gate_slices[id(module.c_attn.bias)] = (q_end, gate_end)
    return gate_slices


# ---------------------------------------------------------------------------
# Heterogeneous DP-SGD step (manual, no PrivacyEngine)
# ---------------------------------------------------------------------------

def hetero_dp_step(
    model, batch, optimizer, *,
    sigma_b: float, sigma_g: float,
    clip_norm_body: float, clip_norm_gate: float,
    gate_slices: dict,
):
    """
    One step of heterogeneous-noise DP-SGD with PER-GROUP CLIPPING.

    This correctly implements direction B: body and gate are treated as
    two independent Gaussian mechanisms with separate sensitivities
    (C_body, C_gate) and noise scales (σ_b, σ_g).

      1. Forward + backward (GradSampleModule populates p.grad_sample)
      2. Compute per-sample norms SEPARATELY for body and gate partitions
      3. Clip body partition with C_body, gate partition with C_gate
      4. Aggregate, add noise:
         - body params and non-gate slices of c_attn: N(0, σ_b² · C_body²)
         - gate slice of c_attn:                     N(0, σ_g² · C_gate²)
      5. optimizer.step()

    Privacy analysis: body sensitivity = C_body, gate sensitivity = C_gate.
    Joint RDP per step (order α) =
        α/2 · [(C_body/σ_b)² + (C_gate/σ_g)²] · (subsampling factors)
    which reduces to the standard per-mechanism Opacus accounting if we
    record noise_multiplier = σ_b for body and σ_g for gate independently
    (because Opacus normalizes sensitivity to 1 internally).
    """
    outputs = model(**batch, use_cache=False)
    loss = outputs.loss
    loss.backward()

    batch_size = batch["input_ids"].size(0)
    device = loss.device

    # 1. Split per-sample grad norms into body and gate partitions
    body_norm_sq = torch.zeros(batch_size, device=device)
    gate_norm_sq = torch.zeros(batch_size, device=device)
    grad_params = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        gs = getattr(p, "grad_sample", None)
        if gs is None:
            continue
        grad_params.append(p)

        if id(p) in gate_slices:
            start, end = gate_slices[id(p)]
            # Gate partition: slice [start:end] along dim 1 of grad_sample
            # (grad_sample has batch as dim 0, then the param dimensions)
            if gs.dim() >= 3:
                gate_part = gs[:, start:end]
                body_part = torch.cat([gs[:, :start], gs[:, end:]], dim=1)
            else:  # 2-D grad_sample: batch + 1 param dim (bias)
                gate_part = gs[:, start:end]
                body_part = torch.cat([gs[:, :start], gs[:, end:]], dim=1)
            gate_norm_sq += gate_part.reshape(batch_size, -1).pow(2).sum(dim=1)
            body_norm_sq += body_part.reshape(batch_size, -1).pow(2).sum(dim=1)
        else:
            body_norm_sq += gs.reshape(batch_size, -1).pow(2).sum(dim=1)

    body_norm = body_norm_sq.sqrt()
    gate_norm = gate_norm_sq.sqrt()

    # 2. Per-group clip factors
    body_clip_factor = (clip_norm_body / (body_norm + 1e-6)).clamp(max=1.0)
    gate_clip_factor = (clip_norm_gate / (gate_norm + 1e-6)).clamp(max=1.0)

    # 3. Aggregate + per-slice noise
    for p in grad_params:
        gs = p.grad_sample

        if id(p) in gate_slices:
            start, end = gate_slices[id(p)]
            # Apply separate clip factors to body and gate slices
            body_cf_shape = [batch_size] + [1] * (gs.dim() - 1)
            gate_cf_shape = [batch_size] + [1] * (gs.dim() - 1)
            body_cf = body_clip_factor.view(*body_cf_shape)
            gate_cf = gate_clip_factor.view(*gate_cf_shape)

            # Clone to avoid in-place aliasing issues
            clipped = gs.clone()
            # Apply gate_cf to gate rows, body_cf to others
            clipped[:, start:end] = gs[:, start:end] * gate_cf
            if start > 0:
                clipped[:, :start] = gs[:, :start] * body_cf
            if end < gs.shape[1]:
                clipped[:, end:] = gs[:, end:] * body_cf

            aggregated = clipped.sum(dim=0)

            # Noise: σ_b · C_body for body rows, σ_g · C_gate for gate rows
            noise = torch.randn_like(aggregated) * (sigma_b * clip_norm_body)
            if p.dim() >= 2:
                gate_shape = (end - start,) + tuple(p.shape[1:])
                gate_noise = torch.randn(
                    *gate_shape, device=p.device, dtype=p.dtype,
                ) * (sigma_g * clip_norm_gate)
                noise[start:end] = gate_noise
            else:
                gate_noise = torch.randn(
                    end - start, device=p.device, dtype=p.dtype,
                ) * (sigma_g * clip_norm_gate)
                noise[start:end] = gate_noise
        else:
            # Pure body param
            cf_shape = [batch_size] + [1] * (gs.dim() - 1)
            clipped = gs * body_clip_factor.view(*cf_shape)
            aggregated = clipped.sum(dim=0)
            noise = torch.randn_like(aggregated) * (sigma_b * clip_norm_body)

        p.grad = (aggregated + noise) / batch_size
        p.grad_sample = None

    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


# ---------------------------------------------------------------------------
# Heterogeneous RDP accountant
# ---------------------------------------------------------------------------

class HeteroRDPAccountant:
    """
    Track two parallel Gaussian mechanisms (body, gate) and report joint
    (ε, δ)-DP via additive RDP composition.
    """

    def __init__(self):
        from opacus.accountants import RDPAccountant
        self.body = RDPAccountant()
        self.gate = RDPAccountant()

    def step(self, sample_rate, sigma_b, sigma_g):
        self.body.step(noise_multiplier=sigma_b, sample_rate=sample_rate)
        self.gate.step(noise_multiplier=sigma_g, sample_rate=sample_rate)

    def get_epsilon(self, delta):
        # Use a common set of orders, sum RDP at each order, take min
        # over orders after converting to (ε, δ).
        from opacus.accountants.analysis.rdp import (
            compute_rdp, get_privacy_spent,
        )
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

        body_history = self.body.history
        gate_history = self.gate.history

        # Each history entry is (noise_multiplier, sample_rate, num_steps)
        joint_rdp = [0.0] * len(orders)
        for nm, sr, n in body_history:
            rdp_b = compute_rdp(q=sr, noise_multiplier=nm, steps=n, orders=orders)
            joint_rdp = [a + b for a, b in zip(joint_rdp, rdp_b)]
        for nm, sr, n in gate_history:
            rdp_g = compute_rdp(q=sr, noise_multiplier=nm, steps=n, orders=orders)
            joint_rdp = [a + b for a, b in zip(joint_rdp, rdp_g)]

        eps, _ = get_privacy_spent(orders=orders, rdp=joint_rdp, delta=delta)
        return eps


# ---------------------------------------------------------------------------
# Evaluation + sparsity measurement
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


@torch.no_grad()
def measure_sparsity(model, batch, device, threshold=0.1):
    _ensure_pkg()
    from modeling_gpt2_gated_dp import model_gate_sparsity
    batch = {k: v.to(device) for k, v in batch.items()}
    input_ids = batch["input_ids"][:1]
    hidden = model.transformer.wte(input_ids)
    pos = torch.arange(input_ids.size(1), device=device).unsqueeze(0)
    hidden = hidden + model.transformer.wpe(pos)
    return model_gate_sparsity(model, hidden, threshold=threshold)


# ---------------------------------------------------------------------------
# Metrics CSV
# ---------------------------------------------------------------------------

class MetricsLogger:
    def __init__(self, path, condition, seed, eps_target):
        self.path = path
        self.condition = condition
        self.seed = seed
        self.eps_target = eps_target
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow([
                "condition", "seed", "eps_target",
                "step", "train_loss", "eval_loss", "eval_ppl",
                "gate_sparsity", "eps_consumed_total",
                "sigma_body", "sigma_gate",
                "clip_body", "clip_gate",
            ])

    def log(self, step, train_loss, eval_loss, eval_ppl, gate_sparsity,
            eps_consumed_total, sigma_b, sigma_g, clip_b, clip_g):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([
                self.condition, self.seed, self.eps_target,
                step, f"{train_loss:.6f}",
                f"{eval_loss:.6f}" if eval_loss is not None else "",
                f"{eval_ppl:.4f}" if eval_ppl is not None else "",
                f"{gate_sparsity:.4f}",
                f"{eps_consumed_total:.4f}",
                f"{sigma_b:.4f}", f"{sigma_g:.4f}",
                f"{clip_b:.4f}", f"{clip_g:.4f}",
            ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DP-SGD training v3 with hetero noise")

    # Model
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--gate-type", default="headwise",
                        choices=["headwise", "elementwise", "none"])
    parser.add_argument("--init-gate-bias", type=float, default=0.5,
                        help="Default 0.5 (best from Exp 1 phase scan).")

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
    parser.add_argument("--noise-multiplier-body", type=float, required=True,
                        help="σ_b applied to body parameters")
    parser.add_argument("--noise-multiplier-gate", type=float, required=True,
                        help="σ_g applied to the gate slice of c_attn")
    parser.add_argument("--clip-norm-body", type=float, default=1.0,
                        help="Per-sample clip norm C_body for body params")
    parser.add_argument("--clip-norm-gate", type=float, default=None,
                        help="Per-sample clip norm C_gate for gate slice "
                             "(default: same as C_body). Set to e.g. 0.25·C_body "
                             "to exploit the natural σ' ≤ 0.25 sensitivity bound.")
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--eps-target", type=float, default=None)
    parser.add_argument("--sample-rate", type=float, default=None,
                        help="Override Poisson sample rate (default: B/N).")

    # Optimiser
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--steps", type=int, default=10000)

    # Logging & eval
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--condition", type=str, default="hetero")
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--sparsity-threshold", type=float, default=0.1)
    parser.add_argument("--metrics-csv", type=str, default="metrics.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    if args.clip_norm_gate is None:
        args.clip_norm_gate = args.clip_norm_body

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

    # Sample rate
    if args.sample_rate is not None:
        sample_rate = args.sample_rate
    else:
        sample_rate = args.batch_size / len(train_loader.dataset)

    print(f"Dataset size: {len(train_loader.dataset)}  Sample rate: {sample_rate:.6f}")

    # Build model
    model = build_model(args).to(args.device)

    # Untie weights
    if hasattr(model, "lm_head") and hasattr(model, "transformer"):
        embed_weight = model.transformer.wte.weight
        if model.lm_head.weight is embed_weight:
            model.lm_head.weight = torch.nn.Parameter(embed_weight.detach().clone())
            model.config.tie_word_embeddings = False

    # Convert Conv1D to Linear
    _ensure_pkg()
    from modeling_gpt2_gated_dp import convert_conv1d_to_linear
    model = convert_conv1d_to_linear(model)

    # Wrap with GradSampleModule (per-sample gradient computation)
    from opacus.grad_sample import GradSampleModule
    model = GradSampleModule(model)
    model.train()

    # Identify gate slices in c_attn
    gate_slices = collect_gate_slices(model)
    print(f"Identified {len(gate_slices)} gate slices in c_attn")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    accountant = HeteroRDPAccountant()
    logger = MetricsLogger(
        path=args.metrics_csv, condition=args.condition,
        seed=args.seed, eps_target=args.eps_target or 0.0,
    )

    # Poisson-sampled DataLoader (mimic Opacus's behavior)
    from opacus.data_loader import DPDataLoader
    dp_loader = DPDataLoader.from_data_loader(train_loader, distributed=False)

    step = 0
    recent_losses = []
    data_iter = infinite_loader(dp_loader)

    while step < args.steps:
        batch = next(data_iter)
        batch = {k: v.to(args.device) for k, v in batch.items()}
        if batch["input_ids"].size(0) == 0:
            continue

        loss_val = hetero_dp_step(
            model, batch, optimizer,
            sigma_b=args.noise_multiplier_body,
            sigma_g=args.noise_multiplier_gate,
            clip_norm_body=args.clip_norm_body,
            clip_norm_gate=args.clip_norm_gate,
            gate_slices=gate_slices,
        )
        accountant.step(sample_rate, args.noise_multiplier_body, args.noise_multiplier_gate)
        step += 1
        recent_losses.append(loss_val)

        if step % args.print_every == 0:
            avg_train = sum(recent_losses[-args.print_every:]) / min(len(recent_losses), args.print_every)
            eps_consumed = accountant.get_epsilon(delta=args.delta)
            print(f"step={step:06d}  loss={avg_train:.4f}  ε_total={eps_consumed:.3f}")

        if args.eval_every and step % args.eval_every == 0:
            eval_loss, eval_ppl = evaluate(model, eval_loader, args.device, args.eval_steps)
            eps_consumed = accountant.get_epsilon(delta=args.delta)
            sparsity = measure_sparsity(model, batch, args.device, args.sparsity_threshold) \
                if args.gate_type != "none" else 0.0
            avg_train = sum(recent_losses[-args.eval_every:]) / min(len(recent_losses), args.eval_every)
            logger.log(step, avg_train, eval_loss, eval_ppl, sparsity,
                       eps_consumed,
                       args.noise_multiplier_body, args.noise_multiplier_gate,
                       args.clip_norm_body, args.clip_norm_gate)
            ppl_str = f"{eval_ppl:.3f}" if eval_ppl is not None else "n/a"
            print(f"  [eval] step={step:06d}  eval_ppl={ppl_str}  "
                  f"sparsity={sparsity:.3f}  ε_total={eps_consumed:.3f}")

    eps_final = accountant.get_epsilon(delta=args.delta)
    print(f"\nTraining complete. ε_total={eps_final:.4f}  δ={args.delta}")
    print(f"Metrics written to: {args.metrics_csv}")


if __name__ == "__main__":
    main()
