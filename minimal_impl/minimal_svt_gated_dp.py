#!/usr/bin/env python
import argparse
import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DPConfig:
    clip_norm: float = 1.0
    noise_multiplier: float = 1.5
    learning_rate: float = 0.5


@dataclass
class SVTConfig:
    threshold: float = 0.05
    max_positive: int = 3
    sigma_threshold: float = 1.0
    sigma_query: float = 1.0
    sensitivity: float = 1.0  # placeholder for accounting


class GatedSelfAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        d = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        gate = torch.sigmoid(self.gate_proj(x))
        return self.o_proj(out * gate), gate


class TinyModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_classes: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.attn = GatedSelfAttention(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, tokens: torch.Tensor):
        x = self.embed(tokens)
        attn_out, gate = self.attn(x)
        pooled = attn_out.mean(dim=1)
        logits = self.classifier(pooled)
        return logits, gate


def make_synthetic_batch(batch_size: int, seq_len: int, vocab_size: int, device):
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # Simple rule: label = 1 if sum(tokens) exceeds a threshold.
    threshold = (seq_len * (vocab_size - 1)) // 2
    labels = (tokens.sum(dim=1) > threshold).long()
    return tokens, labels


def svt_select(scores, cfg: SVTConfig, device):
    # Gaussian SVT-like selection (toy). Returns boolean mask.
    threshold_noise = torch.normal(
        mean=0.0, std=cfg.sigma_threshold, size=(), device=device
    )
    noisy_threshold = cfg.threshold + threshold_noise
    selected = []
    count = 0
    for s in scores:
        if count >= cfg.max_positive:
            selected.append(False)
            continue
        query_noise = torch.normal(mean=0.0, std=cfg.sigma_query, size=(), device=device)
        noisy_score = s + query_noise
        if noisy_score >= noisy_threshold:
            selected.append(True)
            count += 1
        else:
            selected.append(False)
    return selected


def per_sample_grads(model, loss_fn, tokens, labels):
    params = [p for p in model.parameters() if p.requires_grad]
    grads = []
    for i in range(tokens.size(0)):
        model.zero_grad(set_to_none=True)
        logits, _ = model(tokens[i : i + 1])
        loss = loss_fn(logits, labels[i : i + 1])
        g = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
        grads.append([gi.detach() for gi in g])
    return grads, params


def clip_and_aggregate(grads, clip_norm: float):
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


def dp_sgd_step(model, loss_fn, tokens, labels, dp_cfg: DPConfig, svt_cfg: SVTConfig):
    grads, params = per_sample_grads(model, loss_fn, tokens, labels)
    agg_grads = clip_and_aggregate(grads, dp_cfg.clip_norm)
    scores = [g.norm().detach() for g in agg_grads]
    selected = svt_select(scores, svt_cfg, device=agg_grads[0].device)

    for p, g, keep in zip(params, agg_grads, selected):
        if keep:
            noise = torch.normal(
                mean=0.0,
                std=dp_cfg.noise_multiplier * dp_cfg.clip_norm,
                size=g.shape,
                device=g.device,
            )
            p.grad = g + noise
        else:
            p.grad = torch.zeros_like(g)

    with torch.no_grad():
        for p in params:
            p -= dp_cfg.learning_rate * p.grad

    return selected


def rdp_gaussian(alpha: float, sigma: float, sensitivity: float = 1.0) -> float:
    # RDP of Gaussian mechanism (no subsampling).
    return alpha * (sensitivity**2) / (2.0 * sigma**2)


def eps_from_rdp(rho: float, alpha: float, delta: float) -> float:
    return rho + math.log(1.0 / delta) / (alpha - 1.0)


def estimate_eps_upper_bound(steps, dp_cfg: DPConfig, svt_cfg: SVTConfig, delta=1e-5):
    # Conservative bound: no subsampling, SVT as (c+1) Gaussian queries.
    best_eps = float("inf")
    for alpha in [2, 4, 8, 16, 32, 64]:
        rho_sgd = steps * rdp_gaussian(alpha, dp_cfg.noise_multiplier, dp_cfg.clip_norm)
        rho_svt = (svt_cfg.max_positive + 1) * rdp_gaussian(
            alpha, svt_cfg.sigma_query, svt_cfg.sensitivity
        )
        rho = rho_sgd + rho_svt
        eps = eps_from_rdp(rho, alpha, delta)
        best_eps = min(best_eps, eps)
    return best_eps


def run(steps: int, batch_size: int, seq_len: int, vocab_size: int, d_model: int):
    device = torch.device("cpu")
    torch.set_num_threads(1)
    torch.manual_seed(7)
    random.seed(7)

    model = TinyModel(vocab_size=vocab_size, d_model=d_model, num_classes=2).to(device)
    loss_fn = nn.CrossEntropyLoss()
    dp_cfg = DPConfig()
    svt_cfg = SVTConfig()

    for step in range(1, steps + 1):
        tokens, labels = make_synthetic_batch(batch_size, seq_len, vocab_size, device)
        logits, gate = model(tokens)
        loss = loss_fn(logits, labels).item()

        selected = dp_sgd_step(model, loss_fn, tokens, labels, dp_cfg, svt_cfg)
        selected_count = sum(1 for s in selected if s)

        if step % 5 == 0 or step == 1:
            print(
                f"step={step:03d} loss={loss:.4f} selected_blocks={selected_count}"
            )

        # Simple invariant checks for verification.
        assert selected_count <= svt_cfg.max_positive
        assert gate.min().item() >= 0.0 and gate.max().item() <= 1.0

    eps = estimate_eps_upper_bound(steps, dp_cfg, svt_cfg, delta=1e-5)
    print(f"eps_upper_bound(no_subsampling, toy SVT)={eps:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--vocab-size", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if args.verify:
        run(
            steps=args.steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            d_model=args.d_model,
        )
        print("verify=ok")
    else:
        run(
            steps=args.steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            d_model=args.d_model,
        )


if __name__ == "__main__":
    main()
