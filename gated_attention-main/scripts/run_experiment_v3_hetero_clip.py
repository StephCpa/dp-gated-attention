#!/usr/bin/env python
"""
run_experiment_v3_hetero_clip.py
=================================
Exp 3: Per-group clipping + heterogeneous noise (direction B, correct version).

Unlike run_experiment_v3_hetero.py (which used global clipping and therefore
could not exploit the sensitivity-ratio advantage), this version properly
implements direction B by using SEPARATE per-sample clip norms for body
(C_body) and gate (C_gate = ρ · C_body).

Sweep
-----
Fix bias = +0.5 (best from Exp 1).
Sweep (ρ, k) where:
  ρ = C_gate / C_body  (clip ratio — respects sensitivity bound)
  k = σ_g / σ_b        (noise ratio — redistributes privacy budget)

Per-step joint RDP:
  α T q² / 2 · [(C_body/σ_b)² + (C_gate/σ_g)²]
  = α T q² / 2 · (C_body/σ_b)² · (1 + (ρ/k)²)

So for fixed total privacy, σ_b² ∝ (1 + (ρ/k)²) · C_body².

Conditions
----------
We pick points that isolate specific mechanisms:

  baseline: ρ=1.0, k=1.0          standard DP-SGD reference
  clip:     ρ=0.25, k=1.0         gate tightly clipped, same noise scale
                                  (σ_g in ABSOLUTE terms = 0.25 · σ_b, achieved
                                   via smaller C rather than smaller σ)
  both:     ρ=0.25, k=0.25        gate sensitivity and noise both scaled 4x
                                  (sensitivity-matched noise)
  extreme:  ρ=0.1,  k=0.5         aggressive clip, moderate noise

Expected outcomes:
  baseline: reproduces k=1.0 result from previous hetero experiment
  clip:     σ_b increases only marginally (√(1+0.0625)≈1.03x), gate gets
            genuinely less noise → gate might finally learn selectivity
  both:     σ_b ≈ 1.41x baseline (cost), gate noise balanced to sensitivity
  extreme:  stress test of aggressive sensitivity reduction

Each condition × ε ∈ {3, 8} × seed ∈ {7, 42, 123} = 24 runs.

Usage
-----
python scripts/run_experiment_v3_hetero_clip.py \
    --hf-model gpt2 --tokenizer gpt2 \
    --gpus 0,7 --eps 3,8 --seeds 7,42,123 \
    --steps 10000 --batch-size 16
"""

import argparse
import csv
import os
import subprocess
import time
import math
from dataclasses import dataclass
from typing import List


# ---------------------------------------------------------------------------
# Condition definitions
# ---------------------------------------------------------------------------

@dataclass
class HeteroClipCondition:
    name: str
    rho: float   # C_gate / C_body
    k: float     # σ_g / σ_b
    description: str = ""


CONDITIONS: List[HeteroClipCondition] = [
    HeteroClipCondition(
        name="baseline_r1.0_k1.0",
        rho=1.0, k=1.0,
        description="Standard DP-SGD reference",
    ),
    HeteroClipCondition(
        name="clip_r0.25_k1.0",
        rho=0.25, k=1.0,
        description="Gate clipped to 0.25·C, same σ as body. "
                    "σ_b barely increases (~3%), gate gets 4x less absolute noise.",
    ),
    HeteroClipCondition(
        name="both_r0.25_k0.25",
        rho=0.25, k=0.25,
        description="Gate clipped AND noised at 0.25x. Sensitivity-matched balance.",
    ),
    HeteroClipCondition(
        name="extreme_r0.1_k0.5",
        rho=0.1, k=0.5,
        description="Aggressive clip (10x), moderate noise (2x). "
                    "Gate highly constrained but privacy-efficient.",
    ),
]


# ---------------------------------------------------------------------------
# Joint (σ_b, σ_g) search with per-group clipping
# ---------------------------------------------------------------------------

def compute_joint_epsilon(steps, sample_rate, sigma_b, sigma_g, delta):
    """
    Joint RDP of two parallel Gaussian mechanisms with unit sensitivity
    (since Opacus normalizes sensitivity). The clip norms enter the training
    dynamics but are divided out in RDP accounting.
    """
    from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    rdp_b = compute_rdp(q=sample_rate, noise_multiplier=sigma_b,
                        steps=steps, orders=orders)
    rdp_g = compute_rdp(q=sample_rate, noise_multiplier=sigma_g,
                        steps=steps, orders=orders)
    joint_rdp = [a + b for a, b in zip(rdp_b, rdp_g)]
    eps, _ = get_privacy_spent(orders=orders, rdp=joint_rdp, delta=delta)
    return eps


def find_sigma_b(target_eps, k, steps, sample_rate, delta,
                 lo=0.1, hi=10.0, iters=40):
    """Binary-search σ_b such that joint ε(σ_b, k·σ_b) = target_eps."""
    while compute_joint_epsilon(steps, sample_rate, hi, k * hi, delta) > target_eps:
        hi *= 2.0
        if hi > 1024:
            raise RuntimeError(f"Cannot satisfy ε={target_eps} with k={k}")
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        eps = compute_joint_epsilon(steps, sample_rate, mid, k * mid, delta)
        if eps > target_eps:
            lo = mid
        else:
            hi = mid
    return hi


def get_dataset_size(dataset, config, split):
    from datasets import load_dataset
    return len(load_dataset(dataset, config, split=split))


# ---------------------------------------------------------------------------
# Run spec
# ---------------------------------------------------------------------------

@dataclass
class RunSpec:
    cond: HeteroClipCondition
    eps: float
    sigma_b: float
    sigma_g: float
    clip_body: float
    clip_gate: float
    seed: int
    cmd: List[str]
    log_path: str
    csv_path: str


def build_runs(args) -> List[RunSpec]:
    if args.dataset_len is None:
        ds_len = get_dataset_size(args.dataset, args.dataset_config, args.dataset_split)
    else:
        ds_len = args.dataset_len

    sample_rate = min(1.0, args.batch_size / ds_len)
    print(f"Dataset size: {ds_len}  Sample rate: {sample_rate:.6f}")

    os.makedirs(args.results_dir, exist_ok=True)

    runs = []
    for eps in args.eps:
        for cond in CONDITIONS:
            print(f"Solving σ_b for ε={eps}, {cond.name}...", end=" ", flush=True)
            sigma_b = find_sigma_b(eps, cond.k, args.steps, sample_rate, args.delta)
            sigma_g = cond.k * sigma_b
            clip_body = args.clip_norm_body
            clip_gate = cond.rho * clip_body
            print(f"σ_b={sigma_b:.4f} σ_g={sigma_g:.4f} "
                  f"C_b={clip_body:.3f} C_g={clip_gate:.3f}")

            for seed in args.seeds:
                csv_path = os.path.join(
                    args.results_dir,
                    f"hc_{cond.name}_eps{eps}_seed{seed}.csv",
                )
                log_path = csv_path.replace(".csv", ".log")

                cmd = [
                    "python", args.train_script,
                    "--hf-model", args.hf_model,
                    "--tokenizer", args.tokenizer,
                    "--dataset", args.dataset,
                    "--dataset-config", args.dataset_config,
                    "--dataset-split", args.dataset_split,
                    "--seq-len", str(args.seq_len),
                    "--batch-size", str(args.batch_size),
                    "--steps", str(args.steps),
                    "--noise-multiplier-body", f"{sigma_b:.6f}",
                    "--noise-multiplier-gate", f"{sigma_g:.6f}",
                    "--clip-norm-body", f"{clip_body:.6f}",
                    "--clip-norm-gate", f"{clip_gate:.6f}",
                    "--lr", str(args.lr),
                    "--delta", str(args.delta),
                    "--eps-target", str(eps),
                    "--gate-type", "headwise",
                    "--init-gate-bias", str(args.init_gate_bias),
                    "--seed", str(seed),
                    "--condition", cond.name,
                    "--eval-every", str(args.eval_every),
                    "--eval-steps", str(args.eval_steps),
                    "--print-every", str(args.print_every),
                    "--metrics-csv", csv_path,
                ]

                runs.append(RunSpec(
                    cond=cond, eps=eps, sigma_b=sigma_b, sigma_g=sigma_g,
                    clip_body=clip_body, clip_gate=clip_gate, seed=seed,
                    cmd=cmd, log_path=log_path, csv_path=csv_path,
                ))
    return runs


# ---------------------------------------------------------------------------
# Parallel GPU launcher
# ---------------------------------------------------------------------------

def run_all(runs: List[RunSpec], gpus: List[int], dry_run: bool = False):
    gpu_queue = list(gpus)
    active = []

    def _launch(run, gpu_id):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        lf = open(run.log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(run.cmd, env=env, stdout=lf, stderr=lf)
        return proc, lf, run, gpu_id

    for run in runs:
        print(f"\n[queue] ε={run.eps}  {run.cond.name}  "
              f"σ_b={run.sigma_b:.3f} σ_g={run.sigma_g:.3f} "
              f"C_b={run.clip_body:.2f} C_g={run.clip_gate:.2f}  seed={run.seed}")
        print(f"        log → {run.log_path}")
        if dry_run:
            continue

        while not gpu_queue:
            time.sleep(5)
            still = []
            for proc, lf, r, gid in active:
                if proc.poll() is None:
                    still.append((proc, lf, r, gid))
                else:
                    lf.close()
                    gpu_queue.append(gid)
                    print(f"[done ] ε={r.eps}  {r.cond.name}  seed={r.seed}  gpu={gid}")
            active[:] = still

        gid = gpu_queue.pop(0)
        active.append(_launch(run, gid))
        print(f"[start] ε={run.eps}  {run.cond.name}  seed={run.seed}  gpu={gid}")

    if not dry_run:
        while active:
            time.sleep(5)
            still = []
            for proc, lf, r, gid in active:
                if proc.poll() is None:
                    still.append((proc, lf, r, gid))
                else:
                    lf.close()
                    gpu_queue.append(gid)
                    print(f"[done ] ε={r.eps}  {r.cond.name}  seed={r.seed}  gpu={gid}")
            active[:] = still


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(runs: List[RunSpec], out_path: str):
    rows = []
    for run in runs:
        if not os.path.exists(run.csv_path):
            continue
        with open(run.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            last_eval = None
            for row in reader:
                if row.get("eval_ppl"):
                    last_eval = row
        if last_eval:
            rows.append({
                "condition": run.cond.name,
                "rho": run.cond.rho, "k": run.cond.k,
                "eps_target": run.eps,
                "sigma_b": run.sigma_b, "sigma_g": run.sigma_g,
                "clip_body": run.clip_body, "clip_gate": run.clip_gate,
                "seed": run.seed,
                "eval_ppl": float(last_eval["eval_ppl"]),
                "gate_sparsity": float(last_eval["gate_sparsity"]),
                "eps_consumed": float(last_eval["eps_consumed_total"]),
            })
    if not rows:
        print("No completed runs found for summary.")
        return

    from collections import defaultdict
    import statistics
    groups = defaultdict(list)
    for r in rows:
        groups[(r["condition"], r["eps_target"])].append(r)

    summary = []
    for (cond_name, eps), group in sorted(groups.items()):
        ppls = [g["eval_ppl"] for g in group]
        sparses = [g["gate_sparsity"] for g in group]
        summary.append({
            "condition": cond_name,
            "rho": group[0]["rho"], "k": group[0]["k"],
            "eps_target": eps, "n_seeds": len(group),
            "sigma_b": group[0]["sigma_b"], "sigma_g": group[0]["sigma_g"],
            "clip_body": group[0]["clip_body"], "clip_gate": group[0]["clip_gate"],
            "ppl_mean": statistics.mean(ppls),
            "ppl_std": statistics.stdev(ppls) if len(ppls) > 1 else 0.0,
            "sparsity_mean": statistics.mean(sparses),
            "sparsity_std": statistics.stdev(sparses) if len(sparses) > 1 else 0.0,
        })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print(f"\nSummary written to: {out_path}")
    print(f"\n{'Condition':<22} {'ε':>3} {'ρ':>5} {'k':>5} "
          f"{'σ_b':>5} {'σ_g':>5}  {'PPL':>8} {'± std':>7}  {'Sparsity':>9}")
    print("-" * 85)
    for r in summary:
        print(f"{r['condition']:<22} {r['eps_target']:>3.0f} "
              f"{r['rho']:>5.2f} {r['k']:>5.2f} "
              f"{r['sigma_b']:>5.2f} {r['sigma_g']:>5.2f}  "
              f"{r['ppl_mean']:>8.3f} ±{r['ppl_std']:>6.3f}  "
              f"{r['sparsity_mean']:>9.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Per-group clip + hetero noise (Exp 3)")

    p.add_argument("--train-script", default="gated_attention-main/dp_train_v3_hetero.py")
    p.add_argument("--hf-model", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--dataset-len", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--clip-norm-body", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--eps", type=str, default="3,8")
    p.add_argument("--seeds", type=str, default="7,42,123")
    p.add_argument("--gpus", type=str, default="0,1")
    p.add_argument("--init-gate-bias", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--results-dir", default="results/v3/hetero_clip")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--summary-only", action="store_true")

    args = p.parse_args()
    args.eps = [float(v) for v in args.eps.split(",")]
    args.seeds = [int(v) for v in args.seeds.split(",")]
    gpus = [int(v) for v in args.gpus.split(",")]

    print("=" * 60)
    print("v3 Exp 3 — Per-group clip + hetero noise")
    print("=" * 60)
    print("Conditions:")
    for c in CONDITIONS:
        print(f"  {c.name:<22} ρ={c.rho:.2f} k={c.k:.2f}  {c.description}")
    print(f"\nε targets:  {args.eps}")
    print(f"Seeds:      {args.seeds}")
    print(f"init_bias:  {args.init_gate_bias}")
    print(f"Total runs: {len(CONDITIONS) * len(args.eps) * len(args.seeds)}")
    print()

    if args.summary_only:
        runs = build_runs(args)
        write_summary(runs, os.path.join(args.results_dir, "summary.csv"))
        return

    runs = build_runs(args)

    if args.dry_run:
        print("\n[DRY RUN] Commands that would be executed:")
        run_all(runs, gpus, dry_run=True)
    else:
        run_all(runs, gpus, dry_run=False)
        write_summary(runs, os.path.join(args.results_dir, "summary.csv"))


if __name__ == "__main__":
    main()
