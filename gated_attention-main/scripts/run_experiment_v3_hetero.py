#!/usr/bin/env python
"""
run_experiment_v3_hetero.py
===========================
Exp 2: Heterogeneous noise sweep for direction B.

Tests the central hypothesis of direction B: the gate's natural
sensitivity (bounded by σ' ≤ 0.25) means it can be trained with less
noise than the body without consuming the full privacy budget.

Sweep
-----
Fix bias = +0.5 (best from Exp 1 phase scan, achieves baseline parity).
Sweep k = σ_g / σ_b ∈ {0.25, 0.5, 1.0}:

  k=1.00  baseline (uniform DP-SGD, equivalent to v2 result)
  k=0.50  gate noise halved
  k=0.25  gate noise quartered (matches sensitivity ratio prediction)

For each k, σ_b is solved so that the JOINT (ε, δ)-DP equals the target
ε ∈ {3, 8}.  This means a smaller k means more noise on body to keep
total budget fixed — testing whether the trade-off is favourable.

Each k × ε ∈ {3, 8} × seed ∈ {7, 42, 123} = 18 runs.

Predictions (per direction-B theory):
  - k=1.00: gate sparsity stays ≈ 0 (replicates Exp 1 finding)
  - k=0.50: some sparsity emerges; PPL ≈ baseline
  - k=0.25: clearer sparsity emergence; PPL may BEAT baseline

Output
------
results/v3/hetero/
  hetero_k1.00_eps3_seed7.csv
  hetero_k0.50_eps3_seed7.csv
  ...
  summary.csv

Usage
-----
python scripts/run_experiment_v3_hetero.py \
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
# k sweep configurations
# ---------------------------------------------------------------------------

K_VALUES: List[float] = [1.0, 0.5, 0.25]


# ---------------------------------------------------------------------------
# Joint (σ_b, σ_g) search for given (ε_target, k)
# ---------------------------------------------------------------------------

def compute_joint_epsilon(steps, sample_rate, sigma_b, sigma_g, delta):
    """Joint RDP composition of two parallel Gaussian mechanisms."""
    from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    rdp_b = compute_rdp(q=sample_rate, noise_multiplier=sigma_b,
                        steps=steps, orders=orders)
    rdp_g = compute_rdp(q=sample_rate, noise_multiplier=sigma_g,
                        steps=steps, orders=orders)
    joint_rdp = [a + b for a, b in zip(rdp_b, rdp_g)]
    eps, _ = get_privacy_spent(orders=orders, rdp=joint_rdp, delta=delta)
    return eps


def find_sigma_b_for_joint(target_eps, k, steps, sample_rate, delta,
                            lo=0.1, hi=10.0, iters=40):
    """
    Binary-search σ_b such that joint ε(σ_b, k·σ_b) = target_eps.
    Larger σ_b → smaller ε.
    """
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
    k: float
    eps: float
    sigma_b: float
    sigma_g: float
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
        for k in K_VALUES:
            print(f"Solving σ_b for ε={eps}, k={k}...", end=" ", flush=True)
            sigma_b = find_sigma_b_for_joint(eps, k, args.steps, sample_rate, args.delta)
            sigma_g = k * sigma_b
            print(f"σ_b={sigma_b:.4f}  σ_g={sigma_g:.4f}")

            for seed in args.seeds:
                k_label = f"k{k:.2f}".replace(".", "_")
                csv_path = os.path.join(
                    args.results_dir,
                    f"hetero_{k_label}_eps{eps}_seed{seed}.csv",
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
                    "--clip-norm", str(args.clip_norm),
                    "--lr", str(args.lr),
                    "--delta", str(args.delta),
                    "--eps-target", str(eps),
                    "--gate-type", "headwise",
                    "--init-gate-bias", str(args.init_gate_bias),
                    "--seed", str(seed),
                    "--condition", k_label,
                    "--eval-every", str(args.eval_every),
                    "--eval-steps", str(args.eval_steps),
                    "--print-every", str(args.print_every),
                    "--metrics-csv", csv_path,
                ]

                runs.append(RunSpec(
                    k=k, eps=eps, sigma_b=sigma_b, sigma_g=sigma_g,
                    seed=seed, cmd=cmd, log_path=log_path, csv_path=csv_path,
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
        print(f"\n[queue] ε={run.eps}  k={run.k:.2f}  σ_b={run.sigma_b:.3f} "
              f"σ_g={run.sigma_g:.3f}  seed={run.seed}")
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
                    print(f"[done ] ε={r.eps}  k={r.k:.2f}  seed={r.seed}  gpu={gid}")
            active[:] = still

        gid = gpu_queue.pop(0)
        active.append(_launch(run, gid))
        print(f"[start] ε={run.eps}  k={run.k:.2f}  seed={run.seed}  gpu={gid}")

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
                    print(f"[done ] ε={r.eps}  k={r.k:.2f}  seed={r.seed}  gpu={gid}")
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
                "k": run.k, "eps_target": run.eps,
                "sigma_b": run.sigma_b, "sigma_g": run.sigma_g,
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
        groups[(r["k"], r["eps_target"])].append(r)

    summary = []
    for (k, eps), group in sorted(groups.items()):
        ppls = [g["eval_ppl"] for g in group]
        sparses = [g["gate_sparsity"] for g in group]
        summary.append({
            "k": k, "eps_target": eps, "n_seeds": len(group),
            "sigma_b": group[0]["sigma_b"], "sigma_g": group[0]["sigma_g"],
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
    print(f"\n{'k':>5} {'ε':>3} {'σ_b':>6} {'σ_g':>6}  {'PPL mean':>9} {'± std':>7}  {'Sparsity':>10}")
    print("-" * 70)
    for r in summary:
        print(f"{r['k']:>5.2f} {r['eps_target']:>3.0f} {r['sigma_b']:>6.3f} "
              f"{r['sigma_g']:>6.3f}  {r['ppl_mean']:>9.3f} ±{r['ppl_std']:>6.3f}  "
              f"{r['sparsity_mean']:>9.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Heterogeneous noise sweep (Exp 2)")

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
    p.add_argument("--clip-norm", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--eps", type=str, default="3,8")
    p.add_argument("--seeds", type=str, default="7,42,123")
    p.add_argument("--gpus", type=str, default="0,1")
    p.add_argument("--init-gate-bias", type=float, default=0.5,
                   help="Default 0.5 (best from Exp 1).")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--results-dir", default="results/v3/hetero")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--summary-only", action="store_true")

    args = p.parse_args()
    args.eps = [float(v) for v in args.eps.split(",")]
    args.seeds = [int(v) for v in args.seeds.split(",")]
    gpus = [int(v) for v in args.gpus.split(",")]

    print("=" * 60)
    print("v3 Exp 2 — Heterogeneous noise sweep")
    print("=" * 60)
    print(f"k values:   {K_VALUES}")
    print(f"ε targets:  {args.eps}")
    print(f"Seeds:      {args.seeds}")
    print(f"init_bias:  {args.init_gate_bias}  (sigmoid={1/(1+math.exp(-args.init_gate_bias)):.3f})")
    print(f"Total runs: {len(K_VALUES) * len(args.eps) * len(args.seeds)}")
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
