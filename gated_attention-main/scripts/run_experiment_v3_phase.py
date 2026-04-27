#!/usr/bin/env python
"""
run_experiment_v3_phase.py
==========================
Exp 1: Phase transition verification for gate bias initialisation.

Theoretical prediction (from refined framework):
------------------------------------------------
A gate with bias b_0 can escape the "slow manifold" (where its own
gradient is dominated by DP noise) only if

    σ'(b_0) > σ · C / ||g_raw||

where σ'(·) is the sigmoid derivative (max 0.25 at x=0), σ is the
noise multiplier, C is the clip norm, and ||g_raw|| is the typical
raw gradient norm (≈2 from condition A's training logs).

Under typical settings (σC ≈ 0.5, ||g_raw|| ≈ 2), this gives
σ'(b_0) > 0.25, i.e. b_0 ∈ [-1.1, 1.1].  Values outside this range
should exhibit catastrophic degradation (PPL blow-up) because the
gate cannot learn selectivity.  Values inside should behave similarly
to each other (all above the critical threshold).

This experiment falsifies (or confirms) the phase-transition prediction
by sweeping b_0 ∈ {-2.2, -1.1, -0.5, 0, 0.5} at fixed σ.

Each bias × ε ∈ {3, 8} × 1 seed = 10 runs (fast validation).

Comparison with v2:
-------------------
- v2 ran 4 conditions × 2 ε × 3 seeds = 24 runs
- v3 runs 5 bias values × 2 ε × 1 seed = 10 runs
- Much faster, designed to isolate the effect of init_gate_bias alone

Output
------
results/v3/phase/
  phase_bias-2.2_eps3_seed7.csv
  phase_bias-1.1_eps3_seed7.csv
  ...
  summary.csv  <- PPL vs. bias curve

Usage
-----
python scripts/run_experiment_v3_phase.py \
    --hf-model gpt2 \
    --tokenizer gpt2 \
    --train-script gated_attention-main/dp_train_v2.py \
    --gpus 0,1 \
    --eps 3,8 \
    --seeds 7 \
    --steps 10000 \
    --batch-size 16
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
# Phase-transition sweep over init_gate_bias
# ---------------------------------------------------------------------------

@dataclass
class BiasPoint:
    name: str          # label for logs/csv
    init_bias: float   # gate bias at init
    sigmoid_val: float # sigmoid(init_bias) for reference
    sigmoid_deriv: float  # σ'(init_bias), the theoretical quantity of interest


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _sigmoid_deriv(x: float) -> float:
    s = _sigmoid(x)
    return s * (1.0 - s)


# Sweep designed to straddle the predicted critical threshold σ' ≈ 0.25
BIAS_POINTS: List[BiasPoint] = [
    BiasPoint("bias-2.2", -2.197, _sigmoid(-2.197), _sigmoid_deriv(-2.197)),  # σ'≈0.09, BELOW threshold
    BiasPoint("bias-1.1", -1.100, _sigmoid(-1.100), _sigmoid_deriv(-1.100)),  # σ'≈0.20, near threshold
    BiasPoint("bias-0.5", -0.500, _sigmoid(-0.500), _sigmoid_deriv(-0.500)),  # σ'≈0.24, just below 0.25
    BiasPoint("bias+0.0",  0.000, _sigmoid( 0.000), _sigmoid_deriv( 0.000)),  # σ'=0.25, at threshold
    BiasPoint("bias+0.5",  0.500, _sigmoid( 0.500), _sigmoid_deriv( 0.500)),  # σ'≈0.24, symmetric check
]


# ---------------------------------------------------------------------------
# σ search (copied from v2 for self-containment)
# ---------------------------------------------------------------------------

def compute_epsilon(steps, sample_rate, noise_multiplier, delta):
    from opacus.accountants import RDPAccountant
    acc = RDPAccountant()
    for _ in range(steps):
        acc.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    return acc.get_epsilon(delta)


def find_sigma(target_eps, steps, sample_rate, delta, lo=0.1, hi=10.0, iters=40):
    while compute_epsilon(steps, sample_rate, hi, delta) > target_eps:
        hi *= 2.0
        if hi > 1024:
            raise RuntimeError(f"Cannot satisfy ε={target_eps}: σ search diverged.")
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        eps = compute_epsilon(steps, sample_rate, mid, delta)
        if eps > target_eps:
            lo = mid
        else:
            hi = mid
    return hi


def get_dataset_size(dataset, config, split):
    from datasets import load_dataset
    return len(load_dataset(dataset, config, split=split))


# ---------------------------------------------------------------------------
# Run specification
# ---------------------------------------------------------------------------

@dataclass
class RunSpec:
    point: BiasPoint
    eps: float
    sigma: float
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
        print(f"Searching σ for ε={eps}...", end=" ", flush=True)
        sigma = find_sigma(eps, args.steps, sample_rate, args.delta)
        print(f"σ={sigma:.6f}")

        for pt in BIAS_POINTS:
            for seed in args.seeds:
                csv_path = os.path.join(
                    args.results_dir,
                    f"phase_{pt.name}_eps{eps}_seed{seed}.csv",
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
                    "--noise-multiplier", f"{sigma:.6f}",
                    "--clip-norm", str(args.clip_norm),
                    "--lr", str(args.lr),
                    "--delta", str(args.delta),
                    "--eps-target", str(eps),
                    "--gate-type", "headwise",
                    "--init-gate-bias", f"{pt.init_bias:.4f}",
                    "--gate-l1-lambda", "0.0",
                    "--seed", str(seed),
                    "--condition", pt.name,
                    "--eval-every", str(args.eval_every),
                    "--eval-steps", str(args.eval_steps),
                    "--print-every", str(args.print_every),
                    "--metrics-csv", csv_path,
                ]
                if args.grad_sample_mode:
                    cmd += ["--grad-sample-mode", args.grad_sample_mode]
                if args.max_physical_batch_size > 0:
                    cmd += ["--max-physical-batch-size", str(args.max_physical_batch_size)]
                if not args.poisson_sampling:
                    cmd += ["--no-poisson-sampling"]

                runs.append(RunSpec(
                    point=pt, eps=eps, sigma=sigma, seed=seed,
                    cmd=cmd, log_path=log_path, csv_path=csv_path,
                ))
    return runs


# ---------------------------------------------------------------------------
# Parallel GPU launcher (same pattern as v2)
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
        print(f"\n[queue] ε={run.eps}  σ={run.sigma:.4f}  bias={run.point.init_bias:+.3f} "
              f"(σ'={run.point.sigmoid_deriv:.3f})  seed={run.seed}")
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
                    print(f"[done ] ε={r.eps}  bias={r.point.init_bias:+.3f}  gpu={gid}")
            active[:] = still

        gid = gpu_queue.pop(0)
        active.append(_launch(run, gid))
        print(f"[start] ε={run.eps}  bias={run.point.init_bias:+.3f}  gpu={gid}")

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
                    print(f"[done ] ε={r.eps}  bias={r.point.init_bias:+.3f}  gpu={gid}")
            active[:] = still


# ---------------------------------------------------------------------------
# Summary: PPL vs bias curve with phase-transition marker
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
                "bias_label": run.point.name,
                "init_bias": run.point.init_bias,
                "sigmoid_val": run.point.sigmoid_val,
                "sigmoid_deriv": run.point.sigmoid_deriv,
                "eps_target": run.eps,
                "seed": run.seed,
                "eval_ppl": float(last_eval["eval_ppl"]),
                "eval_loss": float(last_eval["eval_loss"]),
                "gate_sparsity": float(last_eval["gate_sparsity"]),
                "eps_consumed": float(last_eval["eps_consumed"]),
            })

    if not rows:
        print("No completed runs found for summary.")
        return

    rows.sort(key=lambda r: (r["eps_target"], r["init_bias"]))

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSummary written to: {out_path}")
    print(f"\n{'bias':>7} {'σ(b)':>6} {'σ′(b)':>6} {'ε':>3}  {'PPL':>8}  {'sparsity':>8}")
    print("-" * 55)
    for r in rows:
        print(f"{r['init_bias']:>+7.3f} {r['sigmoid_val']:>6.3f} "
              f"{r['sigmoid_deriv']:>6.3f} {r['eps_target']:>3.0f}  "
              f"{r['eval_ppl']:>8.2f}  {r['gate_sparsity']:>8.3f}")

    # Flag phase transition
    print("\n[phase transition analysis]")
    print("Theory: PPL blow-up expected for σ'(b) < σC/||g_raw|| ≈ 0.25")
    print("        (i.e. |bias| > 1.1 approximately)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Phase transition verification (Exp 1)")

    p.add_argument("--train-script", default="gated_attention-main/dp_train_v2.py")
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
    p.add_argument("--seeds", type=str, default="7")
    p.add_argument("--gpus", type=str, default="0,1")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--grad-sample-mode", default="functorch",
                   choices=["hooks", "functorch"])
    p.add_argument("--max-physical-batch-size", type=int, default=0)
    p.add_argument("--poisson-sampling", action="store_true", default=True)
    p.add_argument("--no-poisson-sampling", action="store_false", dest="poisson_sampling")
    p.add_argument("--results-dir", default="results/v3/phase")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--summary-only", action="store_true")

    args = p.parse_args()
    args.eps = [float(v) for v in args.eps.split(",")]
    args.seeds = [int(v) for v in args.seeds.split(",")]
    gpus = [int(v) for v in args.gpus.split(",")]

    print("=" * 60)
    print("v3 Exp 1 — Phase transition verification for init_gate_bias")
    print("=" * 60)
    print("\nSweep:")
    print(f"{'label':<10} {'bias':>7} {'σ(b)':>6} {'σ′(b)':>6}")
    for pt in BIAS_POINTS:
        marker = " <- below critical" if pt.sigmoid_deriv < 0.25 else ""
        print(f"{pt.name:<10} {pt.init_bias:>+7.3f} {pt.sigmoid_val:>6.3f} "
              f"{pt.sigmoid_deriv:>6.3f}{marker}")
    print(f"\nε targets:  {args.eps}")
    print(f"Seeds:      {args.seeds}")
    print(f"Total runs: {len(BIAS_POINTS) * len(args.eps) * len(args.seeds)}")
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
