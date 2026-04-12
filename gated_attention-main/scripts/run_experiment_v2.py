#!/usr/bin/env python
"""
run_experiment_v2.py
====================
Driver for the four-condition ablation study of DP-aware gated attention.

Conditions
----------
The experiment tests the causal chain of improvements.  Each condition
adds exactly one variable over the previous, so the contribution of each
component can be isolated.

  A  none              — Baseline GPT-2, no gate
  B  gate_naive        — Gate with default init (bias=0, sigmoid≈0.5)
                         Replicates the original experiment.
                         Expected: slightly worse than A (confirms current finding)
  C  gate_dp_init      — Gate with DP-aware init (bias=-2.197, sigmoid≈0.1)
                         No L1 regularisation.
                         Expected: ≈ A or slightly better
  D  gate_dp_init_l1   — Gate with DP-aware init + ε-adaptive L1 regularisation
                         Expected: better than A

Each condition × ε ∈ {3, 8} × seed ∈ {7, 42, 123} = 24 runs.

All runs use GPT-2 on WikiText-2-raw-v1.  σ is searched once per (ε, seed)
and shared across conditions A–D, which is valid because the headwise gate
adds only ~0.09% extra parameters to GPT-2 (negligible ε difference).

Output
------
results/
  condition_A_eps3_seed7.csv
  condition_B_eps3_seed7.csv
  ...
  summary.csv          <- aggregated final-step metrics across all runs

Usage
-----
python scripts/run_experiment_v2.py \
    --hf-model gpt2 \
    --tokenizer gpt2 \
    --train-script gated_attention-main/dp_train_v2.py \
    --gpus 0,1,2,3 \
    --eps 3,8 \
    --seeds 7,42,123 \
    --steps 10000 \
    --batch-size 16

Dry run (print commands only, do not execute):
    python scripts/run_experiment_v2.py --dry-run ...
"""

import argparse
import csv
import os
import subprocess
import time
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Condition definitions
# ---------------------------------------------------------------------------

@dataclass
class Condition:
    name: str
    gate_type: str
    init_gate_bias: float     # 0.0 = naive;  -2.197 = DP-aware
    gate_l1_lambda: float     # 0.0 = no regularisation
    gate_l1_gamma: float = 1.0
    description: str = ""


_DP_INIT = math.log(0.1 / 0.9)   # ≈ -2.197

CONDITIONS: List[Condition] = [
    Condition(
        name="A_none",
        gate_type="none",
        init_gate_bias=0.0,
        gate_l1_lambda=0.0,
        description="Baseline — no gate, standard DP-SGD",
    ),
    Condition(
        name="B_gate_naive",
        gate_type="headwise",
        init_gate_bias=0.0,         # sigmoid(0) = 0.5
        gate_l1_lambda=0.0,
        description="Naive gate — default init, no L1. Replicates original experiment.",
    ),
    Condition(
        name="C_gate_dp_init",
        gate_type="headwise",
        init_gate_bias=_DP_INIT,    # sigmoid(-2.197) ≈ 0.1
        gate_l1_lambda=0.0,
        description="DP-aware init only — gate starts maximally sparse.",
    ),
    Condition(
        name="D_gate_dp_init_l1",
        gate_type="headwise",
        init_gate_bias=_DP_INIT,
        gate_l1_lambda=0.01,        # tune if needed
        gate_l1_gamma=1.0,
        description="DP-aware init + ε-adaptive L1 — full proposed method.",
    ),
]


# ---------------------------------------------------------------------------
# σ search (binary search for noise_multiplier given ε target)
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
    condition: Condition
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

        for cond in CONDITIONS:
            for seed in args.seeds:
                csv_path = os.path.join(
                    args.results_dir,
                    f"condition_{cond.name}_eps{eps}_seed{seed}.csv",
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
                    "--gate-type", cond.gate_type,
                    "--init-gate-bias", str(cond.init_gate_bias),
                    "--gate-l1-lambda", str(cond.gate_l1_lambda),
                    "--gate-l1-gamma", str(cond.gate_l1_gamma),
                    "--seed", str(seed),
                    "--condition", cond.name,
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
                    condition=cond,
                    eps=eps,
                    sigma=sigma,
                    seed=seed,
                    cmd=cmd,
                    log_path=log_path,
                    csv_path=csv_path,
                ))
    return runs


# ---------------------------------------------------------------------------
# Parallel GPU launcher
# ---------------------------------------------------------------------------

def run_all(runs: List[RunSpec], gpus: List[int], dry_run: bool = False):
    gpu_queue = list(gpus)
    active = []   # (proc, log_file, run, gpu_id)

    def _launch(run, gpu_id):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        lf = open(run.log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(run.cmd, env=env, stdout=lf, stderr=lf)
        return proc, lf, run, gpu_id

    for run in runs:
        cond_name = run.condition.name
        print(f"\n[queue] ε={run.eps}  σ={run.sigma:.4f}  cond={cond_name}  seed={run.seed}")
        print(f"        {' '.join(run.cmd[:8])} ...")
        print(f"        log → {run.log_path}")
        if dry_run:
            continue

        # Wait for a GPU slot
        while not gpu_queue:
            time.sleep(5)
            still = []
            for proc, lf, r, gid in active:
                if proc.poll() is None:
                    still.append((proc, lf, r, gid))
                else:
                    lf.close()
                    gpu_queue.append(gid)
                    print(f"[done ] ε={r.eps}  cond={r.condition.name}  seed={r.seed}  gpu={gid}")
            active[:] = still

        gid = gpu_queue.pop(0)
        active.append(_launch(run, gid))
        print(f"[start] ε={run.eps}  cond={cond_name}  seed={run.seed}  gpu={gid}")

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
                    print(f"[done ] ε={r.eps}  cond={r.condition.name}  seed={r.seed}  gpu={gid}")
            active[:] = still


# ---------------------------------------------------------------------------
# Post-hoc summary CSV
# ---------------------------------------------------------------------------

def write_summary(runs: List[RunSpec], out_path: str):
    """
    Read the last eval row from each run's CSV and aggregate into a summary
    with mean ± std across seeds for each (condition, ε) cell.
    """
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
                "condition": run.condition.name,
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

    # Group and compute mean ± std
    from collections import defaultdict
    import statistics
    groups = defaultdict(list)
    for r in rows:
        key = (r["condition"], r["eps_target"])
        groups[key].append(r)

    summary_rows = []
    for (cond, eps), group in sorted(groups.items()):
        ppls = [g["eval_ppl"] for g in group]
        sparses = [g["gate_sparsity"] for g in group]
        summary_rows.append({
            "condition": cond,
            "eps_target": eps,
            "n_seeds": len(group),
            "eval_ppl_mean": statistics.mean(ppls),
            "eval_ppl_std": statistics.stdev(ppls) if len(ppls) > 1 else 0.0,
            "gate_sparsity_mean": statistics.mean(sparses),
            "gate_sparsity_std": statistics.stdev(sparses) if len(sparses) > 1 else 0.0,
        })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSummary written to: {out_path}")
    print(f"\n{'Condition':<25} {'ε':>4}  {'PPL mean':>9} {'± std':>7}  {'Sparsity':>8}")
    print("-" * 60)
    for r in summary_rows:
        print(
            f"{r['condition']:<25} {r['eps_target']:>4.0f}  "
            f"{r['eval_ppl_mean']:>9.3f} ±{r['eval_ppl_std']:>6.3f}  "
            f"{r['gate_sparsity_mean']:>7.3f}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="4-condition ablation for DP-Gated Attention")

    p.add_argument("--train-script", default="gated_attention-main/dp_train_v2.py")
    p.add_argument("--hf-model", required=True, help="e.g. gpt2")
    p.add_argument("--tokenizer", required=True, help="e.g. gpt2")
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--dataset-len", type=int, default=None,
                   help="Override dataset length (avoids loading the full dataset twice).")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--clip-norm", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--eps", type=str, default="3,8",
                   help="Comma-separated ε targets.")
    p.add_argument("--seeds", type=str, default="7,42,123",
                   help="Comma-separated random seeds.")
    p.add_argument("--gpus", type=str, default="0,1,2,3",
                   help="Comma-separated CUDA device IDs.")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--grad-sample-mode", default="hooks",
                   choices=["hooks", "functorch"])
    p.add_argument("--max-physical-batch-size", type=int, default=0)
    p.add_argument("--poisson-sampling", action="store_true", default=True)
    p.add_argument("--no-poisson-sampling", action="store_false", dest="poisson_sampling")
    p.add_argument("--results-dir", default="results/v2")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without running them.")
    p.add_argument("--summary-only", action="store_true",
                   help="Skip launching runs; just produce summary.csv from existing CSVs.")

    args = p.parse_args()
    args.eps = [float(v) for v in args.eps.split(",")]
    args.seeds = [int(v) for v in args.seeds.split(",")]
    gpus = [int(v) for v in args.gpus.split(",")]

    print("=" * 60)
    print("DP-Gated Attention v2 — 4-condition ablation")
    print("=" * 60)
    print(f"Conditions: {[c.name for c in CONDITIONS]}")
    print(f"ε targets:  {args.eps}")
    print(f"Seeds:      {args.seeds}")
    print(f"Total runs: {len(CONDITIONS) * len(args.eps) * len(args.seeds)}")
    print()

    if args.summary_only:
        # Reconstruct run specs to know CSV paths, without re-launching
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
