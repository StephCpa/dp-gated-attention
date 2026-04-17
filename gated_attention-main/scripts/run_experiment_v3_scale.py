#!/usr/bin/env python
"""
run_experiment_v3_scale.py
==========================
Exp 4: Scale-dependent phase transition for gate selectivity.

Tests the constructive prediction from the negative result analysis:
gated attention selectivity requires head redundancy, which scales with
model size. Specifically:

    Var_h[∂L/∂G_h] must exceed σ²C² / (B · σ'²)

With more heads (H), the per-head load decreases, creating more
redundancy and potentially larger inter-head variance.

Model sweep:
    gpt2         — 12 heads, 768 hidden,  117M params (reference)
    gpt2-medium  — 16 heads, 1024 hidden, 345M params
    gpt2-large   — 20 heads, 1280 hidden, 774M params

For each model: no-gate baseline vs gated (bias=+0.5).
Use ε=8 only (maximises signal by reducing noise).
Single seed for fast validation; extend to 3 seeds if promising.

Total: 3 models × 2 gate_types × 1 ε × 1 seed = 6 runs.
If sparsity > 0 emerges at larger scale, extend with 3 seeds.

Expected outcomes:
    gpt2 (12h):     sparsity = 0, gated ≈ baseline  (replicates Exp 1)
    gpt2-medium (16h): sparsity ≈ 0 or marginally > 0
    gpt2-large (20h):  sparsity possibly > 0 → phase transition evidence

Usage
-----
python scripts/run_experiment_v3_scale.py \
    --gpus 0,7 --seed 7 --steps 10000 --results-dir results/v3/scale

For larger models, may need --max-physical-batch-size to fit in GPU memory.
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
# Model configurations
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    name: str           # HuggingFace model name
    label: str          # short label for logs/csv
    num_heads: int      # for reference in output
    batch_size: int     # may need to reduce for larger models
    max_phys_bs: int    # max_physical_batch_size for memory management


MODEL_CONFIGS: List[ModelConfig] = [
    ModelConfig("gpt2",        "gpt2_12h",  12, batch_size=16, max_phys_bs=0),
    ModelConfig("gpt2-medium", "gpt2m_16h", 16, batch_size=8,  max_phys_bs=4),
    ModelConfig("gpt2-large",  "gpt2l_20h", 20, batch_size=4,  max_phys_bs=2),
]

GATE_TYPES = [
    ("none", 0.0),       # no gate (baseline)
    ("headwise", 0.5),   # gated with bias=+0.5
]


# ---------------------------------------------------------------------------
# σ search
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
            raise RuntimeError(f"Cannot satisfy ε={target_eps}")
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
# Run spec
# ---------------------------------------------------------------------------

@dataclass
class RunSpec:
    model_cfg: ModelConfig
    gate_type: str
    init_bias: float
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

    os.makedirs(args.results_dir, exist_ok=True)

    runs = []
    for mcfg in MODEL_CONFIGS:
        sample_rate = min(1.0, mcfg.batch_size / ds_len)
        print(f"\n{mcfg.label}: batch={mcfg.batch_size}, sample_rate={sample_rate:.6f}")

        for eps in args.eps:
            sigma = find_sigma(eps, args.steps, sample_rate, args.delta)
            print(f"  ε={eps}: σ={sigma:.4f}")

            for gate_type, init_bias in GATE_TYPES:
                for seed in args.seeds:
                    gate_label = "gated" if gate_type == "headwise" else "none"
                    csv_path = os.path.join(
                        args.results_dir,
                        f"scale_{mcfg.label}_{gate_label}_eps{eps}_seed{seed}.csv",
                    )
                    log_path = csv_path.replace(".csv", ".log")

                    cmd = [
                        "python", args.train_script,
                        "--hf-model", mcfg.name,
                        "--tokenizer", mcfg.name,
                        "--dataset", args.dataset,
                        "--dataset-config", args.dataset_config,
                        "--dataset-split", args.dataset_split,
                        "--seq-len", str(args.seq_len),
                        "--batch-size", str(mcfg.batch_size),
                        "--steps", str(args.steps),
                        "--noise-multiplier", f"{sigma:.6f}",
                        "--clip-norm", str(args.clip_norm),
                        "--lr", str(args.lr),
                        "--delta", str(args.delta),
                        "--eps-target", str(eps),
                        "--gate-type", gate_type,
                        "--init-gate-bias", str(init_bias),
                        "--gate-l1-lambda", "0.0",
                        "--seed", str(seed),
                        "--condition", f"{mcfg.label}_{gate_label}",
                        "--eval-every", str(args.eval_every),
                        "--eval-steps", str(args.eval_steps),
                        "--print-every", str(args.print_every),
                        "--metrics-csv", csv_path,
                        "--grad-sample-mode", "functorch",
                    ]
                    if mcfg.max_phys_bs > 0:
                        cmd += ["--max-physical-batch-size", str(mcfg.max_phys_bs)]

                    runs.append(RunSpec(
                        model_cfg=mcfg, gate_type=gate_type, init_bias=init_bias,
                        eps=eps, sigma=sigma, seed=seed,
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
        gate_label = "gated" if run.gate_type == "headwise" else "none"
        print(f"\n[queue] {run.model_cfg.label} {gate_label}  "
              f"ε={run.eps} σ={run.sigma:.3f} seed={run.seed}")
        print(f"        log → {run.log_path}")
        if dry_run:
            continue

        while not gpu_queue:
            time.sleep(10)
            still = []
            for proc, lf, r, gid in active:
                if proc.poll() is None:
                    still.append((proc, lf, r, gid))
                else:
                    lf.close()
                    gpu_queue.append(gid)
                    gl = "gated" if r.gate_type == "headwise" else "none"
                    print(f"[done ] {r.model_cfg.label} {gl}  gpu={gid}")
            active[:] = still

        gid = gpu_queue.pop(0)
        active.append(_launch(run, gid))
        gate_label = "gated" if run.gate_type == "headwise" else "none"
        print(f"[start] {run.model_cfg.label} {gate_label}  gpu={gid}")

    if not dry_run:
        while active:
            time.sleep(10)
            still = []
            for proc, lf, r, gid in active:
                if proc.poll() is None:
                    still.append((proc, lf, r, gid))
                else:
                    lf.close()
                    gpu_queue.append(gid)
                    gl = "gated" if r.gate_type == "headwise" else "none"
                    print(f"[done ] {r.model_cfg.label} {gl}  gpu={gid}")
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
            gate_label = "gated" if run.gate_type == "headwise" else "none"
            rows.append({
                "model": run.model_cfg.label,
                "num_heads": run.model_cfg.num_heads,
                "gate": gate_label,
                "eps_target": run.eps,
                "sigma": run.sigma,
                "seed": run.seed,
                "eval_ppl": float(last_eval["eval_ppl"]),
                "gate_sparsity": float(last_eval["gate_sparsity"]),
                "eps_consumed": float(last_eval["eps_consumed"]),
            })

    if not rows:
        print("No completed runs found for summary.")
        return

    rows.sort(key=lambda r: (r["num_heads"], r["gate"], r["eps_target"]))

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSummary written to: {out_path}")
    print(f"\n{'Model':<12} {'H':>3} {'Gate':>6} {'ε':>3}  "
          f"{'PPL':>8}  {'Sparsity':>9}  {'Δ PPL':>7}")
    print("-" * 65)

    # Compute gated - baseline PPL delta
    baseline_ppl = {}
    for r in rows:
        if r["gate"] == "none":
            baseline_ppl[(r["model"], r["eps_target"])] = r["eval_ppl"]

    for r in rows:
        key = (r["model"], r["eps_target"])
        delta_ppl = ""
        if r["gate"] == "gated" and key in baseline_ppl:
            d = r["eval_ppl"] - baseline_ppl[key]
            delta_ppl = f"{d:>+7.2f}"
        print(f"{r['model']:<12} {r['num_heads']:>3} {r['gate']:>6} "
              f"{r['eps_target']:>3.0f}  {r['eval_ppl']:>8.2f}  "
              f"{r['gate_sparsity']:>9.4f}  {delta_ppl}")

    # Phase transition check
    print("\n[SCALE TRANSITION CHECK]")
    for r in rows:
        if r["gate"] == "gated" and r["gate_sparsity"] > 0.01:
            print(f"  *** SPARSITY DETECTED: {r['model']} sparsity={r['gate_sparsity']:.4f} ***")
    else:
        has_any = any(r["gate_sparsity"] > 0.01 for r in rows if r["gate"] == "gated")
        if not has_any:
            print("  No sparsity detected at any scale. Negative result holds.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Scale-dependent transition (Exp 4)")

    p.add_argument("--train-script", default="gated_attention-main/dp_train_v2.py")
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--dataset-len", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--clip-norm", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--eps", type=str, default="8",
                   help="ε target(s). Default: 8 only for clearer signal.")
    p.add_argument("--seeds", type=str, default="7",
                   help="Single seed for fast validation. Use 7,42,123 for full run.")
    p.add_argument("--gpus", type=str, default="0,7")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--results-dir", default="results/v3/scale")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--summary-only", action="store_true")

    args = p.parse_args()
    args.eps = [float(v) for v in args.eps.split(",")]
    args.seeds = [int(v) for v in args.seeds.split(",")]
    gpus = [int(v) for v in args.gpus.split(",")]

    print("=" * 60)
    print("v3 Exp 4 — Scale-dependent phase transition")
    print("=" * 60)
    print("\nModels:")
    for m in MODEL_CONFIGS:
        print(f"  {m.label:<12} {m.name:<15} {m.num_heads:>2} heads  "
              f"batch={m.batch_size}  max_phys_bs={m.max_phys_bs}")
    print(f"\nGate types: {[g for g, _ in GATE_TYPES]}")
    print(f"ε targets:  {args.eps}")
    print(f"Seeds:      {args.seeds}")
    print(f"Total runs: {len(MODEL_CONFIGS) * len(GATE_TYPES) * len(args.eps) * len(args.seeds)}")
    print()

    if args.summary_only:
        runs = build_runs(args)
        write_summary(runs, os.path.join(args.results_dir, "summary.csv"))
        return

    runs = build_runs(args)

    if args.dry_run:
        print("\n[DRY RUN]")
        run_all(runs, gpus, dry_run=True)
    else:
        run_all(runs, gpus, dry_run=False)
        write_summary(runs, os.path.join(args.results_dir, "summary.csv"))


if __name__ == "__main__":
    main()
