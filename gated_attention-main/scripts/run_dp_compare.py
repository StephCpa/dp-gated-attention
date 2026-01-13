#!/usr/bin/env python
import argparse
import os
import subprocess
import time
from dataclasses import dataclass


@dataclass
class RunSpec:
    eps: float
    sigma: float
    gate_type: str
    seed: int
    cmd: list[str]
    log_path: str


def _parse_csv(value, cast=str):
    return [cast(v.strip()) for v in value.split(",") if v.strip()]


def _compute_epsilon(steps, sample_rate, noise_multiplier, delta):
    from opacus.accountants import RDPAccountant

    accountant = RDPAccountant()
    for _ in range(steps):
        accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    return accountant.get_epsilon(delta)


def _find_sigma(target_eps, steps, sample_rate, delta, lo=0.1, hi=10.0, iters=30):
    while _compute_epsilon(steps, sample_rate, hi, delta) > target_eps:
        hi *= 2.0
        if hi > 512:
            raise RuntimeError("Sigma search failed; eps target is too small.")

    for _ in range(iters):
        mid = (lo + hi) / 2.0
        eps = _compute_epsilon(steps, sample_rate, mid, delta)
        if eps > target_eps:
            lo = mid
        else:
            hi = mid
    return hi


def _get_dataset_len(dataset, dataset_config, dataset_split):
    from datasets import load_dataset

    ds = load_dataset(dataset, dataset_config, split=dataset_split)
    return len(ds)


def build_runs(args):
    if args.dataset_len is None:
        dataset_len = _get_dataset_len(args.dataset, args.dataset_config, args.dataset_split)
    else:
        dataset_len = args.dataset_len

    if dataset_len <= 0:
        raise ValueError("dataset_len must be > 0")

    sample_rate = min(1.0, args.batch_size / dataset_len)

    runs = []
    for eps in args.eps:
        sigma = _find_sigma(eps, args.steps, sample_rate, args.delta)
        for gate_type in args.gate_types:
            for seed in args.seeds:
                log_name = f"eps{eps}_sigma{sigma:.4f}_{gate_type}_seed{seed}.log"
                log_path = os.path.join(args.log_dir, log_name)
                cmd = [
                    "python",
                    args.train_script,
                    "--model",
                    args.model,
                    "--hf-model",
                    args.hf_model,
                    "--tokenizer",
                    args.tokenizer,
                    "--dataset",
                    args.dataset,
                    "--dataset-config",
                    args.dataset_config,
                    "--dataset-split",
                    args.dataset_split,
                    "--seq-len",
                    str(args.seq_len),
                    "--batch-size",
                    str(args.batch_size),
                    "--steps",
                    str(args.steps),
                    "--noise-multiplier",
                    f"{sigma:.6f}",
                    "--clip-norm",
                    str(args.clip_norm),
                    "--lr",
                    str(args.lr),
                    "--use-opacus",
                    "--gate-type",
                    gate_type,
                    "--seed",
                    str(seed),
                    "--grad-sample-mode",
                    args.grad_sample_mode,
                    "--eval-every",
                    str(args.eval_every),
                    "--eval-steps",
                    str(args.eval_steps),
                ]
                if args.max_physical_batch_size > 0:
                    cmd += ["--max-physical-batch-size", str(args.max_physical_batch_size)]
                if not args.poisson_sampling:
                    cmd += ["--no-poisson-sampling"]
                if args.extra_args:
                    cmd += args.extra_args.split()

                runs.append(
                    RunSpec(
                        eps=eps,
                        sigma=sigma,
                        gate_type=gate_type,
                        seed=seed,
                        cmd=cmd,
                        log_path=log_path,
                    )
                )
    return runs


def run_parallel(runs, gpus, dry_run=False):
    os.makedirs(os.path.dirname(runs[0].log_path), exist_ok=True)

    gpu_queue = gpus[:]
    active = []

    def _launch(run, gpu_id):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        log_file = open(run.log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(run.cmd, env=env, stdout=log_file, stderr=log_file)
        return proc, log_file, run, gpu_id

    for run in runs:
        while not gpu_queue:
            time.sleep(5)
            still_active = []
            for proc, log_file, r, gpu_id in active:
                ret = proc.poll()
                if ret is None:
                    still_active.append((proc, log_file, r, gpu_id))
                else:
                    log_file.close()
                    gpu_queue.append(gpu_id)
            active = still_active

        gpu_id = gpu_queue.pop(0)
        cmd_str = " ".join(run.cmd)
        print(f"[launch] gpu={gpu_id} eps={run.eps} gate={run.gate_type} seed={run.seed}")
        print(f"  log={run.log_path}")
        print(f"  cmd={cmd_str}")
        if dry_run:
            gpu_queue.append(gpu_id)
            continue
        active.append(_launch(run, gpu_id))

    if dry_run:
        return

    while active:
        time.sleep(5)
        still_active = []
        for proc, log_file, r, gpu_id in active:
            ret = proc.poll()
            if ret is None:
                still_active.append((proc, log_file, r, gpu_id))
            else:
                log_file.close()
                gpu_queue.append(gpu_id)
        active = still_active


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-script", type=str, default="gated_attention-main/dp_sgd_train_minimal.py")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--hf-model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--dataset-len", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--eps", type=str, default="3,8")
    parser.add_argument("--gate-types", type=str, default="headwise,none")
    parser.add_argument("--seeds", type=str, default="7")
    parser.add_argument("--gpus", type=str, default="2,3,4,5")
    parser.add_argument("--log-dir", type=str, default="logs/dp_compare")
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--grad-sample-mode", type=str, default="functorch")
    parser.add_argument("--poisson-sampling", action="store_true", default=True)
    parser.add_argument("--no-poisson-sampling", action="store_false", dest="poisson_sampling")
    parser.add_argument("--max-physical-batch-size", type=int, default=1)
    parser.add_argument("--extra-args", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.eps = [float(v) for v in _parse_csv(args.eps, float)]
    args.gate_types = _parse_csv(args.gate_types, str)
    args.seeds = _parse_csv(args.seeds, int)
    gpus = _parse_csv(args.gpus, int)
    if not gpus:
        raise ValueError("No GPUs provided via --gpus")

    runs = build_runs(args)
    run_parallel(runs, gpus, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
