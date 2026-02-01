#!/usr/bin/env python3
"""Batch runner for optimizer sweeps - runs 5 trials each for priority optimizers."""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Configuration
RESULTS_DIR = Path(__file__).parent.parent / "results"
NUM_RUNS = 5

# Priority order (most interesting first)
OPTIMIZERS = [
    "adam",
    "adamw",
    "sgd",
    "muon",
    "sgd_metric",
    "sgd_log_metric",
    "sgd_rms",
    "sgd_learn_scalar",
    "sgd_learn_scalar_log",
    "sgd_learn_diag",
    "sgd_learn_diag_log",
    "sgd_offdiag_0_l",
    "sgd_offdiag_0_m",
    "sgd_offdiag_0_theta",
    "sgd_offdiag_l_0",
    "sgd_offdiag_l_l",
    "sgd_offdiag_l_m",
    "sgd_offdiag_l_theta",
    "sgd_offdiag_m_0",
    "sgd_offdiag_m_l",
    "sgd_offdiag_m_m",
    "sgd_offdiag_m_theta",
    "sgd_offdiag_theta_0",
    "sgd_offdiag_theta_l",
    "sgd_offdiag_theta_m",
    "sgd_offdiag_theta_theta",
]


def timestamp():
    """Return current time in HH:MM:SS format."""
    return datetime.now().strftime("%H:%M:%S")


def check_missing_runs(optimizer, iteration):
    """Check which run indices are missing for an optimizer in a specific iteration.

    Args:
        optimizer: Optimizer name
        iteration: Iteration/batch number

    Returns:
        list: Indices of missing runs (0-4)
    """
    runs_to_do = []
    for i in range(NUM_RUNS):
        result_file = RESULTS_DIR / "mnist_mlp" / optimizer / f"itr_{iteration}" / f"run_{i}.json"
        if not result_file.exists():
            runs_to_do.append(i)
    return runs_to_do


def run_sweep(optimizer, run_idx, iteration):
    """Run a single sweep for the given optimizer and run index.

    Args:
        optimizer: Optimizer name
        run_idx: Run index (0-4)
        iteration: Iteration/batch number

    Returns:
        bool: True if successful, False if failed
    """
    cmd = [
        sys.executable,
        "sweep_mnist_mlp.py",
        "--optimiser", optimizer,
        "--num_runs", "1",
        "--index", str(run_idx),
        "--iteration", str(iteration),
        "--backend", "local",
        "--results_dir", str(RESULTS_DIR),
    ]

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    """Run all optimizer sweeps."""
    parser = argparse.ArgumentParser(description="Batch runner for optimizer sweeps")
    parser.add_argument(
        "--iteration", type=int, default=0,
        help="Iteration/batch number for organizing multiple experiment runs (default: 0)"
    )
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)

    print(f"Running batch sweeps for iteration {args.iteration}", flush=True)
    print(flush=True)

    for opt in OPTIMIZERS:
        print("=" * 42, flush=True)
        print(f"{timestamp()} Starting: {opt}", flush=True)
        print("=" * 42, flush=True)

        # Check which run indices already exist
        runs_to_do = check_missing_runs(opt, args.iteration)

        # Skip if all runs are complete
        if not runs_to_do:
            print(f"  All {NUM_RUNS} runs already complete, skipping", flush=True)
            continue

        print(f"  Running {len(runs_to_do)} missing runs: {runs_to_do}", flush=True)

        # Run each missing index
        for run_idx in runs_to_do:
            success = run_sweep(opt, run_idx, args.iteration)
            if not success:
                print(f"  FAILED: {opt} run {run_idx}", flush=True)
                continue

        print(f"{timestamp()} Completed: {opt}", flush=True)
        print(flush=True)

    print("=" * 42, flush=True)
    print(f"{timestamp()} All sweeps completed!", flush=True)
    print("=" * 42, flush=True)


if __name__ == "__main__":
    main()
