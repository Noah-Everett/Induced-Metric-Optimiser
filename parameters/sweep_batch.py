#!/usr/bin/env python3
"""Batch runner for optimizer sweeps - runs multiple trials for multiple optimizers.

Usage::

    python sweep_batch.py --task sweep_mnist_mlp.py --iteration 0 --num_runs 5
    python sweep_batch.py --task sweep_mnist_mlp.py --optimisers adam sgd muon --backend local
    python sweep_batch.py --task sweep_mnist_mlp.py --iteration 1 --skip_completed
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from optimizer_registry import ALL_OPTIMIZERS

# Default configuration
DEFAULT_RESULTS_DIR = Path(__file__).parent.parent / "results"
DEFAULT_NUM_RUNS = 5
DEFAULT_TASK = "sweep_mnist_mlp.py"

# Default priority order (most interesting first)
DEFAULT_OPTIMIZERS = [
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


def check_missing_runs(optimizer, iteration, task_tag, num_runs, results_dir):
    """Check which run indices are missing for an optimizer in a specific iteration.

    Args:
        optimizer: Optimizer name
        iteration: Iteration/batch number
        task_tag: Task identifier (e.g., "mnist_mlp")
        num_runs: Total number of runs expected
        results_dir: Base results directory

    Returns:
        list: Indices of missing runs
    """
    runs_to_do = []
    for i in range(num_runs):
        result_file = results_dir / task_tag / optimizer / f"itr_{iteration}" / f"run_{i}.json"
        if not result_file.exists():
            runs_to_do.append(i)
    return runs_to_do


def run_sweep(optimizer, run_idx, iteration, task_script, backend, results_dir, verbose=False):
    """Run a single sweep for the given optimizer and run index.

    Args:
        optimizer: Optimizer name
        run_idx: Run index
        iteration: Iteration/batch number
        task_script: Path to the sweep task script
        backend: Backend to use ("wandb" or "local")
        results_dir: Base results directory
        verbose: If True, show full output from sweep script

    Returns:
        bool: True if successful, False if failed
    """
    cmd = [
        sys.executable,
        task_script,
        "--optimiser", optimizer,
        "--num_runs", "1",
        "--index", str(run_idx),
        "--iteration", str(iteration),
        "--backend", backend,
        "--results_dir", str(results_dir),
    ]

    # Suppress output unless verbose mode or if it fails
    if verbose:
        result = subprocess.run(cmd, capture_output=False)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Print error output if the command failed
            print(f"  ERROR OUTPUT:", flush=True)
            if result.stdout:
                print(result.stdout, flush=True)
            if result.stderr:
                print(result.stderr, flush=True)

    return result.returncode == 0


def main():
    """Run batch optimizer sweeps with full CLI support."""
    parser = argparse.ArgumentParser(
        description="Batch runner for optimizer sweeps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run default optimizers with 5 runs each (quiet mode)
  python sweep_batch.py --task sweep_mnist_mlp.py --iteration 0

  # Run specific optimizers with verbose output
  python sweep_batch.py --optimisers adam sgd muon --num_runs 10 --verbose

  # Skip already completed runs
  python sweep_batch.py --skip_completed --iteration 1

  # Use WandB backend
  python sweep_batch.py --backend wandb --task sweep_mnist_mlp.py
        """
    )

    # Task configuration
    parser.add_argument(
        "--task", type=str, default=DEFAULT_TASK,
        help=f"Sweep task script to run (default: {DEFAULT_TASK})"
    )
    parser.add_argument(
        "--iteration", type=int, default=0,
        help="Iteration/batch number for organizing multiple experiment runs (default: 0)"
    )

    # Optimizer selection
    parser.add_argument(
        "--optimisers", type=str, nargs="+", default=None,
        choices=ALL_OPTIMIZERS,
        help="List of optimizers to run (default: run all default optimizers)"
    )
    parser.add_argument(
        "--num_runs", type=int, default=DEFAULT_NUM_RUNS,
        help=f"Number of runs per optimizer (default: {DEFAULT_NUM_RUNS})"
    )

    # Backend configuration
    parser.add_argument(
        "--backend", type=str, default="local",
        choices=["wandb", "local"],
        help="Sweep backend: 'wandb' for W&B sweeps, 'local' for Optuna + JSON (default: local)"
    )
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help=f"Base directory for results (default: {DEFAULT_RESULTS_DIR})"
    )

    # Execution options
    parser.add_argument(
        "--skip_completed", action="store_true",
        help="Skip runs that already have results (default: False)"
    )
    parser.add_argument(
        "--continue_on_failure", action="store_true",
        help="Continue to next optimizer if one fails (default: False)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show full output from individual sweep runs (default: False)"
    )

    args = parser.parse_args()

    # Set up paths and configuration
    os.chdir(Path(__file__).parent)

    optimizers = args.optimisers if args.optimisers else DEFAULT_OPTIMIZERS
    results_dir = Path(args.results_dir) if args.results_dir else DEFAULT_RESULTS_DIR

    # Extract task tag from task script filename
    task_tag = Path(args.task).stem.replace("sweep_", "")

    # Print configuration
    print("=" * 60, flush=True)
    print("BATCH SWEEP CONFIGURATION", flush=True)
    print("=" * 60, flush=True)
    print(f"Task script:     {args.task}", flush=True)
    print(f"Task tag:        {task_tag}", flush=True)
    print(f"Iteration:       {args.iteration}", flush=True)
    print(f"Backend:         {args.backend}", flush=True)
    print(f"Results dir:     {results_dir}", flush=True)
    print(f"Runs per opt:    {args.num_runs}", flush=True)
    print(f"Optimizers:      {len(optimizers)} total", flush=True)
    print(f"Skip completed:  {args.skip_completed}", flush=True)
    print(f"Verbose output:  {args.verbose}", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

    # Run sweeps for each optimizer
    total_optimizers = len(optimizers)
    completed_optimizers = 0
    failed_optimizers = []

    for idx, opt in enumerate(optimizers, 1):
        print("=" * 60, flush=True)
        print(f"{timestamp()} [{idx}/{total_optimizers}] Starting: {opt}", flush=True)
        print("=" * 60, flush=True)

        # Check which run indices already exist
        if args.skip_completed:
            runs_to_do = check_missing_runs(opt, args.iteration, task_tag, args.num_runs, results_dir)
        else:
            runs_to_do = list(range(args.num_runs))

        # Skip if all runs are complete
        if not runs_to_do:
            print(f"  All {args.num_runs} runs already complete, skipping", flush=True)
            completed_optimizers += 1
            print(flush=True)
            continue

        # Run each index with progress bar
        optimizer_failed = False

        if args.verbose:
            print(f"  Running {len(runs_to_do)} runs: {runs_to_do}", flush=True)
            iterator = runs_to_do
        else:
            iterator = tqdm(runs_to_do, desc=f"  {opt}", unit="run", ncols=80, leave=True)

        for run_idx in iterator:
            success = run_sweep(opt, run_idx, args.iteration, args.task, args.backend, results_dir, args.verbose)

            if not success:
                if not args.verbose:
                    iterator.close()
                print(f"  FAILED: {opt} run {run_idx}", flush=True)
                optimizer_failed = True
                if not args.continue_on_failure:
                    print(f"  Stopping batch sweep due to failure", flush=True)
                    failed_optimizers.append(opt)
                    break

        if optimizer_failed:
            failed_optimizers.append(opt)
            if not args.continue_on_failure:
                break
        else:
            completed_optimizers += 1

        print(f"{timestamp()} Completed: {opt}", flush=True)
        print(flush=True)

    # Print summary
    print("=" * 60, flush=True)
    print(f"{timestamp()} BATCH SWEEP SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Total optimizers:     {total_optimizers}", flush=True)
    print(f"Completed:            {completed_optimizers}", flush=True)
    print(f"Failed:               {len(failed_optimizers)}", flush=True)
    if failed_optimizers:
        print(f"Failed optimizers:    {', '.join(failed_optimizers)}", flush=True)
    print("=" * 60, flush=True)

    # Exit with error code if any failures
    if failed_optimizers and not args.continue_on_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
