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
import time
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
    "sgd_learn_diag_curv",
    "sgd_learn_diag_curv_log",
    "sgd_newton_diag",
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


def format_duration(seconds):
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


def count_existing_runs(optimizer, iteration, task_tag, results_dir):
    """Count how many run files already exist for an optimizer.

    Args:
        optimizer: Optimizer name
        iteration: Iteration/batch number
        task_tag: Task identifier (e.g., "mnist_mlp")
        results_dir: Base results directory

    Returns:
        int: Number of existing run files
    """
    run_dir = results_dir / task_tag / optimizer / f"itr_{iteration}"
    if not run_dir.exists():
        return 0
    return len(list(run_dir.glob("run_*.json")))


def extract_diagnostics(stdout):
    """Extract the first diagnostics section from output.

    Returns:
        dict: Extracted diagnostic info (backend, device, batches) or None
    """
    if not stdout:
        return None

    info = {}
    for line in stdout.split('\n'):
        if 'JAX backend:' in line:
            info['backend'] = line.split(':', 1)[1].strip()
        elif 'JAX devices:' in line:
            # Extract just the device type, e.g., "CudaDevice(id=0)" -> "CUDA:0"
            devices_str = line.split(':', 1)[1].strip()
            if 'CudaDevice' in devices_str:
                info['device'] = 'GPU (CUDA)'
            elif 'Metal' in devices_str:
                info['device'] = 'GPU (Metal)'
            else:
                info['device'] = devices_str
        elif 'Training batches:' in line:
            info['batches'] = line.split(':', 1)[1].strip()

    return info if info else None


def run_sweep(optimizer, num_runs, run_offset, iteration, task_script, backend,
              results_dir, verbose=False, show_diagnostics=False):
    """Run a sweep for the given optimizer with all runs in a single subprocess.

    Args:
        optimizer: Optimizer name
        num_runs: Number of runs to execute
        run_offset: Starting run index (for resuming)
        iteration: Iteration/batch number
        task_script: Path to the sweep task script
        backend: Backend to use ("wandb" or "local")
        results_dir: Base results directory
        verbose: If True, show full output from sweep script
        show_diagnostics: If True, extract and return diagnostics info

    Returns:
        tuple: (success: bool, diagnostics: dict or None)
    """
    cmd = [
        sys.executable,
        task_script,
        "--optimiser", optimizer,
        "--num_runs", str(num_runs),
        "--run_offset", str(run_offset),
        "--iteration", str(iteration),
        "--backend", backend,
        "--results_dir", str(results_dir),
    ]

    diagnostics = None

    if verbose:
        # Stream output directly to terminal
        result = subprocess.run(cmd)
    else:
        # Capture output
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Extract diagnostics if requested (only for first optimizer)
        if show_diagnostics and result.stdout:
            diagnostics = extract_diagnostics(result.stdout)

        # Show errors
        if result.returncode != 0:
            print(f"\n  ERROR:", flush=True)
            if result.stdout:
                # Show last 20 lines of stdout
                lines = result.stdout.strip().split('\n')
                for line in lines[-20:]:
                    print(f"    {line}", flush=True)
            if result.stderr:
                print(f"  STDERR:", flush=True)
                for line in result.stderr.strip().split('\n')[-10:]:
                    print(f"    {line}", flush=True)

    return result.returncode == 0, diagnostics


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

    # Print configuration (compact)
    print("=" * 70, flush=True)
    print(f"  BATCH SWEEP: {task_tag} | iteration {args.iteration} | {args.backend} backend", flush=True)
    print(f"  {len(optimizers)} optimizers x {args.num_runs} runs | results: {results_dir}", flush=True)
    print("=" * 70, flush=True)

    # Run sweeps for each optimizer
    total_optimizers = len(optimizers)
    completed_optimizers = 0
    skipped_optimizers = 0
    failed_optimizers = []
    batch_start_time = time.time()
    diagnostics_shown = False
    device_info = ""

    # Create progress bar
    pbar = tqdm(
        optimizers,
        desc="Starting...",
        unit="opt",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        leave=True,
    )

    for opt in pbar:
        # Check how many runs already exist
        existing = count_existing_runs(opt, args.iteration, task_tag, results_dir)

        if args.skip_completed:
            remaining = args.num_runs - existing
            run_offset = existing
            if remaining <= 0:
                pbar.set_description(f"{opt}: skipped")
                skipped_optimizers += 1
                continue
            status_note = f" +{existing}" if existing > 0 else ""
        else:
            remaining = args.num_runs
            run_offset = 0
            status_note = ""

        # Update progress bar description
        pbar.set_description(f"{opt}: {remaining} trials{status_note}")

        opt_start = time.time()
        success, diagnostics = run_sweep(
            opt, remaining, run_offset, args.iteration,
            args.task, args.backend, results_dir, args.verbose,
            show_diagnostics=(not diagnostics_shown)
        )
        elapsed = time.time() - opt_start

        if success:
            completed_optimizers += 1
            pbar.set_description(f"{opt}: done ({format_duration(elapsed)})")

            # Capture diagnostics once after first successful optimizer
            if diagnostics and not diagnostics_shown:
                diagnostics_shown = True
                device = diagnostics.get('device', diagnostics.get('backend', 'unknown'))
                batches = diagnostics.get('batches', 'unknown')
                device_info = f"  Device: {device} | Batches: {batches}"
        else:
            failed_optimizers.append(opt)
            pbar.set_description(f"{opt}: FAILED ({format_duration(elapsed)})")
            if not args.continue_on_failure:
                tqdm.write("  Stopping due to failure (use --continue_on_failure to proceed)")
                break

    pbar.close()

    # Print summary
    total_elapsed = time.time() - batch_start_time
    print("=" * 70, flush=True)
    if device_info:
        print(device_info, flush=True)
    print(f"  {completed_optimizers} completed, {skipped_optimizers} skipped, {len(failed_optimizers)} failed | {format_duration(total_elapsed)}", flush=True)
    if failed_optimizers:
        print(f"  Failed: {', '.join(failed_optimizers)}", flush=True)
    print("=" * 70, flush=True)

    # Exit with error code if any failures
    if failed_optimizers and not args.continue_on_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
