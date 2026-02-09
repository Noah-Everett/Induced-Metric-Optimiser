"""
Shared utilities for analysis notebooks.

Supports both WandB and local JSON backends for loading sweep results.

Usage::

    from analysis_utils import (
        load_best_runs,
        load_top_n_runs,
        extract_history,
        compute_wall_times,
        plot_2d_histograms,
        plot_training_curves,
        plot_speedrun_results,
        plot_time_to_best,
        plot_convergence_heatmap,
        plot_efficiency_scatter,
        plot_ranking_bar,
        plot_performance_summary,
        print_speedrun_table,
        print_best_hyperparameters,
        save_best_hyperparameters,
        get_optimizer_colors,
        ALL_OPTIMIZERS,
    )
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for optimizer_registry import
sys.path.insert(0, str(Path(__file__).parent.parent / "parameters"))

from optimizer_registry import get_optimizer_colors, ALL_OPTIMIZERS


# ---------------------------------------------------------------------------
# Data loading - WandB backend
# ---------------------------------------------------------------------------

def _load_best_runs_wandb(project, task_tag, run_index, optimizers,
                          entity=None, sort_metric="sweep_metric", sort_order="+"):
    """Load best runs from WandB API."""
    from wandb.apis.public import Api

    api = Api()
    best_runs = {}

    for optimizer in optimizers:
        print(f"Finding best run for {optimizer}...")

        runs = api.runs(
            f"{entity}/{project}" if entity else project,
            filters={
                "$and": [
                    {"tags": task_tag},
                    {"tags": f"run_{run_index}"},
                    {"tags": optimizer},
                ]
            },
            order=f"{sort_order}summary_metrics.{sort_metric}",
        )

        if len(runs) > 0:
            best_run = runs[0]
            best_runs[optimizer] = {
                "run": best_run,
                "config": {k: v for k, v in best_run.config.items()
                           if not k.startswith("_")},
                "summary": dict(best_run.summary),
                "name": best_run.name,
            }
            print(f"  {optimizer}: {best_run.name}")
        else:
            print(f"  {optimizer}: No runs found")

    return best_runs


def _load_top_n_runs_wandb(project, task_tag, run_index, optimizers,
                           n=50, entity=None, sort_metric="sweep_metric", sort_order="+"):
    """Load top N runs for each optimizer from WandB API."""
    from wandb.apis.public import Api

    api = Api()
    top_n_runs = {}

    for optimizer in optimizers:
        print(f"Loading top {n} runs for {optimizer}...")

        runs = api.runs(
            f"{entity}/{project}" if entity else project,
            filters={
                "$and": [
                    {"tags": task_tag},
                    {"tags": f"run_{run_index}"},
                    {"tags": optimizer},
                ]
            },
            order=f"{sort_order}summary_metrics.{sort_metric}",
        )

        top_n_runs[optimizer] = runs[:n]
        print(f"  {optimizer}: {len(top_n_runs[optimizer])} runs")

    return top_n_runs


def _extract_history_wandb(best_runs, metric_keys):
    """Extract training history from WandB runs.

    Missing metric values are stored as ``np.nan`` to keep all arrays
    aligned with the epoch array.
    """
    optimizer_data = {}

    for optimizer, run_info in best_runs.items():
        print(f"Processing {optimizer}...")
        run = run_info["run"]

        history = run.scan_history()

        data = {key: [] for key in metric_keys}
        data["epoch"] = []

        for row in history:
            # Support both "epoch" (training sweeps) and "iteration" (small_examples)
            epoch_val = row.get("epoch", row.get("iteration", None))
            if epoch_val is not None:
                data["epoch"].append(epoch_val)
                for key in metric_keys:
                    data[key].append(row.get(key, np.nan))

        for key in data:
            data[key] = np.array(data[key], dtype=float)

        optimizer_data[optimizer] = data

    return optimizer_data


# ---------------------------------------------------------------------------
# Data loading - Local backend
# ---------------------------------------------------------------------------

def _load_local_runs(results_dir, task_tag, optimizer, iteration=0):
    """Load all runs for an optimizer from local JSON files.

    Args:
        results_dir: Base results directory
        task_tag: Task identifier (e.g., "mnist_mlp")
        optimizer: Optimizer name
        iteration: Iteration/batch number (default: 0)

    Returns:
        list: List of run data dicts, or empty list if no files found
    """
    itr_dir = Path(results_dir) / task_tag / optimizer / f"itr_{iteration}"

    if not itr_dir.exists():
        return []

    runs = []
    for result_file in sorted(itr_dir.glob("run_*.json")):
        with open(result_file) as f:
            data = json.load(f)
            data["_file"] = str(result_file)
            runs.append(data)

    return runs


def _load_best_runs_local(results_dir, task_tag, optimizers,
                          metric_key="sweep_metric", direction="minimize",
                          iteration=0):
    """Load best runs from local JSON files.

    Args:
        results_dir: Base results directory
        task_tag: Task identifier
        optimizers: List of optimizer names
        metric_key: Metric for sorting
        direction: "minimize" or "maximize"
        iteration: Iteration/batch number
    """
    best_runs = {}

    for optimizer in optimizers:
        print(f"Finding best run for {optimizer}...")

        runs = _load_local_runs(results_dir, task_tag, optimizer, iteration=iteration)

        if runs:
            key_fn = lambda r: r["summary"].get(
                metric_key, float("inf") if direction == "minimize" else float("-inf")
            )
            runs_sorted = sorted(runs, key=key_fn, reverse=(direction == "maximize"))
            best = runs_sorted[0]

            best_runs[optimizer] = {
                "config": best["config"],
                "summary": best["summary"],
                "history": best["history"],
                "name": Path(best["_file"]).stem,
            }
            print(f"  {optimizer}: {best_runs[optimizer]['name']}")
        else:
            print(f"  {optimizer}: No runs found")

    return best_runs


def _load_top_n_runs_local(results_dir, task_tag, optimizers,
                           n=50, metric_key="sweep_metric", direction="minimize",
                           iteration=0):
    """Load top N runs for each optimizer from local JSON files.

    Args:
        results_dir: Base results directory
        task_tag: Task identifier
        optimizers: List of optimizer names
        n: Number of top runs to return
        metric_key: Metric for sorting
        direction: "minimize" or "maximize"
        iteration: Iteration/batch number
    """
    top_n_runs = {}

    for optimizer in optimizers:
        print(f"Loading top {n} runs for {optimizer}...")

        runs = _load_local_runs(results_dir, task_tag, optimizer, iteration=iteration)

        if runs:
            key_fn = lambda r: r["summary"].get(
                metric_key, float("inf") if direction == "minimize" else float("-inf")
            )
            runs_sorted = sorted(runs, key=key_fn, reverse=(direction == "maximize"))
            top_n_runs[optimizer] = runs_sorted[:n]
            print(f"  {optimizer}: {len(top_n_runs[optimizer])} runs")
        else:
            top_n_runs[optimizer] = []
            print(f"  {optimizer}: No runs found")

    return top_n_runs


def _extract_history_local(best_runs, metric_keys):
    """Extract training history from local run data.

    Keeps all arrays aligned to the epoch array.  Missing metric values
    are stored as ``np.nan`` so downstream code can use ``np.nanmean``
    etc. without length mismatches.
    """
    optimizer_data = {}

    for optimizer, run_info in best_runs.items():
        print(f"Processing {optimizer}...")

        history = run_info.get("history", [])

        data = {key: [] for key in metric_keys}
        data["epoch"] = []

        for row in history:
            # Support both "epoch" (training sweeps) and "iteration" (small_examples)
            epoch_val = row.get("epoch", row.get("iteration", None))
            if epoch_val is not None:
                data["epoch"].append(epoch_val)
                for key in metric_keys:
                    data[key].append(row.get(key, np.nan))

        for key in data:
            data[key] = np.array(data[key], dtype=float)

        optimizer_data[optimizer] = data

    return optimizer_data


# ---------------------------------------------------------------------------
# Unified data loading API
# ---------------------------------------------------------------------------

def load_best_runs(
    backend,
    optimizers,
    task_tag=None,
    project=None,
    entity=None,
    results_dir="results",
    metric_key="sweep_metric",
    direction="minimize",
    sort_metric=None,
    sort_order=None,
    iteration=0,
):
    """
    Load best runs for each optimizer.

    Parameters
    ----------
    backend : str
        "wandb" or "local"
    optimizers : list
        List of optimizer names
    task_tag : str
        Task tag (e.g., "mnist_mlp", "small_examples_beale")
    project : str
        WandB project name (wandb backend only)
    entity : str
        WandB entity name (wandb backend only)
    results_dir : str
        Base directory for local results (local backend only)
    metric_key : str
        Metric key for sorting (local backend)
    direction : str
        "minimize" or "maximize" (local backend)
    sort_metric : str, optional
        WandB summary metric to sort by (default: metric_key)
    sort_order : str, optional
        "+" ascending or "-" descending (default: "+" for minimize, "-" for maximize)
    iteration : int
        Iteration/batch number (default: 0).
        For WandB: used as tag filter (run_{iteration}).
        For local: used as folder path (itr_{iteration}/).

    Returns
    -------
    dict
        Mapping from optimizer name to run info dict
    """
    if backend == "wandb":
        if sort_metric is None:
            sort_metric = metric_key
        if sort_order is None:
            sort_order = "+" if direction == "minimize" else "-"
        return _load_best_runs_wandb(
            project, task_tag, iteration, optimizers, entity, sort_metric, sort_order
        )
    else:
        return _load_best_runs_local(
            results_dir, task_tag, optimizers, metric_key, direction,
            iteration=iteration
        )


def load_top_n_runs(
    backend,
    optimizers,
    n=50,
    task_tag=None,
    project=None,
    entity=None,
    results_dir="results",
    metric_key="sweep_metric",
    direction="minimize",
    sort_metric=None,
    sort_order=None,
    iteration=0,
):
    """
    Load top N runs for each optimizer.

    Parameters
    ----------
    backend : str
        "wandb" or "local"
    optimizers : list
        List of optimizer names
    n : int
        Number of top runs to load
    task_tag : str
        Task tag (e.g., "mnist_mlp", "small_examples_beale")
    project : str
        WandB project name (wandb backend only)
    entity : str
        WandB entity name (wandb backend only)
    results_dir : str
        Base directory for local results (local backend only)
    metric_key : str
        Metric key for sorting (local backend)
    direction : str
        "minimize" or "maximize" (local backend)
    sort_metric : str, optional
        WandB summary metric to sort by (default: metric_key)
    sort_order : str, optional
        "+" ascending or "-" descending (default: "+" for minimize, "-" for maximize)
    iteration : int
        Iteration/batch number (default: 0).
        For WandB: used as tag filter (run_{iteration}).
        For local: used as folder path (itr_{iteration}/).

    Returns
    -------
    dict
        Mapping from optimizer name to list of runs
    """
    if backend == "wandb":
        if sort_metric is None:
            sort_metric = metric_key
        if sort_order is None:
            sort_order = "+" if direction == "minimize" else "-"
        return _load_top_n_runs_wandb(
            project, task_tag, iteration, optimizers, n, entity, sort_metric, sort_order
        )
    else:
        return _load_top_n_runs_local(
            results_dir, task_tag, optimizers, n, metric_key, direction,
            iteration=iteration
        )


def extract_history(backend, best_runs, metric_keys):
    """
    Extract training history from best runs.

    Parameters
    ----------
    backend : str
        "wandb" or "local"
    best_runs : dict
        Result from load_best_runs()
    metric_keys : list
        List of metric keys to extract (e.g., ["train_loss", "test_acc"])

    Returns
    -------
    dict
        Mapping from optimizer name to data dict with numpy arrays
    """
    if backend == "wandb":
        return _extract_history_wandb(best_runs, metric_keys)
    else:
        return _extract_history_local(best_runs, metric_keys)


# ---------------------------------------------------------------------------
# Wall time computation
# ---------------------------------------------------------------------------

def compute_wall_times(train_time_values):
    """Convert train_time_seconds values to wall-clock seconds from epoch 0.

    The sweep scripts accumulate time as::

        epoch_start = time.time()
        ... training ...
        train_time += time.time() - epoch_start

    which produces a monotonically increasing cumulative value.  This helper
    normalizes those values to start from 0.

    Parameters
    ----------
    train_time_values : array-like
        Cumulative ``train_time_seconds`` values from training history.

    Returns
    -------
    np.ndarray
        Wall-clock seconds starting from 0.
    """
    vals = np.asarray(train_time_values, dtype=float)
    if len(vals) == 0:
        return np.zeros(0)
    # Normalize to start from 0
    return vals - vals[0]


def add_wall_times(optimizer_data, time_key="train_time_seconds"):
    """Add ``wall_times`` to each entry in optimizer_data in-place.

    When *time_key* is sparse (contains NaN for epochs where timing was
    not logged), the function linearly interpolates to fill all epochs.

    Parameters
    ----------
    optimizer_data : dict
        Result from extract_history().  Must contain *time_key* arrays.
    time_key : str
        Key for the raw timing values.

    Returns
    -------
    dict
        The same optimizer_data dict, modified in place.
    """
    for opt, data in optimizer_data.items():
        raw = data.get(time_key, np.array([]))
        epochs = data.get("epoch", np.array([]))
        n_epochs = len(epochs)

        if len(raw) == 0 or n_epochs == 0:
            data["wall_times"] = np.arange(n_epochs, dtype=float)
            continue

        # Find valid (non-NaN) time entries
        valid_mask = ~np.isnan(raw)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            data["wall_times"] = np.arange(n_epochs, dtype=float)
            continue

        # Compute wall times for valid entries
        valid_times = compute_wall_times(raw[valid_mask])

        if len(valid_indices) == n_epochs:
            # All entries have timing — no interpolation needed
            data["wall_times"] = valid_times
        else:
            # Interpolate to fill all epochs
            data["wall_times"] = np.interp(
                np.arange(n_epochs), valid_indices, valid_times
            )

    return optimizer_data


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def plot_2d_histograms(
    top_n_runs,
    backend,
    x_metric,
    y_metric,
    optimizers=None,
    figsize=(20, 15),
    title_prefix="",
):
    """
    Create 2D histograms for each optimizer showing distribution of two metrics.

    Parameters
    ----------
    top_n_runs : dict
        Result from load_top_n_runs()
    backend : str
        "wandb" or "local"
    x_metric : str
        Metric name for x-axis
    y_metric : str
        Metric name for y-axis
    optimizers : list, optional
        Subset of optimizers to plot (defaults to all in top_n_runs)
    figsize : tuple
        Figure size
    title_prefix : str
        Prefix for subplot titles

    Returns
    -------
    Figure
        Matplotlib figure
    """
    if optimizers is None:
        optimizers = list(top_n_runs.keys())

    n_plots = len(optimizers)
    ncols = min(4, n_plots)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Collect global ranges
    all_x, all_y = [], []
    for opt in optimizers:
        for run in top_n_runs.get(opt, []):
            if backend == "wandb":
                x_val = run.summary.get(x_metric)
                y_val = run.summary.get(y_metric)
            else:
                x_val = run["summary"].get(x_metric)
                y_val = run["summary"].get(y_metric)
            if x_val is not None and y_val is not None:
                all_x.append(x_val)
                all_y.append(y_val)

    if all_x and all_y:
        x_range = max(all_x) - min(all_x)
        y_range = max(all_y) - min(all_y)
        x_bins = np.linspace(min(all_x) - x_range * 0.05, max(all_x) + x_range * 0.05, 21)
        y_bins = np.linspace(min(all_y) - y_range * 0.05, max(all_y) + y_range * 0.05, 21)

    for i, opt in enumerate(optimizers):
        runs = top_n_runs.get(opt, [])
        x_vals, y_vals = [], []

        for run in runs:
            if backend == "wandb":
                x_val = run.summary.get(x_metric)
                y_val = run.summary.get(y_metric)
            else:
                x_val = run["summary"].get(x_metric)
                y_val = run["summary"].get(y_metric)
            if x_val is not None and y_val is not None:
                x_vals.append(x_val)
                y_vals.append(y_val)

        if x_vals:
            h = axes[i].hist2d(x_vals, y_vals, bins=[x_bins, y_bins], cmap="viridis", alpha=0.8)
            plt.colorbar(h[3], ax=axes[i])

        axes[i].set_title(f"{title_prefix}{opt.upper()}\n(n={len(x_vals)})")
        axes[i].set_xlabel(x_metric)
        axes[i].set_ylabel(y_metric)
        axes[i].grid(True, alpha=0.3)

    for j in range(len(optimizers), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    return fig


def plot_training_curves(
    optimizer_data,
    x_keys,
    y_keys,
    optimizers=None,
    colors=None,
    figsize=(15, 10),
    marker="o-",
    log_scale=False,
):
    """
    Plot training curves for multiple optimizers.

    Parameters
    ----------
    optimizer_data : dict
        Result from extract_history()
    x_keys : list
        X-axis metric for each subplot column (e.g., ["wall_times", "epoch"])
    y_keys : list
        Y-axis metric for each subplot row (e.g., ["train_acc", "test_acc"])
    optimizers : list, optional
        Subset of optimizers to plot (defaults to all in optimizer_data)
    colors : dict, optional
        Mapping from optimizer name to color (defaults to registry colors)
    figsize : tuple
        Figure size
    marker : str
        Matplotlib marker style
    log_scale : bool
        Use log-log scale on all axes

    Returns
    -------
    Figure
        Matplotlib figure
    """
    if optimizers is None:
        optimizers = list(optimizer_data.keys())

    if colors is None:
        colors = get_optimizer_colors(optimizers)

    nrows = len(y_keys)
    ncols = len(x_keys)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for row, y_key in enumerate(y_keys):
        for col, x_key in enumerate(x_keys):
            ax = axes[row, col]

            for opt in optimizers:
                data = optimizer_data.get(opt, {})
                x = data.get(x_key, data.get("epoch", []))
                y = data.get(y_key, [])

                if len(x) > 0 and len(y) > 0:
                    min_len = min(len(x), len(y))
                    plot_fn = ax.loglog if log_scale else ax.plot
                    plot_fn(x[:min_len], y[:min_len], marker,
                            linewidth=2, markersize=4, color=colors.get(opt), label=opt)

            ax.set_xlabel(x_key)
            ax.set_ylabel(y_key)
            ax.set_title(f"{y_key} vs {x_key}")
            ax.grid(True, alpha=0.3)
            ax.legend()

    plt.tight_layout()
    return fig


def plot_time_to_best(
    best_runs,
    optimizer_data,
    best_epoch_key,
    best_metric_key=None,
    time_key="wall_times",
    optimizers=None,
    colors=None,
    figsize=(15, 6),
):
    """
    Plot wall-time and epoch of best metric for each optimizer's best run.

    Parameters
    ----------
    best_runs : dict
        Result from load_best_runs()
    optimizer_data : dict
        Result from extract_history() with wall_times added
    best_epoch_key : str
        Summary key for the epoch of best metric (e.g., "max_acc_epoch")
    best_metric_key : str, optional
        Summary key for the best metric value to print
    time_key : str
        Key in optimizer_data for time values
    optimizers : list, optional
        Subset of optimizers
    colors : dict, optional
        Color mapping
    figsize : tuple
        Figure size

    Returns
    -------
    Figure
        Matplotlib figure
    """
    if optimizers is None:
        optimizers = [o for o in best_runs if o in optimizer_data]

    if colors is None:
        colors = get_optimizer_colors(optimizers)

    best_times = {}
    print(f"Time to reach best metric for each optimizer:")
    print("=" * 70)

    for opt in optimizers:
        run_info = best_runs.get(opt)
        data = optimizer_data.get(opt)
        if run_info is None or data is None:
            continue

        best_epoch = run_info["summary"].get(best_epoch_key)
        if best_epoch is None:
            continue

        epochs = data.get("epoch", [])
        times = data.get(time_key, [])

        epoch_indices = np.where(np.asarray(epochs) == best_epoch)[0]
        if len(epoch_indices) > 0 and epoch_indices[0] < len(times):
            idx = epoch_indices[0]
            wall_time = times[idx]
            metric_val = ""
            if best_metric_key:
                vals = data.get(best_metric_key, [])
                if idx < len(vals):
                    metric_val = f" ({best_metric_key}: {vals[idx]:.6f})"

            best_times[opt] = {"epoch": best_epoch, "wall_time": wall_time}
            print(f"  {opt.upper():<20}: {wall_time:>8.2f}s at epoch {int(best_epoch):>3d}{metric_val}")

    if not best_times:
        print("  No data available.")
        return plt.figure()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    opts = list(best_times.keys())
    wall_times = [best_times[o]["wall_time"] for o in opts]
    epochs = [best_times[o]["epoch"] for o in opts]
    clrs = [colors.get(o, "#333333") for o in opts]

    bars1 = ax1.bar(range(len(opts)), wall_times, color=clrs, alpha=0.7)
    ax1.set_xlabel("Optimizer")
    ax1.set_ylabel("Wall Time (seconds)")
    ax1.set_title("Time to Reach Best Metric")
    ax1.set_xticks(range(len(opts)))
    ax1.set_xticklabels([o.upper() for o in opts], rotation=45, ha="right")
    ax1.grid(True, alpha=0.3)
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.2f}s", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    bars2 = ax2.bar(range(len(opts)), epochs, color=clrs, alpha=0.7)
    ax2.set_xlabel("Optimizer")
    ax2.set_ylabel("Epoch")
    ax2.set_title("Epoch of Best Metric")
    ax2.set_xticks(range(len(opts)))
    ax2.set_xticklabels([o.upper() for o in opts], rotation=45, ha="right")
    ax2.grid(True, alpha=0.3)
    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    return fig


def plot_speedrun_results(
    optimizer_data,
    targets,
    metric_key,
    direction,
    time_key="wall_times",
    optimizers=None,
    colors=None,
    figsize=(15, 6),
):
    """
    Plot time to reach target metrics (speedrun analysis).

    Parameters
    ----------
    optimizer_data : dict
        Result from extract_history() with wall_times added
    targets : list
        List of target values (e.g., [0.9, 0.95, 0.99] for accuracy)
    metric_key : str
        Metric to compare against targets
    direction : str
        "above" (metric >= target) or "below" (metric <= target)
    time_key : str
        Key for time values in optimizer_data
    optimizers : list, optional
        Subset of optimizers to plot
    colors : dict, optional
        Mapping from optimizer name to color
    figsize : tuple
        Figure size

    Returns
    -------
    Figure
        Matplotlib figure
    """
    if optimizers is None:
        optimizers = list(optimizer_data.keys())

    if colors is None:
        colors = get_optimizer_colors(optimizers)

    results = compute_speedrun_results(optimizer_data, targets, metric_key, direction, time_key, optimizers)

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    x = np.arange(len(targets))
    width = 0.8 / max(len(optimizers), 1)

    for i, opt in enumerate(optimizers):
        times = [results[opt].get(t) for t in targets]
        times_plot = [t if t is not None else 0 for t in times]
        offset = (i - len(optimizers) / 2 + 0.5) * width
        ax.bar(x + offset, times_plot, width, label=opt, color=colors.get(opt), alpha=0.8)

    ax.set_xlabel("Target")
    ax.set_ylabel("Time to reach target")
    ax.set_title(f"Speedrun: Time to reach {metric_key} targets")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}" for t in targets])
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def compute_speedrun_results(optimizer_data, targets, metric_key, direction,
                             time_key="wall_times", optimizers=None):
    """Compute time-to-target for each optimizer.

    Returns
    -------
    dict
        ``{optimizer: {target: wall_time_or_None, ...}, ...}``
    """
    if optimizers is None:
        optimizers = list(optimizer_data.keys())

    results = {}
    for opt in optimizers:
        data = optimizer_data.get(opt, {})
        metric_vals = np.asarray(data.get(metric_key, []))
        time_vals = np.asarray(data.get(time_key, data.get("epoch", [])))

        results[opt] = {}
        for target in targets:
            if direction == "above":
                indices = np.where(metric_vals >= target)[0]
            else:
                indices = np.where(metric_vals <= target)[0]

            if len(indices) > 0 and indices[0] < len(time_vals):
                results[opt][target] = float(time_vals[indices[0]])
            else:
                results[opt][target] = None

    return results


def print_speedrun_table(optimizer_data, targets, metric_key, direction,
                         time_key="wall_times", optimizers=None):
    """Print a formatted speedrun results table."""
    if optimizers is None:
        optimizers = list(optimizer_data.keys())

    results = compute_speedrun_results(optimizer_data, targets, metric_key, direction, time_key, optimizers)

    header = f"{'Optimizer':<25}"
    for t in targets:
        header += f"{t:<10}"
    print(header)
    print("-" * (25 + 10 * len(targets)))

    for opt in optimizers:
        row = f"{opt:<25}"
        for t in targets:
            val = results[opt].get(t)
            row += f"{val:<10.1f}" if val is not None else f"{'--':<10}"
        print(row)


def plot_convergence_heatmap(
    matrix=None,
    row_labels=None,
    col_labels=None,
    optimizer_data=None,
    targets=None,
    metric_key=None,
    direction=None,
    optimizers=None,
    ax=None,
    figsize=(8, 6),
    cmap="RdYlGn",
    title="Convergence Rate Heatmap",
    xlabel=None,
    ylabel="Optimizer",
    show_colorbar=True,
    annotation_fmt="percent",
    vmin=0,
    vmax=1,
):
    """
    Plot a heatmap showing convergence rates or threshold achievement.

    Can be called in two modes:
    1. Direct mode: Provide `matrix`, `row_labels`, `col_labels`
    2. Compute mode: Provide `optimizer_data`, `targets`, `metric_key`, `direction`

    Parameters
    ----------
    matrix : array-like or DataFrame, optional
        Pre-computed matrix of values (rows=optimizers, cols=targets/functions)
    row_labels : list, optional
        Labels for rows (optimizers)
    col_labels : list, optional
        Labels for columns (targets/functions)
    optimizer_data : dict, optional
        Result from extract_history() - used in compute mode
    targets : list, optional
        Target thresholds - used in compute mode
    metric_key : str, optional
        Metric to check against targets - used in compute mode
    direction : str, optional
        "above" or "below" - used in compute mode
    optimizers : list, optional
        Subset of optimizers
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    figsize : tuple
        Figure size
    cmap : str
        Colormap
    title : str
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str
        Y-axis label
    show_colorbar : bool
        Whether to show colorbar
    annotation_fmt : str
        "percent" for percentage, "check" for ✓/✗, or format string
    vmin, vmax : float
        Value range for colormap

    Returns
    -------
    Figure or None
    """
    # Compute mode: build matrix from optimizer_data
    if matrix is None and optimizer_data is not None:
        if optimizers is None:
            optimizers = list(optimizer_data.keys())

        convergence_matrix = []
        valid_optimizers = []
        for opt in optimizers:
            if opt not in optimizer_data:
                continue
            valid_optimizers.append(opt)
            row = []
            data = optimizer_data[opt]
            metric_vals = data.get(metric_key, [])

            for target in targets:
                if direction == "above":
                    reached = any(v >= target for v in metric_vals)
                else:
                    reached = any(v <= target for v in metric_vals)
                row.append(1.0 if reached else 0.0)
            convergence_matrix.append(row)

        if not convergence_matrix:
            return None

        matrix = np.array(convergence_matrix)
        row_labels = valid_optimizers
        col_labels = [f"{t:.0%}" if t < 1 else f"{t}" for t in targets]
        if xlabel is None:
            xlabel = "Target Threshold"

    # Convert DataFrame to array if needed
    if hasattr(matrix, 'values'):
        if row_labels is None:
            row_labels = list(matrix.index)
        if col_labels is None:
            col_labels = list(matrix.columns)
        matrix = matrix.values

    if matrix is None or len(matrix) == 0:
        return None

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if show_colorbar:
        plt.colorbar(im, ax=ax)

    # Add annotations
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            if annotation_fmt == "percent":
                text = f"{val:.0%}"
                fontsize = 7
            elif annotation_fmt == "check":
                text = '✓' if val == 1.0 else '✗'
                fontsize = 10
            else:
                text = f"{val:{annotation_fmt}}"
                fontsize = 7
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=fontsize, color=color)

    return fig


def plot_efficiency_scatter(
    data=None,
    x_col=None,
    y_col=None,
    group_col=None,
    best_runs=None,
    optimizer_data=None,
    epoch_key=None,
    time_key="wall_times",
    optimizers=None,
    colors=None,
    ax=None,
    figsize=(8, 6),
    log_scale=True,
    title="Efficiency: Iterations vs Runtime",
    xlabel=None,
    ylabel=None,
    marker_size=80,
    alpha=0.7,
):
    """
    Plot a scatter showing efficiency (iterations/epochs vs runtime).

    Can be called in two modes:
    1. DataFrame mode: Provide `data`, `x_col`, `y_col`, `group_col`
    2. Best runs mode: Provide `best_runs`, `optimizer_data`, `epoch_key`

    Parameters
    ----------
    data : DataFrame, optional
        DataFrame with scatter data
    x_col : str, optional
        Column name for x-axis
    y_col : str, optional
        Column name for y-axis
    group_col : str, optional
        Column name for grouping (optimizer)
    best_runs : dict, optional
        Result from load_best_runs()
    optimizer_data : dict, optional
        Result from extract_history()
    epoch_key : str, optional
        Summary key for epoch of best metric
    time_key : str
        Key for time values
    optimizers : list, optional
        Subset of optimizers
    colors : dict, optional
        Color mapping
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    figsize : tuple
        Figure size
    log_scale : bool
        Use log-log scale
    title : str
        Plot title
    xlabel, ylabel : str, optional
        Axis labels
    marker_size : int
        Scatter marker size
    alpha : float
        Marker transparency

    Returns
    -------
    Figure or None
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # DataFrame mode
    if data is not None:
        if group_col:
            groups = data[group_col].unique()
            if colors is None:
                colors = get_optimizer_colors(list(groups))
            for grp in groups:
                grp_data = data[data[group_col] == grp]
                ax.scatter(grp_data[x_col], grp_data[y_col],
                           label=grp, color=colors.get(grp, 'gray'),
                           s=marker_size, alpha=alpha)
        else:
            ax.scatter(data[x_col], data[y_col], s=marker_size, alpha=alpha)

        if xlabel is None:
            xlabel = x_col
        if ylabel is None:
            ylabel = y_col

    # Best runs mode
    elif best_runs is not None and optimizer_data is not None:
        if optimizers is None:
            optimizers = [o for o in best_runs if o in optimizer_data]

        if colors is None:
            colors = get_optimizer_colors(optimizers)

        for opt in optimizers:
            if opt not in best_runs or opt not in optimizer_data:
                continue

            summary = best_runs[opt].get("summary", {})
            epochs_to_best = summary.get(epoch_key, 0)

            opt_data = optimizer_data[opt]
            wall_times = opt_data.get(time_key, [])
            n_times = len(wall_times)
            best_idx = min(epochs_to_best, n_times - 1) if n_times > 0 else 0
            time_to_best = wall_times[best_idx] if n_times > 0 and best_idx < n_times else 0

            if epochs_to_best > 0 and time_to_best > 0:
                ax.scatter(epochs_to_best, time_to_best,
                           label=opt, color=colors.get(opt, 'gray'),
                           s=marker_size + 20, alpha=alpha + 0.1)

        if xlabel is None:
            xlabel = 'Epochs to Best'
        if ylabel is None:
            ylabel = 'Wall Time to Best (s)'

    ax.set_xlabel(xlabel or 'X')
    ax.set_ylabel(ylabel or 'Y')
    ax.set_title(title)
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)

    return fig


def plot_ranking_bar(
    best_runs=None,
    metric_key=None,
    optimizers=None,
    colors=None,
    ax=None,
    figsize=(8, 6),
    ascending=False,
    title="Optimizer Rankings",
    xlabel=None,
    rankings=None,
    value_format=".3f",
):
    """
    Plot a horizontal bar chart ranking optimizers by a metric.

    Parameters
    ----------
    best_runs : dict, optional
        Result from load_best_runs(). Either provide this with metric_key,
        or provide rankings directly.
    metric_key : str, optional
        Summary key to rank by (e.g., "final_max_val_acc")
    optimizers : list, optional
        Subset of optimizers to include
    colors : dict, optional
        Mapping from optimizer name to color
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    figsize : tuple
        Figure size (used only if ax is None)
    ascending : bool
        If True, lower values are better (sorted ascending)
    title : str
        Plot title
    xlabel : str, optional
        X-axis label (defaults to metric_key)
    rankings : list of tuples, optional
        Pre-computed rankings as [(optimizer, value), ...]. If provided,
        best_runs and metric_key are ignored.
    value_format : str
        Format string for value labels (default: ".3f")

    Returns
    -------
    Figure or None
        Matplotlib figure (None if ax was provided)
    """
    # Build rankings from best_runs if not provided directly
    if rankings is None:
        if best_runs is None:
            return None

        if optimizers is None:
            optimizers = list(best_runs.keys())

        rankings = []
        for opt in optimizers:
            if opt not in best_runs:
                continue
            summary = best_runs[opt].get("summary", {})
            val = summary.get(metric_key)
            if val is not None:
                rankings.append((opt, val))

    if not rankings:
        return None

    rankings = sorted(rankings, key=lambda x: x[1], reverse=not ascending)
    opts = [r[0] for r in rankings]
    vals = [r[1] for r in rankings]

    if colors is None:
        colors = get_optimizer_colors(opts)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(range(len(opts)), vals, color=[colors.get(o, 'gray') for o in opts])
    ax.set_yticks(range(len(opts)))
    ax.set_yticklabels(opts, fontsize=8)
    ax.set_xlabel(xlabel or metric_key or "Value")
    ax.set_title(title)

    # Adjust x-limits for better visibility
    val_range = max(vals) - min(vals) if len(vals) > 1 else max(vals) * 0.1
    margin = val_range * 0.1 if val_range > 0 else 0.01
    ax.set_xlim(min(vals) - margin, max(vals) + margin * 2)
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    fmt = f"{{:{value_format}}}"
    for i, (opt, val) in enumerate(zip(opts, vals)):
        ax.text(val + margin * 0.5, i, fmt.format(val), va='center', fontsize=8)

    return fig


def plot_performance_summary(
    best_runs,
    optimizer_data,
    targets,
    metric_key,
    direction,
    epoch_key,
    ranking_metric_key,
    optimizers=None,
    colors=None,
    figsize=(20, 6),
):
    """
    Create a 3-panel performance summary: heatmap, efficiency scatter, and rankings.

    Parameters
    ----------
    best_runs : dict
        Result from load_best_runs()
    optimizer_data : dict
        Result from extract_history() with wall_times added
    targets : list
        Target thresholds for heatmap
    metric_key : str
        Metric for convergence checking (e.g., "test_acc")
    direction : str
        "above" or "below" for convergence checking
    epoch_key : str
        Summary key for epoch of best metric
    ranking_metric_key : str
        Summary key for ranking bar chart
    optimizers : list, optional
        Subset of optimizers
    colors : dict, optional
        Color mapping
    figsize : tuple
        Figure size

    Returns
    -------
    Figure
        Matplotlib figure with 3 subplots
    """
    if optimizers is None:
        optimizers = [o for o in best_runs if o in optimizer_data]

    if colors is None:
        colors = get_optimizer_colors(optimizers)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    plot_convergence_heatmap(
        optimizer_data=optimizer_data,
        targets=targets,
        metric_key=metric_key,
        direction=direction,
        optimizers=optimizers,
        ax=axes[0],
        annotation_fmt="check",
    )

    plot_efficiency_scatter(
        best_runs=best_runs,
        optimizer_data=optimizer_data,
        epoch_key=epoch_key,
        optimizers=optimizers,
        colors=colors,
        ax=axes[1],
    )

    plot_ranking_bar(
        best_runs=best_runs,
        metric_key=ranking_metric_key,
        optimizers=optimizers,
        colors=colors,
        ax=axes[2],
        title="Optimizer Rankings by Final Accuracy",
        xlabel="Best Test Accuracy",
    )

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Results export utilities
# ---------------------------------------------------------------------------

def print_best_hyperparameters(best_runs, optimizers=None):
    """Print hyperparameters for best runs."""
    if optimizers is None:
        optimizers = list(best_runs.keys())

    print("Hyperparameters for Best Runs")
    print("=" * 60)

    for opt in optimizers:
        run_info = best_runs.get(opt)
        if run_info is None:
            continue

        print(f"\n{opt.upper()}:")
        print(f"  Run name: {run_info.get('name', 'unknown')}")
        print("  Config:")

        config = run_info.get("config", {})
        for key, value in config.items():
            if not key.startswith("_"):
                print(f"    {key}: {value}")

        summary = run_info.get("summary", {})
        for key in ["sweep_metric", "final_max_val_acc", "final_min_val_loss",
                     "final_min_val_perplexity", "max_val_acc", "min_val_loss",
                     "min_val_perplexity"]:
            if key in summary:
                print(f"  {key}: {summary[key]:.6f}")

        print("-" * 40)


def save_best_hyperparameters(best_runs, filename, optimizers=None):
    """Save best hyperparameters to CSV.

    Returns
    -------
    DataFrame
    """
    if optimizers is None:
        optimizers = list(best_runs.keys())

    rows = []
    for opt in optimizers:
        run_info = best_runs.get(opt)
        if run_info is None:
            continue

        row = {"optimizer": opt, "run_name": run_info.get("name", "unknown")}
        row.update(run_info.get("config", {}))
        row.update(run_info.get("summary", {}))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Saved hyperparameters to {filename}")

    return df


# ---------------------------------------------------------------------------
# Re-export optimizer utilities
# ---------------------------------------------------------------------------

__all__ = [
    "load_best_runs",
    "load_top_n_runs",
    "extract_history",
    "compute_wall_times",
    "add_wall_times",
    "plot_2d_histograms",
    "plot_training_curves",
    "plot_speedrun_results",
    "plot_time_to_best",
    "plot_convergence_heatmap",
    "plot_efficiency_scatter",
    "plot_ranking_bar",
    "plot_performance_summary",
    "compute_speedrun_results",
    "print_speedrun_table",
    "print_best_hyperparameters",
    "save_best_hyperparameters",
    "get_optimizer_colors",
    "ALL_OPTIMIZERS",
]
