"""
Shared utilities for analysis notebooks.

Supports both WandB and local CSV/JSON backends for loading sweep results.

Usage::

    from analysis_utils import (
        load_best_runs,
        load_top_n_runs,
        load_all_runs,
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
        plot_hp_sensitivity,
        print_speedrun_table,
        print_best_hyperparameters,
        save_best_hyperparameters,
        print_ranking_table,
        print_variant_comparison,
        print_variant_comparison_allruns,
        print_hp_sensitivity_table,
        print_offdiag_ranking,
        print_cross_task_summary,
        get_optimizer_colors,
        ALL_OPTIMIZERS,
    )
"""

import csv
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

from optimizer_registry import get_optimizer_color, get_optimizer_colors, ALL_OPTIMIZERS


# ---------------------------------------------------------------------------
# HP sensitivity — module-level constants
# ---------------------------------------------------------------------------

# HPs that should use log scale on axes (mirrors _PARAM_BOUNDS log_scale field)
_HP_LOG_SCALE = frozenset({
    "learning_rate", "weight_decay", "eps", "xi",
    "metric_lr", "metric_reg", "gamma",
})

# Params that are infrastructure / task config, not true optimiser HPs
_NON_HP_PARAMS = frozenset({"batch_size", "n_epochs", "seed"})


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

def _load_csv_run(filepath):
    """Load a single run from a CSV file with JSON metadata comment header.

    The first line is ``# {JSON}`` containing ``config`` and ``summary``.
    The remaining lines are standard CSV with the training history.
    """
    with open(filepath) as f:
        first_line = f.readline().strip()
        if first_line.startswith("# "):
            metadata = json.loads(first_line[2:])
        else:
            metadata = {"config": {}, "summary": {}}
            f.seek(0)  # no comment header — re-read from start

        reader = csv.DictReader(f)
        history = {}
        for row in reader:
            for key, val in row.items():
                if key not in history:
                    history[key] = []
                if val == "":
                    history[key].append(None)
                else:
                    try:
                        history[key].append(float(val))
                    except ValueError:
                        history[key].append(val)

    return {
        "config": metadata.get("config", {}),
        "history": history,
        "summary": metadata.get("summary", {}),
    }


def _load_json_run(filepath):
    """Load a single run from a legacy JSON file and normalise history."""
    with open(filepath) as f:
        data = json.load(f)

    # Convert list-of-dicts history to dict-of-lists
    history_list = data.get("history", [])
    if isinstance(history_list, list):
        history = {}
        for i, row in enumerate(history_list):
            all_keys = set(history.keys()) | set(row.keys())
            for key in all_keys:
                if key not in history:
                    history[key] = [None] * i
                history[key].append(row.get(key, None))
        data["history"] = history

    return data


_SUMMARY_KEYS = frozenset({
    "function", "final_value", "iterations", "converged", "pruned",
    "runtime_seconds", "sweep_metric",
    "total_training_time_sec", "total_training_time_min", "seed", "optimizer",
    # training-task variants
    "train_loss", "val_loss", "test_loss", "train_acc", "val_acc", "test_acc",
    "best_val_acc", "best_val_loss", "best_test_acc",
    "n_epochs", "epoch",
    # best-epoch / best-metric keys used by 2D histograms
    "final_max_acc_epoch", "final_max_val_acc",
    "final_min_perp_epoch", "final_min_val_perplexity",
})


def _load_local_runs_parquet(itr_dir: Path, skip_history: bool = False) -> list[dict]:
    """Load runs from consolidated Parquet files (fast path).

    Reconstructs the same run-dict format as _load_csv_run so all
    downstream code works without modification.

    When *skip_history* is True, trajectory data is not loaded — only
    summary and config.  This dramatically reduces memory for analyses
    that only need per-run metrics (e.g. HP sensitivity).
    """
    summary_path = itr_dir / "summary.parquet"
    traj_path = itr_dir / "trajectories.parquet"

    summary_df = pd.read_parquet(summary_path)

    # Load trajectories grouped by run_id (only if file exists)
    traj_by_run: dict = {}
    if not skip_history and traj_path.exists():
        traj_df = pd.read_parquet(traj_path)
        for run_id, group in traj_df.groupby("run_id"):
            traj_by_run[run_id] = group.drop(columns="run_id")

    runs = []
    id_col = {"run_id"}
    for _, row in summary_df.iterrows():
        run_id = row.get("run_id", -1)

        config = {
            k: (None if (isinstance(v, float) and np.isnan(v)) else v)
            for k, v in row.items()
            if k not in _SUMMARY_KEYS and k not in id_col
        }
        summary = {
            k: (None if (isinstance(v, float) and np.isnan(v)) else v)
            for k, v in row.items()
            if k in _SUMMARY_KEYS
        }

        if run_id in traj_by_run:
            group = traj_by_run[run_id]
            history = {col: group[col].tolist() for col in group.columns}
        else:
            history = {}

        runs.append({
            "config": config,
            "history": history,
            "summary": summary,
            "_file": str(itr_dir / f"run_{run_id}.csv"),
        })

    return runs


def _load_local_runs(results_dir, task_tag, optimizer, iteration=0,
                     skip_history=False):
    """Load all runs for an optimizer from local Parquet or CSV/JSON files.

    Prefers consolidated Parquet files (written by consolidate_results.py)
    and falls back to scanning individual run_*.csv / run_*.json files.

    Args:
        results_dir: Base results directory
        task_tag: Task identifier (e.g., "mnist_mlp")
        optimizer: Optimizer name
        iteration: Iteration/batch number (default: 0)
        skip_history: If True, skip loading trajectory data (saves memory)

    Returns:
        list: List of run data dicts, or empty list if no files found
    """
    itr_dir = Path(results_dir) / task_tag / optimizer / f"itr_{iteration}"

    if not itr_dir.exists():
        return []

    # Fast path: consolidated Parquet files
    if (itr_dir / "summary.parquet").exists():
        return _load_local_runs_parquet(itr_dir, skip_history=skip_history)

    # Fallback: scan individual CSV/JSON files
    runs = []
    result_files = sorted(itr_dir.glob("run_*.csv")) + sorted(itr_dir.glob("run_*.json"))
    for result_file in result_files:
        if result_file.suffix == ".csv":
            data = _load_csv_run(result_file)
        else:
            data = _load_json_run(result_file)
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
            _default = float("inf") if direction == "minimize" else float("-inf")
            key_fn = lambda r, d=_default: (v if (v := r["summary"].get(metric_key)) is not None else d)
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
            _default = float("inf") if direction == "minimize" else float("-inf")
            key_fn = lambda r, d=_default: (v if (v := r["summary"].get(metric_key)) is not None else d)
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

        history = run_info.get("history", {})

        # history is dict-of-lists (normalised on load for both CSV and JSON)
        epochs = history.get("epoch", history.get("iteration", []))

        data = {"epoch": np.array(
            [v if v is not None else np.nan for v in epochs], dtype=float
        )}

        for key in metric_keys:
            col = history.get(key, [])
            if len(col) == len(epochs):
                data[key] = np.array(
                    [v if v is not None else np.nan for v in col], dtype=float
                )
            else:
                data[key] = np.full(len(epochs), np.nan)

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


def load_all_runs(
    backend,
    optimizers,
    task_tag=None,
    results_dir="results",
    iteration=0,
):
    """Load every run for each optimizer (no top-N filtering).

    Only supports the local backend.  Use this instead of
    ``load_top_n_runs`` when you need the full distribution of runs,
    e.g. for HP sensitivity analysis.

    Parameters
    ----------
    backend : str
        Must be ``"local"``.
    optimizers : list[str]
        Optimizer names to load.
    task_tag : str
        Task identifier (e.g. ``"mnist_mlp"``).
    results_dir : str
        Base directory for local results.
    iteration : int
        Iteration/batch number (default 0).

    Returns
    -------
    dict[str, list[dict]]
        Maps optimizer name → list of all run dicts.
    """
    if backend != "local":
        raise NotImplementedError("load_all_runs only supports the local backend")

    all_runs = {}
    for optimizer in optimizers:
        print(f"Loading all runs for {optimizer}...")
        runs = _load_local_runs(results_dir, task_tag, optimizer,
                                iteration=iteration, skip_history=True)
        all_runs[optimizer] = runs
        print(f"  {optimizer}: {len(runs)} runs")

    return all_runs


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
    ncols = min(6, n_plots)
    nrows = (n_plots + ncols - 1) // ncols

    if figsize == (20, 15):  # default — scale dynamically
        figsize = (ncols * 4, nrows * 3.5)

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
            if x_val is not None and y_val is not None and np.isfinite(x_val) and np.isfinite(y_val):
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
            if x_val is not None and y_val is not None and np.isfinite(x_val) and np.isfinite(y_val):
                x_vals.append(x_val)
                y_vals.append(y_val)

        if x_vals:
            h = axes[i].hist2d(x_vals, y_vals, bins=[x_bins, y_bins], cmap="viridis", alpha=0.8)
            plt.colorbar(h[3], ax=axes[i])

        axes[i].set_title(f"{title_prefix}{opt.upper()} (n={len(x_vals)})", fontsize=8)
        axes[i].set_xlabel(x_metric, fontsize=7)
        axes[i].set_ylabel(y_metric, fontsize=7)
        axes[i].tick_params(labelsize=6)
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
    ylim=None,
    highlight=None,
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

    n_opt = len(optimizers)
    nrows = len(y_keys)
    ncols = len(x_keys)

    # Scale figure wider when many optimizers need an external legend
    if figsize == (15, 10) and n_opt > 10:
        figsize = (20, 5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    # Thinner lines and no markers when many optimizers
    lw = 1.2 if n_opt > 10 else 2
    ms = 0 if n_opt > 10 else 4
    fmt = "-" if n_opt > 10 else marker

    for row, y_key in enumerate(y_keys):
        for col, x_key in enumerate(x_keys):
            ax = axes[row, col]

            for opt in optimizers:
                data = optimizer_data.get(opt, {})
                x = data.get(x_key, data.get("epoch", []))
                y = data.get(y_key, [])

                if len(x) > 0 and len(y) > 0:
                    min_len = min(len(x), len(y))
                    plot_fn = ax.loglog if log_scale else (ax.semilogy if n_opt > 10 else ax.plot)
                    if highlight is not None:
                        is_hl = opt in highlight
                        _lw = lw * 2.0 if is_hl else lw
                        _zo = 5 if is_hl else 1
                    else:
                        _lw, _zo = lw, 1
                    plot_fn(x[:min_len], y[:min_len], fmt,
                            linewidth=_lw, markersize=ms, color=colors.get(opt),
                            label=opt, alpha=0.8, zorder=_zo)

            if ylim is not None:
                if isinstance(ylim, dict):
                    if y_key in ylim:
                        ax.set_ylim(ylim[y_key])
                else:
                    ax.set_ylim(ylim)

            ax.set_xlabel(x_key)
            ax.set_ylabel(y_key)
            ax.set_title(f"{y_key} vs {x_key}")
            ax.grid(True, alpha=0.3)
            if n_opt > 10:
                ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
                          fontsize=7, ncol=1)
            else:
                ax.legend()

    plt.tight_layout()
    return fig


def plot_convergence_curves(
    optimizer_data,
    y_keys,
    x_keys=None,
    best_runs=None,
    mark_best=True,
    best_epoch_key=None,
    best_metric_key=None,
    converged_key=None,
    colors=None,
    log_y="auto",
    title=None,
    y_labels=None,
    x_labels=None,
    x_scales=None,
    figsize=None,
    highlight=None,
    linewidth=2,
    alpha=0.8,
):
    """
    Plot convergence / training curves with optional semilogy scale and
    best-point markers.

    Creates a grid of subplots with one row per *y_key* and one column per
    *x_key*.  Automatically uses semilogy for loss-like metrics.

    Parameters
    ----------
    optimizer_data : dict
        Result from ``extract_history()``, with optional ``wall_times``
        added via ``add_wall_times()``.
    y_keys : list of str
        Metric keys for y-axes (e.g., ``["train_loss", "test_acc"]``).
    x_keys : list of str, optional
        Metric keys for x-axes (default: ``["epoch", "wall_times"]``).
    best_runs : dict, optional
        Result from ``load_best_runs()``.  When provided and *mark_best*
        is True, the best / convergence point is marked with a star.
    mark_best : bool
        Whether to draw star markers at the best point (requires
        *best_runs* and *best_epoch_key*).
    best_epoch_key : str, optional
        Summary key for the epoch / iteration of the best metric (e.g.,
        ``"iterations"`` for small-examples, ``"final_max_acc_epoch"``
        for training tasks).
    best_metric_key : str, optional
        When set, star markers are only drawn on rows where *y_key*
        matches this value.  For example, set to ``"test_acc"`` so the
        marker only appears on the accuracy subplot and not on the loss
        subplot.  When *None* (default), markers appear on all rows.
    converged_key : str, optional
        Summary key for a boolean convergence flag.  When set, markers are
        only drawn for runs where the flag is True, and the legend shows a
        checkmark / cross indicator.
    colors : dict, optional
        Mapping from optimizer name to matplotlib colour.
    log_y : str, bool, or dict
        ``"auto"`` (default) enables semilogy for y_keys whose names
        contain ``loss``, ``value``, ``mse``, ``perplexity``, or
        ``error``.  A bool applies uniformly.  A dict maps individual
        y_keys to bools.
    title : str, optional
        Figure suptitle.
    y_labels : dict, optional
        Custom y-axis labels mapping y_key to display string.
    x_labels : dict, optional
        Custom x-axis labels mapping x_key to display string.
    x_scales : dict, optional
        Scaling factors for x-axes (maps x_key to float multiplier).
        E.g., ``{"wall_times": 1000}`` converts seconds to milliseconds.
    figsize : tuple, optional
        Figure size (auto-calculated if *None*).
    linewidth : float
        Curve line width.
    alpha : float
        Curve transparency.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    if x_keys is None:
        x_keys = ["epoch", "wall_times"]
    if colors is None:
        colors = get_optimizer_colors(list(optimizer_data.keys()))
    if y_labels is None:
        y_labels = {}
    if x_labels is None:
        x_labels = {}
    if x_scales is None:
        x_scales = {}

    def _should_log(y_key):
        if isinstance(log_y, dict):
            return log_y.get(y_key, False)
        if isinstance(log_y, bool):
            return log_y
        # auto mode
        return any(s in y_key.lower()
                   for s in ("loss", "value", "mse", "perplexity", "error"))

    nrows = len(y_keys)
    ncols = len(x_keys)
    if figsize is None:
        figsize = (9 * ncols, 6 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    if title:
        fig.suptitle(title, fontsize=14)

    for row, y_key in enumerate(y_keys):
        use_log = _should_log(y_key)
        for col, x_key in enumerate(x_keys):
            ax = axes[row, col]
            plot_fn = ax.semilogy if use_log else ax.plot

            for opt, data in optimizer_data.items():
                y = np.asarray(data.get(y_key, []), dtype=float)
                x = np.asarray(
                    data.get(x_key, data.get("epoch", np.arange(len(y)))),
                    dtype=float,
                )

                if len(y) == 0:
                    continue

                min_len = min(len(x), len(y))
                x_plot = x[:min_len]
                y_plot = y[:min_len]

                scale = x_scales.get(x_key, 1.0)
                if scale != 1.0:
                    x_plot = x_plot * scale

                # Build legend label
                label = opt
                if best_runs and opt in best_runs:
                    summary = best_runs[opt].get("summary", {})
                    if converged_key is not None:
                        conv = summary.get(converged_key, False)
                        n_iter = int(summary.get(best_epoch_key, len(y)))
                        sym = "\u2713" if conv else "\u2717"
                        label = f"{opt} ({sym}, {n_iter} iter)"
                    elif best_epoch_key:
                        best_ep = summary.get(best_epoch_key)
                        if best_ep is not None:
                            label = f"{opt} (best @ {int(best_ep)})"

                if highlight is not None:
                    is_hl = opt in highlight
                    lw = linewidth * 2.0 if is_hl else linewidth
                    zo = 5 if is_hl else 1
                else:
                    lw, zo = linewidth, 1

                plot_fn(x_plot, y_plot, label=label,
                        color=colors.get(opt), linewidth=lw, alpha=alpha, zorder=zo)

                # Mark best / convergence point
                if (mark_best and best_runs and opt in best_runs
                        and best_epoch_key
                        and (best_metric_key is None
                             or y_key == best_metric_key)):
                    summary = best_runs[opt].get("summary", {})
                    should_mark = True
                    if converged_key is not None:
                        should_mark = summary.get(converged_key, False)

                    if should_mark:
                        best_ep = summary.get(best_epoch_key)
                        if best_ep is not None:
                            epochs = np.asarray(
                                data.get("epoch", np.arange(len(y))),
                                dtype=float,
                            )[:min_len]
                            idx = int(np.argmin(
                                np.abs(epochs - float(best_ep))
                            ))
                            if 0 <= idx < min_len:
                                ax.scatter(
                                    x_plot[idx], y_plot[idx],
                                    color=colors.get(opt), s=100, marker='*',
                                    edgecolor='black', zorder=10,
                                )

            x_lbl = x_labels.get(x_key, x_key)
            y_lbl = y_labels.get(y_key, y_key)
            ax.set_xlabel(x_lbl)
            ax.set_ylabel(y_lbl)
            ax.set_title(f"{y_lbl} vs {x_lbl}")
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
            ax.grid(True, alpha=0.3)

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

    n_opt = len(best_times)
    use_horizontal = n_opt > 10

    if use_horizontal:
        # Sort by wall time ascending (best at top)
        sorted_opts = sorted(best_times.keys(),
                             key=lambda o: best_times[o]["wall_time"],
                             reverse=True)
        if figsize == (15, 6):
            figsize = (16, max(6, n_opt * 0.4))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        wall_times = [best_times[o]["wall_time"] for o in sorted_opts]
        epochs = [best_times[o]["epoch"] for o in sorted_opts]
        clrs = [colors.get(o, "#333333") for o in sorted_opts]
        y_pos = range(len(sorted_opts))

        ax1.barh(y_pos, wall_times, color=clrs, alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([o.upper() for o in sorted_opts], fontsize=7)
        ax1.set_xlabel("Wall Time (seconds)")
        ax1.set_title("Time to Reach Best Metric")
        ax1.grid(True, alpha=0.3, axis="x")
        for i, v in enumerate(wall_times):
            ax1.text(v + max(wall_times) * 0.01, i, f"{v:.1f}s", va="center", fontsize=6)

        # Sort by epoch ascending for second panel
        sorted_opts_ep = sorted(best_times.keys(),
                                key=lambda o: best_times[o]["epoch"],
                                reverse=True)
        epochs_sorted = [best_times[o]["epoch"] for o in sorted_opts_ep]
        clrs_ep = [colors.get(o, "#333333") for o in sorted_opts_ep]

        ax2.barh(y_pos, epochs_sorted, color=clrs_ep, alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([o.upper() for o in sorted_opts_ep], fontsize=7)
        ax2.set_xlabel("Epoch")
        ax2.set_title("Epoch of Best Metric")
        ax2.grid(True, alpha=0.3, axis="x")
        for i, v in enumerate(epochs_sorted):
            ax2.text(v + max(epochs_sorted) * 0.01, i, f"{int(v)}", va="center", fontsize=6)
    else:
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

    n_opt = len(optimizers)

    if n_opt > 10:
        # Line plot: each optimizer is a line, x-axis is targets
        if figsize == (15, 6):
            figsize = (18, 8)
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        for opt in optimizers:
            times = [results[opt].get(t) for t in targets]
            # Only plot non-None segments
            xs, ys = [], []
            for j, t in enumerate(times):
                if t is not None:
                    xs.append(j)
                    ys.append(t)
            if xs:
                ax.plot(xs, ys, "o-", color=colors.get(opt), label=opt,
                        linewidth=1.2, markersize=4, alpha=0.8)

        ax.set_xlabel("Target")
        ax.set_ylabel("Time to reach target (s)")
        ax.set_title(f"Speedrun: Time to reach {metric_key} targets")
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels([f"{t}" for t in targets])
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, ncol=1)
        ax.grid(True, alpha=0.3)
    else:
        # Grouped bar plot for small number of optimizers
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        x = np.arange(len(targets))
        width = 0.8 / max(n_opt, 1)

        for i, opt in enumerate(optimizers):
            times = [results[opt].get(t) for t in targets]
            times_plot = [t if t is not None else 0 for t in times]
            offset = (i - n_opt / 2 + 0.5) * width
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

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_efficiency_scatter(
        best_runs=best_runs,
        optimizer_data=optimizer_data,
        epoch_key=epoch_key,
        optimizers=optimizers,
        colors=colors,
        ax=axes[0],
    )

    plot_ranking_bar(
        best_runs=best_runs,
        metric_key=ranking_metric_key,
        optimizers=optimizers,
        colors=colors,
        ax=axes[1],
        title="Optimizer Rankings by Final Accuracy",
        xlabel="Best Test Accuracy",
    )

    plt.tight_layout()
    return fig


def plot_grouped_bars(
    df,
    x_col,
    y_col,
    group_col,
    ax=None,
    log_y=False,
    title=None,
    ylabel=None,
    colormap="tab20",
    legend_kwargs=None,
    epsilon=1e-6,
):
    """
    Plot a grouped bar chart from a DataFrame.

    Parameters
    ----------
    df : DataFrame
        Data to plot.
    x_col : str
        Column for x-axis categories.
    y_col : str
        Column for bar heights.
    group_col : str
        Column for bar groups (e.g., optimizer).
    ax : Axes, optional
        Matplotlib axes.  Creates a new figure if *None*.
    log_y : bool
        Use log scale on y-axis.
    title : str, optional
        Plot title.
    ylabel : str, optional
        Y-axis label (defaults to *y_col*).
    colormap : str or Colormap
        Matplotlib colormap.
    legend_kwargs : dict, optional
        Keyword arguments for ``ax.legend()``.
    epsilon : float
        Small value to clip data before log scale.

    Returns
    -------
    Figure or None
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    if len(df) == 0:
        ax.set_title(title or "")
        return fig

    pivot = df.pivot(index=x_col, columns=group_col, values=y_col)
    if log_y:
        pivot = pivot.clip(lower=epsilon)
    pivot.plot(kind='bar', ax=ax, rot=45, colormap=colormap)
    if log_y:
        ax.set_yscale('log')

    ax.set_title(title or f"{y_col} by {x_col}")
    ax.set_ylabel(ylabel or y_col)
    if legend_kwargs is None:
        legend_kwargs = dict(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.legend(**legend_kwargs)

    return fig


def build_summary_dataframe(all_best_runs, optimizers=None):
    """
    Build a summary DataFrame from multi-function best runs.

    Designed for convergence-based benchmarks (e.g., small-examples) where
    the summary contains ``converged``, ``iterations``, ``final_value``, and
    ``runtime_seconds``.

    Parameters
    ----------
    all_best_runs : dict
        Mapping from function name to best_runs dict (result of
        ``load_best_runs()``).
    optimizers : list, optional
        Ordered list of optimizers for categorical sorting.

    Returns
    -------
    DataFrame
        Columns: function, optimizer, converged, iterations, final_value,
        runtime_seconds, runtime_ms.
    """
    rows = []
    for func_name, best_runs in all_best_runs.items():
        for opt, run_info in best_runs.items():
            s = run_info.get("summary", {})
            rows.append({
                'function': func_name,
                'optimizer': opt,
                'converged': s.get('converged', False),
                'iterations': s.get('iterations', 0),
                'final_value': s.get('final_value', float('inf')),
                'runtime_seconds': s.get('runtime_seconds', 0),
                'runtime_ms': s.get('runtime_seconds', 0) * 1000,
            })

    if not rows:
        return pd.DataFrame(columns=['function', 'optimizer', 'converged',
                                      'iterations', 'final_value',
                                      'runtime_seconds', 'runtime_ms'])

    df = pd.DataFrame(rows)
    if optimizers:
        df['optimizer'] = pd.Categorical(
            df['optimizer'], categories=optimizers, ordered=True
        )
    df = df.sort_values(['function', 'optimizer']).reset_index(drop=True)
    return df


def plot_multi_function_performance(
    summary_df,
    optimizers=None,
    figsize=(32, 12),
    suptitle="Optimizer Performance Comparison",
):
    """
    Create a 6-panel performance comparison across multiple test functions.

    Panels:
      1. Iterations to convergence (bar chart, log scale)
      2. Runtime to convergence (bar chart, log scale)
      3. Final function values (bar chart, log scale)
      4. Convergence rate heatmap
      5. Efficiency scatter (iterations vs runtime)
      6. Rankings by average iterations

    Parameters
    ----------
    summary_df : DataFrame
        Must have columns: function, optimizer, converged, iterations,
        final_value, runtime_seconds, runtime_ms.  See
        ``build_summary_dataframe()``.
    optimizers : list, optional
        Ordered list of optimizer names.
    figsize : tuple
        Figure size.
    suptitle : str
        Figure title.

    Returns
    -------
    Figure
        Matplotlib figure with 6 subplots.
    """
    if len(summary_df) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No results to plot.", ha='center', va='center',
                transform=ax.transAxes)
        return fig

    converged_df = summary_df[summary_df['converged'] == True]
    cmap = plt.cm.tab20

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(suptitle, fontsize=16)

    # 1. Iterations to convergence
    if len(converged_df) > 0:
        plot_grouped_bars(
            converged_df, x_col='function', y_col='iterations',
            group_col='optimizer', ax=axes[0, 0], log_y=True,
            title='Iterations to Convergence', ylabel='Iterations',
            colormap=cmap,
        )

    # 2. Runtime to convergence
    if len(converged_df) > 0:
        plot_grouped_bars(
            converged_df, x_col='function', y_col='runtime_ms',
            group_col='optimizer', ax=axes[0, 1], log_y=True,
            title='Runtime to Convergence', ylabel='Runtime (ms)',
            colormap=cmap,
        )

    # 3. Final values
    plot_grouped_bars(
        summary_df, x_col='function', y_col='final_value',
        group_col='optimizer', ax=axes[0, 2], log_y=True,
        title='Final Function Values (Log Scale)', ylabel='Function Value',
        colormap=cmap,
    )

    # 4. Convergence heatmap
    conv_matrix = summary_df.groupby(
        ['optimizer', 'function']
    )['converged'].mean().unstack()
    plot_convergence_heatmap(
        matrix=conv_matrix, ax=axes[1, 0],
        title='Convergence Rate Heatmap', xlabel='Function',
        annotation_fmt="percent",
    )

    # 5. Efficiency scatter
    if len(converged_df) > 0:
        plot_efficiency_scatter(
            data=converged_df, x_col='iterations', y_col='runtime_ms',
            group_col='optimizer', ax=axes[1, 1],
            title='Efficiency: Iterations vs Runtime',
            xlabel='Iterations', ylabel='Runtime (ms)',
        )

    # 6. Rankings
    if len(converged_df) > 0:
        avg_iter = converged_df.groupby('optimizer')['iterations'].mean().sort_values()
        rankings = [(opt, val) for opt, val in avg_iter.items()]
        plot_ranking_bar(
            rankings=rankings, ax=axes[1, 2], ascending=True,
            title='Average Iterations to Convergence',
            xlabel='Average Iterations', value_format=".1f",
        )

    plt.tight_layout()
    return fig


def print_convergence_summary(best_runs, func_name=None, optimizers=None):
    """
    Print a text summary of convergence results.

    Parameters
    ----------
    best_runs : dict
        Result from ``load_best_runs()``.  Each entry's summary should
        contain ``converged``, ``iterations``, ``runtime_seconds``, and
        ``final_value``.
    func_name : str, optional
        Name of the function / task for the header.
    optimizers : list, optional
        Ordered list of optimizers to print (defaults to all in
        *best_runs*).
    """
    if func_name:
        print(f"\n{func_name.title()} Function Summary:")
    print("-" * 60)

    if optimizers is None:
        optimizers = list(best_runs.keys())

    for opt in optimizers:
        if opt not in best_runs:
            continue
        s = best_runs[opt].get("summary", {})
        status = "Converged" if s.get("converged", False) else "Failed"
        print(
            f"  {opt:25s}: {status:10s} | "
            f"{s.get('iterations', 0):4d} iter | "
            f"{s.get('runtime_seconds', 0) * 1000:6.1f} ms | "
            f"Final: {s.get('final_value', float('inf')):.2e}"
        )


# ---------------------------------------------------------------------------
# HP sensitivity analysis
# ---------------------------------------------------------------------------

def plot_hp_sensitivity(
    runs,
    optimizer_name,
    *,
    convergence_percentile=50,
    convergence_threshold=None,
    convergence_key=None,
    metric_key="sweep_metric",
    direction="minimize",
    exclude_params=None,
    figsize=None,
    title=None,
    conv_color="#1f77b4",
    div_color="#d62728",
    alpha_conv=0.4,
    alpha_div=0.25,
    point_size=12,
):
    """Grid of pairwise HP scatter plots coloured by convergence.

    Each off-diagonal panel shows a 2-D scatter of two hyperparameters:
    converged runs as filled circles (○) and diverged runs as crosses (×).
    Diagonal panels show overlapping histograms for each HP.  Axes use log
    scale for log-uniform parameters (learning rate, eps, xi, etc.).

    Parameters
    ----------
    runs : list[dict]
        Run dicts from ``load_all_runs``, each containing ``"config"``
        and ``"summary"`` sub-dicts.
    optimizer_name : str
        Used for the plot title and default colour.
    convergence_percentile : float
        Percentile split when no hard threshold is given.  Runs in the
        better ``convergence_percentile`` percent are labelled converged.
        E.g. 50 → median split; 25 → only top quarter are converged.
    convergence_threshold : float or None
        Hard metric threshold.  Overrides ``convergence_percentile``.
        With ``direction="minimize"`` a run is converged if its metric
        is *below* this value.
    convergence_key : str or None
        If set, read convergence directly from ``run["summary"][convergence_key]``
        as a boolean.  Overrides both ``convergence_threshold`` and
        ``convergence_percentile``.  Useful when the summary already contains
        a ``"converged"`` flag (e.g. small-examples tasks).
    metric_key : str
        Summary key used to determine convergence (default ``"sweep_metric"``).
    direction : {"minimize", "maximize"}
        Whether lower or higher metric values mean better performance.
    exclude_params : collection[str] or None
        HP names to drop from the plot.
    figsize : tuple or None
        Figure size; defaults to ``(3*N, 3*N)`` where N is the number of HPs.
    title : str or None
        Figure title; defaults to ``"{optimizer_name} — HP Sensitivity"``.
    conv_color : str
        Colour for converged runs (default ``"#1f77b4"`` — matplotlib blue).
        Fixed across all optimizers so plots are directly comparable.
    div_color : str
        Colour for diverged runs.
    alpha_conv, alpha_div : float
        Transparency for scatter points.
    point_size : float
        Marker size in scatter plots.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of matplotlib.axes.Axes, shape (N, N)
    """
    from matplotlib.lines import Line2D

    # ------------------------------------------------------------------
    # 1. Build DataFrame
    # ------------------------------------------------------------------
    records = []
    for run in runs:
        cfg = run.get("config")
        summ = run.get("summary")
        if not cfg or not summ:
            continue
        row = dict(cfg)
        row["_metric"] = summ.get(metric_key, np.nan)
        row["_pruned"] = bool(summ.get("pruned", False))
        if convergence_key is not None:
            row["_conv_key"] = bool(summ.get(convergence_key, False))
        records.append(row)

    if not records:
        raise ValueError(f"No valid runs found for {optimizer_name!r}")

    df = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # 2. Determine convergence
    # ------------------------------------------------------------------
    metrics = df["_metric"].to_numpy(dtype=float)
    valid = np.isfinite(metrics)

    if convergence_key is not None:
        converged_mask = df["_conv_key"].to_numpy(dtype=bool)
    elif convergence_threshold is not None:
        if direction == "minimize":
            converged_mask = metrics < convergence_threshold
        else:
            converged_mask = metrics > convergence_threshold
    else:
        if direction == "minimize":
            q = np.nanpercentile(metrics, convergence_percentile)
            converged_mask = metrics <= q
        else:
            q = np.nanpercentile(metrics, 100 - convergence_percentile)
            converged_mask = metrics >= q

    # Pruned runs are never converged
    converged_mask = converged_mask & ~df["_pruned"].to_numpy(dtype=bool)
    df["_converged"] = converged_mask

    # ------------------------------------------------------------------
    # 3. Select HP columns (numeric, variable, not excluded)
    # ------------------------------------------------------------------
    _exclude = _NON_HP_PARAMS | set(exclude_params or [])
    hp_cols = [
        c for c in df.columns
        if not c.startswith("_")
        and c not in _exclude
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].nunique() > 1
    ]

    if not hp_cols:
        raise ValueError(
            f"No variable numeric HP columns found for {optimizer_name!r}. "
            f"Columns: {list(df.columns)}"
        )

    N = len(hp_cols)
    df_conv = df[df["_converged"]]
    df_div = df[~df["_converged"]]
    n_conv = int(converged_mask.sum())
    n_div = int((~converged_mask).sum())

    # ------------------------------------------------------------------
    # 4. Figure setup
    # ------------------------------------------------------------------
    figsize = figsize or (max(3 * N, 6), max(3 * N, 6))
    fig, axes = plt.subplots(N, N, figsize=figsize, squeeze=False)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    # ------------------------------------------------------------------
    # 5. Fill each panel
    # ------------------------------------------------------------------
    for i, yp in enumerate(hp_cols):
        for j, xp in enumerate(hp_cols):
            ax = axes[i, j]
            use_xlog = xp in _HP_LOG_SCALE
            use_ylog = yp in _HP_LOG_SCALE

            if j > i:
                # ---- Upper triangle: hide ----
                ax.set_visible(False)
                continue

            if i == j:
                # ---- Diagonal: overlapping histograms ----
                all_vals = df[xp].dropna().to_numpy(dtype=float)
                if use_xlog:
                    all_vals = all_vals[all_vals > 0]
                if len(all_vals) == 0:
                    continue

                if use_xlog:
                    lo = np.log10(all_vals.min())
                    hi = np.log10(all_vals.max())
                    bins = np.logspace(lo, hi, 30)
                    ax.set_xscale("log")
                else:
                    bins = np.linspace(all_vals.min(), all_vals.max(), 30)

                vals_div = df_div[xp].dropna().to_numpy(dtype=float)
                vals_conv = df_conv[xp].dropna().to_numpy(dtype=float)
                if use_xlog:
                    vals_div = vals_div[vals_div > 0]
                    vals_conv = vals_conv[vals_conv > 0]

                ax.hist(vals_div, bins=bins, color=div_color, alpha=0.5, linewidth=0)
                ax.hist(vals_conv, bins=bins, color=conv_color, alpha=0.6, linewidth=0)
                ax.set_yticks([])

            else:
                # ---- Lower triangle: scatter ----
                # Diverged first so converged sits on top
                ax.scatter(
                    df_div[xp].to_numpy(dtype=float),
                    df_div[yp].to_numpy(dtype=float),
                    marker="x", c=div_color, alpha=alpha_div,
                    s=point_size, linewidths=0.7, rasterized=True,
                )
                ax.scatter(
                    df_conv[xp].to_numpy(dtype=float),
                    df_conv[yp].to_numpy(dtype=float),
                    marker="o", c=conv_color, alpha=alpha_conv,
                    s=point_size, linewidths=0, rasterized=True,
                )

                if use_xlog:
                    ax.set_xscale("log")
                if use_ylog:
                    ax.set_yscale("log")

            # ---- Axis labels: bottom row gets x-labels, left column gets y-labels ----
            if i == N - 1:
                ax.set_xlabel(xp, fontsize=9)
            else:
                ax.tick_params(labelbottom=False)

            if j == 0 and i != j:
                ax.set_ylabel(yp, fontsize=9)
            else:
                ax.tick_params(labelleft=False)

            ax.tick_params(labelsize=7)

    # ------------------------------------------------------------------
    # 6. Legend and title
    # ------------------------------------------------------------------
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=conv_color,
               markersize=8, label=f"Converged  (n={n_conv})"),
        Line2D([0], [0], marker="x", color=div_color, markersize=8,
               markeredgewidth=1.2, label=f"Diverged  (n={n_div})"),
    ]
    fig.legend(
        handles=legend_handles, loc="upper right",
        bbox_to_anchor=(1.0, 1.0), framealpha=0.9, fontsize=9,
    )

    title = title or f"{optimizer_name} — HP Sensitivity"
    fig.suptitle(title, fontsize=13, y=1.01)

    return fig, axes


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


# ---------------------------------------------------------------------------
# Text-based analysis functions (paper-ready tables)
# ---------------------------------------------------------------------------

# Variant comparison pairs: (base, variant, label)
_VARIANT_PAIRS = [
    ("sgd_learn_diag", "sgd_learn_diag_log", "log embedding"),
    ("sgd_learn_diag", "sgd_learn_diag_curv", "curvature correction"),
    ("sgd_learn_diag_log", "sgd_learn_diag_curv_log", "curvature correction (log)"),
    ("sgd_learn_diag_curv", "sgd_learn_diag_curv_log", "log embedding (curv)"),
    ("sgd_learn_scalar", "sgd_learn_diag", "diagonal vs scalar"),
    ("sgd_learn_scalar_log", "sgd_learn_diag_log", "diagonal vs scalar (log)"),
]


def print_ranking_table(best_runs, metric_key="sweep_metric",
                        direction="maximize", optimizers=None, title=None):
    """Print a sorted ranking table of optimizer performance.

    Parameters
    ----------
    best_runs : dict
        Result from ``load_best_runs()``.  Maps optimizer name to run dict.
    metric_key : str
        Key in run['summary'] to rank by.
    direction : str
        "maximize" or "minimize".
    optimizers : list, optional
        Subset of optimizers to include.
    title : str, optional
        Header for the table.
    """
    if optimizers is None:
        optimizers = list(best_runs.keys())

    rows = []
    for opt in optimizers:
        run = best_runs.get(opt)
        if run is None:
            continue
        val = run.get("summary", {}).get(metric_key)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        rows.append((opt, float(val)))

    reverse = direction == "maximize"
    rows.sort(key=lambda r: r[1], reverse=reverse)

    if title:
        print(f"\n{title}")
    print(f"{'Rank':<6}{'Optimizer':<30}{metric_key:<20}")
    print("-" * 56)
    for i, (opt, val) in enumerate(rows, 1):
        print(f"{i:<6}{opt:<30}{val:<20.6f}")
    print()


def print_variant_comparison(best_runs, metric_key="sweep_metric",
                             direction="maximize", pairs=None, title=None):
    """Print a comparison table between variant pairs.

    Compares base vs variant (e.g., plain vs log, base vs curv) using the
    best run for each optimizer.

    Parameters
    ----------
    best_runs : dict
        Result from ``load_best_runs()``.
    metric_key : str
        Key in run['summary'] to compare.
    direction : str
        "maximize" or "minimize".
    pairs : list of (str, str, str), optional
        List of (base_name, variant_name, label) tuples.
        Defaults to ``_VARIANT_PAIRS``.
    title : str, optional
        Header for the table.
    """
    if pairs is None:
        pairs = _VARIANT_PAIRS

    if title:
        print(f"\n{title}")
    print(f"{'Comparison':<35}{'Base':<12}{'Variant':<12}{'Delta':<12}{'Winner':<12}")
    print("-" * 83)

    for base_name, var_name, label in pairs:
        base_run = best_runs.get(base_name)
        var_run = best_runs.get(var_name)
        if base_run is None or var_run is None:
            continue

        base_val = base_run.get("summary", {}).get(metric_key)
        var_val = var_run.get("summary", {}).get(metric_key)
        if base_val is None or var_val is None:
            continue

        delta = float(var_val) - float(base_val)
        if direction == "maximize":
            winner = var_name.split("_")[-1] if delta > 0 else base_name.split("_")[-1]
        else:
            winner = var_name.split("_")[-1] if delta < 0 else base_name.split("_")[-1]

        print(f"{label:<35}{float(base_val):<12.6f}{float(var_val):<12.6f}"
              f"{delta:<+12.6f}{winner:<12}")
    print()


def print_variant_comparison_allruns(all_runs, metric_key="sweep_metric",
                                     direction="maximize", pairs=None,
                                     title=None):
    """Compare variant pairs using all runs (not just best), with statistics.

    Uses Mann-Whitney U test for significance and reports median, IQR, and
    p-value.

    Parameters
    ----------
    all_runs : dict
        Result from ``load_all_runs()``.  Maps optimizer name to list of
        run dicts.
    metric_key : str
        Key in run['summary'] to compare.
    direction : str
        "maximize" or "minimize".
    pairs : list of (str, str, str), optional
        Defaults to ``_VARIANT_PAIRS``.
    title : str, optional
        Header for the table.
    """
    from scipy.stats import mannwhitneyu

    if pairs is None:
        pairs = _VARIANT_PAIRS

    if title:
        print(f"\n{title}")
    print(f"{'Comparison':<30}{'Base med':<11}{'Var med':<11}"
          f"{'n_base':<8}{'n_var':<8}{'p-value':<10}{'Sig?':<6}")
    print("-" * 84)

    for base_name, var_name, label in pairs:
        base_list = all_runs.get(base_name, [])
        var_list = all_runs.get(var_name, [])

        base_vals = [float(v) for r in base_list
                     if (v := r.get("summary", {}).get(metric_key)) is not None
                     and not np.isnan(float(v))]
        var_vals = [float(v) for r in var_list
                    if (v := r.get("summary", {}).get(metric_key)) is not None
                    and not np.isnan(float(v))]

        if len(base_vals) < 3 or len(var_vals) < 3:
            continue

        base_med = np.median(base_vals)
        var_med = np.median(var_vals)

        try:
            alt = "greater" if direction == "maximize" else "less"
            _, p = mannwhitneyu(var_vals, base_vals, alternative=alt)
        except ValueError:
            p = 1.0

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        print(f"{label:<30}{base_med:<11.6f}{var_med:<11.6f}"
              f"{len(base_vals):<8}{len(var_vals):<8}{p:<10.4f}{sig:<6}")
    print()


def print_hp_sensitivity_table(runs, metric_key="sweep_metric",
                               exclude_params=None, title=None):
    """Print HP importance ranking using Spearman correlation with performance.

    Parameters
    ----------
    runs : list[dict]
        List of run dicts (from one optimizer).
    metric_key : str
        Key in run['summary'] to correlate against.
    exclude_params : set, optional
        Parameter names to skip.
    title : str, optional
        Header for the table.
    """
    from scipy.stats import spearmanr

    if exclude_params is None:
        exclude_params = _NON_HP_PARAMS

    metrics = []
    configs = []
    for r in runs:
        val = r.get("summary", {}).get(metric_key)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        metrics.append(float(val))
        configs.append(r.get("config", {}))

    if len(metrics) < 10:
        print("  (too few runs for HP sensitivity)")
        return

    if title:
        print(f"\n{title}")
    print(f"{'HP':<25}{'Spearman rho':<15}{'p-value':<12}{'Sig?':<6}{'Direction':<12}")
    print("-" * 70)

    results = []
    all_keys = set()
    for c in configs:
        all_keys.update(c.keys())

    for hp in sorted(all_keys - exclude_params):
        vals = []
        perf = []
        for m, c in zip(metrics, configs):
            v = c.get(hp)
            if v is not None and isinstance(v, (int, float)):
                vals.append(float(v))
                perf.append(m)

        if len(vals) < 10:
            continue

        # Use log scale for log-scale HPs
        if hp in _HP_LOG_SCALE:
            vals = [np.log10(max(v, 1e-15)) for v in vals]

        rho, p = spearmanr(vals, perf)
        if np.isnan(rho):
            continue
        results.append((hp, rho, p))

    results.sort(key=lambda x: -abs(x[1]))
    for hp, rho, p in results:
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        direction = "higher=better" if rho > 0 else "lower=better"
        print(f"{hp:<25}{rho:<+15.3f}{p:<12.4f}{sig:<6}{direction:<12}")
    print()


def print_offdiag_ranking(best_runs, metric_key="sweep_metric",
                          direction="maximize", title=None):
    """Print ranking of off-diagonal (a,b) mode combinations.

    Filters to optimizers matching ``sgd_offdiag_*`` and ranks them.

    Parameters
    ----------
    best_runs : dict
        Result from ``load_best_runs()``.
    metric_key : str
        Key in run['summary'] to rank by.
    direction : str
        "maximize" or "minimize".
    title : str, optional
        Header for the table.
    """
    offdiag = {k: v for k, v in best_runs.items()
               if k.startswith("sgd_offdiag_")}
    if not offdiag:
        print("  (no off-diagonal optimizers found)")
        return

    rows = []
    for opt, run in offdiag.items():
        val = run.get("summary", {}).get(metric_key)
        if val is None:
            continue
        # Parse (a, b) from name: sgd_offdiag_{a}_{b}
        parts = opt.replace("sgd_offdiag_", "").split("_", 1)
        a_mode = parts[0] if len(parts) > 0 else "?"
        b_mode = parts[1] if len(parts) > 1 else "?"
        rows.append((opt, a_mode, b_mode, float(val)))

    reverse = direction == "maximize"
    rows.sort(key=lambda r: r[3], reverse=reverse)

    if title:
        print(f"\n{title}")
    print(f"{'Rank':<6}{'a':<8}{'b':<8}{metric_key:<20}{'Name':<30}")
    print("-" * 72)
    for i, (opt, a, b, val) in enumerate(rows, 1):
        print(f"{i:<6}{a:<8}{b:<8}{val:<20.6f}{opt:<30}")
    print()


def print_cross_task_summary(task_results, metric_keys=None,
                             direction="maximize", title=None):
    """Print a cross-task summary showing each optimizer's rank per task.

    Parameters
    ----------
    task_results : dict
        Mapping from task name to best_runs dict.
    metric_keys : dict, optional
        Mapping from task name to metric key.  Defaults to ``sweep_metric``.
    direction : str
        "maximize" or "minimize".
    title : str, optional
        Header for the table.
    """
    if metric_keys is None:
        metric_keys = {t: "sweep_metric" for t in task_results}

    # Compute ranks per task
    task_ranks = {}
    all_opts = set()
    for task, best_runs in task_results.items():
        mk = metric_keys.get(task, "sweep_metric")
        rows = []
        for opt, run in best_runs.items():
            val = run.get("summary", {}).get(mk)
            if val is not None and not np.isnan(float(val)):
                rows.append((opt, float(val)))
                all_opts.add(opt)
        reverse = direction == "maximize"
        rows.sort(key=lambda r: r[1], reverse=reverse)
        task_ranks[task] = {opt: rank for rank, (opt, _) in enumerate(rows, 1)}

    tasks = list(task_results.keys())

    if title:
        print(f"\n{title}")
    header = f"{'Optimizer':<30}" + "".join(f"{t:<15}" for t in tasks) + f"{'Avg rank':<10}"
    print(header)
    print("-" * len(header))

    opt_avg = []
    for opt in sorted(all_opts):
        ranks = [task_ranks[t].get(opt) for t in tasks]
        valid = [r for r in ranks if r is not None]
        avg = np.mean(valid) if valid else 999
        opt_avg.append((opt, ranks, avg))

    opt_avg.sort(key=lambda x: x[2])
    for opt, ranks, avg in opt_avg:
        row = f"{opt:<30}"
        for r in ranks:
            row += f"{r:<15}" if r is not None else f"{'--':<15}"
        row += f"{avg:<10.1f}"
        print(row)
    print()


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
    "load_all_runs",
    "extract_history",
    "compute_wall_times",
    "add_wall_times",
    "plot_2d_histograms",
    "plot_training_curves",
    "plot_convergence_curves",
    "plot_speedrun_results",
    "plot_time_to_best",
    "plot_convergence_heatmap",
    "plot_efficiency_scatter",
    "plot_ranking_bar",
    "plot_performance_summary",
    "plot_grouped_bars",
    "plot_multi_function_performance",
    "plot_hp_sensitivity",
    "build_summary_dataframe",
    "compute_speedrun_results",
    "print_speedrun_table",
    "print_convergence_summary",
    "print_best_hyperparameters",
    "save_best_hyperparameters",
    "print_ranking_table",
    "print_variant_comparison",
    "print_variant_comparison_allruns",
    "print_hp_sensitivity_table",
    "print_offdiag_ranking",
    "print_cross_task_summary",
    "get_optimizer_colors",
    "ALL_OPTIMIZERS",
]
