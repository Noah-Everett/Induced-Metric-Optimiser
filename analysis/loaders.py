"""
Data loading for analysis notebooks.

Supports WandB and local CSV/JSON/Parquet backends.
"""

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from utils import _SUMMARY_KEYS


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

def _load_csv_run(filepath, columns=None):
    """Load a single run from a CSV file with JSON metadata comment header.

    The first line is ``# {JSON}`` containing ``config`` and ``summary``.
    The remaining lines are standard CSV with the training history.

    When *columns* is given, only those columns (plus ``epoch``/
    ``iteration``) are kept from the CSV body.
    """
    keep = None
    if columns is not None:
        keep = set(columns) | {"epoch", "iteration"}

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
                if keep is not None and key not in keep:
                    continue
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


def _load_json_run(filepath, columns=None):
    """Load a single run from a legacy JSON file and normalise history."""
    keep = None
    if columns is not None:
        keep = set(columns) | {"epoch", "iteration"}

    with open(filepath) as f:
        data = json.load(f)

    # Convert list-of-dicts history to dict-of-lists
    history_list = data.get("history", [])
    if isinstance(history_list, list):
        history = {}
        for i, row in enumerate(history_list):
            all_keys = set(history.keys()) | set(row.keys())
            for key in all_keys:
                if keep is not None and key not in keep:
                    continue
                if key not in history:
                    history[key] = [None] * i
                history[key].append(row.get(key, None))
        data["history"] = history

    return data


def _load_local_runs_parquet(itr_dir: Path, skip_history: bool = False,
                             columns: list[str] | None = None) -> list[dict]:
    """Load runs from consolidated Parquet files (fast path).

    Reconstructs the same run-dict format as _load_csv_run so all
    downstream code works without modification.

    When *skip_history* is True, trajectory data is not loaded — only
    summary and config.  This dramatically reduces memory for analyses
    that only need per-run metrics (e.g. HP sensitivity).

    When *columns* is given, only those columns (plus ``run_id`` and
    ``epoch``/``iteration``) are read from trajectories.parquet.  This
    avoids loading large diagnostic columns that aren't needed.
    """
    summary_path = itr_dir / "summary.parquet"
    traj_path = itr_dir / "trajectories.parquet"

    summary_df = pd.read_parquet(summary_path)

    # Load trajectories grouped by run_id (only if file exists)
    traj_by_run: dict = {}
    if not skip_history and traj_path.exists():
        read_cols = None
        if columns is not None:
            # Always include run_id + epoch/iteration for grouping/alignment
            required = {"run_id", "epoch", "iteration"}
            read_cols = list(required | set(columns))
            # Filter to columns that actually exist in the file
            import pyarrow.parquet as pq
            available = set(pq.read_schema(traj_path).names)
            read_cols = [c for c in read_cols if c in available]

        traj_df = pd.read_parquet(traj_path, columns=read_cols)
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
                     skip_history=False, columns=None):
    """Load all runs for an optimizer from local Parquet or CSV/JSON files.

    Prefers consolidated Parquet files (written by consolidate_results.py)
    and falls back to scanning individual run_*.csv / run_*.json files.

    Args:
        results_dir: Base results directory
        task_tag: Task identifier (e.g., "mnist_mlp")
        optimizer: Optimizer name
        iteration: Iteration/batch number (default: 0)
        skip_history: If True, skip loading trajectory data (saves memory)
        columns: If given, only read these trajectory columns from Parquet
                 (plus run_id/epoch/iteration). None reads all columns.

    Returns:
        list: List of run data dicts, or empty list if no files found
    """
    itr_dir = Path(results_dir) / task_tag / optimizer / f"itr_{iteration}"

    if not itr_dir.exists():
        return []

    # Fast path: consolidated Parquet files
    if (itr_dir / "summary.parquet").exists():
        return _load_local_runs_parquet(itr_dir, skip_history=skip_history,
                                        columns=columns)

    # Fallback: scan individual CSV/JSON files
    runs = []
    result_files = sorted(itr_dir.glob("run_*.csv")) + sorted(itr_dir.glob("run_*.json"))
    for result_file in result_files:
        if skip_history:
            # Summary-only: read metadata but skip CSV body
            with open(result_file) as f:
                first_line = f.readline().strip()
            if first_line.startswith("# "):
                meta = json.loads(first_line[2:])
            else:
                meta = {"config": {}, "summary": {}}
            data = {"config": meta.get("config", {}),
                    "summary": meta.get("summary", {}),
                    "history": {}}
        elif result_file.suffix == ".csv":
            data = _load_csv_run(result_file, columns=columns)
        else:
            data = _load_json_run(result_file, columns=columns)
        data["_file"] = str(result_file)
        runs.append(data)

    return runs


def _sort_runs(runs, metric_key, direction):
    """Sort runs by a summary metric, returning best first."""
    _default = float("inf") if direction == "minimize" else float("-inf")
    key_fn = lambda r, d=_default: (v if (v := r["summary"].get(metric_key)) is not None else d)
    return sorted(runs, key=key_fn, reverse=(direction == "maximize"))


def _reload_run_history(run, results_dir, task_tag, optimizer, iteration,
                        columns=None):
    """Reload a single run's history data (used after summary-only loading).

    If the run already has non-empty history, returns it as-is.
    Otherwise, reloads from the original file (Parquet or CSV/JSON).

    When *columns* is given, only those columns (plus epoch/iteration)
    are read from the Parquet file.
    """
    if run.get("history"):
        return run

    itr_dir = Path(results_dir) / task_tag / optimizer / f"itr_{iteration}"
    run_file = run.get("_file", "")

    # Try Parquet: load just this run's trajectory
    traj_path = itr_dir / "trajectories.parquet"
    if traj_path.exists() and "run_" in run_file:
        try:
            run_id = int(Path(run_file).stem.replace("run_", ""))

            read_cols = None
            if columns is not None:
                required = {"run_id", "epoch", "iteration"}
                read_cols = list(required | set(columns))
                import pyarrow.parquet as pq
                available = set(pq.read_schema(traj_path).names)
                read_cols = [c for c in read_cols if c in available]

            traj_df = pd.read_parquet(traj_path, columns=read_cols)
            group = traj_df[traj_df["run_id"] == run_id].drop(columns="run_id")
            if len(group) > 0:
                run["history"] = {col: group[col].tolist() for col in group.columns}
                return run
        except (ValueError, KeyError):
            pass

    # Fallback: re-read the individual file
    if os.path.exists(run_file):
        if run_file.endswith(".csv"):
            full = _load_csv_run(run_file, columns=columns)
        else:
            full = _load_json_run(run_file, columns=columns)
        run["history"] = full.get("history", {})

    return run


def _load_best_runs_local(results_dir, task_tag, optimizers,
                          metric_key="sweep_metric", direction="minimize",
                          iteration=0, columns=None):
    """Load best runs from local files.

    Loads summaries first (skip_history=True) to find the best run,
    then reloads only the selected run's history.  This avoids loading
    hundreds of full training trajectories into memory just to discard
    all but one.

    Args:
        results_dir: Base results directory
        task_tag: Task identifier
        optimizers: List of optimizer names
        metric_key: Metric for sorting
        direction: "minimize" or "maximize"
        iteration: Iteration/batch number
        columns: If given, only load these trajectory columns from Parquet
    """
    best_runs = {}

    for optimizer in optimizers:
        print(f"Finding best run for {optimizer}...")

        # Phase 1: load summaries only (fast, low memory)
        runs = _load_local_runs(results_dir, task_tag, optimizer,
                                iteration=iteration, skip_history=True)

        if runs:
            runs_sorted = _sort_runs(runs, metric_key, direction)
            best = runs_sorted[0]

            # Phase 2: reload history for the single best run
            best = _reload_run_history(best, results_dir, task_tag, optimizer,
                                       iteration, columns=columns)

            best_runs[optimizer] = {
                "config": best["config"],
                "summary": best["summary"],
                "history": best.get("history", {}),
                "name": Path(best["_file"]).stem,
            }
            print(f"  {optimizer}: {best_runs[optimizer]['name']}")
        else:
            print(f"  {optimizer}: No runs found")

    return best_runs


def _load_top_n_runs_local(results_dir, task_tag, optimizers,
                           n=50, metric_key="sweep_metric", direction="minimize",
                           iteration=0, columns=None):
    """Load top N runs for each optimizer from local files.

    Loads summaries first (skip_history=True) to find the top N runs,
    then reloads only their histories.

    Args:
        results_dir: Base results directory
        task_tag: Task identifier
        optimizers: List of optimizer names
        n: Number of top runs to return
        metric_key: Metric for sorting
        direction: "minimize" or "maximize"
        iteration: Iteration/batch number
        columns: If given, only load these trajectory columns from Parquet
    """
    top_n_runs = {}

    for optimizer in optimizers:
        print(f"Loading top {n} runs for {optimizer}...")

        # Phase 1: load summaries only
        runs = _load_local_runs(results_dir, task_tag, optimizer,
                                iteration=iteration, skip_history=True)

        if runs:
            runs_sorted = _sort_runs(runs, metric_key, direction)
            selected = runs_sorted[:n]

            # Phase 2: reload histories for selected runs
            for i, run in enumerate(selected):
                selected[i] = _reload_run_history(
                    run, results_dir, task_tag, optimizer, iteration,
                    columns=columns,
                )

            top_n_runs[optimizer] = selected
            print(f"  {optimizer}: {len(selected)} runs")
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
    columns=None,
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
    columns : list[str], optional
        Trajectory columns to load (local/Parquet only).  When given,
        only these columns (plus run_id/epoch/iteration) are read from
        trajectories.parquet.  ``None`` reads all columns.

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
            iteration=iteration, columns=columns,
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
    columns=None,
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
    columns : list[str], optional
        Trajectory columns to load (local/Parquet only).  When given,
        only these columns (plus run_id/epoch/iteration) are read from
        trajectories.parquet.  ``None`` reads all columns.

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
            iteration=iteration, columns=columns,
        )


def load_all_runs(
    backend,
    optimizers,
    task_tag=None,
    results_dir="results",
    iteration=0,
    columns=None,
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
    columns : list[str], optional
        Trajectory columns to load (Parquet only).  ``None`` reads all.

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
                                iteration=iteration, skip_history=True,
                                columns=columns)
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
