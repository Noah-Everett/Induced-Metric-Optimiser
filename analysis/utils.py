"""
Shared constants, helpers, and re-exports for the analysis subpackage.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Re-export optimizer utilities
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent.parent / "parameters"))

from optimizer_registry import get_optimizer_color, get_optimizer_colors, ALL_OPTIMIZERS  # noqa: E402

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

# Summary keys recognised when splitting Parquet rows into config vs summary
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
    "final_min_loss_epoch", "final_min_val_loss",
})

# ---------------------------------------------------------------------------
# Predefined diagnostic metric groups for quick access
# ---------------------------------------------------------------------------

DIAG_KEYS_METRIC_SCALE = [
    "diag/r", "diag/v_hat", "diag/metric_ema",
]
DIAG_KEYS_LEARNABLE = [
    "diag/eff_lr/mean", "diag/metric_condition", "diag/clipped_frac",
    "diag/metric_entropy_norm", "diag/mean_center_offset",
]
DIAG_KEYS_CURVATURE = [
    "diag/H_hat/mean", "diag/positive_curv_frac",
    "diag/curv_corr_ratio", "diag/curv_sign/mean",
]
DIAG_KEYS_TRAINING = [
    "diag/grad_norm", "diag/update_norm", "diag/relative_step",
    "diag/eff_lr_empirical", "diag/mom_grad_cos",
]
DIAG_KEYS_OFFDIAG = [
    "diag/woodbury_det", "diag/M00", "diag/M11",
    "diag/cramer_w0", "diag/cramer_w1",
]


# ---------------------------------------------------------------------------
# Data transformation helpers
# ---------------------------------------------------------------------------

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


def ema(arr, alpha=0.3):
    """Exponential moving average smoothing."""
    result = np.empty_like(arr, dtype=float)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result
