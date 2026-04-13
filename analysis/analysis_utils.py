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

This module re-exports everything from the analysis submodules:
  - utils:   constants, helpers, optimizer registry re-exports
  - loaders: data loading (WandB, local, Parquet)
  - plots:   all plotting functions
  - tables:  text-based tables and exports
  - stats:   statistical analysis methods
"""

import sys
from pathlib import Path

# Ensure this directory is on sys.path so sibling modules resolve
_DIR = str(Path(__file__).parent)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

# ---------------------------------------------------------------------------
# Re-export from submodules
# ---------------------------------------------------------------------------

from utils import (  # noqa: F401, E402
    get_optimizer_color,
    get_optimizer_colors,
    ALL_OPTIMIZERS,
    _HP_LOG_SCALE,
    _NON_HP_PARAMS,
    _SUMMARY_KEYS,
    DIAG_KEYS_METRIC_SCALE,
    DIAG_KEYS_LEARNABLE,
    DIAG_KEYS_CURVATURE,
    DIAG_KEYS_TRAINING,
    DIAG_KEYS_OFFDIAG,
    compute_speedrun_results,
    build_summary_dataframe,
    ema,
)

from loaders import (  # noqa: F401, E402
    load_best_runs,
    load_top_n_runs,
    load_all_runs,
    extract_history,
    compute_wall_times,
    add_wall_times,
)

from plots import (  # noqa: F401, E402
    plot_2d_histograms,
    plot_training_curves,
    plot_convergence_curves,
    plot_speedrun_results,
    plot_time_to_best,
    plot_convergence_heatmap,
    plot_efficiency_scatter,
    plot_ranking_bar,
    plot_performance_summary,
    plot_grouped_bars,
    plot_multi_function_performance,
    plot_hp_sensitivity,
    plot_diagnostic_curves,
    plot_effective_lr_comparison,
    plot_all_diagnostics,
    plot_convergence_curves_grid,
)

from tables import (  # noqa: F401, E402
    _VARIANT_PAIRS,
    print_speedrun_table,
    print_convergence_summary,
    print_best_hyperparameters,
    save_best_hyperparameters,
    print_ranking_table,
    print_variant_comparison,
    print_variant_comparison_allruns,
    print_hp_sensitivity_table,
    print_offdiag_ranking,
    print_cross_task_summary,
)

from stats import *  # noqa: F401, F403, E402

from task_configs import TASK_CONFIGS  # noqa: F401, E402

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # utils
    "get_optimizer_color",
    "get_optimizer_colors",
    "ALL_OPTIMIZERS",
    "DIAG_KEYS_METRIC_SCALE",
    "DIAG_KEYS_LEARNABLE",
    "DIAG_KEYS_CURVATURE",
    "DIAG_KEYS_TRAINING",
    "DIAG_KEYS_OFFDIAG",
    "compute_speedrun_results",
    "build_summary_dataframe",
    "ema",
    # loaders
    "load_best_runs",
    "load_top_n_runs",
    "load_all_runs",
    "extract_history",
    "compute_wall_times",
    "add_wall_times",
    # plots
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
    "plot_diagnostic_curves",
    "plot_effective_lr_comparison",
    "plot_all_diagnostics",
    "plot_convergence_curves_grid",
    # tables
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
    # stats
    "bootstrap_best_run_ci",
    "print_topk_comparison",
    "print_robustness_table",
    "plot_best_of_budget",
    "plot_best_of_budget_grid",
    # task configs
    "TASK_CONFIGS",
]
