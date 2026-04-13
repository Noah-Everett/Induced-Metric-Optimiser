"""
Statistical analysis methods for optimizer benchmarking.

Budget-aware comparisons, bootstrap confidence intervals, and
robustness metrics.
"""

import numpy as np
import matplotlib.pyplot as plt

from utils import get_optimizer_colors

__all__ = [
    "bootstrap_best_run_ci",
    "print_topk_comparison",
    "print_robustness_table",
    "plot_best_of_budget",
    "plot_best_of_budget_grid",
]


# ---------------------------------------------------------------------------
# Bootstrap CI on maximum order statistic (IMO-79)
# ---------------------------------------------------------------------------

def bootstrap_best_run_ci(all_runs, metric_key="sweep_metric",
                          direction="minimize", optimizers=None,
                          n_bootstrap=10000, ci_level=0.95, title=None):
    """Bootstrap confidence interval on the best-run metric.

    Resamples N runs with replacement, takes the best from each
    resample, and reports percentile-based confidence intervals.
    Quantifies how much the "best run" depends on lucky HP draws.

    Parameters
    ----------
    all_runs : dict
        Result from ``load_all_runs()``.  Maps optimizer name to list
        of run dicts.
    metric_key : str
        Key in run['summary'] to evaluate.
    direction : str
        "minimize" or "maximize".
    optimizers : list, optional
        Subset of optimizers to include.
    n_bootstrap : int
        Number of bootstrap resamples (default 10,000).
    ci_level : float
        Confidence level (default 0.95 for 95% CI).
    title : str, optional
        Header for the table.

    Returns
    -------
    dict
        Maps optimizer name to dict with keys: best, ci_lo, ci_hi, n_runs.
    """
    if optimizers is None:
        optimizers = list(all_runs.keys())

    alpha = (1 - ci_level) / 2
    best_fn = np.min if direction == "minimize" else np.max

    if title:
        print(f"\n{title}")
    print(f"{'Optimizer':<30}{'Best':<12}{'CI lo':<12}{'CI hi':<12}{'n_runs':<8}")
    print("-" * 74)

    results = {}
    for opt in optimizers:
        run_list = all_runs.get(opt, [])
        vals = np.array([
            float(v) for r in run_list
            if (v := r.get("summary", {}).get(metric_key)) is not None
            and np.isfinite(float(v))
        ])

        if len(vals) < 2:
            continue

        observed_best = float(best_fn(vals))

        # Bootstrap: resample and take best each time
        rng = np.random.default_rng(42)
        boot_bests = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            sample = rng.choice(vals, size=len(vals), replace=True)
            boot_bests[i] = best_fn(sample)

        ci_lo = float(np.percentile(boot_bests, 100 * alpha))
        ci_hi = float(np.percentile(boot_bests, 100 * (1 - alpha)))

        results[opt] = {
            "best": observed_best,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "n_runs": len(vals),
        }

        print(f"{opt:<30}{observed_best:<12.6f}{ci_lo:<12.6f}{ci_hi:<12.6f}{len(vals):<8}")

    print()
    return results


# ---------------------------------------------------------------------------
# Top-k comparison (IMO-77)
# ---------------------------------------------------------------------------

def print_topk_comparison(all_runs, metric_key="sweep_metric",
                          direction="minimize", optimizers=None,
                          k_values=(1, 5, 10, 50), title=None):
    """Print mean metric for the top-k runs of each optimizer.

    Reveals whether performance advantages are structural (persist at
    large k) or noise (only appear at k=1 due to lucky HP draws).

    Parameters
    ----------
    all_runs : dict
        Result from ``load_all_runs()``.
    metric_key : str
        Key in run['summary'] to evaluate.
    direction : str
        "minimize" or "maximize".
    optimizers : list, optional
        Subset of optimizers to include.
    k_values : tuple of int
        Values of k to report (default: 1, 5, 10, 50).
    title : str, optional
        Header for the table.
    """
    if optimizers is None:
        optimizers = list(all_runs.keys())

    if title:
        print(f"\n{title}")

    # Header
    header = f"{'Optimizer':<30}"
    for k in k_values:
        header += f"{'top-' + str(k):<14}"
    header += f"{'n_runs':<8}"
    print(header)
    print("-" * (30 + 14 * len(k_values) + 8))

    reverse = direction == "maximize"

    rows = []
    for opt in optimizers:
        run_list = all_runs.get(opt, [])
        vals = sorted(
            [float(v) for r in run_list
             if (v := r.get("summary", {}).get(metric_key)) is not None
             and np.isfinite(float(v))],
            reverse=reverse,
        )

        if not vals:
            continue

        means = []
        for k in k_values:
            top_k = vals[:min(k, len(vals))]
            means.append(np.mean(top_k))

        # Sort rows by top-1 for display
        rows.append((opt, means, len(vals)))

    # Sort by top-1 (best first)
    rows.sort(key=lambda x: x[1][0], reverse=reverse)

    for opt, means, n in rows:
        row = f"{opt:<30}"
        for m in means:
            row += f"{m:<14.6f}"
        row += f"{n:<8}"
        print(row)
    print()


# ---------------------------------------------------------------------------
# Robustness table (IMO-80)
# ---------------------------------------------------------------------------

def print_robustness_table(all_runs, metric_key="sweep_metric",
                           direction="minimize", optimizers=None,
                           failure_threshold=None, failure_percentile=90,
                           title=None):
    """Print robustness metrics: median, IQR, and failure rate.

    Shows how sensitive each optimizer is to HP choice.  An optimizer
    with a low IQR and low failure rate is robust.

    Parameters
    ----------
    all_runs : dict
        Result from ``load_all_runs()``.
    metric_key : str
        Key in run['summary'] to evaluate.
    direction : str
        "minimize" or "maximize".
    optimizers : list, optional
        Subset of optimizers to include.
    failure_threshold : float, optional
        Metric value beyond which a run is considered a failure.
        If None, computed automatically as the ``failure_percentile``
        of the worst optimizer's distribution.
    failure_percentile : float
        Percentile used to auto-compute failure threshold (default 90).
    title : str, optional
        Header for the table.
    """
    if optimizers is None:
        optimizers = list(all_runs.keys())

    # Collect all values per optimizer
    opt_vals = {}
    for opt in optimizers:
        run_list = all_runs.get(opt, [])
        vals = np.array([
            float(v) for r in run_list
            if (v := r.get("summary", {}).get(metric_key)) is not None
            and np.isfinite(float(v))
        ])
        if len(vals) > 0:
            opt_vals[opt] = vals

    if not opt_vals:
        print("  (no data for robustness table)")
        return

    # Auto-compute failure threshold if not given
    if failure_threshold is None:
        all_vals = np.concatenate(list(opt_vals.values()))
        if direction == "minimize":
            failure_threshold = float(np.percentile(all_vals, failure_percentile))
        else:
            failure_threshold = float(np.percentile(all_vals, 100 - failure_percentile))

    if title:
        print(f"\n{title}")
    print(f"  Failure threshold: {failure_threshold:.6f} "
          f"({'>' if direction == 'minimize' else '<'} = failure)")
    print(f"{'Optimizer':<30}{'Median':<12}{'Q1':<12}{'Q3':<12}"
          f"{'IQR':<12}{'Fail%':<8}{'n':<8}")
    print("-" * 94)

    rows = []
    for opt, vals in opt_vals.items():
        q1 = float(np.percentile(vals, 25))
        median = float(np.median(vals))
        q3 = float(np.percentile(vals, 75))
        iqr = q3 - q1

        if direction == "minimize":
            fail_rate = float(np.mean(vals > failure_threshold))
        else:
            fail_rate = float(np.mean(vals < failure_threshold))

        rows.append((opt, median, q1, q3, iqr, fail_rate, len(vals)))

    # Sort by median (best first)
    reverse = direction == "maximize"
    rows.sort(key=lambda x: x[1], reverse=reverse)

    for opt, median, q1, q3, iqr, fail_rate, n in rows:
        print(f"{opt:<30}{median:<12.6f}{q1:<12.6f}{q3:<12.6f}"
              f"{iqr:<12.6f}{fail_rate:<8.1%}{n:<8}")
    print()


# ---------------------------------------------------------------------------
# Best-of-budget incumbent curve (IMO-76)
# ---------------------------------------------------------------------------

def plot_best_of_budget(all_runs, metric_key="sweep_metric",
                        direction="minimize", optimizers=None,
                        max_budget=None, n_bootstrap=1000,
                        ci_level=0.95, colors=None, figsize=(10, 6),
                        title=None):
    """Plot AlgoPerf-style incumbent (best-so-far) curves.

    For each optimizer, shows how the best metric improves as the HP
    tuning budget (number of trials) increases.  Bootstrap CI bands
    quantify uncertainty from the random trial order.

    Parameters
    ----------
    all_runs : dict
        Result from ``load_all_runs()``.
    metric_key : str
        Key in run['summary'] to evaluate.
    direction : str
        "minimize" or "maximize".
    optimizers : list, optional
        Subset of optimizers to include.
    max_budget : int, optional
        Maximum number of trials to show on x-axis.
        Defaults to the smallest optimizer's run count.
    n_bootstrap : int
        Number of bootstrap resamples for CI bands (default 1000).
    ci_level : float
        Confidence level (default 0.95).
    colors : dict, optional
        Optimizer name to colour mapping.
    figsize : tuple
        Figure size.
    title : str, optional
        Plot title.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    if optimizers is None:
        optimizers = list(all_runs.keys())
    if colors is None:
        colors = get_optimizer_colors(optimizers)

    alpha = (1 - ci_level) / 2
    best_fn = np.minimum.accumulate if direction == "minimize" else np.maximum.accumulate

    # Collect valid metric values per optimizer
    opt_vals = {}
    for opt in optimizers:
        run_list = all_runs.get(opt, [])
        vals = np.array([
            float(v) for r in run_list
            if (v := r.get("summary", {}).get(metric_key)) is not None
            and np.isfinite(float(v))
        ])
        if len(vals) >= 2:
            opt_vals[opt] = vals

    if not opt_vals:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha='center', va='center',
                transform=ax.transAxes)
        return fig

    if max_budget is None:
        max_budget = min(len(v) for v in opt_vals.values())

    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(42)

    for opt, vals in opt_vals.items():
        n = min(len(vals), max_budget)
        budgets = np.arange(1, n + 1)

        # Bootstrap: resample run order, compute incumbent at each budget
        boot_curves = np.empty((n_bootstrap, n))
        for b in range(n_bootstrap):
            perm = rng.permutation(vals)[:n]
            boot_curves[b] = best_fn(perm)

        median_curve = np.median(boot_curves, axis=0)
        ci_lo = np.percentile(boot_curves, 100 * alpha, axis=0)
        ci_hi = np.percentile(boot_curves, 100 * (1 - alpha), axis=0)

        c = colors.get(opt, None)
        ax.plot(budgets, median_curve, label=opt, color=c, linewidth=1.5)
        ax.fill_between(budgets, ci_lo, ci_hi, alpha=0.15, color=c)

    ax.set_xlabel("HP tuning budget (number of trials)")
    ax.set_ylabel(f"Best {metric_key} so far")
    ax.set_title(title or f"Best-of-budget: {metric_key}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_best_of_budget_grid(all_runs_by_func, functions, metric_key="sweep_metric",
                              direction="minimize", colors=None, ncols=3,
                              figsize_per_subplot=(5, 4), n_bootstrap=1000,
                              ci_level=0.95, max_budget=None):
    """Best-of-budget incumbent curves for multiple functions in a single figure.

    Parameters
    ----------
    all_runs_by_func : dict
        ``{func_name: {optimizer: [run, ...], ...}}``
    functions : list[str]
        Function names to plot.
    """
    nrows = int(np.ceil(len(functions) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(figsize_per_subplot[0] * ncols,
                                       figsize_per_subplot[1] * nrows),
                              squeeze=False)

    best_fn = np.minimum.accumulate if direction == "minimize" else np.maximum.accumulate
    alpha = (1 - ci_level) / 2
    rng = np.random.default_rng(42)

    for idx, func_name in enumerate(functions):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        all_runs = all_runs_by_func.get(func_name, {})

        opt_vals = {}
        for opt, runs in all_runs.items():
            vals = np.array([
                float(r["summary"][metric_key])
                for r in runs
                if (v := r.get("summary", {}).get(metric_key)) is not None
                and np.isfinite(float(v))
            ])
            if len(vals) >= 2:
                opt_vals[opt] = vals

        if not opt_vals:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(func_name)
            continue

        budget = max_budget or min(len(v) for v in opt_vals.values())

        for opt, vals in opt_vals.items():
            n = min(len(vals), budget)
            budgets = np.arange(1, n + 1)
            boot_curves = np.empty((n_bootstrap, n))
            for b in range(n_bootstrap):
                perm = rng.permutation(vals)[:n]
                boot_curves[b] = best_fn(perm)
            median_curve = np.median(boot_curves, axis=0)
            ci_lo = np.percentile(boot_curves, 100 * alpha, axis=0)
            ci_hi = np.percentile(boot_curves, 100 * (1 - alpha), axis=0)
            c = colors.get(opt, None) if colors else None
            ax.plot(budgets, median_curve, label=opt, color=c, linewidth=1.2)
            ax.fill_between(budgets, ci_lo, ci_hi, alpha=0.12, color=c)

        ax.set_title(func_name, fontsize=10)
        ax.set_xlabel("Budget (trials)", fontsize=8)
        ax.set_ylabel(f"Best {metric_key}", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for idx in range(len(functions), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    # Single legend from the last populated axis
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc="upper left",
               fontsize=7)
    fig.suptitle("Best-of-Budget Incumbent Curves", fontsize=12)
    fig.tight_layout()
    return fig
