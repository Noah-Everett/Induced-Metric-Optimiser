"""
Text-based analysis functions (paper-ready tables and exports).
"""

import numpy as np
import pandas as pd

from utils import _HP_LOG_SCALE, _NON_HP_PARAMS, compute_speedrun_results


# Variant comparison pairs: (base, variant, label)
_VARIANT_PAIRS = [
    ("sgd_learn_diag", "sgd_learn_diag_log", "log embedding"),
    ("sgd_learn_diag", "sgd_learn_diag_curv", "curvature correction"),
    ("sgd_learn_diag_log", "sgd_learn_diag_curv_log", "curvature correction (log)"),
    ("sgd_learn_diag_curv", "sgd_learn_diag_curv_log", "log embedding (curv)"),
    ("sgd_learn_scalar", "sgd_learn_diag", "diagonal vs scalar"),
    ("sgd_learn_scalar_log", "sgd_learn_diag_log", "diagonal vs scalar (log)"),
]


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

    Uses Mann-Whitney U test for significance and reports median, IQR,
    p-value, and rank-biserial effect size.

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
          f"{'n_base':<8}{'n_var':<8}{'U':<12}{'r':<8}{'Effect':<12}{'p-value':<10}{'Sig?':<6}")
    print("-" * 116)

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
            U, p = mannwhitneyu(var_vals, base_vals, alternative=alt)
        except ValueError:
            U = 0.0
            p = 1.0

        # Rank-biserial effect size: r = 1 - 2U/(n1*n2)
        n1, n2 = len(var_vals), len(base_vals)
        r = 1.0 - 2.0 * U / (n1 * n2) if n1 * n2 > 0 else 0.0
        abs_r = abs(r)
        if abs_r > 0.5:
            effect = "large"
        elif abs_r > 0.3:
            effect = "medium"
        elif abs_r > 0.1:
            effect = "small"
        else:
            effect = "negligible"

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        print(f"{label:<30}{base_med:<11.6f}{var_med:<11.6f}"
              f"{n1:<8}{n2:<8}{U:<12.0f}{r:<+8.3f}{effect:<12}{p:<10.4f}{sig:<6}")
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
