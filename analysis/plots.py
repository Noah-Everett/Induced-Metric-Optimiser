"""
Plotting utilities for analysis notebooks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    get_optimizer_colors,
    _HP_LOG_SCALE, _NON_HP_PARAMS,
    compute_speedrun_results, build_summary_dataframe,
)


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

    # Helper to extract metric values from a run
    def _get_xy(run):
        if backend == "wandb":
            return run.summary.get(x_metric), run.summary.get(y_metric)
        return run["summary"].get(x_metric), run["summary"].get(y_metric)

    # Collect global ranges
    all_x, all_y = [], []
    for opt in optimizers:
        for run in top_n_runs.get(opt, []):
            x_val, y_val = _get_xy(run)
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
            x_val, y_val = _get_xy(run)
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
        Summary key for the epoch / iteration of the best metric.
    best_metric_key : str, optional
        When set, star markers are only drawn on rows where *y_key*
        matches this value.
    converged_key : str, optional
        Summary key for a boolean convergence flag.
    colors : dict, optional
        Mapping from optimizer name to matplotlib colour.
    log_y : str, bool, or dict
        ``"auto"`` (default) enables semilogy for y_keys whose names
        contain ``loss``, ``value``, ``mse``, ``perplexity``, or
        ``error``.
    title : str, optional
        Figure suptitle.
    y_labels : dict, optional
        Custom y-axis labels mapping y_key to display string.
    x_labels : dict, optional
        Custom x-axis labels mapping x_key to display string.
    x_scales : dict, optional
        Scaling factors for x-axes (maps x_key to float multiplier).
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
        Summary key for the epoch of best metric
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
        List of target values
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
        Pre-computed matrix of values
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
        "percent" for percentage, "check" for checkmark, or format string
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
                text = '\u2713' if val == 1.0 else '\u2717'
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
        Summary key to rank by
    optimizers : list, optional
        Subset of optimizers to include
    colors : dict, optional
        Mapping from optimizer name to color
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    figsize : tuple
        Figure size
    ascending : bool
        If True, lower values are better
    title : str
        Plot title
    xlabel : str, optional
        X-axis label
    rankings : list of tuples, optional
        Pre-computed rankings as [(optimizer, value), ...]
    value_format : str
        Format string for value labels

    Returns
    -------
    Figure or None
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
    Create a 2-panel performance summary: efficiency scatter and rankings.

    Returns
    -------
    Figure
        Matplotlib figure with 2 subplots
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
    converged runs as filled circles and diverged runs as crosses.
    Diagonal panels show overlapping histograms for each HP.  Axes use log
    scale for log-uniform parameters (learning rate, eps, xi, etc.).

    Parameters
    ----------
    runs : list[dict]
        Run dicts from ``load_all_runs``, each containing ``"config"``
        and ``"summary"`` sub-dicts.
    optimizer_name : str
        Used for the plot title.
    convergence_percentile : float
        Percentile split when no hard threshold is given.
    convergence_threshold : float or None
        Hard metric threshold.
    convergence_key : str or None
        If set, read convergence directly from summary as a boolean.
    metric_key : str
        Summary key used to determine convergence.
    direction : {"minimize", "maximize"}
        Whether lower or higher metric values mean better performance.
    exclude_params : collection[str] or None
        HP names to drop from the plot.
    figsize : tuple or None
        Figure size.
    title : str or None
        Figure title.
    conv_color : str
        Colour for converged runs.
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

            # ---- Axis labels ----
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

    title = title or f"{optimizer_name} \u2014 HP Sensitivity"
    fig.suptitle(title, fontsize=13, y=1.01)

    return fig, axes


# ---------------------------------------------------------------------------
# Diagnostic visualisation
# ---------------------------------------------------------------------------

def _has_diag_columns(history_data):
    """Check whether any optimizer in history_data has diag/ columns."""
    for opt_data in history_data.values():
        if any(k.startswith("diag/") for k in opt_data):
            return True
    return False


def plot_diagnostic_curves(
    history_data,
    metric_keys,
    title="Optimizer Diagnostics",
    xlabel="Epoch",
    ncols=2,
    figsize_per_plot=(6, 3.5),
    log_y=False,
    colors=None,
):
    """Plot diagnostic metric curves for multiple optimizers.

    Gracefully skips if no diagnostic data is present (returns None).

    Parameters
    ----------
    history_data : dict
        Output of ``extract_history()`` — maps optimizer name to dict of
        numpy arrays.
    metric_keys : list[str]
        List of ``diag/`` metric keys to plot.
    title : str
        Overall figure title.
    xlabel : str
        X-axis label.
    ncols : int
        Columns in subplot grid.
    figsize_per_plot : tuple
        (width, height) per subplot.
    log_y : bool
        Use log scale for y-axis.
    colors : dict, optional
        Optimizer name -> color.

    Returns
    -------
    fig or None
    """
    if not history_data or not _has_diag_columns(history_data):
        print(f"  [skip] No diagnostic data for: {title}")
        return None

    if colors is None:
        colors = get_optimizer_colors(list(history_data.keys()))

    # Filter to keys that actually have data
    available_keys = []
    for key in metric_keys:
        for opt_data in history_data.values():
            arr = opt_data.get(key)
            if arr is not None and not np.all(np.isnan(arr)):
                available_keys.append(key)
                break

    if not available_keys:
        print(f"  [skip] No data for requested keys in: {title}")
        return None

    nrows = max(1, (len(available_keys) + ncols - 1) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
        squeeze=False,
    )

    for idx, key in enumerate(available_keys):
        ax = axes[idx // ncols][idx % ncols]
        short_name = key.replace("diag/", "")
        for opt_name, opt_data in history_data.items():
            epochs = opt_data.get("epoch", opt_data.get("iteration"))
            vals = opt_data.get(key)
            if epochs is None or vals is None:
                continue
            mask = ~np.isnan(vals)
            if not np.any(mask):
                continue
            ax.plot(epochs[mask], vals[mask], label=opt_name,
                    color=colors.get(opt_name, None), alpha=0.8, linewidth=1.2)
        ax.set_title(short_name, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        if log_y:
            ax.set_yscale("log")
        ax.tick_params(labelsize=8)

    # Hide unused axes
    for idx in range(len(available_keys), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Single legend for entire figure
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   ncol=min(len(labels), 6), fontsize=8,
                   bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(title, fontsize=12, y=1.05)
    fig.tight_layout()
    return fig


def plot_effective_lr_comparison(
    history_data,
    title="Effective Learning Rate Comparison",
    xlabel="Epoch",
    figsize=(10, 5),
    colors=None,
):
    """Plot Adam's effective LR vs our eta*r*exp(s_i) side by side.

    Gracefully skips if no diagnostic data is present (returns None).
    """
    if not history_data or not _has_diag_columns(history_data):
        print(f"  [skip] No diagnostic data for: {title}")
        return None

    if colors is None:
        colors = get_optimizer_colors(list(history_data.keys()))

    # Collect optimizers that have effective LR data
    adam_like = {}   # diag/eff_lr/mean from Adam/AdamW
    ours = {}        # diag/eff_lr/mean from our learnable optimizers

    for opt_name, opt_data in history_data.items():
        eff_mean = opt_data.get("diag/eff_lr/mean")
        if eff_mean is None or np.all(np.isnan(eff_mean)):
            continue
        if opt_name in ("adam", "adamw"):
            adam_like[opt_name] = opt_data
        elif opt_name.startswith("sgd_learn"):
            ours[opt_name] = opt_data

    if not adam_like and not ours:
        print(f"  [skip] No effective LR data for: {title}")
        return None

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for opt_name, opt_data in adam_like.items():
        epochs = opt_data.get("epoch", opt_data.get("iteration"))
        mean = opt_data.get("diag/eff_lr/mean")
        if epochs is None or mean is None:
            continue
        mask = ~np.isnan(mean)
        c = colors.get(opt_name)
        axes[0].plot(epochs[mask], mean[mask], label=opt_name, color=c)
        lo = opt_data.get("diag/eff_lr/min")
        hi = opt_data.get("diag/eff_lr/max")
        if lo is not None and hi is not None:
            axes[0].fill_between(epochs[mask], lo[mask], hi[mask],
                                  alpha=0.15, color=c)
    axes[0].set_title(r"Adam: $\eta / (\sqrt{\hat{v}_i} + \varepsilon)$")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Effective LR")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)

    for opt_name, opt_data in ours.items():
        epochs = opt_data.get("epoch", opt_data.get("iteration"))
        mean = opt_data.get("diag/eff_lr/mean")
        if epochs is None or mean is None:
            continue
        mask = ~np.isnan(mean)
        c = colors.get(opt_name)
        axes[1].plot(epochs[mask], mean[mask], label=opt_name, color=c)
        lo = opt_data.get("diag/eff_lr/min")
        hi = opt_data.get("diag/eff_lr/max")
        if lo is not None and hi is not None:
            axes[1].fill_between(epochs[mask], lo[mask], hi[mask],
                                  alpha=0.15, color=c)
    axes[1].set_title(r"Ours: $\eta \cdot r \cdot e^{s_i}$")
    axes[1].set_xlabel(xlabel)
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig
