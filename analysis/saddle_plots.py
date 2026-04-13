"""Plotting functions for saddle-point analysis.

Extracted from ``saddle_analysis.ipynb`` so that notebooks can
``from saddle_plots import *`` and stay thin.  Every function that
was originally defined inline in a notebook cell lives here; closure
variables have been converted to explicit parameters.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------------------------------------------------------------------------
# Constants (from notebook cell 2 / cell 4)
# ---------------------------------------------------------------------------

FIGSIZE: Tuple[int, int] = (6, 4)
FIGSIZE_WIDE: Tuple[int, int] = (10, 4.2)  # type: ignore[assignment]
DPI: int = 200

FUNCTION_TITLES: Dict[str, str] = {
    "saddle": r"$L = x^2 - y^2$",
    "weighted_saddle": r"$L = 3x^2 - y^2$",
}

HESSIAN_DIAG: Dict[str, Tuple[float, float]] = {
    "saddle": (2.0, -2.0),
    "weighted_saddle": (6.0, -2.0),
}

FULL_TENSOR_KEYS: List[str] = [
    "diag/full/param/root/0", "diag/full/param/root/1",
    "diag/full/log_diag/root/0", "diag/full/log_diag/root/1",
    "diag/full/grad/root/0", "diag/full/grad/root/1",
    "diag/full/H_hat/0", "diag/full/H_hat/1",
    "diag/full/curv_sign/0", "diag/full/curv_sign/1",
    "diag/full/mom/root/0", "diag/full/mom/root/1",
    "diag/full/update/root/0", "diag/full/update/root/1",
]

# Short aliases for readability in plotting code
KEY_PARAM_X: str = "diag/full/param/root/0"
KEY_PARAM_Y: str = "diag/full/param/root/1"
KEY_S0: str = "diag/full/log_diag/root/0"
KEY_S1: str = "diag/full/log_diag/root/1"
KEY_H0: str = "diag/full/H_hat/0"
KEY_H1: str = "diag/full/H_hat/1"
KEY_CS0: str = "diag/full/curv_sign/0"
KEY_CS1: str = "diag/full/curv_sign/1"


# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------

def set_publication_style() -> None:
    """Apply publication-quality matplotlib rcParams.

    Matches the style block in cell 0 of ``saddle_analysis.ipynb``.
    """
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 12,
        "legend.fontsize": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "lines.linewidth": 1.8,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "grid.alpha": 0.35,
        "grid.linewidth": 0.5,
    })


# ---------------------------------------------------------------------------
# Helper: loss grid (JAX imported locally)
# ---------------------------------------------------------------------------

def _loss_grid(
    func_name: str,
    xlim: Tuple[float, float] = (-2, 2),
    ylim: Tuple[float, float] = (-2, 2),
    n: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute loss values on a grid for contour plotting.

    JAX is imported inside the function body so that this module can be
    imported without triggering JAX initialisation.
    """
    import jax  # noqa: E402  — intentionally function-local
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    # Import the test-function registry from sweep_saddle
    sys.path.insert(0, os.path.join(os.getcwd(), "..", "parameters"))
    from sweep_saddle import TEST_FUNCTIONS  # type: ignore[import-untyped]

    func = TEST_FUNCTIONS[func_name]["func"]
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], n)
    X, Y = np.meshgrid(xs, ys)
    Z = np.array([[float(func(jnp.array([x, y]))) for x in xs] for y in ys])
    return X, Y, Z


# ---------------------------------------------------------------------------
# Helper: auto axis limits
# ---------------------------------------------------------------------------

def _auto_lims(
    history_data: Dict[str, Dict[str, Any]],
    pad: float = 0.2,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute axis limits from all optimizer trajectories, with padding."""
    all_x: List[float] = []
    all_y: List[float] = []
    for _opt_name, data in history_data.items():
        px = data.get(KEY_PARAM_X)
        py = data.get(KEY_PARAM_Y)
        if px is not None and py is not None:
            all_x.extend(px.tolist())
            all_y.extend(py.tolist())
    if not all_x:
        return (-2, 2), (-2, 2)
    # Clamp to finite values and cap range to avoid contour failures
    all_x = [v for v in all_x if np.isfinite(v)]
    all_y = [v for v in all_y if np.isfinite(v)]
    if not all_x:
        return (-2, 2), (-2, 2)
    xmin, xmax = max(min(all_x), -50), min(max(all_x), 50)
    ymin, ymax = max(min(all_y), -50), min(max(all_y), 50)
    dx = max(xmax - xmin, 0.5) * pad
    dy = max(ymax - ymin, 0.5) * pad
    return (xmin - dx, xmax + dx), (ymin - dy, ymax + dy)


# ---------------------------------------------------------------------------
# Plot 1: Trajectory comparison  (cell 6)
# ---------------------------------------------------------------------------

def plot_trajectories(
    func_name: str,
    history_data: Dict[str, Dict[str, Any]],
    optimizers_to_plot: Optional[Sequence[str]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    n_contour: int = 200,
    *,
    optimizers: Optional[Sequence[str]] = None,
    function_titles: Optional[Dict[str, str]] = None,
) -> Optional[plt.Figure]:
    """Plot optimizer trajectories on loss contours.

    Parameters
    ----------
    func_name : str
        Key into the test-function registry (e.g. ``"saddle"``).
    history_data : dict
        ``{optimizer_name: {key: array, ...}, ...}`` for *one* function.
    optimizers_to_plot : list[str], optional
        Subset of optimizers to show.  Defaults to all keys in
        *history_data* that also appear in *optimizers*.
    xlim, ylim : tuple, optional
        Manual axis limits.  Auto-scaled from trajectories when *None*.
    n_contour : int
        Resolution of the background contour grid.
    optimizers : list[str], optional
        Full optimizer list (used for default filtering).
    function_titles : dict, optional
        ``{func_name: LaTeX title}``.  Falls back to module-level
        ``FUNCTION_TITLES``.
    """
    if function_titles is None:
        function_titles = FUNCTION_TITLES
    if optimizers is None:
        optimizers = list(history_data.keys())

    if optimizers_to_plot is None:
        optimizers_to_plot = [k for k in history_data if k in optimizers]

    n_opts = len(optimizers_to_plot)
    if n_opts == 0:
        print(f"  [skip] No trajectory data for {func_name}")
        return None

    # Auto-scale to trajectory extent if limits not given
    if xlim is None or ylim is None:
        auto_xl, auto_yl = _auto_lims(history_data)
        xlim = xlim or auto_xl
        ylim = ylim or auto_yl

    X, Y, Z = _loss_grid(func_name, xlim, ylim, n_contour)
    title = function_titles.get(func_name, func_name)

    fig, axes = plt.subplots(1, n_opts, figsize=(5 * n_opts, 4.5), squeeze=False)
    axes = axes[0]

    for ax, opt_name in zip(axes, optimizers_to_plot):
        data = history_data.get(opt_name, {})
        px = data.get(KEY_PARAM_X)
        py = data.get(KEY_PARAM_Y)

        ax.contour(X, Y, Z, levels=20, colors="0.75", linewidths=0.5)
        ax.contourf(X, Y, Z, levels=20, cmap="RdBu_r", alpha=0.3)

        if px is not None and py is not None:
            n_steps = len(px)
            cmap_traj = cm.viridis
            step_colors = [cmap_traj(i / max(n_steps - 1, 1)) for i in range(n_steps)]
            ax.scatter(px, py, c=step_colors, s=8, zorder=3, edgecolors="none")
            ax.plot(px, py, color="0.4", linewidth=0.5, alpha=0.5, zorder=2)
            ax.plot(px[0], py[0], "ko", markersize=7, zorder=4, label="Start")
            ax.plot(px[-1], py[-1], "r*", markersize=10, zorder=4, label="End")

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_title(opt_name.replace("_", r"\_"))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if ax is axes[0]:
            ax.legend(fontsize=8)

    fig.suptitle(f"Trajectories: {title}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ---------------------------------------------------------------------------
# Plot 2: Metric parameter (s_i) evolution  (cell 8)
# ---------------------------------------------------------------------------

def plot_s_evolution(
    func_name: str,
    history_data: Dict[str, Dict[str, Any]],
    optimizers_to_plot: Optional[Sequence[str]] = None,
    *,
    optimizers: Optional[Sequence[str]] = None,
    function_titles: Optional[Dict[str, str]] = None,
) -> Optional[plt.Figure]:
    """Plot s_i evolution for each optimizer side by side."""
    if function_titles is None:
        function_titles = FUNCTION_TITLES
    if optimizers is None:
        optimizers = list(history_data.keys())

    if optimizers_to_plot is None:
        optimizers_to_plot = [k for k in history_data if k in optimizers]

    n_opts = len(optimizers_to_plot)
    if n_opts == 0:
        return None

    title = function_titles.get(func_name, func_name)
    fig, axes = plt.subplots(1, n_opts, figsize=(5 * n_opts, 4), squeeze=False)
    axes = axes[0]

    for ax, opt_name in zip(axes, optimizers_to_plot):
        data = history_data.get(opt_name, {})
        s0 = data.get(KEY_S0)
        s1 = data.get(KEY_S1)

        if s0 is not None and s1 is not None:
            steps = np.arange(len(s0))
            ax.plot(steps, s0, color="#1f77b4", label=r"$s_1$ (x-dir)")
            ax.plot(steps, s1, color="#d62728", label=r"$s_2$ (y-dir)")
            ax.axhline(0, color="0.5", linewidth=0.5, linestyle=":")
        else:
            ax.text(
                0.5, 0.5,
                "No $s_i$ data\n(not a learnable optimizer)",
                transform=ax.transAxes, ha="center", va="center", fontsize=10,
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$s_i$")
        ax.set_title(opt_name.replace("_", r"\_"))
        ax.legend(fontsize=8)
        ax.grid(True)

    fig.suptitle(f"Metric parameters: {title}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ---------------------------------------------------------------------------
# Plot 3: Loss convergence  (cell 10)
# ---------------------------------------------------------------------------

def plot_saddle_loss_curves(
    func_name: str,
    history_data: Dict[str, Dict[str, Any]],
    optimizers_to_plot: Optional[Sequence[str]] = None,
    *,
    optimizers: Optional[Sequence[str]] = None,
    colors: Optional[Dict[str, str]] = None,
    function_titles: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """Plot loss vs iteration for all optimizers on one axis.

    Renamed from ``plot_loss_curves`` (cell 10) to
    ``plot_saddle_loss_curves`` to avoid collision with the generic helper
    in ``analysis_utils``.
    """
    if function_titles is None:
        function_titles = FUNCTION_TITLES
    if optimizers is None:
        optimizers = list(history_data.keys())
    if colors is None:
        colors = {}

    if optimizers_to_plot is None:
        optimizers_to_plot = [k for k in history_data if k in optimizers]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    title = function_titles.get(func_name, func_name)

    for opt_name in optimizers_to_plot:
        data = history_data.get(opt_name, {})
        loss = data.get("function_value")
        if loss is not None:
            steps = np.arange(len(loss))
            ax.plot(
                steps, np.abs(loss),
                color=colors.get(opt_name, "#333"),
                label=opt_name.replace("_", r"\_"),
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$|L(\theta)|$")
    ax.set_yscale("log")
    ax.set_title(f"Loss convergence: {title}")
    ax.legend(fontsize=9)
    ax.grid(True)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 4: Secant Hessian estimates  (cell 12)
# ---------------------------------------------------------------------------

def plot_hessian_estimates(
    func_name: str,
    history_data: Dict[str, Dict[str, Any]],
    curv_optimizer: str = "sgd_learn_diag_curv",
    *,
    function_titles: Optional[Dict[str, str]] = None,
    hessian_diag: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Optional[plt.Figure]:
    """Plot secant Hessian estimates and curv_sign for the curvature-aware optimizer."""
    if function_titles is None:
        function_titles = FUNCTION_TITLES
    if hessian_diag is None:
        hessian_diag = HESSIAN_DIAG

    data = history_data.get(curv_optimizer, {})
    h0 = data.get(KEY_H0)
    h1 = data.get(KEY_H1)
    cs0 = data.get(KEY_CS0)
    cs1 = data.get(KEY_CS1)

    if h0 is None or h1 is None:
        print(f"  [skip] No H_hat data for {func_name} ({curv_optimizer})")
        return None

    title = function_titles.get(func_name, func_name)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    steps = np.arange(len(h0))

    # Left: H_hat values
    ax1.plot(steps, h0, color="#1f77b4", alpha=0.7, label=r"$\hat{H}_{11}$ (x-dir)")
    ax1.plot(steps, h1, color="#d62728", alpha=0.7, label=r"$\hat{H}_{22}$ (y-dir)")

    # Ground truth dashed lines
    if func_name in hessian_diag:
        h_true = hessian_diag[func_name]
        ax1.axhline(
            h_true[0], color="#1f77b4", linestyle="--", linewidth=1, alpha=0.5,
            label=f"True $H_{{11}} = {h_true[0]:.0f}$",
        )
        ax1.axhline(
            h_true[1], color="#d62728", linestyle="--", linewidth=1, alpha=0.5,
            label=f"True $H_{{22}} = {h_true[1]:.0f}$",
        )

    ax1.axhline(0, color="0.5", linewidth=0.5, linestyle=":")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel(r"$\hat{H}_{ii}$")
    ax1.set_title("Secant Hessian diagonal estimates")
    ax1.legend(fontsize=8)
    ax1.grid(True)

    # Right: curv_sign = tanh(H_hat / tau)
    if cs0 is not None and cs1 is not None:
        ax2.plot(steps, cs0, color="#1f77b4", alpha=0.7,
                 label=r"$\tanh(\hat{H}_{11}/\tau)$")
        ax2.plot(steps, cs1, color="#d62728", alpha=0.7,
                 label=r"$\tanh(\hat{H}_{22}/\tau)$")

        if func_name in hessian_diag:
            h_true = hessian_diag[func_name]
            ax2.axhline(
                np.tanh(h_true[0]), color="#1f77b4", linestyle="--",
                linewidth=1, alpha=0.5,
            )
            ax2.axhline(
                np.tanh(h_true[1]), color="#d62728", linestyle="--",
                linewidth=1, alpha=0.5,
            )

    ax2.axhline(0, color="0.5", linewidth=0.5, linestyle=":")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel(r"$\tanh(\hat{H}_{ii} / \tau)$")
    ax2.set_title("Curvature sign (used in update rule)")
    ax2.legend(fontsize=8)
    ax2.grid(True)

    fig.suptitle(
        f"Hessian estimates: {title} ({curv_optimizer.replace('_', chr(92) + '_')})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ---------------------------------------------------------------------------
# Plot 5: Summary grid — all functions  (cell 14)
# ---------------------------------------------------------------------------

def plot_saddle_summary_grid(
    all_history_data: Dict[str, Dict[str, Dict[str, Any]]],
    functions: Sequence[str],
    optimizers: Sequence[str],
    colors: Dict[str, str],
    *,
    function_titles: Optional[Dict[str, str]] = None,
) -> Optional[plt.Figure]:
    """Three-column grid (trajectory, s_i, loss) with one row per function.

    Parameters
    ----------
    all_history_data : dict
        ``{func_name: {optimizer: {key: array}}}``.
    functions : list[str]
        Ordered list of function names to include.
    optimizers : list[str]
        Canonical optimizer order (used for filtering and colour lookup).
    colors : dict
        ``{optimizer_name: colour_string}``.
    function_titles : dict, optional
        ``{func_name: LaTeX title}``.
    """
    if function_titles is None:
        function_titles = FUNCTION_TITLES

    available_funcs = [f for f in functions if f in all_history_data]
    n_funcs = len(available_funcs)

    if n_funcs == 0:
        print("No data available for grid plot.")
        return None

    fig, axes = plt.subplots(n_funcs, 3, figsize=(15, 4 * n_funcs))
    if n_funcs == 1:
        axes = axes[np.newaxis, :]

    for row, func_name in enumerate(available_funcs):
        history_data = all_history_data[func_name]
        title = function_titles.get(func_name, func_name)
        opts = [k for k in history_data if k in optimizers]

        # Auto-scale to trajectory extent
        xlim, ylim = _auto_lims(history_data)

        # Column 0: Trajectories overlaid
        ax = axes[row, 0]
        X, Y, Z = _loss_grid(func_name, xlim, ylim)
        ax.contour(X, Y, Z, levels=20, colors="0.75", linewidths=0.4)
        ax.contourf(X, Y, Z, levels=20, cmap="RdBu_r", alpha=0.2)
        for opt_name in opts:
            data = history_data.get(opt_name, {})
            px = data.get(KEY_PARAM_X)
            py = data.get(KEY_PARAM_Y)
            if px is not None and py is not None:
                ax.plot(
                    px, py,
                    color=colors.get(opt_name, "#333"),
                    linewidth=1.2, alpha=0.8,
                    label=opt_name.replace("_", r"\_"),
                )
                ax.plot(px[0], py[0], "ko", markersize=5, zorder=4)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if row == 0:
            ax.set_title("Trajectory")
            ax.legend(fontsize=6, loc="upper right")
        ax.text(
            -0.15, 0.5, title,
            transform=ax.transAxes, fontsize=11,
            va="center", ha="right", rotation=90,
        )

        # Column 1: s_i evolution (overlaid, diag and curv only)
        ax = axes[row, 1]
        has_s = False
        for opt_name in opts:
            data = history_data.get(opt_name, {})
            s0 = data.get(KEY_S0)
            s1 = data.get(KEY_S1)
            if s0 is not None and s1 is not None:
                has_s = True
                c = colors.get(opt_name, "#333")
                steps = np.arange(len(s0))
                lbl = opt_name.replace("_", r"\_")
                ax.plot(steps, s0, color=c, linestyle="-", alpha=0.8,
                        label=f"{lbl} $s_1$")
                ax.plot(steps, s1, color=c, linestyle="--", alpha=0.8,
                        label=f"{lbl} $s_2$")
        ax.axhline(0, color="0.5", linewidth=0.5, linestyle=":")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$s_i$")
        if row == 0:
            ax.set_title(r"Metric parameters ($s_1$ solid, $s_2$ dashed)")
        if has_s:
            ax.legend(fontsize=6, ncol=2)
        ax.grid(True)

        # Column 2: Loss curves (all optimizers)
        ax = axes[row, 2]
        for opt_name in opts:
            data = history_data.get(opt_name, {})
            loss = data.get("function_value")
            if loss is not None:
                steps = np.arange(len(loss))
                ax.plot(
                    steps, np.abs(loss),
                    color=colors.get(opt_name, "#333"),
                    label=opt_name.replace("_", r"\_"),
                )
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$|L(\theta)|$")
        ax.set_yscale("log")
        if row == 0:
            ax.set_title("Loss convergence")
            ax.legend(fontsize=6)
        ax.grid(True)

    fig.suptitle("Saddle experiment: all functions", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 6: Metric anisotropy  (cell 16)
# ---------------------------------------------------------------------------

def plot_metric_anisotropy(
    func_name: str,
    history_data: Dict[str, Dict[str, Any]],
    learnable_opts: Sequence[str] = ("sgd_learn_diag", "sgd_learn_diag_curv"),
    *,
    colors: Optional[Dict[str, str]] = None,
    function_titles: Optional[Dict[str, str]] = None,
) -> Optional[plt.Figure]:
    r"""Plot :math:`|s_1 - s_2|` for learnable optimizers."""
    if function_titles is None:
        function_titles = FUNCTION_TITLES
    if colors is None:
        colors = {}

    title = function_titles.get(func_name, func_name)
    fig, ax = plt.subplots(figsize=FIGSIZE)

    has_data = False
    for opt_name in learnable_opts:
        data = history_data.get(opt_name, {})
        s0 = data.get(KEY_S0)
        s1 = data.get(KEY_S1)
        if s0 is not None and s1 is not None:
            has_data = True
            aniso = np.abs(s0 - s1)
            steps = np.arange(len(aniso))
            ax.plot(
                steps, aniso,
                color=colors.get(opt_name, "#333"),
                label=opt_name.replace("_", r"\_"),
            )

    if not has_data:
        plt.close(fig)
        print(f"  [skip] No s_i data for anisotropy plot ({func_name})")
        return None

    ax.axhline(0, color="0.5", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$|s_1 - s_2|$")
    ax.set_title(f"Metric anisotropy: {title}")
    ax.legend(fontsize=9)
    ax.grid(True)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 7: Clip comparison  (cell 18)
# ---------------------------------------------------------------------------

def plot_clip_comparison(
    func_name: str,
    all_data: Dict[int, Dict[str, Dict[str, Dict[str, Any]]]],
    learnable_opts: Sequence[str] = ("sgd_learn_diag", "sgd_learn_diag_curv"),
    *,
    function_titles: Optional[Dict[str, str]] = None,
) -> Optional[plt.Figure]:
    """Full comparison: clipped (itr 0) vs unclipped (itr 1).

    4 rows: trajectory, s_i evolution, anisotropy, loss.
    2 columns: one per learnable optimizer.
    Each panel overlays both iterations.

    Parameters
    ----------
    func_name : str
        Function key (e.g. ``"saddle"``).
    all_data : dict
        ``{iteration_index: {func_name: {optimizer: {key: array}}}}``.
        Must contain keys ``0`` (clipped) and ``1`` (unclipped).
    learnable_opts : sequence of str
        Which learnable optimizers to compare.
    function_titles : dict, optional
        ``{func_name: LaTeX title}``.
    """
    if function_titles is None:
        function_titles = FUNCTION_TITLES

    d0 = all_data.get(0, {}).get(func_name)
    d1 = all_data.get(1, {}).get(func_name)
    if d0 is None or d1 is None:
        print(f"  [skip] Need both iterations for clip comparison ({func_name})")
        return None

    title = function_titles.get(func_name, func_name)
    n_opts = len(learnable_opts)
    fig, axes = plt.subplots(4, n_opts, figsize=(5 * n_opts, 16), squeeze=False)

    itr_styles = {0: ("-", 1.0, "With clip"), 1: ("--", 0.7, "No clip")}

    for col, opt_name in enumerate(learnable_opts):
        opt_label = opt_name.replace("_", r"\_")

        # Row 0: Trajectory ------------------------------------------------
        ax = axes[0, col]
        # Recompute auto lims from both iterations for this optimizer
        all_px: List[float] = []
        all_py: List[float] = []
        for itr in [0, 1]:
            d = all_data[itr].get(func_name, {}).get(opt_name, {})
            px, py = d.get(KEY_PARAM_X), d.get(KEY_PARAM_Y)
            if px is not None:
                all_px.extend(px.tolist())
                all_py.extend(py.tolist())
        if all_px:
            pad = 0.2
            xmn, xmx = min(all_px), max(all_px)
            ymn, ymx = min(all_py), max(all_py)
            dx = max(xmx - xmn, 0.5) * pad
            dy = max(ymx - ymn, 0.5) * pad
            xlim: Tuple[float, float] = (xmn - dx, xmx + dx)
            ylim: Tuple[float, float] = (ymn - dy, ymx + dy)
        else:
            xlim, ylim = (-2, 2), (-2, 2)

        X, Y, Z = _loss_grid(func_name, xlim, ylim, n=100)
        ax.contour(X, Y, Z, levels=20, colors="0.75", linewidths=0.4)
        ax.contourf(X, Y, Z, levels=20, cmap="RdBu_r", alpha=0.2)
        for itr, (ls, alpha, lbl) in itr_styles.items():
            d = all_data[itr].get(func_name, {}).get(opt_name, {})
            px, py = d.get(KEY_PARAM_X), d.get(KEY_PARAM_Y)
            if px is not None:
                c = "#1f77b4" if itr == 0 else "#d62728"
                ax.plot(px, py, color=c, linestyle=ls, linewidth=1.2,
                        alpha=alpha, label=lbl)
                ax.plot(px[0], py[0], "ko", markersize=5, zorder=4)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"{opt_label}: trajectory")
        ax.legend(fontsize=8)

        # Row 1: s_i evolution ---------------------------------------------
        ax = axes[1, col]
        for itr, (ls, alpha, lbl) in itr_styles.items():
            d = all_data[itr].get(func_name, {}).get(opt_name, {})
            s0, s1 = d.get(KEY_S0), d.get(KEY_S1)
            if s0 is not None:
                steps = np.arange(len(s0))
                ax.plot(steps, s0, color="#1f77b4", linestyle=ls, alpha=alpha,
                        label=f"$s_1$ ({lbl})")
                ax.plot(steps, s1, color="#d62728", linestyle=ls, alpha=alpha,
                        label=f"$s_2$ ({lbl})")
        ax.axhline(0, color="0.5", linewidth=0.5, linestyle=":")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$s_i$")
        ax.set_title(f"{opt_label}: metric parameters")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True)

        # Row 2: Anisotropy |s_1 - s_2| ------------------------------------
        ax = axes[2, col]
        for itr, (ls, alpha, lbl) in itr_styles.items():
            d = all_data[itr].get(func_name, {}).get(opt_name, {})
            s0, s1 = d.get(KEY_S0), d.get(KEY_S1)
            if s0 is not None:
                aniso = np.abs(s0 - s1)
                steps = np.arange(len(aniso))
                c = "#1f77b4" if itr == 0 else "#d62728"
                ax.plot(steps, aniso, color=c, linestyle=ls, label=lbl)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$|s_1 - s_2|$")
        ax.set_title(f"{opt_label}: metric anisotropy")
        ax.legend(fontsize=8)
        ax.grid(True)

        # Row 3: Loss -------------------------------------------------------
        ax = axes[3, col]
        for itr, (ls, alpha, lbl) in itr_styles.items():
            d = all_data[itr].get(func_name, {}).get(opt_name, {})
            loss = d.get("function_value")
            if loss is not None:
                steps = np.arange(len(loss))
                c = "#1f77b4" if itr == 0 else "#d62728"
                ax.plot(steps, np.abs(loss), color=c, linestyle=ls,
                        alpha=alpha, label=lbl)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$|L(\theta)|$")
        ax.set_yscale("log")
        ax.set_title(f"{opt_label}: loss")
        ax.legend(fontsize=8)
        ax.grid(True)

    fig.suptitle(f"Clip comparison: {title}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig
