"""Generate step-size vs gradient profile plots for induced metric optimisers."""

import numpy as np
import matplotlib.pyplot as plt
import os

# Output directory (same as this script's location)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Global style
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

FIGSIZE = (6, 4)
DPI = 200
l = np.linspace(-5, 5, 500)


# ---------------------------------------------------------------------------
# Plot 1 -- Fixed diagonal metric
# ---------------------------------------------------------------------------
def plot_fixed():
    """Match Figure 2 of Harvey (2025): vary background curvature a_i.

    In N dimensions with gamma=I, xi=1, the update for component i is
        delta theta^i / eta = -l_i / (a_i + l_i^2)
    where a_i = 1 + sum_{k!=i} l_k^2  is the curvature from other params.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # sum_{k!=i} l_k^2 values; denominator is 1 + this + l_i^2
    bg_vals = [0, 1, 4, 9]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    curves = [-l / (1 + bg + l**2) for bg in bg_vals]
    y_max = max(c.max() for c in curves) * 1.15
    y_min = min(c.min() for c in curves) * 1.15

    # SGD reference
    ax.plot(l, -l, "--", color="0.45", label="SGD")

    for bg, c, curve in zip(bg_vals, colors, curves):
        ax.plot(l, curve, color=c,
                label=rf"$\sum_{{k\neq i}} l_k^2 = {bg}$")

    ax.set_xlabel(r"$l_i$")
    ax.set_ylabel(r"$\delta\theta^i / \eta$")
    ax.set_title(r"Fixed diagonal: step-size profile ($\gamma=I,\;\xi=1$)")
    ax.legend()
    ax.grid(True)
    ax.set_ylim(y_min, y_max)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "step_profile_fixed.png"), dpi=DPI)
    plt.close(fig)
    print("  saved step_profile_fixed.png")


# ---------------------------------------------------------------------------
# Plot 2 -- Learnable scalar metric
# ---------------------------------------------------------------------------
def plot_scalar():
    """Learnable scalar: delta theta^i / eta = -e^s l_i / (a_i + xi e^s l_i^2).

    Fix a_i=1, xi=1, vary s.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    xi, a = 1.0, 1.0

    s_vals = [-2, -1, 0, 1, 2]
    colors = ["#9467bd", "#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"]

    curves = [-np.exp(s) * l / (a + xi * np.exp(s) * l**2) for s in s_vals]
    y_max = max(c.max() for c in curves) * 1.15
    y_min = min(c.min() for c in curves) * 1.15

    for s, c, curve in zip(s_vals, colors, curves):
        style = "--" if s == 0 else "-"
        lbl = rf"$s={s}$" + (" (fixed diag)" if s == 0 else "")
        ax.plot(l, curve, style, color=c, label=lbl)

    ax.set_xlabel(r"$l_i$")
    ax.set_ylabel(r"$\delta\theta^i / \eta$")
    ax.set_title(r"Learnable scalar: effect of metric scale $s$")
    ax.legend()
    ax.grid(True)
    ax.set_ylim(y_min, y_max)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "step_profile_scalar.png"), dpi=DPI)
    plt.close(fig)
    print("  saved step_profile_scalar.png")


# ---------------------------------------------------------------------------
# Plot 3 -- Off-diagonal metric (1-D special cases)
# ---------------------------------------------------------------------------
def plot_offdiag():
    """Off-diagonal variants in 1D (signed, full range)."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    xi = 1.0

    c_fixed = -l / (1 + xi * l**2)
    c_one   = -l / (1 + 2 * xi * l**2)
    c_sym   = -l / (1 + 3 * xi * l**2)
    y_max = c_fixed.max() * 1.15
    y_min = c_fixed.min() * 1.15

    ax.plot(l, -l, "--", color="0.45", label="SGD")
    ax.plot(l, c_fixed, color="#1f77b4",
            label=r"$a=b=0$ (fixed diag, eff.\ $\xi=1$)")
    ax.plot(l, c_one, color="#d62728",
            label=r"$a{=}0,b{=}\ell$ (eff.\ $\xi=2$)")
    ax.plot(l, c_sym, color="#2ca02c",
            label=r"$a=b=\ell$ (eff.\ $\xi=3$)")

    ax.set_xlabel(r"$l_i$")
    ax.set_ylabel(r"$\delta\theta^i / \eta$")
    ax.set_title(r"Off-diagonal: coupling variants ($\xi=1$)")
    ax.legend()
    ax.grid(True)
    ax.set_ylim(y_min, y_max)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "step_profile_offdiag.png"), dpi=DPI)
    plt.close(fig)
    print("  saved step_profile_offdiag.png")


# ---------------------------------------------------------------------------
# Plot 4 -- All variants together
# ---------------------------------------------------------------------------
def plot_comparison():
    """All variants on one signed plot."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    xi = 1.0

    es = np.exp(1.0)
    c_fixed   = -l / (1 + xi * l**2)
    c_scalar  = -es * l / (1 + xi * es * l**2)
    c_offdiag = -l / (1 + 3 * xi * l**2)
    all_curves = [c_fixed, c_scalar, c_offdiag]
    y_max = max(c.max() for c in all_curves) * 1.15
    y_min = min(c.min() for c in all_curves) * 1.15

    ax.plot(l, -l, "--", color="0.45", label="SGD")
    ax.plot(l, c_fixed, color="#1f77b4", label=r"Fixed diag ($\xi=1$)")
    ax.plot(l, c_scalar, color="#d62728",
            label=r"Learnable scalar ($\xi=1,\,s=1$)")
    ax.plot(l, c_offdiag, color="#2ca02c",
            label=r"Off-diag $a{=}b{=}\ell$ ($\xi=1$)")

    ax.set_xlabel(r"$l_i$")
    ax.set_ylabel(r"$\delta\theta^i / \eta$")
    ax.set_title("Step-size profiles: all variants")
    ax.legend()
    ax.grid(True)
    ax.set_ylim(y_min, y_max)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "step_profile_comparison.png"), dpi=DPI)
    plt.close(fig)
    print("  saved step_profile_comparison.png")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating step-size profile plots ...")
    plot_fixed()
    plot_scalar()
    plot_offdiag()
    plot_comparison()
    print("Done.")
