"""Reproduce the geodesic convexity analysis plots (originally from Mathematica).

Key result: for the induced metric g = I + xi l x l,
    (Hess_g L)_ij = H_ij / (1 + xi ||nabla L||^2)

Eigenvalues scale uniformly, condition number is preserved, sign structure unchanged.

Generates:
  geodesic_condition_number.png
  geodesic_scaling_factor.png
  geodesic_nonconvexity_map.png
  geodesic_eigenvalue_min.png
  geodesic_saddle_eigenvalues.png
"""

import numpy as np
import matplotlib.pyplot as plt
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Style matching generate_step_profiles.py
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


# ---- Test functions --------------------------------------------------------

def rosenbrock_grad(x, y):
    dLdx = -2 * (1 - x) + 200 * (y - x**2) * (-2 * x)
    dLdy = 200 * (y - x**2)
    return np.array([dLdx, dLdy])


def rosenbrock_hessian(x, y):
    return np.array([
        [2 - 400 * (y - x**2) + 800 * x**2, -400 * x],
        [-400 * x, 200],
    ])


# ---- Plot 1: Condition Number ----------------------------------------------

def plot_condition_number():
    r"""kappa(H) vs kappa(Hess_g) for L = x^2 + 4y^2.  Both equal 4."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    xi_vals = np.linspace(0, 100, 500)
    kappa = np.full_like(xi_vals, 4.0)

    ax.plot(xi_vals, kappa, color="#d62728", linewidth=2.5,
            label=r"$\kappa(H)$")
    ax.plot(xi_vals, kappa, "--", color="#1f77b4", linewidth=2.5,
            label=r"$\kappa(\mathrm{Hess}_g\, L)$")

    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel("Condition number")
    ax.set_title(r"Condition number vs $\xi$ \quad ($L = x^2 + 4y^2$)")
    ax.legend()
    ax.grid(True)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "geodesic_condition_number.png"), dpi=DPI)
    plt.close(fig)
    print("  saved geodesic_condition_number.png")


# ---- Plot 2: Scaling Factor -----------------------------------------------

def plot_scaling_factor():
    r"""1/(1 + xi ||nabla L||^2) along the Rosenbrock valley y = x^2.

    Along y = x^2 the gradient simplifies to ||nabla L||^2 = 4(1-x)^2,
    giving a clean bell-shaped profile peaking at the minimum (1, 1).
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)

    t = np.linspace(0, 1, 2000)
    x = -1 + 4 * t       # x from -1 to 3
    grad_norm_sq = 4 * (1 - x)**2   # analytic along y = x^2

    for xi_val, color, label in [
        (1, "#1f77b4", r"$\xi = 1$"),
        (10, "#ff7f0e", r"$\xi = 10$"),
        (100, "#2ca02c", r"$\xi = 100$"),
    ]:
        scaling = 1.0 / (1 + xi_val * grad_norm_sq)
        ax.plot(t, scaling, color=color, label=label)

    ax.set_xlabel(r"$t$ (path parameter)")
    ax.set_ylabel(r"$1/(1 + \xi\|\nabla L\|^2)$")
    ax.set_title(r"Scaling factor along Rosenbrock valley $y = x^2$")
    ax.legend()
    ax.grid(True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "geodesic_scaling_factor.png"), dpi=DPI)
    plt.close(fig)
    print("  saved geodesic_scaling_factor.png")


# ---- Plot 3: Nonconvexity Map ---------------------------------------------

def plot_nonconvexity_map():
    r"""Side-by-side: lambda_min(H) vs lambda_min(Hess_g L) for Rosenbrock."""
    N = 300
    xs = np.linspace(-2, 2, N)
    ys = np.linspace(-2, 2, N)
    X, Y = np.meshgrid(xs, ys)

    lam_min_H = np.empty_like(X)
    lam_min_Hg = np.empty_like(X)
    xi = 10.0

    for i in range(N):
        for j in range(N):
            H = rosenbrock_hessian(X[i, j], Y[i, j])
            eig_min = np.linalg.eigvalsh(H)[0]
            lam_min_H[i, j] = eig_min

            grad = rosenbrock_grad(X[i, j], Y[i, j])
            scale = 1.0 / (1 + xi * np.dot(grad, grad))
            lam_min_Hg[i, j] = eig_min * scale

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    im1 = ax1.pcolormesh(X, Y, lam_min_H, cmap="RdBu_r", shading="auto",
                          vmin=-500, vmax=200, rasterized=True)
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")
    ax1.set_title(r"$\lambda_{\min}(H)$")
    ax1.set_aspect("equal")
    fig.colorbar(im1, ax=ax1, shrink=0.82)

    im2 = ax2.pcolormesh(X, Y, lam_min_Hg, cmap="RdBu_r", shading="auto",
                          vmin=-0.05, vmax=0.25, rasterized=True)
    ax2.set_xlabel(r"$x$")
    ax2.set_ylabel(r"$y$")
    ax2.set_title(r"$\lambda_{\min}(\mathrm{Hess}_g\, L)$, $\xi = 10$")
    ax2.set_aspect("equal")
    fig.colorbar(im2, ax=ax2, shrink=0.82)

    fig.suptitle(r"Rosenbrock --- minimum Hessian eigenvalue", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT_DIR, "geodesic_nonconvexity_map.png"),
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  saved geodesic_nonconvexity_map.png")


# ---- Plot 4: Eigenvalue Min Along Path ------------------------------------

def plot_eigenvalue_min():
    r"""lambda_min(H) and lambda_min(Hess_g) along a semicircular arc that
    passes through non-convex regions of the Rosenbrock landscape."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    t = np.linspace(0, 1, 2000)
    # Semicircular arc centred at (0, 0), radius 1.5
    # from (1.5, 0) through (0, 1.5) to (-1.5, 0)
    theta = np.pi * t
    x = 1.5 * np.cos(theta)
    y = 1.5 * np.sin(theta)

    lam_min_H = np.empty_like(t)
    grad_norm_sq = np.empty_like(t)
    for k in range(len(t)):
        H = rosenbrock_hessian(x[k], y[k])
        lam_min_H[k] = np.linalg.eigvalsh(H)[0]
        g = rosenbrock_grad(x[k], y[k])
        grad_norm_sq[k] = np.dot(g, g)

    ax.plot(t, lam_min_H, color="#1f77b4", linewidth=2.0,
            label=r"$\lambda_{\min}(H)$")

    for xi_val, color, label in [
        (1, "#ff7f0e", r"$\xi = 1$"),
        (10, "#2ca02c", r"$\xi = 10$"),
        (100, "#d62728", r"$\xi = 100$"),
    ]:
        scale = 1.0 / (1 + xi_val * grad_norm_sq)
        ax.plot(t, lam_min_H * scale, color=color, label=label)

    ax.axhline(0, color="0.5", linewidth=0.8, linestyle=":")
    ax.set_xlabel(r"$t$ (path parameter)")
    ax.set_ylabel(r"Minimum eigenvalue")
    ax.set_title(r"$\lambda_{\min}$ along semicircular arc (Rosenbrock)")
    ax.legend()
    ax.grid(True)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "geodesic_eigenvalue_min.png"), dpi=DPI)
    plt.close(fig)
    print("  saved geodesic_eigenvalue_min.png")


# ---- Plot 5: Saddle Eigenvalues -------------------------------------------

def plot_saddle_eigenvalues():
    r"""Eigenvalues of H and Hess_g for L = x^2 - y^2 on the unit circle."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    theta = np.linspace(0, 2 * np.pi, 500)
    x = np.cos(theta)
    y = np.sin(theta)

    # H = diag(2, -2) everywhere; eigenvalues +/- 2
    lam_max_H = np.full_like(theta, 2.0)
    lam_min_H = np.full_like(theta, -2.0)

    # grad L = (2x, -2y),  ||grad||^2 = 4(x^2 + y^2) = 4 on the unit circle
    xi = 10.0
    grad_norm_sq = 4 * (x**2 + y**2)  # = 4
    scale = 1.0 / (1 + xi * grad_norm_sq)
    lam_max_Hg = 2.0 * scale
    lam_min_Hg = -2.0 * scale

    ax.plot(theta, lam_max_H, "--", color="#1f77b4", linewidth=2.0,
            label=r"$\lambda_{\max}(H) = 2$")
    ax.plot(theta, lam_min_H, "--", color="#ff7f0e", linewidth=2.0,
            label=r"$\lambda_{\min}(H) = -2$")
    ax.plot(theta, lam_max_Hg, color="#2ca02c", linewidth=2.0,
            label=r"$\lambda_{\max}(\mathrm{Hess}_g)$")
    ax.plot(theta, lam_min_Hg, color="#d62728", linewidth=2.0,
            label=r"$\lambda_{\min}(\mathrm{Hess}_g)$")

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(r"Saddle $L = x^2 - y^2$: eigenvalues on unit circle ($\xi = 10$)")
    ax.legend(loc="center right")
    ax.grid(True)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "geodesic_saddle_eigenvalues.png"), dpi=DPI)
    plt.close(fig)
    print("  saved geodesic_saddle_eigenvalues.png")


# ---- Main ------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating geodesic convexity analysis plots ...")
    plot_scaling_factor()
    plot_nonconvexity_map()
    plot_eigenvalue_min()
    plot_saddle_eigenvalues()
    print("Done.")
