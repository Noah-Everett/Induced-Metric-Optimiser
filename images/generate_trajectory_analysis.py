"""Trajectory-level geodesic convexity analysis: numerical experiments.

Validates claims from notes/trajectory-geodesic-convexity.md:
  1. Critical-point obstruction (Hess_g = H at nabla L = 0)
  2. Basin enlargement: geodesic convexity basin vs Euclidean convexity basin
  3. Smoothness of optimal sigma along a trajectory
  4. Stability of coupled (theta, s) dynamics
  5. Correction composition over training steps

Generates:
  trajectory_critical_point_obstruction.png
  trajectory_basin_enlargement.png
  trajectory_optimal_sigma_smoothness.png
  trajectory_coupled_stability.png
  trajectory_correction_composition.png
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Style matching other scripts in the project
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
FIGSIZE_WIDE = (10, 4.2)
DPI = 200


# ---- Test functions --------------------------------------------------------

def saddle_grad(x, y):
    return np.array([2 * x, -2 * y])

def saddle_hessian(x, y):
    return np.array([[2.0, 0.0], [0.0, -2.0]])

def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_grad(x, y):
    dLdx = -2 * (1 - x) + 200 * (y - x**2) * (-2 * x)
    dLdy = 200 * (y - x**2)
    return np.array([dLdx, dLdy])

def rosenbrock_hessian(x, y):
    return np.array([
        [2 - 400 * (y - x**2) + 800 * x**2, -400 * x],
        [-400 * x, 200],
    ])

def quartic_well(x, y):
    """x^4 + y^4 - 2x^2 - 2y^2: four minima, saddles in between."""
    return x**4 + y**4 - 2*x**2 - 2*y**2

def quartic_well_grad(x, y):
    return np.array([4*x**3 - 4*x, 4*y**3 - 4*y])

def quartic_well_hessian(x, y):
    return np.array([[12*x**2 - 4, 0], [0, 12*y**2 - 4]])


# ---- Correction computation ------------------------------------------------

def christoffel_correction_diag(sigma, ell, s=None):
    """Curvature correction C_ij for diagonal metric gamma = diag(exp(s)).

    sigma: 2x2 Jacobian ds_i/dtheta_k
    ell: gradient (2,)
    s: metric log-scales (2,), default zeros (identity)

    Returns C as a 2x2 matrix.
    """
    if s is None:
        s = np.zeros(2)

    N = 2
    gamma_inv = np.diag(np.exp(-s))
    gamma = np.diag(np.exp(s))

    # Christoffel symbols for diagonal metric
    # Gamma^k_ij = (1/2) gamma^{kk} (d_i gamma_{jk} + d_j gamma_{ik} - d_k gamma_{ij})
    # For diagonal gamma_{mm} = exp(s_m), d_i gamma_{mm} = sigma_{m,i} * exp(s_m)
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            val = 0.0
            for k in range(N):
                # Christoffel bracket
                # d_i g_{jk} + d_j g_{ik} - d_k g_{ij}
                # g_{mm} = exp(s_m), d_i g_{mm} = sigma_{m,i} * exp(s_m)
                # g_{mn} = 0 for m != n (diagonal)
                d_i_g_jk = sigma[j, i] * np.exp(s[j]) if j == k else 0.0
                d_j_g_ik = sigma[i, j] * np.exp(s[i]) if i == k else 0.0
                d_k_g_ij = sigma[i, k] * np.exp(s[i]) if i == j else 0.0

                bracket = d_i_g_jk + d_j_g_ik - d_k_g_ij
                gamma_kk_inv = np.exp(-s[k])
                Gamma_k_ij = 0.5 * gamma_kk_inv * bracket
                val += Gamma_k_ij * ell[k]
            C[i, j] = val
    return C


def riemannian_hessian_diag(H, ell, sigma, s=None, xi=1.0):
    """Compute Riemannian Hessian for diagonal learnable metric."""
    if s is None:
        s = np.zeros(2)
    C = christoffel_correction_diag(sigma, ell, s)
    gamma_inv = np.diag(np.exp(-s))
    ell_gamma_sq = ell @ gamma_inv @ ell
    denom = 1.0 + xi * ell_gamma_sq
    return (H - C) / denom


def optimal_sigma(H, ell, budget=5.0, xi=1.0, s=None):
    """Find sigma that maximizes lambda_min(Hess_g L).

    The problem is concave (max of min-eigenvalue of affine matrix function),
    so a single-start solve suffices for the global optimum.
    """
    if s is None:
        s = np.zeros(2)

    # Fast path: if gradient is near-zero, correction is ~0 regardless of sigma
    if np.linalg.norm(ell) < 1e-10:
        return np.zeros((2, 2)), np.linalg.eigvalsh(H)[0]

    def neg_lam_min(sigma_flat):
        sigma = sigma_flat.reshape(2, 2)
        Hg = riemannian_hessian_diag(H, ell, sigma, s, xi)
        return -np.linalg.eigvalsh(Hg)[0]

    def constraint_norm(sigma_flat):
        return budget - np.linalg.norm(sigma_flat)

    # Warm start: anti-correlation heuristic (shrink positive curvature,
    # grow negative curvature directions)
    H_diag = np.diag(H)
    x0 = np.zeros(4)
    if np.any(H_diag != 0):
        target = -np.sign(H_diag) * budget / (2 * np.sqrt(2))
        x0[0], x0[3] = target[0], target[1]  # sigma_11, sigma_22

    res = minimize(neg_lam_min, x0, method="SLSQP",
                   constraints={"type": "ineq", "fun": constraint_norm},
                   options={"maxiter": 300, "ftol": 1e-10})
    return res.x.reshape(2, 2), -res.fun


# ---- Plot 1: Critical-point obstruction ------------------------------------

def plot_critical_point_obstruction():
    """Show that Hess_g L = H at critical points, regardless of metric."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # L = x^2 - y^2: approach the saddle at origin along y=x
    distances = np.linspace(0.01, 2.0, 60)

    for budget, color, label in [
        (0, "#1f77b4", r"$R = 0$ (no correction)"),
        (2, "#ff7f0e", r"$R = 2$"),
        (5, "#2ca02c", r"$R = 5$"),
        (10, "#d62728", r"$R = 10$"),
    ]:
        lam_mins = []
        for d in distances:
            x, y = d, d  # approach along diagonal
            H = saddle_hessian(x, y)
            ell = saddle_grad(x, y)
            if budget == 0:
                sigma = np.zeros((2, 2))
                lam_min = np.linalg.eigvalsh(
                    riemannian_hessian_diag(H, ell, sigma))[0]
            else:
                _, lam_min = optimal_sigma(H, ell, budget=budget)
            lam_mins.append(lam_min)
        ax1.plot(distances, lam_mins, color=color, label=label)

    ax1.axhline(0, color="0.5", linewidth=0.8, linestyle=":")
    ax1.set_xlabel(r"Distance from saddle $\|\theta\|$")
    ax1.set_ylabel(r"$\lambda_{\min}(\mathrm{Hess}_g\, L)$")
    ax1.set_title(r"$L = x^2 - y^2$: approach along $y = x$")
    ax1.legend(fontsize=8)
    ax1.grid(True)

    # Right panel: ratio Hess_g / H as function of distance
    for budget, color, label in [
        (2, "#ff7f0e", r"$R = 2$"),
        (5, "#2ca02c", r"$R = 5$"),
        (10, "#d62728", r"$R = 10$"),
    ]:
        correction_power = []
        for d in distances:
            x, y = d, d
            H = saddle_hessian(x, y)
            ell = saddle_grad(x, y)
            _, lam_min_g = optimal_sigma(H, ell, budget=budget)
            lam_min_H = np.linalg.eigvalsh(H)[0]
            # How much of the negative eigenvalue is corrected
            correction_power.append(
                (lam_min_g - lam_min_H) / abs(lam_min_H)
                if abs(lam_min_H) > 1e-10 else 0
            )
        ax2.plot(distances, correction_power, color=color, label=label)

    ax2.axhline(0, color="0.5", linewidth=0.8, linestyle=":")
    ax2.axhline(1, color="0.5", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.set_xlabel(r"Distance from saddle $\|\theta\|$")
    ax2.set_ylabel("Fractional correction")
    ax2.set_title("Correction power vs.\ distance from saddle")
    ax2.legend(fontsize=8)
    ax2.grid(True)

    fig.suptitle(
        r"Theorem 1: correction vanishes at critical points ($\nabla L = 0$)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(
        os.path.join(OUT_DIR, "trajectory_critical_point_obstruction.png"),
        dpi=DPI, bbox_inches="tight",
    )
    plt.close(fig)
    print("  saved trajectory_critical_point_obstruction.png")


# ---- Plot 2: Basin enlargement ---------------------------------------------

def plot_basin_enlargement():
    """Compare Euclidean convexity basin vs geodesic convexity basin."""
    N = 40  # coarse grid since we're optimizing sigma at each point
    xs = np.linspace(-1.8, 1.8, N)
    ys = np.linspace(-1.8, 1.8, N)
    X, Y = np.meshgrid(xs, ys)

    lam_min_H = np.empty_like(X)
    lam_min_Hg_R2 = np.empty_like(X)
    lam_min_Hg_R5 = np.empty_like(X)

    for i in range(N):
        for j in range(N):
            H = quartic_well_hessian(X[i, j], Y[i, j])
            ell = quartic_well_grad(X[i, j], Y[i, j])
            lam_min_H[i, j] = np.linalg.eigvalsh(H)[0]

            _, lm2 = optimal_sigma(H, ell, budget=2.0)
            lam_min_Hg_R2[i, j] = lm2

            _, lm5 = optimal_sigma(H, ell, budget=5.0)
            lam_min_Hg_R5[i, j] = lm5

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2))

    # Binary convexity maps: blue = convex, red = non-convex
    for ax, data, title in [
        (ax1, lam_min_H, r"Euclidean ($R = 0$)"),
        (ax2, lam_min_Hg_R2, r"Geodesic ($R = 2$)"),
        (ax3, lam_min_Hg_R5, r"Geodesic ($R = 5$)"),
    ]:
        convex = (data > 0).astype(float)
        ax.contourf(X, Y, convex, levels=[-0.5, 0.5, 1.5],
                    colors=["#ef9a9a", "#a5d6a7"], alpha=0.7)
        ax.contour(X, Y, data, levels=[0], colors="k", linewidths=1.5)
        # Mark minima at (+/-1, +/-1)
        for mx, my in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            ax.plot(mx, my, "k*", markersize=8)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_title(title)
        ax.set_aspect("equal")

    fig.suptitle(
        r"Basin enlargement: $L = x^4 + y^4 - 2x^2 - 2y^2$ "
        r"(green = geodesically convex)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(
        os.path.join(OUT_DIR, "trajectory_basin_enlargement.png"),
        dpi=DPI, bbox_inches="tight",
    )
    plt.close(fig)
    print("  saved trajectory_basin_enlargement.png")


# ---- Plot 3: Smoothness of optimal sigma -----------------------------------

def plot_optimal_sigma_smoothness():
    """Show that optimal sigma varies smoothly on the unit circle for L = x^2 - y^2."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    angles = np.linspace(0, 2 * np.pi, 80)
    sigma_entries = {(i, j): [] for i in range(2) for j in range(2)}
    lam_mins = []

    for angle in angles:
        x, y = np.cos(angle), np.sin(angle)
        H = saddle_hessian(x, y)
        ell = saddle_grad(x, y)
        sigma_opt, lam_min = optimal_sigma(H, ell, budget=5.0)
        for i in range(2):
            for j in range(2):
                sigma_entries[(i, j)].append(sigma_opt[i, j])
        lam_mins.append(lam_min)

    labels = {
        (0, 0): r"$\sigma_{11}$",
        (0, 1): r"$\sigma_{12}$",
        (1, 0): r"$\sigma_{21}$",
        (1, 1): r"$\sigma_{22}$",
    }
    colors = {
        (0, 0): "#1f77b4",
        (0, 1): "#ff7f0e",
        (1, 0): "#2ca02c",
        (1, 1): "#d62728",
    }

    for key in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        ax1.plot(angles, sigma_entries[key], color=colors[key],
                 label=labels[key])

    ax1.set_xlabel(r"Angle $\phi$ on unit circle")
    ax1.set_ylabel(r"$\sigma^*_{ij}(\phi)$")
    ax1.set_title(r"Optimal $\sigma^*$ components")
    ax1.legend(fontsize=9)
    ax1.grid(True)
    ax1.set_xlim(0, 2 * np.pi)
    ax1.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax1.set_xticklabels(
        [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

    ax2.plot(angles, lam_mins, color="#2ca02c", linewidth=2)
    ax2.axhline(0, color="0.5", linewidth=0.8, linestyle=":")
    ax2.set_xlabel(r"Angle $\phi$ on unit circle")
    ax2.set_ylabel(r"$\lambda_{\min}(\mathrm{Hess}_g\, L)$")
    ax2.set_title(r"Achieved $\lambda_{\min}$ with optimal $\sigma^*$")
    ax2.grid(True)
    ax2.set_xlim(0, 2 * np.pi)
    ax2.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax2.set_xticklabels(
        [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

    fig.suptitle(
        r"Proposition 4: $\sigma^*$ varies smoothly "
        r"($L = x^2 - y^2$, $R = 5$, unit circle)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(
        os.path.join(OUT_DIR, "trajectory_optimal_sigma_smoothness.png"),
        dpi=DPI, bbox_inches="tight",
    )
    plt.close(fig)
    print("  saved trajectory_optimal_sigma_smoothness.png")


# ---- Plot 4: Coupled stability ---------------------------------------------

def plot_coupled_stability():
    """Simulate coupled (theta, s) dynamics on quartic well, show convergence."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    eta = 0.005     # parameter learning rate
    mu = 0.01       # metric learning rate
    xi = 1.0
    lam_s = 10.0
    beta = 0.5
    n_steps = 2000
    metric_clip = 3.0

    theta_init = np.array([0.1, 0.5])  # near saddle at origin
    s_init = np.zeros(2)

    # Simulate with proposed curvature-aware rule
    thetas_prop = [theta_init.copy()]
    ss_prop = [s_init.copy()]
    lam_mins_prop = []

    theta = theta_init.copy()
    s = s_init.copy()
    prev_ell = quartic_well_grad(*theta)

    for t in range(n_steps):
        ell = quartic_well_grad(*theta)
        H = quartic_well_hessian(*theta)

        # Secant Hessian estimate
        delta_theta = theta - thetas_prop[-1] if t > 0 else np.ones(2) * 1e-6
        delta_ell = ell - prev_ell if t > 0 else np.zeros(2)
        H_hat_diag = np.where(
            np.abs(delta_theta) > 1e-8,
            delta_ell / delta_theta,
            np.diag(H),
        )

        # Metric update (proposed rule)
        sign_H = np.tanh(H_hat_diag / 0.5)  # smooth sign
        drive = (xi - beta * sign_H) * np.exp(s) * ell**2
        s = s + mu * drive - mu * lam_s * s
        s = np.clip(s, -metric_clip, metric_clip)

        # Parameter update
        gamma_inv = np.diag(np.exp(-s))
        ell_gamma_sq = ell @ gamma_inv @ ell
        step = ell / (1 + xi * ell_gamma_sq)
        prev_ell = ell.copy()
        theta = theta - eta * step

        thetas_prop.append(theta.copy())
        ss_prop.append(s.copy())

        # Compute Riemannian Hessian lambda_min (approximate, using sigma from s changes)
        if t > 0:
            sigma_approx = np.outer(
                ss_prop[-1] - ss_prop[-2],
                1.0 / np.where(np.abs(delta_theta) > 1e-10, delta_theta, 1e-10),
            )
            Hg = riemannian_hessian_diag(H, ell, sigma_approx, s, xi)
            lam_mins_prop.append(np.linalg.eigvalsh(Hg)[0])
        else:
            lam_mins_prop.append(np.linalg.eigvalsh(H)[0])

    # Also simulate with current rule (no curvature correction)
    thetas_curr = [theta_init.copy()]
    ss_curr = [s_init.copy()]
    lam_mins_curr = []

    theta = theta_init.copy()
    s = s_init.copy()

    for t in range(n_steps):
        ell = quartic_well_grad(*theta)
        H = quartic_well_hessian(*theta)

        # Current rule (no sign correction)
        drive = xi * np.exp(s) * ell**2
        s = s + mu * drive - mu * lam_s * s
        s = np.clip(s, -metric_clip, metric_clip)

        gamma_inv = np.diag(np.exp(-s))
        ell_gamma_sq = ell @ gamma_inv @ ell
        step = ell / (1 + xi * ell_gamma_sq)
        theta = theta - eta * step

        thetas_curr.append(theta.copy())
        ss_curr.append(s.copy())
        lam_mins_curr.append(np.linalg.eigvalsh(H)[0])

    thetas_prop = np.array(thetas_prop)
    thetas_curr = np.array(thetas_curr)
    ss_prop = np.array(ss_prop)
    ss_curr = np.array(ss_curr)

    # Top-left: trajectories in (x, y) space
    ax = axes[0, 0]
    # Contour of quartic well
    xs = np.linspace(-1.8, 1.8, 100)
    ys = np.linspace(-1.8, 1.8, 100)
    Xg, Yg = np.meshgrid(xs, ys)
    Zg = quartic_well(Xg, Yg)
    ax.contour(Xg, Yg, Zg, levels=15, colors="0.7", linewidths=0.5)
    ax.plot(thetas_curr[:, 0], thetas_curr[:, 1], ".-", color="#1f77b4",
            markersize=1, linewidth=0.8, alpha=0.7, label="Current rule")
    ax.plot(thetas_prop[:, 0], thetas_prop[:, 1], ".-", color="#d62728",
            markersize=1, linewidth=0.8, alpha=0.7, label="Proposed rule")
    ax.plot(*theta_init, "ko", markersize=6)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title("Optimization trajectories")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    # Top-right: loss vs step
    ax = axes[0, 1]
    losses_curr = [quartic_well(*th) for th in thetas_curr]
    losses_prop = [quartic_well(*th) for th in thetas_prop]
    ax.plot(losses_curr, color="#1f77b4", label="Current rule")
    ax.plot(losses_prop, color="#d62728", label="Proposed rule")
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$L(\theta)$")
    ax.set_title("Loss convergence")
    ax.legend(fontsize=8)
    ax.grid(True)

    # Bottom-left: s values over time
    ax = axes[1, 0]
    ax.plot(ss_curr[:, 0], color="#1f77b4", linestyle="-",
            label=r"$s_1$ (current)")
    ax.plot(ss_curr[:, 1], color="#1f77b4", linestyle="--",
            label=r"$s_2$ (current)")
    ax.plot(ss_prop[:, 0], color="#d62728", linestyle="-",
            label=r"$s_1$ (proposed)")
    ax.plot(ss_prop[:, 1], color="#d62728", linestyle="--",
            label=r"$s_2$ (proposed)")
    ax.axhline(0, color="0.5", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$s_i$")
    ax.set_title("Metric parameters")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True)

    # Bottom-right: lambda_min of Riemannian Hessian
    ax = axes[1, 1]
    ax.plot(lam_mins_curr, color="#1f77b4", alpha=0.7, label="Current rule")
    ax.plot(lam_mins_prop, color="#d62728", alpha=0.7, label="Proposed rule")
    ax.axhline(0, color="0.5", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$\lambda_{\min}(H)$ or $\lambda_{\min}(\mathrm{Hess}_g)$")
    ax.set_title("Hessian min eigenvalue along trajectory")
    ax.legend(fontsize=8)
    ax.grid(True)

    fig.suptitle(
        r"Proposition 5: coupled $(\theta, s)$ stability on $x^4+y^4-2x^2-2y^2$",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(
        os.path.join(OUT_DIR, "trajectory_coupled_stability.png"),
        dpi=DPI, bbox_inches="tight",
    )
    plt.close(fig)
    print("  saved trajectory_coupled_stability.png")


# ---- Plot 5: Correction composition ----------------------------------------

def plot_correction_composition():
    """Show metric anti-correlation pattern and convergence for different beta.

    Demonstrates that the curvature-aware rule (beta > 0) produces
    direction-dependent s values (anti-correlation with curvature sign)
    while preserving convergence.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    eta = 0.01
    mu = 0.05
    xi = 1.0
    lam_s = 5.0
    n_steps = 1500
    metric_clip = 3.0

    theta_init = np.array([0.15, 0.6])

    results = {}
    for beta, color, label in [
        (0.0, "#1f77b4", r"$\beta = 0$ (current)"),
        (0.5, "#ff7f0e", r"$\beta = 0.5$"),
        (0.9, "#2ca02c", r"$\beta = 0.9$"),
    ]:
        theta = theta_init.copy()
        s = np.zeros(2)
        prev_theta = theta.copy()
        prev_ell = quartic_well_grad(*theta)

        losses = []
        s1_hist, s2_hist = [], []
        anisotropy = []  # s1 - s2: measures direction-dependent correction

        for t in range(n_steps):
            ell = quartic_well_grad(*theta)
            H = quartic_well_hessian(*theta)

            delta_theta = theta - prev_theta
            delta_ell = ell - prev_ell

            H_hat_diag = np.where(np.abs(delta_theta) > 1e-8,
                                  delta_ell / np.where(
                                      np.abs(delta_theta) > 1e-8,
                                      delta_theta, 1.0),
                                  np.diag(H))

            sign_H = np.tanh(H_hat_diag / 0.5)
            drive = (xi - beta * sign_H) * np.exp(s) * ell**2
            s = s + mu * drive - mu * lam_s * s
            s = np.clip(s, -metric_clip, metric_clip)

            losses.append(quartic_well(*theta))
            s1_hist.append(s[0])
            s2_hist.append(s[1])
            anisotropy.append(s[0] - s[1])

            prev_theta = theta.copy()
            prev_ell = ell.copy()

            gamma_inv = np.diag(np.exp(-s))
            ell_gamma_sq = ell @ gamma_inv @ ell
            step = ell / (1 + xi * ell_gamma_sq)
            theta = theta - eta * step

        results[beta] = {
            "losses": losses, "s1": s1_hist, "s2": s2_hist,
            "aniso": anisotropy, "color": color, "label": label,
        }

    # Panel 1: Loss convergence (all beta)
    ax = axes[0]
    for beta, r in results.items():
        ax.plot(r["losses"], color=r["color"], label=r["label"], alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$L(\theta)$")
    ax.set_title("Loss convergence")
    ax.legend(fontsize=8)
    ax.grid(True)

    # Panel 2: s1 and s2 for beta=0 vs beta=0.9 (show anti-correlation)
    ax = axes[1]
    for beta in [0.0, 0.9]:
        r = results[beta]
        ls = "-" if beta == 0.0 else "-"
        ax.plot(r["s1"], color=r["color"], linestyle="-",
                label=r["label"] + r" $s_1$", alpha=0.8)
        ax.plot(r["s2"], color=r["color"], linestyle="--",
                label=r["label"] + r" $s_2$", alpha=0.8)
    ax.axhline(0, color="0.5", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$s_i$")
    ax.set_title(r"Metric parameters: $s_1$ (solid), $s_2$ (dashed)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True)

    # Panel 3: Anisotropy s1 - s2 (measures direction-dependent correction)
    ax = axes[2]
    for beta, r in results.items():
        ax.plot(r["aniso"], color=r["color"], label=r["label"], alpha=0.8)
    ax.axhline(0, color="0.5", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$s_1 - s_2$")
    ax.set_title("Metric anisotropy (anti-correlation signature)")
    ax.legend(fontsize=8)
    ax.grid(True)

    fig.suptitle(
        r"Correction composition: curvature-aware rule drives metric anisotropy "
        r"($x^4+y^4-2x^2-2y^2$)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(
        os.path.join(OUT_DIR, "trajectory_correction_composition.png"),
        dpi=DPI, bbox_inches="tight",
    )
    plt.close(fig)
    print("  saved trajectory_correction_composition.png")


# ---- Main ------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    print("Generating trajectory-level geodesic convexity analysis plots ...")
    plot_critical_point_obstruction()
    plot_optimal_sigma_smoothness()
    plot_basin_enlargement()
    plot_coupled_stability()
    plot_correction_composition()
    print("Done.")
