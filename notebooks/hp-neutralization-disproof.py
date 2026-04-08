"""
IMO-59: Disproof that the denominator self-neutralises at metric equilibrium.

Claim tested
------------
"At metric equilibrium with mean-centering, v_hat -> 0 and therefore
r = 1/(1 + |v_hat|) -> 1.  xi only controls metric adaptation
asymptotically, not the effective learning rate."

Result: FALSE
-------------
Mean-centering introduces a Lagrange multiplier mu.  The true equilibrium is

    xi * exp(s_i) * g_i^2 = lam * s_i + mu     for all i
    sum(s_i) = 0

Summing the first equation:

    v = xi * sum[exp(s_i) * g_i^2] = lam * sum(s_i) + N * mu = N * mu

If mu = 0, then xi * exp(s_i) * g_i^2 = lam * s_i, which forces s_i > 0
for every i (LHS is strictly positive).  But N positive numbers cannot sum
to zero.  CONTRADICTION.  Therefore mu > 0 and v = N * mu > 0 always.

The original proof's error was using the *unconstrained* equilibrium
equation (xi*exp(s)*g^2 = lam*s) inside the sum.  The mean-centering
constraint changes the equilibrium via the Lagrange multiplier mu, and the
mu term is what was missed.

Quantitative result (weak-regularisation limit, lam << xi * G_geo):
    v ~ N * xi * GeometricMean(g^2)
    r ~ 1 / (1 + N * xi * GeoMean(g^2))

With clipping active, v can be much larger (clipped components contribute
xi * exp(clip) * g_i^2), making r even smaller.

Key finding: xi controls BOTH metric adaptation AND the denominator scaling
at ALL times, not just transiently.
"""

import numpy as np
from scipy.stats import loguniform
from scipy.optimize import fsolve


# ================================================================
# Simulation helpers
# ================================================================

def simulate(xi, metric_lr, metric_reg, g_sq, n_steps, clip,
             use_curv=False, curv_beta=0.5, curv_tau=1.0,
             beta_ema=0.999):
    """Simulate metric dynamics with mean-centering and track v.

    Parameters
    ----------
    xi : float
        Metric driving strength.
    metric_lr : float
        Metric learning rate (mu).
    metric_reg : float
        Metric regularisation (lambda).
    g_sq : ndarray
        Per-component gradient variance estimates.
    n_steps : int
        Number of simulation steps.
    clip : float
        Metric clip bound.
    use_curv : bool
        Whether to include a curvature correction term.
    curv_beta, curv_tau : float
        Curvature correction hyperparameters.
    beta_ema : float
        EMA decay for v_hat tracking.

    Returns
    -------
    v_history, v_hat_history, r_history, s_history
    """
    N = len(g_sq)
    s = np.zeros(N)
    v_ema = 0.0

    v_history = np.zeros(n_steps)
    v_hat_history = np.zeros(n_steps)
    r_history = np.zeros(n_steps)
    s_history = np.zeros((n_steps, N))

    # Fixed fake curvature (random per-component)
    H = np.random.randn(N) * 0.1 if use_curv else np.zeros(N)

    for t in range(n_steps):
        exp_s = np.exp(s)
        drive = xi * exp_s * g_sq

        if use_curv:
            curv_corr = curv_beta * np.tanh(H / curv_tau) * exp_s * g_sq
        else:
            curv_corr = 0.0

        ds = metric_lr * (drive - curv_corr) - metric_lr * metric_reg * s
        s = s + ds
        s = s - np.mean(s)
        s = np.clip(s, -clip, clip)

        v = xi * np.sum(np.exp(s) * g_sq)
        v_ema = beta_ema * v_ema + (1 - beta_ema) * v
        v_hat = v_ema / (1 - beta_ema ** (t + 1))
        r = 1.0 / (1.0 + np.abs(v_hat))

        v_history[t] = v
        v_hat_history[t] = v_hat
        r_history[t] = r
        s_history[t] = s.copy()

    return v_history, v_hat_history, r_history, s_history


def simulate_no_clip(xi, metric_lr, metric_reg, g_sq, n_steps):
    """No clipping, just mean-centering."""
    N = len(g_sq)
    s = np.zeros(N)
    v_history = np.zeros(n_steps)

    for t in range(n_steps):
        exp_s = np.exp(s)
        drive = xi * exp_s * g_sq
        ds = metric_lr * drive - metric_lr * metric_reg * s
        s = s + ds
        s = s - np.mean(s)
        v = xi * np.sum(np.exp(s) * g_sq)
        v_history[t] = v
        if np.any(np.isnan(s)) or np.any(np.abs(s) > 50):
            print(f"    Diverged at step {t}, max|s| = {np.max(np.abs(s)):.1f}")
            return v_history[:t + 1], s

    return v_history, s


def fixed_point_eqs(x, xi, lam, g_sq):
    """Fixed-point equations for mean-centered metric equilibrium.

    Variables: x = [s_0, ..., s_{N-1}, mu]

    Equations:
        xi * exp(s_i) * g_i^2 - lam * s_i - mu = 0    for i = 0..N-1
        sum(s_i) = 0
    """
    N = len(g_sq)
    s = x[:N]
    mu = x[N]
    eqs = np.zeros(N + 1)
    for i in range(N):
        eqs[i] = xi * np.exp(s[i]) * g_sq[i] - lam * s[i] - mu
    eqs[N] = np.sum(s)
    return eqs


# ================================================================
# Main verification
# ================================================================

if __name__ == "__main__":
    np.random.seed(42)

    # Problem setup
    N = 20
    N_STEPS = 100_000
    XI = 0.1
    METRIC_LR = 1e-3
    METRIC_REG = 1e-3
    BETA = 0.999
    CURV_BETA = 0.5
    CURV_TAU = 1.0

    # Heterogeneous gradient variances spanning 4 orders of magnitude
    g_sq = np.sort(loguniform.rvs(1e-2, 1e2, size=N, random_state=42))
    geo_mean = np.exp(np.mean(np.log(g_sq)))

    print("=" * 70)
    print("NUMERICAL VERIFICATION: v at metric equilibrium")
    print("=" * 70)
    print(f"N = {N}, xi = {XI}, metric_lr = {METRIC_LR}, "
          f"metric_reg = {METRIC_REG}")
    print(f"g^2 range: [{g_sq.min():.4e}, {g_sq.max():.4e}]")
    print(f"Geometric mean of g^2: {geo_mean:.4e}")
    print()

    # ------------------------------------------------------------------
    # Test 1: No curvature, clip=4.0
    # ------------------------------------------------------------------
    print("=" * 70)
    print("TEST 1: No curvature correction, clip=4.0")
    print("=" * 70)

    v_hist, v_hat_hist, r_hist, s_hist = simulate(
        XI, METRIC_LR, METRIC_REG, g_sq, N_STEPS, clip=4.0,
    )

    for step in [1000, 10000, 50000, N_STEPS]:
        label = f"  v at step {step}:"
        print(f"{label:<22s} {v_hist[step - 1]:.6f}")
    print(f"  v_hat at end:       {v_hat_hist[-1]:.6f}")
    print(f"  r at end:           {r_hist[-1]:.6f}")
    print(f"  s range at end:     [{s_hist[-1].min():.4f}, "
          f"{s_hist[-1].max():.4f}]")
    print(f"  sum(s) at end:      {s_hist[-1].sum():.2e}")
    is_zero = abs(v_hist[-1]) < 1e-3
    print(f"  Is v ~ 0? {'YES' if is_zero else 'NO, v = ' + f'{v_hist[-1]:.4f}'}")
    print()

    v_predicted = N * XI * geo_mean
    print(f"  Theoretical prediction (weak-reg limit): "
          f"v = N*xi*GeoMean(g^2) = {v_predicted:.6f}")
    print(f"  Ratio actual/predicted: {v_hist[-1] / v_predicted:.4f}")
    print()

    # ------------------------------------------------------------------
    # Test 2: No curvature, no clip
    # ------------------------------------------------------------------
    print("=" * 70)
    print("TEST 2: No curvature correction, clip=100.0 (effectively no clip)")
    print("=" * 70)

    v_hist2, _, r_hist2, s_hist2 = simulate(
        XI, METRIC_LR, METRIC_REG, g_sq, N_STEPS, clip=100.0,
    )

    print(f"  v at step {N_STEPS}: {v_hist2[-1]:.6f}")
    print(f"  r at end:           {r_hist2[-1]:.6f}")
    print(f"  s range at end:     [{s_hist2[-1].min():.4f}, "
          f"{s_hist2[-1].max():.4f}]")
    is_zero = abs(v_hist2[-1]) < 1e-3
    print(f"  Is v ~ 0? {'YES' if is_zero else 'NO, v = ' + f'{v_hist2[-1]:.4f}'}")
    print(f"  Predicted v = {v_predicted:.6f}, "
          f"ratio = {v_hist2[-1] / v_predicted:.4f}")
    print()

    # ------------------------------------------------------------------
    # Test 3-4: With curvature correction
    # ------------------------------------------------------------------
    for clip_val, label in [(4.0, "TEST 3"), (100.0, "TEST 4")]:
        clip_desc = f"clip={clip_val}" + (" (no clip)" if clip_val > 50 else "")
        print("=" * 70)
        print(f"{label}: With curvature correction, {clip_desc}")
        print("=" * 70)

        v_h, _, r_h, s_h = simulate(
            XI, METRIC_LR, METRIC_REG, g_sq, N_STEPS, clip=clip_val,
            use_curv=True, curv_beta=CURV_BETA, curv_tau=CURV_TAU,
        )

        print(f"  v at step {N_STEPS}: {v_h[-1]:.6f}")
        print(f"  r at end:           {r_h[-1]:.6f}")
        print(f"  s range at end:     [{s_h[-1].min():.4f}, "
              f"{s_h[-1].max():.4f}]")
        is_zero = abs(v_h[-1]) < 1e-3
        print(f"  Is v ~ 0? {'YES' if is_zero else 'NO, v = ' + f'{v_h[-1]:.4f}'}")
        print()

    # ------------------------------------------------------------------
    # Equilibrium analysis: verify the Lagrange multiplier
    # ------------------------------------------------------------------
    print("=" * 70)
    print("EQUILIBRIUM ANALYSIS (Test 1, final state)")
    print("=" * 70)

    s_eq = s_hist[-1]
    exp_s_eq = np.exp(s_eq)

    # At equilibrium: xi*exp(s_i)*g_i^2 - lam*s_i = mu (constant)
    residuals = XI * exp_s_eq * g_sq - METRIC_REG * s_eq

    print(f"  Residual xi*exp(s)*g^2 - lam*s for each component:")
    print(f"    mean (= mu): {residuals.mean():.6f}")
    print(f"    std:          {residuals.std():.6e}")
    print(f"  v = N*mu = {N * residuals.mean():.6f}")
    print(f"  v (direct) = {v_hist[-1]:.6f}")
    print()

    # ------------------------------------------------------------------
    # Part A: Exact fixed-point solutions via fsolve
    # ------------------------------------------------------------------
    print("=" * 70)
    print("EXACT FIXED-POINT SOLUTIONS (scipy fsolve, no clipping)")
    print("=" * 70)

    header = (f"{'xi':>8s} {'lam':>8s} | {'mu':>12s} {'v=N*mu':>12s} "
              f"{'v_pred':>12s} {'ratio':>8s} {'r':>8s} | "
              f"{'s_min':>8s} {'s_max':>8s}")
    print(header)
    print("-" * 100)

    for xi_val in [0.001, 0.01, 0.1]:
        for lam_val in [0.1, 0.01, 0.001, 0.0001]:
            s_init = -np.log(g_sq / geo_mean)
            s_init = s_init - np.mean(s_init)
            mu_init = xi_val * geo_mean
            x0 = np.append(s_init, mu_init)

            try:
                x_sol = fsolve(fixed_point_eqs, x0,
                               args=(xi_val, lam_val, g_sq),
                               full_output=False)
                s_sol = x_sol[:N]
                mu_sol = x_sol[N]
                v_sol = N * mu_sol
                v_pred = N * xi_val * geo_mean
                ratio = v_sol / v_pred if v_pred > 0 else float("inf")
                r_sol = 1.0 / (1.0 + abs(v_sol))

                resid = fixed_point_eqs(x_sol, xi_val, lam_val, g_sq)
                max_resid = np.max(np.abs(resid))

                if max_resid > 1e-6:
                    print(f"{xi_val:>8.3f} {lam_val:>8.4f} | "
                          f"{'FAILED':>12s} (residual = {max_resid:.2e})")
                else:
                    print(f"{xi_val:>8.3f} {lam_val:>8.4f} | "
                          f"{mu_sol:>12.6f} {v_sol:>12.4f} "
                          f"{v_pred:>12.4f} {ratio:>8.4f} "
                          f"{r_sol:>8.6f} | "
                          f"{s_sol.min():>8.3f} {s_sol.max():>8.3f}")
            except Exception as e:
                print(f"{xi_val:>8.3f} {lam_val:>8.4f} | ERROR: {e}")

    print()

    # ------------------------------------------------------------------
    # Parameter sweep: v_eq vs xi
    # ------------------------------------------------------------------
    print("=" * 70)
    print("PARAMETER SWEEP: v_eq vs xi")
    print("=" * 70)

    xi_values = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    header = (f"{'xi':>8s} | {'v_eq':>12s} | {'v_predicted':>12s} | "
              f"{'ratio':>8s} | {'r_eq':>8s}")
    print(header)
    print("-" * 60)

    for xi_val in xi_values:
        v_h, _, r_h, _ = simulate(
            xi_val, METRIC_LR, METRIC_REG, g_sq, N_STEPS, clip=4.0,
        )
        v_pred = N * xi_val * geo_mean
        ratio = v_h[-1] / v_pred if v_pred > 0 else float("inf")
        print(f"{xi_val:>8.3f} | {v_h[-1]:>12.6f} | {v_pred:>12.6f} | "
              f"{ratio:>8.4f} | {r_h[-1]:>8.4f}")

    print()

    # ------------------------------------------------------------------
    # Homogeneous gradients (simplest case)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("HOMOGENEOUS GRADIENTS (simplest possible case)")
    print("=" * 70)

    N_h = 10
    g_sq_h = np.ones(N_h)
    xi_h = 0.0001
    lam_h = 0.01

    print(f"N={N_h}, all g^2 = 1, xi={xi_h}, lam={lam_h}")

    s_h = np.zeros(N_h)
    for t in range(500_000):
        ds = xi_h * np.exp(s_h) * g_sq_h - lam_h * s_h
        s_h = s_h + 1e-3 * ds
        s_h = s_h - np.mean(s_h)

    v_h = xi_h * np.sum(np.exp(s_h) * g_sq_h)
    print(f"  By symmetry, all s_i = 0 at equilibrium.")
    print(f"  s values: {s_h}")
    print(f"  v = xi * N * exp(0) * 1 = xi * N = {xi_h * N_h}")
    print(f"  v (computed) = {v_h:.8f}")
    print(f"  r = 1/(1+v) = {1 / (1 + v_h):.6f}")
    print()
    print("Even in the simplest case (all gradients identical), v = xi*N != 0.")
    print("The driving force xi*exp(0)*G at s=0 is NOT balanced by lam*0 = 0;")
    print("it is balanced by the Lagrange multiplier mu = xi*G from "
          "mean-centering.")

    # ------------------------------------------------------------------
    # Convergence check
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("CONVERGENCE CHECK (500k steps)")
    print("=" * 70)

    v_long, _, r_long, _ = simulate(
        XI, METRIC_LR, METRIC_REG, g_sq, 500_000, clip=4.0,
    )

    windows = [
        ("Steps 1k-10k", v_long[1000:10000]),
        ("Steps 10k-50k", v_long[10000:50000]),
        ("Steps 50k-100k", v_long[50000:100000]),
        ("Steps 100k-200k", v_long[100000:200000]),
        ("Steps 200k-500k", v_long[200000:500000]),
    ]

    print(f"{'Window':>20s} | {'mean(v)':>12s} | {'std(v)':>12s}")
    print("-" * 50)
    for name, vals in windows:
        print(f"{name:>20s} | {vals.mean():>12.6f} | {vals.std():>12.2e}")

    print()
    print(f"  v at step 500k: {v_long[-1]:.6f}")
    print(f"  r at step 500k: {r_long[-1]:.6f}")
    print(f"  Change in v over last 300k steps: "
          f"{abs(v_long[-1] - v_long[200000]):.2e}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
CLAIM: At metric equilibrium with mean-centering, v -> 0 and r -> 1.

RESULT: The claim is FALSE.

ANALYTICAL PROOF:
  The fixed point of the mean-centered dynamics satisfies:
    xi * exp(s_i) * g_i^2 - lam * s_i = mu   (same constant for all i)
    sum(s_i) = 0

  Summing over i:
    v = xi * sum(exp(s_i) * g_i^2) = lam * sum(s_i) + N * mu = N * mu

  If mu = 0, then xi * exp(s_i) * g_i^2 = lam * s_i, which requires
  s_i > 0 for all i (since LHS > 0). But sum(s_i) = 0 with all s_i > 0
  is impossible.  CONTRADICTION.  Therefore mu != 0 and v != 0.

WHAT DETERMINES v:
  In the weak-regularisation limit (lam << xi * G_geo):
    v ~ N * xi * GeometricMean(g^2)

  The denominator r = 1/(1 + |v_hat|) converges to a value strictly
  less than 1.  xi controls BOTH the metric adaptation AND the
  denominator scaling.  The denominator does NOT self-neutralise.

ADDITIONAL COMPLICATION -- CLIPPING:
  When clipping is active, v can be MUCH larger than the unclipped
  prediction because clipped components contribute xi*exp(clip)*g_i^2.
  In the practical clipped regime, r can be very small (~0.001).
""")
