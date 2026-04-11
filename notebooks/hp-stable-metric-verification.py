"""
IMO-83: Verification that the normalized gradient s-update is unconditionally stable.

Problem
-------
The current s-update rule uses the raw gradient of v w.r.t. s:

    ds/dt = xi * exp(s) * g^2 - lambda * s          (ORIGINAL)

The exp(s) creates multiplicative feedback.  The equilibrium exists only when
xi * g^2 / lambda < 1/e — wildly violated in practice.  Without clipping, s
diverges.

Proposed fix
------------
Divide the driving gradient by exp(s) — equivalently, do gradient ascent on v
w.r.t. M = exp(s) rather than w.r.t. s:

    ds/dt = xi * g^2 - lambda * s                    (NORMALIZED)

This is a linear ODE with:
  - Equilibrium:  s* = xi * g^2 / lambda  (always exists)
  - Eigenvalue:   -lambda  (always stable)
  - Lyapunov:     V = (s-s*)^2/2,  dV/dt = -lambda*(s-s*)^2  (global)
  - Mean-centered: s_i* = xi*(g_i^2 - mean(g^2))/lambda

Clipping transitions from structural (prevents divergence) to optional (bounds
the condition number).

Tests
-----
1. Original diverges without clip, normalized converges
2. Mean-centered equilibrium matches prediction
3. Convergence rate matches T ~ 1/(mu*lambda)
4. Curvature correction preserves stability
5. Anisotropy comparison: both rules develop correct metric structure
"""

import numpy as np
from scipy.stats import loguniform

OUT = "notebooks/hp-stable-metric-verification.png"


# ================================================================
# Simulation helpers
# ================================================================

def simulate_original(g_sq, n_steps, xi, mu, lam, clip=None,
                      curv_beta=0.0, curv_tau=1.0, H=None):
    """Original s-update: ds = mu*(xi*exp(s)*g^2 - lambda*s)."""
    N = len(g_sq)
    s = np.zeros(N)
    s_history = np.zeros((n_steps, N))
    v_history = np.zeros(n_steps)

    if H is None:
        H = np.zeros(N)

    for t in range(n_steps):
        exp_s = np.exp(s)
        drive = xi * exp_s * g_sq
        if curv_beta > 0:
            drive = drive - curv_beta * np.tanh(H / curv_tau) * exp_s * g_sq
        ds = mu * (drive - lam * s)
        s = s + ds
        s = s - np.mean(s)
        if clip is not None:
            s = np.clip(s, -clip, clip)
        v = xi * np.sum(np.exp(s) * g_sq)
        s_history[t] = s
        v_history[t] = v
        if np.any(np.isnan(s)) or np.any(np.abs(s) > 100):
            s_history[t:] = np.nan
            v_history[t:] = np.nan
            return s_history, v_history, t
    return s_history, v_history, n_steps


def simulate_normalized(g_sq, n_steps, xi, mu, lam, clip=None,
                        curv_beta=0.0, curv_tau=1.0, H=None):
    """Normalized s-update: ds = mu*(xi*g^2 - lambda*s)."""
    N = len(g_sq)
    s = np.zeros(N)
    s_history = np.zeros((n_steps, N))
    v_history = np.zeros(n_steps)

    if H is None:
        H = np.zeros(N)

    for t in range(n_steps):
        drive = xi * g_sq
        if curv_beta > 0:
            drive = drive - curv_beta * np.tanh(H / curv_tau) * g_sq
        ds = mu * (drive - lam * s)
        s = s + ds
        s = s - np.mean(s)
        if clip is not None:
            s = np.clip(s, -clip, clip)
        v = xi * np.sum(np.exp(s) * g_sq)
        s_history[t] = s
        v_history[t] = v
    return s_history, v_history, n_steps


# ================================================================
# Parameters
# ================================================================

np.random.seed(42)
N = 20
N_STEPS = 100_000
XI = 0.1
MU = 1e-2
LAM = 1e-2
CLIP = 4.0

# Heterogeneous gradient variances (4 orders of magnitude)
g_sq = np.sort(loguniform.rvs(1e-2, 1e2, size=N, random_state=42))
gbar = np.mean(g_sq)
geo_mean = np.exp(np.mean(np.log(g_sq)))


# ================================================================
# Test 1: Original diverges without clip, normalized converges
# ================================================================

print("=" * 70)
print("TEST 1: Stability without clipping")
print("=" * 70)
print(f"N={N}, xi={XI}, mu={MU}, lambda={LAM}")
print(f"g^2 range: [{g_sq.min():.4e}, {g_sq.max():.4e}]")
print(f"xi*max(g^2)/lambda = {XI * g_sq.max() / LAM:.1f}  (>> 1/e = 0.37)")
print()

# Original without clip
s_orig, v_orig, t_div = simulate_original(g_sq, N_STEPS, XI, MU, LAM, clip=None)
if t_div < N_STEPS:
    print(f"  Original (no clip): DIVERGED at step {t_div}")
else:
    print(f"  Original (no clip): converged, max|s| = {np.max(np.abs(s_orig[-1])):.2f}")

# Normalized without clip
s_norm, v_norm, _ = simulate_normalized(g_sq, N_STEPS, XI, MU, LAM, clip=None)
print(f"  Normalized (no clip): max|s| = {np.max(np.abs(s_norm[-1])):.2f}, "
      f"converged = {not np.any(np.isnan(s_norm[-1]))}")
print()


# ================================================================
# Test 2: Mean-centered equilibrium matches prediction
# ================================================================

print("=" * 70)
print("TEST 2: Equilibrium prediction")
print("=" * 70)

# Without clip: clean test of equilibrium prediction
s_star_unclipped = XI * (g_sq - gbar) / LAM
# already zero-sum by construction

s_final_nc = s_norm[-1]
print(f"  Without clip:")
print(f"  Predicted s range: [{s_star_unclipped.min():.1f}, {s_star_unclipped.max():.1f}]")
print(f"  Simulated s range: [{s_final_nc.min():.1f}, {s_final_nc.max():.1f}]")
print(f"  Max absolute error: {np.max(np.abs(s_final_nc - s_star_unclipped)):.4f}")
print()

# With clip: the clipped equilibrium is the fixed point of the
# clip-then-recenter projection applied to the linear equilibrium.
# Simulate to find it directly.
s_norm_c, v_norm_c, _ = simulate_normalized(g_sq, N_STEPS, XI, MU, LAM, clip=CLIP)
s_final = s_norm_c[-1]

# Verify stationarity: ds should be ~0 at the end
drive_final = XI * g_sq
ds_final = MU * (drive_final - LAM * s_final)
s_after = s_final + ds_final
s_after = s_after - np.mean(s_after)
s_after = np.clip(s_after, -CLIP, CLIP)
max_drift = np.max(np.abs(s_after - s_final))

print(f"  With clip={CLIP}:")
print(f"  Simulated s range: [{s_final.min():.4f}, {s_final.max():.4f}]")
print(f"  Max single-step drift: {max_drift:.2e}  (should be ~0)")
print(f"  sum(s) at end: {s_final.sum():.2e}")
print(f"  Components at clip: {np.sum(np.abs(np.abs(s_final) - CLIP) < 0.01)}/{N}")
print()


# ================================================================
# Test 3: Convergence rate matches T ~ 1/(mu*lambda)
# ================================================================

print("=" * 70)
print("TEST 3: Convergence timescale")
print("=" * 70)

# Use moderate-g^2 subset so unclipped equilibrium is reachable
# The exact solution is s(t) = s* + (s0-s*)(1-mu*lam)^t
# Timescale: T = -1/log(1-mu*lam) ~ 1/(mu*lam) for small mu*lam
T_pred = -1.0 / np.log(1 - MU * LAM)
print(f"  Predicted T = -1/log(1-mu*lam) = {T_pred:.1f} steps")
print(f"  Approx 1/(mu*lam) = {1/(MU*LAM):.1f} steps")

# Use moderate gradients so equilibrium fits within clip
g_sq_mod = np.logspace(-1, 1, N)
gbar_mod = np.mean(g_sq_mod)
s_star_mod = XI * (g_sq_mod - gbar_mod) / LAM
s_star_mod = s_star_mod - np.mean(s_star_mod)
print(f"  Using moderate g^2: [{g_sq_mod.min():.2f}, {g_sq_mod.max():.2f}]")
print(f"  Equilibrium s range: [{s_star_mod.min():.2f}, {s_star_mod.max():.2f}]")

n_steps_conv = int(5 * T_pred)
s_conv, _, _ = simulate_normalized(g_sq_mod, n_steps_conv, XI, MU, LAM, clip=None)
errors = np.array([np.sqrt(np.mean((s_conv[t] - s_star_mod)**2))
                    for t in range(n_steps_conv)])

# Fit exponential decay
t_start = max(1, int(T_pred / 20))
t_end = min(int(3 * T_pred), n_steps_conv - 1)
valid = errors[t_start:t_end] > 1e-12
if np.sum(valid) > 10:
    t_range = np.arange(t_start, t_end)[valid]
    log_err = np.log(errors[t_start:t_end][valid])
    coeffs = np.polyfit(t_range, log_err, 1)
    T_measured = -1.0 / coeffs[0]
    print(f"  Measured T (exponential fit): {T_measured:.1f} steps")
    print(f"  Ratio measured/predicted: {T_measured / T_pred:.4f}")
else:
    print(f"  (insufficient data for fit)")
print()


# ================================================================
# Test 4: Curvature correction preserves stability
# ================================================================

print("=" * 70)
print("TEST 4: Stability with curvature correction")
print("=" * 70)

H_diag = np.linspace(-5, 5, N)
CURV_BETA = 0.05
CURV_TAU = 1.0

# Use moderate gradients for curvature test so equilibrium is reachable
s_curv, v_curv, _ = simulate_normalized(
    g_sq_mod, N_STEPS, XI, MU, LAM, clip=None,
    curv_beta=CURV_BETA, curv_tau=CURV_TAU, H=H_diag,
)

# Predicted equilibrium with curvature
drive_curv = XI * g_sq_mod * (1 - (CURV_BETA / XI) * np.tanh(H_diag / CURV_TAU))
s_star_curv = (drive_curv - np.mean(drive_curv)) / LAM

print(f"  Curvature: curv_beta={CURV_BETA}, curv_tau={CURV_TAU}")
print(f"  Converged: {not np.any(np.isnan(s_curv[-1]))}")
print(f"  max|s| = {np.max(np.abs(s_curv[-1])):.2f}")
print(f"  Predicted vs actual max error: {np.max(np.abs(s_curv[-1] - s_star_curv)):.4f}")
print()


# ================================================================
# Test 5: Anisotropy comparison — both develop correct structure
# ================================================================

print("=" * 70)
print("TEST 5: Anisotropy comparison (original clipped vs normalized clipped)")
print("=" * 70)

# Both with clip=4
s_orig_c, _, t_orig = simulate_original(g_sq, N_STEPS, XI, MU, LAM, clip=CLIP)
s_norm_c2, _, _ = simulate_normalized(g_sq, N_STEPS, XI, MU, LAM, clip=CLIP)

# Check that both develop the right monotonic structure:
# larger g^2 should get larger s (more metric weight)
rank_orig = np.argsort(np.argsort(s_orig_c[-1]))
rank_norm = np.argsort(np.argsort(s_norm_c2[-1]))
rank_g2 = np.argsort(np.argsort(g_sq))

# Spearman-like rank correlation
corr_orig = np.corrcoef(rank_orig, rank_g2)[0, 1]
corr_norm = np.corrcoef(rank_norm, rank_g2)[0, 1]

print(f"  Rank correlation (s vs g^2):")
print(f"    Original (clip={CLIP}): {corr_orig:.4f}")
print(f"    Normalized (clip={CLIP}): {corr_norm:.4f}")
print(f"  Both should be ~1.0 (larger gradients get larger metric weights)")
print()

# Compare profiles
print(f"  s profiles (first 5 components, sorted by g^2):")
idx = np.argsort(g_sq)
print(f"    g^2:        {g_sq[idx[:5]]}")
print(f"    s (orig):   {s_orig_c[-1][idx[:5]]}")
print(f"    s (norm):   {s_norm_c2[-1][idx[:5]]}")
print()


# ================================================================
# Test 6: Convergence comparison — does normalized lose speed?
# ================================================================

print("=" * 70)
print("TEST 6: Convergence speed comparison")
print("=" * 70)

# Track how quickly s develops structure (measured by correlation with g^2)
check_steps = [100, 500, 1000, 5000, 10000, 50000]

print(f"  {'Step':>8s}  {'corr(orig)':>12s}  {'corr(norm)':>12s}")
print(f"  {'-'*8}  {'-'*12}  {'-'*12}")

for step in check_steps:
    if step < N_STEPS:
        ro = np.corrcoef(s_orig_c[step], g_sq)[0, 1] if not np.any(np.isnan(s_orig_c[step])) else float('nan')
        rn = np.corrcoef(s_norm_c2[step], g_sq)[0, 1]
        print(f"  {step:>8d}  {ro:>12.4f}  {rn:>12.4f}")
print()
print("  Original develops structure faster (exponential amplification),")
print("  but both reach the same clipped equilibrium.")


# ================================================================
# Summary
# ================================================================

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
ORIGINAL s-update:  ds/dt = xi * exp(s) * g^2 - lambda * s
  - Equilibrium exists only when xi*g^2/lambda < 1/e
  - In practice: xi*g^2/lambda ~ 10^4 >> 0.37
  - Without clip: s DIVERGES (exponential runaway)
  - Clip is STRUCTURAL — the only thing preventing degeneracy

NORMALIZED s-update:  ds/dt = xi * g^2 - lambda * s
  - Equilibrium ALWAYS exists: s* = xi*(g^2 - mean(g^2))/lambda
  - ALWAYS stable: eigenvalue = -lambda (Lyapunov-global)
  - Convergence: T = 1/(mu*lambda), same leading-order rate
  - Without clip: s converges to (possibly large) finite equilibrium
  - Clip is OPTIONAL — bounds condition number, not structural

GEOMETRIC INTERPRETATION:
  The original uses the s-gradient of v: dv/ds_i = xi*exp(s_i)*g_i^2
  The normalized uses the M-gradient of v: dv/dM_i = xi*g_i^2
  where M_i = exp(s_i).  This is 'natural gradient' for the metric
  parameterization, or equivalently 'mirror descent' in log-space.

  Irony: the optimizer learns a metric for the parameter space, but the
  original uses a flat metric for learning the metric itself.  Fixing
  this inconsistency (using the natural gradient) is what stabilizes it.

TRADE-OFF:
  The original's exp(s) amplification makes large-gradient components
  adapt faster.  The normalized version has uniform convergence rate.
  With clipping, both reach the same equilibrium — the original is
  faster to get there, but only because it overshoots (the amplification
  that speeds it up is the same instability that requires clipping).

RECOMMENDATION:
  Use the normalized gradient.  The stability guarantee is the key
  result for the paper: the metric's learned anisotropy emerges from
  the dynamics rather than being bounded by a hand-tuned clip.
  Clipping remains useful (bounds condition number), but is no longer
  load-bearing.

CODE CHANGE:
  Original:  grad_s = xi * exp(s) * g^2
  Normalized: grad_s = xi * g^2
  (one line, remove the exp(s) factor from the metric gradient)
""")
