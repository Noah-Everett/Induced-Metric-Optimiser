"""
IMO-61: Numerical verification of the metric convergence timescale.

Claim tested
------------
"The metric convergence timescale is T ~ 1/(metric_lr * metric_reg),
i.e. T ~ 1/(mu * lam)."

Result: CONFIRMED with correction
----------------------------------
The exact e-folding time from linearised stability analysis is:

    T = -1 / ln|1 - mu*lam*(1 - s*_max)|

Series-expanding for small mu*lam:

    T = 1 / (mu*lam*(1 - s*_max))  -  1/2  +  O(mu*lam)

where s*_max is the largest equilibrium metric value, obtained from the
Lambert W solution s* = -W_0(-xi*g^2/lam).

Key findings
------------
1. T is proportional to 1/(mu*lam) to leading order.  CONFIRMED.

2. The correction factor 1/(1 - s*_max) depends on xi*max(g^2)/lam, which
   introduces an INDIRECT dependence on lam (not just the product).
   When lam >> xi*max(g^2), s* -> 0 and T -> 1/(mu*lam) exactly.

3. Stability boundary: mu*lam*(1 - s*_max) < 2, i.e.
       (mu*lam)_crit = 2 / (1 - s*_max)
   For s*=0: (mu*lam)_crit = 2.0.

4. Mean-centering removes the uniform mode (eigenvalue = 0) but does NOT
   change the convergence timescale for non-uniform modes.

5. Convergence step floors (lam=50, tol=1e-6, from s=0):
       10k steps:  mu*lam >= ~7e-4
       50k steps:  mu*lam >= ~2e-4
       100k steps: mu*lam >= ~1e-4
"""

import numpy as np


# ================================================================
# Core dynamics
# ================================================================

N = 20
GSQ = np.logspace(-2, 2, N)   # g_i^2 from 0.01 to 100
XI = 0.1
CLIP_VAL = 4.0


def step_fn(s, mu, lam):
    """One step of the metric dynamics: drive + regularise + center + clip."""
    A = XI * GSQ
    s_grad = A * np.exp(s)
    s_new = s + mu * s_grad - mu * lam * s
    s_new = s_new - np.mean(s_new)
    s_new = np.clip(s_new, -CLIP_VAL, CLIP_VAL)
    return s_new


def find_equilibrium(mu, lam, max_steps=2_000_000, tol=1e-14):
    """Run dynamics to convergence and return the equilibrium."""
    s = np.zeros(N)
    for t in range(max_steps):
        s_new = step_fn(s, mu, lam)
        if np.max(np.abs(s_new - s)) < tol:
            return s_new, t + 1, True
        s = s_new
    return s, max_steps, False


def measure_spectral_radius(mu, lam, s_eq, eps=1e-8):
    """Numerical Jacobian at s_eq, returning the spectral radius.

    The Jacobian of the discrete-time map Phi(s) at equilibrium determines
    the convergence rate.  One eigenvalue is ~0 (the uniform mode killed
    by mean-centering); the spectral radius of the remaining eigenvalues
    gives the per-step contraction factor.
    """
    J = np.zeros((N, N))
    f0 = step_fn(s_eq, mu, lam)
    for j in range(N):
        s_pert = s_eq.copy()
        s_pert[j] += eps
        f_pert = step_fn(s_pert, mu, lam)
        J[:, j] = (f_pert - f0) / eps
    eigs = np.linalg.eigvals(J)
    # Sort by magnitude; skip the smallest (the ~0 uniform eigenvalue)
    eigs_sorted = sorted(eigs, key=lambda x: abs(x))
    nonzero_eigs = eigs_sorted[1:]
    return max(abs(e) for e in nonzero_eigs)


def convergence_steps(mu, lam, max_steps=500_000, tol=1e-6):
    """Count steps from s=0 to convergence at the given tolerance."""
    s = np.zeros(N)
    for t in range(max_steps):
        s_new = step_fn(s, mu, lam)
        if np.max(np.abs(s_new - s)) < tol:
            return t + 1
        s = s_new
    return max_steps


# ================================================================
# Experiments
# ================================================================

if __name__ == "__main__":

    # ==============================================================
    # Experiment 1: Fix lam, vary mu => vary mu*lam
    # Verify T_efold ~ 1/(mu*lam) with a correction factor
    # ==============================================================
    print("=" * 90)
    print("EXPERIMENT 1: Fix lam=50, vary mu")
    print("All components reach interior equilibria (no clipping binds).")
    print("Verify T_efold is proportional to 1/(mu*lam).")
    print("=" * 90)

    lam_fixed = 50.0
    mu_vals = np.logspace(-5, -1, 20)

    header = ("{:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>12s}  "
              "{:>12s}  {:>10s}  {:>8s}")
    print(header.format("mu", "lam", "mu*lam", "spec_rad",
                         "T_efold", "1/(mu*lam)", "T_ratio", "n_clip"))
    print("-" * 95)

    exp1_data = []
    for mu in mu_vals:
        mulam = mu * lam_fixed
        s_eq, eq_steps, conv = find_equilibrium(mu, lam_fixed)
        n_clip = int(np.sum(np.abs(s_eq) > 3.99))

        if n_clip == 0:
            sr = measure_spectral_radius(mu, lam_fixed, s_eq)
            if 0 < sr < 1:
                T = -1.0 / np.log(sr)
                pred = 1.0 / mulam
                exp1_data.append((mulam, T, pred))
                print("{:10.2e}  {:10.1f}  {:10.4f}  {:10.8f}  "
                      "{:12.2f}  {:12.2f}  {:8.4f}  {:3d}/{:d}".format(
                          mu, lam_fixed, mulam, sr, T, pred,
                          T / pred, n_clip, N))
            else:
                print("{:10.2e}  {:10.1f}  {:10.4f}  {:10.8f}  "
                      "{:>12s}  {:12.2f}  {:>8s}  {:3d}/{:d}".format(
                          mu, lam_fixed, mulam, sr, "UNSTABLE",
                          1.0 / mulam, "---", n_clip, N))
        else:
            print("{:10.2e}  {:10.1f}  {:10.4f}  {:>10s}  "
                  "{:>12s}  {:12.2f}  {:>8s}  {:3d}/{:d}".format(
                      mu, lam_fixed, mulam, "clipped", "clipped",
                      1.0 / mulam, "---", n_clip, N))

    print()

    # ==============================================================
    # Experiment 2: Fix mu*lam, vary mu/lam ratio
    # Test whether the ratio affects convergence independently
    # ==============================================================
    print("=" * 100)
    print("EXPERIMENT 2: Ratio independence test")
    print("Fix mu*lam product, vary (mu, lam) pairs with lam >= 30")
    print("(interior equilibria guaranteed).")
    print("=" * 100)

    for mulam in [0.005, 0.01, 0.05, 0.1, 0.5]:
        print("\n--- mu*lam = {:.3f} ---".format(mulam))
        print("{:>10s}  {:>10s}  {:>12s}  {:>12s}  {:>12s}  "
              "{:>10s}  {:>10s}".format(
                  "mu", "lam", "mu/lam", "spec_rad", "T_efold",
                  "max_sEq", "T_ratio"))

        T_values = []
        for lam in [30, 50, 100, 200, 500]:
            mu = mulam / lam
            if mu > 0.1:
                continue

            s_eq, _, conv = find_equilibrium(mu, lam)
            n_clip = int(np.sum(np.abs(s_eq) > 3.99))
            if n_clip > 0:
                print("{:10.2e}  {:10.1f}  {:12.2e}  {:>12s}  {:>12s}  "
                      "{:>10s}  {:>10s}".format(
                          mu, lam, mu / lam, "CLIPPED", "---",
                          "---", "---"))
                continue

            sr = measure_spectral_radius(mu, lam, s_eq)
            if 0 < sr < 1:
                T = -1.0 / np.log(sr)
                pred = 1.0 / mulam
                max_s = np.max(s_eq)
                T_values.append(T)
                print("{:10.2e}  {:10.1f}  {:12.2e}  {:12.10f}  "
                      "{:12.4f}  {:10.6f}  {:10.4f}".format(
                          mu, lam, mu / lam, sr, T, max_s, T / pred))
            else:
                print("{:10.2e}  {:10.1f}  {:12.2e}  {:12.8f}  "
                      "{:>12s}  {:>10s}  {:>10s}".format(
                          mu, lam, mu / lam, sr, "UNSTABLE",
                          "---", "---"))

        if len(T_values) >= 2:
            spread = ((max(T_values) - min(T_values))
                      / min(T_values) * 100)
            print("  => T spread: {:.1f}% (min={:.2f}, max={:.2f})".format(
                spread, min(T_values), max(T_values)))
            print("  => Ratio matters through equilibrium s*: "
                  "T varies by {:.1f}%".format(spread))

    print()

    # ==============================================================
    # Experiment 3: Stability boundary
    # ==============================================================
    print("=" * 80)
    print("STABILITY BOUNDARY")
    print("=" * 80)

    print("\nFine scan of stability boundary (lam=50):")
    print("{:>10s}  {:>10s}  {:>12s}  {:>8s}".format(
        "mu*lam", "spec_rad", "T_efold", "stable?"))
    print("-" * 50)

    for mulam in np.arange(0.5, 2.5, 0.1):
        mu = mulam / lam_fixed
        s_eq, steps, conv = find_equilibrium(mu, lam_fixed)
        n_clip = int(np.sum(np.abs(s_eq) > 3.99))

        if n_clip > 0:
            print("{:10.2f}  {:>10s}  {:>12s}  CLIPPED".format(
                mulam, "---", "---"))
            continue

        sr = measure_spectral_radius(mu, lam_fixed, s_eq)
        if sr < 1:
            T = -1.0 / np.log(sr)
            print("{:10.2f}  {:10.6f}  {:12.4f}  YES".format(
                mulam, sr, T))
        else:
            print("{:10.2f}  {:10.6f}  {:>12s}  NO (unstable!)".format(
                mulam, sr, "diverges"))

    # Theoretical prediction
    mu_test = 0.0001
    s_eq_test, _, _ = find_equilibrium(mu_test, lam_fixed)
    max_sEq = np.max(s_eq_test)
    crit_mulam = 2.0 / (1.0 - max_sEq)
    print("\nPredicted critical mu*lam = 2/(1-s*_max) "
          "= 2/(1-{:.4f}) = {:.4f}".format(max_sEq, crit_mulam))

    # ==============================================================
    # Experiment 4: Proportionality check T ~ C/(mu*lam)
    # ==============================================================
    print()
    print("=" * 100)
    print("EXPERIMENT 4: Is T_efold proportional to 1/(mu*lam)?")
    print("Using lam=50 (all interior equilibria)")
    print("=" * 100)

    mu_vals_e4 = np.logspace(-5, -2, 30)

    data = []
    for mu in mu_vals_e4:
        mulam = mu * lam_fixed
        s_eq, _, conv = find_equilibrium(mu, lam_fixed)
        n_clip = int(np.sum(np.abs(s_eq) > 3.99))
        if n_clip > 0:
            continue
        sr = measure_spectral_radius(mu, lam_fixed, s_eq)
        if 0 < sr < 1:
            T = -1.0 / np.log(sr)
            pred = 1.0 / mulam
            data.append((mulam, T, pred))

    print("{:>10s}  {:>12s}  {:>12s}  {:>10s}".format(
        "mu*lam", "T_efold", "1/(mu*lam)", "T_ratio"))
    print("-" * 50)
    ratios = []
    for mulam, T, pred in data:
        r = T / pred
        ratios.append(r)
        print("{:10.4f}  {:12.4f}  {:12.4f}  {:10.4f}".format(
            mulam, T, pred, r))

    print()
    print("T_ratio = T_efold / [1/(mu*lam)]:")
    print("  Mean: {:.4f}".format(np.mean(ratios)))
    print("  Std:  {:.4f}".format(np.std(ratios)))
    print("  Min:  {:.4f}".format(np.min(ratios)))
    print("  Max:  {:.4f}".format(np.max(ratios)))
    print()
    print("CONCLUSION: T_efold = {:.4f} / (mu*lam) with {:.1f}% "
          "variation".format(np.mean(ratios),
                             np.std(ratios) / np.mean(ratios) * 100))

    # ==============================================================
    # Experiment 5: Convergence step thresholds
    # ==============================================================
    print()
    print("=" * 80)
    print("CONVERGENCE STEP THRESHOLDS")
    print("=" * 80)

    print("{:>10s}  {:>10s}  {:>12s}  {:>12s}".format(
        "mu*lam", "conv_steps", "1/(mu*lam)", "steps*mu*lam"))
    print("-" * 55)

    mu_scan = np.logspace(-5, -1.5, 30)
    multipliers = []
    for mu in mu_scan:
        mulam = mu * lam_fixed
        if mulam > 1.5:
            continue
        steps = convergence_steps(mu, lam_fixed)
        mult = steps * mulam
        multipliers.append(mult)
        print("{:10.5f}  {:10d}  {:12.2f}  {:12.4f}".format(
            mulam, steps, 1.0 / mulam, mult))

    print()
    print("steps * mu*lam (should be roughly constant if proportional):")
    print("  Mean: {:.2f}".format(np.mean(multipliers)))
    print("  Std:  {:.2f}".format(np.std(multipliers)))
    print("  Range: {:.2f} to {:.2f}".format(
        np.min(multipliers), np.max(multipliers)))

    C = np.mean(multipliers)
    print()
    print("Minimum mu*lam for convergence within N steps:")
    print("  Using conv_steps ~ {:.1f} / (mu*lam)".format(C))
    for target in [10000, 50000, 100000]:
        min_mulam = C / target
        print("  {:>6d} steps: mu*lam >= {:.6f}  (={:.2e})".format(
            target, min_mulam, min_mulam))

    # ==============================================================
    # Summary
    # ==============================================================
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
CLAIM: T ~ 1/(mu*lam).

RESULT: CONFIRMED with correction.

EXACT FORMULA:
    T = -1 / ln|1 - mu*lam*(1 - s*_max)|

LEADING-ORDER:
    T = 1 / (mu*lam * (1 - s*_max))

where s*_max = max_i[-W_0(-xi*g_i^2/lam)] is the largest equilibrium
metric value.

KEY POINTS:
  1. The product mu*lam controls the timescale.  The ratio mu/lam has
     only an INDIRECT effect through the equilibrium s*, which depends
     on xi*g^2/lam.

  2. When lam >> xi*max(g^2), the correction factor 1/(1-s*_max) -> 1
     and T -> 1/(mu*lam) exactly.

  3. Stability requires mu*lam < 2/(1-s*_max).  Optimal convergence
     (spectral radius = 0) occurs at mu*lam = 1/(1-s*_max).

  4. Mean-centering removes the uniform mode but does NOT change the
     convergence timescale for non-uniform modes.
""")
