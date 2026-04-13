"""
IMO-93: Theory verification -- robustness theorem and condition number reduction.

Verifies the robustness theorem from IMO-85: the Newton-targeted preconditioner
achieves kappa_prec <= (1+delta)/(1-delta) for relative Hessian estimation error
|eps_i| <= delta, independent of dimension N and original condition number.

Also verifies that Hutchinson + EMA + clip matches perfect Newton on non-diagonal
Hessians while the secant estimator provides only modest improvement.

Tests
-----
1. Robustness bound: kappa_prec <= (1+delta)/(1-delta) across N, kappa, delta
2. Log compression: 50% error in h_hat => ~4% error in s-target for kappa=10^4
3. Full optimizer on dense non-diagonal H (reproduces IMO-85 Section 8c)
4. Full optimizer on block-diagonal H (NN-like structure)

Run: PYTHONPATH=repo .conda/bin/python repo/optimisers/verify_robustness_theorem.py
"""

import sys
import jax
import jax.numpy as jnp
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)

from optimisers.jax_derived_newton_diag import (
    derived_newton_diag,
    DerivedNewtonDiagState,
    hutchinson_hvp_diag,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS_COUNT = 0
FAIL_COUNT = 0


def report(name, passed, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if passed:
        PASS_COUNT += 1
        print(f"  PASS  {name}")
    else:
        FAIL_COUNT += 1
        print(f"  FAIL  {name}  {detail}")


def make_quadratic_dense(H_matrix):
    """L = 1/2 theta^T H theta for dense symmetric PD H.

    Returns (loss_fn, grad_fn, h_diag_exact).
    """
    H = jnp.array(H_matrix, dtype=jnp.float64)
    loss_fn = jax.jit(lambda theta: 0.5 * theta @ H @ theta)
    grad_fn = jax.jit(jax.grad(loss_fn))
    h_diag_exact = jnp.diag(H)
    return loss_fn, grad_fn, h_diag_exact


def make_dense_H(n, kappa, seed):
    """Dense non-diagonal PD matrix with target condition number ~kappa.

    Diagonal: logspace(0, log(kappa), n).
    Off-diagonal: 0.1 * rand(-1,1) * sqrt(h_i*h_j) / (n-1), symmetrised.
    """
    rng = np.random.RandomState(seed)
    h = np.exp(np.linspace(0, np.log(kappa), n))
    H = np.diag(h)
    for i in range(n):
        for j in range(i + 1, n):
            off = 0.1 * rng.uniform(-1, 1) * np.sqrt(h[i] * h[j]) / (n - 1)
            H[i, j] = off
            H[j, i] = off
    eigs = np.linalg.eigvalsh(H)
    if eigs[0] < 0:
        H += (-eigs[0] + 0.01) * np.eye(n)
    return jnp.array(H, dtype=jnp.float64)


def make_block_diag_H(block_sizes, block_kappas, coupling, seed):
    """Block-diagonal PD matrix with intra-block off-diagonal coupling."""
    rng = np.random.RandomState(seed)
    n_total = sum(block_sizes)
    H = np.zeros((n_total, n_total))
    pos = 0
    for bs, bk in zip(block_sizes, block_kappas):
        diag = np.exp(rng.uniform(0, np.log(bk), bs))
        block = np.diag(diag)
        for i in range(bs):
            for j in range(i + 1, bs):
                off = coupling * rng.uniform(-1, 1) * np.sqrt(diag[i] * diag[j])
                block[i, j] = off
                block[j, i] = off
        eigs = np.linalg.eigvalsh(block)
        if eigs[0] < 0:
            block += (-eigs[0] + 0.1) * np.eye(bs)
        H[pos:pos + bs, pos:pos + bs] = block
        pos += bs
    return jnp.array(H, dtype=jnp.float64)


def secant_h_diag(g, prev_g, theta, prev_theta, eps=1e-5):
    """Secant Hessian diagonal estimate: (g - g_prev) / (theta - theta_prev)."""
    dt = theta - prev_theta
    dg = g - prev_g
    safe = jnp.abs(dt) > eps
    denom = jnp.where(safe, dt, jnp.ones_like(dt))
    return jnp.where(safe, dg / denom, jnp.ones_like(g))


def run_optimizer_full(loss_fn, grad_fn, H, theta0, n_steps, method,
                       lr=0.0005, beta_s=0.005, clip=3.0, hess_eps=0.01,
                       rng_seed=55):
    """Run derived_newton_diag on a dense quadratic.

    method: "none" | "perfect" | "hutchinson" | "secant"
    Returns final loss.
    """
    n = theta0.shape[0]
    h_diag_exact = jnp.diag(H)

    use_beta_s = 0.0 if method == "none" else beta_s
    opt = derived_newton_diag(
        learning_rate=lr, momentum=0.0, beta_s=use_beta_s,
        weight_decay=0.0, metric_clip=clip, hess_eps=hess_eps,
    )

    theta = theta0.copy()
    state = opt.init(theta)
    key = jax.random.PRNGKey(rng_seed)
    prev_theta = theta
    prev_g = grad_fn(theta)

    for t in range(n_steps):
        g = grad_fn(theta)

        if method == "none":
            h_diag = jnp.ones(n, dtype=jnp.float64)
        elif method == "perfect":
            h_diag = h_diag_exact
        elif method == "hutchinson":
            key, subkey = jax.random.split(key)
            h_diag = hutchinson_hvp_diag(loss_fn, theta, subkey)
        elif method == "secant":
            if t == 0:
                h_diag = jnp.ones(n, dtype=jnp.float64)
            else:
                h_diag = secant_h_diag(g, prev_g, theta, prev_theta)
        else:
            raise ValueError(f"Unknown method: {method}")

        updates, state = opt.update(g, state, h_diag, params=theta)
        prev_theta = theta
        prev_g = g
        theta = optax.apply_updates(theta, updates)

    return float(loss_fn(theta))


# ---------------------------------------------------------------------------
# Test 1: Robustness bound -- kappa_prec <= (1+delta)/(1-delta)
# ---------------------------------------------------------------------------

def test_robustness_bound():
    print("\nTest 1: kappa_prec <= (1+delta)/(1-delta)")
    print("-" * 60)
    n_trials = 100
    deltas = [0.1, 0.3, 0.5, 0.7, 0.9]

    for N in [2, 10, 100, 1000]:
        for log_kappa in [2, 4, 6, 10]:
            kappa = 10.0 ** log_kappa
            h = np.logspace(0, log_kappa, N)
            all_pass = True
            worst_ratio = 0.0

            for delta in deltas:
                bound = (1.0 + delta) / (1.0 - delta)
                max_kappa_prec = 0.0

                for trial in range(n_trials):
                    rng = np.random.RandomState(42 + trial)
                    eps = rng.uniform(-delta, delta, N)
                    h_hat = h * (1.0 + eps)
                    h_tilde = h / h_hat  # preconditioned eigenvalues
                    kp = h_tilde.max() / h_tilde.min()
                    max_kappa_prec = max(max_kappa_prec, kp)

                ok = max_kappa_prec <= bound + 1e-10
                if not ok:
                    all_pass = False
                worst_ratio = max(worst_ratio, max_kappa_prec / bound)

            report(
                f"N={N:4d}  kappa=10^{log_kappa:2d}  "
                f"worst_ratio={worst_ratio:.6f}",
                all_pass,
                detail=f"bound violated" if not all_pass else "",
            )


# ---------------------------------------------------------------------------
# Test 2: Log compression and improvement factor table
# ---------------------------------------------------------------------------

def test_log_compression():
    print("\nTest 2: Log compression and improvement factors")
    print("-" * 60)

    # Part A: verify improvement factor table (deterministic)
    kappa_orig = 1e4
    expected = [
        (0.1, 1.222, 8182),
        (0.3, 1.857, 5385),
        (0.5, 3.000, 3333),
        (0.7, 5.667, 1765),
        (0.9, 19.00, 526),
    ]
    for delta, kp_bound, improvement in expected:
        computed_bound = (1.0 + delta) / (1.0 - delta)
        computed_improvement = int(kappa_orig / computed_bound)
        bound_ok = abs(computed_bound - kp_bound) < 0.001
        impr_ok = abs(computed_improvement - improvement) <= 1
        report(
            f"delta={delta}  kp_bound={computed_bound:.3f}  "
            f"improvement={computed_improvement}x",
            bound_ok and impr_ok,
        )

    # Part B: verify log compression numerically
    N = 100
    kappa = 1e4
    delta = 0.5
    n_trials = 100
    h = np.logspace(0, np.log10(kappa), N)
    log_h = np.log(h)
    s_star = -log_h + np.mean(log_h)
    s_range = s_star.max() - s_star.min()

    rel_errors = []
    for trial in range(n_trials):
        rng = np.random.RandomState(100 + trial)
        eps = rng.uniform(-delta, delta, N)
        h_hat = h * (1.0 + eps)
        log_h_hat = np.log(h_hat)
        s_hat = -log_h_hat + np.mean(log_h_hat)
        rel_err = np.max(np.abs(s_hat - s_star)) / s_range
        rel_errors.append(rel_err)

    median_err = np.median(rel_errors)
    max_err = np.max(rel_errors)
    # Per-component bound is log(1.5)/log(1e4) ~ 4.4%, but mean-centering
    # shifts errors, so the max-over-components median is ~7% and worst ~9%.
    report(
        f"log compression  delta={delta}  kappa={kappa:.0e}  "
        f"median_err={median_err:.3f}  max_err={max_err:.3f}",
        median_err < 0.08 and max_err < 0.12,
    )


# ---------------------------------------------------------------------------
# Test 3: Full optimizer on dense non-diagonal H
# ---------------------------------------------------------------------------

def test_dense_non_diagonal():
    print("\nTest 3: Dense non-diagonal H (kappa~100, 20k steps)")
    print("-" * 60)
    n = 10
    H = make_dense_H(n, 100.0, seed=55)
    loss_fn, grad_fn, _ = make_quadratic_dense(H)

    kappa_actual = float(jnp.linalg.cond(H))
    print(f"  Matrix condition number: {kappa_actual:.1f}")

    theta0 = jnp.array(
        np.random.RandomState(55).uniform(-1, 1, n), dtype=jnp.float64
    )

    results = {}
    for method in ["none", "perfect", "hutchinson", "secant"]:
        loss = run_optimizer_full(
            loss_fn, grad_fn, H, theta0, n_steps=20000, method=method,
        )
        results[method] = loss
        log_loss = np.log10(max(loss, 1e-300))
        print(f"  {method:12s}  loss = {loss:.2e}  (10^{log_loss:.1f})")

    ll = {k: np.log10(max(v, 1e-300)) for k, v in results.items()}

    report(
        "perfect vastly better than none",
        ll["perfect"] < ll["none"] - 50,
        detail=f"gap={ll['none'] - ll['perfect']:.1f} (need >50)",
    )
    report(
        "Hutchinson matches perfect",
        abs(ll["hutchinson"] - ll["perfect"]) < 5,
        detail=f"gap={abs(ll['hutchinson'] - ll['perfect']):.1f} (need <5)",
    )
    report(
        "secant improves over none",
        results["secant"] < results["none"],
    )
    report(
        "Hutchinson beats secant",
        results["hutchinson"] < results["secant"],
    )


# ---------------------------------------------------------------------------
# Test 4: Full optimizer on block-diagonal H
# ---------------------------------------------------------------------------

def test_block_diagonal():
    print("\nTest 4: Block-diagonal H (NN-like, 20k steps)")
    print("-" * 60)
    H = make_block_diag_H(
        block_sizes=[5, 3, 2], block_kappas=[20, 50, 10],
        coupling=0.3, seed=42,
    )
    n = H.shape[0]
    loss_fn, grad_fn, _ = make_quadratic_dense(H)

    kappa_actual = float(jnp.linalg.cond(H))
    print(f"  Matrix condition number: {kappa_actual:.1f}")

    theta0 = jnp.array(
        np.random.RandomState(42).uniform(-1, 1, n), dtype=jnp.float64
    )

    results = {}
    for method in ["none", "perfect", "hutchinson", "secant"]:
        loss = run_optimizer_full(
            loss_fn, grad_fn, H, theta0, n_steps=20000, method=method,
            rng_seed=42,
        )
        results[method] = loss
        log_loss = np.log10(max(loss, 1e-300))
        print(f"  {method:12s}  loss = {loss:.2e}  (10^{log_loss:.1f})")

    report(
        "Hutchinson significantly better than none",
        results["hutchinson"] < results["none"] * 1e-3,
    )
    report(
        "secant improves over none",
        results["secant"] < results["none"],
    )
    report(
        "Hutchinson beats secant",
        results["hutchinson"] < results["secant"],
    )
    report(
        "sanity: problem solvable",
        results["none"] < 1e-10,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("IMO-93: Robustness theorem verification")
    print("=" * 60)

    test_robustness_bound()
    test_log_compression()
    test_dense_non_diagonal()
    test_block_diagonal()

    print("\n" + "=" * 60)
    total = PASS_COUNT + FAIL_COUNT
    print(f"Results: {PASS_COUNT}/{total} passed")
    if FAIL_COUNT == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{FAIL_COUNT} TESTS FAILED")
        sys.exit(1)
    print("=" * 60)
