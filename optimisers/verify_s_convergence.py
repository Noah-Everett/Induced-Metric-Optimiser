"""
IMO-92: Theory verification -- s convergence on diagonal quadratics.

Verifies the derived Newton-targeted optimizer's metric dynamics on
diagonal quadratic losses L = 1/2 theta^T diag(h) theta with various
condition numbers kappa.

Tests
-----
1. s_i converges to -log(h_i) + mean(log h) for kappa in {10, 100, 1000, 1e6}
2. Convergence timescale matches tau = 1/beta_s (exponential decay)
3. Preconditioned eigenvalues exp(s_i)*h_i equalize to h_geo
4. Convergence rate matches |1 - eta*h_geo| at various eta
5. Single-step convergence matches perfect Newton oracle

Run: PYTHONPATH=repo .conda/bin/python repo/optimisers/verify_s_convergence.py
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


def make_quadratic(h_vec):
    """L = 1/2 sum_i h_i theta_i^2.  Returns (loss_fn, grad_fn, h_diag_fn, h)."""
    h = jnp.array(h_vec, dtype=jnp.float64)
    loss_fn = jax.jit(lambda theta: 0.5 * jnp.sum(h * theta ** 2))
    grad_fn = jax.jit(jax.grad(loss_fn))
    h_diag_fn = lambda _: h          # exact constant Hessian diagonal
    return loss_fn, grad_fn, h_diag_fn, h


def s_target(h_vec):
    """Theoretical fixed point: s_i* = -log(h_i) + mean(log h)."""
    log_h = jnp.log(jnp.array(h_vec, dtype=jnp.float64))
    return -log_h + jnp.mean(log_h)


def converge_s(h_vec, beta_s=0.3, n_steps=300, metric_clip=15.0):
    """Run optimizer with negligible lr to converge s without moving theta."""
    n = len(h_vec)
    _, grad_fn, h_diag_fn, _ = make_quadratic(h_vec)
    opt = derived_newton_diag(
        learning_rate=1e-10, momentum=0.0, beta_s=beta_s,
        weight_decay=0.0, metric_clip=metric_clip,
    )
    theta = jnp.ones(n, dtype=jnp.float64)
    state = opt.init(theta)
    for _ in range(n_steps):
        grads = grad_fn(theta)
        h = h_diag_fn(theta)
        updates, state = opt.update(grads, state, h, params=theta)
        theta = optax.apply_updates(theta, updates)
    return state.log_diag


# ---------------------------------------------------------------------------
# Test 1: s convergence to Newton target across kappa values
# ---------------------------------------------------------------------------

def test_s_convergence():
    print("\nTest 1: s_i -> -log(h_i) + mean(log h)")
    print("-" * 55)
    n = 10
    for kappa in [10.0, 100.0, 1000.0, 1e6]:
        h_vec = np.logspace(0, np.log10(kappa), n)
        s_conv = converge_s(h_vec)
        s_star = s_target(h_vec)
        err = float(jnp.max(jnp.abs(s_conv - s_star)))
        report(f"kappa={kappa:.0e}  max|s-s*|={err:.2e}", err < 0.01)


# ---------------------------------------------------------------------------
# Test 2: Convergence timescale matches tau = 1/beta_s
# ---------------------------------------------------------------------------

def test_timescale():
    """Error should decay as (1-beta_s)^t.

    At t = tau = 1/beta_s steps, the error ratio err(tau)/err(0) should
    equal (1-beta_s)^tau ~ 1/e ~ 0.368.
    """
    print("\nTest 2: Convergence timescale ~ 1/beta_s")
    print("-" * 55)
    n = 5
    h_vec = np.logspace(0, 2, n)     # kappa = 100
    _, grad_fn, h_diag_fn, _ = make_quadratic(h_vec)
    s_star = s_target(h_vec)

    for beta_s in [0.01, 0.05, 0.1, 0.2]:
        opt = derived_newton_diag(
            learning_rate=1e-10, momentum=0.0, beta_s=beta_s,
            weight_decay=0.0, metric_clip=15.0,
        )
        n_steps = max(int(20.0 / beta_s), 50)
        theta = jnp.ones(n, dtype=jnp.float64)
        state = opt.init(theta)

        errors = [float(jnp.max(jnp.abs(state.log_diag - s_star)))]
        for _ in range(n_steps):
            grads = grad_fn(theta)
            h = h_diag_fn(theta)
            updates, state = opt.update(grads, state, h, params=theta)
            theta = optax.apply_updates(theta, updates)
            errors.append(float(jnp.max(jnp.abs(state.log_diag - s_star))))

        tau = int(round(1.0 / beta_s))
        measured = errors[tau] / errors[0]
        expected = (1.0 - beta_s) ** tau
        rel_err = abs(measured - expected) / (expected + 1e-15)
        report(
            f"beta_s={beta_s}  err(tau)/err(0)={measured:.4f}  "
            f"expected={expected:.4f}",
            rel_err < 0.05,
        )


# ---------------------------------------------------------------------------
# Test 3: Preconditioned eigenvalues equalize to h_geo
# ---------------------------------------------------------------------------

def test_eigenvalue_equalization():
    print("\nTest 3: exp(s_i)*h_i -> h_geo")
    print("-" * 55)
    n = 10
    for kappa in [10.0, 100.0, 1000.0, 1e6]:
        h_vec = np.logspace(0, np.log10(kappa), n)
        h = jnp.array(h_vec, dtype=jnp.float64)
        h_geo = float(jnp.exp(jnp.mean(jnp.log(h))))
        s_conv = converge_s(h_vec)

        precond = jnp.exp(s_conv) * h
        rel_std = float(jnp.std(precond) / jnp.mean(precond))
        mean_p = float(jnp.mean(precond))
        h_geo_err = abs(mean_p - h_geo) / h_geo

        report(
            f"kappa={kappa:.0e}  h_geo={h_geo:.4f}  "
            f"mean_precond={mean_p:.4f}  rel_std={rel_std:.2e}",
            rel_std < 0.01 and h_geo_err < 0.01,
        )


# ---------------------------------------------------------------------------
# Test 4: Convergence rate = |1 - eta * h_geo|
# ---------------------------------------------------------------------------

def test_convergence_rate():
    """After s converges, one-step contraction = |1 - eta * h_geo|.

    We inject pre-converged s into a fresh state with beta_s=0 (frozen
    metric) and measure the norm ratio ||theta_1|| / ||theta_0|| for a
    single step with momentum=0.
    """
    print("\nTest 4: Convergence rate = |1 - eta*h_geo|")
    print("-" * 55)
    n = 5
    kappa = 100.0
    h_vec = np.logspace(0, np.log10(kappa), n)
    h = jnp.array(h_vec, dtype=jnp.float64)
    h_geo = float(jnp.exp(jnp.mean(jnp.log(h))))
    _, grad_fn, h_diag_fn, _ = make_quadratic(h_vec)

    s_conv = converge_s(h_vec)
    theta0 = jnp.ones(n, dtype=jnp.float64)

    for eta_frac in [0.5, 0.8, 0.95, 1.0]:
        eta = eta_frac / h_geo
        expected_rate = abs(1.0 - eta_frac)

        opt = derived_newton_diag(
            learning_rate=eta, momentum=0.0, beta_s=0.0,
            weight_decay=0.0, metric_clip=15.0,
        )
        state = DerivedNewtonDiagState(
            step=jnp.zeros([], dtype=jnp.int32),
            momentum=jax.tree.map(jnp.zeros_like, theta0),
            log_diag=s_conv,
        )

        grads = grad_fn(theta0)
        h_diag = h_diag_fn(theta0)
        updates, _ = opt.update(grads, state, h_diag, params=theta0)
        theta1 = optax.apply_updates(theta0, updates)

        measured_rate = float(jnp.linalg.norm(theta1) / jnp.linalg.norm(theta0))
        report(
            f"eta={eta_frac}/h_geo  measured={measured_rate:.8f}  "
            f"expected={expected_rate:.8f}",
            abs(measured_rate - expected_rate) < 1e-6,
        )


# ---------------------------------------------------------------------------
# Test 5: Newton oracle comparison -- single-step convergence
# ---------------------------------------------------------------------------

def test_newton_oracle():
    """At eta=1/h_geo with converged s, one step should drive theta to zero,
    matching the Newton oracle theta_{t+1} = theta_t - H^{-1} g = 0.
    """
    print("\nTest 5: Match Newton oracle (single-step convergence)")
    print("-" * 55)
    n = 10
    for kappa in [10.0, 100.0, 1000.0, 1e6]:
        h_vec = np.logspace(0, np.log10(kappa), n)
        h = jnp.array(h_vec, dtype=jnp.float64)
        h_geo = float(jnp.exp(jnp.mean(jnp.log(h))))
        _, grad_fn, h_diag_fn, _ = make_quadratic(h_vec)
        theta0 = jnp.ones(n, dtype=jnp.float64) * 0.5

        s_conv = converge_s(h_vec)
        eta = 1.0 / h_geo

        opt = derived_newton_diag(
            learning_rate=eta, momentum=0.0, beta_s=0.0,
            weight_decay=0.0, metric_clip=15.0,
        )
        state = DerivedNewtonDiagState(
            step=jnp.zeros([], dtype=jnp.int32),
            momentum=jax.tree.map(jnp.zeros_like, theta0),
            log_diag=s_conv,
        )

        grads = grad_fn(theta0)
        h_diag = h_diag_fn(theta0)
        updates, _ = opt.update(grads, state, h_diag, params=theta0)
        theta1 = optax.apply_updates(theta0, updates)
        residual = float(jnp.linalg.norm(theta1))

        report(
            f"kappa={kappa:.0e}  ||theta_1||={residual:.2e}  (Newton gives 0)",
            residual < 1e-8,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("IMO-92: s convergence verification on diagonal quadratics")
    print("=" * 60)

    test_s_convergence()
    test_timescale()
    test_eigenvalue_equalization()
    test_convergence_rate()
    test_newton_oracle()

    print("\n" + "=" * 60)
    total = PASS_COUNT + FAIL_COUNT
    print(f"Results: {PASS_COUNT}/{total} passed")
    if FAIL_COUNT == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{FAIL_COUNT} TESTS FAILED")
        sys.exit(1)
    print("=" * 60)
