"""
IMO-94: Theory verification -- coupled (theta, s) convergence and Lyapunov function.

Verifies the coupled convergence results from IMO-16 using the actual
derived Newton-targeted optimizer (jax_derived_newton_diag).  Confirms
the joint Lyapunov function V(theta, s) = L(theta) + alpha/2 ||s - s*||^2
decreases monotonically and that no timescale separation is needed.

Tests
-----
1. Coupled convergence across mu/eta ratios (1e-6 to 10) on diagonal quadratics
2. Lyapunov V(theta, s) monotonic decrease over 20k steps (zero violations)
3. Both theta and s components converge for various condition numbers
4. Spectral radii match analytical predictions at equilibrium
5. Rosenbrock (non-quadratic) local convergence near the minimum

Run: PYTHONPATH=repo .conda/bin/python repo/optimisers/verify_lyapunov_coupled.py
"""

import sys
import os
import jax
import jax.numpy as jnp
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)

from optimisers.jax_derived_newton_diag import (
    derived_newton_diag,
    DerivedNewtonDiagState,
)

# Import coupled_jacobian_newton from the notebooks directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "notebooks"))
from importlib import import_module
_ttc = import_module("hp-two-timescale-convergence-verification")
coupled_jacobian_newton = _ttc.coupled_jacobian_newton

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
    h_diag_fn = lambda _: h
    return loss_fn, grad_fn, h_diag_fn, h


def s_target(h_vec):
    """Theoretical fixed point: s_i* = -log(h_i) + mean(log h)."""
    log_h = jnp.log(jnp.array(h_vec, dtype=jnp.float64))
    return -log_h + jnp.mean(log_h)


def run_optimizer(h_vec, eta, beta_s, n_steps, momentum=0.0, metric_clip=15.0,
                  theta0=None, record_every=1):
    """Run derived_newton_diag on a diagonal quadratic, returning trajectories."""
    n = len(h_vec)
    loss_fn, grad_fn, h_diag_fn, h = make_quadratic(h_vec)
    s_star = s_target(h_vec)

    opt = derived_newton_diag(
        learning_rate=eta, momentum=momentum, beta_s=beta_s,
        weight_decay=0.0, metric_clip=metric_clip,
    )

    theta = jnp.ones(n, dtype=jnp.float64) if theta0 is None else jnp.array(theta0, dtype=jnp.float64)
    state = opt.init(theta)

    losses = []
    s_errors = []
    lyapunov_vals = []
    steps_rec = []

    for t in range(1, n_steps + 1):
        grads = grad_fn(theta)
        h_diag = h_diag_fn(theta)
        updates, state = opt.update(grads, state, h_diag, params=theta)
        theta = optax.apply_updates(theta, updates)

        if t % record_every == 0:
            loss = float(loss_fn(theta))
            s = state.log_diag
            s_err = float(jnp.linalg.norm(s - s_star))
            lyap = loss + 0.5 * float(jnp.sum((s - s_star) ** 2))

            losses.append(loss)
            s_errors.append(s_err)
            lyapunov_vals.append(lyap)
            steps_rec.append(t)

    return {
        "losses": np.array(losses),
        "s_errors": np.array(s_errors),
        "lyapunov": np.array(lyapunov_vals),
        "steps": np.array(steps_rec),
        "final_loss": losses[-1] if losses else np.inf,
        "final_theta": np.array(theta),
        "final_s": np.array(state.log_diag),
        "s_star": np.array(s_star),
    }


# ---------------------------------------------------------------------------
# Test 1: Coupled convergence across mu/eta ratios
# ---------------------------------------------------------------------------

def test_convergence_across_ratios():
    """Newton-targeted should converge for ALL mu/eta ratios."""
    print("\nTest 1: Coupled convergence across mu/eta ratios")
    print("-" * 60)
    h_vec = [1.0, 10.0, 100.0]
    eta = 0.005
    # beta_s = mu * lam; with lam=1, mu = eta * ratio => beta_s = eta * ratio
    ratios = [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
    results = {}

    for ratio in ratios:
        beta_s = eta * ratio
        # Clamp beta_s to (0, 1) for the EMA
        beta_s = min(beta_s, 0.99)
        res = run_optimizer(h_vec, eta=eta, beta_s=beta_s, n_steps=20000,
                            momentum=0.0, record_every=100)
        results[ratio] = res
        passed = res["final_loss"] < 1e-5
        report(
            f"mu/eta={ratio:.0e}  beta_s={beta_s:.2e}  "
            f"final_loss={res['final_loss']:.2e}",
            passed,
        )

    return results


# ---------------------------------------------------------------------------
# Test 2: Lyapunov monotonic decrease (20k steps, zero violations)
# ---------------------------------------------------------------------------

def test_lyapunov_monotonic():
    """V(theta, s) = L(theta) + 0.5 ||s - s*||^2 should decrease monotonically."""
    print("\nTest 2: Lyapunov V(theta, s) monotonic decrease (20k steps)")
    print("-" * 60)
    h_vec = [1.0, 10.0, 100.0]
    eta = 0.005
    beta_s = 0.01

    res = run_optimizer(h_vec, eta=eta, beta_s=beta_s, n_steps=20000,
                        momentum=0.0, record_every=1)

    lyap = res["lyapunov"]
    diffs = np.diff(lyap)
    n_increases = int(np.sum(diffs > 1e-15))

    print(f"  Initial Lyapunov: {lyap[0]:.6f}")
    print(f"  Final Lyapunov:   {lyap[-1]:.2e}")
    print(f"  Monotonicity violations: {n_increases}/{len(diffs)}")

    report(
        f"zero violations over 20k steps (got {n_increases})",
        n_increases == 0,
    )
    return res


# ---------------------------------------------------------------------------
# Test 3: Both theta and s converge for various kappa
# ---------------------------------------------------------------------------

def test_joint_convergence():
    """Both ||theta|| and ||s - s*|| should be small after 20k steps."""
    print("\nTest 3: Joint (theta, s) convergence across condition numbers")
    print("-" * 60)
    n = 10

    for kappa in [10.0, 100.0, 1000.0, 1e6]:
        h_vec = np.logspace(0, np.log10(kappa), n).tolist()
        h_geo = float(np.exp(np.mean(np.log(h_vec))))
        # eta must satisfy eta * h_geo < 2 for stability after equalization.
        # Use beta_s=0.5 for fast metric convergence, limiting the transient
        # instability before s reaches the Newton target.
        eta = min(0.005, 0.5 / h_geo)
        beta_s = 0.5
        res = run_optimizer(h_vec, eta=eta, beta_s=beta_s, n_steps=20000,
                            momentum=0.0, record_every=100)
        theta_norm = float(np.linalg.norm(res["final_theta"]))
        s_err = float(np.linalg.norm(res["final_s"] - res["s_star"]))

        passed = theta_norm < 1e-5 and s_err < 1e-2
        report(
            f"kappa={kappa:.0e}  eta={eta:.1e}  ||theta||={theta_norm:.2e}  "
            f"||s-s*||={s_err:.2e}",
            passed,
        )


# ---------------------------------------------------------------------------
# Test 4: Spectral radii match analytical predictions
# ---------------------------------------------------------------------------

def test_spectral_radius():
    """Discrete spectral radius at equilibrium should match theory."""
    print("\nTest 4: Spectral radii match analytical predictions")
    print("-" * 60)
    h = np.array([1.0, 10.0, 100.0])
    N = len(h)

    configs = [
        {"eta": 0.005, "beta_s": 0.001},
        {"eta": 0.005, "beta_s": 0.01},
        {"eta": 0.005, "beta_s": 0.1},
        {"eta": 0.01,  "beta_s": 0.01},
    ]

    print(f"\n  {'eta':>8s} {'beta_s':>8s} | {'rho_theta':>10s} {'rho_s':>10s} "
          f"{'rho_joint':>10s} {'rho_num':>10s} | {'match':>5s}")
    print("  " + "-" * 72)

    all_passed = True
    for cfg in configs:
        eta = cfg["eta"]
        beta_s = cfg["beta_s"]

        h_geo = float(np.exp(np.mean(np.log(h))))
        s_star = -np.log(h) + np.mean(np.log(h))

        # Analytical (discrete map eigenvalues)
        rho_theta = abs(1.0 - eta * h_geo)
        rho_s = abs(1.0 - beta_s)
        rho_joint = max(rho_theta, rho_s)

        # Numerical: Jacobian at equilibrium
        # In the derived optimizer, beta_s = mu*lam.  Use lam=1, mu=beta_s, xi=0.
        J = coupled_jacobian_newton(
            np.zeros(N), s_star, h,
            eta=eta, mu=beta_s, lam=1.0, xi=0.0,
        )
        discrete_map = np.eye(2 * N) + J
        eigs = np.linalg.eigvals(discrete_map)
        rho_num = float(np.max(np.abs(eigs)))

        match = abs(rho_num - rho_joint) < 1e-10
        all_passed &= match

        print(f"  {eta:8.4f} {beta_s:8.4f} | {rho_theta:10.6f} {rho_s:10.6f} "
              f"{rho_joint:10.6f} {rho_num:10.6f} | {'YES' if match else 'NO':>5s}")

        report(
            f"eta={eta} beta_s={beta_s}  |rho_num - rho_analytical| = "
            f"{abs(rho_num - rho_joint):.2e}",
            match,
        )


# ---------------------------------------------------------------------------
# Test 5: Rosenbrock local convergence (non-quadratic)
# ---------------------------------------------------------------------------

def test_rosenbrock():
    """Local convergence near the Rosenbrock minimum [1, 1].

    Uses exact Hessian diagonal (not Hutchinson) to isolate the optimizer's
    convergence properties from estimation noise.  The Rosenbrock Hessian has
    large off-diagonal elements, making single-sample Hutchinson estimates
    too noisy for a clean convergence test.
    """
    print("\nTest 5: Rosenbrock local convergence (non-quadratic)")
    print("-" * 60)

    def rosenbrock(theta):
        x, y = theta[0], theta[1]
        return (1.0 - x) ** 2 + 100.0 * (y - x ** 2) ** 2

    def rosenbrock_hess_diag(theta):
        x, y = theta[0], theta[1]
        d2dx2 = 2.0 + 1200.0 * x ** 2 - 400.0 * y
        d2dy2 = jnp.full_like(x, 200.0)
        return jnp.array([d2dx2, d2dy2])

    loss_fn = jax.jit(rosenbrock)
    grad_fn = jax.jit(jax.grad(rosenbrock))

    theta = jnp.array([0.9, 0.81], dtype=jnp.float64)
    eta = 0.001
    beta_s = 0.1
    n_steps = 20000

    opt = derived_newton_diag(
        learning_rate=eta, momentum=0.0, beta_s=beta_s,
        weight_decay=0.0, metric_clip=8.0, hess_eps=1e-6,
    )
    state = opt.init(theta)

    losses = []
    for t in range(1, n_steps + 1):
        grads = grad_fn(theta)
        h_diag = rosenbrock_hess_diag(theta)
        updates, state = opt.update(grads, state, h_diag, params=theta)
        theta = optax.apply_updates(theta, updates)

        if t % 1000 == 0:
            loss = float(loss_fn(theta))
            losses.append(loss)

    final_loss = float(loss_fn(theta))
    final_theta = np.array(theta)
    dist_to_min = float(np.linalg.norm(final_theta - np.array([1.0, 1.0])))

    print(f"  Initial: theta=[0.9, 0.81], loss={rosenbrock(jnp.array([0.9, 0.81])):.6f}")
    print(f"  Final:   theta=[{final_theta[0]:.6f}, {final_theta[1]:.6f}], loss={final_loss:.2e}")
    print(f"  Distance to minimum: {dist_to_min:.2e}")

    report(
        f"final_loss={final_loss:.2e} (need < 1e-6)",
        final_loss < 1e-6,
    )
    return {"losses": losses, "final_loss": final_loss}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(ratio_results, lyapunov_result):
    """Save convergence plots to disk."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  [skip plots: matplotlib not available]")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Loss vs step for each mu/eta ratio
    ax = axes[0]
    for ratio, res in sorted(ratio_results.items()):
        valid = res["losses"] > 0
        if np.any(valid):
            ax.semilogy(res["steps"][valid], res["losses"][valid],
                        label=f"mu/eta={ratio:.0e}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Convergence across mu/eta ratios")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Lyapunov trajectory
    ax = axes[1]
    lyap = lyapunov_result["lyapunov"]
    steps = lyapunov_result["steps"]
    valid = lyap > 0
    ax.semilogy(steps[valid], lyap[valid], "k-", linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("V(theta, s)")
    ax.set_title("Lyapunov function V = L + 0.5||s - s*||^2")
    ax.grid(True, alpha=0.3)

    # Panel 3: ||s - s*|| trajectory
    ax = axes[2]
    s_err = lyapunov_result["s_errors"]
    valid_s = s_err > 0
    ax.semilogy(steps[valid_s], s_err[valid_s], "b-", linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("||s - s*||")
    ax.set_title("Metric convergence to Newton target")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "lyapunov_coupled_plots.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n  Plots saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("IMO-94: Coupled (theta, s) convergence and Lyapunov verification")
    print("=" * 65)

    ratio_results = test_convergence_across_ratios()
    lyapunov_result = test_lyapunov_monotonic()
    test_joint_convergence()
    test_spectral_radius()
    test_rosenbrock()

    make_plots(ratio_results, lyapunov_result)

    print("\n" + "=" * 65)
    total = PASS_COUNT + FAIL_COUNT
    print(f"Results: {PASS_COUNT}/{total} passed")
    if FAIL_COUNT == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{FAIL_COUNT} TESTS FAILED")
        sys.exit(1)
    print("=" * 65)
