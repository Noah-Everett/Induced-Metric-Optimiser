# optimisers/test_newton_diag.py
"""
Functional tests for the derived Newton-targeted diagonal optimizer.

Tests:
1. Metric recovery on diagonal quadratic (exact H passed)
2. Hutchinson estimator accuracy (averaged Rademacher samples)
3. beta_s=0 freezes metric at initialisation
4. Saddle differentiation (positive vs negative curvature)
5. Rosenbrock convergence (end-to-end with Hutchinson helper)

Run: PYTHONPATH=repo .conda/bin/python repo/optimisers/test_newton_diag.py
"""

import jax
import jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)

from optimisers.jax_derived_newton_diag import (
    derived_newton_diag,
    hutchinson_hvp_diag,
    DerivedNewtonDiagState,
)


def run_optimizer_hvp(opt, params_init, grad_fn, h_diag_fn, n_steps,
                      needs_params=True):
    """Run optimizer for n_steps with externally provided h_diag."""
    params = params_init.copy()
    state = opt.init(params)
    losses = []
    for _ in range(n_steps):
        grads = grad_fn(params)
        h_diag = h_diag_fn(params)
        if needs_params:
            updates, state = opt.update(grads, state, h_diag, params=params)
        else:
            updates, state = opt.update(grads, state, h_diag)
        params = optax.apply_updates(params, updates)
    return state, params


# ---- Test 1: Metric recovery on diagonal quadratic ----

def test_metric_recovery():
    """On L = a*x^2 + b*y^2, with exact H passed, s should converge to Newton target.

    After mean-centering, s_1 - s_2 should converge to -log(2a) + log(2b) = log(b/a).
    """
    a, b = 5.0, 0.5
    loss_fn = jax.jit(lambda p: a * p[0]**2 + b * p[1]**2)
    grad_fn = jax.jit(jax.grad(loss_fn))
    # Exact Hessian diagonal: diag(2a, 2b)
    h_diag_fn = lambda p: jnp.array([2.0 * a, 2.0 * b])

    opt = derived_newton_diag(
        learning_rate=0.01, momentum=0.0, beta_s=0.1,
        weight_decay=0.0, metric_clip=10.0,
    )
    p0 = jnp.array([1.0, 1.0])
    state, params = run_optimizer_hvp(opt, p0, grad_fn, h_diag_fn, 200)

    s = state.log_diag
    s_diff = float(s[0] - s[1])
    expected_diff = jnp.log(b / a)  # = log(0.5/5) = log(0.1) ~ -2.303
    print(f"[Metric recovery] s_diff={s_diff:.4f}, expected={float(expected_diff):.4f}")
    assert abs(s_diff - float(expected_diff)) < 0.1, (
        f"s_diff={s_diff:.4f} should be close to {float(expected_diff):.4f}"
    )
    print("PASS: metric recovery")


# ---- Test 2: Hutchinson estimator accuracy ----

def test_hutchinson_accuracy():
    """Averaged Hutchinson samples should match true Hessian diagonal."""
    a, b = 3.0, 7.0
    loss_fn = lambda p: a * p[0]**2 + b * p[1]**2
    params = jnp.array([1.0, 1.0])
    true_diag = jnp.array([2.0 * a, 2.0 * b])

    n_samples = 500
    key = jax.random.PRNGKey(42)
    h_sum = jnp.zeros(2)
    for i in range(n_samples):
        key, subkey = jax.random.split(key)
        h = hutchinson_hvp_diag(loss_fn, params, subkey)
        h_sum = h_sum + h
    h_mean = h_sum / n_samples

    err = jnp.max(jnp.abs(h_mean - true_diag))
    print(f"[Hutchinson accuracy] mean={h_mean}, true={true_diag}, max_err={float(err):.2e}")
    # For a purely diagonal quadratic, each Rademacher sample is exact
    # (v_i * H_ii * v_i = H_ii), so the mean should match to float precision.
    assert float(err) < 1e-10, f"Hutchinson mean too far from truth: err={float(err)}"
    print("PASS: Hutchinson accuracy")


# ---- Test 3: beta_s=0 freezes metric ----

def test_beta_s_zero():
    """With beta_s=0, s should stay at initialisation (all zeros)."""
    loss_fn = jax.jit(lambda p: p[0]**2 + p[1]**2)
    grad_fn = jax.jit(jax.grad(loss_fn))
    h_diag_fn = lambda p: jnp.array([2.0, 2.0])

    opt = derived_newton_diag(
        learning_rate=0.01, momentum=0.9, beta_s=0.0,
        weight_decay=0.0, metric_clip=4.0,
    )
    p0 = jnp.array([1.0, 1.0])
    state, _ = run_optimizer_hvp(opt, p0, grad_fn, h_diag_fn, 50)

    s_max = float(jnp.max(jnp.abs(state.log_diag)))
    print(f"[beta_s=0] max |s| = {s_max:.2e}")
    assert s_max < 1e-10, f"s should be zero with beta_s=0, got max |s|={s_max}"
    print("PASS: beta_s=0 freezes metric")


# ---- Test 4: Saddle differentiation ----

def test_saddle_differentiation():
    """On L = x^2 - y^2, H = diag(2, -2).

    Positive curvature (H_11=2 > eps): s_1 = -log(2) < 0
    Negative curvature (H_22=-2, clamped to eps): s_2 = -log(eps) >> 0
    So s_1 < s_2 after mean-centering.
    """
    loss_fn = jax.jit(lambda p: p[0]**2 - p[1]**2)
    grad_fn = jax.jit(jax.grad(loss_fn))
    # Pass exact Hessian: diag(2, -2)
    h_diag_fn = lambda p: jnp.array([2.0, -2.0])

    opt = derived_newton_diag(
        learning_rate=0.01, momentum=0.0, beta_s=0.2,
        weight_decay=0.0, metric_clip=10.0, hess_eps=1e-6,
    )
    p0 = jnp.array([1.0, 1.0])
    state, _ = run_optimizer_hvp(opt, p0, grad_fn, h_diag_fn, 50)

    s1, s2 = float(state.log_diag[0]), float(state.log_diag[1])
    print(f"[Saddle] s1={s1:.4f} (H=2), s2={s2:.4f} (H=-2)")
    assert s1 < s2, (
        f"Expected s1 < s2 for positive vs negative curvature, got s1={s1:.4f}, s2={s2:.4f}"
    )
    print("PASS: saddle differentiation")


# ---- Test 5: Rosenbrock convergence (end-to-end with Hutchinson) ----

def test_rosenbrock_convergence():
    """Verify convergence on Rosenbrock using the Hutchinson helper."""
    loss_fn = jax.jit(lambda p: (1.0 - p[0])**2 + 100.0 * (p[1] - p[0]**2)**2)
    p0 = jnp.array([0.0, 0.0])

    opt = derived_newton_diag(
        learning_rate=0.001, momentum=0.9, beta_s=0.05,
        weight_decay=0.0, metric_clip=4.0, hess_eps=1e-6,
    )
    params = p0.copy()
    state = opt.init(params)
    key = jax.random.PRNGKey(0)
    n_steps = 1000

    initial_loss = float(loss_fn(params))
    for i in range(n_steps):
        key, subkey = jax.random.split(key)
        grads = jax.grad(loss_fn)(params)
        h_diag = hutchinson_hvp_diag(loss_fn, params, subkey)
        updates, state = opt.update(grads, state, h_diag, params=params)
        params = optax.apply_updates(params, updates)

    final_loss = float(loss_fn(params))
    print(f"[Rosenbrock] initial={initial_loss:.4f}, final={final_loss:.6f}")
    # Sanity check: should make meaningful progress, not just epsilon improvement
    assert final_loss < 0.9 * initial_loss, (
        f"Should converge meaningfully: final={final_loss}, initial={initial_loss}"
    )
    print("PASS: Rosenbrock convergence")


# ---- Test 6: Weight decay produces different trajectory ----

def test_weight_decay():
    """With weight_decay > 0 and params passed, trajectory should differ from wd=0."""
    loss_fn = jax.jit(lambda p: p[0]**2 + p[1]**2)
    grad_fn = jax.jit(jax.grad(loss_fn))
    h_diag_fn = lambda p: jnp.array([2.0, 2.0])
    p0 = jnp.array([1.0, 1.0])

    hparams = dict(learning_rate=0.01, momentum=0.9, beta_s=0.1, metric_clip=4.0)

    state_nowd, params_nowd = run_optimizer_hvp(
        derived_newton_diag(**hparams, weight_decay=0.0),
        p0, grad_fn, h_diag_fn, 50, needs_params=True,
    )
    state_wd, params_wd = run_optimizer_hvp(
        derived_newton_diag(**hparams, weight_decay=0.1),
        p0, grad_fn, h_diag_fn, 50, needs_params=True,
    )

    diff = float(jnp.max(jnp.abs(params_nowd - params_wd)))
    print(f"[Weight decay] param diff (wd=0 vs wd=0.1): {diff:.6f}")
    assert diff > 1e-4, f"Weight decay should change trajectory, got diff={diff}"
    print("PASS: weight decay")


# ---- Test 7: params=None path works without error ----

def test_params_none():
    """Calling update without params keyword should work (no weight decay)."""
    loss_fn = jax.jit(lambda p: p[0]**2 + p[1]**2)
    grad_fn = jax.jit(jax.grad(loss_fn))
    h_diag_fn = lambda p: jnp.array([2.0, 2.0])
    p0 = jnp.array([1.0, 1.0])

    opt = derived_newton_diag(learning_rate=0.01, momentum=0.9, beta_s=0.1)
    state, params = run_optimizer_hvp(
        opt, p0, grad_fn, h_diag_fn, 20, needs_params=False,
    )

    final_loss = float(loss_fn(params))
    print(f"[params=None] final loss={final_loss:.6f}")
    assert final_loss < 2.0, f"Should still converge: final_loss={final_loss}"
    assert jnp.all(jnp.isfinite(params)), "Params should be finite"
    print("PASS: params=None path")


if __name__ == "__main__":
    print("=" * 60)
    print("Derived Newton-targeted optimizer functional tests")
    print("=" * 60)
    print()
    test_metric_recovery()
    print()
    test_hutchinson_accuracy()
    print()
    test_beta_s_zero()
    print()
    test_saddle_differentiation()
    print()
    test_weight_decay()
    print()
    test_params_none()
    print()
    test_rosenbrock_convergence()
    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
