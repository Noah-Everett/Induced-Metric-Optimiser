"""
Functional tests for alternative metric parameterizations (IMO-73).

Tests:
1. Recovery: metric_param="exp" matches original optimizer behaviour exactly.
2. Softplus identity init: softplus(init_s) == 1.0.
3. Equilibrium: non-exp parameterizations reach finite equilibrium without clip.
4. Adaptive clip: condition number stays below target.
"""

import jax
import jax.numpy as jnp
import numpy as np

from jax_learnable_diag import (
    custom_sgd_learnable_diag,
    _SOFTPLUS_IDENTITY_INIT,
)
from jax_learnable_scalar import custom_sgd_learnable_scalar


def _quadratic_grads(params, a=1.0, b=1.0):
    """Gradient of L = a*x^2 + b*y^2 at (x, y)."""
    x, y = params['x'], params['y']
    return {'x': 2.0 * a * x, 'y': 2.0 * b * y}


def _run_diag_optimizer(metric_param, n_steps=1000, metric_clip=100.0,
                        max_condition_number=None, **kwargs):
    """Run diagonal learnable optimizer on L = x^2 + y^2 and return final state."""
    opt = custom_sgd_learnable_diag(
        learning_rate=0.01,
        momentum=0.9,
        xi=0.1,
        beta=0.8,
        metric_lr=1e-2,
        metric_reg=1e-3,
        metric_clip=metric_clip,
        metric_param=metric_param,
        max_condition_number=max_condition_number,
        **kwargs,
    )

    params = {'x': jnp.array(1.0), 'y': jnp.array(1.0)}
    state = opt.init(params)

    for _ in range(n_steps):
        grads = _quadratic_grads(params)
        updates, state = opt.update(grads, state, params)
        params = jax.tree.map(lambda p, u: p + u, params, updates)

    return params, state


def _run_diag_constant_grads(metric_param, n_steps=1000, metric_clip=20.0,
                              max_condition_number=None, **kwargs):
    """Run optimizer with constant anisotropic gradients.

    Uses a vector param so mean-centering preserves relative s differences.
    Gradient is [2.0, 0.1] -- strongly anisotropic to drive s apart.
    Clip defaults to 20.0 (prevents exp overflow while showing divergence).
    """
    opt = custom_sgd_learnable_diag(
        learning_rate=0.001,  # small LR so params don't blow up
        momentum=0.0,  # no momentum for clean s dynamics
        xi=0.1,
        beta=0.8,
        metric_lr=0.005,
        metric_reg=1e-2,
        metric_clip=metric_clip,
        metric_param=metric_param,
        max_condition_number=max_condition_number,
        **kwargs,
    )

    params = {'w': jnp.array([1.0, 1.0])}
    state = opt.init(params)
    # Anisotropic constant gradient
    grads = {'w': jnp.array([2.0, 0.1])}

    for _ in range(n_steps):
        updates, state = opt.update(grads, state, params)
        params = jax.tree.map(lambda p, u: p + u, params, updates)

    return params, state


def test_recovery():
    """metric_param='exp' must produce identical results to default."""
    opt_default = custom_sgd_learnable_diag(
        learning_rate=0.01, momentum=0.9, xi=0.1, beta=0.8,
        metric_lr=1e-3, metric_reg=1e-4, metric_clip=4.0,
    )
    opt_exp = custom_sgd_learnable_diag(
        learning_rate=0.01, momentum=0.9, xi=0.1, beta=0.8,
        metric_lr=1e-3, metric_reg=1e-4, metric_clip=4.0,
        metric_param="exp",
    )

    params = {'x': jnp.array(1.0), 'y': jnp.array(0.5)}
    state_d = opt_default.init(params)
    state_e = opt_exp.init(params)

    params_d = params.copy()
    params_e = params.copy()

    for _ in range(50):
        grads_d = _quadratic_grads(params_d)
        grads_e = _quadratic_grads(params_e)
        upd_d, state_d = opt_default.update(grads_d, state_d, params_d)
        upd_e, state_e = opt_exp.update(grads_e, state_e, params_e)
        params_d = jax.tree.map(lambda p, u: p + u, params_d, upd_d)
        params_e = jax.tree.map(lambda p, u: p + u, params_e, upd_e)

    np.testing.assert_allclose(params_d['x'], params_e['x'], atol=1e-6)
    np.testing.assert_allclose(params_d['y'], params_e['y'], atol=1e-6)
    np.testing.assert_allclose(state_d.log_diag['x'], state_e.log_diag['x'], atol=1e-6)
    np.testing.assert_allclose(state_d.log_diag['y'], state_e.log_diag['y'], atol=1e-6)
    print("PASS: test_recovery")


def test_softplus_identity_init():
    """softplus(init_s) should equal 1.0 for fair comparison with exp(0) = 1."""
    init_val = _SOFTPLUS_IDENTITY_INIT
    result = float(jax.nn.softplus(jnp.array(init_val)))
    np.testing.assert_allclose(result, 1.0, atol=1e-6)

    # Also verify via optimizer init
    opt = custom_sgd_learnable_diag(metric_param="softplus")
    params = {'w': jnp.ones((3,))}
    state = opt.init(params)
    init_sigma = float(jnp.mean(jax.nn.softplus(state.log_diag['w'])))
    np.testing.assert_allclose(init_sigma, 1.0, atol=1e-6)
    print("PASS: test_softplus_identity_init")


def test_equilibrium_exp_hits_clip():
    """With constant grads, exp parameterization should saturate at clip bounds."""
    _, state = _run_diag_constant_grads("exp", n_steps=2000, metric_clip=20.0)
    s_vals = state.log_diag['w']
    max_s = float(jnp.max(jnp.abs(s_vals)))
    # exp has no equilibrium -- it drives s to the clip
    assert max_s > 15.0, f"Expected s to hit clip, but max |s| = {max_s:.2f}"
    print(f"PASS: test_equilibrium_exp_hits_clip (max |s| = {max_s:.2f})")


def test_equilibrium_softplus_stable():
    """Softplus should reach finite equilibrium well below the clip."""
    _, state = _run_diag_constant_grads("softplus", n_steps=2000, metric_clip=20.0)
    s_vals = state.log_diag['w']
    max_s = float(jnp.max(jnp.abs(s_vals)))
    s_range = float(jnp.max(s_vals) - jnp.min(s_vals))
    # Softplus should stabilize well below clip
    assert max_s < 15.0, f"Expected softplus to stabilize, but max |s| = {max_s:.2f}"
    print(f"PASS: test_equilibrium_softplus_stable (max |s| = {max_s:.2f}, range = {s_range:.2f})")


def test_equilibrium_norm_grad_stable():
    """Normalized gradient should reach finite equilibrium well below the clip."""
    _, state = _run_diag_constant_grads("exp_norm_grad", n_steps=2000, metric_clip=20.0)
    s_vals = state.log_diag['w']
    max_s = float(jnp.max(jnp.abs(s_vals)))
    s_range = float(jnp.max(s_vals) - jnp.min(s_vals))
    assert max_s < 15.0, f"Expected norm_grad to stabilize, but max |s| = {max_s:.2f}"
    print(f"PASS: test_equilibrium_norm_grad_stable (max |s| = {max_s:.2f}, range = {s_range:.2f})")


def test_matched_reg_runs_without_nan():
    """Matched reg should not produce NaN with practical clip."""
    _, state = _run_diag_constant_grads("exp_matched_reg", n_steps=1000, metric_clip=20.0)
    s_vals = state.log_diag['w']
    assert jnp.all(jnp.isfinite(s_vals)), "exp_matched_reg produced NaN"
    max_s = float(jnp.max(jnp.abs(s_vals)))
    print(f"PASS: test_matched_reg_runs_without_nan (max |s| = {max_s:.2f})")


def test_adaptive_clip_condition_number():
    """Adaptive clip should enforce the condition number target."""
    target_cond = 100.0
    _, state = _run_diag_constant_grads(
        "exp_adaptive_clip", n_steps=500, metric_clip=4.0,
        max_condition_number=target_cond,
    )
    s_vals = state.log_diag['w']
    exp_s = jnp.exp(s_vals)
    actual_cond = float(jnp.max(exp_s) / jnp.min(exp_s))
    assert actual_cond <= target_cond + 1.0, (
        f"Condition number {actual_cond:.1f} exceeds target {target_cond}")
    print(f"PASS: test_adaptive_clip_condition_number (cond = {actual_cond:.1f})")


def test_scalar_all_params():
    """All metric_param values should work with scalar learnable without error."""
    for mp in ["exp", "exp_matched_reg", "softplus", "exp_norm_grad", "exp_adaptive_clip"]:
        opt = custom_sgd_learnable_scalar(
            learning_rate=0.01, metric_param=mp, metric_clip=4.0)
        params = {'w': jnp.ones((3,))}
        state = opt.init(params)
        grads = {'w': jnp.array([0.1, 0.2, 0.3])}
        updates, state = opt.update(grads, state, params)
        # Just verify no error and finite output
        assert jnp.all(jnp.isfinite(updates['w'])), f"Non-finite update for metric_param={mp}"
    print("PASS: test_scalar_all_params")


def test_invalid_metric_param():
    """Invalid metric_param should raise ValueError."""
    try:
        custom_sgd_learnable_diag(metric_param="invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid" in str(e)
    print("PASS: test_invalid_metric_param")


if __name__ == "__main__":
    test_recovery()
    test_softplus_identity_init()
    test_equilibrium_exp_hits_clip()
    test_matched_reg_runs_without_nan()
    test_equilibrium_softplus_stable()
    test_equilibrium_norm_grad_stable()
    test_adaptive_clip_condition_number()
    test_scalar_all_params()
    test_invalid_metric_param()
    print("\nAll tests passed!")
