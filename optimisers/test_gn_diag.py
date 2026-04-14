# optimisers/test_gn_diag.py
"""
Functional tests for the exact GN diagonal computation.

Tests:
1. Shape match (h_diag pytree matches params)
2. Positive semi-definite (all values >= 0)
3. Single-layer analytic verification
4. Multi-layer against brute-force vmap(grad)
5. Cross-entropy output gradient correctness
6. MSE output gradient correctness
7. Activation derivative correctness
8. Determinism (two calls → identical)
9. Optimizer integration (plug into derived_newton_diag)
10. loss_grad_and_gn_diag consistency

Run: PYTHONPATH=repo .conda/bin/python repo/optimisers/test_gn_diag.py
"""

import jax
import jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)

from optimisers.jax_gn_diag import (
    gn_diag_mlp,
    loss_grad_and_gn_diag,
    cross_entropy_output_grad,
    mse_output_grad,
    _activation_deriv,
)
from optimisers.jax_derived_newton_diag import derived_newton_diag


def _init_mlp_params(key, in_dim, hidden, out_dim):
    """Create Flax-style MLP params dict for testing."""
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    return {'params': {
        'dense1': {
            'kernel': jax.random.normal(k1, (in_dim, hidden)) * 0.1,
            'bias': jax.random.normal(k2, (hidden,)) * 0.01,
        },
        'dense2': {
            'kernel': jax.random.normal(k3, (hidden, hidden)) * 0.1,
            'bias': jax.random.normal(k4, (hidden,)) * 0.01,
        },
        'dense3': {
            'kernel': jax.random.normal(k5, (hidden, out_dim)) * 0.1,
            'bias': jax.random.normal(k6, (out_dim,)) * 0.01,
        },
    }}


def _mlp_forward(params, x, activation_fn):
    """Manual MLP forward pass matching Flax MLP."""
    p = params['params']
    h = activation_fn(x @ p['dense1']['kernel'] + p['dense1']['bias'])
    h = activation_fn(h @ p['dense2']['kernel'] + p['dense2']['bias'])
    return h @ p['dense3']['kernel'] + p['dense3']['bias']


# ---- Test 1: Shape match ----

def test_shape_match():
    """h_diag pytree must have same structure and leaf shapes as params."""
    key = jax.random.PRNGKey(0)
    params = _init_mlp_params(key, 8, 4, 3)
    x = jax.random.normal(key, (16, 8))
    y = jax.random.randint(key, (16,), 0, 3)

    h_diag = gn_diag_mlp(params, x, y, jax.nn.gelu,
                          cross_entropy_output_grad)

    for layer in ('dense1', 'dense2', 'dense3'):
        for field in ('kernel', 'bias'):
            expected = params['params'][layer][field].shape
            actual = h_diag['params'][layer][field].shape
            assert actual == expected, (
                f"{layer}/{field}: expected {expected}, got {actual}")

    print("PASS: shape match")


# ---- Test 2: Positive semi-definite ----

def test_positive_semidefinite():
    """All GN diagonal values must be >= 0."""
    key = jax.random.PRNGKey(1)
    params = _init_mlp_params(key, 8, 4, 3)
    x = jax.random.normal(key, (16, 8))
    y = jax.random.randint(key, (16,), 0, 3)

    h_diag = gn_diag_mlp(params, x, y, jax.nn.gelu,
                          cross_entropy_output_grad)

    for layer in ('dense1', 'dense2', 'dense3'):
        for field in ('kernel', 'bias'):
            vals = h_diag['params'][layer][field]
            min_val = float(jnp.min(vals))
            assert min_val >= 0.0, (
                f"{layer}/{field}: min={min_val} < 0")

    print("PASS: positive semi-definite")


# ---- Test 3: Single-layer analytic ----

def test_single_layer_analytic():
    """For a 1-layer linear net, verify GN diagonal against analytic formula."""
    key = jax.random.PRNGKey(2)
    k1, k2, k3 = jax.random.split(key, 3)
    in_dim, out_dim = 4, 3
    B = 8

    params = {'params': {
        'dense1': {
            'kernel': jax.random.normal(k1, (in_dim, out_dim)) * 0.1,
            'bias': jax.random.normal(k2, (out_dim,)) * 0.01,
        },
    }}
    x = jax.random.normal(k3, (B, in_dim))
    y = jax.random.randint(key, (B,), 0, out_dim)

    h_diag = gn_diag_mlp(params, x, y, jax.nn.gelu,
                          cross_entropy_output_grad,
                          layer_keys=('dense1',))

    # Compute analytically
    p = params['params']
    logits = x @ p['dense1']['kernel'] + p['dense1']['bias']
    g = jax.nn.softmax(logits) - jax.nn.one_hot(y, out_dim)
    expected_kernel = (x ** 2).T @ (g ** 2) / B
    expected_bias = jnp.mean(g ** 2, axis=0)

    err_k = float(jnp.max(jnp.abs(h_diag['params']['dense1']['kernel'] - expected_kernel)))
    err_b = float(jnp.max(jnp.abs(h_diag['params']['dense1']['bias'] - expected_bias)))
    print(f"[single-layer] kernel err={err_k:.2e}, bias err={err_b:.2e}")
    assert err_k < 1e-12, f"kernel err too large: {err_k}"
    assert err_b < 1e-12, f"bias err too large: {err_b}"
    print("PASS: single-layer analytic")


# ---- Test 4: Multi-layer against brute-force ----

def test_multilayer_vs_bruteforce():
    """Compare GN diagonal against brute-force per-sample gradient squared.

    The empirical Fisher diagonal is:
        F_ii = (1/B) sum_d (dL_d/dtheta_i)^2

    Compute this via vmap(grad) and compare against gn_diag_mlp.
    """
    key = jax.random.PRNGKey(3)
    in_dim, hidden, out_dim = 4, 6, 3
    B = 16
    params = _init_mlp_params(key, in_dim, hidden, out_dim)
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (B, in_dim))
    y = jax.random.randint(k2, (B,), 0, out_dim)

    h_diag = gn_diag_mlp(params, x, y, jax.nn.gelu,
                          cross_entropy_output_grad)

    # Brute force: per-sample gradient squared, averaged
    def per_sample_loss(params, xi, yi):
        logits = _mlp_forward(params, xi[None, :], jax.nn.gelu)
        return optax.softmax_cross_entropy_with_integer_labels(logits, yi[None]).squeeze()

    per_sample_grads = jax.vmap(
        jax.grad(per_sample_loss), in_axes=(None, 0, 0)
    )(params, x, y)

    brute_force = jax.tree.map(
        lambda g: jnp.mean(g ** 2, axis=0), per_sample_grads
    )

    for layer in ('dense1', 'dense2', 'dense3'):
        for field in ('kernel', 'bias'):
            gn_val = h_diag['params'][layer][field]
            bf_val = brute_force['params'][layer][field]
            rel_err = float(jnp.max(jnp.abs(gn_val - bf_val) /
                                     jnp.maximum(jnp.abs(bf_val), 1e-10)))
            print(f"[multi-layer] {layer}/{field}: max_rel_err={rel_err:.2e}")
            assert rel_err < 1e-4, (
                f"{layer}/{field}: rel_err={rel_err} too large")

    print("PASS: multi-layer vs brute-force")


# ---- Test 5: Cross-entropy output gradient ----

def test_cross_entropy_grad():
    """Verify cross_entropy_output_grad matches softmax - one_hot."""
    logits = jnp.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
    y = jnp.array([0, 2])

    g = cross_entropy_output_grad(logits, y)
    expected = jax.nn.softmax(logits) - jax.nn.one_hot(y, 3)
    err = float(jnp.max(jnp.abs(g - expected)))
    assert err < 1e-12, f"cross-entropy grad err={err}"
    print("PASS: cross-entropy output gradient")


# ---- Test 6: MSE output gradient ----

def test_mse_grad():
    """Verify mse_output_grad for scalar and vector outputs."""
    # Scalar output (output_dim=1)
    logits_1d = jnp.array([[1.5], [2.0], [0.5]])
    y_1d = jnp.array([1.0, 3.0, 0.0])
    g = mse_output_grad(logits_1d, y_1d)
    expected = 2.0 * (logits_1d.squeeze(-1) - y_1d)[:, None]
    assert g.shape == (3, 1), f"shape={g.shape}"
    err = float(jnp.max(jnp.abs(g - expected)))
    assert err < 1e-12, f"MSE 1D err={err}"

    # Multi-dim output
    logits_2d = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y_2d = jnp.array([[0.5, 1.5], [2.5, 3.5]])
    g2 = mse_output_grad(logits_2d, y_2d)
    expected2 = 2.0 * (logits_2d - y_2d)
    err2 = float(jnp.max(jnp.abs(g2 - expected2)))
    assert err2 < 1e-12, f"MSE 2D err={err2}"

    print("PASS: MSE output gradient")


# ---- Test 7: Activation derivative ----

def test_activation_deriv():
    """Verify _activation_deriv against finite differences."""
    z = jnp.linspace(-3.0, 3.0, 100)
    eps = 1e-6

    for act_fn, name in [(jax.nn.gelu, "gelu"), (jax.nn.relu, "relu")]:
        deriv = _activation_deriv(act_fn, z)
        fd = (act_fn(z + eps) - act_fn(z - eps)) / (2 * eps)
        err = float(jnp.max(jnp.abs(deriv - fd)))
        print(f"[activation deriv] {name}: max_err={err:.2e}")
        assert err < 1e-5, f"{name} deriv err={err}"

    print("PASS: activation derivative")


# ---- Test 8: Determinism ----

def test_determinism():
    """Two identical calls must produce bit-for-bit identical output."""
    key = jax.random.PRNGKey(4)
    params = _init_mlp_params(key, 8, 4, 3)
    x = jax.random.normal(key, (16, 8))
    y = jax.random.randint(key, (16,), 0, 3)

    h1 = gn_diag_mlp(params, x, y, jax.nn.gelu, cross_entropy_output_grad)
    h2 = gn_diag_mlp(params, x, y, jax.nn.gelu, cross_entropy_output_grad)

    for layer in ('dense1', 'dense2', 'dense3'):
        for field in ('kernel', 'bias'):
            diff = float(jnp.max(jnp.abs(h1['params'][layer][field] -
                                          h2['params'][layer][field])))
            assert diff == 0.0, f"{layer}/{field}: diff={diff}"

    print("PASS: determinism")


# ---- Test 9: Optimizer integration ----

def test_optimizer_integration():
    """Plug GN diagonal into derived_newton_diag and verify finite updates."""
    key = jax.random.PRNGKey(5)
    params = _init_mlp_params(key, 8, 4, 3)
    x = jax.random.normal(key, (16, 8))
    y = jax.random.randint(key, (16,), 0, 3)

    h_diag = gn_diag_mlp(params, x, y, jax.nn.gelu, cross_entropy_output_grad)

    def loss_fn(p):
        logits = _mlp_forward(p, x, jax.nn.gelu)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    grads = jax.grad(loss_fn)(params)

    opt = derived_newton_diag(
        learning_rate=0.01, momentum=0.9, beta_s=0.1,
        weight_decay=0.0, metric_clip=4.0,
    )
    state = opt.init(params)
    updates, new_state = opt.update(grads, state, h_diag, params=params)

    for layer in ('dense1', 'dense2', 'dense3'):
        for field in ('kernel', 'bias'):
            u = updates['params'][layer][field]
            assert jnp.all(jnp.isfinite(u)), (
                f"{layer}/{field}: non-finite updates")

    print("PASS: optimizer integration")


# ---- Test 10: loss_grad_and_gn_diag consistency ----

def test_fused_consistency():
    """loss_grad_and_gn_diag must match separate value_and_grad + gn_diag_mlp."""
    key = jax.random.PRNGKey(6)
    params = _init_mlp_params(key, 8, 4, 3)
    x = jax.random.normal(key, (16, 8))
    y = jax.random.randint(key, (16,), 0, 3)

    def loss_fn(p):
        logits = _mlp_forward(p, x, jax.nn.gelu)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    # Fused
    loss_f, grads_f, h_diag_f = loss_grad_and_gn_diag(
        params, x, y, loss_fn, jax.nn.gelu, cross_entropy_output_grad)

    # Separate
    loss_s, grads_s = jax.value_and_grad(loss_fn)(params)
    h_diag_s = gn_diag_mlp(params, x, y, jax.nn.gelu,
                            cross_entropy_output_grad)

    # Compare loss
    assert abs(float(loss_f) - float(loss_s)) < 1e-12, "loss mismatch"

    # Compare grads
    for layer in ('dense1', 'dense2', 'dense3'):
        for field in ('kernel', 'bias'):
            diff_g = float(jnp.max(jnp.abs(
                grads_f['params'][layer][field] - grads_s['params'][layer][field])))
            diff_h = float(jnp.max(jnp.abs(
                h_diag_f['params'][layer][field] - h_diag_s['params'][layer][field])))
            assert diff_g < 1e-12, f"grad {layer}/{field}: diff={diff_g}"
            assert diff_h < 1e-12, f"h_diag {layer}/{field}: diff={diff_h}"

    print("PASS: fused consistency")


# ---- Test 11: MSE single-layer analytic ----

def test_mse_single_layer():
    """For a 1-layer linear net with MSE, verify GN diagonal analytically."""
    key = jax.random.PRNGKey(7)
    k1, k2, k3 = jax.random.split(key, 3)
    in_dim, out_dim = 4, 1
    B = 8

    params = {'params': {
        'dense1': {
            'kernel': jax.random.normal(k1, (in_dim, out_dim)) * 0.1,
            'bias': jax.random.normal(k2, (out_dim,)) * 0.01,
        },
    }}
    x = jax.random.normal(k3, (B, in_dim))
    y = jax.random.normal(key, (B,))

    h_diag = gn_diag_mlp(params, x, y, jax.nn.gelu,
                          mse_output_grad,
                          layer_keys=('dense1',))

    # Analytic
    p = params['params']
    logits = x @ p['dense1']['kernel'] + p['dense1']['bias']
    g = 2.0 * (logits.squeeze(-1) - y)[:, None]  # (B, 1)
    expected_kernel = (x ** 2).T @ (g ** 2) / B  # (in_dim, 1)
    expected_bias = jnp.mean(g ** 2, axis=0)      # (1,)

    err_k = float(jnp.max(jnp.abs(h_diag['params']['dense1']['kernel'] - expected_kernel)))
    err_b = float(jnp.max(jnp.abs(h_diag['params']['dense1']['bias'] - expected_bias)))
    print(f"[MSE single-layer] kernel err={err_k:.2e}, bias err={err_b:.2e}")
    assert err_k < 1e-12, f"kernel err={err_k}"
    assert err_b < 1e-12, f"bias err={err_b}"
    print("PASS: MSE single-layer analytic")


# ---- Test 12: Multi-step convergence with GN diagonal ----

def test_multistep_convergence():
    """Run optimizer for several steps with GN diagonal, verify loss decreases."""
    key = jax.random.PRNGKey(8)
    in_dim, hidden, out_dim = 8, 4, 3
    B = 32
    params = _init_mlp_params(key, in_dim, hidden, out_dim)
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (B, in_dim))
    y = jax.random.randint(k2, (B,), 0, out_dim)

    def loss_fn(p):
        logits = _mlp_forward(p, x, jax.nn.gelu)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    opt = derived_newton_diag(
        learning_rate=0.01, momentum=0.9, beta_s=0.1, metric_clip=4.0)
    state = opt.init(params)

    initial_loss = float(loss_fn(params))
    for _ in range(50):
        grads = jax.grad(loss_fn)(params)
        h_diag = gn_diag_mlp(params, x, y, jax.nn.gelu,
                              cross_entropy_output_grad)
        updates, state = opt.update(grads, state, h_diag, params=params)
        params = optax.apply_updates(params, updates)

    final_loss = float(loss_fn(params))
    print(f"[multi-step] initial={initial_loss:.4f}, final={final_loss:.4f}")
    assert final_loss < initial_loss, (
        f"Loss should decrease: {initial_loss} -> {final_loss}")
    print("PASS: multi-step convergence")


# ---- Test 13: Multi-layer MSE vs brute-force ----

def test_mse_multilayer_vs_bruteforce():
    """Compare multi-layer GN diagonal with MSE against brute-force."""
    key = jax.random.PRNGKey(10)
    in_dim, hidden, out_dim = 4, 6, 1
    B = 16
    params = _init_mlp_params(key, in_dim, hidden, out_dim)
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (B, in_dim))
    y = jax.random.normal(k2, (B,))

    h_diag = gn_diag_mlp(params, x, y, jax.nn.gelu, mse_output_grad)

    def per_sample_loss(params, xi, yi):
        logits = _mlp_forward(params, xi[None, :], jax.nn.gelu)
        return (logits.squeeze() - yi) ** 2

    per_sample_grads = jax.vmap(
        jax.grad(per_sample_loss), in_axes=(None, 0, 0)
    )(params, x, y)
    brute_force = jax.tree.map(
        lambda g: jnp.mean(g ** 2, axis=0), per_sample_grads
    )

    for layer in ('dense1', 'dense2', 'dense3'):
        for field in ('kernel', 'bias'):
            gn_val = h_diag['params'][layer][field]
            bf_val = brute_force['params'][layer][field]
            rel_err = float(jnp.max(jnp.abs(gn_val - bf_val) /
                                     jnp.maximum(jnp.abs(bf_val), 1e-10)))
            print(f"[MSE multi-layer] {layer}/{field}: max_rel_err={rel_err:.2e}")
            assert rel_err < 1e-4, (
                f"{layer}/{field}: rel_err={rel_err} too large")

    print("PASS: MSE multi-layer vs brute-force")


if __name__ == "__main__":
    print("=" * 60)
    print("GN diagonal functional tests")
    print("=" * 60)
    print()
    test_shape_match()
    print()
    test_positive_semidefinite()
    print()
    test_single_layer_analytic()
    print()
    test_multilayer_vs_bruteforce()
    print()
    test_cross_entropy_grad()
    print()
    test_mse_grad()
    print()
    test_activation_deriv()
    print()
    test_determinism()
    print()
    test_optimizer_integration()
    print()
    test_fused_consistency()
    print()
    test_mse_single_layer()
    print()
    test_multistep_convergence()
    print()
    test_mse_multilayer_vs_bruteforce()
    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
