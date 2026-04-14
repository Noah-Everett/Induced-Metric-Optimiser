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
    loss_grad_and_hvp_diag,
    _group_leaves,
    DerivedNewtonDiagState,
)


def run_optimizer_hvp(opt, params_init, grad_fn, h_diag_fn, n_steps,
                      needs_params=True):
    """Run optimizer for n_steps with externally provided h_diag."""
    params = params_init.copy()
    state = opt.init(params)
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
        learning_rate=0.003, momentum=0.9, beta_s=0.05,
        weight_decay=0.0, metric_clip=4.0, hess_eps=1e-6,
    )
    params = p0.copy()
    state = opt.init(params)
    key = jax.random.PRNGKey(0)
    n_steps = 5000

    initial_loss = float(loss_fn(params))
    for i in range(n_steps):
        key, subkey = jax.random.split(key)
        grads = jax.grad(loss_fn)(params)
        h_diag = hutchinson_hvp_diag(loss_fn, params, subkey)
        updates, state = opt.update(grads, state, h_diag, params=params)
        params = optax.apply_updates(params, updates)

    final_loss = float(loss_fn(params))
    print(f"[Rosenbrock] initial={initial_loss:.4f}, final={final_loss:.6f}")
    assert final_loss < 0.1, (
        f"Should converge near optimum: final={final_loss}"
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


# ---- Test 8: _group_leaves correctness ----

def test_group_leaves():
    """Verify _group_leaves partitions leaves correctly."""
    # Single leaf -> single group
    leaves_1 = [jnp.zeros(10)]
    assert _group_leaves(leaves_1) == [[0]], "single leaf should be one group"

    # All large -> each its own group
    leaves_large = [jnp.zeros(2000), jnp.zeros(3000), jnp.zeros(5000)]
    groups = _group_leaves(leaves_large)
    assert len(groups) == 3
    assert all(len(g) == 1 for g in groups)
    print(f"[group_leaves] all-large: {groups}")

    # All small -> bundled
    leaves_small = [jnp.zeros(10), jnp.zeros(20), jnp.zeros(30)]
    groups = _group_leaves(leaves_small)
    assert len(groups) == 1
    assert sorted(groups[0]) == [0, 1, 2]
    print(f"[group_leaves] all-small: {groups}")

    # Mixed -> large separate, small bundled
    leaves_mixed = [jnp.zeros(5000), jnp.zeros(10), jnp.zeros(3000), jnp.zeros(20)]
    groups = _group_leaves(leaves_mixed)
    large_g = [g for g in groups if any(leaves_mixed[i].size > 1024 for i in g)]
    small_g = [g for g in groups if all(leaves_mixed[i].size <= 1024 for i in g)]
    assert len(large_g) == 2  # indices 0 and 2
    assert len(small_g) == 1  # indices 1 and 3
    print(f"[group_leaves] mixed: {groups}")

    print("PASS: group_leaves")


# ---- Test 9: Per-leaf Hutchinson on multi-group pytree ----

def test_per_leaf_hutchinson_multi_group():
    """Per-leaf Hutchinson with leaves in separate groups should match true diagonal.

    Uses leaves > _SMALL_LEAF_THRESHOLD (1024) to ensure they get separate groups,
    and a non-separable loss to verify cross-leaf noise is eliminated.
    """
    # L = ||w||^2 + 0.5*||u||^2 + w.sum()*u.sum()
    # H_ww = 2*I, H_uu = I, H_wu = ones (dense cross-block)
    # True diagonal: w -> 2.0, u -> 1.0
    # Per-leaf zeroes the cross-block, so each sample is exact.
    n_w, n_u = 2000, 1500  # both > 1024 -> separate groups
    def loss_fn(p):
        return jnp.sum(p["w"] ** 2) + 0.5 * jnp.sum(p["u"] ** 2) + p["w"].sum() * p["u"].sum()

    params = {"u": jnp.ones(n_u), "w": jnp.ones(n_w)}
    true_diag = {"u": jnp.ones(n_u), "w": 2.0 * jnp.ones(n_w)}

    n_samples = 100
    key = jax.random.PRNGKey(42)
    h_sum = jax.tree.map(jnp.zeros_like, params)
    for _ in range(n_samples):
        key, subkey = jax.random.split(key)
        h = hutchinson_hvp_diag(loss_fn, params, subkey)
        h_sum = jax.tree.map(lambda a, b: a + b, h_sum, h)
    h_mean = jax.tree.map(lambda s: s / n_samples, h_sum)

    for k in params:
        err = float(jnp.max(jnp.abs(h_mean[k] - true_diag[k])))
        print(f"[multi-group Hutchinson] {k}: max_err={err:.2e}")
        # Diagonal H within each leaf -> exact per sample -> tight tolerance
        assert err < 1e-6, f"Leaf '{k}' mean too far from truth: err={err}"

    print("PASS: per-leaf Hutchinson multi-group")


# ---- Test 10: Per-leaf matches global for single-leaf params ----

def test_per_leaf_matches_global_single_leaf():
    """For a single flat array, per-leaf should produce identical results."""
    loss_fn = lambda p: 3.0 * p[0]**2 + 7.0 * p[1]**2
    params = jnp.array([1.0, 1.0])
    key = jax.random.PRNGKey(99)

    h = hutchinson_hvp_diag(loss_fn, params, key)
    # With single leaf, per-leaf == global. Verify it's a valid estimate.
    # For diagonal quadratic, each sample is exact.
    true_diag = jnp.array([6.0, 14.0])
    err = float(jnp.max(jnp.abs(h - true_diag)))
    print(f"[single-leaf match] h={h}, err={err:.2e}")
    assert err < 1e-10, f"Single leaf should be exact on diagonal quadratic: err={err}"
    print("PASS: per-leaf matches global single-leaf")


# ---- Test 11: loss_grad_and_hvp_diag with multiple groups ----

def test_loss_grad_hvp_multi_group():
    """loss_grad_and_hvp_diag with 2+ groups returns correct loss, grads, and h_diag.

    Ensures the groups[1:] codepath (grad-only JVP) is exercised.
    """
    a, b = 5.0, 2.0
    n_w, n_v = 2000, 1500  # both > 1024 -> separate groups
    def loss_fn(p):
        return a * jnp.sum(p["w"] ** 2) + b * jnp.sum(p["v"] ** 2)

    params = {"v": jnp.ones(n_v), "w": jnp.ones(n_w)}
    key = jax.random.PRNGKey(7)

    loss, grads, h_diag = loss_grad_and_hvp_diag(loss_fn, params, key)

    # Check loss
    expected_loss = float(a * n_w + b * n_v)
    print(f"[fused multi-group] loss={float(loss):.4f}, expected={expected_loss:.4f}")
    assert abs(float(loss) - expected_loss) < 1e-3

    # Check grads
    assert float(jnp.max(jnp.abs(grads["w"] - 2.0 * a))) < 1e-6
    assert float(jnp.max(jnp.abs(grads["v"] - 2.0 * b))) < 1e-6

    # Check h_diag (separable diagonal quadratic -> exact per sample)
    err_w = float(jnp.max(jnp.abs(h_diag["w"] - 2.0 * a)))
    err_v = float(jnp.max(jnp.abs(h_diag["v"] - 2.0 * b)))
    print(f"[fused multi-group] h_diag err: w={err_w:.2e}, v={err_v:.2e}")
    assert err_w < 1e-6 and err_v < 1e-6

    print("PASS: loss_grad_and_hvp_diag multi-group")


# ---- Test 12: Fused vs separate consistency ----

def test_fused_matches_separate():
    """loss_grad_and_hvp_diag h_diag must match hutchinson_hvp_diag for same key."""
    n_w, n_b = 2000, 1500
    def loss_fn(p):
        return jnp.sum(p["w"] ** 2) + 0.5 * jnp.sum(p["b"] ** 2) + p["w"].sum() * p["b"].sum()

    params = {"b": jnp.ones(n_b), "w": jnp.ones(n_w)}
    key = jax.random.PRNGKey(123)

    h_separate = hutchinson_hvp_diag(loss_fn, params, key)
    _, _, h_fused = loss_grad_and_hvp_diag(loss_fn, params, key)

    for k in params:
        diff = float(jnp.max(jnp.abs(h_separate[k] - h_fused[k])))
        print(f"[fused vs separate] {k}: max_diff={diff:.2e}")
        assert diff < 1e-10, f"h_diag should be identical for same key: {k} diff={diff}"

    print("PASS: fused matches separate")


# ---- Test 13: Variance reduction vs global baseline ----

def test_variance_reduction():
    """Per-leaf Hutchinson should have lower variance than global with cross-leaf coupling."""
    # L = ||w||^2 + 0.5*||b||^2 + w.sum()*b.sum()
    # H_ww = 2*I, H_bb = I, H_wb = ones (dense cross-block)
    # Per-leaf: H_ww diagonal -> var = 0
    # Global: cross-leaf off-diag ones contribute massive noise
    n_w, n_b = 2000, 1500

    def loss_fn(p):
        return jnp.sum(p["w"] ** 2) + 0.5 * jnp.sum(p["b"] ** 2) + p["w"].sum() * p["b"].sum()

    params = {"b": jnp.ones(n_b), "w": jnp.ones(n_w)}
    key = jax.random.PRNGKey(42)

    n_samples = 200
    h_samples_w = []
    for i in range(n_samples):
        key, subkey = jax.random.split(key)
        h = hutchinson_hvp_diag(loss_fn, params, subkey)
        h_samples_w.append(h["w"][0])

    h_arr = jnp.array(h_samples_w)
    per_leaf_var = float(jnp.var(h_arr))
    h_mean = float(jnp.mean(h_arr))

    print(f"[variance reduction] per-leaf: mean={h_mean:.4f}, var={per_leaf_var:.6f}")
    assert abs(h_mean - 2.0) < 0.1, f"Mean should be ~2.0, got {h_mean}"
    assert per_leaf_var < 0.01, (
        f"Per-leaf variance should be ~0 for diagonal H_ww block: {per_leaf_var}"
    )

    # Compare against global Hutchinson (single-group baseline)
    from optimisers.jax_derived_newton_diag import _rademacher_like
    key2 = jax.random.PRNGKey(42)
    h_global_w = []
    grad_fn = jax.grad(loss_fn)
    for i in range(n_samples):
        key2, subkey = jax.random.split(key2)
        v = _rademacher_like(subkey, params)
        _, hvp = jax.jvp(grad_fn, (params,), (v,))
        h_global = jax.tree.map(lambda vi, hi: vi * hi, v, hvp)
        h_global_w.append(h_global["w"][0])

    global_arr = jnp.array(h_global_w)
    global_var = float(jnp.var(global_arr))
    print(f"[variance reduction] global:   mean={float(jnp.mean(global_arr)):.4f}, var={global_var:.4f}")
    assert global_var > 100 * per_leaf_var, (
        f"Global variance ({global_var}) should be >> per-leaf variance ({per_leaf_var})"
    )

    print("PASS: variance reduction")


# ---- Test 14: Three leaf groups ----

def test_three_groups():
    """Per-leaf Hutchinson with 3 separate groups (2 large + 1 small batch)."""
    n_w1, n_w2, n_b = 2000, 1500, 10  # w1, w2 separate; b small -> bundled
    def loss_fn(p):
        return (jnp.sum(p["w1"] ** 2) + 2.0 * jnp.sum(p["w2"] ** 2)
                + 0.5 * jnp.sum(p["b"] ** 2)
                + p["w1"].sum() * p["w2"].sum())  # cross-leaf coupling

    params = {"b": jnp.ones(n_b), "w1": jnp.ones(n_w1), "w2": jnp.ones(n_w2)}
    # True diag: w1 -> 2, w2 -> 4, b -> 1
    true_diag = {"b": jnp.ones(n_b), "w1": 2.0 * jnp.ones(n_w1), "w2": 4.0 * jnp.ones(n_w2)}

    # Verify grouping: should be 3 groups (b is small -> own group, w1 + w2 large)
    leaves, _ = jax.tree.flatten(params)
    groups = _group_leaves(leaves)
    print(f"[three groups] groups={groups} (sizes: {[leaves[g[0]].size for g in groups]})")
    assert len(groups) == 3, f"Expected 3 groups, got {len(groups)}"

    n_samples = 100
    key = jax.random.PRNGKey(77)
    h_sum = jax.tree.map(jnp.zeros_like, params)
    for _ in range(n_samples):
        key, subkey = jax.random.split(key)
        h = hutchinson_hvp_diag(loss_fn, params, subkey)
        h_sum = jax.tree.map(lambda a, b: a + b, h_sum, h)
    h_mean = jax.tree.map(lambda s: s / n_samples, h_sum)

    for k in params:
        err = float(jnp.max(jnp.abs(h_mean[k] - true_diag[k])))
        print(f"[three groups] {k}: max_err={err:.2e}")
        assert err < 1e-6, f"Leaf '{k}' mean too far from truth: err={err}"

    print("PASS: three groups")


# ---- Test 15: Nested pytree ----

def test_nested_pytree():
    """Per-leaf Hutchinson on a nested pytree preserves structure."""
    n = 1500  # > 1024 -> separate groups
    def loss_fn(p):
        return jnp.sum(p["l1"]["w"] ** 2) + 2.0 * jnp.sum(p["l2"]["w"] ** 2)

    params = {"l1": {"w": jnp.ones(n)}, "l2": {"w": jnp.ones(n)}}
    key = jax.random.PRNGKey(55)

    h = hutchinson_hvp_diag(loss_fn, params, key)
    # Diagonal quadratic -> exact
    err_l1 = float(jnp.max(jnp.abs(h["l1"]["w"] - 2.0)))
    err_l2 = float(jnp.max(jnp.abs(h["l2"]["w"] - 4.0)))
    print(f"[nested pytree] l1/w err={err_l1:.2e}, l2/w err={err_l2:.2e}")
    assert err_l1 < 1e-6 and err_l2 < 1e-6

    # Also test fused version
    loss, grads, h2 = loss_grad_and_hvp_diag(loss_fn, params, key)
    assert abs(float(loss) - 3.0 * n) < 1e-3
    err_l1_f = float(jnp.max(jnp.abs(h2["l1"]["w"] - 2.0)))
    err_l2_f = float(jnp.max(jnp.abs(h2["l2"]["w"] - 4.0)))
    assert err_l1_f < 1e-6 and err_l2_f < 1e-6

    print("PASS: nested pytree")


# ---- Test 16: xi=0 backward compatibility ----

def test_xi_zero_backward_compat():
    """Explicit xi=0.0 should produce identical results to the default (no xi arg)."""
    a, b = 5.0, 0.5
    grad_fn = jax.jit(jax.grad(lambda p: a * p[0]**2 + b * p[1]**2))
    h_diag_fn = lambda p: jnp.array([2.0 * a, 2.0 * b])
    p0 = jnp.array([1.0, 1.0])
    hparams = dict(learning_rate=0.01, momentum=0.9, beta_s=0.1, metric_clip=4.0)

    _, params_default = run_optimizer_hvp(
        derived_newton_diag(**hparams), p0, grad_fn, h_diag_fn, 50,
    )
    _, params_xi0 = run_optimizer_hvp(
        derived_newton_diag(**hparams, xi=0.0), p0, grad_fn, h_diag_fn, 50,
    )

    diff = float(jnp.max(jnp.abs(params_default - params_xi0)))
    print(f"[xi=0 compat] max diff = {diff:.2e}")
    assert diff < 1e-10, f"xi=0 should be identical to default, got diff={diff}"
    print("PASS: xi=0 backward compatibility")


# ---- Test 17: xi>0 denominator effect ----

def test_xi_denominator_effect():
    """With xi>0, updates should be smaller due to the denominator > 1.

    Setup: h_diag=[2,2], params=[1,1], s=[0,0] (step 0).
    E_w[lambda] = sum(h^2 * p^2 * exp(s)) / sum(h * p^2) = (4+4)/(2+2) = 2.
    With xi=2: r = 1/(1 + 1*2) = 1/3.
    So updates with xi=2 should be ~1/3 of updates with xi=0.
    """
    h_diag_fn = lambda p: jnp.array([2.0, 2.0])
    grad_fn = lambda p: jnp.array([0.1, 0.1])
    p0 = jnp.array([1.0, 1.0])
    hparams = dict(learning_rate=0.01, momentum=0.0, beta_s=0.0, metric_clip=4.0)

    opt_no_xi = derived_newton_diag(**hparams, xi=0.0)
    st0 = opt_no_xi.init(p0)
    updates_no_xi, _ = opt_no_xi.update(grad_fn(p0), st0, h_diag_fn(p0), params=p0)

    opt_xi = derived_newton_diag(**hparams, xi=2.0)
    st_xi = opt_xi.init(p0)
    updates_xi, _ = opt_xi.update(grad_fn(p0), st_xi, h_diag_fn(p0), params=p0)

    ratio = float(jnp.abs(updates_xi[0]) / jnp.abs(updates_no_xi[0]))
    expected_ratio = 1.0 / 3.0  # r = 1/(1 + (2/2)*2) = 1/3
    print(f"[xi denominator] ratio={ratio:.6f}, expected={expected_ratio:.6f}")
    assert abs(ratio - expected_ratio) < 1e-5, (
        f"Update ratio should be ~{expected_ratio}, got {ratio}"
    )
    print("PASS: xi denominator effect")


# ---- Test 18: xi>0 requires params ----

def test_xi_requires_params():
    """xi>0 without params should raise ValueError."""
    grad_fn = lambda p: jnp.array([0.1, 0.1])
    h_diag_fn = lambda p: jnp.array([2.0, 2.0])
    p0 = jnp.array([1.0, 1.0])

    opt = derived_newton_diag(learning_rate=0.01, momentum=0.0, beta_s=0.0, xi=0.5)
    state = opt.init(p0)
    try:
        opt.update(grad_fn(p0), state, h_diag_fn(p0))  # no params
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "params" in str(e)
    print("PASS: xi>0 requires params")


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
    test_group_leaves()
    print()
    test_per_leaf_hutchinson_multi_group()
    print()
    test_per_leaf_matches_global_single_leaf()
    print()
    test_loss_grad_hvp_multi_group()
    print()
    test_fused_matches_separate()
    print()
    test_variance_reduction()
    print()
    test_three_groups()
    print()
    test_nested_pytree()
    print()
    test_xi_zero_backward_compat()
    print()
    test_xi_denominator_effect()
    print()
    test_xi_requires_params()
    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
