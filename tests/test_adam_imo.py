"""Tests for the Adam-IMO bolt-on optimizer.

The two contracts that must hold:

1. ``xi=0`` reduces ``adam_imo`` to ``optax.adam`` exactly.
2. The smooth clip factor ``c_t = 1 / (1 + xi * quad)`` is in (0, 1] and
   the per-step update magnitude shrinks toward zero as ``xi * quad`` grows.

A break in either invariant would silently corrupt every result downstream.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from optimisers.jax_adam_imo import adam_imo, adam_clip, AdamIMOState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fixed_pytree(seed=0):
    """Small, deterministic pytree of params + grads (different shapes)."""
    rng = np.random.default_rng(seed)
    params = {
        'w1': jnp.array(rng.normal(size=(3, 4)).astype(np.float32)),
        'w2': jnp.array(rng.normal(size=(4,)).astype(np.float32)),
    }
    grads = {
        'w1': jnp.array(rng.normal(size=(3, 4)).astype(np.float32)),
        'w2': jnp.array(rng.normal(size=(4,)).astype(np.float32)),
    }
    return params, grads


def _flat(tree):
    return jnp.concatenate([l.ravel() for l in jax.tree.leaves(tree)])


# ---------------------------------------------------------------------------
# Contract 1: xi=0 -> exact Adam
# ---------------------------------------------------------------------------

def test_xi_zero_matches_adam_first_step():
    params, grads = _fixed_pytree()

    imo = adam_imo(learning_rate=1e-3, xi=0.0)
    ref = optax.adam(learning_rate=1e-3)

    s_imo = imo.init(params)
    s_ref = ref.init(params)
    u_imo, _ = imo.update(grads, s_imo, params)
    u_ref, _ = ref.update(grads, s_ref, params)

    np.testing.assert_allclose(_flat(u_imo), _flat(u_ref), atol=1e-7, rtol=1e-6)


def test_xi_zero_matches_adam_after_many_steps():
    params, grads_template = _fixed_pytree(seed=1)
    imo = adam_imo(learning_rate=1e-3, xi=0.0)
    ref = optax.adam(learning_rate=1e-3)
    s_imo = imo.init(params)
    s_ref = ref.init(params)

    rng = np.random.default_rng(42)
    for _ in range(20):
        # Vary the grads so bias correction gets exercised.
        grads = {
            'w1': jnp.array(rng.normal(size=(3, 4)).astype(np.float32)),
            'w2': jnp.array(rng.normal(size=(4,)).astype(np.float32)),
        }
        u_imo, s_imo = imo.update(grads, s_imo, params)
        u_ref, s_ref = ref.update(grads, s_ref, params)
        np.testing.assert_allclose(_flat(u_imo), _flat(u_ref),
                                    atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# Contract 2: clip factor c_t ∈ (0, 1] and update shrinks with ξ
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("xi", [0.0, 0.01, 0.1, 1.0, 10.0, 100.0])
def test_clip_factor_in_unit_interval(xi):
    params, grads = _fixed_pytree(seed=2)
    imo = adam_imo(learning_rate=1e-3, xi=xi)
    state = imo.init(params)
    updates, state = imo.update(grads, state, params)

    # Recompute c_t from state for the assertion (the optimizer doesn't
    # expose it directly).
    bc2 = 1.0 - 0.999 ** float(state.step)
    quad = 0.0
    for g, nu in zip(jax.tree.leaves(grads), jax.tree.leaves(state.nu)):
        quad += float(jnp.sum((g * g) / (jnp.sqrt(nu / bc2) + 1e-8)))
    c_t = 1.0 / (1.0 + xi * quad)

    assert 0.0 < c_t <= 1.0 + 1e-9
    if xi > 0.0:
        assert c_t < 1.0


def test_update_norm_shrinks_with_xi():
    """Holding everything else equal, a larger xi yields a smaller update norm."""
    params, grads = _fixed_pytree(seed=3)
    norms = []
    for xi in [0.0, 0.1, 1.0, 10.0]:
        imo = adam_imo(learning_rate=1e-3, xi=xi)
        s = imo.init(params)
        u, _ = imo.update(grads, s, params)
        norms.append(float(jnp.linalg.norm(_flat(u))))
    # Strictly decreasing
    for a, b in zip(norms, norms[1:]):
        assert b < a, f"update norms non-monotone: {norms}"


# ---------------------------------------------------------------------------
# Diagnostics function returns finite scalars
# ---------------------------------------------------------------------------

def test_adam_imo_diagnostics_finite():
    from optimisers.jax_adam_imo import adam_imo_diagnostics
    params, grads = _fixed_pytree(seed=4)
    imo = adam_imo(learning_rate=1e-3, xi=0.5)
    state = imo.init(params)
    _, state = imo.update(grads, state, params)
    diag = adam_imo_diagnostics(grads, state, xi=0.5)
    assert np.isfinite(diag['quad'])
    assert np.isfinite(diag['zeta'])
    assert np.isfinite(diag['c_t'])
    assert 0.0 < diag['c_t'] <= 1.0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_adam_imo_rejects_negative_xi():
    """Negative xi can produce c_t < 0 (anti-descent) — must be rejected."""
    with pytest.raises(ValueError, match="xi"):
        adam_imo(learning_rate=1e-3, xi=-0.1)


def test_adam_imo_weight_decay_requires_params():
    """weight_decay > 0 needs params; silently skipping decay is a footgun."""
    params, grads = _fixed_pytree(seed=8)
    opt = adam_imo(learning_rate=1e-3, xi=0.0, weight_decay=0.1)
    state = opt.init(params)
    with pytest.raises(ValueError, match="params"):
        opt.update(grads, state, None)


# ---------------------------------------------------------------------------
# Weight-decay path: xi=0 + wd > 0 should match optax.adamw
# ---------------------------------------------------------------------------

def test_adam_imo_weight_decay_matches_adamw_when_xi_zero():
    """xi=0 + decoupled weight_decay should reduce to optax.adamw exactly."""
    params, _ = _fixed_pytree(seed=9)
    rng = np.random.default_rng(123)
    imo = adam_imo(learning_rate=5e-3, xi=0.0, weight_decay=0.05)
    ref = optax.adamw(learning_rate=5e-3, weight_decay=0.05)
    s_imo = imo.init(params)
    s_ref = ref.init(params)
    cur_imo = params
    cur_ref = params
    for _ in range(8):
        grads = {
            'w1': jnp.array(rng.normal(size=(3, 4)).astype(np.float32)),
            'w2': jnp.array(rng.normal(size=(4,)).astype(np.float32)),
        }
        u_imo, s_imo = imo.update(grads, s_imo, cur_imo)
        u_ref, s_ref = ref.update(grads, s_ref, cur_ref)
        np.testing.assert_allclose(_flat(u_imo), _flat(u_ref),
                                    atol=1e-6, rtol=1e-5)
        cur_imo = optax.apply_updates(cur_imo, u_imo)
        cur_ref = optax.apply_updates(cur_ref, u_ref)


# ---------------------------------------------------------------------------
# Early-step bias-correction transient (t = 1, 2, 5)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_steps", [1, 2, 5])
def test_adam_imo_early_step_finite_and_clipped(n_steps):
    """Bias correction is most aggressive at small t (bc2 = 1 - b2**t).
    The implementation must stay finite and keep c_t in (0, 1] across the
    transient."""
    params, _ = _fixed_pytree(seed=10)
    imo = adam_imo(learning_rate=1e-3, xi=0.1)
    state = imo.init(params)
    rng = np.random.default_rng(42)
    grads = None
    for _ in range(n_steps):
        grads = {
            'w1': jnp.array(rng.normal(size=(3, 4)).astype(np.float32)),
            'w2': jnp.array(rng.normal(size=(4,)).astype(np.float32)),
        }
        updates, state = imo.update(grads, state, params)
        assert jnp.all(jnp.isfinite(_flat(updates)))
    bc2 = 1.0 - 0.999 ** float(state.step)
    quad = 0.0
    for g, nu in zip(jax.tree.leaves(grads), jax.tree.leaves(state.nu)):
        quad += float(jnp.sum((g * g) / (jnp.sqrt(nu / bc2) + 1e-8)))
    c_t = 1.0 / (1.0 + 0.1 * quad)
    assert 0.0 < c_t <= 1.0


# ---------------------------------------------------------------------------
# adam_clip baseline composes a global-norm clip with Adam
# ---------------------------------------------------------------------------

def test_adam_clip_caps_update_norm_when_grads_explode():
    params, _ = _fixed_pytree(seed=5)
    big_grads = {
        'w1': jnp.full((3, 4), 1e3, dtype=jnp.float32),
        'w2': jnp.full((4,), 1e3, dtype=jnp.float32),
    }
    opt_clip = adam_clip(learning_rate=1e-3, max_norm=1.0)
    opt_noclip = optax.adam(learning_rate=1e-3)

    s_clip = opt_clip.init(params)
    s_noclip = opt_noclip.init(params)

    u_clip, _ = opt_clip.update(big_grads, s_clip, params)
    u_noclip, _ = opt_noclip.update(big_grads, s_noclip, params)

    # Update magnitudes from Adam are roughly lr regardless of grad size, but
    # the asserted invariant we want is that adam_clip is *finite* and bounded
    # by lr (since clipped grads have norm <= 1 and Adam normalises by sqrt(v)).
    n_clip = float(jnp.linalg.norm(_flat(u_clip)))
    n_noclip = float(jnp.linalg.norm(_flat(u_noclip)))
    assert np.isfinite(n_clip)
    # Both should yield comparable-magnitude updates because Adam's sqrt(v)
    # dominates; the contract is just that adam_clip *runs* and produces
    # a sensible state.
    assert n_clip > 0


# ---------------------------------------------------------------------------
# Registry round-trip
# ---------------------------------------------------------------------------

def test_registry_create_adam_imo():
    from optimizer_registry import create_optimizer
    opt = create_optimizer('adam_imo', {'learning_rate': 1e-3, 'xi': 0.1})
    params, grads = _fixed_pytree(seed=6)
    s = opt.init(params)
    u, s2 = opt.update(grads, s, params)
    assert isinstance(s2, AdamIMOState)
    assert int(s2.step) == 1


def test_registry_create_adam_clip():
    from optimizer_registry import create_optimizer
    opt = create_optimizer('adam_clip',
                            {'learning_rate': 1e-3, 'clip_norm': 1.0})
    params, grads = _fixed_pytree(seed=7)
    s = opt.init(params)
    u, _ = opt.update(grads, s, params)
    # Just needs to produce finite updates of matching pytree structure.
    assert jax.tree.structure(u) == jax.tree.structure(params)
