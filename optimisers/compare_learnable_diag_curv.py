# optimisers/compare_learnable_diag_curv.py
"""
Parity check between JAX and PyTorch curvature-aware learnable-diagonal variants.

Runs two update steps (need at least 2 so the secant estimator activates on step 2)
with identical hyperparameters and initial states, then prints max abs diff.

Also verifies that curv_beta=0 recovers the same output as the base diagonal learnable.
"""

import numpy as np

# JAX side
import jax, jax.numpy as jnp
import optax
from optimisers.jax_learnable_diag_curv import custom_sgd_learnable_diag_curv
from optimisers.jax_learnable_diag import custom_sgd_learnable_diag

# Torch side
import torch
from optimisers.torch_learnable_diag_curv import SGDLearnableDiagCurv
from optimisers.torch_learnable_diag import SGDLearnableDiag


HPARAMS = dict(
    learning_rate=0.01, momentum=0.9, xi=0.1, beta=0.8,
    weight_decay=0.01, metric_lr=1e-3, metric_reg=1e-4, metric_clip=4.0,
)
CURV_HPARAMS = dict(curv_beta=0.05, curv_tau=1.0, secant_eps=1e-5)


def _make_data():
    np.random.seed(42)
    w1 = np.random.randn(4, 3).astype(np.float32)
    b1 = np.random.randn(3).astype(np.float32)
    g1_step1 = np.random.randn(4, 3).astype(np.float32) * 0.1
    g2_step1 = np.random.randn(3).astype(np.float32) * 0.1
    g1_step2 = np.random.randn(4, 3).astype(np.float32) * 0.1
    g2_step2 = np.random.randn(3).astype(np.float32) * 0.1
    return w1, b1, g1_step1, g2_step1, g1_step2, g2_step2


def test_jax_torch_parity():
    """Check that JAX and PyTorch curvature-aware variants produce the same params."""
    w1, b1, g1_s1, g2_s1, g1_s2, g2_s2 = _make_data()

    # ---- JAX: two steps
    params_j = {"w1": jnp.array(w1), "b1": jnp.array(b1)}
    opt_j = custom_sgd_learnable_diag_curv(**HPARAMS, **CURV_HPARAMS)
    st_j = opt_j.init(params_j)

    grads_s1 = {"w1": jnp.array(g1_s1), "b1": jnp.array(g2_s1)}
    updates_j, st_j = opt_j.update(grads_s1, st_j, params=params_j)
    params_j = optax.apply_updates(params_j, updates_j)

    grads_s2 = {"w1": jnp.array(g1_s2), "b1": jnp.array(g2_s2)}
    updates_j, st_j = opt_j.update(grads_s2, st_j, params=params_j)
    params_j = optax.apply_updates(params_j, updates_j)

    # ---- Torch: two steps
    w1_t = torch.tensor(w1.copy(), requires_grad=True)
    b1_t = torch.tensor(b1.copy(), requires_grad=True)
    opt_t = SGDLearnableDiagCurv(
        [w1_t, b1_t],
        lr=HPARAMS['learning_rate'], momentum=HPARAMS['momentum'],
        xi=HPARAMS['xi'], beta=HPARAMS['beta'],
        weight_decay=HPARAMS['weight_decay'],
        metric_lr=HPARAMS['metric_lr'], metric_reg=HPARAMS['metric_reg'],
        metric_clip=HPARAMS['metric_clip'],
        **CURV_HPARAMS,
    )
    # Step 1
    with torch.no_grad():
        w1_t.grad = torch.tensor(g1_s1)
        b1_t.grad = torch.tensor(g2_s1)
    opt_t.step()
    # Step 2
    with torch.no_grad():
        w1_t.grad = torch.tensor(g1_s2)
        b1_t.grad = torch.tensor(g2_s2)
    opt_t.step()

    # Compare
    j_w1 = np.array(params_j["w1"])
    j_b1 = np.array(params_j["b1"])
    diff_w1 = np.max(np.abs(j_w1 - w1_t.detach().numpy()))
    diff_b1 = np.max(np.abs(j_b1 - b1_t.detach().numpy()))
    print(f"[JAX vs Torch curv] max|diff w1|: {diff_w1:.2e}")
    print(f"[JAX vs Torch curv] max|diff b1|: {diff_b1:.2e}")
    assert diff_w1 < 1e-5, f"w1 parity failed: {diff_w1}"
    assert diff_b1 < 1e-5, f"b1 parity failed: {diff_b1}"
    print("PASS: JAX/Torch curvature-aware parity")


def test_curv_beta_zero_recovers_base():
    """Check that curv_beta=0 gives the same result as the base diagonal learnable."""
    w1, b1, g1_s1, g2_s1, g1_s2, g2_s2 = _make_data()

    # ---- Base diagonal learnable (JAX)
    params_base = {"w1": jnp.array(w1), "b1": jnp.array(b1)}
    opt_base = custom_sgd_learnable_diag(**HPARAMS)
    st_base = opt_base.init(params_base)

    grads_s1 = {"w1": jnp.array(g1_s1), "b1": jnp.array(g2_s1)}
    updates_base, st_base = opt_base.update(grads_s1, st_base, params=params_base)
    params_base = optax.apply_updates(params_base, updates_base)

    grads_s2 = {"w1": jnp.array(g1_s2), "b1": jnp.array(g2_s2)}
    updates_base, st_base = opt_base.update(grads_s2, st_base, params=params_base)
    params_base = optax.apply_updates(params_base, updates_base)

    # ---- Curvature-aware with curv_beta=0 (JAX)
    params_curv = {"w1": jnp.array(w1), "b1": jnp.array(b1)}
    opt_curv = custom_sgd_learnable_diag_curv(**HPARAMS, curv_beta=0.0)
    st_curv = opt_curv.init(params_curv)

    grads_s1 = {"w1": jnp.array(g1_s1), "b1": jnp.array(g2_s1)}
    updates_curv, st_curv = opt_curv.update(grads_s1, st_curv, params=params_curv)
    params_curv = optax.apply_updates(params_curv, updates_curv)

    grads_s2 = {"w1": jnp.array(g1_s2), "b1": jnp.array(g2_s2)}
    updates_curv, st_curv = opt_curv.update(grads_s2, st_curv, params=params_curv)
    params_curv = optax.apply_updates(params_curv, updates_curv)

    # Compare
    diff_w1 = np.max(np.abs(np.array(params_base["w1"]) - np.array(params_curv["w1"])))
    diff_b1 = np.max(np.abs(np.array(params_base["b1"]) - np.array(params_curv["b1"])))
    print(f"[curv_beta=0 vs base] max|diff w1|: {diff_w1:.2e}")
    print(f"[curv_beta=0 vs base] max|diff b1|: {diff_b1:.2e}")
    assert diff_w1 < 1e-6, f"w1 recovery failed: {diff_w1}"
    assert diff_b1 < 1e-6, f"b1 recovery failed: {diff_b1}"
    print("PASS: curv_beta=0 recovers base diagonal learnable")


def main():
    test_jax_torch_parity()
    print()
    test_curv_beta_zero_recovers_base()


if __name__ == "__main__":
    main()
