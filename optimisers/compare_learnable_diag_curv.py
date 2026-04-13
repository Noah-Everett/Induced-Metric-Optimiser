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


CURV_HPARAMS = dict(curv_beta=0.05, curv_tau=1.0, secant_eps=1e-5)


def _make_data(n_steps):
    np.random.seed(42)
    w1 = np.random.randn(4, 3).astype(np.float32)
    b1 = np.random.randn(3).astype(np.float32)
    rng = np.random.RandomState(123)
    grads_w1 = [rng.randn(4, 3).astype(np.float32) * 0.1 for _ in range(n_steps)]
    grads_b1 = [rng.randn(3).astype(np.float32) * 0.1 for _ in range(n_steps)]
    return w1, b1, grads_w1, grads_b1


def test_jax_torch_parity(hparams, n_steps, label=""):
    """Check that JAX and PyTorch curvature-aware variants produce the same params."""
    w1, b1, grads_w1, grads_b1 = _make_data(n_steps)

    # ---- JAX
    params_j = {"w1": jnp.array(w1), "b1": jnp.array(b1)}
    opt_j = custom_sgd_learnable_diag_curv(**hparams, **CURV_HPARAMS)
    st_j = opt_j.init(params_j)
    for i in range(n_steps):
        grads_j = {"w1": jnp.array(grads_w1[i]), "b1": jnp.array(grads_b1[i])}
        updates_j, st_j = opt_j.update(grads_j, st_j, params=params_j)
        params_j = optax.apply_updates(params_j, updates_j)

    # ---- Torch
    w1_t = torch.tensor(w1.copy(), requires_grad=True)
    b1_t = torch.tensor(b1.copy(), requires_grad=True)
    opt_t = SGDLearnableDiagCurv(
        [w1_t, b1_t],
        lr=hparams['learning_rate'], momentum=hparams['momentum'],
        xi=hparams['xi'], beta=hparams['beta'],
        weight_decay=hparams['weight_decay'],
        metric_lr=hparams['metric_lr'], metric_reg=hparams['metric_reg'],
        metric_clip=hparams['metric_clip'],
        **CURV_HPARAMS,
    )
    for i in range(n_steps):
        with torch.no_grad():
            w1_t.grad = torch.tensor(grads_w1[i])
            b1_t.grad = torch.tensor(grads_b1[i])
        opt_t.step()

    # Compare
    j_w1 = np.array(params_j["w1"])
    j_b1 = np.array(params_j["b1"])
    diff_w1 = float(np.max(np.abs(j_w1 - w1_t.detach().numpy())))
    diff_b1 = float(np.max(np.abs(j_b1 - b1_t.detach().numpy())))
    print(f"[{label}] After {n_steps} steps:")
    print(f"  max|diff w1|: {diff_w1:.2e}")
    print(f"  max|diff b1|: {diff_b1:.2e}")
    tol = 1e-5
    assert diff_w1 < tol, f"w1 parity failed ({label}): {diff_w1}"
    assert diff_b1 < tol, f"b1 parity failed ({label}): {diff_b1}"
    print(f"  PASS (tol={tol})")


def test_curv_beta_zero_recovers_base():
    """Check that curv_beta=0 gives the same result as the base diagonal learnable."""
    hparams = dict(
        learning_rate=0.01, momentum=0.9, xi=0.1, beta=0.8,
        weight_decay=0.01, metric_lr=1e-3, metric_reg=1e-4, metric_clip=4.0,
    )
    n_steps = 25
    w1, b1, grads_w1, grads_b1 = _make_data(n_steps)

    # ---- Base diagonal learnable (JAX)
    params_base = {"w1": jnp.array(w1), "b1": jnp.array(b1)}
    opt_base = custom_sgd_learnable_diag(**hparams)
    st_base = opt_base.init(params_base)
    for i in range(n_steps):
        grads = {"w1": jnp.array(grads_w1[i]), "b1": jnp.array(grads_b1[i])}
        updates_base, st_base = opt_base.update(grads, st_base, params=params_base)
        params_base = optax.apply_updates(params_base, updates_base)

    # ---- Curvature-aware with curv_beta=0 (JAX)
    params_curv = {"w1": jnp.array(w1), "b1": jnp.array(b1)}
    opt_curv = custom_sgd_learnable_diag_curv(**hparams, curv_beta=0.0)
    st_curv = opt_curv.init(params_curv)
    for i in range(n_steps):
        grads = {"w1": jnp.array(grads_w1[i]), "b1": jnp.array(grads_b1[i])}
        updates_curv, st_curv = opt_curv.update(grads, st_curv, params=params_curv)
        params_curv = optax.apply_updates(params_curv, updates_curv)

    # Compare
    diff_w1 = float(np.max(np.abs(np.array(params_base["w1"]) - np.array(params_curv["w1"]))))
    diff_b1 = float(np.max(np.abs(np.array(params_base["b1"]) - np.array(params_curv["b1"]))))
    print(f"[curv_beta=0 vs base] max|diff w1|: {diff_w1:.2e}")
    print(f"[curv_beta=0 vs base] max|diff b1|: {diff_b1:.2e}")
    assert diff_w1 < 1e-6, f"w1 recovery failed: {diff_w1}"
    assert diff_b1 < 1e-6, f"b1 recovery failed: {diff_b1}"
    print("PASS: curv_beta=0 recovers base diagonal learnable")


def main():
    hparams_std = dict(
        learning_rate=0.01, momentum=0.9, xi=0.1, beta=0.8,
        weight_decay=0.01, metric_lr=1e-3, metric_reg=1e-4, metric_clip=4.0,
    )
    test_jax_torch_parity(hparams_std, n_steps=25, label="standard")

    hparams_hiwd = dict(
        learning_rate=0.1, momentum=0.9, xi=0.1, beta=0.8,
        weight_decay=0.1, metric_lr=1e-3, metric_reg=1e-4, metric_clip=4.0,
    )
    test_jax_torch_parity(hparams_hiwd, n_steps=25, label="high-wd")

    print()
    test_curv_beta_zero_recovers_base()


if __name__ == "__main__":
    main()
