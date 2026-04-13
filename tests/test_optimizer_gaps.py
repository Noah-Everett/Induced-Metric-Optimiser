"""
Tests filling specific gaps from the codebase audit remediation:
- DerivedNewtonDiag __init__ validation
- p.grad is None on requires_grad=True param
- Log-loss parity (SGDLearnableDiag log_loss=True vs JAX)
- Learnable scalar parity with assertions
"""

import numpy as np
import pytest
import torch


# ---- Init validation ----

def test_init_validation_momentum_1():
    from optimisers.torch_derived_newton_diag import DerivedNewtonDiag
    w = torch.randn(3, requires_grad=True)
    with pytest.raises(ValueError, match="momentum"):
        DerivedNewtonDiag([w], momentum=1.0)


def test_init_validation_negative_lr():
    from optimisers.torch_derived_newton_diag import DerivedNewtonDiag
    w = torch.randn(3, requires_grad=True)
    with pytest.raises(ValueError, match="lr"):
        DerivedNewtonDiag([w], lr=-0.1)


def test_init_validation_negative_wd():
    from optimisers.torch_derived_newton_diag import DerivedNewtonDiag
    w = torch.randn(3, requires_grad=True)
    with pytest.raises(ValueError, match="weight_decay"):
        DerivedNewtonDiag([w], weight_decay=-0.01)


def test_init_validation_zero_hess_eps():
    from optimisers.torch_derived_newton_diag import DerivedNewtonDiag
    w = torch.randn(3, requires_grad=True)
    with pytest.raises(ValueError, match="hess_eps"):
        DerivedNewtonDiag([w], hess_eps=0.0)


@pytest.mark.parametrize("momentum", [0.0, 0.5, 0.99])
def test_init_validation_valid_momentum(momentum):
    from optimisers.torch_derived_newton_diag import DerivedNewtonDiag
    w = torch.randn(3, requires_grad=True)
    DerivedNewtonDiag([w], momentum=momentum)  # should not raise


@pytest.mark.parametrize("beta_s", [0.0, 0.5, 1.0])
def test_init_validation_valid_beta_s(beta_s):
    from optimisers.torch_derived_newton_diag import DerivedNewtonDiag
    w = torch.randn(3, requires_grad=True)
    DerivedNewtonDiag([w], beta_s=beta_s)  # should not raise


# ---- p.grad is None ----

def test_grad_none_no_crash():
    """requires_grad=True param with grad=None: step doesn't crash, param unchanged."""
    from optimisers.torch_derived_newton_diag import DerivedNewtonDiag
    w1 = torch.randn(3, requires_grad=True)
    w2 = torch.randn(3, requires_grad=True)
    opt = DerivedNewtonDiag([w1, w2], lr=0.01, momentum=0.9, beta_s=0.1)

    w1.grad = torch.randn(3)
    w2_before = w2.data.clone()
    # h_diag only for w1 (w2 skipped because grad is None)
    opt.step(h_diag=[torch.abs(torch.randn(3)) + 0.1])
    assert torch.equal(w2.data, w2_before)


# ---- Log-loss parity ----

def test_log_loss_parity_diag():
    """SGDLearnableDiag(log_loss=True) vs JAX custom_sgd_log_learnable_diag."""
    import jax.numpy as jnp
    import optax
    from optimisers.jax_learnable_diag import custom_sgd_log_learnable_diag
    from optimisers.torch_learnable_diag import SGDLearnableDiag

    np.random.seed(1)
    w = np.random.randn(4, 3).astype(np.float32)
    g = np.random.randn(4, 3).astype(np.float32) * 0.1
    loss_val = 1.5

    hparams = dict(
        learning_rate=0.01, momentum=0.9, xi=0.1, beta=0.8,
        weight_decay=0.0, metric_lr=1e-3, metric_reg=1e-4, metric_clip=4.0,
    )

    # JAX
    params_j = {"w": jnp.array(w)}
    grads_j = {"w": jnp.array(g)}
    opt_j = custom_sgd_log_learnable_diag(**hparams)
    st = opt_j.init(params_j)
    updates, _ = opt_j.update(grads_j, st, jnp.float32(loss_val), params_j)
    new_j = optax.apply_updates(params_j, updates)

    # Torch
    w_t = torch.tensor(w.copy(), requires_grad=True)
    opt_t = SGDLearnableDiag(
        [w_t], lr=hparams["learning_rate"], momentum=hparams["momentum"],
        xi=hparams["xi"], beta=hparams["beta"],
        weight_decay=hparams["weight_decay"],
        metric_lr=hparams["metric_lr"], metric_reg=hparams["metric_reg"],
        metric_clip=hparams["metric_clip"], log_loss=True,
    )
    with torch.no_grad():
        w_t.grad = torch.tensor(g)
    opt_t.step(loss=loss_val)

    diff = float(np.max(np.abs(np.array(new_j["w"]) - w_t.detach().numpy())))
    assert diff < 1e-5, f"Log-loss parity mismatch: {diff}"


# ---- Learnable scalar parity ----

def test_learnable_scalar_parity():
    """Port of compare_learnable_scalar.py with actual assertions."""
    import jax
    import jax.numpy as jnp
    import optax
    from optimisers.jax_learnable_scalar import custom_sgd_learnable_scalar
    from optimisers.torch_learnable_scalar import SGDLearnableScalar

    np.random.seed(0)
    w = np.random.randn(4, 3).astype(np.float32)
    b = np.random.randn(3).astype(np.float32)

    hparams = dict(
        learning_rate=0.01, momentum=0.9, xi=0.1, beta=0.8,
        weight_decay=0.01, metric_lr=1e-3, metric_reg=1e-4, metric_clip=4.0,
    )

    # JAX
    params_j = {"w": jnp.array(w), "b": jnp.array(b)}
    grads_j = jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, params_j)
    opt_j = custom_sgd_learnable_scalar(**hparams)
    st = opt_j.init(params_j)
    updates, _ = opt_j.update(grads_j, st, params=params_j)
    new_j = optax.apply_updates(params_j, updates)

    # Torch
    w_t = torch.tensor(w.copy(), requires_grad=True)
    b_t = torch.tensor(b.copy(), requires_grad=True)
    opt_t = SGDLearnableScalar(
        [w_t, b_t], lr=hparams["learning_rate"], momentum=hparams["momentum"],
        xi=hparams["xi"], beta=hparams["beta"],
        weight_decay=hparams["weight_decay"],
        metric_lr=hparams["metric_lr"], metric_reg=hparams["metric_reg"],
        metric_clip=hparams["metric_clip"], log_loss=False,
    )
    with torch.no_grad():
        w_t.grad = torch.ones_like(w_t) * 0.1
        b_t.grad = torch.ones_like(b_t) * 0.1
    opt_t.step()

    diff_w = float(np.max(np.abs(np.array(new_j["w"]) - w_t.detach().numpy())))
    diff_b = float(np.max(np.abs(np.array(new_j["b"]) - b_t.detach().numpy())))
    assert diff_w < 1e-5, f"w parity failed: {diff_w}"
    assert diff_b < 1e-5, f"b parity failed: {diff_b}"
