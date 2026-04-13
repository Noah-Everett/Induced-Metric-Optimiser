# optimisers/compare_learnable.py
"""
Minimal parity check between JAX and PyTorch learnable-diagonal variants.

This script constructs a small set of tensors, runs a single update step in JAX and
PyTorch with identical hyperparameters and initial states (including s=0 so that
gamma^{-1}=I at start), and prints the max abs diff.
"""

import numpy as np

# JAX side
import jax, jax.numpy as jnp
import optax
from optimisers.jax_learnable_diag import custom_sgd_learnable_diag

# Torch side
import torch
from optimisers.torch_learnable_diag import SGDLearnableDiag

def run_parity(hparams, n_steps, label=""):
    """Run JAX vs PyTorch parity check with varying gradients."""
    np.random.seed(0)
    w1 = np.random.randn(4, 3).astype(np.float32)
    b1 = np.random.randn(3).astype(np.float32)

    rng = np.random.RandomState(42)
    grads_w1 = [rng.randn(4, 3).astype(np.float32) * 0.1 for _ in range(n_steps)]
    grads_b1 = [rng.randn(3).astype(np.float32) * 0.1 for _ in range(n_steps)]

    # ---- JAX run
    params_j = {"w1": jnp.array(w1), "b1": jnp.array(b1)}
    opt_j = custom_sgd_learnable_diag(**hparams)
    st_j = opt_j.init(params_j)
    for i in range(n_steps):
        grads_j = {"w1": jnp.array(grads_w1[i]), "b1": jnp.array(grads_b1[i])}
        updates_j, st_j = opt_j.update(grads_j, st_j, params=params_j)
        params_j = optax.apply_updates(params_j, updates_j)

    # ---- Torch run
    w1_t = torch.tensor(w1.copy(), requires_grad=True)
    b1_t = torch.tensor(b1.copy(), requires_grad=True)
    opt_t = SGDLearnableDiag(
        [w1_t, b1_t], lr=hparams["learning_rate"], momentum=hparams["momentum"],
        xi=hparams["xi"], beta=hparams["beta"],
        weight_decay=hparams["weight_decay"],
        metric_lr=hparams["metric_lr"], metric_reg=hparams["metric_reg"],
        metric_clip=hparams["metric_clip"], log_loss=False,
    )
    for i in range(n_steps):
        with torch.no_grad():
            w1_t.grad = torch.tensor(grads_w1[i])
            b1_t.grad = torch.tensor(grads_b1[i])
        opt_t.step()

    # Compare
    j_w1, j_b1 = np.array(params_j["w1"]), np.array(params_j["b1"])
    diff_w1 = float(np.max(np.abs(j_w1 - w1_t.detach().numpy())))
    diff_b1 = float(np.max(np.abs(j_b1 - b1_t.detach().numpy())))
    print(f"[{label}] After {n_steps} steps:")
    print(f"  max|diff w1|: {diff_w1:.2e}")
    print(f"  max|diff b1|: {diff_b1:.2e}")
    tol = 1e-5
    assert diff_w1 < tol, f"w1 parity failed ({label}): {diff_w1}"
    assert diff_b1 < tol, f"b1 parity failed ({label}): {diff_b1}"
    print(f"  PASS (tol={tol})")


def main():
    # Standard test
    run_parity(
        dict(learning_rate=0.01, momentum=0.9, xi=0.1, beta=0.8,
             weight_decay=0.01, metric_lr=1e-3, metric_reg=1e-4, metric_clip=4.0),
        n_steps=25, label="standard",
    )
    # High lr*wd stress test
    run_parity(
        dict(learning_rate=0.1, momentum=0.9, xi=0.1, beta=0.8,
             weight_decay=0.1, metric_lr=1e-3, metric_reg=1e-4, metric_clip=4.0),
        n_steps=25, label="high-wd",
    )

if __name__ == "__main__":
    main()
