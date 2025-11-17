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
from jax_learnable import custom_sgd_learnable_diag

# Torch side
import torch
from torch_learnable import SGDLearnableDiag

def main():
    key = jax.random.PRNGKey(0)
    # synthetic params
    np.random.seed(0)
    w1 = np.random.randn(4, 3).astype(np.float32)
    b1 = np.random.randn(3).astype(np.float32)

    # ---- JAX run
    params_j = {"w1": jnp.array(w1), "b1": jnp.array(b1)}
    grads_j = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 0.1, params_j)

    opt_j = custom_sgd_learnable_diag(
        learning_rate=0.01, momentum=0.9, xi=0.1, beta=0.8,
        weight_decay=0.01, metric_lr=1e-3, metric_reg=1e-4, metric_clip=4.0
    )
    st_j = opt_j.init(params_j)
    updates_j, st_j2 = opt_j.update(grads_j, st_j, params=params_j)
    new_params_j = optax.apply_updates(params_j, updates_j)

    # ---- Torch run
    w1_t = torch.tensor(w1, requires_grad=True)
    b1_t = torch.tensor(b1, requires_grad=True)
    model_params = [w1_t, b1_t]
    opt_t = SGDLearnableDiag(
        model_params, lr=0.01, momentum=0.9, xi=0.1, beta=0.8,
        weight_decay=0.01, metric_lr=1e-3, metric_reg=1e-4, metric_clip=4.0, log_loss=False
    )
    # supply matching grads
    with torch.no_grad():
        w1_t.grad = torch.ones_like(w1_t) * 0.1
        b1_t.grad = torch.ones_like(b1_t) * 0.1
    opt_t.step()

    # Compare
    j_w1, j_b1 = np.array(new_params_j["w1"]), np.array(new_params_j["b1"])
    diff_w1 = np.max(np.abs(j_w1 - w1_t.detach().numpy()))
    diff_b1 = np.max(np.abs(j_b1 - b1_t.detach().numpy()))
    print("max|diff w1|:", float(diff_w1))
    print("max|diff b1|:", float(diff_b1))

if __name__ == "__main__":
    main()
