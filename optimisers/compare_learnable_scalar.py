# optimisers/compare_scalar_learnable.py
import numpy as np
import jax, jax.numpy as jnp, optax
import torch

from jax_learnable_scalar import custom_sgd_learnable_scalar
from torch_learnable_scalar import SGDLearnableScalar

def main():
    np.random.seed(0)
    w = np.random.randn(4, 3).astype(np.float32)
    b = np.random.randn(3).astype(np.float32)

    # JAX
    params_j = {"w": jnp.array(w), "b": jnp.array(b)}
    grads_j = jax.tree.map(lambda x: jnp.ones_like(x)*0.1, params_j)
    opt_j = custom_sgd_learnable_scalar(learning_rate=0.01, momentum=0.9, xi=0.1, beta=0.8,
                                        weight_decay=0.01, metric_lr=1e-3, metric_reg=1e-4, metric_clip=4.0)
    st = opt_j.init(params_j)
    updates, st2 = opt_j.update(grads_j, st, params=params_j)
    new_params_j = optax.apply_updates(params_j, updates)

    # Torch
    w_t = torch.tensor(w, requires_grad=True)
    b_t = torch.tensor(b, requires_grad=True)
    opt_t = SGDLearnableScalar([w_t, b_t], lr=0.01, momentum=0.9, xi=0.1, beta=0.8,
                               weight_decay=0.01, metric_lr=1e-3, metric_reg=1e-4, metric_clip=4.0, log_loss=False)
    with torch.no_grad():
        w_t.grad = torch.ones_like(w_t)*0.1
        b_t.grad = torch.ones_like(b_t)*0.1
    opt_t.step()

    print("max|diff w|:", np.max(np.abs(np.array(new_params_j["w"]) - w_t.detach().numpy())))
    print("max|diff b|:", np.max(np.abs(np.array(new_params_j["b"]) - b_t.detach().numpy())))

if __name__ == "__main__":
    main()
