# optimisers/compare_derived_newton_diag.py
"""
Parity check between JAX and PyTorch derived Newton-targeted diagonal optimizers.

Constructs identical parameters and Hessian diagonal estimates, runs a few steps
in both frameworks, and verifies the results match to within floating-point
tolerance.

Run: PYTHONPATH=repo .conda/bin/python repo/optimisers/compare_derived_newton_diag.py
"""

import numpy as np

# JAX side
import jax
import jax.numpy as jnp
import optax
from optimisers.jax_derived_newton_diag import derived_newton_diag

# Torch side
import torch
from optimisers.torch_derived_newton_diag import DerivedNewtonDiag


def main():
    np.random.seed(42)
    w1 = np.random.randn(4, 3).astype(np.float32)
    b1 = np.random.randn(3).astype(np.float32)

    # Deterministic h_diag (simulating Hessian diagonal)
    h_w1 = np.abs(np.random.randn(4, 3).astype(np.float32)) + 0.1
    h_b1 = np.abs(np.random.randn(3).astype(np.float32)) + 0.1

    # Constant grads for reproducibility
    g_w1 = np.random.randn(4, 3).astype(np.float32) * 0.1
    g_b1 = np.random.randn(3).astype(np.float32) * 0.1

    hparams = dict(
        learning_rate=0.01, momentum=0.9, beta_s=0.1,
        weight_decay=0.01, metric_clip=4.0, hess_eps=1e-6,
    )
    n_steps = 5

    # ---- JAX run ----
    params_j = {"w1": jnp.array(w1), "b1": jnp.array(b1)}
    grads_j = {"w1": jnp.array(g_w1), "b1": jnp.array(g_b1)}
    h_diag_j = {"w1": jnp.array(h_w1), "b1": jnp.array(h_b1)}

    opt_j = derived_newton_diag(**hparams)
    st_j = opt_j.init(params_j)
    for _ in range(n_steps):
        updates_j, st_j = opt_j.update(grads_j, st_j, h_diag_j, params=params_j)
        params_j = optax.apply_updates(params_j, updates_j)

    # ---- Torch run ----
    w1_t = torch.tensor(w1, requires_grad=True)
    b1_t = torch.tensor(b1, requires_grad=True)
    model_params = [w1_t, b1_t]

    # Remap hparams for PyTorch API (learning_rate -> lr)
    opt_t = DerivedNewtonDiag(
        model_params, lr=hparams["learning_rate"], momentum=hparams["momentum"],
        beta_s=hparams["beta_s"], weight_decay=hparams["weight_decay"],
        metric_clip=hparams["metric_clip"], hess_eps=hparams["hess_eps"],
    )
    h_diag_t = [torch.tensor(h_w1), torch.tensor(h_b1)]

    for _ in range(n_steps):
        with torch.no_grad():
            w1_t.grad = torch.tensor(g_w1)
            b1_t.grad = torch.tensor(g_b1)
        opt_t.step(h_diag=h_diag_t)

    # ---- Compare ----
    j_w1 = np.array(params_j["w1"])
    j_b1 = np.array(params_j["b1"])
    t_w1 = w1_t.detach().numpy()
    t_b1 = b1_t.detach().numpy()

    diff_w1 = float(np.max(np.abs(j_w1 - t_w1)))
    diff_b1 = float(np.max(np.abs(j_b1 - t_b1)))

    print(f"After {n_steps} steps:")
    print(f"  max|diff w1|: {diff_w1:.2e}")
    print(f"  max|diff b1|: {diff_b1:.2e}")

    # Also compare internal metric state (log_diag / s)
    s_j_w1 = np.array(st_j.log_diag["w1"])
    s_j_b1 = np.array(st_j.log_diag["b1"])
    s_t_w1 = opt_t.state[w1_t]['log_diag'].numpy()
    s_t_b1 = opt_t.state[b1_t]['log_diag'].numpy()
    diff_s_w1 = float(np.max(np.abs(s_j_w1 - s_t_w1)))
    diff_s_b1 = float(np.max(np.abs(s_j_b1 - s_t_b1)))
    print(f"  max|diff s_w1|: {diff_s_w1:.2e}")
    print(f"  max|diff s_b1|: {diff_s_b1:.2e}")

    tol = 1e-5
    ok = all(d < tol for d in [diff_w1, diff_b1, diff_s_w1, diff_s_b1])
    if ok:
        print(f"\nPARITY CHECK PASSED (tol={tol})")
    else:
        print(f"\nPARITY CHECK FAILED (tol={tol})")
        raise AssertionError("JAX/PyTorch parity check failed")


if __name__ == "__main__":
    main()
