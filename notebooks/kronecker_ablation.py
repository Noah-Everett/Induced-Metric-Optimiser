"""
Kronecker optimizer ablation (IMO-152).

Focused comparison: Kronecker vs diagonal Newton vs Adam.
Full-batch, 1000 steps, MNIST MLP.

Run:
    PYTHONPATH=repo .conda/bin/python3.11 repo/notebooks/kronecker_ablation.py
"""

import sys
import os
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
sys.path.insert(0, repo_dir)

import jax
import jax.numpy as jnp
import optax
import tensorflow as tf

from optimisers.jax_derived_newton_diag import derived_newton_diag
from optimisers.jax_gn_diag import gn_diag_mlp, cross_entropy_output_grad
from optimisers.jax_kronecker import kronecker_factors_mlp, kronecker_newton
from parameters.shared_models import MLP


def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = jnp.array(x_train.reshape(-1, 784) / 255.0, dtype=jnp.float32)
    x_test = jnp.array(x_test.reshape(-1, 784) / 255.0, dtype=jnp.float32)
    y_train = jnp.array(y_train)
    y_test = jnp.array(y_test)
    return x_train, y_train, x_test, y_test


def loss_fn(params, x, y, model):
    logits = model.apply(params, x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


def accuracy(params, x, y, model):
    logits = model.apply(params, x)
    return float(jnp.mean(jnp.argmax(logits, axis=1) == y))


def train_kronecker(params, model, x, y, x_test, y_test,
                    lr, momentum, damping, n_steps, eval_every, label):
    opt = kronecker_newton(learning_rate=lr, momentum=momentum)
    state = opt.init(params)
    trajectory = []
    t0 = time.time()
    for step in range(n_steps):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, x, y, model))(params)
        kf = kronecker_factors_mlp(params, x, y, jax.nn.gelu,
                                   cross_entropy_output_grad, damping=damping)
        updates, state = opt.update(grads, state, kf)
        params = optax.apply_updates(params, updates)
        if step % eval_every == 0 or step == n_steps - 1:
            acc = accuracy(params, x_test, y_test, model)
            elapsed = time.time() - t0
            trajectory.append({'step': step, 'loss': float(loss),
                               'test_acc': acc, 'elapsed': elapsed})
            if step % (eval_every * 4) == 0:
                print(f"  [{label}] step={step:4d}  loss={float(loss):.4f}  "
                      f"acc={acc:.4f}  ({elapsed:.1f}s)")
    return params, trajectory


def train_newton(params, model, x, y, x_test, y_test,
                 lr, momentum, beta_s, clip, n_steps, eval_every, label):
    opt = derived_newton_diag(learning_rate=lr, momentum=momentum,
                              beta_s=beta_s, metric_clip=clip, hess_eps=1e-6)
    state = opt.init(params)
    trajectory = []
    t0 = time.time()
    for step in range(n_steps):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, x, y, model))(params)
        h_diag = gn_diag_mlp(params, x, y, jax.nn.gelu, cross_entropy_output_grad)
        updates, state = opt.update(grads, state, h_diag, params=params)
        params = optax.apply_updates(params, updates)
        if step % eval_every == 0 or step == n_steps - 1:
            acc = accuracy(params, x_test, y_test, model)
            elapsed = time.time() - t0
            trajectory.append({'step': step, 'loss': float(loss),
                               'test_acc': acc, 'elapsed': elapsed})
            if step % (eval_every * 4) == 0:
                print(f"  [{label}] step={step:4d}  loss={float(loss):.4f}  "
                      f"acc={acc:.4f}  ({elapsed:.1f}s)")
    return params, trajectory


def train_adam(params, model, x, y, x_test, y_test,
              lr, n_steps, eval_every, label):
    opt = optax.adam(learning_rate=lr)
    state = opt.init(params)
    trajectory = []
    t0 = time.time()
    for step in range(n_steps):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, x, y, model))(params)
        updates, state = opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        if step % eval_every == 0 or step == n_steps - 1:
            acc = accuracy(params, x_test, y_test, model)
            elapsed = time.time() - t0
            trajectory.append({'step': step, 'loss': float(loss),
                               'test_acc': acc, 'elapsed': elapsed})
            if step % (eval_every * 4) == 0:
                print(f"  [{label}] step={step:4d}  loss={float(loss):.4f}  "
                      f"acc={acc:.4f}  ({elapsed:.1f}s)")
    return params, trajectory


def main():
    print("Loading MNIST...")
    x_train, y_train, x_test, y_test = load_mnist()

    model = MLP(features=64, output_dim=10)
    key = jax.random.PRNGKey(42)
    init_params = model.init(key, jnp.ones((1, 784)))

    n_steps = 1000
    eval_every = 50
    results = {}

    # All experiments use full-batch
    x, y = x_train, y_train

    # --- Adam baseline ---
    print("\n--- Adam (lr=1e-3) ---")
    _, results['adam'] = train_adam(
        init_params, model, x, y, x_test, y_test,
        lr=1e-3, n_steps=n_steps, eval_every=eval_every, label="adam")

    # --- Diagonal Newton (best configs from IMO-147) ---
    print("\n--- Diagonal Newton (clip=1.3) ---")
    _, results['newton_c1.3'] = train_newton(
        init_params, model, x, y, x_test, y_test,
        lr=0.5, momentum=0.9, beta_s=0.1, clip=1.3,
        n_steps=n_steps, eval_every=eval_every, label="newton,c=1.3")

    print("\n--- Diagonal Newton (clip=4) ---")
    _, results['newton_c4'] = train_newton(
        init_params, model, x, y, x_test, y_test,
        lr=0.5, momentum=0.9, beta_s=0.1, clip=4.0,
        n_steps=n_steps, eval_every=eval_every, label="newton,c=4")

    # --- Kronecker: damping sweep ---
    for damp in [1e-4, 1e-2, 1e-1]:
        print(f"\n--- Kronecker (d={damp}, lr=0.1, mom=0.9) ---")
        _, results[f'kron_d{damp}'] = train_kronecker(
            init_params, model, x, y, x_test, y_test,
            lr=0.1, momentum=0.9, damping=damp,
            n_steps=n_steps, eval_every=eval_every,
            label=f"kron,d={damp}")

    # --- Kronecker: LR sweep at best damping ---
    for kron_lr in [0.01, 0.5]:
        print(f"\n--- Kronecker (d=1e-4, lr={kron_lr}, mom=0.9) ---")
        _, results[f'kron_lr{kron_lr}'] = train_kronecker(
            init_params, model, x, y, x_test, y_test,
            lr=kron_lr, momentum=0.9, damping=1e-4,
            n_steps=n_steps, eval_every=eval_every,
            label=f"kron,lr={kron_lr}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY (test accuracy at gradient step count)")
    print("=" * 70)

    checkpoints = [50, 100, 200, 500, 999]
    fmt = "{:<25s}" + " {:>8s}" * len(checkpoints) + " {:>8s}"
    print(fmt.format("config", *[f"@{s}" for s in checkpoints], "time(s)"))
    print("-" * (25 + 9 * (len(checkpoints) + 1)))

    for name in sorted(results.keys()):
        traj = results[name]
        acc_at = {t['step']: t for t in traj}
        vals = []
        for s in checkpoints:
            closest = min(acc_at.keys(), key=lambda k: abs(k - s)) if acc_at else None
            if closest is not None and abs(closest - s) <= 10:
                vals.append(f"{acc_at[closest]['test_acc']:.4f}")
            else:
                vals.append("N/A")
        last = traj[-1] if traj else {}
        vals.append(f"{last.get('elapsed', 0):.1f}")
        print(fmt.format(name, *vals))


if __name__ == "__main__":
    main()
