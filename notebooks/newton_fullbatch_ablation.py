"""
IMO-147: Newton diagonal full-batch ablation.

Tests whether the Newton target works as the theory predicts when
mini-batch noise is eliminated and clip bounds are relaxed.

Experiments:
1. Full-batch vs mini-batch (B=1024) at clip=4.0
2. Clip sweep: clip in {0.5, 1.0, 2.0, 4.0, 8.0, 20.0} with full-batch
3. No mean-centering ablation (full-batch, clip=20)
4. Adam and SGD baselines for comparison

Run:
    PYTHONPATH=repo .conda/bin/python3.11 repo/notebooks/imo147_newton_fullbatch_ablation.py
"""

import sys
import os
import time

# Add repo to path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
sys.path.insert(0, repo_dir)

import jax
import jax.numpy as jnp
import optax
import tensorflow as tf

from optimisers.jax_derived_newton_diag import derived_newton_diag
from optimisers.jax_gn_diag import gn_diag_mlp, cross_entropy_output_grad
from parameters.shared_models import MLP


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = jnp.array(x_train.reshape(-1, 784) / 255.0, dtype=jnp.float32)
    x_test = jnp.array(x_test.reshape(-1, 784) / 255.0, dtype=jnp.float32)
    y_train = jnp.array(y_train)
    y_test = jnp.array(y_test)
    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def loss_fn(params, x, y, model):
    logits = model.apply(params, x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


def accuracy(params, x, y, model):
    logits = model.apply(params, x)
    return float(jnp.mean(jnp.argmax(logits, axis=1) == y))


def train_newton(params, model, x_train, y_train, x_test, y_test,
                 batch_size, lr, momentum, beta_s, metric_clip,
                 n_epochs=200, label=""):
    """Train with newton_diag and return trajectory."""
    opt = derived_newton_diag(
        learning_rate=lr, momentum=momentum, beta_s=beta_s,
        weight_decay=0.0, metric_clip=metric_clip, hess_eps=1e-6, xi=0.0,
    )
    state = opt.init(params)

    # Batch the data
    n_batches = len(x_train) // batch_size
    x_batched = x_train[:n_batches * batch_size].reshape(n_batches, batch_size, 784)
    y_batched = y_train[:n_batches * batch_size].reshape(n_batches, batch_size)

    trajectory = []
    for epoch in range(n_epochs):
        t0 = time.time()
        epoch_losses = []
        for b in range(n_batches):
            x, y = x_batched[b], y_batched[b]
            loss, grads = jax.value_and_grad(lambda p: loss_fn(p, x, y, model))(params)
            h_diag = gn_diag_mlp(params, x, y, jax.nn.gelu, cross_entropy_output_grad)
            updates, state = opt.update(grads, state, h_diag, params=params)
            params = optax.apply_updates(params, updates)
            epoch_losses.append(float(loss))

        dt = time.time() - t0
        avg_loss = sum(epoch_losses) / len(epoch_losses)

        # Metric diagnostics
        s_leaves = jax.tree.leaves(state.log_diag)
        all_s = jnp.concatenate([l.ravel() for l in s_leaves])
        clipped_frac = float(jnp.mean(jnp.abs(all_s) >= metric_clip - 1e-6))
        condition = float(jnp.exp(jnp.max(all_s)) / jnp.exp(jnp.min(all_s)))
        s_std = float(jnp.std(all_s))

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            test_acc = accuracy(params, x_test, y_test, model)
            train_acc = accuracy(params, x_train, y_train, model)
            trajectory.append({
                'epoch': epoch, 'loss': avg_loss,
                'train_acc': train_acc, 'test_acc': test_acc,
                'clipped_frac': clipped_frac, 'condition': condition,
                's_std': s_std, 'time': dt,
            })
            if epoch % 50 == 0:
                print(f"  [{label}] ep={epoch:3d}  loss={avg_loss:.4f}  "
                      f"acc={test_acc:.4f}  clip={clipped_frac:.3f}  "
                      f"cond={condition:.1f}  s_std={s_std:.2f}  "
                      f"({dt:.2f}s)")

    return params, trajectory


def train_baseline(params, model, x_train, y_train, x_test, y_test,
                   batch_size, optimizer, n_epochs=200, label=""):
    """Train with a baseline optimizer (adam/sgd)."""
    state = optimizer.init(params)

    n_batches = len(x_train) // batch_size
    x_batched = x_train[:n_batches * batch_size].reshape(n_batches, batch_size, 784)
    y_batched = y_train[:n_batches * batch_size].reshape(n_batches, batch_size)

    trajectory = []
    for epoch in range(n_epochs):
        t0 = time.time()
        epoch_losses = []
        for b in range(n_batches):
            x, y = x_batched[b], y_batched[b]
            loss, grads = jax.value_and_grad(lambda p: loss_fn(p, x, y, model))(params)
            updates, state = optimizer.update(grads, state, params)
            params = optax.apply_updates(params, updates)
            epoch_losses.append(float(loss))

        dt = time.time() - t0
        avg_loss = sum(epoch_losses) / len(epoch_losses)

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            test_acc = accuracy(params, x_test, y_test, model)
            train_acc = accuracy(params, x_train, y_train, model)
            trajectory.append({
                'epoch': epoch, 'loss': avg_loss,
                'train_acc': train_acc, 'test_acc': test_acc,
                'time': dt,
            })
            if epoch % 50 == 0:
                print(f"  [{label}] ep={epoch:3d}  loss={avg_loss:.4f}  "
                      f"acc={test_acc:.4f}  ({dt:.2f}s)")

    return params, trajectory


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def main():
    print("Loading MNIST...")
    x_train, y_train, x_test, y_test = load_mnist()

    model = MLP(features=64, output_dim=10)
    key = jax.random.PRNGKey(42)
    init_params = model.init(key, jnp.ones((1, 784)))

    full_batch = len(x_train)  # 60000
    mini_batch = 1024
    n_epochs = 200

    # Good HP values (from best itr_7 trials)
    lr = 0.5
    momentum = 0.9
    beta_s = 0.1

    results = {}

    # =================================================================
    # Experiment 1: Full-batch vs mini-batch at clip=4.0
    # =================================================================
    print("\n" + "=" * 70)
    print("EXP 1: Full-batch vs mini-batch (clip=4.0)")
    print("=" * 70)

    print(f"\n--- Mini-batch (B={mini_batch}) ---")
    _, results['mini_clip4'] = train_newton(
        init_params, model, x_train, y_train, x_test, y_test,
        batch_size=mini_batch, lr=lr, momentum=momentum, beta_s=beta_s,
        metric_clip=4.0, n_epochs=n_epochs, label="mini,clip=4")

    print(f"\n--- Full-batch (B={full_batch}) ---")
    _, results['full_clip4'] = train_newton(
        init_params, model, x_train, y_train, x_test, y_test,
        batch_size=full_batch, lr=lr, momentum=momentum, beta_s=beta_s,
        metric_clip=4.0, n_epochs=n_epochs, label="full,clip=4")

    # =================================================================
    # Experiment 2: Clip sweep with full-batch
    # =================================================================
    print("\n" + "=" * 70)
    print("EXP 2: Clip sweep (full-batch)")
    print("=" * 70)

    for clip in [0.5, 1.0, 2.0, 4.0, 8.0, 20.0]:
        # Adjust LR for stability: eta * exp(clip) * h_max < 2
        # With large clip, need smaller LR
        adj_lr = min(lr, 1.0 / jnp.exp(clip) * 0.5)
        print(f"\n--- clip={clip}, lr={adj_lr:.4f} ---")
        _, results[f'full_clip{clip}'] = train_newton(
            init_params, model, x_train, y_train, x_test, y_test,
            batch_size=full_batch, lr=adj_lr, momentum=momentum, beta_s=beta_s,
            metric_clip=clip, n_epochs=n_epochs, label=f"full,clip={clip}")

    # =================================================================
    # Experiment 3: Baselines (full-batch)
    # =================================================================
    print("\n" + "=" * 70)
    print("EXP 3: Baselines (full-batch)")
    print("=" * 70)

    print(f"\n--- Adam (full-batch) ---")
    _, results['adam_full'] = train_baseline(
        init_params, model, x_train, y_train, x_test, y_test,
        batch_size=full_batch,
        optimizer=optax.adam(learning_rate=1e-3),
        n_epochs=n_epochs, label="adam,full")

    print(f"\n--- SGD+momentum (full-batch) ---")
    _, results['sgd_full'] = train_baseline(
        init_params, model, x_train, y_train, x_test, y_test,
        batch_size=full_batch,
        optimizer=optax.sgd(learning_rate=lr, momentum=momentum),
        n_epochs=n_epochs, label="sgd,full")

    # =================================================================
    # Experiment 4: Baselines (mini-batch) for reference
    # =================================================================
    print("\n" + "=" * 70)
    print("EXP 4: Baselines (mini-batch)")
    print("=" * 70)

    print(f"\n--- Adam (mini-batch) ---")
    _, results['adam_mini'] = train_baseline(
        init_params, model, x_train, y_train, x_test, y_test,
        batch_size=mini_batch,
        optimizer=optax.adam(learning_rate=1e-3),
        n_epochs=n_epochs, label="adam,mini")

    print(f"\n--- SGD+momentum (mini-batch) ---")
    _, results['sgd_mini'] = train_baseline(
        init_params, model, x_train, y_train, x_test, y_test,
        batch_size=mini_batch,
        optimizer=optax.sgd(learning_rate=lr, momentum=momentum),
        n_epochs=n_epochs, label="sgd,mini")

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    fmt = "{:<30s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}"
    print(fmt.format("config", "acc@10", "acc@50", "acc@100", "acc@199", "clip_f"))
    print("-" * 75)

    for name, traj in sorted(results.items()):
        acc_at = {}
        clip_at = {}
        for t in traj:
            acc_at[t['epoch']] = t['test_acc']
            clip_at[t['epoch']] = t.get('clipped_frac', '')

        def _a(ep):
            return f"{acc_at.get(ep, 0):.4f}" if ep in acc_at else "N/A"
        def _c(ep):
            v = clip_at.get(ep, '')
            return f"{v:.3f}" if isinstance(v, float) else ""

        # Use clipped_frac from epoch 10 if available
        cf = _c(10) if _c(10) else _c(0)
        print(fmt.format(name, _a(10), _a(50), _a(100), _a(199), cf))


if __name__ == "__main__":
    main()
