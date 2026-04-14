"""
Newton diagonal full-batch ablation (IMO-147).

Tests whether the Newton target works as the theory predicts when
mini-batch noise is eliminated and clip bounds are relaxed.

All comparisons are at EQUAL GRADIENT STEP COUNT (not equal epochs).
Full-batch = 1 step/epoch, mini-batch (B=1024) = 58 steps/epoch.

Experiments:
1. Step-matched comparison: full-batch 1000 steps vs mini-batch ~1000 steps
2. Clip sweep with full-batch, fixed LR (no auto-adjustment)
3. Adam and SGD baselines at both batch sizes

Run:
    PYTHONPATH=repo .conda/bin/python3.11 repo/notebooks/newton_fullbatch_ablation.py
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
from optimisers.jax_kronecker import kronecker_factors_mlp, kronecker_newton
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
# Training (step-based, not epoch-based)
# ---------------------------------------------------------------------------

def loss_fn(params, x, y, model):
    logits = model.apply(params, x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


def accuracy(params, x, y, model):
    logits = model.apply(params, x)
    return float(jnp.mean(jnp.argmax(logits, axis=1) == y))


def train_newton_steps(params, model, x_train, y_train, x_test, y_test,
                       batch_size, lr, momentum, beta_s, metric_clip,
                       n_steps=1000, eval_every=50, label=""):
    """Train for a fixed number of gradient steps."""
    opt = derived_newton_diag(
        learning_rate=lr, momentum=momentum, beta_s=beta_s,
        weight_decay=0.0, metric_clip=metric_clip, hess_eps=1e-6, xi=0.0,
    )
    state = opt.init(params)

    n_batches = len(x_train) // batch_size
    x_batched = x_train[:n_batches * batch_size].reshape(n_batches, batch_size, 784)
    y_batched = y_train[:n_batches * batch_size].reshape(n_batches, batch_size)

    trajectory = []
    step = 0
    t0 = time.time()
    while step < n_steps:
        for b in range(n_batches):
            if step >= n_steps:
                break
            x, y = x_batched[b], y_batched[b]
            loss, grads = jax.value_and_grad(lambda p: loss_fn(p, x, y, model))(params)
            h_diag = gn_diag_mlp(params, x, y, jax.nn.gelu, cross_entropy_output_grad)
            updates, state = opt.update(grads, state, h_diag, params=params)
            params = optax.apply_updates(params, updates)

            if step % eval_every == 0 or step == n_steps - 1:
                s_leaves = jax.tree.leaves(state.log_diag)
                all_s = jnp.concatenate([l.ravel() for l in s_leaves])
                clipped_frac = float(jnp.mean(jnp.abs(all_s) >= metric_clip - 1e-6))
                condition = float(jnp.exp(jnp.max(all_s)) / jnp.exp(jnp.min(all_s)))
                test_acc = accuracy(params, x_test, y_test, model)
                elapsed = time.time() - t0
                trajectory.append({
                    'step': step, 'loss': float(loss), 'test_acc': test_acc,
                    'clipped_frac': clipped_frac, 'condition': condition,
                    'elapsed': elapsed,
                })
                if step % (eval_every * 4) == 0:
                    print(f"  [{label}] step={step:5d}  loss={float(loss):.4f}  "
                          f"acc={test_acc:.4f}  clip={clipped_frac:.3f}  "
                          f"cond={condition:.1f}  ({elapsed:.1f}s)")
            step += 1

    return params, trajectory


def train_kronecker_steps(params, model, x_train, y_train, x_test, y_test,
                          batch_size, lr, momentum, damping,
                          n_steps=1000, eval_every=50, label=""):
    """Train with Kronecker-preconditioned gradient for fixed steps."""
    opt = kronecker_newton(learning_rate=lr, momentum=momentum)
    state = opt.init(params)

    n_batches = len(x_train) // batch_size
    x_batched = x_train[:n_batches * batch_size].reshape(n_batches, batch_size, 784)
    y_batched = y_train[:n_batches * batch_size].reshape(n_batches, batch_size)

    trajectory = []
    step = 0
    t0 = time.time()
    while step < n_steps:
        for b in range(n_batches):
            if step >= n_steps:
                break
            x, y = x_batched[b], y_batched[b]
            loss, grads = jax.value_and_grad(lambda p: loss_fn(p, x, y, model))(params)
            kron_factors = kronecker_factors_mlp(
                params, x, y, jax.nn.gelu, cross_entropy_output_grad,
                damping=damping)
            updates, state = opt.update(grads, state, kron_factors)
            params = optax.apply_updates(params, updates)

            if step % eval_every == 0 or step == n_steps - 1:
                test_acc = accuracy(params, x_test, y_test, model)
                elapsed = time.time() - t0
                trajectory.append({
                    'step': step, 'loss': float(loss), 'test_acc': test_acc,
                    'elapsed': elapsed,
                })
                if step % (eval_every * 4) == 0:
                    print(f"  [{label}] step={step:5d}  loss={float(loss):.4f}  "
                          f"acc={test_acc:.4f}  ({elapsed:.1f}s)")
            step += 1

    return params, trajectory


def train_baseline_steps(params, model, x_train, y_train, x_test, y_test,
                         batch_size, optimizer, n_steps=1000,
                         eval_every=50, label=""):
    """Train a baseline optimizer for a fixed number of gradient steps."""
    state = optimizer.init(params)

    n_batches = len(x_train) // batch_size
    x_batched = x_train[:n_batches * batch_size].reshape(n_batches, batch_size, 784)
    y_batched = y_train[:n_batches * batch_size].reshape(n_batches, batch_size)

    trajectory = []
    step = 0
    t0 = time.time()
    while step < n_steps:
        for b in range(n_batches):
            if step >= n_steps:
                break
            x, y = x_batched[b], y_batched[b]
            loss, grads = jax.value_and_grad(lambda p: loss_fn(p, x, y, model))(params)
            updates, state = optimizer.update(grads, state, params)
            params = optax.apply_updates(params, updates)

            if step % eval_every == 0 or step == n_steps - 1:
                test_acc = accuracy(params, x_test, y_test, model)
                elapsed = time.time() - t0
                trajectory.append({
                    'step': step, 'loss': float(loss), 'test_acc': test_acc,
                    'elapsed': elapsed,
                })
                if step % (eval_every * 4) == 0:
                    print(f"  [{label}] step={step:5d}  loss={float(loss):.4f}  "
                          f"acc={test_acc:.4f}  ({elapsed:.1f}s)")
            step += 1

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
    n_steps = 1000  # all experiments run for exactly 1000 gradient steps

    # Fixed HP values
    lr = 0.5
    momentum = 0.9
    beta_s = 0.1

    results = {}

    # =================================================================
    # Experiment 1: Full-batch vs mini-batch, step-matched
    # =================================================================
    print("\n" + "=" * 70)
    print(f"EXP 1: Full-batch vs mini-batch, {n_steps} gradient steps each")
    print(f"  Full-batch: {n_steps} epochs x 1 batch = {n_steps} steps")
    print(f"  Mini-batch: {n_steps//58} epochs x 58 batches ~ {n_steps} steps")
    print("=" * 70)

    print(f"\n--- Newton full-batch (clip=4) ---")
    _, results['newton_full_c4'] = train_newton_steps(
        init_params, model, x_train, y_train, x_test, y_test,
        batch_size=full_batch, lr=lr, momentum=momentum, beta_s=beta_s,
        metric_clip=4.0, n_steps=n_steps, label="newton,full,c=4")

    print(f"\n--- Newton mini-batch (clip=4) ---")
    _, results['newton_mini_c4'] = train_newton_steps(
        init_params, model, x_train, y_train, x_test, y_test,
        batch_size=mini_batch, lr=lr, momentum=momentum, beta_s=beta_s,
        metric_clip=4.0, n_steps=n_steps, label="newton,mini,c=4")

    print(f"\n--- Newton full-batch (clip=1.3, best sweep config) ---")
    _, results['newton_full_c1.3'] = train_newton_steps(
        init_params, model, x_train, y_train, x_test, y_test,
        batch_size=full_batch, lr=lr, momentum=momentum, beta_s=beta_s,
        metric_clip=1.3, n_steps=n_steps, label="newton,full,c=1.3")

    print(f"\n--- Newton mini-batch (clip=1.3) ---")
    _, results['newton_mini_c1.3'] = train_newton_steps(
        init_params, model, x_train, y_train, x_test, y_test,
        batch_size=mini_batch, lr=lr, momentum=momentum, beta_s=beta_s,
        metric_clip=1.3, n_steps=n_steps, label="newton,mini,c=1.3")

    # =================================================================
    # Experiment 2: Clip sweep, full-batch, FIXED LR
    # =================================================================
    print("\n" + "=" * 70)
    print(f"EXP 2: Clip sweep (full-batch, lr={lr}, {n_steps} steps)")
    print("  Same LR for all clips — tests whether larger clip helps or hurts")
    print("=" * 70)

    for clip in [0.5, 1.0, 2.0, 4.0, 8.0]:
        print(f"\n--- clip={clip} ---")
        _, results[f'clip_sweep_{clip}'] = train_newton_steps(
            init_params, model, x_train, y_train, x_test, y_test,
            batch_size=full_batch, lr=lr, momentum=momentum, beta_s=beta_s,
            metric_clip=clip, n_steps=n_steps, label=f"full,c={clip}")

    # =================================================================
    # Experiment 3: Baselines, step-matched
    # =================================================================
    print("\n" + "=" * 70)
    print(f"EXP 3: Baselines ({n_steps} steps each)")
    print("=" * 70)

    for bs, bs_label in [(full_batch, "full"), (mini_batch, "mini")]:
        print(f"\n--- Adam ({bs_label}-batch, lr=1e-3) ---")
        _, results[f'adam_{bs_label}'] = train_baseline_steps(
            init_params, model, x_train, y_train, x_test, y_test,
            batch_size=bs, optimizer=optax.adam(learning_rate=1e-3),
            n_steps=n_steps, label=f"adam,{bs_label}")

        print(f"\n--- SGD+mom ({bs_label}-batch, lr=0.01) ---")
        _, results[f'sgd_{bs_label}'] = train_baseline_steps(
            init_params, model, x_train, y_train, x_test, y_test,
            batch_size=bs, optimizer=optax.sgd(learning_rate=0.01, momentum=momentum),
            n_steps=n_steps, label=f"sgd,{bs_label}")

    # =================================================================
    # Experiment 4: Kronecker vs diagonal Newton (IMO-152)
    # =================================================================
    print("\n" + "=" * 70)
    print(f"EXP 4: Kronecker-preconditioned Newton ({n_steps} steps, full-batch)")
    print("=" * 70)

    # 4a: Damping sweep
    for damp in [1e-6, 1e-4, 1e-2, 1e-1]:
        print(f"\n--- Kronecker (damping={damp}, lr=0.1, mom=0.9) ---")
        _, results[f'kron_d{damp}'] = train_kronecker_steps(
            init_params, model, x_train, y_train, x_test, y_test,
            batch_size=full_batch, lr=0.1, momentum=momentum,
            damping=damp, n_steps=n_steps, label=f"kron,d={damp}")

    # 4b: LR sweep at best damping (1e-4 as starting point)
    for kron_lr in [0.01, 0.05, 0.5]:
        print(f"\n--- Kronecker (damping=1e-4, lr={kron_lr}, mom=0.9) ---")
        _, results[f'kron_lr{kron_lr}'] = train_kronecker_steps(
            init_params, model, x_train, y_train, x_test, y_test,
            batch_size=full_batch, lr=kron_lr, momentum=momentum,
            damping=1e-4, n_steps=n_steps, label=f"kron,lr={kron_lr}")

    # 4c: No-momentum Kronecker
    print(f"\n--- Kronecker (no momentum, damping=1e-4, lr=0.1) ---")
    _, results['kron_nomom'] = train_kronecker_steps(
        init_params, model, x_train, y_train, x_test, y_test,
        batch_size=full_batch, lr=0.1, momentum=0.0,
        damping=1e-4, n_steps=n_steps, label="kron,nomom")

    # =================================================================
    # Summary at key step counts
    # =================================================================
    print("\n" + "=" * 70)
    print("SUMMARY (test accuracy at gradient step count)")
    print("=" * 70)

    checkpoints = [50, 100, 200, 500, 999]
    fmt = "{:<25s}" + " {:>8s}" * len(checkpoints)
    print(fmt.format("config", *[f"@{s}" for s in checkpoints]))
    print("-" * (25 + 9 * len(checkpoints)))

    for name in sorted(results.keys()):
        traj = results[name]
        # Build step -> acc lookup
        acc_at = {t['step']: t['test_acc'] for t in traj}
        vals = []
        for s in checkpoints:
            # Find closest step
            closest = min(acc_at.keys(), key=lambda k: abs(k - s)) if acc_at else None
            if closest is not None and abs(closest - s) <= 10:
                vals.append(f"{acc_at[closest]:.4f}")
            else:
                vals.append("N/A")
        print(fmt.format(name, *vals))

    # Clip sweep detail
    print()
    print("--- Clip sweep detail (full-batch, step 999) ---")
    fmt2 = "{:<10s} {:>8s} {:>8s} {:>8s} {:>8s}"
    print(fmt2.format("clip", "acc", "loss", "clip_f", "cond"))
    print("-" * 45)
    for clip in [0.5, 1.0, 2.0, 4.0, 8.0]:
        key = f'clip_sweep_{clip}'
        if key in results and results[key]:
            last = results[key][-1]
            print(fmt2.format(
                str(clip),
                f"{last['test_acc']:.4f}",
                f"{last['loss']:.4f}",
                f"{last.get('clipped_frac', 0):.3f}",
                f"{last.get('condition', 0):.1f}",
            ))


    # Kronecker summary
    print()
    print("--- Kronecker sweep detail (full-batch, step 999) ---")
    fmt3 = "{:<25s} {:>8s} {:>8s}"
    print(fmt3.format("config", "acc", "loss"))
    print("-" * 45)
    kron_keys = sorted([k for k in results if k.startswith('kron')])
    for key in kron_keys:
        if results[key]:
            last = results[key][-1]
            print(fmt3.format(key, f"{last['test_acc']:.4f}", f"{last['loss']:.4f}"))


if __name__ == "__main__":
    main()
