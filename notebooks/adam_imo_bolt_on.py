"""
Adam-IMO bolt-on probe on CIFAR-10 SmallCNN.

One-off probe — not a permanent task in ``analysis/task_configs.py``.  Output
lands in the standard ``results/cifar10_simple_cnn/<optimiser>/itr_0/run_*.csv``
layout via ``SweepLogger`` so anything downstream can read it like any other
sweep if someone wants to revisit later.

Findings: ``notes/adam-imo-bolt-on-results.md``.

Each invocation runs the explicit 5-LR x 3-seed grid for one (optimiser, ξ /
clip_norm) combination.  SLURM array submission is in
``notebooks/adam_imo_bolt_on.sbatch``.

Usage::

    python notebooks/adam_imo_bolt_on.py --optimiser adam
    python notebooks/adam_imo_bolt_on.py --optimiser adam_imo --xi 0.1
    python notebooks/adam_imo_bolt_on.py --optimiser adam_clip --clip_norm 1.0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


# Disable XLA autotuning (small kernels, autotune is pure overhead).
_xla_flag = '--xla_gpu_autotune_level=0'
_existing = os.environ.get('XLA_FLAGS', '')
if _xla_flag not in _existing:
    os.environ['XLA_FLAGS'] = f'{_existing} {_xla_flag}'.strip()
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'true')
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')

# Make sibling packages importable when invoked from any cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..'))
for _p in (_REPO, os.path.join(_REPO, 'parameters'),
            os.path.join(_REPO, 'analysis')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from optimizer_registry import create_optimizer
from optimizer_diagnostics import collect_diagnostics
from sweep_utils import SweepLogger


# ---------------------------------------------------------------------------
# Model — tiny CIFAR-10 CNN, ~90K params, used only by this probe.
# ---------------------------------------------------------------------------

class SmallCNN(nn.Module):
    """3 conv blocks (16 -> 32 -> 64), Dense(1024 -> 64) -> Dense(64 -> 10).

    Inlined here because no other task uses it; if a future probe wants the
    same model it can import this class from notebooks.adam_imo_bolt_on.
    """
    num_classes: int = 10

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.gelu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.gelu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.gelu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=64)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x


# ---------------------------------------------------------------------------
# Data + augmentation
# ---------------------------------------------------------------------------

_MEAN = jnp.array([0.4914, 0.4822, 0.4465], dtype=jnp.float32)
_STD = jnp.array([0.2470, 0.2435, 0.2616], dtype=jnp.float32)


def load_cifar10():
    import tensorflow as tf
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    assert x_train.shape == (50000, 32, 32, 3), x_train.shape

    def _norm(x):
        x = jnp.array(x, dtype=jnp.float32) / 255.0
        return (x - _MEAN) / _STD

    return (
        jax.device_put(_norm(x_train)),
        jax.device_put(jnp.array(y_train.flatten(), dtype=jnp.int32)),
        jax.device_put(_norm(x_test)),
        jax.device_put(jnp.array(y_test.flatten(), dtype=jnp.int32)),
    )


def _augment_batch(images, key):
    """Random 32x32 crop with 4-px reflect pad + horizontal flip @ 0.5."""
    pad = 4
    padded = jnp.pad(images, ((0, 0), (pad, pad), (pad, pad), (0, 0)),
                      mode='reflect')
    k_top, k_left, k_flip = jax.random.split(key, 3)
    n = images.shape[0]
    tops = jax.random.randint(k_top, (n,), 0, 2 * pad + 1)
    lefts = jax.random.randint(k_left, (n,), 0, 2 * pad + 1)
    flips = jax.random.bernoulli(k_flip, p=0.5, shape=(n,))

    def crop_one(p, top, left, flip):
        cropped = jax.lax.dynamic_slice(p, (top, left, 0), (32, 32, 3))
        return jnp.where(flip, cropped[:, ::-1, :], cropped)

    return jax.vmap(crop_one)(padded, tops, lefts, flips)


# ---------------------------------------------------------------------------
# Single (lr, seed) run — logs to a SweepLogger.
# ---------------------------------------------------------------------------

def train_one(*, optimiser_name, base_lr, seed, n_epochs, batch_size,
                extra_config, data, logger, diagnostics, diverge_mult=10.0):
    x_train, y_train, x_test, y_test = data
    n_train = x_train.shape[0]
    steps_per_epoch = n_train // batch_size
    n_test_batches = x_test.shape[0] // batch_size
    total_steps = steps_per_epoch * n_epochs

    schedule = optax.cosine_decay_schedule(
        init_value=base_lr, decay_steps=max(total_steps, 1), alpha=0.0,
    )
    config = dict(extra_config)
    config['learning_rate'] = schedule

    optimizer = create_optimizer(optimiser_name, config)
    init_key = jax.random.PRNGKey(seed)
    model = SmallCNN(num_classes=10)
    variables = model.init(init_key, jnp.ones((1, 32, 32, 3)))
    params = variables['params']
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(p, x, y):
        logits = model.apply({'params': p}, x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    @jax.jit
    def train_step(carry, batch):
        params, opt_state = carry
        x, y = batch
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    @jax.jit
    def run_epoch(params, opt_state, xb, yb):
        return jax.lax.scan(train_step, (params, opt_state), (xb, yb))

    @jax.jit
    def count_correct(params, xb, yb):
        def _count(total, batch):
            x, y = batch
            logits = model.apply({'params': params}, x)
            return total + jnp.sum(jnp.argmax(logits, axis=-1) == y), None
        total, _ = jax.lax.scan(_count, jnp.array(0, dtype=jnp.int32), (xb, yb))
        return total

    x_test_b = x_test[:n_test_batches * batch_size].reshape(
        n_test_batches, batch_size, 32, 32, 3)
    y_test_b = y_test[:n_test_batches * batch_size].reshape(
        n_test_batches, batch_size)
    x_train_eval_b = x_train[:steps_per_epoch * batch_size].reshape(
        steps_per_epoch, batch_size, 32, 32, 3)
    y_train_eval_b = y_train[:steps_per_epoch * batch_size].reshape(
        steps_per_epoch, batch_size)

    initial_loss = None
    diverged = False
    diverge_epoch = -1
    max_test_acc = 0.0
    max_test_acc_epoch = 0
    train_time = 0.0
    last_train_loss = float('nan')

    for epoch in range(n_epochs):
        epoch_key = jax.random.PRNGKey(seed * 1_000_003 + epoch)
        perm_key, aug_key = jax.random.split(epoch_key)
        perm = jax.random.permutation(perm_key, n_train)
        x_epoch = x_train[perm][:steps_per_epoch * batch_size]
        y_epoch = y_train[perm][:steps_per_epoch * batch_size]
        x_epoch = _augment_batch(x_epoch, aug_key)
        x_epoch = x_epoch.reshape(steps_per_epoch, batch_size, 32, 32, 3)
        y_epoch = y_epoch.reshape(steps_per_epoch, batch_size)

        t0 = time.time()
        (params, opt_state), losses = run_epoch(params, opt_state,
                                                  x_epoch, y_epoch)
        avg_loss = float(jnp.mean(losses))
        train_time += time.time() - t0
        last_train_loss = avg_loss

        if initial_loss is None:
            initial_loss = avg_loss
        if not np.isfinite(avg_loss) or avg_loss > diverge_mult * initial_loss:
            diverged = True
            diverge_epoch = epoch
            logger.log({
                'epoch': epoch, 'train_loss': avg_loss,
                'train_acc': float('nan'), 'test_acc': float('nan'),
                'train_time_seconds': train_time, 'diverged': 1,
            })
            break

        train_correct = int(count_correct(params, x_train_eval_b, y_train_eval_b))
        test_correct = int(count_correct(params, x_test_b, y_test_b))
        train_acc = train_correct / float(steps_per_epoch * batch_size)
        test_acc = test_correct / float(n_test_batches * batch_size)
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            max_test_acc_epoch = epoch

        row = {
            'epoch': epoch, 'train_loss': avg_loss,
            'train_acc': train_acc, 'test_acc': test_acc,
            'train_time_seconds': train_time, 'diverged': 0,
        }
        if diagnostics:
            xb0, yb0 = x_epoch[0], y_epoch[0]
            dl, dg = jax.value_and_grad(loss_fn)(params, xb0, yb0)
            du, _ = optimizer.update(dg, opt_state, params)
            row.update(collect_diagnostics(
                optimiser_name, opt_state, params,
                grads=dg, updates=du, loss=dl, config=config,
            ))
        logger.log(row)

    summary = {
        # Canonical keys (match other tasks' summary shape).
        'final_max_val_acc': max_test_acc,
        'final_max_acc_epoch': max_test_acc_epoch,
        'sweep_metric': -max_test_acc,
        'pruned': False,
        'architecture': 'SmallCNN',
        'optimizer': optimiser_name,
        'seed': seed,
        # Probe-specific extras.
        'diverged': diverged,
        'diverge_epoch': diverge_epoch,
        'final_train_loss': last_train_loss,
        'wall_time_seconds': train_time,
    }
    return summary


# ---------------------------------------------------------------------------
# Run-index assignment — keeps indices unique across SLURM array tasks
# writing to the same ``adam_imo/itr_0/`` directory.
# ---------------------------------------------------------------------------

_LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
_XI_GRID = [0.01, 0.1, 1.0]
_NUM_SEEDS = 3


def _run_index(optimiser, lr, seed, xi):
    lr_idx = _LR_GRID.index(lr)
    if optimiser == 'adam_imo':
        xi_idx = _XI_GRID.index(xi)
        return xi_idx * len(_LR_GRID) * _NUM_SEEDS + lr_idx * _NUM_SEEDS + seed
    return lr_idx * _NUM_SEEDS + seed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--optimiser', required=True,
                        choices=['adam', 'adamw', 'adam_imo', 'adam_clip'])
    parser.add_argument('--xi', type=float, default=None,
                        help='Required for adam_imo')
    parser.add_argument('--clip_norm', type=float, default=None,
                        help='Required for adam_clip')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--iteration', type=int, default=0)
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--diagnostics', action='store_true')
    args = parser.parse_args()

    if args.optimiser == 'adam_imo' and args.xi is None:
        parser.error('--xi is required for adam_imo')
    if args.optimiser == 'adam_clip' and args.clip_norm is None:
        parser.error('--clip_norm is required for adam_clip')

    extra_config = {
        'beta1': args.beta1,
        'beta2': args.beta2,
        'eps': args.eps,
        'weight_decay': args.weight_decay,
    }
    if args.optimiser == 'adam_imo':
        extra_config['xi'] = args.xi
    if args.optimiser == 'adam_clip':
        extra_config['clip_norm'] = args.clip_norm

    print('Loading CIFAR-10 ...', flush=True)
    data = load_cifar10()
    print(f'Devices: {jax.devices()}', flush=True)

    out_dir = (Path(args.results_dir) / 'cifar10_simple_cnn'
                / args.optimiser / f'itr_{args.iteration}')
    out_dir.mkdir(parents=True, exist_ok=True)

    for lr in _LR_GRID:
        for seed in range(_NUM_SEEDS):
            run_index = _run_index(args.optimiser, lr, seed, args.xi)
            tag_extra = ''
            if args.optimiser == 'adam_imo':
                tag_extra = f' xi={args.xi:g}'
            elif args.optimiser == 'adam_clip':
                tag_extra = f' clip={args.clip_norm:g}'
            print(f"\n=== run_{run_index}: {args.optimiser}{tag_extra}  "
                  f"lr={lr:g}  seed={seed} ===", flush=True)

            logger = SweepLogger(
                backend='local',
                project='induced_metric',
                tags=[args.optimiser, 'cifar10_simple_cnn',
                      f'itr_{args.iteration}', f'run_{run_index}'],
                local_dir=str(out_dir),
                run_index=run_index,
            )
            run_config = {
                'learning_rate': float(lr),
                'optimizer': args.optimiser,
                'n_epochs': args.n_epochs,
                'batch_size': args.batch_size,
                **extra_config,
            }
            logger.init_run(run_config)

            t_start = time.time()
            summary = train_one(
                optimiser_name=args.optimiser,
                base_lr=float(lr),
                seed=seed,
                n_epochs=args.n_epochs,
                batch_size=args.batch_size,
                extra_config=extra_config,
                data=data,
                logger=logger,
                diagnostics=args.diagnostics,
            )
            summary['total_wall_seconds'] = time.time() - t_start
            logger.finish(summary)

            print(f"  -> max_test_acc={summary['final_max_val_acc']:.4f}  "
                  f"diverged={summary['diverged']}  "
                  f"wall={summary['total_wall_seconds']:.1f}s", flush=True)


if __name__ == '__main__':
    main()
