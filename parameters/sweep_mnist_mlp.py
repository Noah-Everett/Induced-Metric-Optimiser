"""
MNIST MLP hyperparameter sweep.

Usage::

    python sweep_mnist_mlp.py --optimiser adam --num_runs 50 --backend wandb
    python sweep_mnist_mlp.py --optimiser sgd_learn_scalar --num_runs 100 --backend local
"""

import functools
import os
import time

# Disable XLA autotuning to avoid compilation hangs on MIG partitions
# Autotuning tries 100+ kernel configs which can hang/timeout on MIG GPUs
_xla_flag = '--xla_gpu_autotune_level=0'
_existing = os.environ.get('XLA_FLAGS', '')
if _xla_flag not in _existing:
    os.environ['XLA_FLAGS'] = f'{_existing} {_xla_flag}'.strip()

# Configure JAX for optimal performance on both GPU and CPU
# These settings work on M1 Pro (Metal/CPU), CUDA GPUs, and CPU-only systems
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'  # Preallocate device memory
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'  # Use platform-specific allocator
# Note: Do NOT set JAX_PLATFORMS - let JAX auto-detect (Metal on M1, CUDA on NVIDIA, CPU otherwise)

import jax
import jax.numpy as jnp
import optax
import tensorflow as tf

from shared_models import MLP
from optimizer_registry import create_optimizer, needs_loss
from sweep_utils import SweepRunner, setup_argparser

# Parse CLI
parser = setup_argparser("MNIST MLP Hyperparameter Sweep")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Convert to JAX arrays and explicitly pin to default device (GPU if available)
    x_train = jax.device_put(jnp.array(x_train.reshape(-1, 784) / 255.0, dtype=jnp.float32))
    x_test = jax.device_put(jnp.array(x_test.reshape(-1, 784) / 255.0, dtype=jnp.float32))
    y_train = jax.device_put(jnp.array(y_train))
    y_test = jax.device_put(jnp.array(y_test))
    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------------
# Loss / accuracy helpers
# ---------------------------------------------------------------------------

def loss_fn(params, x, y, model):
    logits = model.apply(params, x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


@functools.partial(jax.jit, static_argnums=(3,))
def count_correct(params, x, y, model):
    logits = model.apply(params, x)
    return jnp.sum(jnp.argmax(logits, axis=1) == y)


# ---------------------------------------------------------------------------
# Generic training function
# ---------------------------------------------------------------------------

def train(config, seed, logger):
    x_train, y_train, x_test, y_test = load_mnist()

    model = MLP(features=64, output_dim=10)
    batch_size = config.get("batch_size", 1024)
    n_epochs = config.get("n_epochs", 200)

    # Shuffle training data once and pre-batch as arrays for jax.lax.scan
    key = jax.random.PRNGKey(seed)
    perm = jax.random.permutation(key, len(x_train))
    x_train_shuffled = x_train[perm]
    y_train_shuffled = y_train[perm]

    n_train_batches = len(x_train) // batch_size
    x_train_batched = x_train_shuffled[:n_train_batches * batch_size].reshape(
        n_train_batches, batch_size, 784
    )
    y_train_batched = y_train_shuffled[:n_train_batches * batch_size].reshape(
        n_train_batches, batch_size
    )

    params = model.init(key, jnp.ones((1, 784)))

    if args.index == 0:
        print(f"\n=== Performance Diagnostics (run_0) ===", flush=True)
        print(f"JAX backend: {jax.default_backend()}", flush=True)
        print(f"JAX devices: {jax.devices()}", flush=True)
        print(f"Training batches: {n_train_batches} x {batch_size} (via jax.lax.scan)", flush=True)
        print(f"Data on device: {x_train.devices()}", flush=True)
        print(f"Params on device: {jax.tree_util.tree_leaves(params)[0].devices()}", flush=True)
        print("=" * 40 + "\n", flush=True)

    optimizer = create_optimizer(args.optimiser, config)
    opt_state = optimizer.init(params)
    use_loss = needs_loss(args.optimiser)

    # Use jax.lax.scan to process all batches in a single GPU dispatch per epoch
    if use_loss:
        @jax.jit
        def train_epoch(params, opt_state, x_batched, y_batched):
            def step(carry, batch):
                params, opt_state = carry
                x, y = batch
                loss, grads = jax.value_and_grad(lambda p: loss_fn(p, x, y, model))(params)
                updates, opt_state = optimizer.update(grads, opt_state, loss, params)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), loss
            (params, opt_state), losses = jax.lax.scan(
                step, (params, opt_state), (x_batched, y_batched)
            )
            return params, opt_state, jnp.mean(losses)
    else:
        @jax.jit
        def train_epoch(params, opt_state, x_batched, y_batched):
            def step(carry, batch):
                params, opt_state = carry
                x, y = batch
                loss, grads = jax.value_and_grad(lambda p: loss_fn(p, x, y, model))(params)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), loss
            (params, opt_state), losses = jax.lax.scan(
                step, (params, opt_state), (x_batched, y_batched)
            )
            return params, opt_state, jnp.mean(losses)

    max_val_acc = 0.0
    max_acc_epoch = 0
    train_time = 0.0
    val_time = 0.0
    pruned = False

    for epoch in range(n_epochs):
        epoch_start = time.time()
        params, opt_state, avg_loss = train_epoch(
            params, opt_state, x_train_batched, y_train_batched
        )
        avg_loss = float(avg_loss)
        epoch_time = time.time() - epoch_start
        train_time += epoch_time

        if epoch == 0 and args.index == 0:
            print(f"First epoch (scan, {n_train_batches} batches): {epoch_time:.3f}s", flush=True)

        if epoch % args.val_freq == 0 or epoch == n_epochs - 1:
            val_start = time.time()
            train_acc = float(count_correct(params, x_train, y_train, model) / len(y_train))
            test_acc = float(count_correct(params, x_test, y_test, model) / len(y_test))
            val_time += time.time() - val_start

            if test_acc > max_val_acc:
                max_val_acc = test_acc
                max_acc_epoch = epoch

            logger.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "train_time_seconds": train_time,
            })

            # Check for pruning (minimize negative accuracy)
            if logger.report_and_check_prune(epoch, -test_acc):
                pruned = True
                break
        else:
            logger.log({"epoch": epoch, "train_loss": avg_loss})

    return {
        "objective": -max_val_acc,
        "summary": {
            "final_max_val_acc": max_val_acc,
            "final_max_acc_epoch": max_acc_epoch,
            "sweep_metric": -max_val_acc,
            "pruned": pruned,
            "architecture": "MLP",
            "val_time_seconds": val_time,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    task_fixed_params = {
        "batch_size": {"values": [args.batch_size if args.batch_size else 1024]},
        "n_epochs": {"value": 200},
    }

    runner = SweepRunner(
        backend=args.backend,
        project="induced_metric",
        task_tag="mnist_mlp",
        optimizer_name=args.optimiser,
        args=args,
        task_fixed_params=task_fixed_params,
        results_dir=args.results_dir,
    )
    runner.run(train)
