"""
MNIST MLP hyperparameter sweep.

Usage::

    python sweep_mnist_mlp.py --optimiser adam --num_runs 50 --backend wandb
    python sweep_mnist_mlp.py --optimiser sgd_learn_scalar --num_runs 100 --backend local
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf

from shared_models import MLP
from optimizer_registry import create_optimizer, needs_loss
from sweep_utils import SweepRunner, setup_argparser

# Print JAX device information
print(f"JAX devices: {jax.devices()}")
print(f"JAX default backend: {jax.default_backend()}")

# Parse CLI
parser = setup_argparser("MNIST MLP Hyperparameter Sweep")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Convert to JAX arrays immediately to avoid CPU→GPU transfers every step
    x_train = jnp.array(x_train.reshape(-1, 784).astype(np.float32) / 255.0)
    x_test = jnp.array(x_test.reshape(-1, 784).astype(np.float32) / 255.0)
    y_train = jnp.array(y_train)
    y_test = jnp.array(y_test)
    return x_train, y_train, x_test, y_test


def create_data_loaders(x_train, y_train, x_test, y_test, batch_size, seed):
    key = jax.random.PRNGKey(seed)
    perm = jax.random.permutation(key, len(x_train))
    x_train = x_train[perm]
    y_train = y_train[perm]

    n_train = len(x_train) // batch_size
    n_test = len(x_test) // batch_size

    train_batches = [
        (x_train[i * batch_size:(i + 1) * batch_size],
         y_train[i * batch_size:(i + 1) * batch_size])
        for i in range(n_train)
    ]
    test_batches = [
        (x_test[i * batch_size:(i + 1) * batch_size],
         y_test[i * batch_size:(i + 1) * batch_size])
        for i in range(n_test)
    ]
    return train_batches, test_batches


# ---------------------------------------------------------------------------
# Loss / accuracy helpers
# ---------------------------------------------------------------------------

def loss_fn(params, x, y, model):
    logits = model.apply(params, x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


def compute_full_accuracy(params, data_batches, model):
    total_correct = 0
    total_samples = 0
    for x_batch, y_batch in data_batches:
        logits = model.apply(params, x_batch)
        total_correct += jnp.sum(jnp.argmax(logits, axis=1) == y_batch)
        total_samples += len(x_batch)
    return total_correct / total_samples


# ---------------------------------------------------------------------------
# Generic training function
# ---------------------------------------------------------------------------

def train(config, seed, logger):
    x_train, y_train, x_test, y_test = load_mnist()

    model = MLP(features=64, output_dim=10)
    batch_size = config.get("batch_size", 1024)
    n_epochs = config.get("n_epochs", 200)
    train_batches, test_batches = create_data_loaders(
        x_train, y_train, x_test, y_test, batch_size, seed
    )

    key = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.ones((1, 784)))

    # Warmup: Force compilation and GPU transfer
    warmup_start = time.time()
    test_array = jnp.ones((1000, 1000))
    _ = jnp.dot(test_array, test_array).block_until_ready()
    warmup_time = time.time() - warmup_start
    print(f"GPU warmup time: {warmup_time:.3f}s", flush=True)
    print(f"Data on device: {train_batches[0][0].devices()}", flush=True)
    print(f"Params on device: {jax.tree_util.tree_leaves(params)[0].devices()}", flush=True)

    optimizer = create_optimizer(args.optimiser, config)
    opt_state = optimizer.init(params)
    use_loss = needs_loss(args.optimiser)

    if use_loss:
        @jax.jit
        def train_step(params, opt_state, x, y):
            loss, grads = jax.value_and_grad(lambda p: loss_fn(p, x, y, model))(params)
            updates, opt_state_new = optimizer.update(grads, opt_state, loss, params)
            return optax.apply_updates(params, updates), opt_state_new, loss
    else:
        @jax.jit
        def train_step(params, opt_state, x, y):
            loss, grads = jax.value_and_grad(lambda p: loss_fn(p, x, y, model))(params)
            updates, opt_state_new = optimizer.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state_new, loss

    max_val_acc = 0.0
    max_acc_epoch = 0
    train_time = 0.0

    for epoch in range(n_epochs):
        epoch_losses = []

        epoch_start = time.time()
        for batch_idx, (x_batch, y_batch) in enumerate(train_batches):
            if epoch == 0 and batch_idx < 3:
                batch_start = time.time()

            params, opt_state, loss = train_step(params, opt_state, x_batch, y_batch)

            if epoch == 0 and batch_idx < 3:
                batch_time = time.time() - batch_start
                label = "with JIT compilation" if batch_idx == 0 else "post-compilation"
                print(f"Batch {batch_idx} ({label}): {batch_time:.3f}s", flush=True)

            epoch_losses.append(loss)

        epoch_time = time.time() - epoch_start
        train_time += epoch_time

        if epoch == 0:
            print(f"First epoch total time: {epoch_time:.3f}s ({len(train_batches)} batches)", flush=True)

        avg_loss = float(jnp.mean(jnp.array(epoch_losses)))

        if epoch % args.val_freq == 0 or epoch == n_epochs - 1:
            train_acc = float(compute_full_accuracy(params, train_batches, model))
            test_acc = float(compute_full_accuracy(params, test_batches, model))
            if test_acc > max_val_acc:
                max_val_acc = test_acc
                max_acc_epoch = epoch

            logger.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "max_val_acc": max_val_acc,
                "max_acc_epoch": max_acc_epoch,
                "train_time_seconds": train_time,
            })
        else:
            logger.log({"epoch": epoch, "train_loss": avg_loss})

    return {
        "objective": -max_val_acc,
        "summary": {
            "final_max_val_acc": max_val_acc,
            "final_max_acc_epoch": max_acc_epoch,
            "sweep_metric": -max_val_acc,
            "architecture": "MLP",
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    task_fixed_params = {
        "batch_size": {"values": [1024]},
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
