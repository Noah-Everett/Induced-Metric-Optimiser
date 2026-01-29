"""
CIFAR-10 ResNet18 hyperparameter sweep.

Usage::

    python sweep_cifar10_resnet18.py --optimiser adam --num_runs 50 --backend wandb
    python sweep_cifar10_resnet18.py --optimiser sgd_learn_scalar --num_runs 100 --backend local
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from shared_models import ResNet18
from optimizer_registry import create_optimizer, needs_loss
from sweep_utils import SweepRunner, setup_argparser

# Print JAX device information
print(f"JAX devices: {jax.devices()}")
print(f"JAX default backend: {jax.default_backend()}")

# Parse CLI
parser = setup_argparser("CIFAR-10 ResNet18 Hyperparameter Sweep")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cifar10():
    """Load and preprocess CIFAR-10 dataset with error handling."""
    import tensorflow as tf
    import shutil

    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Loading CIFAR-10 (attempt {attempt + 1}/{max_retries})...")
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

            assert x_train.shape == (50000, 32, 32, 3), f"Unexpected train shape: {x_train.shape}"
            assert x_test.shape == (10000, 32, 32, 3), f"Unexpected test shape: {x_test.shape}"

            print("CIFAR-10 loaded successfully!")
            break

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Clearing cache and retrying...")
                cache_dir = os.path.expanduser("~/.keras/datasets/")
                for item in ["cifar-10-batches-py.tar.gz", "cifar-10-batches-py"]:
                    path = os.path.join(cache_dir, item)
                    if os.path.exists(path):
                        if os.path.isdir(path):
                            shutil.rmtree(path)
                        else:
                            os.remove(path)
                time.sleep(2)
            else:
                raise e

    x_train = jnp.array(x_train, dtype=jnp.float32) / 255.0
    x_test = jnp.array(x_test, dtype=jnp.float32) / 255.0
    y_train = jnp.array(y_train.flatten(), dtype=jnp.int32)
    y_test = jnp.array(y_test.flatten(), dtype=jnp.int32)
    return x_train, y_train, x_test, y_test


def create_data_loaders(x_train, y_train, x_test, y_test, batch_size, seed):
    n_test_batches = len(x_test) // batch_size
    test_batches = [
        (x_test[i * batch_size:(i + 1) * batch_size],
         y_test[i * batch_size:(i + 1) * batch_size])
        for i in range(n_test_batches)
    ]
    return (x_train, y_train, batch_size), test_batches


def create_shuffled_batches(x_train, y_train, batch_size, epoch_seed):
    key = jax.random.PRNGKey(epoch_seed)
    perm = jax.random.permutation(key, len(x_train))
    x_train_shuffled = x_train[perm]
    y_train_shuffled = y_train[perm]

    n_train_batches = len(x_train) // batch_size
    train_batches = [
        (x_train_shuffled[i * batch_size:(i + 1) * batch_size],
         y_train_shuffled[i * batch_size:(i + 1) * batch_size])
        for i in range(n_train_batches)
    ]
    return train_batches


# ---------------------------------------------------------------------------
# Loss / accuracy helpers
# ---------------------------------------------------------------------------

def compute_full_accuracy(variables, data_batches, model):
    total_acc = 0.0
    total_samples = 0
    for x_batch, y_batch in data_batches:
        logits = model.apply(variables, x_batch, train=False)
        predictions = jnp.argmax(logits, axis=-1)
        batch_acc = jnp.mean(predictions == y_batch)
        batch_size = len(x_batch)
        total_acc += batch_acc * batch_size
        total_samples += batch_size
    return total_acc / total_samples


# ---------------------------------------------------------------------------
# Generic training function
# ---------------------------------------------------------------------------

def train(config, seed, logger):
    x_train, y_train, x_test, y_test = load_cifar10()

    model = ResNet18(num_classes=10)
    batch_size = config.get("batch_size", 1024)
    n_epochs = config.get("n_epochs", 400)
    train_data, test_batches = create_data_loaders(
        x_train, y_train, x_test, y_test, batch_size, seed
    )
    x_train, y_train, batch_size = train_data

    key = jax.random.PRNGKey(seed)
    dummy_input = jnp.ones((1, 32, 32, 3))
    variables = model.init(key, dummy_input, train=True)
    params = variables["params"]
    batch_stats = variables.get("batch_stats", {})

    optimizer = create_optimizer(args.optimiser, config)
    opt_state = optimizer.init(params)
    use_loss = needs_loss(args.optimiser)

    if use_loss:
        @jax.jit
        def train_step(params, batch_stats, opt_state, x, y):
            def loss_fn_with_batch_stats(p):
                variables = {"params": p, "batch_stats": batch_stats}
                logits, new_batch_stats = model.apply(
                    variables, x, train=True, mutable=["batch_stats"]
                )
                return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean(), new_batch_stats

            (loss, new_batch_stats), grads = jax.value_and_grad(
                loss_fn_with_batch_stats, has_aux=True
            )(params)
            updates, opt_state_new = optimizer.update(grads, opt_state, loss, params)
            params_new = optax.apply_updates(params, updates)
            return params_new, new_batch_stats["batch_stats"], opt_state_new, loss
    else:
        @jax.jit
        def train_step(params, batch_stats, opt_state, x, y):
            def loss_fn_with_batch_stats(p):
                variables = {"params": p, "batch_stats": batch_stats}
                logits, new_batch_stats = model.apply(
                    variables, x, train=True, mutable=["batch_stats"]
                )
                return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean(), new_batch_stats

            (loss, new_batch_stats), grads = jax.value_and_grad(
                loss_fn_with_batch_stats, has_aux=True
            )(params)
            updates, opt_state_new = optimizer.update(grads, opt_state, params)
            params_new = optax.apply_updates(params, updates)
            return params_new, new_batch_stats["batch_stats"], opt_state_new, loss

    max_val_acc = 0.0
    max_acc_epoch = 0
    train_time = 0.0

    for epoch in range(n_epochs):
        train_batches = create_shuffled_batches(x_train, y_train, batch_size, seed + epoch)
        epoch_losses = []

        train_time += time.time()
        for x_batch, y_batch in train_batches:
            params, batch_stats, opt_state, loss = train_step(
                params, batch_stats, opt_state, x_batch, y_batch
            )
            epoch_losses.append(loss)
        train_time -= time.time()

        avg_loss = float(jnp.mean(jnp.array(epoch_losses)))

        if epoch % args.val_freq == 0 or epoch == n_epochs - 1:
            variables = {"params": params, "batch_stats": batch_stats}
            train_acc = float(compute_full_accuracy(variables, train_batches, model))
            test_acc = float(compute_full_accuracy(variables, test_batches, model))
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
            "architecture": "ResNet18",
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    task_fixed_params = {
        "batch_size": {"values": [1024]},
        "n_epochs": {"value": 400},
    }

    runner = SweepRunner(
        backend=args.backend,
        project="induced_metric",
        task_tag="cifar10_resnet18",
        optimizer_name=args.optimiser,
        args=args,
        task_fixed_params=task_fixed_params,
        results_dir=args.results_dir,
    )
    runner.run(train)
