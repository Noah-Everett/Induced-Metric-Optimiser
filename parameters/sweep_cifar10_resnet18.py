"""
CIFAR-10 ResNet18 hyperparameter sweep.

Usage::

    python sweep_cifar10_resnet18.py --optimiser adam --num_runs 50 --backend wandb
    python sweep_cifar10_resnet18.py --optimiser sgd_learn_scalar --num_runs 100 --backend local
"""

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

from shared_models import ResNet18
from optimizer_registry import create_optimizer, needs_loss, needs_hvp
from sweep_utils import SweepRunner, setup_argparser
from optimizer_diagnostics import collect_diagnostics
from optimisers.jax_derived_newton_diag import hutchinson_hvp_diag

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
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            assert x_train.shape == (50000, 32, 32, 3), f"Unexpected train shape: {x_train.shape}"
            assert x_test.shape == (10000, 32, 32, 3), f"Unexpected test shape: {x_test.shape}"
            break
        except Exception as e:
            if attempt < max_retries - 1:
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

    # Convert to JAX arrays and explicitly pin to default device (GPU if available)
    x_train = jax.device_put(jnp.array(x_train, dtype=jnp.float32) / 255.0)
    x_test = jax.device_put(jnp.array(x_test, dtype=jnp.float32) / 255.0)
    y_train = jax.device_put(jnp.array(y_train.flatten(), dtype=jnp.int32))
    y_test = jax.device_put(jnp.array(y_test.flatten(), dtype=jnp.int32))
    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train(config, seed, logger):
    x_train, y_train, x_test, y_test = load_cifar10()

    model = ResNet18(num_classes=10)
    batch_size = config.get("batch_size", 1024)
    n_epochs = config.get("n_epochs", 400)

    # Pre-batch data as arrays for jax.lax.scan (not Python lists)
    n_train_batches = len(x_train) // batch_size
    n_test_batches = len(x_test) // batch_size

    # Fixed batched arrays for accuracy computation (no shuffle needed)
    x_train_acc = x_train[:n_train_batches * batch_size].reshape(
        n_train_batches, batch_size, 32, 32, 3
    )
    y_train_acc = y_train[:n_train_batches * batch_size].reshape(
        n_train_batches, batch_size
    )
    x_test_batched = x_test[:n_test_batches * batch_size].reshape(
        n_test_batches, batch_size, 32, 32, 3
    )
    y_test_batched = y_test[:n_test_batches * batch_size].reshape(
        n_test_batches, batch_size
    )

    key = jax.random.PRNGKey(seed)
    dummy_input = jnp.ones((1, 32, 32, 3))
    variables = model.init(key, dummy_input, train=True)
    params = variables["params"]
    batch_stats = variables.get("batch_stats", {})

    optimizer = create_optimizer(args.optimiser, config)
    opt_state = optimizer.init(params)
    use_loss = needs_loss(args.optimiser)
    use_hvp = needs_hvp(args.optimiser)

    # Use jax.lax.scan to process all batches in a single GPU dispatch per epoch
    if use_hvp:
        @jax.jit
        def train_epoch(params, batch_stats, opt_state, x_batched, y_batched, rng_key):
            def step(carry, batch):
                params, batch_stats, opt_state, key = carry
                x, y = batch
                key, subkey = jax.random.split(key)
                def loss_fn(p):
                    variables = {"params": p, "batch_stats": batch_stats}
                    logits, mutated = model.apply(
                        variables, x, train=True, mutable=["batch_stats"]
                    )
                    return optax.softmax_cross_entropy_with_integer_labels(
                        logits, y
                    ).mean(), mutated
                def scalar_loss_fn(p):
                    return loss_fn(p)[0]
                (loss, mutated), grads = jax.value_and_grad(
                    loss_fn, has_aux=True
                )(params)
                batch_stats_new = mutated["batch_stats"]
                h_diag = hutchinson_hvp_diag(scalar_loss_fn, params, subkey)
                updates, opt_state_new = optimizer.update(grads, opt_state, h_diag, params)
                params_new = optax.apply_updates(params, updates)
                return (params_new, batch_stats_new, opt_state_new, key), loss
            (params, batch_stats, opt_state, _), losses = jax.lax.scan(
                step, (params, batch_stats, opt_state, rng_key), (x_batched, y_batched)
            )
            return params, batch_stats, opt_state, jnp.mean(losses)
    elif use_loss:
        @jax.jit
        def train_epoch(params, batch_stats, opt_state, x_batched, y_batched):
            def step(carry, batch):
                params, batch_stats, opt_state = carry
                x, y = batch
                def loss_fn(p):
                    variables = {"params": p, "batch_stats": batch_stats}
                    logits, mutated = model.apply(
                        variables, x, train=True, mutable=["batch_stats"]
                    )
                    return optax.softmax_cross_entropy_with_integer_labels(
                        logits, y
                    ).mean(), mutated
                (loss, mutated), grads = jax.value_and_grad(
                    loss_fn, has_aux=True
                )(params)
                batch_stats_new = mutated["batch_stats"]
                updates, opt_state_new = optimizer.update(grads, opt_state, loss, params)
                params_new = optax.apply_updates(params, updates)
                return (params_new, batch_stats_new, opt_state_new), loss
            (params, batch_stats, opt_state), losses = jax.lax.scan(
                step, (params, batch_stats, opt_state), (x_batched, y_batched)
            )
            return params, batch_stats, opt_state, jnp.mean(losses)
    else:
        @jax.jit
        def train_epoch(params, batch_stats, opt_state, x_batched, y_batched):
            def step(carry, batch):
                params, batch_stats, opt_state = carry
                x, y = batch
                def loss_fn(p):
                    variables = {"params": p, "batch_stats": batch_stats}
                    logits, mutated = model.apply(
                        variables, x, train=True, mutable=["batch_stats"]
                    )
                    return optax.softmax_cross_entropy_with_integer_labels(
                        logits, y
                    ).mean(), mutated
                (loss, mutated), grads = jax.value_and_grad(
                    loss_fn, has_aux=True
                )(params)
                batch_stats_new = mutated["batch_stats"]
                updates, opt_state_new = optimizer.update(grads, opt_state, params)
                params_new = optax.apply_updates(params, updates)
                return (params_new, batch_stats_new, opt_state_new), loss
            (params, batch_stats, opt_state), losses = jax.lax.scan(
                step, (params, batch_stats, opt_state), (x_batched, y_batched)
            )
            return params, batch_stats, opt_state, jnp.mean(losses)

    # Scan-based accuracy (full dataset too large for single forward pass)
    @jax.jit
    def count_correct_scan(params, batch_stats, x_batched, y_batched):
        variables = {"params": params, "batch_stats": batch_stats}
        def count_batch(total, batch):
            x, y = batch
            logits = model.apply(variables, x, train=False)
            return total + jnp.sum(jnp.argmax(logits, axis=-1) == y), None
        total_correct, _ = jax.lax.scan(
            count_batch, jnp.array(0, dtype=jnp.int32), (x_batched, y_batched)
        )
        return total_correct

    max_val_acc = 0.0
    max_acc_epoch = 0
    train_time = 0.0
    pruned = False
    prev_diag_loss = None
    hvp_key = jax.random.PRNGKey(seed + 999)

    for epoch in range(n_epochs):
        # Shuffle and reshape training data for this epoch
        epoch_key = jax.random.PRNGKey(seed + epoch)
        perm = jax.random.permutation(epoch_key, len(x_train))
        x_epoch = x_train[perm][:n_train_batches * batch_size].reshape(
            n_train_batches, batch_size, 32, 32, 3
        )
        y_epoch = y_train[perm][:n_train_batches * batch_size].reshape(
            n_train_batches, batch_size
        )

        epoch_start = time.time()
        if use_hvp:
            hvp_key, hvp_epoch_key = jax.random.split(hvp_key)
            params, batch_stats, opt_state, avg_loss = train_epoch(
                params, batch_stats, opt_state, x_epoch, y_epoch, hvp_epoch_key
            )
        else:
            params, batch_stats, opt_state, avg_loss = train_epoch(
                params, batch_stats, opt_state, x_epoch, y_epoch
            )
        avg_loss = float(avg_loss)
        epoch_time = time.time() - epoch_start
        train_time += epoch_time

        if epoch == 0 and args.index == 0:
            print(f"First epoch (scan, {n_train_batches} batches): {epoch_time:.3f}s", flush=True)

        # Diagnostics (one fwd/bwd on first batch, train=False to avoid batch_stats mutation)
        diag_metrics = {}
        if getattr(args, 'diagnostics', False):
            def _diag_loss(p):
                variables = {"params": p, "batch_stats": batch_stats}
                logits = model.apply(variables, x_epoch[0], train=False)
                return optax.softmax_cross_entropy_with_integer_labels(
                    logits, y_epoch[0]
                ).mean()
            dl, dg = jax.value_and_grad(_diag_loss)(params)
            if use_hvp:
                hvp_key, diag_key = jax.random.split(hvp_key)
                dh = hutchinson_hvp_diag(_diag_loss, params, diag_key)
                du, _ = optimizer.update(dg, opt_state, dh, params)
            elif use_loss:
                du, _ = optimizer.update(dg, opt_state, dl, params)
            else:
                du, _ = optimizer.update(dg, opt_state, params)
            diag_metrics = collect_diagnostics(
                args.optimiser, opt_state, params,
                grads=dg, updates=du, loss=dl,
                config=config, prev_loss=prev_diag_loss,
            )
            prev_diag_loss = dl

        if epoch % args.val_freq == 0 or epoch == n_epochs - 1:
            n_train_samples = n_train_batches * batch_size
            n_test_samples = n_test_batches * batch_size
            train_correct = count_correct_scan(
                params, batch_stats, x_train_acc, y_train_acc
            )
            test_correct = count_correct_scan(
                params, batch_stats, x_test_batched, y_test_batched
            )
            train_acc = float(train_correct / n_train_samples)
            test_acc = float(test_correct / n_test_samples)

            if test_acc > max_val_acc:
                max_val_acc = test_acc
                max_acc_epoch = epoch

            logger.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "train_time_seconds": train_time,
                **diag_metrics,
            })

            # Check for pruning (minimize negative accuracy)
            if logger.report_and_check_prune(epoch, -test_acc):
                pruned = True
                break
        else:
            logger.log({"epoch": epoch, "train_loss": avg_loss, **diag_metrics})

    return {
        "objective": -max_val_acc,
        "summary": {
            "final_max_val_acc": max_val_acc,
            "final_max_acc_epoch": max_acc_epoch,
            "sweep_metric": -max_val_acc,
            "pruned": pruned,
            "architecture": "ResNet18",
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    task_fixed_params = {
        "batch_size": {"values": [args.batch_size if args.batch_size else 1024]},
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
