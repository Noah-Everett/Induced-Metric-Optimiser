"""
Regression hyperparameter sweep on synthetic polynomial data.

Usage::

    python sweep_regression.py --optimiser adam --num_runs 50 --backend wandb
    python sweep_regression.py --optimiser sgd_learn_scalar --num_runs 100 --backend local
"""

import time
from itertools import product

import jax
import jax.numpy as jnp
import optax

from shared_models import MLP
from optimizer_registry import create_optimizer, needs_loss
from sweep_utils import SweepRunner, setup_argparser

# Parse CLI
parser = setup_argparser("Regression Hyperparameter Sweep")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def random_polynomial(key, order, degree):
    """Generate coefficients for a random polynomial with cross terms."""
    power_combinations = []
    for total_degree in range(degree + 1):
        for powers in product(range(total_degree + 1), repeat=order):
            if sum(powers) == total_degree:
                power_combinations.append(powers)

    num_terms = len(power_combinations)
    coeffs = jax.random.normal(key, (num_terms,))
    return coeffs, power_combinations


def evaluate_polynomial(coeffs, power_combinations, x):
    """Evaluate polynomial with cross terms at given points."""
    result = jnp.zeros(x.shape[0])
    for coeff, powers in zip(coeffs, power_combinations):
        term = jnp.ones(x.shape[0])
        for j, power in enumerate(powers):
            if power > 0:
                term *= x[:, j] ** power
        result += coeff * term
    return result


# Generate fixed synthetic data
point_key = jax.random.PRNGKey(0)
func_key = jax.random.PRNGKey(1)
order = 6
degree = 6

poly_coeffs, power_combinations = random_polynomial(func_key, order, degree)

x = jax.random.normal(point_key, (13312, 4))
y = evaluate_polynomial(poly_coeffs, power_combinations, x)
y_mean = jnp.mean(y)
y_std = jnp.std(y)
y = (y - y_mean) / y_std

x_train, y_train = x[:8192], y[:8192]
x_test, y_test = x[8192:], y[8192:]


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
# Loss / MSE helpers
# ---------------------------------------------------------------------------

def mse_fn(params, x, y, model):
    predictions = model.apply(params, x).squeeze()
    return jnp.mean((predictions - y) ** 2)


def compute_full_mse(params, data_batches, model):
    total_mse = 0.0
    total_samples = 0
    for x_batch, y_batch in data_batches:
        batch_mse = mse_fn(params, x_batch, y_batch, model)
        batch_size = len(x_batch)
        total_mse += batch_mse * batch_size
        total_samples += batch_size
    return total_mse / total_samples


# ---------------------------------------------------------------------------
# Generic training function
# ---------------------------------------------------------------------------

def train(config, seed, logger):
    model = MLP(features=32, output_dim=1)
    batch_size = config.get("batch_size", 1024)
    n_epochs = config.get("n_epochs", 200)
    train_batches, test_batches = create_data_loaders(
        x_train, y_train, x_test, y_test, batch_size, seed
    )

    key = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.ones((1, 4)))

    optimizer = create_optimizer(args.optimiser, config)
    opt_state = optimizer.init(params)
    use_loss = needs_loss(args.optimiser)

    if use_loss:
        @jax.jit
        def train_step(params, opt_state, x, y):
            loss, grads = jax.value_and_grad(lambda p: mse_fn(p, x, y, model))(params)
            updates, opt_state_new = optimizer.update(grads, opt_state, loss, params)
            return optax.apply_updates(params, updates), opt_state_new, loss
    else:
        @jax.jit
        def train_step(params, opt_state, x, y):
            loss, grads = jax.value_and_grad(lambda p: mse_fn(p, x, y, model))(params)
            updates, opt_state_new = optimizer.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state_new, loss

    min_val_loss = float("inf")
    min_loss_epoch = 0
    train_time = 0.0
    pruned = False

    for epoch in range(n_epochs):
        epoch_losses = []

        epoch_start = time.time()
        for x_batch, y_batch in train_batches:
            params, opt_state, loss = train_step(params, opt_state, x_batch, y_batch)
            epoch_losses.append(loss)
        train_time += time.time() - epoch_start

        avg_loss = float(jnp.mean(jnp.array(epoch_losses)))

        if epoch % args.val_freq == 0 or epoch == n_epochs - 1:
            train_mse = float(compute_full_mse(params, train_batches, model))
            test_mse = float(compute_full_mse(params, test_batches, model))
            if test_mse < min_val_loss:
                min_val_loss = test_mse
                min_loss_epoch = epoch

            logger.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "min_val_loss": min_val_loss,
                "min_loss_epoch": min_loss_epoch,
                "train_time_seconds": train_time,
            })

            # Check for pruning (minimize test MSE)
            if logger.report_and_check_prune(epoch, test_mse):
                pruned = True
                break
        else:
            logger.log({"epoch": epoch, "train_loss": avg_loss})

    return {
        "objective": min_val_loss,
        "summary": {
            "final_min_val_loss": min_val_loss,
            "final_min_loss_epoch": min_loss_epoch,
            "sweep_metric": min_val_loss,
            "pruned": pruned,
            "architecture": "MLP",
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
        task_tag="regression",
        optimizer_name=args.optimiser,
        args=args,
        task_fixed_params=task_fixed_params,
        results_dir=args.results_dir,
    )
    runner.run(train)
