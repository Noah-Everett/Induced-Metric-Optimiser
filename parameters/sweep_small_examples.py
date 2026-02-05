"""
Hyperparameter sweep for 2D test function optimisation.

Runs gradient descent on classic test functions (Beale, Rosenbrock,
Himmelblau, Ackley, Rastrigin) and records convergence metrics.

Usage::

    # Local (Optuna) backend
    python sweep_small_examples.py --optimiser adam --num_runs 100 --backend local

    # WandB backend
    python sweep_small_examples.py --optimiser sgd_metric --num_runs 50 --backend wandb

    # Sweep all functions at once
    python sweep_small_examples.py --optimiser adam --num_runs 100 --backend local --all_functions
"""

import time

import jax
import jax.numpy as jnp
import optax

from optimizer_registry import create_optimizer, needs_loss
from sweep_utils import SweepLogger, SweepRunner, setup_argparser

# Enable 64-bit precision for small-scale optimisation
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

@jax.jit
def beale_function(params):
    x, y = params
    return (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2

@jax.jit
def rosenbrock_function(params):
    x, y = params
    return (1.0 - x)**2 + 100.0 * (y - x**2)**2

@jax.jit
def himmelblau_function(params):
    x, y = params
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

@jax.jit
def ackley_function(params):
    x, y = params
    a, b, c = 20.0, 0.2, 2 * jnp.pi
    return (-a * jnp.exp(-b * jnp.sqrt(0.5 * (x**2 + y**2)))
            - jnp.exp(0.5 * (jnp.cos(c * x) + jnp.cos(c * y)))
            + a + jnp.e)

@jax.jit
def rastrigin_function(params):
    x, y = params
    A = 10.0
    return A * 2 + (x**2 - A * jnp.cos(2 * jnp.pi * x)) + (y**2 - A * jnp.cos(2 * jnp.pi * y))


TEST_FUNCTIONS = {
    "beale": {
        "func": beale_function,
        "grad": jax.jit(jax.grad(beale_function)),
        "initial": jnp.array([0.0, 0.0]),
        "tolerance": 1e-7,
    },
    "rosenbrock": {
        "func": rosenbrock_function,
        "grad": jax.jit(jax.grad(rosenbrock_function)),
        "initial": jnp.array([0.0, 0.0]),
        "tolerance": 1e-7,
    },
    "himmelblau": {
        "func": himmelblau_function,
        "grad": jax.jit(jax.grad(himmelblau_function)),
        "initial": jnp.array([0.0, 0.0]),
        "tolerance": 1e-6,
    },
    "ackley": {
        "func": ackley_function,
        "grad": jax.jit(jax.grad(ackley_function)),
        "initial": jnp.array([2.0, 2.0]),
        "tolerance": 1e-6,
    },
    "rastrigin": {
        "func": rastrigin_function,
        "grad": jax.jit(jax.grad(rastrigin_function)),
        "initial": jnp.array([2.0, 2.0]),
        "tolerance": 1e-6,
    },
}


# ---------------------------------------------------------------------------
# Fixed configuration
# ---------------------------------------------------------------------------

MAX_ITERATIONS = 1000


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train(config, seed, logger, optimizer_name, function_name):
    """Run gradient descent on a single test function.

    Returns
    -------
    dict
        Results dictionary with ``objective`` key for Optuna and ``summary``.
    """
    func_info = TEST_FUNCTIONS[function_name]
    func = func_info["func"]
    grad_func = func_info["grad"]
    tolerance = func_info["tolerance"]

    optimizer = create_optimizer(optimizer_name, config)
    opt_state = optimizer.init(func_info["initial"])
    params = func_info["initial"].copy()
    use_loss = needs_loss(optimizer_name)

    converged = False
    iterations = 0
    runtime = 0.0

    for i in range(MAX_ITERATIONS):
        start = time.time()

        grads = grad_func(params)

        if use_loss:
            current_loss = func(params)
            updates, opt_state = optimizer.update(grads, opt_state, current_loss, params)
        else:
            updates, opt_state = optimizer.update(grads, opt_state, params)

        params = optax.apply_updates(params, updates)
        runtime += max(time.time() - start, 0.0)

        current_value = float(func(params))
        iterations = i + 1

        logger.log({
            "iteration": i,
            "function_value": current_value,
            "runtime_seconds": runtime,
        })

        if current_value <= tolerance:
            converged = True
            break

    final_value = float(func(params))

    summary = {
        "function": function_name,
        "final_value": final_value,
        "iterations": iterations,
        "converged": converged,
        "runtime_seconds": runtime,
        "sweep_metric": final_value + (0.0 if converged else 100.0) + 0.01 * iterations,
    }

    return {"objective": summary["sweep_metric"], "summary": summary}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = setup_argparser("Small Examples (2D Test Functions) Hyperparameter Sweep")
    parser.add_argument(
        "--function", type=str, default=None,
        choices=list(TEST_FUNCTIONS.keys()),
        help="Test function to optimise (default: run all)",
    )
    parser.add_argument(
        "--all_functions", action="store_true",
        help="Run sweep for all test functions",
    )
    args = parser.parse_args()

    functions_to_run = list(TEST_FUNCTIONS.keys()) if (args.all_functions or args.function is None) else [args.function]

    for function_name in functions_to_run:
        task_tag = f"small_examples_{function_name}"

        task_fixed_params = {
            "max_iterations": {"value": MAX_ITERATIONS},
        }

        def make_train_fn(fn_name):
            def _train(config, seed, logger):
                return train(config, seed, logger, args.optimiser, fn_name)
            return _train

        runner = SweepRunner(
            backend=args.backend,
            project="induced_metric",
            task_tag=task_tag,
            optimizer_name=args.optimiser,
            args=args,
            task_fixed_params=task_fixed_params,
            results_dir=args.results_dir,
        )
        runner.run(make_train_fn(function_name))


if __name__ == "__main__":
    main()
