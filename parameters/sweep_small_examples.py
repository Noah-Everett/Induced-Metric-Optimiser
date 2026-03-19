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
# Task-specific parameter overrides
# ---------------------------------------------------------------------------
# These match the original small_examples_analysis.ipynb analysis notebook.
# Key differences from the unified defaults:
# - beta=0 for fixed metric optimizers (no stochasticity to smooth)
# - eps searched for Adam/Muon/RMS (more sensitive for 2D functions)
# - weight_decay allows 0 for offdiag (irrelevant for 2D functions)
# - xi range extended for offdiag (1e-4 to 1e1)

def get_param_overrides(optimizer_name):
    """Return task-specific parameter overrides for small examples."""
    
    # Common: eps searched instead of fixed (for Adam-like optimizers)
    eps_searched = {"min": 1e-10, "max": 1e-6, "log": True}
    
    # beta=0 for fixed metric optimizers (deterministic gradients, no smoothing needed)
    beta_zero = {"value": 0.0}
    
    # weight_decay allowing zero (uniform, not log) - irrelevant for 2D functions
    weight_decay_uniform = {"min": 0.0, "max": 1e-2, "log": False}
    
    # Extended xi range for offdiag
    xi_extended = {"min": 1e-4, "max": 1e1, "log": True}
    
    if optimizer_name == "adam":
        return {"eps": eps_searched}
    
    if optimizer_name == "adamw":
        return {"eps": eps_searched}
    
    if optimizer_name == "muon":
        return {"eps": eps_searched}
    
    if optimizer_name in ("sgd_metric", "sgd_log_metric"):
        return {"beta": beta_zero}
    
    if optimizer_name == "sgd_rms":
        return {"beta": beta_zero, "eps": eps_searched}
    
    if optimizer_name in ("sgd_learn_scalar", "sgd_learn_scalar_log",
                          "sgd_learn_diag", "sgd_learn_diag_log"):
        # Learnable variants still search beta (they learn the metric)
        return {}
    
    if optimizer_name.startswith("sgd_offdiag_"):
        return {
            "xi": xi_extended,
            "weight_decay": weight_decay_uniform,
        }
    
    # sgd and others: use defaults
    return {}


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

    # JIT-compile the training step (gradient + optimizer update + param update)
    if use_loss:
        @jax.jit
        def train_step(params, opt_state):
            grads = grad_func(params)
            current_loss = func(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, current_loss, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state
    else:
        @jax.jit
        def train_step(params, opt_state):
            grads = grad_func(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state

    converged = False
    pruned = False
    iterations = 0
    runtime = 0.0

    for i in range(MAX_ITERATIONS):
        start = time.time()

        params, opt_state = train_step(params, opt_state)
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

        # Check for pruning (early stopping of unpromising trials)
        if logger.report_and_check_prune(i, current_value):
            pruned = True
            break

    final_value = float(func(params))

    summary = {
        "function": function_name,
        "final_value": final_value,
        "iterations": iterations,
        "converged": converged,
        "pruned": pruned,
        "runtime_seconds": runtime,
        "sweep_metric": final_value + (0.0 if converged else 100.0) + (50.0 if pruned else 0.0) + 0.01 * iterations,
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

        # Get task-specific parameter overrides
        param_overrides = get_param_overrides(args.optimiser)

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
            param_overrides=param_overrides,
        )
        runner.run(make_train_fn(function_name))


if __name__ == "__main__":
    main()
