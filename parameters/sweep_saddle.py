"""
Hyperparameter sweep for controlled 2D saddle experiments.

Runs gradient descent on saddle and non-convex test functions and records
per-step trajectory data (position, loss, s_i values, Hessian estimates).

Test functions:
- saddle:          x^2 - y^2  (known saddle at origin, H = diag(2, -2))
- weighted_saddle: 3x^2 - y^2 (asymmetric eigenvalues, H = diag(6, -2))

Usage::

    # Single function, single optimizer
    python sweep_saddle.py --optimiser sgd_learn_diag_curv --function saddle --num_runs 50 --backend local --diagnostics

    # All functions
    python sweep_saddle.py --optimiser adam --num_runs 100 --backend local --all_functions --diagnostics

IMO-69
"""

import time

import jax
import jax.numpy as jnp
import optax

from optimizer_registry import create_optimizer, needs_loss, needs_hvp
from sweep_utils import SweepLogger, SweepRunner, setup_argparser
from optimizer_diagnostics import collect_diagnostics
from optimisers.jax_derived_newton_diag import hutchinson_hvp_diag, loss_grad_and_hvp_diag

# Enable 64-bit precision for small-scale optimisation
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

@jax.jit
def saddle_function(params):
    """L = x^2 - y^2.  Saddle at origin, H = diag(2, -2)."""
    x, y = params
    return x**2 - y**2


@jax.jit
def weighted_saddle_function(params):
    """L = 3x^2 - y^2.  Asymmetric saddle, H = diag(6, -2)."""
    x, y = params
    return 3.0 * x**2 - y**2


TEST_FUNCTIONS = {
    "saddle": {
        "func": saddle_function,
        "grad": jax.jit(jax.grad(saddle_function)),
        "initial": jnp.array([1.0, 1.0]),
        # No finite minimum — run full trajectory (tolerance never reached)
        "tolerance": float("-inf"),
    },
    "weighted_saddle": {
        "func": weighted_saddle_function,
        "grad": jax.jit(jax.grad(weighted_saddle_function)),
        "initial": jnp.array([1.0, 1.0]),
        "tolerance": float("-inf"),
    },
}


# ---------------------------------------------------------------------------
# Fixed configuration
# ---------------------------------------------------------------------------

MAX_ITERATIONS = 1000


# ---------------------------------------------------------------------------
# Task-specific parameter overrides
# ---------------------------------------------------------------------------
# Mirrors sweep_small_examples.py overrides — deterministic gradients,
# no stochasticity to smooth for fixed metric optimizers.

def get_param_overrides(optimizer_name):
    """Return task-specific parameter overrides for saddle experiments."""

    # eps searched instead of fixed (for Adam-like optimizers)
    eps_searched = {"min": 1e-10, "max": 1e-6, "log": True}

    # beta=0 for fixed metric optimizers (deterministic gradients)
    beta_zero = {"value": 0.0}

    # weight_decay allowing zero (uniform, not log)
    weight_decay_uniform = {"min": 0.0, "max": 1e-2, "log": False}

    # Extended xi range for offdiag
    xi_extended = {"min": 1e-4, "max": 1e1, "log": True}

    if optimizer_name in ("adam", "adamw"):
        return {"eps": eps_searched}

    if optimizer_name == "muon":
        return {"eps": eps_searched}

    if optimizer_name in ("sgd_metric", "sgd_log_metric"):
        return {"beta": beta_zero}

    if optimizer_name == "sgd_rms":
        return {"beta": beta_zero, "eps": eps_searched}

    if optimizer_name in ("sgd_learn_scalar", "sgd_learn_scalar_log",
                          "sgd_learn_diag", "sgd_learn_diag_log",
                          "sgd_learn_diag_curv", "sgd_learn_diag_curv_log"):
        return {}

    if optimizer_name.startswith("sgd_offdiag_"):
        return {
            "xi": xi_extended,
            "weight_decay": weight_decay_uniform,
        }

    return {}


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train(config, seed, logger, optimizer_name, function_name, diagnostics=False):
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
    use_hvp = needs_hvp(optimizer_name)
    hvp_key = jax.random.PRNGKey(seed + 999)

    # JIT-compile the training step
    if use_hvp:
        @jax.jit
        def train_step(params, opt_state, rng_key):
            _, grads, h_diag = loss_grad_and_hvp_diag(func, params, rng_key)
            updates, new_opt_state = optimizer.update(grads, opt_state, h_diag, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state
    elif use_loss:
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
    prev_diag_loss = None
    do_diag = diagnostics

    for i in range(MAX_ITERATIONS):
        start = time.time()

        if use_hvp:
            hvp_key, step_key = jax.random.split(hvp_key)
            params, opt_state = train_step(params, opt_state, step_key)
        else:
            params, opt_state = train_step(params, opt_state)
        runtime += max(time.time() - start, 0.0)

        current_value = float(func(params))
        iterations = i + 1

        # Diagnostics (2D functions — full tensor mode)
        diag_metrics = {}
        if do_diag:
            dg = grad_func(params)
            dl = func(params)
            if use_hvp:
                hvp_key, diag_key = jax.random.split(hvp_key)
                dh = hutchinson_hvp_diag(func, params, diag_key)
                du, _ = optimizer.update(dg, opt_state, dh, params)
            elif use_loss:
                du, _ = optimizer.update(dg, opt_state, dl, params)
            else:
                du, _ = optimizer.update(dg, opt_state, params)
            diag_metrics = collect_diagnostics(
                optimizer_name, opt_state, params,
                grads=dg, updates=du, loss=dl,
                config=config, prev_loss=prev_diag_loss,
            )
            prev_diag_loss = dl

        logger.log({
            "iteration": i,
            "function_value": current_value,
            "runtime_seconds": runtime,
            **diag_metrics,
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
    parser = setup_argparser("Saddle Experiment Hyperparameter Sweep (IMO-69)")
    parser.add_argument(
        "--function", type=str, default=None,
        choices=list(TEST_FUNCTIONS.keys()),
        help="Test function to optimise (default: run all)",
    )
    parser.add_argument(
        "--all_functions", action="store_true",
        help="Run sweep for all test functions",
    )
    parser.add_argument(
        "--no_clip", action="store_true",
        help="Disable metric_clip (set to 100.0) for learnable optimizers",
    )
    args = parser.parse_args()

    functions_to_run = list(TEST_FUNCTIONS.keys()) if (args.all_functions or args.function is None) else [args.function]

    for function_name in functions_to_run:
        task_tag = function_name

        task_fixed_params = {
            "max_iterations": {"value": MAX_ITERATIONS},
        }

        param_overrides = get_param_overrides(args.optimiser)

        if args.no_clip:
            param_overrides["metric_clip"] = {"value": 100.0}

        def make_train_fn(fn_name):
            def _train(config, seed, logger):
                return train(config, seed, logger, args.optimiser, fn_name,
                             diagnostics=getattr(args, 'diagnostics', False))
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
