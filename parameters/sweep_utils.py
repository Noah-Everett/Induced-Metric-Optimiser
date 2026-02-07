"""
Shared sweep infrastructure with WandB-optional backend.

Provides:
- ``SweepLogger``: abstraction over WandB vs local JSON logging.
- ``SweepRunner``: abstraction over WandB sweeps vs Optuna.
- ``setup_argparser``: shared CLI argument parser.
"""

import argparse
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np

from optimizer_registry import ALL_OPTIMIZERS, get_sweep_parameters, suggest_optuna_parameters


# ---------------------------------------------------------------------------
# Logging backend
# ---------------------------------------------------------------------------

class SweepLogger:
    """Thin abstraction over WandB vs local JSON logging.

    Usage::

        logger = SweepLogger(backend, project=..., tags=..., local_dir=...)
        logger.init_run(config)
        for epoch in range(n_epochs):
            logger.log({"epoch": epoch, "train_loss": loss})
            # Optional: report for pruning (local backend with Optuna)
            if logger.should_prune(epoch, train_loss):
                break
        logger.finish({"final_metric": value})
    """

    def __init__(self, backend="wandb", project=None, tags=None, local_dir=None, run_index=None, trial=None):
        self.backend = backend
        self.project = project
        self.tags = tags or []
        self.local_dir = local_dir
        self.run_index = run_index
        self.trial = trial  # Optuna trial for pruning support
        self._history = []
        self._config = {}
        self._run_dir = None

    def init_run(self, config=None):
        if self.backend == "wandb":
            import wandb
            wandb.init(project=self.project, tags=self.tags, config=config)
        else:
            self._config = dict(config) if config else {}
            self._history = []
            self._run_dir = Path(self.local_dir)
            self._run_dir.mkdir(parents=True, exist_ok=True)

            # Use run index instead of UUID for predictable filenames
            if self.run_index is not None:
                self._run_file = self._run_dir / f"run_{self.run_index}.json"
            else:
                # Fallback to UUID if no run_index provided (backward compatibility)
                run_id = uuid.uuid4().hex[:8]
                self._run_file = self._run_dir / f"{run_id}.json"

    def log(self, metrics):
        if self.backend == "wandb":
            import wandb
            wandb.log(metrics)
        else:
            self._history.append(metrics)

    def report_and_check_prune(self, step, value):
        """Report intermediate value and check if trial should be pruned.
        
        Parameters
        ----------
        step : int
            Current step/iteration/epoch number.
        value : float
            Intermediate objective value to report.
            
        Returns
        -------
        bool
            True if the trial should be pruned, False otherwise.
            
        Raises
        ------
        optuna.TrialPruned
            If trial should be pruned (only if raise_on_prune=True).
        """
        if self.backend == "local" and self.trial is not None:
            self.trial.report(value, step)
            return self.trial.should_prune()
        return False

    def finish(self, summary=None):
        if self.backend == "wandb":
            import wandb
            if summary:
                wandb.log(summary)
            wandb.finish()
        else:
            data = {
                "config": self._config,
                "history": self._history,
                "summary": summary or {},
            }
            with open(self._run_file, "w") as f:
                json.dump(data, f, indent=2, default=_json_default)

    @property
    def config(self):
        if self.backend == "wandb":
            import wandb
            return wandb.config
        return self._config


def _json_default(obj):
    """JSON serialiser for numpy, JAX, and other scalar/array types."""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # JAX arrays and other objects with .item() (scalars) or .tolist() (arrays)
    if hasattr(obj, "item") and callable(obj.item):
        return obj.item()
    if hasattr(obj, "tolist") and callable(obj.tolist):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

class SweepRunner:
    """Run a hyperparameter sweep using either WandB or Optuna.

    Parameters
    ----------
    backend : str
        ``"wandb"`` or ``"local"``.
    project : str
        WandB project name (used for both backends as an organizational label).
    task_tag : str
        Tag identifying the task (e.g. ``"mnist_mlp"``).
    optimizer_name : str
        Optimizer to sweep.
    args : argparse.Namespace
        Parsed CLI arguments (must have ``index``, ``search``, ``num_runs``).
    task_fixed_params : dict
        Fixed parameters added to every sweep config (e.g. batch_size, n_epochs).
    results_dir : str
        Base directory for local results.
    param_overrides : dict, optional
        Task-specific parameter overrides. Keys are parameter names, values are
        dicts with ``{"value": x}`` for fixed, ``{"values": [x, y]}`` for
        categorical, or ``{"min": lo, "max": hi, "log": bool}`` for ranges.
    """

    def __init__(self, backend, project, task_tag, optimizer_name, args,
                 task_fixed_params=None, results_dir="results", param_overrides=None):
        self.backend = backend
        self.project = project
        self.task_tag = task_tag
        self.optimizer_name = optimizer_name
        self.args = args
        self.task_fixed_params = task_fixed_params or {}
        self.results_dir = results_dir
        self.param_overrides = param_overrides or {}

    def run(self, train_fn):
        """Execute the sweep.

        Parameters
        ----------
        train_fn : callable
            ``train_fn(config, seed, logger)`` that runs one training run.
            It must call ``logger.log()`` and return a results dict.
        """
        if self.backend == "wandb":
            self._run_wandb(train_fn)
        else:
            self._run_local(train_fn)

    # ---- WandB backend ----

    def _run_wandb(self, train_fn):
        import wandb

        sweep_params = get_sweep_parameters(self.optimizer_name, overrides=self.param_overrides)
        sweep_params.update(self.task_fixed_params)

        sweep_config = {
            "name": f"{self.task_tag}_{self.optimizer_name}_{self.args.index}",
            "method": self.args.search,
            "metric": {"name": "sweep_metric", "goal": "minimize"},
            "early_terminate": {"type": "hyperband", "min_iter": 20},
            "parameters": sweep_params,
        }

        sweep_id = wandb.sweep(sweep_config, project=self.project)
        tags = [self.optimizer_name, self.task_tag,
                f"run_{self.args.index}", self.args.search]

        def _agent_fn():
            logger = SweepLogger("wandb", project=self.project, tags=tags)
            logger.init_run()
            config = dict(wandb.config)
            seed = np.random.randint(1, 1_000_000)
            start = time.time()
            results = train_fn(config, seed, logger)
            elapsed = time.time() - start
            summary = {
                "total_training_time_sec": elapsed,
                "total_training_time_min": elapsed / 60,
                "seed": seed,
                "optimizer": self.optimizer_name,
            }
            summary.update(results.get("summary", {}))
            logger.finish(summary)

        print(f"Starting WandB sweep: {self.task_tag}_{self.optimizer_name}")
        print(f"Sweep ID: {sweep_id}")
        wandb.agent(sweep_id, _agent_fn, count=self.args.num_runs)
        print("Sweep completed!")

    # ---- Local / Optuna backend ----

    def _run_local(self, train_fn):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Use iteration-based directory structure
        iteration = getattr(self.args, 'iteration', 0)  # Default to 0 if not provided
        local_dir = os.path.join(
            self.results_dir, self.task_tag, self.optimizer_name,
            f"itr_{iteration}"
        )

        # Select sampler based on --search argument (matches WandB backend behavior)
        search_method = getattr(self.args, 'search', 'random')
        seed = getattr(self.args, 'seed', 42)
        
        if search_method == "bayes":
            # TPE (Tree-structured Parzen Estimator) - Optuna's Bayesian optimization
            sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
        elif search_method == "grid":
            # Grid search requires predefined search space - fall back to QMC for better coverage
            sampler = optuna.samplers.QMCSampler(seed=seed)
        else:  # "random" or default
            sampler = optuna.samplers.RandomSampler(seed=seed)

        # Select pruner based on --pruner argument
        pruner_type = getattr(self.args, 'pruner', 'none')
        if pruner_type == "hyperband":
            # Hyperband - aggressive early stopping, good for many trials
            pruner = optuna.pruners.HyperbandPruner(
                min_resource=getattr(self.args, 'min_resource', 10),
                reduction_factor=3,
            )
        elif pruner_type == "median":
            # Median pruner - prune if worse than median of previous trials
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=getattr(self.args, 'min_resource', 10),
            )
        elif pruner_type == "percentile":
            # Percentile pruner - more aggressive than median
            pruner = optuna.pruners.PercentilePruner(
                percentile=25.0,
                n_startup_trials=5,
                n_warmup_steps=getattr(self.args, 'min_resource', 10),
            )
        else:  # "none" or default
            pruner = optuna.pruners.NopPruner()

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            study_name=f"{self.task_tag}_{self.optimizer_name}",
        )

        def objective(trial):
            config = suggest_optuna_parameters(self.optimizer_name, trial, overrides=self.param_overrides)
            for k, v in self.task_fixed_params.items():
                if isinstance(v, dict) and "value" in v:
                    config[k] = v["value"]
                elif isinstance(v, dict) and "values" in v:
                    config[k] = v["values"][0]
                else:
                    config[k] = v

            # Use trial.number + offset for unique, predictable filenames
            run_offset = getattr(self.args, 'run_offset', 0)
            logger = SweepLogger("local", local_dir=local_dir, run_index=run_offset + trial.number, trial=trial)
            logger.init_run(config)
            seed = np.random.randint(1, 1_000_000)
            start = time.time()
            try:
                results = train_fn(config, seed, logger)
            except optuna.TrialPruned:
                # Trial was pruned - still save partial results
                logger.finish({"pruned": True, "seed": seed})
                raise
            elapsed = time.time() - start
            summary = {
                "total_training_time_sec": elapsed,
                "total_training_time_min": elapsed / 60,
                "seed": seed,
                "optimizer": self.optimizer_name,
            }
            summary.update(results.get("summary", {}))
            logger.finish(summary)
            return results.get("objective", 0.0)

        print(f"Starting local sweep: {self.task_tag}_{self.optimizer_name}")
        print(f"Results dir: {local_dir}")
        study.optimize(objective, n_trials=self.args.num_runs, show_progress_bar=True)
        print("Sweep completed!")


# ---------------------------------------------------------------------------
# Shared argparser
# ---------------------------------------------------------------------------

def setup_argparser(description):
    """Create an argument parser with the standard sweep CLI arguments.

    Returns the parser (not yet parsed) so sweep scripts can add their own
    task-specific arguments before calling ``parser.parse_args()``.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--optimiser", type=str, required=True,
        choices=ALL_OPTIMIZERS,
        help="Optimizer to use for training",
    )
    parser.add_argument(
        "--num_runs", type=int, default=50,
        help="Number of runs for the sweep",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--val_freq", type=int, default=1,
        help="Frequency of validation computation (every N epochs)",
    )
    parser.add_argument(
        "--search", type=str, default="random",
        choices=["bayes", "grid", "random"],
        help="Search method: 'random' (default), 'bayes' (TPE), 'grid' (QMC for local)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampler reproducibility (local backend)",
    )
    parser.add_argument(
        "--backend", type=str, default="wandb",
        choices=["wandb", "local"],
        help="Sweep backend: 'wandb' for W&B sweeps, 'local' for Optuna + JSON",
    )
    parser.add_argument(
        "--pruner", type=str, default="none",
        choices=["none", "hyperband", "median", "percentile"],
        help="Pruner for early stopping (local backend): 'none', 'hyperband', 'median', 'percentile'",
    )
    parser.add_argument(
        "--min_resource", type=int, default=10,
        help="Minimum steps before pruning can occur (local backend)",
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Base directory for local results (local backend only)",
    )
    parser.add_argument(
        "--iteration", type=int, default=0,
        help="Iteration/batch number for organizing multiple experiment runs",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override batch size (if not set, uses task default)",
    )
    parser.add_argument(
        "--run_offset", type=int, default=0,
        help="Starting run index offset (for resuming partial sweeps)",
    )
    return parser
