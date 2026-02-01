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
        logger.finish({"final_metric": value})
    """

    def __init__(self, backend="wandb", project=None, tags=None, local_dir=None, run_index=None):
        self.backend = backend
        self.project = project
        self.tags = tags or []
        self.local_dir = local_dir
        self.run_index = run_index
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
    """JSON serialiser for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, bool):
        return bool(obj)
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
    """

    def __init__(self, backend, project, task_tag, optimizer_name, args,
                 task_fixed_params=None, results_dir="results"):
        self.backend = backend
        self.project = project
        self.task_tag = task_tag
        self.optimizer_name = optimizer_name
        self.args = args
        self.task_fixed_params = task_fixed_params or {}
        self.results_dir = results_dir

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

        sweep_params = get_sweep_parameters(self.optimizer_name)
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

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.RandomSampler(seed=42),
            study_name=f"{self.task_tag}_{self.optimizer_name}",
        )

        def objective(trial):
            config = suggest_optuna_parameters(self.optimizer_name, trial)
            for k, v in self.task_fixed_params.items():
                if isinstance(v, dict) and "value" in v:
                    config[k] = v["value"]
                elif isinstance(v, dict) and "values" in v:
                    config[k] = v["values"][0]
                else:
                    config[k] = v

            # Pass run_index to logger for predictable filenames
            logger = SweepLogger("local", local_dir=local_dir, run_index=self.args.index)
            logger.init_run(config)
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
        "--search", type=str, default="bayes",
        choices=["bayes", "grid", "random"],
        help="Search method for hyperparameter tuning (WandB backend only)",
    )
    parser.add_argument(
        "--backend", type=str, default="wandb",
        choices=["wandb", "local"],
        help="Sweep backend: 'wandb' for W&B sweeps, 'local' for Optuna + JSON",
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Base directory for local results (local backend only)",
    )
    parser.add_argument(
        "--iteration", type=int, default=0,
        help="Iteration/batch number for organizing multiple experiment runs",
    )
    return parser
