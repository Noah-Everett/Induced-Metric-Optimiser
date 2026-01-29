"""
Central optimizer registry.

Single source of truth for all optimizer names, factory functions,
hyperparameter search spaces, and display metadata.

Adding a new optimizer requires only editing this file.
"""

import os
import sys

import optax

# Ensure the repo root is on the path so optimisers/ can be imported
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from optimisers.jax_fixed import custom_sgd, custom_sgd_log, custom_sgd_rms
from optimisers.jax_learnable_scalar import (
    custom_sgd_learnable_scalar,
    custom_sgd_log_learnable_scalar,
)
from optimisers.jax_learnable_diag import (
    custom_sgd_learnable_diag,
    custom_sgd_log_learnable_diag,
)
from optimisers.jax_offdiag import custom_sgd_offdiag


# ---------------------------------------------------------------------------
# Canonical optimizer list
# ---------------------------------------------------------------------------

ALL_OPTIMIZERS = [
    "adam",
    "adamw",
    "sgd",
    "muon",
    "sgd_metric",
    "sgd_log_metric",
    "sgd_rms",
    "sgd_learn_scalar",
    "sgd_learn_scalar_log",
    "sgd_learn_diag",
    "sgd_learn_diag_log",
    "sgd_offdiag_0_l",
    "sgd_offdiag_0_m",
    "sgd_offdiag_0_theta",
    "sgd_offdiag_l_0",
    "sgd_offdiag_l_l",
    "sgd_offdiag_l_m",
    "sgd_offdiag_l_theta",
    "sgd_offdiag_m_0",
    "sgd_offdiag_m_l",
    "sgd_offdiag_m_m",
    "sgd_offdiag_m_theta",
    "sgd_offdiag_theta_0",
    "sgd_offdiag_theta_l",
    "sgd_offdiag_theta_m",
    "sgd_offdiag_theta_theta",
]

LOG_OPTIMIZERS = frozenset({
    "sgd_log_metric",
    "sgd_learn_scalar_log",
    "sgd_learn_diag_log",
})

# Off-diagonal mode shorthand -> full mode name
_MODE_MAP = {
    "0": "zero",
    "l": "grad",
    "m": "momentum",
    "theta": "params",
}


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def _pick(config, key, default=None):
    """Get a value from config, returning default if absent."""
    return config[key] if key in config else default


def create_optimizer(name, config):
    """Create an optax GradientTransformation from a name and config dict.

    Parameters
    ----------
    name : str
        One of the names in ALL_OPTIMIZERS.
    config : dict
        Hyperparameters. Irrelevant keys are ignored.

    Returns
    -------
    optax.GradientTransformation
    """
    if name == "adam":
        return optax.adam(
            learning_rate=config["learning_rate"],
            b1=config.get("beta1", 0.9),
            b2=config.get("beta2", 0.999),
            eps=config.get("eps", 1e-8),
        )

    if name == "adamw":
        return optax.adamw(
            learning_rate=config["learning_rate"],
            b1=config.get("beta1", 0.9),
            b2=config.get("beta2", 0.999),
            eps=config.get("eps", 1e-8),
            weight_decay=config.get("weight_decay", 0.0),
        )

    if name == "sgd":
        return optax.sgd(
            learning_rate=config["learning_rate"],
            momentum=config.get("momentum", 0.9),
        )

    if name == "muon":
        return optax.contrib.muon(
            learning_rate=config["learning_rate"],
            adam_b1=config.get("adam_b1", 0.9),
            adam_b2=config.get("adam_b2", 0.999),
            eps=config.get("eps", 1e-8),
            beta=config.get("beta", 0.95),
            weight_decay=config.get("weight_decay", 0.0),
        )

    if name == "sgd_metric":
        return custom_sgd(
            learning_rate=config["learning_rate"],
            momentum=config.get("momentum", 0.9),
            xi=config.get("xi", 0.1),
            beta=config.get("beta", 0.8),
            weight_decay=config.get("weight_decay", 0.0),
        )

    if name == "sgd_log_metric":
        return custom_sgd_log(
            learning_rate=config["learning_rate"],
            momentum=config.get("momentum", 0.9),
            xi=config.get("xi", 0.1),
            beta=config.get("beta", 0.8),
            weight_decay=config.get("weight_decay", 0.0),
        )

    if name == "sgd_rms":
        return custom_sgd_rms(
            learning_rate=config["learning_rate"],
            momentum=config.get("momentum", 0.9),
            xi=config.get("xi", 0.1),
            beta=config.get("beta", 0.8),
            beta_rms=config.get("beta_rms", 0.99),
            weight_decay=config.get("weight_decay", 0.0),
            eps=config.get("eps", 1e-8),
        )

    if name == "sgd_learn_scalar":
        return custom_sgd_learnable_scalar(
            learning_rate=config["learning_rate"],
            momentum=config.get("momentum", 0.9),
            xi=config.get("xi", 0.1),
            beta=config.get("beta", 0.8),
            weight_decay=config.get("weight_decay", 0.0),
            metric_lr=config.get("metric_lr", 1e-3),
            metric_reg=config.get("metric_reg", 1e-4),
            metric_clip=config.get("metric_clip", 4.0),
        )

    if name == "sgd_learn_scalar_log":
        return custom_sgd_log_learnable_scalar(
            learning_rate=config["learning_rate"],
            momentum=config.get("momentum", 0.9),
            xi=config.get("xi", 0.1),
            beta=config.get("beta", 0.8),
            weight_decay=config.get("weight_decay", 0.0),
            metric_lr=config.get("metric_lr", 1e-3),
            metric_reg=config.get("metric_reg", 1e-4),
            metric_clip=config.get("metric_clip", 4.0),
        )

    if name == "sgd_learn_diag":
        return custom_sgd_learnable_diag(
            learning_rate=config["learning_rate"],
            momentum=config.get("momentum", 0.9),
            xi=config.get("xi", 0.1),
            beta=config.get("beta", 0.8),
            weight_decay=config.get("weight_decay", 0.0),
            metric_lr=config.get("metric_lr", 1e-3),
            metric_reg=config.get("metric_reg", 1e-4),
            metric_clip=config.get("metric_clip", 4.0),
        )

    if name == "sgd_learn_diag_log":
        return custom_sgd_log_learnable_diag(
            learning_rate=config["learning_rate"],
            momentum=config.get("momentum", 0.9),
            xi=config.get("xi", 0.1),
            beta=config.get("beta", 0.8),
            weight_decay=config.get("weight_decay", 0.0),
            metric_lr=config.get("metric_lr", 1e-3),
            metric_reg=config.get("metric_reg", 1e-4),
            metric_clip=config.get("metric_clip", 4.0),
        )

    if name.startswith("sgd_offdiag_"):
        parts = name.split("_")
        a_short, b_short = parts[2], parts[3]
        a_mode = _MODE_MAP[a_short]
        b_mode = _MODE_MAP[b_short]
        return custom_sgd_offdiag(
            learning_rate=config["learning_rate"],
            momentum=config.get("momentum", 0.9),
            xi=config.get("xi", 0.1),
            weight_decay=config.get("weight_decay", 0.0),
            gamma=config.get("gamma", 1.0),
            base_mode=config.get("base_mode", "grad"),
            a_mode=a_mode,
            b_mode=b_mode,
            use_momentum_for_update=config.get("use_momentum_for_update", True),
        )

    raise ValueError(f"Unknown optimizer: {name!r}. Valid names: {ALL_OPTIMIZERS}")


def needs_loss(name):
    """Return True if optimizer.update() expects loss as a positional arg."""
    return name in LOG_OPTIMIZERS


# ---------------------------------------------------------------------------
# WandB sweep parameter definitions
# ---------------------------------------------------------------------------

def get_sweep_parameters(name):
    """Return WandB sweep parameter dict for the given optimizer.

    These are merged into the sweep config's ``parameters`` key.
    Task-specific fixed parameters (batch_size, n_epochs, etc.) are added
    by each sweep script.
    """
    if name == "adam":
        return {
            "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1},
            "beta1": {"distribution": "uniform", "min": 0.8, "max": 0.99},
            "beta2": {"distribution": "uniform", "min": 0.9, "max": 0.999},
            "eps": {"values": [1e-8]},
        }

    if name == "adamw":
        return {
            "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1},
            "beta1": {"distribution": "uniform", "min": 0.8, "max": 0.99},
            "beta2": {"distribution": "uniform", "min": 0.9, "max": 0.999},
            "eps": {"values": [1e-8]},
            "weight_decay": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2},
        }

    if name == "sgd":
        return {
            "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1},
            "momentum": {"distribution": "uniform", "min": 0.1, "max": 0.99},
        }

    if name == "muon":
        return {
            "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1},
            "adam_b1": {"distribution": "uniform", "min": 0.8, "max": 0.99},
            "adam_b2": {"distribution": "uniform", "min": 0.9, "max": 0.999},
            "eps": {"values": [1e-8]},
            "beta": {"distribution": "uniform", "min": 0.9, "max": 0.99},
            "weight_decay": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2},
        }

    if name in ("sgd_metric", "sgd_log_metric"):
        return {
            "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1},
            "momentum": {"distribution": "uniform", "min": 0.8, "max": 0.99},
            "xi": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-1},
            "beta": {"distribution": "uniform", "min": 0.6, "max": 0.9},
            "weight_decay": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2},
        }

    if name == "sgd_rms":
        return {
            "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1},
            "momentum": {"distribution": "uniform", "min": 0.8, "max": 0.99},
            "xi": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-1},
            "beta": {"distribution": "uniform", "min": 0.6, "max": 0.9},
            "beta_rms": {"distribution": "uniform", "min": 0.9, "max": 0.999},
            "eps": {"values": [1e-8]},
            "weight_decay": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2},
        }

    if name in ("sgd_learn_scalar", "sgd_learn_scalar_log",
                 "sgd_learn_diag", "sgd_learn_diag_log"):
        return {
            "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1},
            "momentum": {"distribution": "uniform", "min": 0.0, "max": 0.99},
            "xi": {"distribution": "log_uniform_values", "min": 1e-3, "max": 1e1},
            "beta": {"distribution": "uniform", "min": 0.5, "max": 0.99},
            "weight_decay": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2},
            "metric_lr": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1.0},
            "metric_reg": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2},
            "metric_clip": {"distribution": "uniform", "min": 1.0, "max": 5.0},
        }

    if name.startswith("sgd_offdiag_"):
        return {
            "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1},
            "momentum": {"distribution": "uniform", "min": 0.0, "max": 0.99},
            "xi": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e1},
            "gamma": {"distribution": "log_uniform_values", "min": 1e-4, "max": 10.0},
            "weight_decay": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2},
            "base_mode": {"values": ["grad", "momentum"]},
            "use_momentum_for_update": {"values": [True, False]},
        }

    raise ValueError(f"Unknown optimizer: {name!r}")


# ---------------------------------------------------------------------------
# Optuna parameter suggestions (for local / offline mode)
# ---------------------------------------------------------------------------

def suggest_optuna_parameters(name, trial, prefix=""):
    """Suggest hyperparameters using an Optuna trial.

    Parameters
    ----------
    name : str
        Optimizer name.
    trial : optuna.Trial
        Active Optuna trial.
    prefix : str
        Optional prefix for parameter names (avoids collisions).

    Returns
    -------
    dict
        Config dict suitable for ``create_optimizer(name, config)``.
    """
    p = prefix

    if name == "adam":
        return {
            "learning_rate": trial.suggest_float(p + "learning_rate", 1e-6, 1e1, log=True),
            "beta1": trial.suggest_float(p + "beta1", 0.0, 0.99),
            "beta2": trial.suggest_float(p + "beta2", 0.8, 0.9999),
            "eps": trial.suggest_float(p + "eps", 1e-10, 1e-6, log=True),
        }

    if name == "adamw":
        return {
            "learning_rate": trial.suggest_float(p + "learning_rate", 1e-6, 1e1, log=True),
            "beta1": trial.suggest_float(p + "beta1", 0.0, 0.99),
            "beta2": trial.suggest_float(p + "beta2", 0.8, 0.9999),
            "eps": trial.suggest_float(p + "eps", 1e-10, 1e-6, log=True),
            "weight_decay": trial.suggest_float(p + "weight_decay", 1e-6, 1e-1, log=True),
        }

    if name == "sgd":
        return {
            "learning_rate": trial.suggest_float(p + "learning_rate", 1e-6, 1e1, log=True),
            "momentum": trial.suggest_float(p + "momentum", 0.0, 0.99),
        }

    if name == "muon":
        return {
            "learning_rate": trial.suggest_float(p + "learning_rate", 1e-6, 1e1, log=True),
            "adam_b1": trial.suggest_float(p + "adam_b1", 0.0, 0.99),
            "adam_b2": trial.suggest_float(p + "adam_b2", 0.8, 0.9999),
            "eps": trial.suggest_float(p + "eps", 1e-10, 1e-6, log=True),
            "beta": trial.suggest_float(p + "beta", 0.5, 0.999),
        }

    if name in ("sgd_metric", "sgd_log_metric"):
        return {
            "learning_rate": trial.suggest_float(p + "learning_rate", 1e-6, 1e1, log=True),
            "momentum": trial.suggest_float(p + "momentum", 0.0, 0.99),
            "xi": trial.suggest_float(p + "xi", 1e-3, 1e1, log=True),
            "beta": 0.0,
        }

    if name == "sgd_rms":
        return {
            "learning_rate": trial.suggest_float(p + "learning_rate", 1e-6, 1e1, log=True),
            "momentum": trial.suggest_float(p + "momentum", 0.0, 0.99),
            "xi": trial.suggest_float(p + "xi", 1e-3, 1e1, log=True),
            "beta": 0.0,
            "beta_rms": trial.suggest_float(p + "beta_rms", 0.8, 0.9999),
            "eps": trial.suggest_float(p + "eps", 1e-10, 1e-6, log=True),
        }

    if name in ("sgd_learn_scalar", "sgd_learn_scalar_log",
                 "sgd_learn_diag", "sgd_learn_diag_log"):
        return {
            "learning_rate": trial.suggest_float(p + "learning_rate", 1e-6, 1e1, log=True),
            "momentum": trial.suggest_float(p + "momentum", 0.0, 0.99),
            "xi": trial.suggest_float(p + "xi", 1e-3, 1e1, log=True),
            "beta": trial.suggest_float(p + "beta", 0.5, 0.99),
            "metric_lr": trial.suggest_float(p + "metric_lr", 1e-5, 1.0, log=True),
            "metric_reg": trial.suggest_float(p + "metric_reg", 1e-6, 1e-2, log=True),
            "metric_clip": trial.suggest_float(p + "metric_clip", 1.0, 5.0),
        }

    if name.startswith("sgd_offdiag_"):
        return {
            "learning_rate": trial.suggest_float(p + "learning_rate", 1e-6, 1e1, log=True),
            "momentum": trial.suggest_float(p + "momentum", 0.0, 0.99),
            "xi": trial.suggest_float(p + "xi", 1e-4, 1e1, log=True),
            "gamma": trial.suggest_float(p + "gamma", 1e-4, 10.0, log=True),
            "weight_decay": trial.suggest_float(p + "weight_decay", 0.0, 1e-2),
            "base_mode": trial.suggest_categorical(p + "base_mode", ["grad", "momentum"]),
            "use_momentum_for_update": trial.suggest_categorical(
                p + "use_momentum_for_update", [True, False]
            ),
        }

    raise ValueError(f"Unknown optimizer: {name!r}")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_OPTIMIZER_COLORS = {
    "adam": "#1f77b4",
    "adamw": "#ff7f0e",
    "sgd": "#2ca02c",
    "muon": "#00bcd4",
    "sgd_metric": "#9467bd",
    "sgd_log_metric": "#e377c2",
    "sgd_rms": "#8c564b",
    "sgd_learn_scalar": "#d62728",
    "sgd_learn_scalar_log": "#ff9896",
    "sgd_learn_diag": "#17becf",
    "sgd_learn_diag_log": "#aec7e8",
    "sgd_offdiag_0_l": "#bcbd22",
    "sgd_offdiag_0_m": "#dbdb8d",
    "sgd_offdiag_0_theta": "#98df8a",
    "sgd_offdiag_l_0": "#c5b0d5",
    "sgd_offdiag_l_l": "#c49c94",
    "sgd_offdiag_l_m": "#f7b6d2",
    "sgd_offdiag_l_theta": "#7f7f7f",
    "sgd_offdiag_m_0": "#c7c7c7",
    "sgd_offdiag_m_l": "#ffbb78",
    "sgd_offdiag_m_m": "#ff9896",
    "sgd_offdiag_m_theta": "#9edae5",
    "sgd_offdiag_theta_0": "#393b79",
    "sgd_offdiag_theta_l": "#637939",
    "sgd_offdiag_theta_m": "#8c6d31",
    "sgd_offdiag_theta_theta": "#843c39",
}


def get_optimizer_color(name):
    """Return a hex color string for the given optimizer."""
    return _OPTIMIZER_COLORS.get(name, "#333333")


def get_optimizer_colors(names=None):
    """Return a dict mapping optimizer names to colors.

    If *names* is None, returns colors for all optimizers.
    """
    if names is None:
        names = ALL_OPTIMIZERS
    return {n: get_optimizer_color(n) for n in names}


def get_all_names():
    """Return a copy of the canonical optimizer list."""
    return list(ALL_OPTIMIZERS)
