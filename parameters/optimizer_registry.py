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
from optimisers.jax_learnable_diag_curv import (
    custom_sgd_learnable_diag_curv,
    custom_sgd_log_learnable_diag_curv,
)
from optimisers.jax_offdiag import custom_sgd_offdiag
from optimisers.jax_derived_newton_diag import derived_newton_diag


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
    "sgd_learn_diag_curv",
    "sgd_learn_diag_curv_log",
    "sgd_newton_diag",
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
    "sgd_learn_diag_curv_log",
})

HVP_OPTIMIZERS = frozenset({
    "sgd_newton_diag",
})

# Off-diagonal mode shorthand -> full mode name
_MODE_MAP = {
    "0": "zero",
    "l": "grad",
    "m": "momentum",
    "theta": "params",
}


# ---------------------------------------------------------------------------
# Unified hyperparameter bounds (single source of truth)
# ---------------------------------------------------------------------------

# Format: (min, max, log_scale, wandb_distribution)
# log_scale=True means log-uniform; False means uniform
_PARAM_BOUNDS = {
    # Common parameters
    "learning_rate": (1e-6, 1e1, True),
    "momentum": (0.0, 0.99, False),
    "weight_decay": (1e-6, 1e-1, True),
    "eps": (1e-10, 1e-6, True),
    # Adam/AdamW parameters
    "beta1": (0.0, 0.99, False),
    "beta2": (0.8, 0.9999, False),
    # Muon parameters
    "adam_b1": (0.0, 0.99, False),
    "adam_b2": (0.8, 0.9999, False),
    "muon_beta": (0.5, 0.999, False),  # muon's momentum parameter
    # Custom SGD metric parameters
    "xi": (1e-3, 1e1, True),
    "beta": (0.5, 0.99, False),
    "beta_rms": (0.8, 0.9999, False),
    # Learnable metric parameters
    "metric_lr": (1e-5, 1.0, True),
    "metric_reg": (1e-6, 1e-2, True),
    "metric_clip": (1.0, 5.0, False),
    "max_condition_number": (10.0, 10000.0, True),
    # Newton-targeted parameters
    "beta_s": (0.01, 0.5, True),
    "hess_eps": (1e-8, 1e-3, True),
    # Curvature-aware parameters
    "curv_beta": (0.001, 1.0, True),
    "curv_tau": (0.5, 10.0, True),     # IMO-48: narrowed from [0.1,100]; sensitivity peaks at tau~|H|
    # Curvature ratio (curv_beta/xi): sign-flip transition at coth(H/tau) ~ 1
    "curv_ratio": (0.2, 5.0, True),    # IMO-48: replaces independent curv_beta in Optuna sweeps
    # Off-diagonal parameters
    "gamma": (1e-4, 10.0, True),
}


def _wandb_param(name, fixed_value=None, categorical_values=None):
    """Generate WandB sweep parameter definition from unified bounds.

    Parameters
    ----------
    name : str
        Parameter name (must exist in _PARAM_BOUNDS unless fixed/categorical).
    fixed_value : any, optional
        If provided, return a fixed value parameter.
    categorical_values : list, optional
        If provided, return a categorical parameter.

    Returns
    -------
    dict
        WandB sweep parameter specification.
    """
    if fixed_value is not None:
        return {"values": [fixed_value]}
    if categorical_values is not None:
        return {"values": categorical_values}

    lo, hi, log_scale = _PARAM_BOUNDS[name]
    if log_scale:
        return {"distribution": "log_uniform_values", "min": lo, "max": hi}
    else:
        return {"distribution": "uniform", "min": lo, "max": hi}


def _optuna_suggest(trial, name, prefix="", fixed_value=None, categorical_values=None):
    """Suggest parameter value from unified bounds using Optuna trial.

    Parameters
    ----------
    trial : optuna.Trial
        Active Optuna trial.
    name : str
        Parameter name (must exist in _PARAM_BOUNDS unless fixed/categorical).
    prefix : str
        Optional prefix for parameter names.
    fixed_value : any, optional
        If provided, return this fixed value.
    categorical_values : list, optional
        If provided, suggest from these values.

    Returns
    -------
    any
        Suggested parameter value.
    """
    if fixed_value is not None:
        return fixed_value
    if categorical_values is not None:
        return trial.suggest_categorical(prefix + name, categorical_values)

    lo, hi, log_scale = _PARAM_BOUNDS[name]
    return trial.suggest_float(prefix + name, lo, hi, log=log_scale)


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
            metric_param=config.get("metric_param", "exp"),
            max_condition_number=config.get("max_condition_number", None),
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
            metric_param=config.get("metric_param", "exp"),
            max_condition_number=config.get("max_condition_number", None),
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
            metric_param=config.get("metric_param", "exp"),
            max_condition_number=config.get("max_condition_number", None),
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
            metric_param=config.get("metric_param", "exp"),
            max_condition_number=config.get("max_condition_number", None),
        )

    if name == "sgd_learn_diag_curv":
        return custom_sgd_learnable_diag_curv(
            learning_rate=config["learning_rate"],
            momentum=config.get("momentum", 0.9),
            xi=config.get("xi", 0.1),
            beta=config.get("beta", 0.8),
            weight_decay=config.get("weight_decay", 0.0),
            metric_lr=config.get("metric_lr", 1e-3),
            metric_reg=config.get("metric_reg", 1e-4),
            metric_clip=config.get("metric_clip", 4.0),
            curv_beta=config.get("curv_beta", 0.05),
            curv_tau=config.get("curv_tau", 1.0),
            metric_param=config.get("metric_param", "exp"),
            max_condition_number=config.get("max_condition_number", None),
        )

    if name == "sgd_learn_diag_curv_log":
        return custom_sgd_log_learnable_diag_curv(
            learning_rate=config["learning_rate"],
            momentum=config.get("momentum", 0.9),
            xi=config.get("xi", 0.1),
            beta=config.get("beta", 0.8),
            weight_decay=config.get("weight_decay", 0.0),
            metric_lr=config.get("metric_lr", 1e-3),
            metric_reg=config.get("metric_reg", 1e-4),
            metric_clip=config.get("metric_clip", 4.0),
            curv_beta=config.get("curv_beta", 0.05),
            curv_tau=config.get("curv_tau", 1.0),
            metric_param=config.get("metric_param", "exp"),
            max_condition_number=config.get("max_condition_number", None),
        )

    if name == "sgd_newton_diag":
        return derived_newton_diag(
            learning_rate=config["learning_rate"],
            momentum=config.get("momentum", 0.9),
            beta_s=config.get("beta_s", 0.1),
            weight_decay=config.get("weight_decay", 0.0),
            metric_clip=config.get("metric_clip", 4.0),
            hess_eps=config.get("hess_eps", 1e-6),
            xi=config.get("xi", 0.0),
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


def needs_hvp(name):
    """Return True if optimizer.update() expects h_diag as a positional arg."""
    return name in HVP_OPTIMIZERS


# ---------------------------------------------------------------------------
# WandB sweep parameter definitions
# ---------------------------------------------------------------------------

def get_sweep_parameters(name, overrides=None):
    """Return WandB sweep parameter dict for the given optimizer.

    These are merged into the sweep config's ``parameters`` key.
    Task-specific fixed parameters (batch_size, n_epochs, etc.) are added
    by each sweep script.

    Uses unified _PARAM_BOUNDS to ensure consistency with Optuna suggestions.

    Parameters
    ----------
    name : str
        Optimizer name.
    overrides : dict, optional
        Parameter overrides. Keys are parameter names, values are dicts with
        either ``{"value": x}`` for fixed values, ``{"values": [x, y]}`` for
        categorical, or ``{"min": lo, "max": hi, "log": bool}`` for ranges.

    Returns
    -------
    dict
        WandB sweep parameter specification.
    """
    overrides = overrides or {}

    def _param(param_name, default_fixed=None, default_categorical=None):
        """Get parameter spec, applying override if present."""
        if param_name in overrides:
            ov = overrides[param_name]
            if "value" in ov:
                return {"values": [ov["value"]]}
            if "values" in ov:
                return {"values": ov["values"]}
            if "min" in ov and "max" in ov:
                log_scale = ov.get("log", False)
                if log_scale:
                    return {"distribution": "log_uniform_values", "min": ov["min"], "max": ov["max"]}
                else:
                    return {"distribution": "uniform", "min": ov["min"], "max": ov["max"]}
        return _wandb_param(param_name, fixed_value=default_fixed, categorical_values=default_categorical)

    if name == "adam":
        return {
            "learning_rate": _param("learning_rate"),
            "beta1": _param("beta1"),
            "beta2": _param("beta2"),
            "eps": _param("eps", default_fixed=1e-8),
        }

    if name == "adamw":
        return {
            "learning_rate": _param("learning_rate"),
            "beta1": _param("beta1"),
            "beta2": _param("beta2"),
            "eps": _param("eps", default_fixed=1e-8),
            "weight_decay": _param("weight_decay"),
        }

    if name == "sgd":
        return {
            "learning_rate": _param("learning_rate"),
            "momentum": _param("momentum"),
        }

    if name == "muon":
        return {
            "learning_rate": _param("learning_rate"),
            "adam_b1": _param("adam_b1"),
            "adam_b2": _param("adam_b2"),
            "eps": _param("eps", default_fixed=1e-8),
            "beta": _param("muon_beta"),
            "weight_decay": _param("weight_decay"),
        }

    if name in ("sgd_metric", "sgd_log_metric"):
        return {
            "learning_rate": _param("learning_rate"),
            "momentum": _param("momentum"),
            "xi": _param("xi"),
            "beta": _param("beta"),
            "weight_decay": _param("weight_decay"),
        }

    if name == "sgd_rms":
        return {
            "learning_rate": _param("learning_rate"),
            "momentum": _param("momentum"),
            "xi": _param("xi"),
            "beta": _param("beta"),
            "beta_rms": _param("beta_rms"),
            "eps": _param("eps", default_fixed=1e-8),
            "weight_decay": _param("weight_decay"),
        }

    if name in ("sgd_learn_scalar", "sgd_learn_scalar_log",
                 "sgd_learn_diag", "sgd_learn_diag_log"):
        return {
            "learning_rate": _param("learning_rate"),
            "momentum": _param("momentum"),
            "xi": _param("xi"),
            "beta": _param("beta"),
            "weight_decay": _param("weight_decay"),
            "metric_lr": _param("metric_lr"),
            "metric_reg": _param("metric_reg"),
            "metric_clip": _param("metric_clip"),
            "metric_param": _param("metric_param", default_categorical=[
                "exp", "exp_matched_reg", "softplus", "exp_norm_grad", "exp_adaptive_clip",
            ]),
            "max_condition_number": _param("max_condition_number"),
        }

    if name in ("sgd_learn_diag_curv", "sgd_learn_diag_curv_log"):
        # Clamp curv_beta override range to _PARAM_BOUNDS (Optuna does this
        # after the ratio transform; keep WandB consistent).
        cb_spec = _param("curv_beta")
        cb_lo, cb_hi, _ = _PARAM_BOUNDS["curv_beta"]
        if "min" in cb_spec:
            cb_spec["min"] = max(cb_lo, cb_spec["min"])
        if "max" in cb_spec:
            cb_spec["max"] = min(cb_hi, cb_spec["max"])
        return {
            "learning_rate": _param("learning_rate"),
            "momentum": _param("momentum"),
            "xi": _param("xi"),
            "beta": _param("beta"),
            "weight_decay": _param("weight_decay"),
            "metric_lr": _param("metric_lr"),
            "metric_reg": _param("metric_reg"),
            "metric_clip": _param("metric_clip"),
            "curv_beta": cb_spec,
            "curv_tau": _param("curv_tau"),
            "metric_param": _param("metric_param", default_categorical=[
                "exp", "exp_matched_reg", "softplus", "exp_norm_grad", "exp_adaptive_clip",
            ]),
            "max_condition_number": _param("max_condition_number"),
        }

    if name == "sgd_newton_diag":
        return {
            "learning_rate": _param("learning_rate"),
            "momentum": _param("momentum"),
            "beta_s": _param("beta_s"),
            "weight_decay": _param("weight_decay"),
            "metric_clip": _param("metric_clip"),
            "hess_eps": _param("hess_eps", default_fixed=1e-6),
            "xi": _param("xi", default_fixed=0.0),
        }

    if name.startswith("sgd_offdiag_"):
        return {
            "learning_rate": _param("learning_rate"),
            "momentum": _param("momentum"),
            "xi": _param("xi"),
            "gamma": _param("gamma"),
            "weight_decay": _param("weight_decay"),
            "base_mode": _param("base_mode", default_categorical=["grad", "momentum"]),
            "use_momentum_for_update": _param(
                "use_momentum_for_update", default_categorical=[True, False]
            ),
        }

    raise ValueError(f"Unknown optimizer: {name!r}")


# ---------------------------------------------------------------------------
# Optuna parameter suggestions (for local / offline mode)
# ---------------------------------------------------------------------------

def suggest_optuna_parameters(name, trial, prefix="", overrides=None):
    """Suggest hyperparameters using an Optuna trial.

    Uses unified _PARAM_BOUNDS to ensure consistency with WandB sweep parameters.

    Parameters
    ----------
    name : str
        Optimizer name.
    trial : optuna.Trial
        Active Optuna trial.
    prefix : str
        Optional prefix for parameter names (avoids collisions).
    overrides : dict, optional
        Parameter overrides. Keys are parameter names, values are dicts with
        either ``{"value": x}`` for fixed values, ``{"values": [x, y]}`` for
        categorical, or ``{"min": lo, "max": hi, "log": bool}`` for ranges.

    Returns
    -------
    dict
        Config dict suitable for ``create_optimizer(name, config)``.
    """
    overrides = overrides or {}
    p = prefix

    def _suggest(param_name, default_fixed=None, default_categorical=None):
        """Suggest parameter, applying override if present."""
        if param_name in overrides:
            ov = overrides[param_name]
            if "value" in ov:
                return ov["value"]
            if "values" in ov:
                return trial.suggest_categorical(p + param_name, ov["values"])
            if "min" in ov and "max" in ov:
                log_scale = ov.get("log", False)
                return trial.suggest_float(p + param_name, ov["min"], ov["max"], log=log_scale)
        return _optuna_suggest(trial, param_name, p, fixed_value=default_fixed, categorical_values=default_categorical)

    if name == "adam":
        return {
            "learning_rate": _suggest("learning_rate"),
            "beta1": _suggest("beta1"),
            "beta2": _suggest("beta2"),
            "eps": _suggest("eps", default_fixed=1e-8),
        }

    if name == "adamw":
        return {
            "learning_rate": _suggest("learning_rate"),
            "beta1": _suggest("beta1"),
            "beta2": _suggest("beta2"),
            "eps": _suggest("eps", default_fixed=1e-8),
            "weight_decay": _suggest("weight_decay"),
        }

    if name == "sgd":
        return {
            "learning_rate": _suggest("learning_rate"),
            "momentum": _suggest("momentum"),
        }

    if name == "muon":
        return {
            "learning_rate": _suggest("learning_rate"),
            "adam_b1": _suggest("adam_b1"),
            "adam_b2": _suggest("adam_b2"),
            "eps": _suggest("eps", default_fixed=1e-8),
            "beta": _suggest("muon_beta"),
            "weight_decay": _suggest("weight_decay"),
        }

    if name in ("sgd_metric", "sgd_log_metric"):
        return {
            "learning_rate": _suggest("learning_rate"),
            "momentum": _suggest("momentum"),
            "xi": _suggest("xi"),
            "beta": _suggest("beta"),
            "weight_decay": _suggest("weight_decay"),
        }

    if name == "sgd_rms":
        return {
            "learning_rate": _suggest("learning_rate"),
            "momentum": _suggest("momentum"),
            "xi": _suggest("xi"),
            "beta": _suggest("beta"),
            "beta_rms": _suggest("beta_rms"),
            "eps": _suggest("eps", default_fixed=1e-8),
            "weight_decay": _suggest("weight_decay"),
        }

    if name in ("sgd_learn_scalar", "sgd_learn_scalar_log",
                 "sgd_learn_diag", "sgd_learn_diag_log"):
        mp = _suggest("metric_param", default_categorical=[
            "exp", "exp_matched_reg", "softplus", "exp_norm_grad", "exp_adaptive_clip",
        ])
        config = {
            "learning_rate": _suggest("learning_rate"),
            "momentum": _suggest("momentum"),
            "xi": _suggest("xi"),
            "beta": _suggest("beta"),
            "weight_decay": _suggest("weight_decay"),
            "metric_lr": _suggest("metric_lr"),
            "metric_reg": _suggest("metric_reg"),
            "metric_clip": _suggest("metric_clip"),
            "metric_param": mp,
        }
        if mp == "exp_adaptive_clip":
            config["max_condition_number"] = _suggest("max_condition_number")
        return config

    if name in ("sgd_learn_diag_curv", "sgd_learn_diag_curv_log"):
        xi_val = _suggest("xi")
        # IMO-48: tie curv_beta to xi via curv_ratio. Sign-flip transition
        # at coth(H/tau) ~ 1; useful range [0.2, 5.0].
        if "curv_beta" in overrides:
            curv_beta_val = _suggest("curv_beta")
        else:
            curv_ratio = _suggest("curv_ratio")
            cb_lo, cb_hi, _ = _PARAM_BOUNDS["curv_beta"]
            curv_beta_val = max(cb_lo, min(cb_hi, xi_val * curv_ratio))
        mp = _suggest("metric_param", default_categorical=[
            "exp", "exp_matched_reg", "softplus", "exp_norm_grad", "exp_adaptive_clip",
        ])
        config = {
            "learning_rate": _suggest("learning_rate"),
            "momentum": _suggest("momentum"),
            "xi": xi_val,
            "beta": _suggest("beta"),
            "weight_decay": _suggest("weight_decay"),
            "metric_lr": _suggest("metric_lr"),
            "metric_reg": _suggest("metric_reg"),
            "metric_clip": _suggest("metric_clip"),
            "curv_beta": curv_beta_val,
            "curv_tau": _suggest("curv_tau"),
            "metric_param": mp,
        }
        if mp == "exp_adaptive_clip":
            config["max_condition_number"] = _suggest("max_condition_number")
        return config

    if name == "sgd_newton_diag":
        return {
            "learning_rate": _suggest("learning_rate"),
            "momentum": _suggest("momentum"),
            "beta_s": _suggest("beta_s"),
            "weight_decay": _suggest("weight_decay"),
            "metric_clip": _suggest("metric_clip"),
            "hess_eps": _suggest("hess_eps", default_fixed=1e-6),
            "xi": _suggest("xi", default_fixed=0.0),
        }

    if name.startswith("sgd_offdiag_"):
        return {
            "learning_rate": _suggest("learning_rate"),
            "momentum": _suggest("momentum"),
            "xi": _suggest("xi"),
            "gamma": _suggest("gamma"),
            "weight_decay": _suggest("weight_decay"),
            "base_mode": _suggest("base_mode", default_categorical=["grad", "momentum"]),
            "use_momentum_for_update": _suggest(
                "use_momentum_for_update", default_categorical=[True, False]
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
    "sgd_learn_diag_curv": "#e6550d",
    "sgd_learn_diag_curv_log": "#fdae6b",
    "sgd_newton_diag": "#7b2d8e",
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
