"""
Tests for the optimizer registry: dispatch correctness, factory functions,
sweep parameter generation, and display metadata.

A broken registry silently runs the wrong optimizer for multi-hour sweeps.
"""

import pytest
import optax

from optimizer_registry import (
    ALL_OPTIMIZERS,
    LOG_OPTIMIZERS,
    HVP_OPTIMIZERS,
    _PARAM_BOUNDS,
    needs_loss,
    needs_hvp,
    create_optimizer,
    get_sweep_parameters,
    get_optimizer_color,
)

# Shared config with all possible keys at plausible defaults
_FULL_CONFIG = {
    "learning_rate": 0.01, "momentum": 0.9, "weight_decay": 0.01,
    "eps": 1e-8, "beta1": 0.9, "beta2": 0.999,
    "adam_b1": 0.9, "adam_b2": 0.999, "beta": 0.8, "muon_beta": 0.95,
    "xi": 0.1, "beta_rms": 0.99,
    "metric_lr": 1e-3, "metric_reg": 1e-4, "metric_clip": 4.0,
    "metric_param": "exp", "max_condition_number": 1000.0,
    "beta_s": 0.1, "hess_eps": 1e-6,
    "curv_beta": 0.05, "curv_tau": 1.0,
    "gamma": 1.0, "base_mode": "grad", "use_momentum_for_update": True,
}


def test_no_duplicates():
    assert len(ALL_OPTIMIZERS) == len(set(ALL_OPTIMIZERS))


def test_log_subset_of_all():
    assert LOG_OPTIMIZERS <= set(ALL_OPTIMIZERS)


def test_hvp_subset_of_all():
    assert HVP_OPTIMIZERS <= set(ALL_OPTIMIZERS)


def test_log_hvp_disjoint():
    assert LOG_OPTIMIZERS.isdisjoint(HVP_OPTIMIZERS)


@pytest.mark.parametrize("name", ALL_OPTIMIZERS)
def test_needs_loss_exact(name):
    assert needs_loss(name) == (name in LOG_OPTIMIZERS)


@pytest.mark.parametrize("name", ALL_OPTIMIZERS)
def test_needs_hvp_exact(name):
    assert needs_hvp(name) == (name in HVP_OPTIMIZERS)


@pytest.mark.parametrize("name", ALL_OPTIMIZERS)
def test_create_optimizer(name):
    opt = create_optimizer(name, _FULL_CONFIG)
    assert isinstance(opt, optax.GradientTransformation)


def test_create_optimizer_unknown_raises():
    with pytest.raises(ValueError):
        create_optimizer("nonexistent_optimizer", {})


@pytest.mark.parametrize("name", ALL_OPTIMIZERS)
def test_sweep_params(name):
    params = get_sweep_parameters(name)
    assert isinstance(params, dict)
    assert len(params) > 0
    assert "learning_rate" in params


@pytest.mark.parametrize("name", ALL_OPTIMIZERS)
def test_optimizer_has_color(name):
    assert get_optimizer_color(name) != "#333333"
