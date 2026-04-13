"""
Tests for analysis/utils.py — _SUMMARY_KEYS collision checks, ema,
compute_speedrun_results.

A collision in _SUMMARY_KEYS silently routes hyperparameters into the summary
column, corrupting HP sensitivity analysis.
"""

import numpy as np
import numpy.testing as npt

from utils import _SUMMARY_KEYS, compute_speedrun_results, ema
from optimizer_registry import _PARAM_BOUNDS, ALL_OPTIMIZERS, get_sweep_parameters


def test_summary_keys_no_hp_collision():
    """No HP name in _PARAM_BOUNDS should appear in _SUMMARY_KEYS."""
    overlap = set(_PARAM_BOUNDS.keys()) & _SUMMARY_KEYS
    assert overlap == set(), f"HP names collide with _SUMMARY_KEYS: {overlap}"


def test_summary_keys_no_sweep_param_collision():
    """Sweep parameter names shouldn't collide with _SUMMARY_KEYS."""
    all_config_keys = set()
    for name in ALL_OPTIMIZERS:
        all_config_keys.update(get_sweep_parameters(name).keys())
    overlap = all_config_keys & _SUMMARY_KEYS
    assert overlap == set(), f"Sweep config keys collide with _SUMMARY_KEYS: {overlap}"


def test_ema_identity():
    """alpha=1.0 means no smoothing."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    npt.assert_array_equal(ema(arr, alpha=1.0), arr)


def test_ema_frozen():
    """alpha=0.0 means fully frozen at first value."""
    arr = np.array([1.0, 2.0, 3.0])
    npt.assert_array_equal(ema(arr, alpha=0.0), [1.0, 1.0, 1.0])


def test_compute_speedrun_below():
    data = {
        "opt_a": {
            "train_loss": np.array([1.0, 0.5, 0.2, 0.1]),
            "wall_times": np.array([0.0, 1.0, 2.0, 3.0]),
        }
    }
    results = compute_speedrun_results(data, [0.5, 0.1], "train_loss", "below")
    assert results["opt_a"][0.5] == 1.0
    assert results["opt_a"][0.1] == 3.0


def test_compute_speedrun_unreachable():
    data = {
        "opt_a": {
            "train_loss": np.array([1.0, 0.5]),
            "wall_times": np.array([0.0, 1.0]),
        }
    }
    results = compute_speedrun_results(data, [0.01], "train_loss", "below")
    assert results["opt_a"][0.01] is None
