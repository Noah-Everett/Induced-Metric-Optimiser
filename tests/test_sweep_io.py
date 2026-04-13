"""
Tests for the sweep I/O pipeline: SweepLogger writes CSV, _parse_run_csv and
_load_csv_run read it back. Serialization edge cases.

If this pipeline is lossy, every number in the paper is wrong.
"""

import math
import os

import numpy as np
import pytest

from sweep_utils import SweepLogger, _round_float, _json_default, _format_value
from consolidate_results import _parse_run_csv
from loaders import _load_csv_run
from pathlib import Path


# ---- _round_float ----

def test_round_float_normal():
    result = _round_float(3.14159265, 4)
    assert abs(result - 3.142) < 0.001


def test_round_float_zero():
    assert _round_float(0.0) == 0.0


def test_round_float_inf():
    assert _round_float(float("inf")) == float("inf")
    assert _round_float(float("-inf")) == float("-inf")


def test_round_float_nan():
    assert math.isnan(_round_float(float("nan")))


def test_round_float_very_small():
    result = _round_float(1e-30, 3)
    assert result != 0.0
    assert abs(result - 1e-30) / 1e-30 < 0.01


# ---- _json_default ----

def test_json_default_numpy_int():
    assert _json_default(np.int64(42)) == 42
    assert isinstance(_json_default(np.int64(42)), int)


def test_json_default_numpy_float():
    assert isinstance(_json_default(np.float32(3.14)), float)


def test_json_default_numpy_bool():
    assert _json_default(np.bool_(True)) is True


def test_json_default_numpy_array():
    assert _json_default(np.array([1, 2, 3])) == [1, 2, 3]


def test_json_default_jax_scalar():
    import jax.numpy as jnp
    result = _json_default(jnp.float32(2.5))
    assert isinstance(result, float)


def test_json_default_unsupported():
    with pytest.raises(TypeError):
        _json_default(object())


# ---- _format_value ----

def test_format_value_none():
    assert _format_value(None) == ""


def test_format_value_int():
    assert _format_value(42) == 42


def test_format_value_float():
    assert isinstance(_format_value(3.14159), float)


def test_format_value_string():
    assert _format_value("hello") == "hello"


def test_format_value_numpy_int():
    assert _format_value(np.int64(7)) == 7


def test_format_value_numpy_float():
    assert isinstance(_format_value(np.float64(2.71)), float)


# ---- Round-trip tests ----

def test_round_trip(tmp_path):
    """Write CSV with SweepLogger, read back with both parsers."""
    config = {"learning_rate": 0.01, "momentum": 0.9, "xi": 0.1}
    summary = {"sweep_metric": 0.123456, "converged": True}
    history = [
        {"epoch": 0, "train_loss": 1.5, "test_acc": 0.1},
        {"epoch": 1, "train_loss": 0.8, "test_acc": 0.5},
        {"epoch": 2, "train_loss": 0.3, "test_acc": 0.85},
    ]

    logger = SweepLogger("local", local_dir=str(tmp_path), run_index=0)
    logger.init_run(config)
    for row in history:
        logger.log(row)
    logger.finish(summary)

    csv_path = tmp_path / "run_0.csv"
    assert csv_path.exists()

    # Via _parse_run_csv
    result = _parse_run_csv(csv_path)
    assert result is not None
    metadata, rows = result
    assert metadata["run_id"] == 0
    assert metadata["config"]["learning_rate"] == 0.01
    assert metadata["summary"]["sweep_metric"] == 0.123456
    assert len(rows) == 3
    assert rows[0]["epoch"] == 0
    assert rows[2]["test_acc"] == 0.85

    # Via _load_csv_run
    run = _load_csv_run(str(csv_path))
    assert run["config"]["learning_rate"] == 0.01
    assert run["summary"]["sweep_metric"] == 0.123456
    assert len(run["history"]["epoch"]) == 3


def test_nan_round_trip(tmp_path):
    """NaN metric values survive write/read cycle."""
    logger = SweepLogger("local", local_dir=str(tmp_path), run_index=1)
    logger.init_run({"lr": 0.01})
    logger.log({"epoch": 0, "loss": float("nan")})
    logger.finish({"final": 0.5})

    csv_path = tmp_path / "run_1.csv"
    run = _load_csv_run(str(csv_path))
    # NaN written via _format_value -> _round_float returns nan ->
    # CSV gets "nan" string -> _load_csv_run float("nan") succeeds
    loss_val = run["history"]["loss"][0]
    assert loss_val is None or (isinstance(loss_val, float) and math.isnan(loss_val))


def test_numpy_scalar_types(tmp_path):
    """numpy scalar types in config and history serialize without error."""
    config = {"lr": np.float32(0.01), "steps": np.int64(100)}
    logger = SweepLogger("local", local_dir=str(tmp_path), run_index=2)
    logger.init_run(config)
    logger.log({"epoch": np.int32(0), "loss": np.float64(1.5)})
    logger.finish({"metric": np.float32(0.5)})

    csv_path = tmp_path / "run_2.csv"
    assert csv_path.exists()
    result = _parse_run_csv(csv_path)
    assert result is not None
