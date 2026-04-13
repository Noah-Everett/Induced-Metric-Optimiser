"""
Tests for analysis/loaders.py — particularly _sort_runs which controls
"best run" selection for every paper figure and table.
"""

from loaders import _sort_runs


def test_sort_minimize():
    runs = [
        {"summary": {"m": 3.0}},
        {"summary": {"m": 1.0}},
        {"summary": {"m": 2.0}},
    ]
    result = _sort_runs(runs, "m", "minimize")
    assert [r["summary"]["m"] for r in result] == [1.0, 2.0, 3.0]


def test_sort_maximize():
    runs = [
        {"summary": {"m": 3.0}},
        {"summary": {"m": 1.0}},
        {"summary": {"m": 2.0}},
    ]
    result = _sort_runs(runs, "m", "maximize")
    assert [r["summary"]["m"] for r in result] == [3.0, 2.0, 1.0]


def test_sort_missing_metric_to_end_minimize():
    runs = [
        {"summary": {"m": 2.0}},
        {"summary": {}},
        {"summary": {"m": 1.0}},
    ]
    result = _sort_runs(runs, "m", "minimize")
    assert result[0]["summary"]["m"] == 1.0
    assert result[1]["summary"]["m"] == 2.0
    assert "m" not in result[2]["summary"]


def test_sort_missing_metric_to_end_maximize():
    runs = [
        {"summary": {}},
        {"summary": {"m": 3.0}},
        {"summary": {"m": 1.0}},
    ]
    result = _sort_runs(runs, "m", "maximize")
    assert result[0]["summary"]["m"] == 3.0
    assert result[1]["summary"]["m"] == 1.0
    assert "m" not in result[2]["summary"]


def test_sort_none_value_to_end():
    runs = [
        {"summary": {"m": 2.0}},
        {"summary": {"m": None}},
        {"summary": {"m": 1.0}},
    ]
    result = _sort_runs(runs, "m", "minimize")
    assert result[0]["summary"]["m"] == 1.0
    assert result[1]["summary"]["m"] == 2.0
    assert result[2]["summary"]["m"] is None


def test_sort_nan_no_crash():
    """NaN values don't crash. Ordering is undefined but shouldn't error."""
    runs = [
        {"summary": {"m": 2.0}},
        {"summary": {"m": float("nan")}},
        {"summary": {"m": 1.0}},
    ]
    result = _sort_runs(runs, "m", "minimize")
    assert len(result) == 3


def test_sort_empty():
    assert _sort_runs([], "m", "minimize") == []


def test_sort_single():
    runs = [{"summary": {"m": 1.0}}]
    result = _sort_runs(runs, "m", "minimize")
    assert len(result) == 1
    assert result[0]["summary"]["m"] == 1.0
