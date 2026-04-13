"""
Tests for analysis/stats.py — bootstrap confidence intervals.

These CIs appear in the paper. Wrong intervals could overstate or understate
optimizer differences.
"""

import numpy as np
from stats import bootstrap_best_run_ci


def test_bootstrap_ci_contains_best():
    rng = np.random.default_rng(42)
    values = rng.normal(loc=5.0, scale=1.0, size=200)
    all_runs = {"opt": [{"summary": {"m": float(v)}} for v in values]}
    results = bootstrap_best_run_ci(
        all_runs, metric_key="m", direction="minimize",
        optimizers=["opt"], n_bootstrap=5000, ci_level=0.95,
    )
    r = results["opt"]
    assert r["ci_lo"] <= r["best"] <= r["ci_hi"]


def test_bootstrap_ci_positive_width():
    values = list(range(1, 51))
    all_runs = {"opt": [{"summary": {"m": float(v)}} for v in values]}
    results = bootstrap_best_run_ci(
        all_runs, metric_key="m", direction="minimize",
        optimizers=["opt"], n_bootstrap=5000,
    )
    assert results["opt"]["ci_hi"] - results["opt"]["ci_lo"] > 0


def test_bootstrap_maximize():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    all_runs = {"opt": [{"summary": {"m": v}} for v in values]}
    results = bootstrap_best_run_ci(
        all_runs, metric_key="m", direction="maximize",
        optimizers=["opt"], n_bootstrap=5000,
    )
    assert results["opt"]["best"] == 10.0
    assert results["opt"]["ci_hi"] >= 9.0


def test_bootstrap_single_run_skipped():
    all_runs = {"lonely": [{"summary": {"m": 1.0}}]}
    results = bootstrap_best_run_ci(
        all_runs, metric_key="m", direction="minimize",
        optimizers=["lonely"], n_bootstrap=1000,
    )
    assert "lonely" not in results


def test_bootstrap_nan_excluded():
    all_runs = {
        "opt": [
            {"summary": {"m": 1.0}},
            {"summary": {"m": float("nan")}},
            {"summary": {"m": 2.0}},
            {"summary": {"m": 3.0}},
        ]
    }
    results = bootstrap_best_run_ci(
        all_runs, metric_key="m", direction="minimize",
        optimizers=["opt"], n_bootstrap=1000,
    )
    assert results["opt"]["best"] == 1.0
    assert results["opt"]["n_runs"] == 3
