"""Tests for uncertainty intervals — bootstrap confidence intervals."""

import numpy as np

from kawkab.core.uncertainty import (
    bootstrap_comparison,
    bootstrap_metric,
    bootstrap_psxg_confidence,
    bootstrap_xg_confidence,
)


class TestBootstrapMetric:
    def test_mean_ci_simple(self):
        values = [0.1, 0.05, 0.3, 0.02, 0.15]
        result = bootstrap_metric(values, n_bootstrap=1000, seed=42)
        assert 0.0 < result["point_estimate"] < 1.0
        assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]
        assert result["std_error"] > 0
        assert result["n"] == 5

    def test_single_value(self):
        result = bootstrap_metric([0.5], n_bootstrap=1000)
        assert result["point_estimate"] == 0.5
        assert result["std_error"] == 0.0

    def test_empty(self):
        result = bootstrap_metric([], n_bootstrap=1000)
        assert result["point_estimate"] == 0.0

    def test_two_values(self):
        result = bootstrap_metric([0.1, 0.9], n_bootstrap=500, seed=42)
        assert result["point_estimate"] == 0.5
        assert result["ci_lower"] <= result["ci_upper"]

    def test_custom_metric_fn(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = bootstrap_metric(values, metric_fn=np.sum, n_bootstrap=500, seed=42)
        assert result["point_estimate"] == 15.0

    def test_return_bootstrap_samples(self):
        values = [0.1, 0.2, 0.3]
        result = bootstrap_metric(values, n_bootstrap=100, seed=0, return_all=True)
        assert len(result["bootstrap_samples"]) == 100

    def test_seeded_reproducible(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        a = bootstrap_metric(values, n_bootstrap=1000, seed=1)
        b = bootstrap_metric(values, n_bootstrap=1000, seed=1)
        assert a["point_estimate"] == b["point_estimate"]
        assert a["ci_lower"] == b["ci_lower"]
        assert a["ci_upper"] == b["ci_upper"]

    def test_higher_ci_wider(self):
        values = [0.05, 0.1, 0.08, 0.12, 0.15, 0.02, 0.2, 0.01, 0.07, 0.03]
        c50 = bootstrap_metric(values, n_bootstrap=500, ci_level=0.50, seed=0)
        c95 = bootstrap_metric(values, n_bootstrap=500, ci_level=0.95, seed=0)
        assert (c95["ci_upper"] - c95["ci_lower"]) >= (c50["ci_upper"] - c50["ci_lower"])

    def test_ci_contains_estimate(self):
        for _ in range(10):
            values = list(np.random.default_rng().uniform(0, 1, 20))
            result = bootstrap_metric(values, n_bootstrap=500, seed=42)
            assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]


class TestBootstrapXgConfidence:
    def test_basic_xg_ci(self):
        xg = [0.3, 0.02, 0.5, 0.1, 0.01]
        goals = [True, False, False, False, False]
        result = bootstrap_xg_confidence(xg, goals, n_bootstrap=1000, seed=42)
        assert result["total_xG"] == 0.93
        assert result["actual_goals"] == 1
        assert result["xG_CI"]["lower"] >= 0
        assert result["n_shots"] == 5

    def test_all_goals(self):
        xg = [0.5, 0.3, 0.8]
        goals = [True, True, True]
        result = bootstrap_xg_confidence(xg, goals, n_bootstrap=500, seed=0)
        assert result["actual_goals"] == 3
        assert result["total_xG"] == 1.6

    def test_no_goals(self):
        xg = [0.05, 0.02, 0.01, 0.1]
        goals = [False, False, False, False]
        result = bootstrap_xg_confidence(xg, goals, n_bootstrap=500, seed=0)
        assert result["actual_goals"] == 0

    def test_single_shot(self):
        result = bootstrap_xg_confidence([0.5], [True], n_bootstrap=500)
        assert result["n_shots"] == 1

    def test_ci_level_present(self):
        xg = [0.1, 0.2, 0.3, 0.4]
        goals = [False, False, True, False]
        result = bootstrap_xg_confidence(xg, goals, n_bootstrap=500, ci_level=0.90, seed=0)
        assert result["ci_level"] == 0.90
        assert "goals_CI" in result

    def test_xg_per_shot_ci(self):
        xg = [0.1, 0.2, 0.3, 0.4]
        goals = [False, False, False, True]
        result = bootstrap_xg_confidence(xg, goals, n_bootstrap=500, seed=42)
        assert result["xG_per_shot_CI"]["lower"] <= result["xG_per_shot_CI"]["upper"]

    def test_seeded_reproducible(self):
        xg = [0.1, 0.2, 0.3, 0.4, 0.5]
        goals = [True, False, False, True, False]
        a = bootstrap_xg_confidence(xg, goals, n_bootstrap=1000, seed=1)
        b = bootstrap_xg_confidence(xg, goals, n_bootstrap=1000, seed=1)
        assert a["xG_CI"]["lower"] == b["xG_CI"]["lower"]
        assert a["xG_CI"]["upper"] == b["xG_CI"]["upper"]


class TestBootstrapPsxgConfidence:
    def test_basic_psxg(self):
        psxg = [0.1, 0.3, 0.05, 0.8]
        goals = [False, True, False, True]
        result = bootstrap_psxg_confidence(psxg, goals, n_bootstrap=500, seed=42)
        assert result["total_xG"] > 0
        assert result["actual_goals"] == 2

    def test_psxg_zero_on_target(self):
        result = bootstrap_psxg_confidence([], [], n_bootstrap=500)
        assert result["n_shots"] == 0


class TestBootstrapComparison:
    def test_basic_comparison(self):
        a = [0.3, 0.4, 0.5, 0.6, 0.7]
        b = [0.1, 0.2, 0.15, 0.25, 0.05]
        result = bootstrap_comparison(a, b, n_bootstrap=1000, seed=42)
        assert result["diff_observed"] > 0
        assert result["diff_CI"]["lower"] <= result["diff_CI"]["upper"]
        assert 0 <= result["p_value"] <= 1

    def test_equal_groups(self):
        a = [0.5, 0.5, 0.5]
        b = [0.5, 0.5, 0.5]
        result = bootstrap_comparison(a, b, n_bootstrap=500, seed=0)
        assert result["diff_observed"] == 0.0
        assert result["diff_CI"]["lower"] <= 0 <= result["diff_CI"]["upper"]

    def test_small_samples(self):
        result = bootstrap_comparison([0.5], [0.3], n_bootstrap=500)
        assert result["diff_observed"] == 0.2

    def test_custom_metric(self):
        a = [1.0, 2.0, 3.0]
        b = [0.5, 1.0, 1.5]
        result = bootstrap_comparison(a, b, metric_fn=np.sum, n_bootstrap=500, seed=0)
        assert result["diff_observed"] == 3.0

    def test_seeded_reproducible(self):
        a = [0.3, 0.4, 0.5, 0.6, 0.7]
        b = [0.1, 0.2, 0.15, 0.25, 0.05]
        r1 = bootstrap_comparison(a, b, n_bootstrap=1000, seed=1)
        r2 = bootstrap_comparison(a, b, n_bootstrap=1000, seed=1)
        assert r1["diff_CI"]["lower"] == r2["diff_CI"]["lower"]
        assert r1["diff_CI"]["upper"] == r2["diff_CI"]["upper"]
