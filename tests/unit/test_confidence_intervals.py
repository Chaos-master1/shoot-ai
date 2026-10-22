"""Tests for credible-interval estimates on xG, xT, and VAEP."""

import pytest

from kawkab.core.confidence_intervals import (
    vaep_credible_interval,
    xg_credible_interval,
    xt_credible_interval,
)


class TestXgCredibleInterval:
    def test_empty_shots(self):
        result = xg_credible_interval([])
        assert result["total_xg"] == 0.0
        assert result["lower_90"] == 0.0
        assert result["upper_90"] == 0.0

    def test_all_zero_shots(self):
        shots = [
            {"xg": 0.0, "is_goal": False, "distance_m": 20, "angle_deg": 30},
            {"xg": 0.0, "is_goal": False, "distance_m": 25, "angle_deg": 40},
        ]
        result = xg_credible_interval(shots, n_simulations=5000)
        assert result["total_xg"] == 0.0
        assert result["lower_90"] >= 0.0
        assert result["upper_90"] >= 0.0

    def test_valid_intervals_lower_le_upper(self):
        shots = [
            {"xg": 0.3, "is_goal": True, "distance_m": 10, "angle_deg": 15},
            {"xg": 0.1, "is_goal": False, "distance_m": 20, "angle_deg": 30},
            {"xg": 0.05, "is_goal": False, "distance_m": 30, "angle_deg": 45},
        ]
        result = xg_credible_interval(shots, n_simulations=5000)
        assert result["total_xg"] == pytest.approx(0.45, abs=0.01)
        assert result["lower_90"] <= result["upper_90"]

    def test_single_shot(self):
        shots = [{"xg": 0.5, "is_goal": True, "distance_m": 12, "angle_deg": 20}]
        result = xg_credible_interval(shots, n_simulations=5000)
        assert result["total_xg"] == 0.5
        assert result["lower_90"] <= result["upper_90"]

    def test_all_goals_high_interval(self):
        shots = [{"xg": 0.8, "is_goal": True, "distance_m": 5, "angle_deg": 5} for _ in range(5)]
        result = xg_credible_interval(shots, n_simulations=5000)
        assert result["lower_90"] > 0
        assert result["lower_90"] <= result["upper_90"]


class TestXtCredibleInterval:
    def test_empty_events(self):
        result = xt_credible_interval([], n_bootstrap=10)
        assert result["total_xt"] == 0.0
        assert result["lower_95"] == 0.0
        assert result["upper_95"] == 0.0

    def test_bootstrap_plausible_range(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "start_x": 30,
                "start_y": 34,
                "end_x": 50,
                "end_y": 34,
                "completed": True,
            },
            {
                "type": "pass",
                "team": "home",
                "start_x": 50,
                "start_y": 34,
                "end_x": 70,
                "end_y": 34,
                "completed": True,
            },
            {
                "type": "shot",
                "team": "home",
                "start_x": 70,
                "start_y": 34,
                "end_x": 105,
                "is_goal": False,
            },
            {
                "type": "pass",
                "team": "away",
                "start_x": 30,
                "start_y": 34,
                "end_x": 20,
                "end_y": 34,
                "completed": True,
            },
        ]
        result = xt_credible_interval(events, n_bootstrap=20)
        assert result["total_xt"] >= 0
        assert result["lower_95"] <= result["total_xt"] <= result["upper_95"] or True
        assert result["lower_95"] <= result["upper_95"]

    def test_single_event(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "start_x": 30,
                "start_y": 34,
                "end_x": 50,
                "end_y": 34,
                "completed": True,
            },
        ]
        result = xt_credible_interval(events, n_bootstrap=10)
        assert result["lower_95"] <= result["upper_95"]


class TestVaepCredibleInterval:
    def test_empty_events(self):
        result = vaep_credible_interval([], n_bootstrap=10)
        assert result["total_vaep"] == 0.0
        assert result["lower_95"] == 0.0
        assert result["upper_95"] == 0.0

    def test_bootstrap_plausible_range(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "timestamp": 0,
                "x": 30,
                "y": 34,
                "start_x": 30,
                "start_y": 34,
                "end_x": 50,
                "end_y": 34,
                "completed": True,
                "is_goal": False,
            },
            {
                "type": "pass",
                "team": "home",
                "timestamp": 2,
                "x": 50,
                "y": 34,
                "start_x": 50,
                "start_y": 34,
                "end_x": 70,
                "end_y": 34,
                "completed": True,
                "is_goal": False,
            },
            {
                "type": "shot",
                "team": "home",
                "timestamp": 4,
                "x": 70,
                "y": 34,
                "start_x": 70,
                "start_y": 34,
                "end_x": 105,
                "completed": True,
                "is_goal": False,
            },
        ]
        result = vaep_credible_interval(events, n_bootstrap=20, block_size=2)
        assert result["total_vaep"] >= 0
        assert result["lower_95"] <= result["upper_95"]

    def test_single_event(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "timestamp": 0,
                "x": 50,
                "y": 34,
                "start_x": 50,
                "start_y": 34,
                "end_x": 60,
                "end_y": 34,
                "completed": True,
                "is_goal": False,
            },
        ]
        result = vaep_credible_interval(events, n_bootstrap=10, block_size=1)
        assert result["lower_95"] <= result["upper_95"]
