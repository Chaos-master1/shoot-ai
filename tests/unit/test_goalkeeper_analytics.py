"""Tests for GoalkeeperAnalytics — PSxG integration, positioning, aerial command, distribution."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import install_kawkab_stubs

install_kawkab_stubs()

import pytest

from kawkab.analysis.goalkeeper_analytics import (
    GKAerialCommand,
    GKDistribution,
    GKPositioningMetrics,
    GKSaveQuality,
    GoalkeeperAnalytics,
)


@pytest.fixture
def gka():
    return GoalkeeperAnalytics()


class TestSaveQuality:
    def test_empty_shots(self, gka):
        sq = gka.compute_save_quality([])
        assert sq.total_shots_faced == 0
        assert sq.save_rate == 0.0
        assert sq.goals_prevented == 0.0

    def test_all_saves(self, gka):
        shots = [
            {
                "placement_x": 0.5,
                "placement_y": 0.5,
                "shot_x": 90,
                "shot_y": 34,
                "speed_mps": 25.0,
                "outcome": "save",
            },
            {
                "placement_x": -0.3,
                "placement_y": 0.8,
                "shot_x": 85,
                "shot_y": 30,
                "speed_mps": 20.0,
                "outcome": "save",
            },
        ]
        sq = gka.compute_save_quality(shots)
        assert sq.saves == 2
        assert sq.goals_conceded == 0
        assert sq.save_rate == 1.0
        assert sq.total_psxg > 0

    def test_mixed_outcomes(self, gka):
        shots = [
            {
                "placement_x": 0.9,
                "placement_y": 0.1,
                "shot_x": 95,
                "shot_y": 34,
                "speed_mps": 30.0,
                "outcome": "goal",
            },
            {
                "placement_x": 0.2,
                "placement_y": 0.5,
                "shot_x": 80,
                "shot_y": 34,
                "speed_mps": 15.0,
                "outcome": "save",
            },
            {
                "placement_x": 0.5,
                "placement_y": 0.5,
                "shot_x": 90,
                "shot_y": 34,
                "speed_mps": 25.0,
                "outcome": "save",
            },
        ]
        sq = gka.compute_save_quality(shots)
        assert sq.goals_conceded == 1
        assert sq.saves == 2
        assert 0 < sq.save_rate < 1
        assert sq.goals_prevented >= 0

    def test_high_quality_saves(self, gka):
        shots = [
            {
                "placement_x": 0.95,
                "placement_y": 0.05,
                "shot_x": 99,
                "shot_y": 34,
                "speed_mps": 35.0,
                "outcome": "save",
            },
            {
                "placement_x": 0.9,
                "placement_y": 0.1,
                "shot_x": 95,
                "shot_y": 34,
                "speed_mps": 30.0,
                "outcome": "save",
            },
        ]
        sq = gka.compute_save_quality(shots)
        assert sq.high_quality_saves >= 1

    def test_one_on_one(self, gka):
        shots = [
            {
                "placement_x": 0.3,
                "placement_y": 0.5,
                "shot_x": 100,
                "shot_y": 34,
                "speed_mps": 20.0,
                "outcome": "save",
                "one_on_one": True,
            },
            {
                "placement_x": 0.7,
                "placement_y": 0.3,
                "shot_x": 100,
                "shot_y": 34,
                "speed_mps": 22.0,
                "outcome": "goal",
                "one_on_one": True,
            },
        ]
        sq = gka.compute_save_quality(shots)
        assert sq.one_on_one_saves == 1
        assert sq.one_on_one_goals == 1
        assert sq.one_on_one_save_rate == 0.5

    def test_avg_psxg_per_shot(self, gka):
        shots = [
            {
                "placement_x": 0.5,
                "placement_y": 0.5,
                "shot_x": 90,
                "shot_y": 34,
                "speed_mps": 25.0,
                "outcome": "save",
            },
            {
                "placement_x": 0.5,
                "placement_y": 0.5,
                "shot_x": 90,
                "shot_y": 34,
                "speed_mps": 25.0,
                "outcome": "goal",
            },
        ]
        sq = gka.compute_save_quality(shots)
        assert sq.avg_psxg_per_shot > 0
        assert sq.avg_psxg_per_shot <= 1.0


class TestPositioning:
    def test_empty(self, gka):
        pm = gka.compute_positioning()
        assert pm.avg_position_x == 0.0
        assert pm.position_samples == 0

    def test_with_positions(self, gka):
        positions = [{"x": 15.0, "y": 34.0}, {"x": 18.0, "y": 33.0}, {"x": 12.0, "y": 35.0}]
        pm = gka.compute_positioning(tracking_positions=positions)
        assert abs(pm.avg_position_x - 15.0) < 0.1
        assert pm.position_samples == 3

    def test_with_sweeps(self, gka):
        sweeps = [
            {"x": 30.0, "y": 34.0, "gk_x": 10.0, "gk_y": 34.0},
            {"x": 25.0, "y": 30.0, "gk_x": 8.0, "gk_y": 34.0},
        ]
        pm = gka.compute_positioning(sweeps=sweeps)
        assert pm.sweep_actions == 2
        assert pm.max_sweep_distance > 0

    def test_with_set_pieces(self, gka):
        sp = [{"x": 5.0, "y": 30.0}, {"x": 6.0, "y": 38.0}]
        pm = gka.compute_positioning(set_piece_positions=sp)
        assert abs(pm.avg_set_piece_position_x - 5.5) < 0.1

    def test_combined(self, gka):
        positions = [{"x": 14.0, "y": 34.0}]
        sweeps = [{"x": 28.0, "y": 34.0, "gk_x": 14.0, "gk_y": 34.0}]
        pm = gka.compute_positioning(
            tracking_positions=positions,
            sweeps=sweeps,
        )
        assert pm.position_samples == 1
        assert pm.sweep_actions == 1


class TestAerialCommand:
    def test_empty(self, gka):
        ac = gka.compute_aerial_command([])
        assert ac.crosses_faced == 0
        assert ac.aerial_success_rate == 0.0

    def test_all_claimed(self, gka):
        crosses = [
            {
                "action_type": "claim",
                "outcome": "complete",
                "under_pressure": False,
                "effective": True,
            },
            {
                "action_type": "claim",
                "outcome": "complete",
                "under_pressure": True,
                "effective": True,
            },
            {
                "action_type": "claim",
                "outcome": "complete",
                "under_pressure": False,
                "effective": True,
            },
        ]
        ac = gka.compute_aerial_command(crosses)
        assert ac.crosses_claimed == 3
        assert ac.claimed_under_pressure == 1
        assert ac.aerial_success_rate == 1.0

    def test_mixed_aerial(self, gka):
        crosses = [
            {
                "action_type": "claim",
                "outcome": "complete",
                "under_pressure": False,
                "effective": True,
            },
            {
                "action_type": "punch",
                "outcome": "complete",
                "under_pressure": False,
                "effective": True,
            },
            {
                "action_type": "miss",
                "outcome": "failed",
                "under_pressure": False,
                "effective": False,
            },
        ]
        ac = gka.compute_aerial_command(crosses)
        assert ac.crosses_claimed == 1
        assert ac.crosses_punched == 1
        assert ac.crosses_faced == 3
        assert ac.aerial_success_rate == 2 / 3


class TestDistribution:
    def test_empty(self, gka):
        gd = gka.compute_distribution([])
        assert gd.short_attempts == 0

    def test_short_distribution(self, gka):
        actions = [
            {
                "action_type": "short_dist",
                "outcome": "complete",
                "distance_m": 10.0,
                "dest_x": 20.0,
                "progressive": False,
            },
            {
                "action_type": "short_dist",
                "outcome": "complete",
                "distance_m": 15.0,
                "dest_x": 30.0,
                "progressive": False,
            },
            {
                "action_type": "short_dist",
                "outcome": "failed",
                "distance_m": 12.0,
                "dest_x": 25.0,
                "progressive": False,
            },
        ]
        gd = gka.compute_distribution(actions)
        assert gd.short_attempts == 3
        assert gd.short_successful == 2
        assert abs(gd.short_accuracy - 2 / 3) < 0.01
        assert abs(gd.avg_distance - 12.33) < 0.1

    def test_long_distribution(self, gka):
        actions = [
            {
                "action_type": "long_dist",
                "outcome": "complete",
                "distance_m": 50.0,
                "dest_x": 80.0,
                "progressive": True,
            },
            {
                "action_type": "long_dist",
                "outcome": "failed",
                "distance_m": 55.0,
                "dest_x": 85.0,
                "progressive": False,
            },
        ]
        gd = gka.compute_distribution(actions)
        assert gd.long_attempts == 2
        assert gd.long_successful == 1
        assert gd.progressive_passes == 1

    def test_zones(self, gka):
        actions = [
            {
                "action_type": "short_dist",
                "outcome": "complete",
                "distance_m": 10.0,
                "dest_x": 10.0,
                "progressive": False,
            },
            {
                "action_type": "long_dist",
                "outcome": "complete",
                "distance_m": 50.0,
                "dest_x": 80.0,
                "progressive": True,
            },
            {
                "action_type": "short_dist",
                "outcome": "complete",
                "distance_m": 15.0,
                "dest_x": 55.0,
                "progressive": False,
            },
        ]
        gd = gka.compute_distribution(actions)
        assert gd.left_zone == 1
        assert gd.center_zone == 1
        assert gd.right_zone == 1


class TestMatchReport:
    def test_empty_report(self, gka):
        report = gka.compute_match_report("home")
        assert report.team == "home"
        assert report.rating == 50.0
        assert len(report.strengths) >= 1
        assert len(report.weaknesses) >= 1

    def test_with_saves(self, gka):
        shots = [
            {
                "placement_x": 0.5,
                "placement_y": 0.5,
                "shot_x": 90,
                "shot_y": 34,
                "speed_mps": 25.0,
                "outcome": "save",
            },
            {
                "placement_x": 0.5,
                "placement_y": 0.5,
                "shot_x": 90,
                "shot_y": 34,
                "speed_mps": 25.0,
                "outcome": "save",
            },
        ]
        report = gka.compute_match_report("home", shots_faced=shots)
        assert report.save_quality.saves == 2
        assert report.rating > 50

    def test_clean_sheet_bonus(self, gka):
        report = gka.compute_match_report("home", clean_sheet=True)
        assert report.clean_sheet is True
        assert report.rating > 50

    def test_with_all_data(self, gka):
        shots = [
            {
                "placement_x": 0.3,
                "placement_y": 0.6,
                "shot_x": 92,
                "shot_y": 34,
                "speed_mps": 20.0,
                "outcome": "save",
            },
        ]
        crosses = [
            {
                "action_type": "claim",
                "outcome": "complete",
                "under_pressure": False,
                "effective": True,
            },
        ]
        dist_actions = [
            {
                "action_type": "short_dist",
                "outcome": "complete",
                "distance_m": 12.0,
                "dest_x": 25.0,
                "progressive": False,
            },
        ]
        positions = [{"x": 12.0, "y": 34.0}]
        sweeps = [{"x": 30.0, "y": 34.0, "gk_x": 12.0, "gk_y": 34.0}]
        report = gka.compute_match_report(
            "home",
            shots_faced=shots,
            cross_actions=crosses,
            distribution_actions=dist_actions,
            tracking_positions=positions,
            sweeps=sweeps,
            clean_sheet=True,
            player_name="Test GK",
        )
        assert report.player_name == "Test GK"
        assert report.save_quality.saves == 1
        assert report.aerial.crosses_claimed == 1
        assert report.distribution.short_attempts == 1
        assert report.positioning.position_samples == 1
        assert report.positioning.sweep_actions == 1


class TestReportDict:
    def test_to_dict(self, gka):
        report = gka.compute_match_report("away", clean_sheet=True)
        d = report.to_dict()
        assert d["team"] == "away"
        assert d["clean_sheet"] is True
        assert "rating" in d
        assert "save_quality" in d
        assert "positioning" in d
        assert "aerial" in d
        assert "distribution" in d
        assert "strengths" in d
        assert "weaknesses" in d

    def test_prevented_metric(self, gka):
        shots = [
            {
                "placement_x": 0.7,
                "placement_y": 0.3,
                "shot_x": 95,
                "shot_y": 34,
                "speed_mps": 28.0,
                "outcome": "save",
            },
            {
                "placement_x": 0.5,
                "placement_y": 0.5,
                "shot_x": 90,
                "shot_y": 34,
                "speed_mps": 25.0,
                "outcome": "save",
            },
        ]
        report = gka.compute_match_report("home", shots_faced=shots)
        d = report.to_dict()
        assert d["save_quality"]["goals_prevented"] >= 0


class TestDataclasses:
    def test_save_quality_defaults(self):
        sq = GKSaveQuality()
        assert sq.save_rate == 0.0
        assert sq.total_psxg == 0.0

    def test_positioning_defaults(self):
        pm = GKPositioningMetrics()
        assert pm.avg_position_x == 0.0
        assert pm.sweep_actions == 0

    def test_aerial_defaults(self):
        ac = GKAerialCommand()
        assert ac.aerial_success_rate == 0.0
        assert ac.crosses_faced == 0

    def test_distribution_defaults(self):
        gd = GKDistribution()
        assert gd.short_accuracy == 0.0
        assert gd.avg_distance == 0.0


class TestAssessStrengthsWeaknesses:
    def test_elite_stopper(self, gka):
        shots = [
            {
                "placement_x": 0.3,
                "placement_y": 0.6,
                "shot_x": 90,
                "shot_y": 34,
                "speed_mps": 20.0,
                "outcome": "save",
            },
            {
                "placement_x": 0.3,
                "placement_y": 0.6,
                "shot_x": 90,
                "shot_y": 34,
                "speed_mps": 20.0,
                "outcome": "save",
            },
            {
                "placement_x": 0.3,
                "placement_y": 0.6,
                "shot_x": 90,
                "shot_y": 34,
                "speed_mps": 20.0,
                "outcome": "save",
            },
        ]
        report = gka.compute_match_report("home", shots_faced=shots)
        assert any("Elite" in s for s in report.strengths)

    def test_below_average(self, gka):
        shots = [
            {
                "placement_x": 0.7,
                "placement_y": 0.3,
                "shot_x": 95,
                "shot_y": 34,
                "speed_mps": 28.0,
                "outcome": "goal",
            },
            {
                "placement_x": 0.7,
                "placement_y": 0.3,
                "shot_x": 95,
                "shot_y": 34,
                "speed_mps": 28.0,
                "outcome": "goal",
            },
            {
                "placement_x": 0.5,
                "placement_y": 0.5,
                "shot_x": 90,
                "shot_y": 34,
                "speed_mps": 25.0,
                "outcome": "save",
            },
        ]
        report = gka.compute_match_report("home", shots_faced=shots)
        assert any("Below-average" in w for w in report.weaknesses)

    def test_aerial_strength(self, gka):
        crosses = [
            {
                "action_type": "claim",
                "outcome": "complete",
                "under_pressure": False,
                "effective": True,
            },
            {
                "action_type": "claim",
                "outcome": "complete",
                "under_pressure": False,
                "effective": True,
            },
            {
                "action_type": "claim",
                "outcome": "complete",
                "under_pressure": False,
                "effective": True,
            },
        ]
        report = gka.compute_match_report("home", cross_actions=crosses)
        assert any("Commanding" in s for s in report.strengths)
