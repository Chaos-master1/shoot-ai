"""Tests for GoalkeeperService."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()
_svc = load_service_module("gk_test", "goalkeeper_service.py")
GoalkeeperService = _svc.GoalkeeperService
GoalkeeperAction = _svc.GoalkeeperAction

import pytest


@pytest.fixture
def svc() -> GoalkeeperService:
    return GoalkeeperService()


class TestXGOT:
    def test_xgot_close_shot_high(self, svc: GoalkeeperService) -> None:
        xgot = svc.compute_xgot_simple(shot_x=100, shot_y=34, body_part="foot")
        assert 0.0 < xgot < 1.0

    def test_xgot_far_shot_lower(self, svc: GoalkeeperService) -> None:
        close = svc.compute_xgot_simple(100, 34, "foot")
        far = svc.compute_xgot_simple(80, 34, "foot")
        assert close > far

    def test_xgot_off_target_zero(self, svc: GoalkeeperService) -> None:
        # Shot at center line should be very low
        xgot = svc.compute_xgot_simple(shot_x=0, shot_y=34, body_part="foot")
        assert xgot >= 0.0

    def test_xgot_header_lower(self, svc: GoalkeeperService) -> None:
        foot = svc.compute_xgot_simple(95, 34, "foot")
        head = svc.compute_xgot_simple(95, 34, "head")
        assert head < foot

    def test_xgot_one_on_one_higher(self, svc: GoalkeeperService) -> None:
        normal = svc.compute_xgot_simple(95, 34, "foot", one_on_one=False)
        one_on_one = svc.compute_xgot_simple(95, 34, "foot", one_on_one=True)
        assert one_on_one > normal

    def test_xgot_bounded(self, svc: GoalkeeperService) -> None:
        for d in range(50, 105, 10):
            for y in [10, 34, 58]:
                for bp in ["foot", "head"]:
                    xgot = svc.compute_xgot_simple(d, y, bp)
                    assert 0.0 <= xgot <= 1.0


class TestStatsBasic:
    def test_no_shots_no_actions(self, svc: GoalkeeperService) -> None:
        stats = svc.compute_stats("home", [], [])
        assert stats.save_rate == 0.0
        assert stats.shots_faced == 0
        assert stats.clean_sheet is False

    def test_save_rate_calculation(self, svc: GoalkeeperService) -> None:
        shots = [
            {"x": 88, "y": 34, "outcome": "save"},
            {"x": 88, "y": 34, "outcome": "save"},
            {"x": 88, "y": 34, "outcome": "goal"},
        ]
        stats = svc.compute_stats("home", [], shots)
        assert abs(stats.save_rate - 2 / 3) < 0.01

    def test_goals_conceded(self, svc: GoalkeeperService) -> None:
        shots = [
            {"x": 88, "y": 34, "outcome": "goal"},
            {"x": 88, "y": 34, "outcome": "save"},
        ]
        stats = svc.compute_stats("home", [], shots)
        assert stats.goals_conceded == 1
        assert stats.saves == 1

    def test_action_breakdown(self, svc: GoalkeeperService) -> None:
        actions = [
            GoalkeeperAction(
                action_type="save_cross", minute=10, second=0, team="home", outcome="complete"
            ),
            GoalkeeperAction(
                action_type="save_cross", minute=20, second=0, team="home", outcome="complete"
            ),
            GoalkeeperAction(
                action_type="sweep", minute=30, second=0, team="home", outcome="complete"
            ),
        ]
        stats = svc.compute_stats("home", actions, [])
        assert stats.crosses_claimed == 2
        assert stats.sweep_actions == 1

    def test_distribution_accuracy(self, svc: GoalkeeperService) -> None:
        actions = [
            GoalkeeperAction(
                action_type="short_dist", minute=10, second=0, team="home", outcome="complete"
            ),
            GoalkeeperAction(
                action_type="short_dist", minute=20, second=0, team="home", outcome="failed"
            ),
            GoalkeeperAction(
                action_type="long_dist", minute=30, second=0, team="home", outcome="complete"
            ),
        ]
        stats = svc.compute_stats("home", actions, [])
        assert stats.short_distribution_attempts == 2
        assert stats.short_distribution_successful == 1
        assert stats.long_distribution_attempts == 1
        assert stats.long_distribution_successful == 1


class TestXGTPerShot:
    def test_xgot_per_shot_average(self, svc: GoalkeeperService) -> None:
        shots = [
            {"x": 95, "y": 34, "outcome": "save", "body_part": "foot"},
            {"x": 88, "y": 34, "outcome": "save", "body_part": "foot"},
        ]
        stats = svc.compute_stats("home", [], shots)
        assert stats.xgot_per_shot > 0

    def test_goals_prevented(self, svc: GoalkeeperService) -> None:
        # xGOT faced - xGOT conceded = goals prevented
        shots = [
            {"x": 95, "y": 34, "outcome": "save", "body_part": "foot"},
            {"x": 95, "y": 34, "outcome": "save", "body_part": "foot"},
            {"x": 95, "y": 34, "outcome": "goal", "body_part": "foot"},
        ]
        stats = svc.compute_stats("home", [], shots)
        assert stats.goals_prevented_xgot > 0


class TestGKArea:
    def test_in_gk_area_home(self, svc: GoalkeeperService) -> None:
        assert svc.is_in_gk_area(3, 34, defending_x=0)

    def test_outside_gk_area(self, svc: GoalkeeperService) -> None:
        assert not svc.is_in_gk_area(20, 34, defending_x=0)

    def test_in_gk_area_away(self, svc: GoalkeeperService) -> None:
        assert svc.is_in_gk_area(102, 34, defending_x=105)


class TestNotesGeneration:
    def test_excellent_save_rate(self, svc: GoalkeeperService) -> None:
        shots = [
            {"x": 88, "y": 34, "outcome": "save"},
            {"x": 88, "y": 34, "outcome": "save"},
            {"x": 88, "y": 34, "outcome": "save"},
            {"x": 88, "y": 34, "outcome": "save"},
        ]
        stats = svc.compute_stats("home", [], shots)
        assert any("Excellent" in n for n in stats.notes)

    def test_low_save_rate(self, svc: GoalkeeperService) -> None:
        shots = [
            {"x": 88, "y": 34, "outcome": "save"},
            {"x": 88, "y": 34, "outcome": "goal"},
            {"x": 88, "y": 34, "outcome": "goal"},
            {"x": 88, "y": 34, "outcome": "goal"},
        ]
        stats = svc.compute_stats("home", [], shots)
        assert any("Low save rate" in n for n in stats.notes)

    def test_clean_sheet_note(self, svc: GoalkeeperService) -> None:
        stats = svc.compute_stats("home", [], [], clean_sheet=True)
        assert any("clean sheet" in n.lower() for n in stats.notes)
