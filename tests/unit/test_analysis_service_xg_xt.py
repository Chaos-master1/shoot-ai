"""Tests for AnalysisService xG/xT/PPDA functions.

The service has heavy dependencies (CV, knowledge), so we test only
the pure-function entry points that take pre-computed data.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()


def _install_cv_stub() -> None:
    if "kawkab.services" in sys.modules:
        return
    services_mod = types.ModuleType("kawkab.services")
    sys.modules["kawkab.services"] = services_mod
    cv_mod = types.ModuleType("kawkab.services.cv_service")

    class FrameDetections:
        pass

    class MatchTrackData:
        pass

    cv_mod.FrameDetections = FrameDetections
    cv_mod.MatchTrackData = MatchTrackData
    sys.modules["kawkab.services.cv_service"] = cv_mod
    services_mod.cv_service = cv_mod


_install_cv_stub()
_as = load_service_module("as_test", "analysis_service.py")
AnalysisService = _as.AnalysisService

import pytest


@pytest.fixture
def svc() -> AnalysisService:
    return AnalysisService()


class TestXgSimple:
    def test_empty_events(self, svc: AnalysisService) -> None:
        result = svc.compute_xg_simple([])
        assert result["home"] == 0.0
        assert result["away"] == 0.0
        assert result["shot_details"] == []

    def test_single_home_shot(self, svc: AnalysisService) -> None:
        events = [
            {
                "type": "shot",
                "team": "home",
                "metadata": {"distance_to_goal_m": 12, "angle_to_goal_deg": 30},
            },
        ]
        result = svc.compute_xg_simple(events)
        assert result["home"] > 0
        assert result["away"] == 0
        assert len(result["shot_details"]) == 1
        assert result["shot_details"][0]["team"] == "home"

    def test_away_shot(self, svc: AnalysisService) -> None:
        events = [
            {
                "type": "shot",
                "team": "away",
                "metadata": {"distance_to_goal_m": 18, "angle_to_goal_deg": 25},
            },
        ]
        result = svc.compute_xg_simple(events)
        assert result["home"] == 0
        assert result["away"] > 0

    def test_close_shot_higher_xg(self, svc: AnalysisService) -> None:
        close = svc.compute_xg_simple(
            [
                {
                    "type": "shot",
                    "team": "home",
                    "metadata": {"distance_to_goal_m": 6, "angle_to_goal_deg": 30},
                }
            ]
        )
        far = svc.compute_xg_simple(
            [
                {
                    "type": "shot",
                    "team": "home",
                    "metadata": {"distance_to_goal_m": 30, "angle_to_goal_deg": 30},
                }
            ]
        )
        assert close["home"] > far["home"]

    def test_central_shot_higher_xg(self, svc: AnalysisService) -> None:
        center = svc.compute_xg_simple(
            [
                {
                    "type": "shot",
                    "team": "home",
                    "metadata": {"distance_to_goal_m": 12, "angle_to_goal_deg": 0},
                }
            ]
        )
        wide = svc.compute_xg_simple(
            [
                {
                    "type": "shot",
                    "team": "home",
                    "metadata": {"distance_to_goal_m": 12, "angle_to_goal_deg": 75},
                }
            ]
        )
        assert center["home"] > wide["home"]

    def test_non_shot_events_ignored(self, svc: AnalysisService) -> None:
        events = [
            {"type": "pass", "team": "home"},
            {"type": "foul", "team": "away"},
        ]
        result = svc.compute_xg_simple(events)
        assert result["home"] == 0
        assert result["away"] == 0
        assert result["shot_details"] == []

    def test_shot_with_timestamp(self, svc: AnalysisService) -> None:
        events = [
            {
                "type": "shot",
                "team": "home",
                "timestamp": 1234.5,
                "metadata": {"distance_to_goal_m": 10, "angle_to_goal_deg": 30},
            }
        ]
        result = svc.compute_xg_simple(events)
        assert result["shot_details"][0]["timestamp"] == 1234.5

    def test_multiple_shots_accumulate(self, svc: AnalysisService) -> None:
        events = [
            {
                "type": "shot",
                "team": "home",
                "metadata": {"distance_to_goal_m": 10, "angle_to_goal_deg": 30},
            },
            {
                "type": "shot",
                "team": "home",
                "metadata": {"distance_to_goal_m": 12, "angle_to_goal_deg": 25},
            },
            {
                "type": "shot",
                "team": "home",
                "metadata": {"distance_to_goal_m": 15, "angle_to_goal_deg": 35},
            },
        ]
        result = svc.compute_xg_simple(events)
        assert result["home"] > 0
        assert len(result["shot_details"]) == 3

    def test_xg_bounded_zero_to_one(self, svc: AnalysisService) -> None:
        events = [
            {
                "type": "shot",
                "team": "home",
                "metadata": {"distance_to_goal_m": 1, "angle_to_goal_deg": 0},
            },
            {
                "type": "shot",
                "team": "home",
                "metadata": {"distance_to_goal_m": 80, "angle_to_goal_deg": 89},
            },
        ]
        result = svc.compute_xg_simple(events)
        for shot in result["shot_details"]:
            assert 0.0 <= shot["xg"] <= 1.0


class TestXtSimple:
    def test_empty_events(self, svc: AnalysisService) -> None:
        result = svc.compute_xt_simple([])
        assert result["home"] == 0.0
        assert result["away"] == 0.0

    def test_single_pass_advances_xt(self, svc: AnalysisService) -> None:
        events = [
            {
                "type": "pass",
                "team": "home",
                "completed": True,
                "metadata": {"start_x_pct": 0.3, "end_x_pct": 0.7},
            },
        ]
        result = svc.compute_xt_simple(events)
        assert result["home"] >= 0
        assert result["away"] == 0

    def test_failed_pass_no_xt(self, svc: AnalysisService) -> None:
        events = [
            {
                "type": "pass",
                "team": "home",
                "completed": False,
                "metadata": {"start_x_pct": 0.3, "end_x_pct": 0.7},
            },
        ]
        result = svc.compute_xt_simple(events)
        assert result["home"] == 0

    def test_pass_in_attacking_third_higher_xt(self, svc: AnalysisService) -> None:
        events = [
            {
                "type": "pass",
                "team": "home",
                "completed": True,
                "metadata": {"start_x_pct": 0.1, "end_x_pct": 0.9},
            },
        ]
        result = svc.compute_xt_simple(events)
        assert result["home"] > 0

    def test_pass_progresses_forward(self, svc: AnalysisService) -> None:
        events = [
            {
                "type": "pass",
                "team": "home",
                "completed": True,
                "metadata": {"start_x_pct": 0.1, "end_x_pct": 0.5},
            },
            {
                "type": "pass",
                "team": "home",
                "completed": True,
                "metadata": {"start_x_pct": 0.5, "end_x_pct": 0.9},
            },
        ]
        result = svc.compute_xt_simple(events)
        assert result["home"] > 0

    def test_backward_pass_no_xt(self, svc: AnalysisService) -> None:
        events = [
            {
                "type": "pass",
                "team": "home",
                "completed": True,
                "metadata": {"start_x_pct": 0.7, "end_x_pct": 0.3},
            },
        ]
        result = svc.compute_xt_simple(events)
        assert result["home"] == 0

    def test_away_passes_credit_away(self, svc: AnalysisService) -> None:
        events = [
            {
                "type": "pass",
                "team": "away",
                "completed": True,
                "metadata": {"start_x_pct": 0.1, "end_x_pct": 0.7},
            },
        ]
        result = svc.compute_xt_simple(events)
        assert result["away"] > 0
        assert result["home"] == 0

    def test_non_pass_events_ignored(self, svc: AnalysisService) -> None:
        events = [
            {"type": "shot", "team": "home", "x": 90, "y": 34},
            {"type": "foul", "team": "home"},
        ]
        result = svc.compute_xt_simple(events)
        assert result["home"] == 0
        assert result["away"] == 0


class TestPpda:
    def test_empty_match_data(self, svc: AnalysisService) -> None:
        from dataclasses import dataclass, field

        @dataclass
        class FakeTrack:
            frames: list = field(default_factory=list)

        result = svc.compute_ppda(FakeTrack(), team="home")
        assert "ppda" in result
        assert result["ppda"] is None
        assert result["intensity"] == "unknown"
