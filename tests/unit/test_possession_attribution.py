"""Tests for PossessionService tackle/loss attribution methods."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()
_ps = load_service_module("pos2_test", "possession_service.py")
PossessionService = _ps.PossessionService

import pytest


@pytest.fixture
def svc() -> PossessionService:
    return PossessionService()


class TestAttributeTackle:
    def test_no_candidates(self, svc: PossessionService) -> None:
        events = [{"x": 0, "y": 0, "player_track_id": 1, "team": "home"}]
        result = svc.attribute_tackle(events, tackler_pos=(50, 34), ball_pos=(50, 34))
        assert result["tackler_track_id"] is None
        assert result["confidence"] == 0.0
        assert result["candidates"] == 0

    def test_nearest_player_chosen(self, svc: PossessionService) -> None:
        events = [
            {"x": 51, "y": 35, "player_track_id": 7, "team": "home"},
            {"x": 48, "y": 33, "player_track_id": 3, "team": "away"},
            {"x": 80, "y": 50, "player_track_id": 11, "team": "home"},
        ]
        result = svc.attribute_tackle(events, tackler_pos=(50, 34), ball_pos=(50, 34))
        assert result["tackler_track_id"] in (3, 7)
        assert result["confidence"] > 0.5

    def test_ball_proximity(self, svc: PossessionService) -> None:
        events = [
            {"x": 50, "y": 34, "player_track_id": 9, "team": "home"},
        ]
        result = svc.attribute_tackle(events, ball_pos=(50, 34))
        assert result["tackler_track_id"] == 9
        assert result["confidence"] > 0.9

    def test_excludes_far_players(self, svc: PossessionService) -> None:
        events = [
            {"x": 0, "y": 0, "player_track_id": 1, "team": "home"},
            {"x": 100, "y": 68, "player_track_id": 11, "team": "away"},
        ]
        result = svc.attribute_tackle(events, ball_pos=(50, 34), max_distance_m=5.0)
        assert result["candidates"] == 0

    def test_custom_max_distance(self, svc: PossessionService) -> None:
        events = [{"x": 53, "y": 34, "player_track_id": 5, "team": "home"}]
        result = svc.attribute_tackle(events, ball_pos=(50, 34), max_distance_m=10.0)
        assert result["tackler_track_id"] == 5


class TestAttributePossessionLoss:
    def test_tackle_cause(self, svc: PossessionService) -> None:
        events = [
            {"type": "pass", "team": "home", "timestamp_s": 100, "completed": True},
            {"type": "tackle", "team": "away", "timestamp_s": 102, "x": 50, "y": 34},
        ]
        loss = {"team": "home", "timestamp_s": 102, "x": 50, "y": 34}
        result = svc.attribute_possession_loss(events, loss)
        assert result["cause"] == "tackle"
        assert result["cause_event"] is not None
        assert result["tackle_count"] == 1

    def test_misplaced_pass_cause(self, svc: PossessionService) -> None:
        events = [
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 100,
                "completed": False,
                "x": 60,
                "y": 40,
            },
        ]
        loss = {"team": "home", "timestamp_s": 100, "x": 60, "y": 40}
        result = svc.attribute_possession_loss(events, loss)
        assert result["cause"] == "misplaced_pass"

    def test_out_of_bounds_cause(self, svc: PossessionService) -> None:
        events = [
            {"type": "out_of_play", "team": "home", "timestamp_s": 100},
        ]
        loss = {"team": "home", "timestamp_s": 100}
        result = svc.attribute_possession_loss(events, loss)
        assert result["cause"] == "out_of_bounds"

    def test_foul_cause(self, svc: PossessionService) -> None:
        events = [
            {"type": "foul", "team": "away", "timestamp_s": 100, "x": 50, "y": 34},
        ]
        loss = {"team": "home", "timestamp_s": 100}
        result = svc.attribute_possession_loss(events, loss)
        assert result["cause"] == "foul"

    def test_no_context_unknown_cause(self, svc: PossessionService) -> None:
        events = [{"type": "shot", "team": "home", "timestamp_s": 50, "x": 90, "y": 34}]
        loss = {"team": "home", "timestamp_s": 200}
        result = svc.attribute_possession_loss(events, loss)
        assert result["cause"] == "unknown"
        assert result["context_count"] == 0

    def test_tackle_takes_precedence_over_pass(self, svc: PossessionService) -> None:
        events = [
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 100,
                "completed": False,
                "x": 50,
                "y": 34,
            },
            {"type": "tackle", "team": "away", "timestamp_s": 101, "x": 50, "y": 34},
        ]
        loss = {"team": "home", "timestamp_s": 101, "x": 50, "y": 34}
        result = svc.attribute_possession_loss(events, loss)
        assert result["cause"] == "tackle"

    def test_far_events_excluded(self, svc: PossessionService) -> None:
        events = [
            {"type": "tackle", "team": "away", "timestamp_s": 50, "x": 30, "y": 34},
        ]
        loss = {"team": "home", "timestamp_s": 200, "x": 50, "y": 34}
        result = svc.attribute_possession_loss(events, loss)
        assert result["context_count"] == 0
        assert result["cause"] == "unknown"
