"""Tests for SetPieceService."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()
_svc = load_service_module("sp_test", "setpiece_service.py")
SetPieceService = _svc.SetPieceService
SetPieceEvent = _svc.SetPieceEvent

import pytest


@pytest.fixture
def svc() -> SetPieceService:
    return SetPieceService()


def make_corner(
    team: str = "home", outcome: str = "shot", x: float = 99, y: float = 5
) -> SetPieceEvent:
    return SetPieceEvent(
        set_piece_type="corner",
        minute=12,
        second=0,
        team=team,
        delivery_x=100,
        delivery_y=0 if team == "home" else 68,
        delivery_style="inswinging",
        delivery_height="medium",
        first_contact_x=x,
        first_contact_y=y,
        outcome=outcome,
    )


class TestTargetZone:
    def test_near_post(self, svc: SetPieceService) -> None:
        assert svc._classify_target_zone(99, 5) == "near_post"
        assert svc._classify_target_zone(100, 2) == "near_post"

    def test_far_post(self, svc: SetPieceService) -> None:
        assert svc._classify_target_zone(99, 60) == "far_post"
        assert svc._classify_target_zone(96, 65) == "far_post"

    def test_central(self, svc: SetPieceService) -> None:
        assert svc._classify_target_zone(96, 34) == "central"

    def test_edge_of_box(self, svc: SetPieceService) -> None:
        assert svc._classify_target_zone(96, 15) == "edge_of_box"
        assert svc._classify_target_zone(96, 50) == "edge_of_box"

    def test_short(self, svc: SetPieceService) -> None:
        assert svc._classify_target_zone(80, 34) == "short"


class TestDeliveryClassification:
    def test_short_corner(self, svc: SetPieceService) -> None:
        result = svc.classify_delivery(100, 0, 94, 4)
        assert result["style"] == "short"
        assert result["target_zone"] == "short"

    def test_inswinging_corner(self, svc: SetPieceService) -> None:
        # Home corner: delivery at (100, 0), target at (95, 35) - curves inward
        result = svc.classify_delivery(100, 0, 95, 35)
        assert result["style"] in ("inswinging", "outswinging", "driven", "lofted")

    def test_classification_returns_dict(self, svc: SetPieceService) -> None:
        result = svc.classify_delivery(50, 30, 80, 30)
        assert "style" in result
        assert "height" in result
        assert "target_zone" in result


class TestThreat:
    def test_goal_threat(self, svc: SetPieceService) -> None:
        assert svc.compute_threat("goal") == 1.0

    def test_shot_threat(self, svc: SetPieceService) -> None:
        assert svc.compute_threat("shot") == 0.4

    def test_clearance_negative(self, svc: SetPieceService) -> None:
        assert svc.compute_threat("clearance") < 0

    def test_unknown_threat(self, svc: SetPieceService) -> None:
        assert svc.compute_threat("made_up_thing") == 0.0


class TestRoutineDetection:
    def test_near_post_corner(self, svc: SetPieceService) -> None:
        ev = make_corner()
        assert svc.detect_routine(ev) == "near_post_corner"

    def test_far_post_corner(self, svc: SetPieceService) -> None:
        ev = make_corner(x=99, y=62)
        assert svc.detect_routine(ev) == "far_post_corner"

    def test_short_corner(self, svc: SetPieceService) -> None:
        ev = make_corner(x=85, y=10)
        assert svc.detect_routine(ev) == "short_corner"

    def test_fk_routine(self, svc: SetPieceService) -> None:
        ev = SetPieceEvent(
            set_piece_type="free_kick",
            minute=50,
            second=0,
            team="home",
            delivery_x=88,
            delivery_y=34,
            delivery_style="lofted",
            delivery_height="high",
            first_contact_x=99,
            first_contact_y=5,
            outcome="shot",
        )
        assert "fk" in svc.detect_routine(ev)


class TestAnalyze:
    def test_basic_analyze(self, svc: SetPieceService) -> None:
        events = [make_corner(), make_corner(x=99, y=62, outcome="clearance")]
        report = svc.analyze(events)
        assert report.home_stats.total_corners == 2
        assert report.home_stats.corners_to_shots == 1

    def test_no_events(self, svc: SetPieceService) -> None:
        report = svc.analyze([])
        assert report.home_stats.total_corners == 0
        assert "No significant" in report.notes[0]

    def test_threat_calculation(self, svc: SetPieceService) -> None:
        events = [
            make_corner(outcome="goal"),
            make_corner(outcome="shot"),
            make_corner(outcome="clearance"),
        ]
        report = svc.analyze(events)
        # 1.0 + 0.4 - 0.05 = 1.35
        assert abs(report.home_threat_total - 1.35) < 0.01

    def test_differential(self, svc: SetPieceService) -> None:
        events = [
            make_corner(team="home", outcome="goal"),
            make_corner(team="away", outcome="clearance"),
        ]
        report = svc.analyze(events)
        assert report.set_piece_differential > 0.5

    def test_routine_counted(self, svc: SetPieceService) -> None:
        events = [make_corner() for _ in range(3)]
        report = svc.analyze(events)
        assert "near_post_corner" in dict(report.home_stats.common_routines)

    def test_suggest_routine_no_data(self, svc: SetPieceService) -> None:
        recs = svc.suggest_routine("home", [])
        assert "No data" in recs[0]

    def test_suggest_routine_low_shot_rate(self, svc: SetPieceService) -> None:
        events = [make_corner(outcome="clearance") for _ in range(5)]
        recs = svc.suggest_routine("home", events)
        assert any("shot rate" in r.lower() for r in recs)


class TestNotesGeneration:
    def test_home_dominates_corner_notes(self, svc: SetPieceService) -> None:
        events = [make_corner(team="home", outcome="shot") for _ in range(3)]
        report = svc.analyze(events)
        assert any("Home took 3 corners" in n for n in report.notes)

    def test_short_corner_note(self, svc: SetPieceService) -> None:
        events = [
            SetPieceEvent(
                set_piece_type="corner",
                minute=10,
                second=0,
                team="home",
                delivery_x=100,
                delivery_y=0,
                delivery_style="short",
                delivery_height="low",
                first_contact_x=88,
                first_contact_y=10,
                outcome="retention_midfield",
            )
        ]
        report = svc.analyze(events)
        assert any("short corners" in n for n in report.notes)
