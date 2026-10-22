"""Tests for PossessionService."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()
_svc = load_service_module("pos_test", "possession_service.py")
PossessionService = _svc.PossessionService
PossessionChain = _svc.PossessionChain
PlayerPossessionStats = _svc.PlayerPossessionStats

import pytest


@pytest.fixture
def svc() -> PossessionService:
    return PossessionService()


class TestAnalyzeBasic:
    def test_no_events_balanced(self, svc: PossessionService) -> None:
        report = svc.analyze("home", "away", [])
        assert report.home_possession_pct == 50.0
        assert report.away_possession_pct == 50.0

    def test_home_dominates(self, svc: PossessionService) -> None:
        events = [
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 1,
                "player_track_id": 7,
                "completed": True,
            },
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 2,
                "player_track_id": 8,
                "completed": True,
            },
            {
                "type": "pass",
                "team": "away",
                "timestamp_s": 3,
                "player_track_id": 3,
                "completed": True,
            },
            {"type": "tackle", "team": "home", "timestamp_s": 4, "player_track_id": 5},
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 5,
                "player_track_id": 9,
                "completed": True,
            },
            {"type": "shot", "team": "home", "timestamp_s": 6, "player_track_id": 9},
        ]
        report = svc.analyze("home", "away", events)
        # Home had longer chains
        assert report.home_possession_pct >= 50.0

    def test_player_touch_tracking(self, svc: PossessionService) -> None:
        events = [
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 1,
                "player_track_id": 7,
                "completed": True,
            },
            {"type": "shot", "team": "home", "timestamp_s": 2, "player_track_id": 7},
        ]
        report = svc.analyze("home", "away", events)
        assert 7 in report.home_player_stats
        assert report.home_player_stats[7].touches == 2

    def test_successful_passes_counted(self, svc: PossessionService) -> None:
        events = [
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 1,
                "player_track_id": 7,
                "completed": True,
            },
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 2,
                "player_track_id": 7,
                "completed": False,
            },
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 3,
                "player_track_id": 7,
                "completed": True,
            },
        ]
        report = svc.analyze("home", "away", events)
        assert report.home_player_stats[7].successful_passes == 2
        assert report.home_player_stats[7].failed_passes == 1


class TestChainAttribution:
    def test_chain_end_by_pass_failure(self, svc: PossessionService) -> None:
        events = [
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 1,
                "player_track_id": 7,
                "completed": True,
            },
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 2,
                "player_track_id": 7,
                "completed": False,
            },
        ]
        report = svc.analyze("home", "away", events)
        # First chain should be ended by pass failure
        assert len(report.home_chains) >= 1
        assert report.home_chains[0].ended_by == "pass_failed"

    def test_chain_end_by_tackle(self, svc: PossessionService) -> None:
        events = [
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 1,
                "player_track_id": 7,
                "completed": True,
            },
            {"type": "tackle", "team": "away", "timestamp_s": 2, "player_track_id": 3},
        ]
        report = svc.analyze("home", "away", events)
        assert report.home_chains[0].ended_by == "tackle"

    def test_chain_end_by_shot(self, svc: PossessionService) -> None:
        events = [
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 1,
                "player_track_id": 7,
                "completed": True,
            },
            {"type": "shot", "team": "home", "timestamp_s": 2, "player_track_id": 9, "xg": 0.3},
        ]
        report = svc.analyze("home", "away", events)
        assert report.home_chains[0].ended_by == "shot"
        assert report.home_chains[0].xg_generated == 0.3


class TestCounterPress:
    def test_counter_press_detected(self, svc: PossessionService) -> None:
        events = [
            # Home has ball
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 1,
                "player_track_id": 7,
                "completed": True,
            },
            # Home loses it
            {"type": "tackle", "team": "away", "timestamp_s": 2, "player_track_id": 3},
            # Away has ball briefly
            {
                "type": "pass",
                "team": "away",
                "timestamp_s": 3,
                "player_track_id": 3,
                "completed": True,
            },
            # Home wins it back within 5s = counter-press
            {"type": "tackle", "team": "home", "timestamp_s": 5, "player_track_id": 5},
        ]
        report = svc.analyze("home", "away", events)
        assert report.counter_presses >= 1

    def test_no_counter_press_outside_window(self, svc: PossessionService) -> None:
        events = [
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 1,
                "player_track_id": 7,
                "completed": True,
            },
            {"type": "tackle", "team": "away", "timestamp_s": 2, "player_track_id": 3},
            {
                "type": "pass",
                "team": "away",
                "timestamp_s": 3,
                "player_track_id": 3,
                "completed": True,
            },
            # 10 seconds later — too slow for counter-press
            {"type": "tackle", "team": "home", "timestamp_s": 13, "player_track_id": 5},
        ]
        report = svc.analyze("home", "away", events)
        assert report.counter_presses == 0


class TestChainDurations:
    def test_avg_chain_duration(self, svc: PossessionService) -> None:
        events = [
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 0,
                "player_track_id": 7,
                "completed": True,
            },
            {"type": "shot", "team": "home", "timestamp_s": 5, "player_track_id": 9},
        ]
        report = svc.analyze("home", "away", events)
        assert report.avg_chain_duration_s == 5.0

    def test_longest_chain(self, svc: PossessionService) -> None:
        events = [
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 0,
                "player_track_id": 7,
                "completed": True,
            },
            {"type": "shot", "team": "home", "timestamp_s": 10, "player_track_id": 9},
        ]
        report = svc.analyze("home", "away", events)
        assert report.longest_chain_s == 10.0


class TestNotes:
    def test_balanced_notes(self, svc: PossessionService) -> None:
        events = [
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 1,
                "player_track_id": 7,
                "completed": True,
            },
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 9,
                "player_track_id": 7,
                "completed": True,
            },
        ]
        report = svc.analyze("home", "away", events)
        assert any(n for n in report.notes)

    def test_dominant_notes(self, svc: PossessionService) -> None:
        events = [
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 1,
                "player_track_id": 7,
                "completed": True,
            },
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 2,
                "player_track_id": 8,
                "completed": True,
            },
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 3,
                "player_track_id": 9,
                "completed": True,
            },
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 4,
                "player_track_id": 7,
                "completed": True,
            },
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 5,
                "player_track_id": 7,
                "completed": True,
            },
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 6,
                "player_track_id": 7,
                "completed": True,
            },
            {"type": "shot", "team": "home", "timestamp_s": 7, "player_track_id": 9},
        ]
        report = svc.analyze("home", "away", events)
        assert any("Home" in n and "dominated" in n for n in report.notes)

    def test_long_chains_note(self, svc: PossessionService) -> None:
        events = [
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 0,
                "player_track_id": 7,
                "completed": True,
            },
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 5,
                "player_track_id": 7,
                "completed": True,
            },
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 10,
                "player_track_id": 7,
                "completed": True,
            },
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 20,
                "player_track_id": 7,
                "completed": True,
            },
        ]
        report = svc.analyze("home", "away", events)
        assert any("Long" in n or "patient" in n for n in report.notes)
