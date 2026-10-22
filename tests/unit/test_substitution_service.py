"""Tests for SubstitutionService."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()
_svc = load_service_module("subs_test", "substitution_service.py")
SubstitutionService = _svc.SubstitutionService
SubstitutionEvent = _svc.SubstitutionEvent
SubstitutionImpact = _svc.SubstitutionImpact

import pytest


@pytest.fixture
def svc() -> SubstitutionService:
    return SubstitutionService()


def make_sub(
    minute: int = 60,
    team: str = "home",
    formation_before: str = "4-4-2",
    formation_after: str = "4-4-2",
    position_changed: bool = False,
) -> SubstitutionEvent:
    return SubstitutionEvent(
        minute=minute,
        second=0,
        team=team,
        player_off_track_id=1,
        player_off_name="Tired Player",
        player_on_track_id=2,
        player_on_name="Fresh Player",
        formation_before=formation_before,
        formation_after=formation_after,
        position_changed=position_changed,
    )


class TestAnalyzeBasic:
    def test_no_subs(self, svc: SubstitutionService) -> None:
        report = svc.analyze("home", [], [])
        assert report.team == "home"
        assert report.best_sub is None
        assert report.impacts == []
        assert report.total_impact == 0.0

    def test_one_sub(self, svc: SubstitutionService) -> None:
        sub = make_sub()
        events = [
            {"type": "shot", "team": "home", "minute": 62, "second": 0, "xg": 0.15},
            {"type": "goal", "team": "home", "minute": 75, "second": 0},
        ]
        report = svc.analyze("home", [sub], events)
        assert len(report.impacts) == 1
        assert report.best_sub is not None
        assert report.total_impact > 0

    def test_wrong_team_filtered(self, svc: SubstitutionService) -> None:
        subs = [make_sub(team="home"), make_sub(team="away")]
        report = svc.analyze("home", subs, [])
        assert len(report.impacts) == 1


class TestImpactCalculation:
    def test_positive_impact(self, svc: SubstitutionService) -> None:
        sub = make_sub(minute=60)
        events = [
            {"type": "shot", "team": "home", "minute": 55, "second": 0, "xg": 0.05},
            {"type": "shot", "team": "home", "minute": 65, "second": 0, "xg": 0.4},
            {"type": "goal", "team": "home", "minute": 68, "second": 0},
        ]
        impact = svc._analyze_single_sub(sub, events)
        assert impact.xg_delta > 0.3
        assert impact.goals_for == 1
        assert impact.rating > 0
        assert impact.verdict == "positive"

    def test_negative_impact(self, svc: SubstitutionService) -> None:
        sub = make_sub(minute=60)
        events = [
            {"type": "shot", "team": "home", "minute": 55, "second": 0, "xg": 0.3},
            {"type": "shot", "team": "away", "minute": 65, "second": 0, "xg": 0.05},
            {"type": "goal", "team": "away", "minute": 68, "second": 0},
        ]
        impact = svc._analyze_single_sub(sub, events)
        assert impact.goals_against == 1
        assert impact.rating < 0
        assert impact.verdict == "negative"

    def test_neutral_impact(self, svc: SubstitutionService) -> None:
        sub = make_sub(minute=60)
        events = []  # No activity around sub
        impact = svc._analyze_single_sub(sub, events)
        assert impact.rating == 0.0
        assert impact.verdict == "neutral"

    def test_formation_change_marks_tactical(self, svc: SubstitutionService) -> None:
        sub = make_sub(formation_before="4-4-2", formation_after="4-3-3", position_changed=True)
        events = [{"type": "shot", "team": "home", "minute": 65, "second": 0, "xg": 0.1}]
        impact = svc._analyze_single_sub(sub, events)
        # Neutral rating + formation change = tactical
        assert any("Formation change" in n for n in impact.notes)

    def test_window_size(self, svc: SubstitutionService) -> None:
        sub = make_sub(minute=60)
        impact = svc._analyze_single_sub(sub, [])
        assert impact.window_minutes == svc.window_minutes


class TestBestWorst:
    def test_best_worst_identified(self, svc: SubstitutionService) -> None:
        sub1 = make_sub(minute=30)
        sub2 = make_sub(minute=60)
        sub3 = make_sub(minute=80)
        events = [
            {"type": "shot", "team": "home", "minute": 60, "second": 0, "xg": 0.4},
            {"type": "goal", "team": "home", "minute": 62, "second": 0},
            {"type": "shot", "team": "home", "minute": 90, "second": 0, "xg": 0.05},
        ]
        report = svc.analyze("home", [sub1, sub2, sub3], events)
        assert report.best_sub is not None
        assert report.worst_sub is not None


class TestRatingComputation:
    def test_rating_bounded(self, svc: SubstitutionService) -> None:
        sub = make_sub(minute=60)
        events = [
            {"type": "shot", "team": "home", "minute": 55, "second": 0, "xg": 0.5},
            {"type": "shot", "team": "home", "minute": 65, "second": 0, "xg": 0.5},
        ]
        impact = svc._analyze_single_sub(sub, events)
        assert -1.0 <= impact.rating <= 1.0

    def test_rating_extreme_negative(self, svc: SubstitutionService) -> None:
        sub = make_sub(minute=60)
        events = [
            {"type": "goal", "team": "away", "minute": 62, "second": 0},
            {"type": "goal", "team": "away", "minute": 65, "second": 0},
            {"type": "goal", "team": "away", "minute": 68, "second": 0},
        ]
        impact = svc._analyze_single_sub(sub, events)
        assert impact.rating <= -0.5


class TestTacticalCount:
    def test_tactical_changes_count(self, svc: SubstitutionService) -> None:
        subs = [
            make_sub(position_changed=True),
            make_sub(minute=70, position_changed=False),
            make_sub(minute=80, position_changed=True),
        ]
        report = svc.analyze("home", subs, [])
        assert report.tactical_changes == 2

    def test_formation_changes_count(self, svc: SubstitutionService) -> None:
        subs = [
            make_sub(formation_before="4-4-2", formation_after="4-3-3"),
            make_sub(minute=70, formation_before="4-3-3", formation_after="4-3-3"),
            make_sub(minute=80, formation_before="4-3-3", formation_after="4-4-1-1"),
        ]
        report = svc.analyze("home", subs, [])
        assert report.formation_changes == 2


class TestCompareSubPairs:
    def test_pair_comparison(self, svc: SubstitutionService) -> None:
        sub1 = make_sub(minute=30)
        sub2 = make_sub(minute=60)
        events = [
            {"type": "shot", "team": "home", "minute": 62, "second": 0, "xg": 0.5},
            {"type": "goal", "team": "home", "minute": 65, "second": 0},
        ]
        report = svc.analyze("home", [sub1, sub2], events)
        pairs = svc.compare_sub_pairs(report.impacts)
        assert len(pairs) > 0
        assert all(len(p) == 3 for p in pairs)
