"""Tests for PlayerDevelopmentService — trend detection, growth rates, projections."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()
_mod = load_service_module("pdev_test", "player_development_service.py")
PlayerDevelopmentService = _mod.PlayerDevelopmentService
PlayerMatchStat = _mod.PlayerMatchStat
PlayerDevelopmentReport = _mod.PlayerDevelopmentReport
PlayerTrend = _mod.PlayerTrend
TrendDirection = _mod.TrendDirection

import pytest


def _stat(
    match_id: int,
    match_date: str,
    minutes: int = 90,
    passes_att: int = 40,
    passes_comp: int = 30,
    distance: float = 10000.0,
    sprints: int = 20,
    xg: float = 0.5,
    xt: float = 1.0,
    pressure: int = 10,
    goals: int = 0,
    assists: int = 0,
    touches: int = 50,
) -> PlayerMatchStat:
    return PlayerMatchStat(
        match_id=match_id,
        match_date=match_date,
        minutes_played=minutes,
        passes_attempted=passes_att,
        passes_completed=passes_comp,
        distance_m=distance,
        sprints=sprints,
        xg=xg,
        xt=xt,
        pressure_actions=pressure,
        goals=goals,
        assists=assists,
        touches=touches,
    )


def _history_improving() -> list[PlayerMatchStat]:
    return [
        _stat(
            1,
            "2024-01-01",
            passes_comp=30,
            passes_att=40,
            xg=0.30,
            xt=0.60,
            sprints=18,
            distance=9500,
            pressure=9,
        ),
        _stat(
            2,
            "2024-01-08",
            passes_comp=31,
            passes_att=40,
            xg=0.32,
            xt=0.63,
            sprints=18,
            distance=9600,
            pressure=9,
        ),
        _stat(
            3,
            "2024-01-15",
            passes_comp=32,
            passes_att=40,
            xg=0.34,
            xt=0.66,
            sprints=19,
            distance=9700,
            pressure=10,
        ),
        _stat(
            4,
            "2024-01-22",
            passes_comp=33,
            passes_att=40,
            xg=0.36,
            xt=0.69,
            sprints=19,
            distance=9800,
            pressure=10,
        ),
        _stat(
            5,
            "2024-01-29",
            passes_comp=34,
            passes_att=40,
            xg=0.38,
            xt=0.72,
            sprints=20,
            distance=9900,
            pressure=11,
        ),
    ]


@pytest.fixture
def svc() -> PlayerDevelopmentService:
    return PlayerDevelopmentService(
        min_matches_for_trend=3, rolling_window=5, improvement_threshold=0.05
    )


class TestServiceInit:
    def test_default_params(self) -> None:
        svc = PlayerDevelopmentService()
        assert svc.min_matches_for_trend == 3
        assert svc.rolling_window == 5
        assert svc.improvement_threshold == 0.05

    def test_custom_params(self) -> None:
        svc = PlayerDevelopmentService(
            min_matches_for_trend=5, rolling_window=10, improvement_threshold=0.1
        )
        assert svc.min_matches_for_trend == 5
        assert svc.rolling_window == 10
        assert svc.improvement_threshold == 0.1

    def test_available_property(self) -> None:
        svc = PlayerDevelopmentService()
        assert svc.available is True


class TestAnalyze:
    def test_insufficient_data_empty(self, svc: PlayerDevelopmentService) -> None:
        report = svc.analyze(1, "Player A", "FW", [])
        assert report.matches_played == 0
        assert report.overall_trend == TrendDirection.INSUFFICIENT_DATA
        assert len(report.trends) == 0
        assert len(report.notes) == 1
        assert "at least 3" in report.notes[0]

    def test_insufficient_data_one_match(self, svc: PlayerDevelopmentService) -> None:
        report = svc.analyze(1, "Player A", "MF", [_stat(1, "2024-01-01")])
        assert report.matches_played == 1
        assert report.overall_trend == TrendDirection.INSUFFICIENT_DATA

    def test_insufficient_data_two_matches(self, svc: PlayerDevelopmentService) -> None:
        report = svc.analyze(1, "Player A", "DF", [_stat(1, "2024-01-01"), _stat(2, "2024-01-08")])
        assert report.matches_played == 2
        assert report.overall_trend == TrendDirection.INSUFFICIENT_DATA

    def test_improving_trend(self, svc: PlayerDevelopmentService) -> None:
        report = svc.analyze(1, "Player A", "FW", _history_improving())
        assert report.matches_played == 5
        assert report.overall_trend == TrendDirection.IMPROVING
        assert len(report.trends) == 6
        for t in report.trends:
            if t.n_matches >= 3:
                assert t.slope_per_match >= 0

    def test_report_metadata(self, svc: PlayerDevelopmentService) -> None:
        report = svc.analyze(1, "Player A", "GK", _history_improving())
        assert report.player_id == 1
        assert report.player_name == "Player A"
        assert report.position == "GK"

    def test_strengths_identified(self, svc: PlayerDevelopmentService) -> None:
        report = svc.analyze(1, "Player A", "FW", _history_improving())
        assert len(report.strengths) > 0
        for s in report.strengths:
            assert "avg" in s

    def test_areas_to_improve_empty_on_improving(self, svc: PlayerDevelopmentService) -> None:
        report = svc.analyze(
            1,
            "Player A",
            "FW",
            [
                _stat(
                    1,
                    "2024-01-01",
                    passes_comp=20,
                    passes_att=40,
                    xg=0.1,
                    xt=0.2,
                    sprints=10,
                    distance=8000,
                    pressure=5,
                ),
                _stat(
                    2,
                    "2024-01-08",
                    passes_comp=22,
                    passes_att=40,
                    xg=0.12,
                    xt=0.22,
                    sprints=11,
                    distance=8200,
                    pressure=6,
                ),
                _stat(
                    3,
                    "2024-01-15",
                    passes_comp=21,
                    passes_att=40,
                    xg=0.11,
                    xt=0.21,
                    sprints=10,
                    distance=8100,
                    pressure=5,
                ),
            ],
        )
        if report.areas_to_improve:
            for a in report.areas_to_improve:
                assert "declining" in a or "variance" in a

    def test_limited_minutes_note(self, svc: PlayerDevelopmentService) -> None:
        stats = [
            _stat(1, "2024-01-01", minutes=90),
            _stat(2, "2024-01-08", minutes=90),
            _stat(3, "2024-01-15", minutes=30),
        ]
        report = svc.analyze(1, "Player A", "MF", stats)
        assert any("limited minutes" in n for n in report.notes)

    def test_metrics_count(self, svc: PlayerDevelopmentService) -> None:
        report = svc.analyze(1, "Player A", "DF", _history_improving())
        metric_names = [t.metric for t in report.trends]
        assert "pass_completion" in metric_names
        assert "distance_per_90" in metric_names
        assert "sprints_per_90" in metric_names
        assert "xg_per_90" in metric_names
        assert "xt_per_90" in metric_names
        assert "pressure_per_90" in metric_names


class TestPer90:
    def test_per_90_standard(self) -> None:
        assert PlayerDevelopmentService._per_90(10000, 90) == 10000.0

    def test_per_90_substitute(self) -> None:
        result = PlayerDevelopmentService._per_90(5000, 45)
        assert result == 10000.0

    def test_per_90_zero_minutes(self) -> None:
        assert PlayerDevelopmentService._per_90(100, 0) == 0.0

    def test_per_90_negative_minutes(self) -> None:
        assert PlayerDevelopmentService._per_90(100, -1) == 0.0


class TestComputeTrend:
    def test_insufficient_data(self, svc: PlayerDevelopmentService) -> None:
        trend = svc._compute_trend("pass_completion", [0.5, 0.6])
        assert trend.direction == TrendDirection.INSUFFICIENT_DATA
        assert trend.n_matches == 2

    def test_slope_positive(self, svc: PlayerDevelopmentService) -> None:
        trend = svc._compute_trend("pass_completion", [0.5, 0.6, 0.7, 0.8, 0.9])
        assert trend.slope_per_match > 0
        assert (
            trend.direction == TrendDirection.IMPROVING
            or trend.direction == TrendDirection.VOLATILE
        )

    def test_slope_negative(self, svc: PlayerDevelopmentService) -> None:
        trend = svc._compute_trend("xg_per_90", [0.8, 0.7, 0.6, 0.5, 0.4])
        assert trend.slope_per_match < 0
        assert trend.direction in (TrendDirection.DECLINING, TrendDirection.VOLATILE)

    def test_slope_flat(self, svc: PlayerDevelopmentService) -> None:
        trend = svc._compute_trend("sprints_per_90", [15.0, 15.0, 15.0, 15.0, 15.0])
        assert abs(trend.slope_per_match) < 0.01

    def test_empty_values(self, svc: PlayerDevelopmentService) -> None:
        trend = svc._compute_trend("pass_completion", [])
        assert trend.direction == TrendDirection.INSUFFICIENT_DATA
        assert trend.last_value == 0.0
        assert trend.best_value == 0.0

    def test_single_value_trend(self, svc: PlayerDevelopmentService) -> None:
        trend = svc._compute_trend("pass_completion", [0.75])
        assert trend.direction == TrendDirection.INSUFFICIENT_DATA
        assert trend.last_value == 0.75


class TestAggregateTrend:
    def test_all_improving(self, svc: PlayerDevelopmentService) -> None:
        trends = [
            PlayerTrend("xg_per_90", TrendDirection.IMPROVING, 0.1, 0.5, 0.1, 5, 0.5, 0.6, 0.3),
            PlayerTrend("xt_per_90", TrendDirection.IMPROVING, 0.1, 0.5, 0.1, 5, 0.5, 0.6, 0.3),
        ]
        assert svc._aggregate_trend(trends) == TrendDirection.IMPROVING

    def test_all_declining(self, svc: PlayerDevelopmentService) -> None:
        trends = [
            PlayerTrend("xg_per_90", TrendDirection.DECLINING, -0.1, 0.5, 0.1, 5, 0.5, 0.6, 0.3),
            PlayerTrend("xt_per_90", TrendDirection.DECLINING, -0.1, 0.5, 0.1, 5, 0.5, 0.6, 0.3),
        ]
        assert svc._aggregate_trend(trends) == TrendDirection.DECLINING

    def test_majority_volatile(self, svc: PlayerDevelopmentService) -> None:
        trends = [
            PlayerTrend("xg_per_90", TrendDirection.VOLATILE, 0.0, 0.5, 0.3, 5, 0.5, 0.8, 0.2),
            PlayerTrend("xt_per_90", TrendDirection.VOLATILE, 0.0, 0.5, 0.3, 5, 0.5, 0.8, 0.2),
            PlayerTrend(
                "sprints_per_90", TrendDirection.IMPROVING, 0.1, 0.5, 0.1, 5, 0.5, 0.6, 0.3
            ),
        ]
        assert svc._aggregate_trend(trends) == TrendDirection.VOLATILE

    def test_stable_tie(self, svc: PlayerDevelopmentService) -> None:
        trends = [
            PlayerTrend("xg_per_90", TrendDirection.IMPROVING, 0.1, 0.5, 0.1, 5, 0.5, 0.6, 0.3),
            PlayerTrend("xt_per_90", TrendDirection.DECLINING, -0.1, 0.5, 0.1, 5, 0.5, 0.6, 0.3),
            PlayerTrend("sprints_per_90", TrendDirection.STABLE, 0.0, 0.5, 0.1, 5, 0.5, 0.6, 0.3),
        ]
        assert svc._aggregate_trend(trends) == TrendDirection.STABLE

    def test_only_insufficient_data(self, svc: PlayerDevelopmentService) -> None:
        trends = [
            PlayerTrend(
                "xg_per_90", TrendDirection.INSUFFICIENT_DATA, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0
            ),
        ]
        assert svc._aggregate_trend(trends) == TrendDirection.INSUFFICIENT_DATA


class TestExtractMetric:
    def test_pass_completion(self) -> None:
        svc = PlayerDevelopmentService()
        s = _stat(1, "2024-01-01", passes_att=40, passes_comp=30)
        assert abs(svc._extract_metric(s, "pass_completion") - 0.75) < 0.001

    def test_pass_completion_zero_attempts(self) -> None:
        svc = PlayerDevelopmentService()
        s = _stat(1, "2024-01-01", passes_att=0, passes_comp=0)
        assert svc._extract_metric(s, "pass_completion") == 0.0

    def test_unknown_metric(self) -> None:
        svc = PlayerDevelopmentService()
        s = _stat(1, "2024-01-01")
        assert svc._extract_metric(s, "nonexistent") == 0.0
