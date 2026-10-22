"""Tests for PeriodizationService."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()
_p = load_service_module("per_test", "periodization_service.py")
PeriodizationService = _p.PeriodizationService
CyclePhase = _p.CyclePhase
CongestionLevel = _p.CongestionLevel

import pytest


@pytest.fixture
def svc() -> PeriodizationService:
    return PeriodizationService()


def make_history() -> list[dict]:
    return [
        {
            "week_start": "2024-01-01",
            "date": "2024-01-03",
            "source": "training",
            "duration_min": 60,
            "rpe": 5,
            "distance_m": 4000,
        },
        {
            "week_start": "2024-01-01",
            "date": "2024-01-05",
            "source": "training",
            "duration_min": 90,
            "rpe": 6,
            "distance_m": 6000,
        },
        {
            "week_start": "2024-01-01",
            "date": "2024-01-07",
            "source": "match",
            "duration_min": 90,
            "rpe": 7,
            "distance_m": 10000,
        },
        {
            "week_start": "2024-01-08",
            "date": "2024-01-10",
            "source": "training",
            "duration_min": 90,
            "rpe": 6,
            "distance_m": 6000,
        },
        {
            "week_start": "2024-01-08",
            "date": "2024-01-12",
            "source": "training",
            "duration_min": 90,
            "rpe": 7,
            "distance_m": 7000,
        },
        {
            "week_start": "2024-01-08",
            "date": "2024-01-14",
            "source": "match",
            "duration_min": 90,
            "rpe": 7,
            "distance_m": 10000,
        },
        {
            "week_start": "2024-01-15",
            "date": "2024-01-17",
            "source": "training",
            "duration_min": 60,
            "rpe": 4,
            "distance_m": 3000,
        },
        {
            "week_start": "2024-01-15",
            "date": "2024-01-19",
            "source": "match",
            "duration_min": 90,
            "rpe": 7,
            "distance_m": 10000,
        },
    ]


class TestPeriodization:
    def test_empty_data(self, svc: PeriodizationService) -> None:
        report = svc.analyze(1, "Test", [])
        assert report.total_weeks == 0
        assert report.load_trend == "unknown"

    def test_basic_report(self, svc: PeriodizationService) -> None:
        report = svc.analyze(1, "Test", make_history())
        assert report.total_weeks == 3
        assert len(report.weeks) == 3
        assert report.avg_weekly_load > 0

    def test_week_summary_match_count(self, svc: PeriodizationService) -> None:
        report = svc.analyze(1, "Test", make_history())
        assert report.weeks[0].matches == 1
        assert report.weeks[1].matches == 1
        assert report.weeks[2].matches == 1

    def test_congestion_classification(self, svc: PeriodizationService) -> None:
        report = svc.analyze(1, "Test", make_history())
        for w in report.weeks:
            assert w.congestion in list(CongestionLevel)

    def test_phase_classification(self, svc: PeriodizationService) -> None:
        report = svc.analyze(1, "Test", make_history())
        for w in report.weeks:
            assert w.phase in list(CyclePhase)

    def test_taper_detected(self, svc: PeriodizationService) -> None:
        history = [
            {
                "week_start": "2024-01-01",
                "date": "2024-01-03",
                "source": "training",
                "duration_min": 120,
                "rpe": 8,
                "distance_m": 8000,
            },
            {
                "week_start": "2024-01-01",
                "date": "2024-01-05",
                "source": "match",
                "duration_min": 90,
                "rpe": 8,
                "distance_m": 10000,
            },
            {
                "week_start": "2024-01-08",
                "date": "2024-01-10",
                "source": "training",
                "duration_min": 30,
                "rpe": 3,
                "distance_m": 1500,
            },
        ]
        report = svc.analyze(1, "Test", history)
        assert len(report.taper_weeks) >= 1
        assert "2024-01-08" in report.taper_weeks

    def test_recovery_week(self, svc: PeriodizationService) -> None:
        history = [
            {
                "week_start": "2024-01-01",
                "date": "2024-01-03",
                "source": "match",
                "duration_min": 90,
                "rpe": 7,
                "distance_m": 10000,
            },
            {
                "week_start": "2024-01-08",
                "date": "2024-01-10",
                "source": "training",
                "duration_min": 45,
                "rpe": 3,
                "distance_m": 2000,
            },
        ]
        report = svc.analyze(1, "Test", history)
        assert report.weeks[1].is_recovery is True

    def test_overloaded_week(self, svc: PeriodizationService) -> None:
        history = [
            {
                "week_start": "2024-01-01",
                "date": f"2024-01-{d:02}",
                "source": "match",
                "duration_min": 90,
                "rpe": 7,
                "distance_m": 10000,
            }
            for d in range(1, 5)
        ]
        report = svc.analyze(1, "Test", history)
        assert report.weeks[0].congestion == CongestionLevel.OVERLOADED
        assert "2024-01-01" in report.congestion_weeks

    def test_load_trend_increasing(self, svc: PeriodizationService) -> None:
        history = []
        for week_idx, base in enumerate([100, 200, 300, 400, 500]):
            history.append(
                {
                    "week_start": f"2024-01-{((week_idx) * 7 + 1):02}",
                    "date": f"2024-01-{((week_idx) * 7 + 1):02}",
                    "source": "training",
                    "duration_min": base,
                    "rpe": 7,
                    "distance_m": base * 50,
                }
            )
        report = svc.analyze(1, "Test", history)
        assert report.load_trend == "increasing"

    def test_macrocycle_classification_well_structured(self, svc: PeriodizationService) -> None:
        weeks = [
            svc._build_week_summary(
                "2024-01-01",
                [
                    {"source": "training", "duration_min": 120, "rpe": 7, "distance_m": 6000},
                    {"source": "training", "duration_min": 90, "rpe": 7, "distance_m": 5000},
                    {"source": "training", "duration_min": 90, "rpe": 7, "distance_m": 5000},
                ],
            ),
            svc._build_week_summary(
                "2024-01-08",
                [
                    {"source": "training", "duration_min": 120, "rpe": 7, "distance_m": 6000},
                    {"source": "training", "duration_min": 90, "rpe": 7, "distance_m": 5000},
                ],
            ),
            svc._build_week_summary(
                "2024-01-15",
                [
                    {"source": "match", "duration_min": 90, "rpe": 7, "distance_m": 10000},
                ],
            ),
            svc._build_week_summary(
                "2024-01-22",
                [
                    {"source": "training", "duration_min": 45, "rpe": 3, "distance_m": 2000},
                ],
            ),
        ]
        result = svc.classify_macrocycle(weeks)
        assert result["cycle_type"] in ("well-structured", "irregular")

    def test_macrocycle_build_heavy(self, svc: PeriodizationService) -> None:
        weeks = []
        for i in range(5):
            weeks.append(
                svc._build_week_summary(
                    f"2024-01-{(i * 7 + 1):02}",
                    [
                        {"source": "training", "duration_min": 120, "rpe": 7, "distance_m": 6000},
                        {"source": "training", "duration_min": 90, "rpe": 7, "distance_m": 5000},
                        {"source": "training", "duration_min": 90, "rpe": 7, "distance_m": 5000},
                    ],
                )
            )
        result = svc.classify_macrocycle(weeks)
        assert result["cycle_type"] == "build-heavy"

    def test_calc_total_load_with_rpe(self, svc: PeriodizationService) -> None:
        records = [
            {"source": "match", "duration_min": 90, "rpe": 7},
            {"source": "training", "duration_min": 60, "rpe": 5},
        ]
        load = svc._calc_total_load(records)
        assert load == 90 * 7 + 60 * 5

    def test_calc_total_load_default(self, svc: PeriodizationService) -> None:
        records = [
            {"source": "match", "duration_min": 90},
            {"source": "training", "duration_min": 60},
        ]
        load = svc._calc_total_load(records)
        assert load == 90 * 7.0 + 60 * 4.0

    def test_classify_congestion(self, svc: PeriodizationService) -> None:
        assert svc._classify_congestion(0) == CongestionLevel.LIGHT
        assert svc._classify_congestion(1) == CongestionLevel.NORMAL
        assert svc._classify_congestion(2) == CongestionLevel.CONGESTED
        assert svc._classify_congestion(3) == CongestionLevel.OVERLOADED

    def test_classify_phase_recovery(self, svc: PeriodizationService) -> None:
        assert svc._classify_phase(0, 100, 0, 1) == CyclePhase.RECOVERY
        assert svc._classify_phase(2, 200, 1500, 0) == CyclePhase.COMPETITION
        assert svc._classify_phase(0, 240, 2500, 4) == CyclePhase.PEAK
        assert svc._classify_phase(0, 200, 1800, 3) == CyclePhase.BUILD
