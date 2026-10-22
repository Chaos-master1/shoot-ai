"""Tests for ModelCalibrator."""

from __future__ import annotations

import pytest

from kawkab.core.calibration import ModelCalibrator


@pytest.fixture
def calibrator() -> ModelCalibrator:
    return ModelCalibrator()


def make_shot(xg: float, is_goal: bool = False) -> dict:
    return {"type": "shot", "xg": xg, "is_goal": is_goal, "metadata": {"xg": xg}}


class TestComputeCalibrationStats:
    def test_empty_events(self, calibrator: ModelCalibrator) -> None:
        result = calibrator.compute_calibration_stats([])
        assert result["total_xg"] == 0.0
        assert result["actual_goals"] == 0
        assert result["n_shots"] == 0
        assert result["brier_score"] == 0.0

    def test_no_goals(self, calibrator: ModelCalibrator) -> None:
        events = [make_shot(0.5), make_shot(0.3)]
        result = calibrator.compute_calibration_stats(events)
        assert result["total_xg"] == 0.8
        assert result["actual_goals"] == 0
        assert result["calibration_error"] == 0.8
        assert result["n_shots"] == 2

    def test_with_goals(self, calibrator: ModelCalibrator) -> None:
        events = [make_shot(0.5, True), make_shot(0.3)]
        result = calibrator.compute_calibration_stats(events)
        assert result["actual_goals"] == 1
        assert result["calibration_error"] == 0.2

    def test_brier_score_perfect(self, calibrator: ModelCalibrator) -> None:
        events = [make_shot(1.0, True), make_shot(0.0)]
        result = calibrator.compute_calibration_stats(events)
        assert result["brier_score"] == 0.0

    def test_brier_score_imperfect(self, calibrator: ModelCalibrator) -> None:
        events = [make_shot(0.5, True)]
        result = calibrator.compute_calibration_stats(events)
        assert result["brier_score"] == pytest.approx((1 - 0.5) ** 2)

    def test_non_shot_events_ignored(self, calibrator: ModelCalibrator) -> None:
        events = [{"type": "pass"}, {"type": "foul"}]
        result = calibrator.compute_calibration_stats(events)
        assert result["n_shots"] == 0

    def test_xg_from_metadata(self, calibrator: ModelCalibrator) -> None:
        events = [{"type": "shot", "metadata": {"xg": 0.7}, "is_goal": True}]
        result = calibrator.compute_calibration_stats(events)
        assert result["total_xg"] == 0.7
        assert result["actual_goals"] == 1


class TestReliabilityCurve:
    def test_empty_events(self, calibrator: ModelCalibrator) -> None:
        curve = calibrator.compute_reliability_curve([])
        assert len(curve) == 10

    def test_all_bins_have_entries(self, calibrator: ModelCalibrator) -> None:
        events = [
            make_shot(0.05),
            make_shot(0.15),
            make_shot(0.25),
            make_shot(0.35),
            make_shot(0.45),
            make_shot(0.55),
            make_shot(0.65),
            make_shot(0.75),
            make_shot(0.85),
            make_shot(0.95),
        ]
        curve = calibrator.compute_reliability_curve(events)
        filled = [b for b in curve if b["count"] > 0]
        assert len(filled) == 10

    def test_observed_rate_is_correct(self, calibrator: ModelCalibrator) -> None:
        events = [make_shot(0.7, True), make_shot(0.7)]
        curve = calibrator.compute_reliability_curve(events, n_bins=10)
        bin_7 = [b for b in curve if b["bin_range"].startswith("0.7")][0]
        assert bin_7["count"] == 2
        assert bin_7["observed_rate"] == 0.5

    def test_empty_bin_returns_none(self, calibrator: ModelCalibrator) -> None:
        curve = calibrator.compute_reliability_curve([], n_bins=10)
        assert all(b["observed_rate"] is None for b in curve)


class TestLogLoss:
    def test_empty_events(self, calibrator: ModelCalibrator) -> None:
        result = calibrator.compute_log_loss([])
        assert result["log_loss"] == 0.0
        assert result["n_shots"] == 0

    def test_perfect_prediction(self, calibrator: ModelCalibrator) -> None:
        events = [make_shot(1.0, True), make_shot(0.0)]
        result = calibrator.compute_log_loss(events)
        assert result["log_loss"] == 0.0

    def test_imperfect_prediction(self, calibrator: ModelCalibrator) -> None:
        events = [make_shot(0.5, True)]
        result = calibrator.compute_log_loss(events)
        assert result["log_loss"] > 0
        assert result["n_shots"] == 1


class TestGenerateCalibrationReport:
    def test_empty_events(self, calibrator: ModelCalibrator) -> None:
        report = calibrator.generate_calibration_report([])
        assert report["status"] == "insufficient_data"
        assert report["n_shots"] == 0

    def test_well_calibrated(self, calibrator: ModelCalibrator) -> None:
        events = [make_shot(1.0, True), make_shot(0.0)]
        report = calibrator.generate_calibration_report(events)
        assert report["status"] == "well_calibrated"

    def test_report_contains_all_keys(self, calibrator: ModelCalibrator) -> None:
        events = [make_shot(0.5, True)]
        report = calibrator.generate_calibration_report(events)
        assert "total_xg" in report
        assert "actual_goals" in report
        assert "calibration_error" in report
        assert "brier_score" in report
        assert "log_loss" in report
        assert "reliability_curve" in report
        assert "status" in report
        assert "n_shots" in report
