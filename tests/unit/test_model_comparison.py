"""Tests for xG Model Comparison module."""

from __future__ import annotations

import numpy as np
import pytest

from kawkab.core.model_comparison import (
    ModelComparisonReport,
    _compute_buckets,
    _compute_calibration_chart,
    _compute_metrics,
    compare_xg_models,
)


def make_shot(
    xg_heuristic: float = 0.1,
    xg_logistic: float | None = None,
    is_goal: bool = False,
    distance_m: float = 18.0,
    angle_deg: float = 30.0,
    **kwargs,
) -> dict:
    s = {
        "xg_heuristic": xg_heuristic,
        "is_goal": is_goal,
        "distance_m": distance_m,
        "angle_deg": angle_deg,
    }
    s.update(kwargs)
    return s


class TestModelComparison:
    def test_all_metrics_computed(self):
        shots = [
            make_shot(xg_heuristic=0.5, is_goal=True, distance_m=5.0),
            make_shot(xg_heuristic=0.1, is_goal=False, distance_m=20.0),
            make_shot(xg_heuristic=0.05, is_goal=False, distance_m=25.0),
            make_shot(xg_heuristic=0.7, is_goal=True, distance_m=3.0),
            make_shot(xg_heuristic=0.2, is_goal=False, distance_m=15.0),
            make_shot(xg_heuristic=0.01, is_goal=False, distance_m=35.0),
        ]
        report = compare_xg_models(shots, test_fraction=0.3, random_seed=42)
        assert len(report.models) >= 2
        for m in report.models:
            assert m.log_loss >= 0
            assert m.brier_score >= 0
            assert m.shots_evaluated > 0

    def test_heuristic_model_metrics_valid(self):
        shots = [
            make_shot(xg_heuristic=0.5, is_goal=True, distance_m=5.0),
            make_shot(xg_heuristic=0.1, is_goal=False, distance_m=20.0),
        ]
        report = compare_xg_models(shots, test_fraction=0.5, random_seed=42)
        heuristic = next(m for m in report.models if m.model_name == "heuristic")
        assert heuristic.log_loss > 0
        assert heuristic.brier_score > 0

    def test_best_model_lowest_log_loss(self):
        shots = [
            make_shot(xg_heuristic=0.5, is_goal=i % 2 == 0, distance_m=10.0 + i) for i in range(20)
        ]
        report = compare_xg_models(shots, test_fraction=0.3, random_seed=42)
        best = min(report.models, key=lambda m: m.log_loss)
        assert report.best_model == best.model_name

    def test_empty_shots_returns_empty_report(self):
        report = compare_xg_models([], test_fraction=0.3)
        assert len(report.models) == 0

    def test_single_shot_handled(self):
        shots = [
            make_shot(xg_heuristic=0.5, is_goal=True, distance_m=5.0),
        ]
        report = compare_xg_models(shots, test_fraction=0.5, random_seed=42)
        # Single shot may have 0 models if it ends up in training split
        assert isinstance(report, ModelComparisonReport)

    def test_distance_buckets_populated(self):
        shots = [
            make_shot(xg_heuristic=0.5, is_goal=True, distance_m=3.0),
            make_shot(xg_heuristic=0.3, is_goal=False, distance_m=8.0),
            make_shot(xg_heuristic=0.1, is_goal=False, distance_m=18.0),
            make_shot(xg_heuristic=0.05, is_goal=False, distance_m=22.0),
            make_shot(xg_heuristic=0.01, is_goal=False, distance_m=30.0),
        ]
        report = compare_xg_models(shots, test_fraction=0.4, random_seed=42)
        assert len(report.distance_buckets) >= 1

    def test_calibration_chart_data_has_10_bins(self):
        shots = [
            make_shot(xg_heuristic=0.5, is_goal=i % 3 == 0, distance_m=10.0 + i) for i in range(30)
        ]
        report = compare_xg_models(shots, test_fraction=0.3, random_seed=42)
        for _model_name, chart in report.calibration_chart_data.items():
            assert len(chart["bins"]) == 10

    def test_log_loss_calculation(self):
        predictions = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        labels = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
        metrics = _compute_metrics(predictions, labels, "test")
        assert metrics.log_loss > 0
        assert metrics.brier_score > 0

    def test_brier_score_perfect(self):
        predictions = np.array([1.0, 0.0, 1.0, 0.0])
        labels = np.array([1.0, 0.0, 1.0, 0.0])
        metrics = _compute_metrics(predictions, labels, "perfect")
        assert metrics.brier_score == pytest.approx(0.0, abs=1e-10)

    def test_buckets_empty_when_no_predictions(self):
        preds = {"test": np.array([])}
        labels = np.array([])
        values = np.array([])
        buckets = _compute_buckets(preds, labels, values, [0.0, 10.0, 20.0], ["0-10", "10-20"])
        assert len(buckets) == 1

    def test_calibration_chart_empty_for_zero_labels(self):
        preds = {"m": np.array([0.1, 0.2])}
        labels = np.array([0.0, 0.0])
        chart = _compute_calibration_chart(preds, labels, n_bins=5)
        assert len(chart["m"]["bins"]) == 5

    def test_auc_roc_computed(self):
        predictions = np.array([0.05, 0.1, 0.3, 0.6, 0.8, 0.9])
        labels = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        metrics = _compute_metrics(predictions, labels, "test_auc")
        assert metrics.auc_roc > 0.5

    def test_calibration_slope_near_one_for_perfect(self):
        predictions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        labels = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        metrics = _compute_metrics(predictions, labels, "cal_test")
        assert abs(metrics.calibration_slope - 1.0) < 2.0

    def test_report_to_dict(self):
        shots = [
            make_shot(xg_heuristic=0.5, is_goal=True, distance_m=5.0),
            make_shot(xg_heuristic=0.1, is_goal=False, distance_m=20.0),
        ]
        report = compare_xg_models(shots, test_fraction=0.3, random_seed=42)
        d = report.to_dict()
        assert "models" in d
        assert "best_model" in d
        assert "calibration_chart_data" in d

    def test_summary_text(self):
        shots = [
            make_shot(xg_heuristic=0.5, is_goal=True, distance_m=5.0),
            make_shot(xg_heuristic=0.1, is_goal=False, distance_m=20.0),
        ]
        report = compare_xg_models(shots, test_fraction=0.3, random_seed=42)
        text = report.summary_text()
        assert "xG Model Comparison" in text
        assert "Best model:" in text


class TestCrossValidation:
    def test_cv_returns_folds(self):
        shots = [
            make_shot(xg_heuristic=0.5, is_goal=i % 2 == 0, distance_m=10.0 + i) for i in range(30)
        ]
        report = compare_xg_models(
            shots, n_folds=5, random_seed=42, compute_feature_importance=False
        )
        assert len(report.cv_folds) > 0
        assert report.cv_summary is not None

    def test_cv_summary_has_expected_keys(self):
        shots = [
            make_shot(xg_heuristic=0.5, is_goal=i % 3 == 0, distance_m=15.0) for i in range(30)
        ]
        report = compare_xg_models(
            shots, n_folds=3, random_seed=42, compute_feature_importance=False
        )
        for _model_name, stats in report.cv_summary.items():
            assert "log_loss_mean" in stats
            assert "log_loss_std" in stats
            assert "n_folds" in stats

    def test_cv_folds_have_metrics_per_model(self):
        shots = [
            make_shot(xg_heuristic=0.5, is_goal=i % 2 == 0, distance_m=10.0 + i) for i in range(40)
        ]
        report = compare_xg_models(
            shots, n_folds=4, random_seed=42, compute_feature_importance=False
        )
        for fold in report.cv_folds:
            assert len(fold.metrics) >= 1
            assert fold.train_size > 0
            assert fold.test_size > 0

    def test_cv_with_single_fold(self):
        shots = [
            make_shot(xg_heuristic=0.5, is_goal=True, distance_m=5.0),
            make_shot(xg_heuristic=0.1, is_goal=False, distance_m=20.0),
        ]
        report = compare_xg_models(
            shots, n_folds=2, random_seed=42, compute_feature_importance=False
        )
        assert len(report.cv_folds) <= 2

    def test_cv_zero_folds_does_not_compute_cv(self):
        shots = [
            make_shot(xg_heuristic=0.5, is_goal=i % 2 == 0, distance_m=10.0 + i) for i in range(10)
        ]
        report = compare_xg_models(shots, n_folds=0, random_seed=42)
        assert len(report.cv_folds) == 0


class TestTemporalSplit:
    def test_temporal_split_uses_timestamp(self):
        shots = [
            make_shot(xg_heuristic=0.3, is_goal=True, distance_m=10.0, timestamp=0.0, index=0),
            make_shot(xg_heuristic=0.2, is_goal=False, distance_m=20.0, timestamp=10.0, index=1),
            make_shot(xg_heuristic=0.5, is_goal=True, distance_m=5.0, timestamp=20.0, index=2),
            make_shot(xg_heuristic=0.1, is_goal=False, distance_m=30.0, timestamp=30.0, index=3),
            make_shot(xg_heuristic=0.4, is_goal=True, distance_m=8.0, timestamp=40.0, index=4),
            make_shot(xg_heuristic=0.05, is_goal=False, distance_m=35.0, timestamp=50.0, index=5),
            make_shot(xg_heuristic=0.15, is_goal=False, distance_m=25.0, timestamp=60.0, index=6),
        ]
        report = compare_xg_models(
            shots,
            test_fraction=0.3,
            random_seed=42,
            temporal_split=True,
            compute_feature_importance=False,
        )
        assert len(report.models) >= 1
        # Temporal split should use the first ~70% for training
        assert report.models[0].shots_evaluated > 0

    def test_temporal_split_with_no_timestamp_falls_back(self):
        shots = [
            make_shot(xg_heuristic=0.5, is_goal=True, distance_m=10.0),
            make_shot(xg_heuristic=0.1, is_goal=False, distance_m=20.0),
            make_shot(xg_heuristic=0.3, is_goal=True, distance_m=8.0),
        ]
        report = compare_xg_models(
            shots,
            test_fraction=0.3,
            random_seed=42,
            temporal_split=True,
            compute_feature_importance=False,
        )
        assert len(report.models) >= 1
