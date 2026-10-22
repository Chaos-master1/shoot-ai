"""Tests for backtesting framework."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.analysis.backtesting import (
    BacktestMetrics,
    CalibrationBin,
    _auc_roc,
    _brier_score,
    _calibration_curve,
    _calibration_error,
    _log_loss,
    calibration_chart_data,
    compare_models,
    evaluate_model,
    evaluate_model_single,
)


def test_log_loss_perfect():
    assert _log_loss([1, 0, 1], [0.99, 0.01, 0.99]) < 0.1


def test_log_loss_poor():
    ll = _log_loss([1, 0, 1], [0.01, 0.99, 0.01])
    assert ll > 2.0


def test_log_loss_empty():
    assert _log_loss([], []) == 0.0


def test_brier_score_perfect():
    assert abs(_brier_score([1, 0], [1.0, 0.0])) < 1e-6


def test_brier_score_half():
    bs = _brier_score([1, 0], [0.5, 0.5])
    assert abs(bs - 0.25) < 1e-6


def test_auc_roc_perfect():
    assert _auc_roc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0


def test_auc_roc_random():
    auc = _auc_roc([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5])
    assert 0.4 <= auc <= 0.6


def test_auc_roc_imperfect():
    # Positive predictions should be mostly (not perfectly) higher than negatives
    y_true = [1, 0, 1, 0, 1, 0]
    y_pred = [0.9, 0.7, 0.6, 0.5, 0.3, 0.2]
    auc = _auc_roc(y_true, y_pred)
    assert 0.5 < auc < 1.0


def test_auc_roc_reverse():
    # Reversed ranking should give AUC < 0.5
    auc = _auc_roc([1, 0, 1, 0], [0.1, 0.9, 0.2, 0.8])
    assert auc < 0.5


def test_auc_roc_all_same():
    assert _auc_roc([0, 0, 0], [0.5, 0.5, 0.5]) == 0.5
    assert _auc_roc([1, 1, 1], [0.5, 0.5, 0.5]) == 0.5


def test_calibration_curve():
    y_true = [1, 0, 1, 0, 1]
    y_pred = [0.9, 0.1, 0.8, 0.2, 0.7]
    bins = _calibration_curve(y_true, y_pred, n_bins=5)
    assert len(bins) > 0
    for b in bins:
        assert b.count > 0
        assert b.bin_low < b.bin_high
        assert 0 <= b.mean_predicted <= 1
        assert 0 <= b.mean_actual <= 1


def test_calibration_error_perfect():
    y_true = [1, 1, 0, 0]
    y_pred = [0.9, 0.8, 0.2, 0.1]
    bins = _calibration_curve(y_true, y_pred, n_bins=4)
    err = _calibration_error(bins)
    assert err < 0.3


def test_calibration_error_empty():
    assert _calibration_error([]) == 0.0


def test_evaluate_model():
    y_true = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    y_pred = [0.8, 0.2, 0.7, 0.3, 0.9, 0.1, 0.6, 0.4, 0.85, 0.15]
    report = evaluate_model(y_true, y_pred, "test_model")
    assert report.metrics.model_name == "test_model"
    assert report.metrics.samples == 10
    assert report.metrics.positives == 5
    assert report.metrics.log_loss > 0
    assert report.metrics.brier_score > 0
    assert report.metrics.auc_roc > 0.5
    assert len(report.calibration_bins) > 0
    assert len(report.predictions) == 10


def test_evaluate_model_empty():
    report = evaluate_model([], [], "empty")
    assert len(report.errors) > 0


def test_evaluate_model_mismatched():
    report = evaluate_model([1, 0], [0.5], "bad")
    assert len(report.errors) > 0


def test_evaluate_model_single():
    def fake_model(sample):
        return 0.7 if sample.get("dangerous") else 0.2

    samples = [
        {"dangerous": True, "is_goal": 1},
        {"dangerous": False, "is_goal": 0},
        {"dangerous": True, "is_goal": 0},
        {"dangerous": False, "is_goal": 0},
    ]
    report = evaluate_model_single(fake_model, samples, label_key="is_goal", model_name="fake")
    assert report.metrics.samples == 4
    assert report.metrics.model_name == "fake"
    assert report.metrics.log_loss > 0


def test_evaluate_model_single_with_error():
    def broken(_s):
        raise ValueError("oops")

    report = evaluate_model_single(broken, [{"is_goal": 1}], model_name="broken")
    assert len(report.errors) > 0


def test_compare_models():
    r1 = evaluate_model([1, 0, 1], [0.9, 0.1, 0.8], "model_a")
    r2 = evaluate_model([1, 0, 1], [0.6, 0.4, 0.55], "model_b")
    result = compare_models([r1, r2])
    assert result["best"] == "model_a"
    assert len(result["ranking"]) == 2


def test_compare_models_empty():
    assert compare_models([]) == {"best": "", "ranking": []}


def test_calibration_chart_data():
    bins = [
        CalibrationBin(bin_low=0, bin_high=0.5, count=5, mean_predicted=0.3, mean_actual=0.4),
        CalibrationBin(bin_low=0.5, bin_high=1.0, count=3, mean_predicted=0.7, mean_actual=0.6),
    ]
    chart = calibration_chart_data(bins)
    assert len(chart["labels"]) == 2
    assert chart["predicted"] == [0.3, 0.7]
    assert chart["counts"] == [5, 3]


def test_backtest_metrics_to_dict():
    m = BacktestMetrics(
        model_name="xG",
        log_loss=0.5,
        brier_score=0.15,
        auc_roc=0.85,
        calibration_error=0.05,
        samples=100,
        positives=30,
    )
    d = m.to_dict()
    assert d["model_name"] == "xG"
    assert d["log_loss"] == 0.5
    assert d["auc_roc"] == 0.85


def test_report_to_dict():
    report = evaluate_model([1, 0, 1], [0.8, 0.2, 0.9], "test")
    d = report.to_dict()
    assert d["metrics"]["samples"] == 3
    assert d["n_predictions"] == 3
    assert len(d["calibration_bins"]) > 0


def test_calibration_bin_to_dict():
    b = CalibrationBin(bin_low=0.0, bin_high=0.5, count=10, mean_predicted=0.3, mean_actual=0.25)
    d = b.to_dict()
    assert d["bin"] == "0.00-0.50"
    assert d["count"] == 10


def test_well_calibrated_model():
    y_true = [
        1 if p > 0.5 else 0
        for _ in range(100)
        for p in [0.9, 0.1, 0.7, 0.3, 0.8, 0.2, 0.6, 0.4, 0.95, 0.05]
    ]
    y_pred = [0.9, 0.1, 0.7, 0.3, 0.8, 0.2, 0.6, 0.4, 0.95, 0.05] * 10
    report = evaluate_model(y_true, y_pred, "well_calibrated", n_bins=5)
    assert report.metrics.calibration_error < 0.1
