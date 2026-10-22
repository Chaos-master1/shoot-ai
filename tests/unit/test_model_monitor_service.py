"""Tests for model monitoring service."""

from __future__ import annotations

import tempfile

import numpy as np

from kawkab.services.model_monitor_service import ModelMonitor, ModelMonitoringService


class TestModelMonitor:
    def test_record_snapshot(self):
        monitor = ModelMonitor()
        preds = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        labels = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
        snap = monitor.record_snapshot("test_model", preds, labels)
        assert snap.model_name == "test_model"
        assert snap.log_loss > 0
        assert snap.brier_score > 0
        assert snap.shots_evaluated == 5

    def test_baseline_auto_set(self):
        monitor = ModelMonitor()
        preds = np.array([0.2, 0.8])
        labels = np.array([0.0, 1.0])
        monitor.record_snapshot("m1", preds, labels)
        assert "m1" in monitor._baseline

    def test_drift_no_alert_when_stable(self):
        monitor = ModelMonitor()
        for _ in range(3):
            preds = np.array([0.2, 0.5, 0.8])
            labels = np.array([0.0, 1.0, 1.0])
            monitor.record_snapshot("stable", preds, labels)
        alerts = monitor.detect_drift()
        assert len(alerts) == 0

    def test_drift_alert_on_degradation(self):
        monitor = ModelMonitor()
        preds_good = np.array([0.2, 0.5, 0.8])
        labels = np.array([0.0, 1.0, 1.0])
        monitor.record_snapshot("bad_model", preds_good, labels)

        preds_bad = np.array([0.5, 0.6, 0.7])
        monitor.record_snapshot("bad_model", preds_bad, labels)
        alerts = monitor.detect_drift()
        assert len(alerts) >= 1

    def test_persist_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            monitor = ModelMonitor(tmp)
            preds = np.array([0.1, 0.9])
            labels = np.array([0.0, 1.0])
            monitor.record_snapshot("m1", preds, labels)

            loaded = ModelMonitor.load(tmp)
            assert len(loaded._snapshots) == 1
            assert loaded._snapshots[0].model_name == "m1"

    def test_get_recent_performance(self):
        monitor = ModelMonitor()
        for i in range(5):
            preds = np.array([0.1 + i * 0.1, 0.9])
            labels = np.array([0.0, 1.0])
            monitor.record_snapshot("trend_model", preds, labels)
        recents = monitor.get_recent_performance("trend_model", 3)
        assert len(recents) == 3

    def test_get_trend(self):
        monitor = ModelMonitor()
        for _i in range(3):
            preds = np.array([0.2, 0.8])
            labels = np.array([0.0, 1.0])
            monitor.record_snapshot("trend2", preds, labels)
        trend = monitor.get_trend("trend2", "brier_score")
        assert len(trend) == 3

    def test_empty_monitor_no_alerts(self):
        monitor = ModelMonitor()
        assert monitor.detect_drift() == []

    def test_model_name_in_snapshot(self):
        monitor = ModelMonitor()
        preds = np.array([0.5])
        labels = np.array([1.0])
        snap = monitor.record_snapshot("unique_name", preds, labels)
        assert snap.model_name == "unique_name"


class TestModelMonitoringService:
    def test_evaluate_and_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = ModelMonitoringService(tmp)
            preds = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
            labels = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
            result = svc.evaluate_and_record("test", preds, labels)
            assert "snapshot" in result
            assert "drift_detected" in result
            assert result["snapshot"]["shots_evaluated"] == 5

    def test_monitoring_dashboard(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = ModelMonitoringService(tmp)
            preds = np.array([0.1, 0.5, 0.9])
            labels = np.array([0.0, 1.0, 1.0])
            svc.evaluate_and_record("m1", preds, labels)
            dashboard = svc.get_monitoring_dashboard()
            assert "models" in dashboard
            assert "m1" in dashboard["models"]
            assert dashboard["total_evaluations"] == 1

    def test_need_retrain_insufficient_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = ModelMonitoringService(tmp)
            result = svc.check_need_retrain("new_model")
            assert not result["needs_retrain"]

    def test_need_retrain_with_enough_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = ModelMonitoringService(tmp)
            preds_good = np.array([0.2, 0.5, 0.8])
            labels = np.array([0.0, 1.0, 1.0])
            for _ in range(5):
                svc.evaluate_and_record("m2", preds_good, labels)
            result = svc.check_need_retrain("m2")
            assert "needs_retrain" in result

    def test_dashboard_empty_when_no_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = ModelMonitoringService(tmp)
            dashboard = svc.get_monitoring_dashboard()
            assert dashboard["models"] == {}
            assert dashboard["total_evaluations"] == 0
