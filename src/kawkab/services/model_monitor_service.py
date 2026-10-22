"""Model monitoring service — drift detection, performance tracking, auto-retraining."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kawkab.core.model_comparison import _compute_metrics

DRIFT_THRESHOLD_BRIER = 0.05
DRIFT_THRESHOLD_LOG_LOSS = 0.10
MONITORING_WINDOW_DAYS = 30


@dataclass
class ModelSnapshot:
    timestamp: float
    model_name: str
    log_loss: float
    brier_score: float
    auc_roc: float
    calibration_error: float
    shots_evaluated: int
    goals_actual: int
    goals_predicted: float


@dataclass
class DriftAlert:
    model_name: str
    metric: str
    current_value: float
    baseline_value: float
    threshold: float
    severity: str
    timestamp: float
    message: str


class ModelMonitor:
    def __init__(self, storage_dir: str | None = None):
        self._snapshots: list[ModelSnapshot] = []
        self._baseline: dict[str, ModelSnapshot] = {}
        self._storage_dir: Path | None = None
        if storage_dir:
            self._storage_dir = Path(storage_dir)
            self._storage_dir.mkdir(parents=True, exist_ok=True)

    def record_snapshot(
        self,
        model_name: str,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> ModelSnapshot:
        metrics = _compute_metrics(predictions, labels, model_name)
        snap = ModelSnapshot(
            timestamp=time.time(),
            model_name=model_name,
            log_loss=metrics.log_loss,
            brier_score=metrics.brier_score,
            auc_roc=metrics.auc_roc,
            calibration_error=metrics.calibration_error,
            shots_evaluated=metrics.shots_evaluated,
            goals_actual=metrics.goals_actual,
            goals_predicted=metrics.goals_predicted,
        )
        self._snapshots.append(snap)

        if model_name not in self._baseline:
            self._baseline[model_name] = snap

        if self._storage_dir:
            self._persist()
        return snap

    def set_baseline(self, model_name: str, snapshot: ModelSnapshot):
        self._baseline[model_name] = snapshot

    def detect_drift(self) -> list[DriftAlert]:
        alerts: list[DriftAlert] = []
        for snap in self._snapshots[-100:]:
            baseline = self._baseline.get(snap.model_name)
            if baseline is None:
                continue

            brier_change = abs(snap.brier_score - baseline.brier_score)
            if brier_change > DRIFT_THRESHOLD_BRIER:
                alerts.append(
                    DriftAlert(
                        model_name=snap.model_name,
                        metric="brier_score",
                        current_value=snap.brier_score,
                        baseline_value=baseline.brier_score,
                        threshold=DRIFT_THRESHOLD_BRIER,
                        severity="HIGH" if brier_change > DRIFT_THRESHOLD_BRIER * 2 else "MEDIUM",
                        timestamp=snap.timestamp,
                        message=f"{snap.model_name} Brier score changed by {brier_change:.4f} "
                        f"({baseline.brier_score:.4f} -> {snap.brier_score:.4f})",
                    )
                )

            ll_change = abs(snap.log_loss - baseline.log_loss)
            if ll_change > DRIFT_THRESHOLD_LOG_LOSS:
                alerts.append(
                    DriftAlert(
                        model_name=snap.model_name,
                        metric="log_loss",
                        current_value=snap.log_loss,
                        baseline_value=baseline.log_loss,
                        threshold=DRIFT_THRESHOLD_LOG_LOSS,
                        severity="HIGH" if ll_change > DRIFT_THRESHOLD_LOG_LOSS * 2 else "MEDIUM",
                        timestamp=snap.timestamp,
                        message=f"{snap.model_name} log-loss changed by {ll_change:.4f} "
                        f"({baseline.log_loss:.4f} -> {snap.log_loss:.4f})",
                    )
                )
        return alerts

    def get_recent_performance(self, model_name: str, n_last: int = 10) -> list[ModelSnapshot]:
        return [s for s in self._snapshots[-n_last:] if s.model_name == model_name]

    def get_trend(self, model_name: str, metric: str = "brier_score") -> list[tuple[float, float]]:
        return [
            (s.timestamp, getattr(s, metric)) for s in self._snapshots if s.model_name == model_name
        ]

    def _persist(self):
        if not self._storage_dir:
            return
        data = {
            "snapshots": [
                {
                    "timestamp": s.timestamp,
                    "model_name": s.model_name,
                    "log_loss": s.log_loss,
                    "brier_score": s.brier_score,
                    "auc_roc": s.auc_roc,
                    "calibration_error": s.calibration_error,
                    "shots_evaluated": s.shots_evaluated,
                    "goals_actual": s.goals_actual,
                    "goals_predicted": s.goals_predicted,
                }
                for s in self._snapshots
            ],
            "baseline": {
                name: {
                    "timestamp": snap.timestamp,
                    "model_name": snap.model_name,
                    "log_loss": snap.log_loss,
                    "brier_score": snap.brier_score,
                    "auc_roc": snap.auc_roc,
                }
                for name, snap in self._baseline.items()
            },
        }
        path = self._storage_dir / "model_monitor.json"
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, storage_dir: str) -> ModelMonitor:
        monitor = cls(storage_dir)
        path = Path(storage_dir) / "model_monitor.json"
        if not path.exists():
            return monitor
        try:
            data = json.loads(path.read_text())
            for s_data in data.get("snapshots", []):
                monitor._snapshots.append(ModelSnapshot(**s_data))
            for name, snap_data in data.get("baseline", {}).items():
                monitor._baseline[name] = ModelSnapshot(**snap_data)
        except Exception:
            pass
        return monitor


class ModelMonitoringService:
    """Service-level API for model monitoring, drift detection, and retraining."""

    def __init__(self, storage_dir: str | None = None):
        self.monitor = ModelMonitor.load(storage_dir) if storage_dir else ModelMonitor()

    def evaluate_and_record(
        self,
        model_name: str,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, Any]:
        snap = self.monitor.record_snapshot(model_name, predictions, labels)
        alerts = self.monitor.detect_drift()
        return {
            "snapshot": {
                "model_name": snap.model_name,
                "log_loss": snap.log_loss,
                "brier_score": snap.brier_score,
                "auc_roc": snap.auc_roc,
                "calibration_error": snap.calibration_error,
                "shots_evaluated": snap.shots_evaluated,
            },
            "alerts": [
                {
                    "metric": a.metric,
                    "severity": a.severity,
                    "message": a.message,
                }
                for a in alerts
            ],
            "drift_detected": len(alerts) > 0,
        }

    def get_monitoring_dashboard(self) -> dict[str, Any]:
        models_seen = {s.model_name for s in self.monitor._snapshots}
        dashboard: dict[str, Any] = {
            "models": {},
            "total_evaluations": len(self.monitor._snapshots),
            "active_alerts": [],
        }

        for model_name in models_seen:
            snapshots = self.monitor.get_recent_performance(model_name, 20)
            if not snapshots:
                continue
            latest = snapshots[-1]
            baseline = self.monitor._baseline.get(model_name)
            trend = self.monitor.get_trend(model_name)

            brier_trend = [v for _, v in trend[-10:]] if trend else []
            brier_direction = "stable"
            if len(brier_trend) >= 3:
                first_half = np.mean(brier_trend[: len(brier_trend) // 2])
                second_half = np.mean(brier_trend[len(brier_trend) // 2 :])
                if second_half > first_half + DRIFT_THRESHOLD_BRIER:
                    brier_direction = "degrading"
                elif first_half > second_half + DRIFT_THRESHOLD_BRIER:
                    brier_direction = "improving"

            drift_alerts = self.monitor.detect_drift()
            model_alerts = [a for a in drift_alerts if a.model_name == model_name]

            dashboard["models"][model_name] = {
                "latest": {
                    "log_loss": latest.log_loss,
                    "brier_score": latest.brier_score,
                    "auc_roc": latest.auc_roc,
                    "calibration_error": latest.calibration_error,
                    "shots_evaluated": latest.shots_evaluated,
                },
                "baseline": {
                    "log_loss": baseline.log_loss if baseline else None,
                    "brier_score": baseline.brier_score if baseline else None,
                }
                if baseline
                else None,
                "trend": {
                    "brier_direction": brier_direction,
                    "data_points": len(trend),
                },
                "alerts": [
                    {
                        "metric": a.metric,
                        "severity": a.severity,
                        "message": a.message,
                    }
                    for a in model_alerts
                ],
            }
            dashboard["active_alerts"].extend(a for a in model_alerts if a.severity == "HIGH")

        return dashboard

    def check_need_retrain(self, model_name: str) -> dict[str, Any]:
        recents = self.monitor.get_recent_performance(model_name, 5)
        if len(recents) < 3:
            return {"needs_retrain": False, "reason": "insufficient_data"}

        baseline = self.monitor._baseline.get(model_name)
        if baseline is None:
            return {"needs_retrain": False, "reason": "no_baseline"}

        recent_brier = np.mean([s.brier_score for s in recents])
        if recent_brier > baseline.brier_score + DRIFT_THRESHOLD_BRIER:
            return {
                "needs_retrain": True,
                "reason": f"Brier degraded from {baseline.brier_score:.4f} to {recent_brier:.4f}",
                "current_brier": round(recent_brier, 4),
                "baseline_brier": round(baseline.brier_score, 4),
            }
        return {"needs_retrain": False, "reason": "within_tolerance"}
