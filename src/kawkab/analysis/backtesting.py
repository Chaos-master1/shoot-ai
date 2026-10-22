"""Backtesting framework — model-agnostic prediction validation.

Computes calibration curves, reliability diagrams, and standard
metrics (log-loss, Brier score, AUC-ROC, calibration error) for
any model that outputs probabilities.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BacktestMetrics:
    model_name: str = ""
    log_loss: float = 0.0
    brier_score: float = 0.0
    auc_roc: float = 0.0
    calibration_error: float = 0.0
    samples: int = 0
    positives: int = 0
    avg_prediction: float = 0.0
    avg_actual: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "log_loss": round(self.log_loss, 4),
            "brier_score": round(self.brier_score, 4),
            "auc_roc": round(self.auc_roc, 4),
            "calibration_error": round(self.calibration_error, 4),
            "samples": self.samples,
            "positives": self.positives,
            "avg_prediction": round(self.avg_prediction, 4),
            "avg_actual": round(self.avg_actual, 4),
        }


@dataclass
class CalibrationBin:
    bin_low: float = 0.0
    bin_high: float = 0.0
    count: int = 0
    mean_predicted: float = 0.0
    mean_actual: float = 0.0

    def to_dict(self) -> dict:
        return {
            "bin": f"{self.bin_low:.2f}-{self.bin_high:.2f}",
            "count": self.count,
            "mean_predicted": round(self.mean_predicted, 4),
            "mean_actual": round(self.mean_actual, 4),
        }


@dataclass
class BacktestReport:
    metrics: BacktestMetrics = field(default_factory=BacktestMetrics)
    calibration_bins: list[CalibrationBin] = field(default_factory=list)
    predictions: list[float] = field(default_factory=list)
    actuals: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.to_dict(),
            "calibration_bins": [b.to_dict() for b in self.calibration_bins],
            "n_predictions": len(self.predictions),
        }


def _log_loss(y_true: list[int], y_pred: list[float], eps: float = 1e-15) -> float:
    y_pred = [max(eps, min(1 - eps, p)) for p in y_pred]
    return -sum(
        t * math.log(p) + (1 - t) * math.log(1 - p) for t, p in zip(y_true, y_pred, strict=False)
    ) / max(1, len(y_true))


def _brier_score(y_true: list[int], y_pred: list[float]) -> float:
    return sum((t - p) ** 2 for t, p in zip(y_true, y_pred, strict=False)) / max(1, len(y_true))


def _auc_roc(y_true: list[int], y_pred: list[float]) -> float:
    n = len(y_true)
    pos = sum(y_true)
    neg = n - pos
    if pos == 0 or neg == 0:
        return 0.5
    pairs = sorted(
        zip(y_pred, y_true, range(n), strict=False),
        key=lambda x: x[0],
    )
    rank_sum = 0.0
    i = 0
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        i = j
    u = rank_sum - pos * (pos + 1) / 2
    return u / (pos * neg)


def _calibration_curve(
    y_true: list[int],
    y_pred: list[float],
    n_bins: int = 10,
) -> list[CalibrationBin]:
    pairs = list(zip(y_pred, y_true, strict=False))
    bins = []
    for i in range(n_bins):
        low = i / n_bins
        high = (i + 1) / n_bins
        in_bin = [(p, t) for p, t in pairs if low <= p < high]
        if not in_bin:
            continue
        mean_pred = sum(p for p, _ in in_bin) / len(in_bin)
        mean_act = sum(t for _, t in in_bin) / len(in_bin)
        bins.append(
            CalibrationBin(
                bin_low=low,
                bin_high=high,
                count=len(in_bin),
                mean_predicted=mean_pred,
                mean_actual=mean_act,
            )
        )
    return bins


def _calibration_error(bins: list[CalibrationBin]) -> float:
    if not bins:
        return 0.0
    total = sum(b.count for b in bins)
    return sum(b.count * abs(b.mean_predicted - b.mean_actual) for b in bins) / max(1, total)


def evaluate_model(
    y_true: list[int],
    y_pred: list[float],
    model_name: str = "model",
    n_bins: int = 10,
) -> BacktestReport:
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return BacktestReport(
            metrics=BacktestMetrics(model_name=model_name),
            errors=["Empty or mismatched predictions/actuals"],
        )

    report = BacktestReport(
        metrics=BacktestMetrics(
            model_name=model_name,
            samples=len(y_true),
            positives=sum(y_true),
            avg_prediction=sum(y_pred) / len(y_pred),
            avg_actual=sum(y_true) / len(y_true),
            log_loss=round(_log_loss(y_true, y_pred), 4),
            brier_score=round(_brier_score(y_true, y_pred), 4),
            auc_roc=round(_auc_roc(y_true, y_pred), 4),
        ),
        calibration_bins=_calibration_curve(y_true, y_pred, n_bins),
        predictions=y_pred,
        actuals=y_true,
    )
    report.metrics.calibration_error = round(_calibration_error(report.calibration_bins), 4)
    return report


def evaluate_model_single(
    model_fn: Callable[[dict], float],
    samples: list[dict[str, Any]],
    label_key: str = "is_goal",
    model_name: str = "model",
    n_bins: int = 10,
) -> BacktestReport:
    y_true = []
    y_pred = []
    errors = []
    for s in samples:
        true_val = s.get(label_key, 0)
        try:
            pred = model_fn(s)
            y_true.append(1 if true_val else 0)
            y_pred.append(float(pred))
        except Exception as e:
            errors.append(str(e))
    report = evaluate_model(y_true, y_pred, model_name, n_bins)
    report.errors.extend(errors)
    return report


def compare_models(
    results: list[BacktestReport],
) -> dict[str, Any]:
    if not results:
        return {"best": "", "ranking": []}
    ranked = sorted(results, key=lambda r: r.metrics.log_loss)
    best = ranked[0].metrics.model_name if ranked else ""
    return {
        "best": best,
        "ranking": [
            {
                "rank": i + 1,
                "model": r.metrics.model_name,
                "log_loss": r.metrics.log_loss,
                "brier_score": r.metrics.brier_score,
                "auc_roc": r.metrics.auc_roc,
                "calibration_error": r.metrics.calibration_error,
            }
            for i, r in enumerate(ranked)
        ],
    }


def calibration_chart_data(
    bins: list[CalibrationBin],
) -> dict[str, Any]:
    return {
        "labels": [f"{b.bin_low:.1f}-{b.bin_high:.1f}" for b in bins],
        "predicted": [round(b.mean_predicted, 3) for b in bins],
        "actual": [round(b.mean_actual, 3) for b in bins],
        "counts": [b.count for b in bins],
    }
