"""Validation package — measures Kawkab models against public ground truth.

Loaders convert StatsBomb open-data events and Metrica Sports tracking data
into Kawkab's native event schema; metrics compute Brier scores,
reliability curves, AUC, and calibration; the CLI emits a reproducible
validation report.

This package has NO I/O side effects at import time and no dependency on
kawkab services — core-only, testable in isolation (same rule as the
rest of ``kawkab.core``).
"""

from kawkab.core.validation.metrica_loader import (
    MetricaMatch,
    load_metrica_match,
)
from kawkab.core.validation.metrics import (
    brier_score,
    brier_skill_score,
    calibration_error,
    log_loss,
    mean_absolute_error,
    pearson_correlation,
    reliability_curve,
    roc_auc,
)
from kawkab.core.validation.statsbomb_loader import (
    StatsBombMatch,
    extract_shots_for_fitting,
    load_statsbomb_corpus,
    load_statsbomb_match,
)

__all__ = [
    "StatsBombMatch",
    "load_statsbomb_match",
    "load_statsbomb_corpus",
    "extract_shots_for_fitting",
    "MetricaMatch",
    "load_metrica_match",
    "brier_score",
    "brier_skill_score",
    "reliability_curve",
    "roc_auc",
    "log_loss",
    "calibration_error",
    "mean_absolute_error",
    "pearson_correlation",
]
