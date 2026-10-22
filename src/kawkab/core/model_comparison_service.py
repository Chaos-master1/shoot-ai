"""Model comparison: evaluates all xG models on the same data.

Service-layer wrapper that uses existing xG models
(heuristic, enhanced, DL) and produces comparison reports.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from kawkab.core.dl_xg_model import DLXgModel
from kawkab.core.xg_calibration import (
    CalibrationCurve,
    compute_auc_roc,
    compute_brier_score,
    compute_calibration_curve,
    compute_log_loss,
)
from kawkab.core.xg_model import EnhancedXgModel, compute_xg


@dataclass
class ModelComparisonReport:
    model_name: str
    brier_score: float
    log_loss: float
    auc_roc: float
    calibration_error: float
    calibration_curve: CalibrationCurve
    n_samples: int


def _predict_heuristic(event: dict) -> float:
    d = float(event.get("distance_m", 18.0))
    a = float(event.get("angle_deg", 30.0))
    return compute_xg(
        distance_m=d,
        angle_deg=a,
        body_part=event.get("body_part", "right_foot"),
        assist_type=event.get("assist_type", "standard"),
        is_one_on_one=bool(event.get("is_one_on_one", False)),
        is_pressed=bool(event.get("was_pressed", False)),
        shot_type=event.get("shot_type", "open_play"),
    )


def _predict_enhanced(event: dict, model: EnhancedXgModel) -> float:
    return model.compute(event)


def _predict_dl(event: dict, model: DLXgModel) -> float:
    features = model.extract_features([event])
    if len(features) == 0:
        return 0.0
    return float(model.predict(features)[0])


def compare_xg_models(events: list[dict]) -> list[ModelComparisonReport]:
    shot_events = [e for e in events if e.get("type") == "shot" and "goal" in e or "is_goal" in e]
    if not shot_events:
        return []

    outcomes = [int(e.get("is_goal", e.get("goal", 0))) for e in shot_events]

    enhanced_model = EnhancedXgModel()
    dl_model = DLXgModel(seed=42)

    train_events = [e for i, e in enumerate(shot_events) if i % 3 != 0]
    test_events = [e for i, e in enumerate(shot_events) if i % 3 == 0]
    test_outcomes = [outcomes[i] for i in range(len(shot_events)) if i % 3 == 0]

    if len(train_events) >= 10:
        try:
            dl_features = dl_model.extract_features(train_events)
            dl_labels = np.array(
                [float(e.get("is_goal", e.get("goal", 0))) for e in train_events],
                dtype=np.float64,
            )
            dl_model.train(
                dl_features,
                dl_labels,
                epochs=20,
                batch_size=min(32, len(train_events)),
                verbose=False,
            )
        except Exception:
            pass

    predictions: dict[str, list[float]] = {}
    predictions["heuristic"] = [_predict_heuristic(e) for e in test_events]
    predictions["enhanced"] = [_predict_enhanced(e, enhanced_model) for e in test_events]
    predictions["dl_xg"] = [_predict_dl(e, dl_model) for e in test_events]

    reports: list[ModelComparisonReport] = []
    for name, preds in predictions.items():
        if not preds:
            continue
        curve = compute_calibration_curve(preds, test_outcomes)
        reports.append(
            ModelComparisonReport(
                model_name=name,
                brier_score=compute_brier_score(preds, test_outcomes),
                log_loss=compute_log_loss(preds, test_outcomes),
                auc_roc=compute_auc_roc(preds, test_outcomes),
                calibration_error=curve.ece,
                calibration_curve=curve,
                n_samples=len(preds),
            )
        )

    return reports


_PERMUTATION_CACHE: dict[str, float] = {}


def compute_feature_importance(events: list[dict]) -> dict[str, float]:
    shot_events = [e for e in events if e.get("type") == "shot"]
    if len(shot_events) < 5:
        return {}

    outcomes = np.array(
        [float(e.get("is_goal", e.get("goal", 0))) for e in shot_events], dtype=np.float64
    )

    model = EnhancedXgModel()
    baseline_preds = np.array([model.compute(e) for e in shot_events], dtype=np.float64)
    baseline_auc = compute_auc_roc(baseline_preds.tolist(), outcomes.astype(int).tolist())

    features = [
        ("distance_m", 18.0),
        ("angle_deg", 30.0),
        ("body_part", "right_foot"),
        ("is_one_on_one", False),
        ("was_pressed", False),
        ("shot_type", "open_play"),
        ("assist_type", "standard"),
    ]

    importances: dict[str, float] = {}
    rng = random.Random(42)
    for feat_name, _default_val in features:
        permuted = []
        for e in shot_events:
            e_copy = dict(e)
            if feat_name == "body_part":
                e_copy["body_part"] = (
                    "head" if e.get("body_part", "right_foot") == "right_foot" else "right_foot"
                )
            elif feat_name == "is_one_on_one":
                e_copy["is_one_on_one"] = not bool(e.get("is_one_on_one", False))
            elif feat_name == "was_pressed":
                e_copy["was_pressed"] = not bool(e.get("was_pressed", False))
            elif feat_name == "shot_type":
                e_copy["shot_type"] = (
                    "volley" if e.get("shot_type", "open_play") != "volley" else "open_play"
                )
            elif feat_name == "assist_type":
                e_copy["assist_type"] = (
                    "cross" if e.get("assist_type", "standard") != "cross" else "standard"
                )
            else:
                e_copy[feat_name] = (
                    rng.uniform(0, 105) if feat_name == "distance_m" else rng.uniform(0, 90)
                )
            permuted.append(e_copy)

        perm_preds = np.array([model.compute(e) for e in permuted], dtype=np.float64)
        perm_auc = compute_auc_roc(perm_preds.tolist(), outcomes.astype(int).tolist())
        drop = baseline_auc - perm_auc
        importances[feat_name] = round(max(0.0, drop), 4)

    return dict(sorted(importances.items(), key=lambda x: -x[1]))
