"""xGOT — Post-Shot Expected Goals Model with Calibration.

Estimates the probability a shot on target results in a goal,
using logistic regression with elastic net regularization,
calibrated via Platt scaling. Supports confidence intervals
via bootstrapping.
"""

from __future__ import annotations

import json
import math
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Default coefficients (heuristic fallback when sklearn unavailable or no training run)
DEFAULT_COEFS = {
    "intercept": -1.2,
    "distance_m": -0.06,
    "angle_deg": 0.015,
    "placement_dist_center": 2.0,
    "is_header": -0.5,
    "one_on_one": 0.4,
    "shot_speed_mps": 0.04,
    "defender_proximity": -0.15,
}


@dataclass
class XgotResult:
    xgot: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    calibrated: bool = False
    features: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "xgot": round(self.xgot, 4),
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
            "calibrated": self.calibrated,
        }


@dataclass
class XgotMatchReport:
    home_xgot: float = 0.0
    away_xgot: float = 0.0
    home_goals: int = 0
    away_goals: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "home_xgot": round(self.home_xgot, 3),
            "away_xgot": round(self.away_xgot, 3),
            "home_goals": self.home_goals,
            "away_goals": self.away_goals,
            "home_xgot_diff": round(self.home_goals - self.home_xgot, 3),
            "away_xgot_diff": round(self.away_goals - self.away_xgot, 3),
            "details": self.details,
        }


def _extract_features(
    distance_m: float,
    angle_deg: float,
    placement_x: float = 0.5,
    placement_y: float = 0.5,
    body_part: str = "right_foot",
    one_on_one: bool = False,
    shot_speed: float = 20.0,
    defender_distance: float = 5.0,
) -> dict[str, float]:
    corner_dist = math.sqrt((placement_x - 0.5) ** 2 + (placement_y - 0.5) ** 2)
    return {
        "distance_m": distance_m,
        "angle_deg": angle_deg,
        "placement_dist_center": corner_dist,
        "is_header": 1.0 if body_part == "head" else 0.0,
        "one_on_one": 1.0 if one_on_one else 0.0,
        "shot_speed_mps": shot_speed,
        "defender_proximity": max(0.0, 1.0 - defender_distance / 10.0),
    }


def _inv_logit(logit: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20, min(20, logit))))


class XgotModel:
    def __init__(self, coefs: dict[str, float] | None = None):
        self._coefs = coefs or dict(DEFAULT_COEFS)
        self._sk_model: Any = None
        self._calibrated = False
        self._bootstrap_coefs: list[dict[str, float]] = []
        self._is_trained = False

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    def compute(
        self,
        distance_m: float,
        angle_deg: float,
        placement_x: float = 0.5,
        placement_y: float = 0.5,
        body_part: str = "right_foot",
        one_on_one: bool = False,
        shot_speed: float = 20.0,
        defender_distance: float = 5.0,
        return_features: bool = False,
    ) -> XgotResult:
        features = _extract_features(
            distance_m,
            angle_deg,
            placement_x,
            placement_y,
            body_part,
            one_on_one,
            shot_speed,
            defender_distance,
        )
        if self._sk_model is not None and HAS_SKLEARN:
            X = np.array([[features[k] for k in sorted(features)]])
            proba = self._sk_model.predict_proba(X)[0, 1]
            xgot = float(proba)
        else:
            logit = self._coefs["intercept"]
            for k, v in features.items():
                logit += self._coefs.get(k, 0.0) * v
            xgot = _inv_logit(logit)

        xgot = max(0.005, min(0.995, xgot))

        ci_lower, ci_upper = self._bootstrap_ci(features, xgot)

        result = XgotResult(
            xgot=round(xgot, 4),
            ci_lower=round(ci_lower, 4),
            ci_upper=round(ci_upper, 4),
            calibrated=self._calibrated,
        )
        if return_features:
            result.features = features
        return result

    def _bootstrap_ci(
        self,
        features: dict[str, float],
        base_xgot: float,
        n_iter: int = 200,
        ci_level: float = 0.95,
    ) -> tuple[float, float]:
        if not self._bootstrap_coefs:
            spread = 0.05
            return max(0.001, base_xgot - spread), min(0.999, base_xgot + spread)
        xgots = []
        for coef_set in random.sample(
            self._bootstrap_coefs, min(n_iter, len(self._bootstrap_coefs))
        ):
            logit = coef_set.get("intercept", 0)
            for k, v in features.items():
                logit += coef_set.get(k, 0.0) * v
            xgots.append(_inv_logit(logit))
        if not xgots:
            return base_xgot, base_xgot
        xgots.sort()
        lower_idx = int((1 - ci_level) / 2 * len(xgots))
        upper_idx = int((1 + ci_level) / 2 * len(xgots))
        ci_low = xgots[lower_idx]
        ci_high = xgots[min(upper_idx, len(xgots) - 1)]
        # Ensure xgot is within bounds (bootstrap samples may be shifted)
        if base_xgot < ci_low or base_xgot > ci_high:
            midpoint = (ci_low + ci_high) / 2
            offset = base_xgot - midpoint
            ci_low = max(0.0, ci_low + offset)
            ci_high = min(1.0, ci_high + offset)
        return ci_low, ci_high

    def compute_match(self, events: list[dict[str, Any]]) -> XgotMatchReport:
        home_xgot = 0.0
        away_xgot = 0.0
        home_goals = 0
        away_goals = 0
        details = []

        for ev in events:
            if ev.get("type") not in ("shot", "shot_ontarget"):
                continue
            team = ev.get("team", "home")
            is_goal = ev.get("is_goal", False)

            result = self.compute(
                distance_m=ev.get("distance_m", 18.0),
                angle_deg=ev.get("angle_deg", 30.0),
                placement_x=ev.get("placement_x", 0.5),
                placement_y=ev.get("placement_y", 0.5),
                body_part=ev.get("body_part", "right_foot"),
                one_on_one=ev.get("one_on_one", False),
                shot_speed=ev.get("shot_speed", 20.0),
                defender_distance=ev.get("defender_distance", 5.0),
            )

            if team == "home":
                home_xgot += result.xgot
                if is_goal:
                    home_goals += 1
            else:
                away_xgot += result.xgot
                if is_goal:
                    away_goals += 1

            details.append(
                {
                    "timestamp": ev.get("timestamp", 0),
                    "team": team,
                    "xgot": result.xgot,
                    "ci_lower": result.ci_lower,
                    "ci_upper": result.ci_upper,
                    "is_goal": is_goal,
                }
            )

        return XgotMatchReport(
            home_xgot=home_xgot,
            away_xgot=away_xgot,
            home_goals=home_goals,
            away_goals=away_goals,
            details=details,
        )

    def train(
        self,
        shot_data: list[dict[str, Any]],
        calibrate: bool = True,
        bootstrap: bool = True,
    ) -> dict[str, Any]:
        if not HAS_SKLEARN or len(shot_data) < 10:
            return {
                "trained": False,
                "reason": "sklearn unavailable or too few samples",
                "samples": len(shot_data),
            }

        X_list = []
        y_list = []
        feature_keys = sorted(_extract_features(18, 30).keys())

        for shot in shot_data:
            feats = _extract_features(
                distance_m=shot.get("distance_m", 18),
                angle_deg=shot.get("angle_deg", 30),
                placement_x=shot.get("placement_x", 0.5),
                placement_y=shot.get("placement_y", 0.5),
                body_part=shot.get("body_part", "right_foot"),
                one_on_one=shot.get("one_on_one", False),
                shot_speed=shot.get("shot_speed", 20),
                defender_distance=shot.get("defender_distance", 5),
            )
            X_list.append([feats[k] for k in feature_keys])
            y_list.append(1 if shot.get("is_goal") else 0)

        X = np.array(X_list)
        y = np.array(y_list)

        base_model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=1.0,
            l1_ratio=0.5,
            max_iter=5000,
            random_state=42,
            class_weight="balanced",
        )
        base_model.fit(X, y)

        if calibrate:
            cal_model = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
            cal_model.fit(X, y)
            self._sk_model = cal_model
            self._calibrated = True
        else:
            self._sk_model = base_model
            self._calibrated = False

        self._coefs = {
            "intercept": float(base_model.intercept_[0]),
        }
        for i, k in enumerate(feature_keys):
            self._coefs[k] = float(base_model.coef_[0][i])
        self._is_trained = True

        if bootstrap and len(shot_data) >= 50:
            self._bootstrap_coefs = []
            for _ in range(500):
                idx = np.random.randint(0, len(X), len(X))
                Xb, yb = X[idx], y[idx]
                try:
                    bm = LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        C=1.0,
                        l1_ratio=0.5,
                        max_iter=2000,
                        random_state=None,
                    )
                    bm.fit(Xb, yb)
                    coefs_b = {"intercept": float(bm.intercept_[0])}
                    for i, k in enumerate(feature_keys):
                        coefs_b[k] = float(bm.coef_[0][i])
                    self._bootstrap_coefs.append(coefs_b)
                except Exception:
                    continue

        return {
            "trained": True,
            "calibrated": self._calibrated,
            "bootstrap_samples": len(self._bootstrap_coefs),
            "samples": len(shot_data),
            "features": feature_keys,
        }

    def save(self, path: str | Path) -> None:
        data = {
            "coefs": self._coefs,
            "calibrated": self._calibrated,
            "is_trained": self._is_trained,
            "bootstrap_coefs": self._bootstrap_coefs,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> XgotModel:
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(coefs=data.get("coefs"))
        model._calibrated = data.get("calibrated", False)
        model._is_trained = data.get("is_trained", False)
        model._bootstrap_coefs = data.get("bootstrap_coefs", [])
        return model

    def save_json(self, path: str | Path) -> None:
        data = {
            "coefs": self._coefs,
            "calibrated": self._calibrated,
            "is_trained": self._is_trained,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> XgotModel:
        with open(path) as f:
            data = json.load(f)
        return cls(coefs=data.get("coefs"))
