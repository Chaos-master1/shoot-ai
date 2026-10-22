"""Data-driven xG model trainer — numpy-only logistic regression.

Trains logistic regression coefficients from shot event data using
batch gradient descent. No scikit-learn/pandas/scipy dependencies.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from kawkab.core.events import ShotEvent
from kawkab.core.xg_model import ENHANCED_COEFFICIENTS

FEATURE_NAMES = [
    "intercept",
    "distance_m",
    "distance_m_sq",
    "angle_sin",
    "angle_deg_sq_sin",
    "is_header",
    "is_through_ball_assist",
    "is_cross_assist",
    "is_one_on_one",
    "is_pressed",
    "is_volley",
    "is_free_kick",
    "gk_distance_m",
    "gk_distance_m_sq",
    "is_rebound",
    "is_big_chance",
]


@dataclass
class FitShot:
    distance_m: float = 18.0
    angle_deg: float = 30.0
    is_header: bool = False
    is_through_ball_assist: bool = False
    is_cross_assist: bool = False
    is_one_on_one: bool = False
    is_pressed: bool = False
    is_volley: bool = False
    is_free_kick: bool = False
    gk_distance_m: float = 0.0
    is_rebound: bool = False
    is_big_chance: bool = False
    is_goal: bool = False


def _build_feature_matrix(shots: list[FitShot]) -> tuple[np.ndarray, np.ndarray]:
    n = len(shots)
    X = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    for i, s in enumerate(shots):
        d = max(s.distance_m, 0.5)
        angle_rad = math.radians(max(s.angle_deg, 0.0))
        gf = math.cos(angle_rad) if angle_rad < math.pi / 2 else 0.0
        X[i, 0] = 1.0
        X[i, 1] = d
        X[i, 2] = d * d
        X[i, 3] = 1.0 - gf
        X[i, 4] = (1.0 - gf) ** 2
        X[i, 5] = 1.0 if s.is_header else 0.0
        X[i, 6] = 1.0 if s.is_through_ball_assist else 0.0
        X[i, 7] = 1.0 if s.is_cross_assist else 0.0
        X[i, 8] = 1.0 if s.is_one_on_one else 0.0
        X[i, 9] = 1.0 if s.is_pressed else 0.0
        X[i, 10] = 1.0 if s.is_volley else 0.0
        X[i, 11] = 1.0 if s.is_free_kick else 0.0
        if s.gk_distance_m > 0:
            X[i, 12] = s.gk_distance_m
            X[i, 13] = s.gk_distance_m**2
        X[i, 14] = 1.0 if s.is_rebound else 0.0
        X[i, 15] = 1.0 if s.is_big_chance else 0.0
        y[i] = 1.0 if s.is_goal else 0.0
    return X, y


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _normalize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalize non-intercept features in-place."""
    mean = np.zeros(X.shape[1], dtype=np.float64)
    std = np.ones(X.shape[1], dtype=np.float64)
    if X.shape[0] > 1:
        for j in range(1, X.shape[1]):
            col = X[:, j]
            m = np.mean(col)
            s = np.std(col)
            if s > 1e-12:
                mean[j] = m
                std[j] = s
                X[:, j] = (col - m) / s
    return X, mean, std


def batch_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    epochs: int = 5000,
    l2: float = 0.001,
    verbose: bool = False,
) -> tuple[np.ndarray, list[float]]:
    n, m = X.shape
    X_norm, feat_mean, feat_std = _normalize(X.copy())
    theta = np.zeros(m, dtype=np.float64)
    loss_history: list[float] = []
    for epoch in range(epochs):
        z = X_norm @ theta
        h = _sigmoid(z)
        error = h - y
        gradient = (X_norm.T @ error) / n
        gradient[1:] += l2 * theta[1:] / n
        theta -= lr * gradient
        if epoch % 500 == 0 or epoch == epochs - 1:
            loss = -np.mean(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))
            loss += (l2 / (2 * n)) * np.sum(theta[1:] ** 2)
            loss_history.append(loss)
            if verbose:
                print(f"  epoch {epoch:5d} loss {loss:.6f}")
    # Denormalize coefficients for interpretability
    theta_orig = np.zeros(m, dtype=np.float64)
    theta_orig[0] = theta[0]
    for j in range(1, m):
        theta_orig[j] = theta[j] / feat_std[j] if feat_std[j] > 1e-12 else 0.0
        theta_orig[0] -= theta[j] * feat_mean[j] / feat_std[j] if feat_std[j] > 1e-12 else 0.0
    return theta_orig, loss_history


def fit_from_events(
    events: list[dict[str, Any]],
    model_name: str = "enhanced",
) -> dict[str, float]:
    shots: list[FitShot] = []
    for ev in events:
        if ev.get("type") != "shot":
            continue
        try:
            se = ShotEvent.from_dict(ev)
        except Exception:
            continue
        d = se.distance_m or 18.0
        a = se.angle_deg or 30.0
        body_part = se.body_part.value if se.body_part else "right_foot"
        shot_type = se.shot_type.value if se.shot_type else "open_play"
        shots.append(
            FitShot(
                distance_m=d,
                angle_deg=a,
                is_header=(body_part == "head"),
                is_through_ball_assist=bool(ev.get("assist_type") == "through_ball"),
                is_cross_assist=bool(ev.get("assist_type") == "cross"),
                is_one_on_one=se.is_one_on_one,
                is_pressed=se.was_pressed,
                is_volley=(shot_type in ("volley", "half_volley")),
                is_free_kick=(shot_type == "free_kick"),
                gk_distance_m=getattr(se, "gk_distance_m", 0.0) or ev.get("gk_distance_m", 0.0),
                is_rebound=ev.get("is_rebound", False),
                is_big_chance=ev.get("is_big_chance", False),
                is_goal=bool(ev.get("is_goal", False)),
            )
        )
    return fit_from_shots(shots, model_name)


def fit_from_shots(
    shots: list[FitShot],
    model_name: str = "enhanced",
) -> dict[str, float]:
    if len(shots) < 10:
        return dict(ENHANCED_COEFFICIENTS)
    X, y = _build_feature_matrix(shots)
    theta, _ = batch_gradient_descent(X, y)
    coeffs = dict(zip(FEATURE_NAMES, theta.tolist(), strict=False))
    coeffs["_model_name"] = model_name
    coeffs["_n_shots"] = len(shots)
    coeffs["_goal_rate"] = float(np.mean(y))
    return coeffs


def generate_synthetic_training_data(
    n_shots: int = 10000,
    seed: int = 42,
) -> list[FitShot]:
    rng = np.random.default_rng(seed)
    shots: list[FitShot] = []
    for _ in range(n_shots):
        d = rng.uniform(1.0, 35.0)
        a = rng.uniform(0.0, 90.0)
        is_header = rng.random() < 0.08
        is_pressed = rng.random() < 0.30
        is_volley = rng.random() < 0.10
        is_free_kick = rng.random() < 0.05
        is_one_on_one = rng.random() < 0.05
        gk_dist = rng.uniform(0.0, 10.0) if rng.random() < 0.4 else 0.0
        is_rebound = rng.random() < 0.03
        is_big_chance = rng.random() < 0.08
        is_through_ball = rng.random() < 0.06
        is_cross = rng.random() < 0.12
        true_proba = _synthetic_proba(
            d,
            a,
            is_header,
            is_pressed,
            is_volley,
            is_free_kick,
            is_one_on_one,
            gk_dist,
            is_rebound,
            is_big_chance,
            is_through_ball,
            is_cross,
        )
        is_goal = rng.random() < true_proba
        shots.append(
            FitShot(
                distance_m=d,
                angle_deg=a,
                is_header=is_header,
                is_pressed=is_pressed,
                is_volley=is_volley,
                is_free_kick=is_free_kick,
                is_one_on_one=is_one_on_one,
                gk_distance_m=gk_dist,
                is_rebound=is_rebound,
                is_big_chance=is_big_chance,
                is_through_ball_assist=is_through_ball,
                is_cross_assist=is_cross,
                is_goal=is_goal,
            )
        )
    return shots


def _synthetic_proba(
    d,
    a,
    is_header,
    is_pressed,
    is_volley,
    is_free_kick,
    is_one_on_one,
    gk_dist,
    is_rebound,
    is_big_chance,
    is_through_ball,
    is_cross,
) -> float:
    logit = -1.5
    logit += -0.10 * max(d, 0.5)
    logit += -0.0005 * (max(d, 0.5) ** 2)
    angle_rad = math.radians(max(a, 0.0))
    gf = math.cos(angle_rad) if angle_rad < math.pi / 2 else 0.0
    logit += 1.0 * (1.0 - gf)
    logit += -0.2 * ((1.0 - gf) ** 2)
    if is_header:
        logit += -0.6
    if is_pressed:
        logit += -0.3
    if is_volley:
        logit += 0.15
    if is_free_kick:
        logit += 0.1
    if is_one_on_one:
        logit += 0.5
    if gk_dist > 0:
        logit += -0.06 * gk_dist
        logit += -0.0003 * (gk_dist**2)
    if is_rebound:
        logit += 0.4
    if is_big_chance:
        logit += 0.6
    if is_through_ball:
        logit += 0.3
    if is_cross:
        logit += -0.15
    return 1.0 / (1.0 + math.exp(-min(logit, 20.0)))


def save_coefficients(coeffs: dict[str, float], path: str) -> None:
    serializable = {k: v for k, v in coeffs.items() if isinstance(v, (int, float, str))}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_coefficients(path: str) -> dict[str, float]:
    with open(path) as f:
        return json.load(f)
