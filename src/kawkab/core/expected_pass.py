"""Expected Pass Completion (EP) probability model using logistic regression.

Features:
  - Pass distance (m) and distance² (non-linear decay)
  - Pass angle relative to goal direction (forward = harder)
  - Through ball, cross, long ball flags
  - Pressured flag (defender within 2m of passer)
  - Headed pass flag
  - Receiver under pressure flag
  - Pass start zone (x-coordinate on pitch, defensive third = easier)
  - Body part used (foot vs head)
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ── Coefficients ─────────────────────────────────────────────────────────────

EP_COEFFICIENTS: dict[str, float] = {
    "intercept": 1.5,
    "distance_m": -0.08,
    "distance_m_sq": -0.001,
    "is_through_ball": -0.8,
    "is_cross": -0.6,
    "is_long_ball": -0.5,
    "is_pressured": -0.7,
    "is_headed": -0.4,
    "receiver_pressured": -0.3,
    "start_x_norm": 0.15,
    "angle_deg": -0.005,
}

# ── Result dataclass ─────────────────────────────────────────────────────────


@dataclass
class ExpectedPassResult:
    """Result of an expected pass computation.

    Attributes:
        ep: Predicted pass completion probability (0-1).
        is_progressive: Whether the pass moves the ball significantly forward.
        difficulty: Qualitative label — "easy", "moderate", "difficult",
            or "very_difficult".
        factors: Per-feature contribution to the log-odds before sigmoid.
    """

    ep: float
    is_progressive: bool
    difficulty: str
    factors: dict[str, float] = field(default_factory=dict)


# ── Difficulty classification ────────────────────────────────────────────────


def _classify_difficulty(ep: float) -> str:
    """Classify pass difficulty from EP value."""
    if ep >= 0.85:
        return "easy"
    if ep >= 0.65:
        return "moderate"
    if ep >= 0.40:
        return "difficult"
    return "very_difficult"


# ── Progressive pass detection ───────────────────────────────────────────────


def _is_progressive(
    start_x: float,
    end_x: float,
    pass_distance: float,
    attacking_direction: int = 1,
) -> bool:
    """Determine if a pass is progressive.

    A pass is progressive when it moves the ball significantly toward the
    opponent's goal (at least 25 % of the pitch length or 20 m forward).

    Args:
        start_x: Start x-coordinate (0 = own goal line, 105 = opponent goal).
        end_x: End x-coordinate.
        pass_distance: Euclidean distance of the pass in metres.
        attacking_direction: 1 = left-to-right attacking, -1 = right-to-left.

    Returns:
        True if the pass is progressive.
    """
    forward_progress = (end_x - start_x) * attacking_direction
    if forward_progress <= 0:
        return False
    return forward_progress >= 0.25 * 105.0 or pass_distance >= 20.0


# ── Feature extraction ───────────────────────────────────────────────────────


def _feature_vector(pass_data: dict[str, Any]) -> np.ndarray:
    """Extract and normalise features into a numpy feature vector.

    Expected keys (all optional with sensible defaults):
        distance_m, angle_deg, is_through_ball, is_cross, is_long_ball,
        is_pressured, is_headed, receiver_pressured, start_x

    Args:
        pass_data: Dictionary of pass attributes.

    Returns:
        1-D numpy array of 11 features matching EP_COEFFICIENTS order.
    """
    distance_m = float(pass_data.get("distance_m", 15.0))
    distance_m_sq = distance_m * distance_m
    angle_deg = float(pass_data.get("angle_deg", 0.0))
    is_through_ball = 1.0 if pass_data.get("is_through_ball", False) else 0.0
    is_cross = 1.0 if pass_data.get("is_cross", False) else 0.0
    is_long_ball = 1.0 if pass_data.get("is_long_ball", False) else 0.0
    is_pressured = 1.0 if pass_data.get("is_pressured", False) else 0.0
    is_headed = 1.0 if pass_data.get("is_headed", False) else 0.0
    receiver_pressured = 1.0 if pass_data.get("receiver_pressured", False) else 0.0

    # Normalise start_x to [0, 1] assuming pitch length 105 m.
    start_x = float(pass_data.get("start_x", 52.5))
    start_x_norm = max(0.0, min(1.0, start_x / 105.0))

    return np.array(
        [
            1.0,  # intercept
            distance_m,
            distance_m_sq,
            is_through_ball,
            is_cross,
            is_long_ball,
            is_pressured,
            is_headed,
            receiver_pressured,
            start_x_norm,
            angle_deg,
        ],
        dtype=np.float64,
    )


_FEATURE_NAMES: list[str] = [
    "intercept",
    "distance_m",
    "distance_m_sq",
    "is_through_ball",
    "is_cross",
    "is_long_ball",
    "is_pressured",
    "is_headed",
    "receiver_pressured",
    "start_x_norm",
    "angle_deg",
]


def _coeff_array() -> np.ndarray:
    """Return coefficients as a 1-D numpy array in feature order."""
    c = EP_COEFFICIENTS
    return np.array(
        [c[k] for k in _FEATURE_NAMES],
        dtype=np.float64,
    )


# ── Single-pass computation ──────────────────────────────────────────────────


def compute_ep(pass_data: dict) -> ExpectedPassResult:
    """Compute expected pass completion probability for a single pass.

    Args:
        pass_data: Dictionary of pass attributes.

    Returns:
        ExpectedPassResult with the EP probability, progressive flag,
        difficulty label, and per-factor contributions.
    """
    return _compute_ep_cached(tuple(sorted(pass_data.items())))


@functools.lru_cache(maxsize=128)
def _compute_ep_cached(pass_data_items: tuple) -> ExpectedPassResult:
    pd = dict(pass_data_items)
    coeffs = _coeff_array()
    features = _feature_vector(pd)

    logit = float(np.dot(coeffs, features))
    if not math.isfinite(logit):
        return ExpectedPassResult(ep=0.0, is_progressive=False, difficulty="unknown", factors={})
    ep = 1.0 / (1.0 + math.exp(-min(logit, 20.0)))

    factors = {name: float(coeffs[i] * features[i]) for i, name in enumerate(_FEATURE_NAMES)}

    distance_m = float(pd.get("distance_m", 15.0))
    start_x = float(pd.get("start_x", 52.5))
    end_x = float(pd.get("end_x", 52.5))
    attacking_direction = int(pd.get("attacking_direction", 1))
    progressive = _is_progressive(start_x, end_x, distance_m, attacking_direction)
    difficulty = _classify_difficulty(ep)

    return ExpectedPassResult(
        ep=round(ep, 4),
        is_progressive=progressive,
        difficulty=difficulty,
        factors=factors,
    )


# ── Batch computation (vectorised) ───────────────────────────────────────────


def compute_ep_batch(passes: list[dict[str, Any]]) -> list[ExpectedPassResult]:
    """Compute EP for multiple passes using vectorised numpy operations.

    Args:
        passes: List of pass-data dictionaries.

    Returns:
        List of ExpectedPassResult objects in the same order as the input.
    """
    if not passes:
        return []

    coeffs = _coeff_array()

    n = len(passes)
    features_list = [_feature_vector(p) for p in passes]
    feature_matrix = np.array(features_list, dtype=np.float64)  # (n, 11)

    logits = feature_matrix @ coeffs  # (n,)
    eps = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))

    results: list[ExpectedPassResult] = []
    for i in range(n):
        p = passes[i]
        distance_m = float(p.get("distance_m", 15.0))
        start_x = float(p.get("start_x", 52.5))
        end_x = float(p.get("end_x", 52.5))
        attacking_direction = int(p.get("attacking_direction", 1))
        progressive = _is_progressive(start_x, end_x, distance_m, attacking_direction)
        difficulty = _classify_difficulty(float(eps[i]))

        factors = {
            name: float(coeffs[j] * feature_matrix[i, j]) for j, name in enumerate(_FEATURE_NAMES)
        }

        results.append(
            ExpectedPassResult(
                ep=round(float(eps[i]), 4),
                is_progressive=progressive,
                difficulty=difficulty,
                factors=factors,
            )
        )

    return results
