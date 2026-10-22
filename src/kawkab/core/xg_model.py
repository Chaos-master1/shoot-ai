"""Expected Goals (xG) model using logistic regression with enhanced features.

Features:
  - Distance and distance² (non-linear distance decay)
  - Angle sin (visible goal fraction)
  - Body part (head, foot)
  - Assist type (through ball, cross, standard)
  - Shot type (open play, volley, free kick, penalty)
  - One-on-one flag
  - Pressure flag
  - Goalkeeper distance (how far GK is from shot)
  - Shot placement (angle to goal center)
  - Rebound flag (shot following a save)
  - Big chance flag (clear-cut opportunity)

Legacy coefficients and the original compute_xg() function are preserved
for backward compatibility. The EnhancedXgModel provides the improved model.
"""

from __future__ import annotations

import functools
import json
import logging
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.stats import beta

logger = logging.getLogger(__name__)

from kawkab.core.coordinate_validator import CoordinateValidator
from kawkab.core.events import (
    ShotEvent,
    event_from_dict,
)
from kawkab.core.perf_timing import timed

# ── Named constants for magic numbers ────────────────────────────────────────

# Shot angle thresholds (degrees)
ANGLE_CENTRAL_DEG = 30.0
ANGLE_SIDE_THRESHOLD_DEG = 35.0

# Penalty kick fixed xG
PENALTY_XG = 0.76

# Logit clipping to prevent exp overflow
LOGIT_CLIP_MIN = -20.0
LOGIT_CLIP_MAX = 20.0

# Distance clamping minimum
MIN_DISTANCE = 0.5

# Default values when event fields are missing
DEFAULT_DISTANCE_M = 18.0

# ── Legacy coefficients (unchanged for backward compat) ─────────────────────

XG_COEFFICIENTS: dict[str, float] = {
    "intercept": -0.5,
    "distance_m": -0.09,
    "distance_m_sq": -0.0005,
    "angle_deg_sin": -0.8,
    "is_header": -0.9,
    "is_through_ball_assist": 0.35,
    "is_cross_assist": -0.35,
    "is_one_on_one": 0.55,
    "is_pressed": -0.25,
    "is_volley": 0.15,
    "is_free_kick": 0.08,
    "is_far_side": -0.10,
    "is_penalty": 2.0,
}

# ── Enhanced model coefficients (calibrated to StatsBomb-like distribution) ─

ENHANCED_COEFFICIENTS: dict[str, float] = {
    "intercept": -1.2,
    "distance_m": -0.12,
    "distance_m_sq": -0.0003,
    "angle_sin": -1.4,
    "is_header": -0.7,
    "is_through_ball_assist": 0.4,
    "is_cross_assist": -0.2,
    "is_one_on_one": 0.6,
    "is_pressed": -0.3,
    "is_volley": 0.2,
    "is_free_kick": 0.15,
    "is_penalty": 2.0,
    "gk_distance_m": -0.08,
    "gk_distance_m_sq": -0.0004,
    "is_rebound": 0.5,
    "is_big_chance": 0.7,
    "angle_deg_sq_sin": -0.3,
}


# ── Trained coefficients (auto-loaded from disk, fallback to enhanced) ──────

TRAINED_COEFFICIENTS: dict[str, float] = dict(ENHANCED_COEFFICIENTS)
_TRAINED_COEFF_PATH = Path(__file__).parent / "trained_xg_coefficients.json"
_TRAINED_LOADED_FROM_DISK = False
if _TRAINED_COEFF_PATH.exists():
    try:
        with open(_TRAINED_COEFF_PATH) as _f:
            _trained = json.load(_f)
        _trained_clean = {
            k: v
            for k, v in _trained.items()
            if isinstance(v, (int, float)) and not k.startswith("_")
        }
        if _trained_clean:
            TRAINED_COEFFICIENTS.update(_trained_clean)
            _TRAINED_LOADED_FROM_DISK = True
    except Exception:
        pass


def trained_model_available() -> bool:
    """True when fitted weights (not the hand-tuned heuristic) are active."""
    return _TRAINED_LOADED_FROM_DISK


def _validate_trained_coefficients() -> None:
    """Check TRAINED_COEFFICIENTS has all ENHANCED_COEFFICIENTS keys; warn on mismatch."""
    missing = set(ENHANCED_COEFFICIENTS) - set(TRAINED_COEFFICIENTS)
    extra = set(TRAINED_COEFFICIENTS) - set(ENHANCED_COEFFICIENTS)
    if missing:
        logger.warning("TRAINED_COEFFICIENTS missing keys: %s", missing)
    if extra:
        logger.warning(
            "TRAINED_COEFFICIENTS has extra keys not in ENHANCED_COEFFICIENTS: %s", extra
        )


_validate_trained_coefficients()

# ── Legacy functions (backward compatible) ──────────────────────────────────


@functools.lru_cache(maxsize=64)
@timed()
def compute_xg(
    distance_m: float,
    angle_deg: float,
    *,
    body_part: str = "right_foot",
    assist_type: str = "standard",
    is_one_on_one: bool = False,
    is_pressed: bool = False,
    shot_type: str = "open_play",
    from_side: bool = False,
) -> float:
    """Compute expected goals using the legacy model."""
    if shot_type == "penalty":
        return PENALTY_XG
    coef = XG_COEFFICIENTS
    logit = coef["intercept"]
    d = max(distance_m, MIN_DISTANCE)
    logit += coef["distance_m"] * d
    logit += coef["distance_m_sq"] * (d * d)
    angle_rad = math.radians(max(angle_deg, 0.0))
    goal_fraction = math.cos(angle_rad) if angle_rad < math.pi / 2 else 0.0
    logit += coef["angle_deg_sin"] * (1.0 - goal_fraction)

    if body_part == "head":
        logit += coef["is_header"]
    if assist_type == "through_ball":
        logit += coef["is_through_ball_assist"]
    elif assist_type == "cross":
        logit += coef["is_cross_assist"]
    if is_one_on_one:
        logit += coef["is_one_on_one"]
    if is_pressed:
        logit += coef["is_pressed"]
    if shot_type in ("volley", "half_volley"):
        logit += coef["is_volley"]
    if shot_type == "free_kick":
        logit += coef["is_free_kick"]
    if from_side:
        logit += coef["is_far_side"]

    return 1.0 / (1.0 + math.exp(-logit))


def compute_xg_from_shot_event(event: ShotEvent) -> float:
    """Compute xG from a ShotEvent object using the legacy model."""
    angle_deg = event.angle_deg or ANGLE_CENTRAL_DEG
    distance_m = event.distance_m or DEFAULT_DISTANCE_M
    body_part = event.body_part.value if event.body_part else "right_foot"
    shot_type = event.shot_type.value if event.shot_type else "open_play"
    from_side = angle_deg > ANGLE_SIDE_THRESHOLD_DEG

    return compute_xg(
        distance_m=distance_m,
        angle_deg=angle_deg,
        body_part=body_part,
        is_one_on_one=event.is_one_on_one,
        is_pressed=event.was_pressed,
        shot_type=shot_type,
        from_side=from_side,
    )


def compute_xg_trained_from_shot_event(
    event: ShotEvent,
    *,
    gk_distance_m: float | None = None,
) -> float:
    """xG from a typed ShotEvent using the ACTIVE (trained) model.

    This is the trained-model replacement for ``compute_xg_from_shot_event``
    (which runs the legacy heuristic and is kept only for backward
    compatibility). The live CV pipeline's ``analyze_match`` path must use
    this one so trained weights actually reach video-derived shots.

    Convention notes — read before touching:
    - ``ShotEvent.angle_deg`` from the CV pipeline is in the
      DEVIATION-from-central convention (0° = central), matching both the
      trained coefficients (negative ``angle_sin`` on ``1 − cos(angle)``)
      and the pipeline's own ``angle_to_goal_deg`` metadata. StatsBomb
      imports store OPENING angle in metadata ``angle_deg`` and are
      converted by ``statsbomb_import_service`` before storage — do not
      feed opening-angle values here unconverted (the 2026-09-16
      wrong-angle-convention bug taught the model that wide shots score
      more; the sanity tests in tests/unit/test_xg_model.py pin this).
    - ``gk_distance_m``: distance from the shot origin to the nearest
      goalkeeper track, in meters. None/0 = feature absent (the trained
      model skips the GK terms rather than treating 0 as a real value).
    """
    features = active_xg_model().extract_features(event)
    if gk_distance_m is not None:
        features.gk_distance_m = max(float(gk_distance_m), 0.0)
    return active_xg_model().compute_single(features)


def compute_xg_from_dict(event_dict: dict[str, Any]) -> float:
    """Compute xG from a raw event dict (legacy)."""
    CoordinateValidator.validate_event_spatial(event_dict)
    event = ShotEvent.from_dict(event_dict)
    return compute_xg_from_shot_event(event)


# ── Trained-model access (the path live code should use) ────────────────────

_ACTIVE_MODEL: EnhancedXgModel | None = None


def active_xg_model() -> EnhancedXgModel:
    """Module-level singleton exposing the best available xG model.

    Returns the trained model (auto-loaded from
    ``trained_xg_coefficients.json`` when present), falling back to the
    enhanced heuristic. Check ``.coeffs_source`` for provenance.
    """
    global _ACTIVE_MODEL
    if _ACTIVE_MODEL is None:
        _ACTIVE_MODEL = EnhancedXgModel()
    return _ACTIVE_MODEL


def compute_xg_trained_from_dict(event_dict: dict[str, Any]) -> float:
    """xG from a raw event dict using the ACTIVE (trained when available) model.

    Same contract as ``compute_xg_from_dict`` but never the legacy model —
    the legacy function is kept only for backward compatibility; new call
    sites must use this one so trained weights actually reach the product.
    """
    model = active_xg_model()
    raw: dict[str, Any] = dict(event_dict)
    if raw.get("type") is None:
        raw["type"] = "shot"
    return model.compute(raw)


def batch_compute_xg(
    events: list[ShotEvent | dict[str, Any]],
) -> list[float]:
    """Compute xG for multiple events using vectorized numpy (legacy)."""
    results: list[float] = []
    shot_events: list[ShotEvent] = []
    for ev in events:
        if isinstance(ev, ShotEvent):
            shot_events.append(ev)
        elif isinstance(ev, dict):
            if ev.get("type") == "shot":
                try:
                    shot_events.append(cast(ShotEvent, event_from_dict(ev)))
                except Exception:
                    results.append(0.0)
                    continue
        else:
            results.append(0.0)
    if not shot_events:
        return results + [0.0] * (len(events) - len(results))

    n_shots = len(shot_events)
    distances = np.fromiter(
        (max(s.distance_m or DEFAULT_DISTANCE_M, MIN_DISTANCE) for s in shot_events),
        dtype=np.float64,
        count=n_shots,
    )
    angles = np.fromiter(
        (s.angle_deg or ANGLE_CENTRAL_DEG for s in shot_events), dtype=np.float64, count=n_shots
    )
    is_header = np.fromiter(
        ((s.body_part is not None and s.body_part.value == "head") for s in shot_events),
        dtype=np.float64,
        count=n_shots,
    )
    is_one_on_one = np.fromiter(
        (getattr(s, "is_one_on_one", False) for s in shot_events), dtype=np.float64, count=n_shots
    )
    was_pressed = np.fromiter(
        (getattr(s, "was_pressed", False) for s in shot_events), dtype=np.float64, count=n_shots
    )

    shot_types = [s.shot_type.value if s.shot_type else "open_play" for s in shot_events]
    is_volley = np.fromiter(
        (t in ("volley", "half_volley") for t in shot_types), dtype=np.float64, count=n_shots
    )
    is_free_kick = np.fromiter(
        (t == "free_kick" for t in shot_types), dtype=np.float64, count=n_shots
    )
    is_penalty = np.fromiter((t == "penalty" for t in shot_types), dtype=np.float64, count=n_shots)

    penalty_mask = is_penalty.astype(bool)
    batch_size = len(distances)
    xg_values = np.zeros(batch_size, dtype=np.float64)

    non_penalty = ~penalty_mask
    if np.any(non_penalty):
        coef = XG_COEFFICIENTS
        d = distances[non_penalty]
        a = angles[non_penalty]
        logit = np.full(int(np.sum(non_penalty)), coef["intercept"], dtype=np.float64)
        logit += coef["distance_m"] * d
        logit += coef["distance_m_sq"] * (d * d)
        angle_rad = np.radians(np.maximum(a, 0.0))
        goal_fraction = np.where(angle_rad < np.pi / 2, np.cos(angle_rad), 0.0)
        logit += coef["angle_deg_sin"] * (1.0 - goal_fraction)
        logit += coef["is_header"] * is_header[non_penalty]
        logit += coef["is_one_on_one"] * is_one_on_one[non_penalty]
        logit += coef["is_pressed"] * was_pressed[non_penalty]
        logit += coef["is_volley"] * is_volley[non_penalty]
        logit += coef["is_free_kick"] * is_free_kick[non_penalty]
        logit = np.clip(logit, LOGIT_CLIP_MIN, LOGIT_CLIP_MAX)
        logit = np.where(np.isfinite(logit), logit, 0.0)
        xg_values[non_penalty] = 1.0 / (1.0 + np.exp(-logit))

    xg_values[penalty_mask] = PENALTY_XG

    return results + xg_values.tolist()


# ── Enhanced Model ──────────────────────────────────────────────────────────


@dataclass
class EnhancedXgFeatures:
    """Feature vector for the enhanced xG model."""

    distance_m: float = DEFAULT_DISTANCE_M
    angle_deg: float = ANGLE_CENTRAL_DEG
    is_header: bool = False
    is_through_ball_assist: bool = False
    is_cross_assist: bool = False
    is_one_on_one: bool = False
    is_pressed: bool = False
    is_volley: bool = False
    is_free_kick: bool = False
    is_penalty: bool = False
    gk_distance_m: float = 0.0
    is_rebound: bool = False
    is_big_chance: bool = False

    def __hash__(self):
        return hash(tuple(getattr(self, f.name) for f in fields(self)))

    def __eq__(self, other):
        if not isinstance(other, EnhancedXgFeatures):
            return NotImplemented
        return all(getattr(self, f.name) == getattr(other, f.name) for f in fields(self))


class EnhancedXgModel:
    """Enhanced xG model with additional features.

    Uses logistic regression with more sophisticated feature engineering:
    goalkeeper distance, rebound detection, big chance flag, angle².

    Args:
        coefficients: Dict of coefficient values. Defaults to ENHANCED_COEFFICIENTS.
        coeffs_source: Label for coefficient provenance ("heuristic" or path).
    """

    def __init__(
        self, coefficients: dict[str, float] | None = None, coeffs_source: str = "heuristic"
    ):
        if coefficients is not None:
            self.coef = coefficients
        else:
            self.coef = TRAINED_COEFFICIENTS
            # Provenance: "heuristic" is only honest when no fitted
            # weights were found on disk. The old label always said
            # "heuristic" even when trained coefficients had auto-loaded.
            self.coeffs_source = (
                "trained (trained_xg_coefficients.json)"
                if _TRAINED_LOADED_FROM_DISK
                else "heuristic"
            )
            return
        self.coeffs_source = coeffs_source

    @classmethod
    def load_trained(cls, path: str) -> EnhancedXgModel:
        from kawkab.core.xg_trainer import load_coefficients

        coeffs = load_coefficients(path)
        return cls(coefficients=coeffs, coeffs_source=path)

    def extract_features(self, event: ShotEvent | dict[str, Any]) -> EnhancedXgFeatures:
        """Extract feature vector from a shot event."""
        # Dict-only extras must be read BEFORE the ShotEvent conversion
        # below replaces `event` -- the old code's post-conversion
        # `isinstance(event, dict)` rebound check was unreachable dead
        # code, so dict-supplied gk_distance_m / is_rebound /
        # is_big_chance were silently dropped (gk_distance is the
        # strongest feature in the trained model; dropping it made every
        # trained-model evaluation run ~0.3 logits hot).
        raw: dict[str, Any] = event if isinstance(event, dict) else {}
        gk_distance_raw = raw.get("gk_distance_m")
        is_rebound_raw = raw.get("is_rebound", False)
        is_big_chance_raw = raw.get("is_big_chance", False)
        assist_raw = raw.get("assist_type", "")

        if isinstance(event, dict):
            event = ShotEvent.from_dict(event)

        angle_deg = event.angle_deg or ANGLE_CENTRAL_DEG
        distance_m = event.distance_m or DEFAULT_DISTANCE_M
        body_part = event.body_part.value if event.body_part else "right_foot"
        shot_type = event.shot_type.value if event.shot_type else "open_play"

        # gk_distance priority: explicit dict value > event field (None
        # when absent -- ShotEvent has no gk_distance_m attribute) > 0.
        gk_distance = (
            float(gk_distance_raw)
            if gk_distance_raw is not None
            else float(getattr(event, "gk_distance_m", 0.0) or 0.0)
        )

        features = EnhancedXgFeatures(
            distance_m=distance_m,
            angle_deg=angle_deg,
            is_header=(body_part == "head"),
            is_through_ball_assist=(assist_raw == "through_ball"),
            is_cross_assist=(assist_raw == "cross"),
            is_one_on_one=event.is_one_on_one,
            is_pressed=event.was_pressed,
            is_volley=(shot_type in ("volley", "half_volley")),
            is_free_kick=(shot_type == "free_kick"),
            is_penalty=(shot_type == "penalty"),
            gk_distance_m=gk_distance,
        )

        # Rebound/big-chance: from the raw dict when available, else any
        # attribute the event object carries.
        if raw:
            features.is_rebound = bool(is_rebound_raw)
            features.is_big_chance = bool(is_big_chance_raw)
        else:
            features.is_rebound = bool(getattr(event, "is_rebound", False))
            features.is_big_chance = bool(getattr(event, "is_big_chance", False))

        return features

    def compute_single(self, features: EnhancedXgFeatures) -> float:
        """Compute xG for a single feature vector.

        Results are memoized per model instance. A shared class-level
        lru_cache here keyed on `features` alone leaked results across
        model instances -- two EnhancedXgModels with different
        coefficients could serve each other's cached xG values.
        """
        cache = self.__dict__.setdefault("_compute_single_cache", {})
        try:
            return cache[features]
        except KeyError:
            pass
        value = self._compute_single_uncached(features)
        if len(cache) >= 128:
            cache.clear()
        cache[features] = value
        return value

    def _compute_single_uncached(self, features: EnhancedXgFeatures) -> float:
        if features.is_penalty:
            return PENALTY_XG

        c = self.coef
        logit = c["intercept"]
        d = max(features.distance_m, MIN_DISTANCE)

        logit += c["distance_m"] * d
        logit += c["distance_m_sq"] * (d * d)

        angle_rad = math.radians(max(features.angle_deg, 0.0))
        goal_fraction = math.cos(angle_rad) if angle_rad < math.pi / 2 else 0.0
        logit += c["angle_sin"] * (1.0 - goal_fraction)
        logit += c["angle_deg_sq_sin"] * ((1.0 - goal_fraction) ** 2)

        if features.is_header:
            logit += c["is_header"]
        if features.is_through_ball_assist:
            logit += c["is_through_ball_assist"]
        if features.is_cross_assist:
            logit += c["is_cross_assist"]
        if features.is_one_on_one:
            logit += c["is_one_on_one"]
        if features.is_pressed:
            logit += c["is_pressed"]
        if features.is_volley:
            logit += c["is_volley"]
        if features.is_free_kick:
            logit += c["is_free_kick"]

        if features.gk_distance_m > 0:
            logit += c["gk_distance_m"] * features.gk_distance_m
            logit += c["gk_distance_m_sq"] * (features.gk_distance_m**2)

        if features.is_rebound:
            logit += c["is_rebound"]
        if features.is_big_chance:
            logit += c["is_big_chance"]

        if not math.isfinite(logit):
            return 0.0
        return 1.0 / (1.0 + math.exp(-min(logit, LOGIT_CLIP_MAX)))

    def compute(self, event: ShotEvent | dict[str, Any]) -> float:
        """Compute xG for a single shot event (enhanced model)."""
        features = self.extract_features(event)
        return self.compute_single(features)

    def batch_compute(self, events: list[ShotEvent | dict[str, Any]]) -> list[float]:
        """Compute xG for multiple events using numpy vectorization."""
        results: list[float] = []
        shot_events: list[ShotEvent] = []

        for ev in events:
            if isinstance(ev, ShotEvent):
                shot_events.append(ev)
            elif isinstance(ev, dict):
                if ev.get("type") == "shot":
                    try:
                        shot_events.append(ShotEvent.from_dict(ev))
                    except Exception:
                        results.append(0.0)
                        continue
            else:
                results.append(0.0)

        if not shot_events:
            return results + [0.0] * (len(events) - len(results))

        n = len(shot_events)
        distances = np.fromiter(
            (max(s.distance_m or DEFAULT_DISTANCE_M, MIN_DISTANCE) for s in shot_events),
            dtype=np.float64,
            count=n,
        )
        angles = np.fromiter(
            (s.angle_deg or ANGLE_CENTRAL_DEG for s in shot_events), dtype=np.float64, count=n
        )
        is_header = np.fromiter(
            ((s.body_part is not None and s.body_part.value == "head") for s in shot_events),
            dtype=np.float64,
            count=n,
        )
        is_one_on_one = np.fromiter(
            (getattr(s, "is_one_on_one", False) for s in shot_events), dtype=np.float64, count=n
        )
        was_pressed = np.fromiter(
            (getattr(s, "was_pressed", False) for s in shot_events), dtype=np.float64, count=n
        )
        shot_types = [s.shot_type.value if s.shot_type else "open_play" for s in shot_events]
        is_volley = np.fromiter(
            (t in ("volley", "half_volley") for t in shot_types), dtype=np.float64, count=n
        )
        is_free_kick = np.fromiter(
            (t == "free_kick" for t in shot_types), dtype=np.float64, count=n
        )
        is_penalty = np.fromiter((t == "penalty" for t in shot_types), dtype=np.float64, count=n)

        gk_dist = np.fromiter(
            (getattr(s, "gk_distance_m", 0.0) for s in shot_events), dtype=np.float64, count=n
        )
        is_rebound = np.fromiter(
            (getattr(s, "is_rebound", False) for s in shot_events), dtype=np.float64, count=n
        )
        is_big_chance = np.fromiter(
            (getattr(s, "is_big_chance", False) for s in shot_events), dtype=np.float64, count=n
        )

        c = self.coef
        xg_values = np.zeros(n, dtype=np.float64)
        penalty_mask = is_penalty.astype(bool)
        non_penalty = ~penalty_mask

        if np.any(non_penalty):
            d = distances[non_penalty]
            a = angles[non_penalty]
            logit = np.full(int(np.sum(non_penalty)), c["intercept"], dtype=np.float64)
            logit += c["distance_m"] * d
            logit += c["distance_m_sq"] * (d * d)
            angle_rad = np.radians(np.maximum(a, 0.0))
            gf = np.where(angle_rad < np.pi / 2, np.cos(angle_rad), 0.0)
            logit += c["angle_sin"] * (1.0 - gf)
            logit += c["angle_deg_sq_sin"] * ((1.0 - gf) ** 2)
            logit += c["is_header"] * is_header[non_penalty]
            logit += c["is_one_on_one"] * is_one_on_one[non_penalty]
            logit += c["is_pressed"] * was_pressed[non_penalty]
            logit += c["is_volley"] * is_volley[non_penalty]
            logit += c["is_free_kick"] * is_free_kick[non_penalty]

            gkd = gk_dist[non_penalty]
            gk_mask = gkd > 0
            if np.any(gk_mask):
                logit[gk_mask] += c["gk_distance_m"] * gkd[gk_mask]
                logit[gk_mask] += c["gk_distance_m_sq"] * (gkd[gk_mask] ** 2)

            logit += c["is_rebound"] * is_rebound[non_penalty]
            logit += c["is_big_chance"] * is_big_chance[non_penalty]

            logit = np.clip(logit, LOGIT_CLIP_MIN, LOGIT_CLIP_MAX)
            logit = np.where(np.isfinite(logit), logit, 0.0)
            xg_values[non_penalty] = 1.0 / (1.0 + np.exp(-logit))

        xg_values[penalty_mask] = PENALTY_XG

        return results + xg_values.tolist()


def compute_xg_enhanced(
    distance_m: float,
    angle_deg: float,
    *,
    gk_distance_m: float = 0.0,
    is_header: bool = False,
    is_one_on_one: bool = False,
    is_pressed: bool = False,
    is_rebound: bool = False,
    is_big_chance: bool = False,
    shot_type: str = "open_play",
) -> float:
    """Convenience function for enhanced xG computation."""
    features = EnhancedXgFeatures(
        distance_m=distance_m,
        angle_deg=angle_deg,
        is_header=is_header,
        is_one_on_one=is_one_on_one,
        is_pressed=is_pressed,
        is_rebound=is_rebound,
        is_big_chance=is_big_chance,
        is_volley=(shot_type in ("volley", "half_volley")),
        is_free_kick=(shot_type == "free_kick"),
        is_penalty=(shot_type == "penalty"),
        gk_distance_m=gk_distance_m,
    )
    return EnhancedXgModel().compute_single(features)


def compute_xg_with_ci(
    distance_m: float,
    angle_deg: float,
    *,
    body_part: str = "right_foot",
    shot_type: str = "open_play",
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """Compute xG with Beta-conjugate credible interval.

    Uses a Beta(1,1) uniform prior updated with effective observed data
    derived from the xG model output.
    """
    xg_val = compute_xg(distance_m, angle_deg, body_part=body_part, shot_type=shot_type)
    effective_shots = 100.0
    goals = xg_val * effective_shots
    n_shots = effective_shots
    alpha_post = goals + 1.0
    beta_post = n_shots - goals + 1.0
    mean = alpha_post / (alpha_post + beta_post)
    lower = beta.ppf((1.0 - confidence_level) / 2.0, alpha_post, beta_post)
    upper = beta.ppf(1.0 - (1.0 - confidence_level) / 2.0, alpha_post, beta_post)
    return {
        "xg": round(mean, 4),
        "ci_lower": round(float(lower), 4),
        "ci_upper": round(float(upper), 4),
        "confidence_level": confidence_level,
    }
