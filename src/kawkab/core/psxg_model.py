"""Post-Shot Expected Goals (PSxG) — goalkeeper save probability.

PSxG measures the probability a shot on target is saved, based on
shot placement relative to the goal frame, shot speed, and angle.
This enables goalkeeper performance evaluation (goals conceded vs PSxG).

Since 2026-09-07 the coefficients are *fitted* from 2,768 StatsBomb
open-data on-target shots (match-level train/val split; Brier 0.165,
AUC 0.759, ECE 0.051 on 549 held-out shots — see
docs/validation/psxg_training_report.json). The hand-tuned values below
remain only as the fallback when trained weights are absent.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Hand-tuned fallback coefficients (superseded by fitted weights when
# trained_psxg_coefficients.json exists — auto-loaded below).
PSXG_COEFFICIENTS: dict[str, float] = {
    "intercept": 0.8,
    "distance_m": -0.04,
    "placement_distance_sq": -2.0,
    "speed_mps": -0.04,
    "angle_deg": 0.012,
    "is_header": 0.5,
    "height_center_distance": -0.6,
}

# Fitted coefficients — the ACTIVE model (see validation.train_psxg).
PSXG_FEATURE_NAMES = [
    "intercept",
    "distance_m",
    "distance_m_sq",
    "angle_opening_deg",
    "placement_height",
    "placement_lateral_abs",
    "lateral_height_interaction",
    "is_header",
    "is_free_kick",
]

TRAINED_PSXG_COEFFICIENTS: dict[str, float] = {}
_PSXG_TRAINED_PATH = Path(__file__).parent / "trained_psxg_coefficients.json"
_PSXG_TRAINED_LOADED = False
if _PSXG_TRAINED_PATH.exists():
    try:
        with open(_PSXG_TRAINED_PATH) as _f:
            _raw = json.load(_f)
        _clean = {
            k: float(v)
            for k, v in _raw.items()
            if isinstance(v, (int, float)) and not k.startswith("_")
        }
        if _clean:
            TRAINED_PSXG_COEFFICIENTS = _clean
            _PSXG_TRAINED_LOADED = True
    except Exception:
        pass


def psxg_model_available() -> bool:
    """True when fitted PSxG weights (not hand-tuned) are active."""
    return _PSXG_TRAINED_LOADED


def _compute_psxg_trained(
    distance_m: float,
    angle_deg: float,
    placement_x: float,
    placement_y: float,
    is_header: bool,
    is_free_kick: bool,
) -> float:
    """Fitted-model PSxG for an on-target shot.

    placement_x here follows the legacy 0..1 convention (0=left post,
    1=right post) — converted to meters from goal center; placement_y
    is 0=ground..1=top. Angle uses the opening-angle convention (deg,
    ~90 for close central, small for far/wide).
    """
    c = TRAINED_PSXG_COEFFICIENTS
    lateral_m = (placement_x - 0.5) * 7.32
    height_m = placement_y * 2.44
    logit = c["intercept"]
    logit += c["distance_m"] * max(distance_m, 0.5)
    logit += c["distance_m_sq"] * (distance_m**2)
    logit += c["angle_opening_deg"] * angle_deg
    logit += c["placement_height"] * height_m
    logit += c["placement_lateral_abs"] * abs(lateral_m)
    logit += c["lateral_height_interaction"] * (abs(lateral_m) * height_m)
    logit += c["is_header"] * (1.0 if is_header else 0.0)
    logit += c["is_free_kick"] * (1.0 if is_free_kick else 0.0)
    psxg = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))
    return max(0.01, min(0.98, psxg))


@dataclass
class PSxGResult:
    psxg: float = 0.0
    save_probability: float = 0.0
    shot_quality: float = 0.0
    placement_x: float = 0.0
    placement_y: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "psxg": round(self.psxg, 4),
            "save_probability": round(self.save_probability, 4),
            "shot_quality": round(self.shot_quality, 4),
        }


@dataclass
class PSxGMatchReport:
    home_psxg: float = 0.0
    away_psxg: float = 0.0
    home_goals_conceded: int = 0
    away_goals_conceded: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "home_psxg": round(self.home_psxg, 3),
            "away_psxg": round(self.away_psxg, 3),
            "home_goals_conceded": self.home_goals_conceded,
            "away_goals_conceded": self.away_goals_conceded,
            "home_g_minus_psxg": round(self.home_goals_conceded - self.home_psxg, 3),
            "away_g_minus_psxg": round(self.away_goals_conceded - self.away_psxg, 3),
        }


def compute_psxg(
    distance_m: float,
    angle_deg: float,
    placement_x: float = 0.5,
    placement_y: float = 0.5,
    shot_speed: float = 20.0,
    on_target: bool = True,
    body_part: str = "right_foot",
) -> PSxGResult:
    """Compute PSxG — probability a shot on target results in a goal.

    Uses a logistic regression with features:
    - distance (closer = higher PSxG)
    - placement distance from center (corners = higher PSxG)
    - shot speed (faster = higher PSxG)
    - angle (wider = lower PSxG, easier for goalkeeper)
    - body part (headers = lower PSxG, easier to save)

    Args:
        distance_m: Distance to goal in meters.
        angle_deg: Angle to goal in degrees.
        placement_x: Horizontal shot placement (0=left post, 1=right post).
        placement_y: Vertical shot placement (0=ground, 1=top corner).
        shot_speed: Shot speed in m/s.
        on_target: Whether the shot is on target.
        body_part: Body part used.

    Returns:
        PSxGResult with save probability and shot quality.
    """
    if not on_target:
        return PSxGResult(
            psxg=0.0,
            save_probability=0.0,
            shot_quality=0.0,
            placement_x=placement_x,
            placement_y=placement_y,
        )

    # Trained model (fitted from StatsBomb on-target shots) takes priority;
    # the hand-tuned heuristic below is the no-weights fallback.
    if _PSXG_TRAINED_LOADED:
        psxg = _compute_psxg_trained(
            distance_m=distance_m,
            angle_deg=angle_deg,
            placement_x=placement_x,
            placement_y=placement_y,
            is_header=(body_part == "head"),
            is_free_kick=False,
        )
        return PSxGResult(
            psxg=psxg,
            save_probability=1.0 - psxg,
            shot_quality=psxg,
            placement_x=placement_x,
            placement_y=placement_y,
        )

    coef = PSXG_COEFFICIENTS
    logit = coef["intercept"]

    # Distance: closer = higher PSxG (logistic: +dist_coef * dist)
    # Smooth decay using sigmoid of distance
    d_norm = distance_m / 40.0  # normalize to ~0-1 range
    logit += coef["distance_m"] * 40.0 * (1.0 - math.exp(-d_norm * 3.0))

    # Placement: distance from center squared (corners = higher PSxG)
    corner_dist = math.sqrt((placement_x - 0.5) ** 2 + (placement_y - 0.5) ** 2)
    logit += coef["placement_distance_sq"] * (-corner_dist * corner_dist * 4.0)

    # Speed: faster = harder to save
    logit += coef["speed_mps"] * shot_speed

    # Angle: wider = easier for keeper
    logit += coef["angle_deg"] * angle_deg

    # Body part: headers easier to save
    if body_part == "head":
        logit += coef["is_header"]

    # Height center distance: mid-height easier for keeper
    height_center = abs(placement_y - 0.5) * 2.0  # 0 at center, 1 at post
    logit += coef["height_center_distance"] * (1.0 - height_center)

    psxg = 1.0 / (1.0 + math.exp(-logit))
    psxg = max(0.01, min(0.98, psxg))

    return PSxGResult(
        psxg=psxg,
        save_probability=1.0 - psxg,
        shot_quality=psxg,
        placement_x=placement_x,
        placement_y=placement_y,
    )


def compute_match_psxg(events: list[dict[str, Any]]) -> PSxGMatchReport:
    """Compute PSxG for all shots in a match.

    Args:
        events: List of event dicts with shot events.

    Returns:
        PSxGMatchReport with per-team totals.
    """
    home_psxg = 0.0
    away_psxg = 0.0
    home_conceded = 0
    away_conceded = 0
    details: list[dict[str, Any]] = []

    for ev in events:
        if ev.get("type") != "shot":
            continue
        team = ev.get("team", "home")
        is_goal = ev.get("is_goal", False)
        on_target = ev.get("on_target", False)
        distance_m = ev.get("distance_m", 18.0)
        angle_deg = ev.get("angle_deg", 30.0)

        result = compute_psxg(
            distance_m=distance_m,
            angle_deg=angle_deg,
            on_target=on_target,
            body_part=ev.get("body_part", "right_foot"),
        )

        if team == "home":
            away_psxg += result.psxg  # PSxG for shots faced
            if is_goal:
                away_conceded += 1
        else:
            home_psxg += result.psxg
            if is_goal:
                home_conceded += 1

        details.append(
            {
                "timestamp": ev.get("timestamp", 0),
                "team_shooting": team,
                "psxg": round(result.psxg, 4),
                "shot_quality": round(result.shot_quality, 3),
                "is_goal": is_goal,
            }
        )

    return PSxGMatchReport(
        home_psxg=home_psxg,
        away_psxg=away_psxg,
        home_goals_conceded=home_conceded,
        away_goals_conceded=away_conceded,
        details=details,
    )
