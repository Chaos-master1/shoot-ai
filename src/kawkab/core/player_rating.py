"""Player performance index with position-specific ratings.

Computes 0-10 ratings for each player based on their position,
using weighted sub-scores for passing, shooting, defending,
physical output, positioning, and dribbling.

Position is inferred from average x-position (defenders deep,
attackers high) or assigned explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class PlayerPosition(Enum):
    GK = "Goalkeeper"
    CB = "Centre-Back"
    FB = "Full-Back"
    CDM = "Defensive Midfield"
    CM = "Central Midfield"
    CAM = "Attacking Midfield"
    WING = "Winger"
    ST = "Striker"
    UNASSIGNED = "Unassigned"


@dataclass
class PlayerRating:
    """Position-specific player rating (0-10 scale)."""

    overall: float = 0.0
    passing: float = 0.0
    shooting: float = 0.0
    defending: float = 0.0
    physical: float = 0.0
    positioning: float = 0.0
    dribbling: float = 0.0
    position: PlayerPosition = PlayerPosition.UNASSIGNED
    n_matches: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": round(self.overall, 1),
            "passing": round(self.passing, 1),
            "shooting": round(self.shooting, 1),
            "defending": round(self.defending, 1),
            "physical": round(self.physical, 1),
            "positioning": round(self.positioning, 1),
            "dribbling": round(self.dribbling, 1),
            "position": self.position.value,
        }


# Position weight templates: how each sub-score contributes to overall
_WEIGHTS: dict[PlayerPosition, dict[str, float]] = {
    PlayerPosition.GK: {
        "passing": 0.30,
        "shooting": 0.00,
        "defending": 0.40,
        "physical": 0.10,
        "positioning": 0.15,
        "dribbling": 0.05,
    },
    PlayerPosition.CB: {
        "passing": 0.15,
        "shooting": 0.05,
        "defending": 0.50,
        "physical": 0.15,
        "positioning": 0.10,
        "dribbling": 0.05,
    },
    PlayerPosition.FB: {
        "passing": 0.20,
        "shooting": 0.05,
        "defending": 0.30,
        "physical": 0.20,
        "positioning": 0.10,
        "dribbling": 0.15,
    },
    PlayerPosition.CDM: {
        "passing": 0.25,
        "shooting": 0.05,
        "defending": 0.35,
        "physical": 0.20,
        "positioning": 0.10,
        "dribbling": 0.05,
    },
    PlayerPosition.CM: {
        "passing": 0.30,
        "shooting": 0.10,
        "defending": 0.15,
        "physical": 0.20,
        "positioning": 0.15,
        "dribbling": 0.10,
    },
    PlayerPosition.CAM: {
        "passing": 0.25,
        "shooting": 0.20,
        "defending": 0.05,
        "physical": 0.10,
        "positioning": 0.20,
        "dribbling": 0.20,
    },
    PlayerPosition.WING: {
        "passing": 0.20,
        "shooting": 0.15,
        "defending": 0.05,
        "physical": 0.15,
        "positioning": 0.20,
        "dribbling": 0.25,
    },
    PlayerPosition.ST: {
        "passing": 0.10,
        "shooting": 0.40,
        "defending": 0.05,
        "physical": 0.15,
        "positioning": 0.20,
        "dribbling": 0.10,
    },
    PlayerPosition.UNASSIGNED: {
        "passing": 0.20,
        "shooting": 0.15,
        "defending": 0.20,
        "physical": 0.20,
        "positioning": 0.10,
        "dribbling": 0.15,
    },
}


def _infer_position_from_x(
    avg_x: float,
    pitch_length: float = 105.0,
) -> PlayerPosition:
    """Infer player position from average x-coordinate.

    Uses thirds of the pitch relative to the team's attacking direction.
    For simplicity assumes 4-3-3 style distribution.
    """
    x_pct = avg_x / pitch_length
    if x_pct < 0.05:
        return PlayerPosition.GK
    elif x_pct < 0.25:
        return PlayerPosition.CB
    elif x_pct < 0.35:
        return PlayerPosition.CDM
    elif x_pct < 0.55:
        return PlayerPosition.CM
    elif x_pct < 0.70:
        return PlayerPosition.CAM
    elif x_pct < 0.85:
        return PlayerPosition.WING
    else:
        return PlayerPosition.ST


def _norm(value: float, min_v: float, max_v: float) -> float:
    """Normalize a value to 0-1 range."""
    if max_v <= min_v:
        return 0.5
    return max(0.0, min(1.0, (value - min_v) / (max_v - min_v)))


def compute_rating(
    *,
    pass_accuracy: float = 0.0,
    passes_completed: int = 0,
    passes_attempted: int = 0,
    progressive_passes: int = 0,
    key_passes: int = 0,
    assists: int = 0,
    shots: int = 0,
    shots_on_target: int = 0,
    goals: float = 0.0,
    xg: float = 0.0,
    tackles: int = 0,
    interceptions: int = 0,
    defensive_actions: int = 0,
    carries: int = 0,
    progressive_carries: int = 0,
    distance_covered_m: float = 0.0,
    max_speed_kmh: float = 0.0,
    sprints: int = 0,
    possession_time_s: float = 0.0,
    minutes_played: float = 90.0,
    avg_x: float | None = None,
    position: PlayerPosition | None = None,
    pitch_length: float = 105.0,
) -> PlayerRating:
    """Compute a 0-10 player rating from match statistics.

    Args:
        pass_accuracy: 0-1 fraction.
        passes_completed: Number of completed passes.
        progressive_passes: Number of progressive passes.
        key_passes: Passes leading to a shot.
        assists: Direct assists.
        shots: Total shots.
        shots_on_target: Shots on target.
        goals: Goals scored (float for xG equivalency).
        xg: Expected goals from shots.
        tackles: Total tackles.
        interceptions: Interceptions made.
        defensive_actions: Total defensive actions.
        carries: Total carries.
        progressive_carries: Progressive carries.
        distance_covered_m: Total distance.
        max_speed_kmh: Peak speed.
        sprints: Sprint count.
        possession_time_s: Time on ball.
        minutes_played: Minutes played (for per-90 normalization).
        avg_x: Average x-coordinate for position inference.
        position: Explicit position override.
        pitch_length: Pitch length for position inference.

    Returns:
        PlayerRating with sub-scores and overall.
    """
    if minutes_played <= 0:
        minutes_played = 90.0

    per90 = 90.0 / minutes_played

    pos = position or (
        _infer_position_from_x(avg_x, pitch_length)
        if avg_x is not None
        else PlayerPosition.UNASSIGNED
    )

    pass_acc_score = _norm(pass_accuracy, 0.4, 0.95)
    pass_vol_score = _norm(passes_completed * per90, 10, 70)

    passing = (
        pass_acc_score * 0.5
        + pass_vol_score * 0.3
        + _norm(progressive_passes * per90, 0, 10) * 0.1
        + _norm(key_passes * per90, 0, 3) * 0.1
    )

    shot_vol = _norm(shots * per90, 0, 5)
    shot_acc = _norm(shots_on_target / max(shots, 1), 0.1, 0.8) if shots > 0 else 0.0
    shooting = (
        shot_vol * 0.3
        + shot_acc * 0.3
        + _norm(goals * per90, 0, 2) * 0.2
        + _norm(xg * per90, 0, 2) * 0.2
    )

    def_vol = _norm(defensive_actions * per90, 0, 20)
    tackling = _norm(tackles * per90, 0, 6)
    intercepting = _norm(interceptions * per90, 0, 4)
    defending = def_vol * 0.3 + tackling * 0.3 + intercepting * 0.4

    dist = _norm(distance_covered_m / 1000.0, 3, 14)
    speed = _norm(max_speed_kmh, 15, 35)
    sprint_score = _norm(sprints * per90, 0, 15)
    physical = dist * 0.4 + speed * 0.3 + sprint_score * 0.3

    pos_time = _norm(possession_time_s * per90, 10, 120)
    positioning = (
        _norm(distance_covered_m * per90 / 1000.0, 3, 14) * 0.3
        + pos_time * 0.3
        + _norm(interceptions * per90, 0, 3) * 0.2
        + _norm(defensive_actions * per90, 0, 15) * 0.2
    )

    carry_vol = _norm(carries * per90, 0, 40)
    prog_carry = _norm(progressive_carries * per90, 0, 8)
    dribbling = carry_vol * 0.4 + prog_carry * 0.4 + _norm(progressive_passes * per90, 0, 6) * 0.2

    scores = {
        "passing": min(10.0, passing * 10.0),
        "shooting": min(10.0, shooting * 10.0),
        "defending": min(10.0, defending * 10.0),
        "physical": min(10.0, physical * 10.0),
        "positioning": min(10.0, positioning * 10.0),
        "dribbling": min(10.0, dribbling * 10.0),
    }

    weights = _WEIGHTS.get(pos, _WEIGHTS[PlayerPosition.UNASSIGNED])
    overall = sum(scores[k] * weights[k] for k in weights)

    return PlayerRating(
        overall=overall,
        passing=scores["passing"],
        shooting=scores["shooting"],
        defending=scores["defending"],
        physical=scores["physical"],
        positioning=scores["positioning"],
        dribbling=scores["dribbling"],
        position=pos,
    )
