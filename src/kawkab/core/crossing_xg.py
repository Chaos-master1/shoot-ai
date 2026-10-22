"""Cross-specific xG model.

Computes expected goals from crosses using cross-specific features
such as cross height, distance from goal, and defender proximity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from kawkab.core.game_constants import GAME

PITCH_LENGTH = GAME.PITCH_LENGTH_M
PITCH_WIDTH = GAME.PITCH_WIDTH_M


@dataclass
class CrossXgFactors:
    distance_m: float = 0.0
    cross_height: str = ""
    defender_distance_m: float = 5.0
    from_byline: bool = False
    headed_chance: float = 0.0
    placement_angle_deg: float = 0.0
    base_xg: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "distance_m": round(self.distance_m, 1),
            "cross_height": self.cross_height,
            "defender_distance_m": round(self.defender_distance_m, 1),
            "from_byline": self.from_byline,
            "headed_chance": round(self.headed_chance, 3),
            "placement_angle_deg": round(self.placement_angle_deg, 1),
            "base_xg": round(self.base_xg, 4),
        }


# Base xG for a cross based on destination zone
CROSS_ZONE_XG: dict[str, float] = {
    "six_yard": 0.12,
    "near_post": 0.08,
    "far_post": 0.06,
    "penalty_spot": 0.05,
    "edge_of_box": 0.03,
    "deep": 0.01,
}

# Height adjustment multipliers
HEIGHT_MULTIPLIER: dict[str, float] = {
    "ground": 1.2,
    "low": 1.0,
    "high": 0.7,
    "lofted": 0.5,
}


def _cross_destination_zone(
    end_x: float,
    end_y: float,
) -> str:
    """Classify cross destination zone."""
    if end_x > PITCH_LENGTH - 5.5:
        return "six_yard"
    if end_x > PITCH_LENGTH - 11.0:
        near_post_dist = min(abs(end_y - PITCH_WIDTH * 0.4), abs(end_y - PITCH_WIDTH * 0.6))
        if near_post_dist < 5:
            return "near_post"
        return "far_post"
    if end_x > PITCH_LENGTH - 20.0:
        return "penalty_spot"
    if end_x > PITCH_LENGTH - 30.0:
        return "edge_of_box"
    return "deep"


def compute_cross_xg(event: dict[str, Any]) -> CrossXgFactors:
    """Compute xG for a cross event.

    Uses cross-specific features:
      - Distance from goal
      - Cross height (ground, low, high, lofted)
      - Defender proximity
      - Whether cross is from the byline

    Args:
        event: Dict with end_x, end_y, start_x, start_y, and optional
            cross_height, defender_distance.

    Returns:
        CrossXgFactors with base_xg and factor breakdown.
    """
    end_x = event.get("end_x", 0.0)
    end_y = event.get("end_y", PITCH_WIDTH / 2)
    start_x = event.get("start_x", 0.0)
    start_y = event.get("start_y", PITCH_WIDTH / 2)

    distance_m = math.hypot(end_x - start_x, end_y - start_y)
    goal_distance = PITCH_LENGTH - end_x

    cross_height = event.get("cross_height", "low")
    if cross_height not in HEIGHT_MULTIPLIER:
        cross_height = "low"

    defender_distance = event.get("defender_distance", event.get("defender_distance_m", 5.0))

    zone = _cross_destination_zone(end_x, end_y)
    base_zone_xg = CROSS_ZONE_XG.get(zone, 0.02)

    height_mult = HEIGHT_MULTIPLIER[cross_height]
    dist_decay = max(0.1, 1.0 - (goal_distance / PITCH_LENGTH))
    def_factor = min(1.5, max(0.3, defender_distance / 5.0))

    from_byline = start_x > PITCH_LENGTH - 5.0
    byline_boost = 1.15 if from_byline else 1.0

    placement_angle = abs(end_y - PITCH_WIDTH / 2) / (PITCH_WIDTH / 2) * 90
    angle_factor = 0.7 + 0.3 * (placement_angle / 90.0)

    headed_chance = (
        0.2 if cross_height in ("high", "lofted") else 0.8 if cross_height == "ground" else 0.5
    )

    xg = base_zone_xg * height_mult * dist_decay * def_factor * byline_boost * angle_factor
    xg = min(xg, 0.35)

    return CrossXgFactors(
        distance_m=round(distance_m, 1),
        cross_height=cross_height,
        defender_distance_m=round(defender_distance, 1),
        from_byline=from_byline,
        headed_chance=round(headed_chance, 3),
        placement_angle_deg=round(placement_angle, 1),
        base_xg=round(xg, 4),
    )
