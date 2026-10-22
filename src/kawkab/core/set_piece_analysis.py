"""Set piece analysis — delivery quality scoring with xG estimation.

Enhanced with zone-weighted delivery quality scores and
expected xG from set piece deliveries.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from kawkab.core.xg_model import EnhancedXgModel

SET_PIECE_TYPES = {"corner_kick", "free_kick", "throw_in", "goal_kick", "penalty"}

# Delivery zone quality weights (16×12 grid mapped from 0-105 x 0-68)
# Higher weight = more dangerous delivery zone
_DELIVERY_QUALITY_GRID: list[list[float]] = [
    # Near goal (high x)
    [0.35, 0.40, 0.45, 0.45, 0.40, 0.35, 0.30, 0.30, 0.35, 0.40, 0.45, 0.45, 0.40, 0.35],
    [0.30, 0.35, 0.40, 0.40, 0.35, 0.30, 0.25, 0.25, 0.30, 0.35, 0.40, 0.40, 0.35, 0.30],
    [0.25, 0.30, 0.35, 0.35, 0.30, 0.25, 0.20, 0.20, 0.25, 0.30, 0.35, 0.35, 0.30, 0.25],
    [0.20, 0.25, 0.30, 0.30, 0.25, 0.20, 0.15, 0.15, 0.20, 0.25, 0.30, 0.30, 0.25, 0.20],
    [0.15, 0.18, 0.22, 0.22, 0.18, 0.15, 0.12, 0.12, 0.15, 0.18, 0.22, 0.22, 0.18, 0.15],
    [0.10, 0.12, 0.15, 0.15, 0.12, 0.10, 0.08, 0.08, 0.10, 0.12, 0.15, 0.15, 0.12, 0.10],
    [0.06, 0.08, 0.10, 0.10, 0.08, 0.06, 0.05, 0.05, 0.06, 0.08, 0.10, 0.10, 0.08, 0.06],
    [0.03, 0.05, 0.07, 0.07, 0.05, 0.03, 0.03, 0.03, 0.03, 0.05, 0.07, 0.07, 0.05, 0.03],
    [0.02, 0.03, 0.05, 0.05, 0.03, 0.02, 0.02, 0.02, 0.02, 0.03, 0.05, 0.05, 0.03, 0.02],
    [0.01, 0.02, 0.03, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.03, 0.02, 0.01],
    [0.005, 0.01, 0.02, 0.02, 0.01, 0.005, 0.005, 0.005, 0.005, 0.01, 0.02, 0.02, 0.01, 0.005],
    [0.0, 0.0, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.01, 0.0, 0.0],
]

# Corner-specific boost factor by delivery zone (near post vs far post)
_CORNER_ZONE_BOOST: dict[str, float] = {
    "Near Post": 1.2,
    "Far Post": 1.15,
    "Centre": 1.0,
    "Edge of Box": 0.7,
    "Midfield": 0.3,
    "Deep": 0.1,
}

_FREE_KICK_ZONE_BOOST: dict[str, float] = {
    "Near Post": 1.3,
    "Far Post": 1.1,
    "Centre": 1.2,
    "Edge of Box": 0.9,
    "Midfield": 0.2,
    "Deep": 0.05,
}


@dataclass
class SetPieceSummary:
    type: str = ""
    count: int = 0
    shots: int = 0
    goals: int = 0
    total_xg: float = 0.0
    avg_xg_per_set_piece: float = 0.0
    conversion_rate: float = 0.0
    threat_rating: float = 0.0
    delivery_quality: float = 0.0
    expected_xg_from_delivery: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "count": self.count,
            "shots": self.shots,
            "goals": self.goals,
            "total_xg": round(self.total_xg, 3),
            "avg_xg_per_set_piece": round(self.avg_xg_per_set_piece, 3),
            "conversion_rate": round(self.conversion_rate, 3),
            "threat_rating": round(self.threat_rating, 3),
            "delivery_quality": round(self.delivery_quality, 3),
            "expected_xg_from_delivery": round(self.expected_xg_from_delivery, 3),
        }


@dataclass
class DeliveryZone:
    label: str = ""
    count: int = 0
    shots: int = 0
    goals: int = 0
    total_xg: float = 0.0
    delivery_quality: float = 0.0
    zone_weight: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "count": self.count,
            "shots": self.shots,
            "goals": self.goals,
            "total_xg": round(self.total_xg, 3),
            "delivery_quality": round(self.delivery_quality, 3),
            "zone_weight": round(self.zone_weight, 3),
        }


@dataclass
class SetPieceReport:
    total_set_pieces: int = 0
    home_set_pieces: int = 0
    away_set_pieces: int = 0
    summaries: list[dict[str, Any]] = field(default_factory=list)
    delivery_zones: list[dict[str, Any]] = field(default_factory=list)
    home_total_xg: float = 0.0
    away_total_xg: float = 0.0
    home_goals: int = 0
    away_goals: int = 0
    overall_delivery_quality: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_set_pieces": self.total_set_pieces,
            "home_set_pieces": self.home_set_pieces,
            "away_set_pieces": self.away_set_pieces,
            "summaries": self.summaries,
            "delivery_zones": self.delivery_zones,
            "home_total_xg": round(self.home_total_xg, 3),
            "away_total_xg": round(self.away_total_xg, 3),
            "home_goals": self.home_goals,
            "away_goals": self.away_goals,
            "overall_delivery_quality": round(self.overall_delivery_quality, 3),
        }


def _classify_set_piece_type(event: dict) -> str | None:
    etype = event.get("event_type", event.get("type", ""))
    if etype in SET_PIECE_TYPES:
        return etype
    if etype == "shot":
        metadata = event.get("metadata", {})
        if isinstance(metadata, dict):
            sp = metadata.get("set_piece", metadata.get("situation", ""))
            if sp in SET_PIECE_TYPES:
                return sp
    return None


def _delivery_zone_label(x: float, y: float) -> str:
    if x < 30:
        return "Deep"
    if x > 90:
        if y < 20:
            return "Near Post"
        if y > 48:
            return "Far Post"
        return "Centre"
    if x < 60:
        return "Midfield"
    return "Edge of Box"


def _compute_delivery_quality(
    end_x: float,
    end_y: float,
    sp_type: str,
    has_shot: bool = False,
    xg_value: float = 0.0,
) -> float:
    """Compute delivery quality score (0-1) for a set piece delivery.

    Args:
        end_x: Delivery endpoint x-coordinate.
        end_y: Delivery endpoint y-coordinate.
        sp_type: Set piece type ("corner_kick", "free_kick", etc).
        has_shot: Whether a shot resulted.
        xg_value: xG of the resulting shot.

    Returns:
        Quality score from 0 (poor) to 1 (excellent).
    """
    # Zone weight from grid
    zx = min(13, max(0, int(end_x / 105.0 * 14)))
    zy = min(11, max(0, int((68 - end_y) / 68.0 * 12)))
    zone_weight = _DELIVERY_QUALITY_GRID[zy][zx]

    # Distance to goal
    dist_to_goal = math.sqrt((105 - end_x) ** 2 + (34 - end_y) ** 2)
    dist_factor = max(0.0, 1.0 - dist_to_goal / 60.0)

    # Type-specific boost
    zone_label = _delivery_zone_label(end_x, end_y)
    if sp_type == "corner_kick":
        boost = _CORNER_ZONE_BOOST.get(zone_label, 1.0)
    elif sp_type == "free_kick":
        boost = _FREE_KICK_ZONE_BOOST.get(zone_label, 1.0)
    else:
        boost = 1.0

    # Outcome bonus
    outcome_bonus = 0.0
    if has_shot:
        outcome_bonus = 0.15
        if xg_value > 0.2:
            outcome_bonus += 0.1

    quality = (zone_weight * 0.4 + dist_factor * 0.3) * boost + outcome_bonus
    return min(1.0, max(0.0, quality))


def _estimate_set_piece_xg(
    end_x: float,
    end_y: float,
    sp_type: str,
    xg_model: EnhancedXgModel | None = None,
) -> float:
    """Estimate expected xG from a set piece delivery to (end_x, end_y).

    Uses delivery quality and historical conversion rates to estimate
    the xG value of the set piece itself (not just the resulting shot).

    Args:
        end_x: Delivery endpoint x-coordinate.
        end_y: Delivery endpoint y-coordinate.
        sp_type: Set piece type.
        xg_model: Optional EnhancedXgModel for shot-based estimation.

    Returns:
        Expected xG from the set piece delivery.
    """
    quality = _compute_delivery_quality(end_x, end_y, sp_type)

    # Base rates by set piece type
    base_rate = {
        "corner_kick": 0.035,
        "free_kick": 0.025,
        "penalty": 0.76,
        "throw_in": 0.005,
        "goal_kick": 0.001,
    }.get(sp_type, 0.01)

    # Penalties are independent of delivery quality
    if sp_type == "penalty":
        return base_rate

    # Quality-adjusted xG
    xg_estimate = base_rate * (0.3 + 0.7 * quality)

    # For direct free kicks (shots from the delivery position), use xG model
    if sp_type == "free_kick" and xg_model is not None:
        # Estimate if this is a direct free kick (shot from set piece)
        dist = math.sqrt((105 - end_x) ** 2 + (34 - end_y) ** 2)
        if dist < 40:
            angle = math.degrees(math.atan2(abs(34 - end_y), 105 - end_x))
            features = {
                "type": "shot",
                "distance_m": dist,
                "angle_deg": angle,
                "is_goal": False,
                "body_part": "right_foot",
                "shot_type": "free_kick",
            }
            xg_model_shot = xg_model.compute(features)
            xg_estimate = max(xg_estimate, xg_model_shot * 0.4)

    return xg_estimate


def analyze_set_pieces(
    events: list[dict[str, Any]],
    xg_model: EnhancedXgModel | None = None,
) -> SetPieceReport:
    """Analyze set pieces with delivery quality scoring.

    Args:
        events: List of event dicts with event_type, x, y, team, is_goal, xg.
        xg_model: Optional EnhancedXgModel for direct free kick estimation.

    Returns:
        SetPieceReport with summaries, delivery zones, and quality scores.
    """
    if not events:
        return SetPieceReport()

    _xg_model = xg_model or EnhancedXgModel()

    type_data: dict[str, dict[str, float | int | float]] = {}
    type_quality: dict[str, list[float]] = defaultdict(list)
    type_xg_estimate: dict[str, list[float]] = defaultdict(list)
    zone_data: dict[str, DeliveryZone] = {}
    home_total_xg = 0.0
    away_total_xg = 0.0
    home_goals = 0
    away_goals = 0
    home_sp_count = 0
    away_sp_count = 0
    all_qualities: list[float] = []

    for ev in events:
        sp_type = _classify_set_piece_type(ev)
        if sp_type is None:
            continue

        team = ev.get("team", "home")
        x = ev.get("x", 0.0)
        y = ev.get("y", 34.0)
        xg = ev.get("xg", 0.0)
        is_goal = ev.get("is_goal", False)
        is_shot = ev.get("type") == "shot"

        if team == "home":
            home_sp_count += 1
            if is_goal:
                home_goals += 1
            home_total_xg += xg
        else:
            away_sp_count += 1
            if is_goal:
                away_goals += 1
            away_total_xg += xg

        if sp_type not in type_data:
            type_data[sp_type] = {"count": 0, "shots": 0, "goals": 0, "total_xg": 0.0}
        td = type_data[sp_type]
        td["count"] += 1
        if is_shot:
            td["shots"] += 1
        if is_goal:
            td["goals"] += 1
        td["total_xg"] += xg

        # Delivery quality for corners, free kicks
        if sp_type in ("corner_kick", "free_kick"):
            pass_x = ev.get("pass_end_x", x)
            pass_y = ev.get("pass_end_y", y + 10)
            quality = _compute_delivery_quality(pass_x, pass_y, sp_type, is_shot, xg)
            all_qualities.append(quality)
            type_quality[sp_type].append(quality)

            xg_est = _estimate_set_piece_xg(pass_x, pass_y, sp_type, _xg_model)
            type_xg_estimate[sp_type].append(xg_est)

            zone_label = _delivery_zone_label(pass_x, pass_y)
            zone_weight = _compute_delivery_quality(pass_x, pass_y, sp_type)
            if zone_label not in zone_data:
                zone_data[zone_label] = DeliveryZone(label=zone_label)
            zd = zone_data[zone_label]
            zd.count += 1
            zd.shots += 1 if is_shot else 0
            zd.goals += 1 if is_goal else 0
            zd.total_xg += xg
            zd.delivery_quality += quality
            zd.zone_weight = max(zd.zone_weight, zone_weight)
        elif sp_type == "penalty":
            quality = 0.95 if is_goal else 0.75
            all_qualities.append(quality)
            type_quality[sp_type].append(quality)
            xg_est = 0.76
            type_xg_estimate[sp_type].append(xg_est)
        else:
            quality = 0.3 if is_shot else 0.1
            all_qualities.append(quality)
            type_quality[sp_type].append(quality)
            xg_est = 0.01
            type_xg_estimate[sp_type].append(xg_est)

    # Average zone delivery quality
    for zd in zone_data.values():
        if zd.count > 0:
            zd.delivery_quality /= zd.count

    summaries: list[SetPieceSummary] = []
    for sp_type in sorted(type_data.keys()):
        td = type_data[sp_type]
        c = td["count"]
        s = td["shots"]
        g = td["goals"]
        txg = td["total_xg"]
        avg_xg = txg / c if c > 0 else 0.0
        conv = g / c if c > 0 else 0.0
        threat = (s * 0.3 + g * 0.7 + txg * 0.5) / max(c, 1)
        q_list = type_quality.get(sp_type, [])
        avg_quality = sum(q_list) / len(q_list) if q_list else 0.0
        xe_list = type_xg_estimate.get(sp_type, [])
        avg_xe = sum(xe_list) / len(xe_list) if xe_list else 0.0
        summaries.append(
            SetPieceSummary(
                type=sp_type,
                count=int(c),
                shots=int(s),
                goals=int(g),
                total_xg=txg,
                avg_xg_per_set_piece=avg_xg,
                conversion_rate=conv,
                threat_rating=threat,
                delivery_quality=avg_quality,
                expected_xg_from_delivery=avg_xe,
            )
        )

    overall_quality = sum(all_qualities) / len(all_qualities) if all_qualities else 0.0

    return SetPieceReport(
        total_set_pieces=sum(int(td["count"]) for td in type_data.values()),
        home_set_pieces=home_sp_count,
        away_set_pieces=away_sp_count,
        summaries=[s.to_dict() for s in summaries],
        delivery_zones=[
            z.to_dict() for z in sorted(zone_data.values(), key=lambda z: z.count, reverse=True)
        ],
        home_total_xg=home_total_xg,
        away_total_xg=away_total_xg,
        home_goals=home_goals,
        away_goals=away_goals,
        overall_delivery_quality=overall_quality,
    )
