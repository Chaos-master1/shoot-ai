"""Carry xT — expected threat gained through ball carries (dribbling).

Separates carry threat from pass threat. A carry that moves the ball
into a higher-value zone creates value even without a pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from kawkab.core.xt_model import ExpectedThreatModel


@dataclass
class CarryXTResult:
    event_index: int = 0
    timestamp: float = 0.0
    team: str = "home"
    start_x: float = 0.0
    start_y: float = 34.0
    end_x: float = 0.0
    end_y: float = 34.0
    carry_distance: float = 0.0
    xt_gained: float = 0.0
    progressive: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "idx": self.event_index,
            "t": round(self.timestamp, 1),
            "team": self.team,
            "start_x": round(self.start_x, 1),
            "start_y": round(self.start_y, 1),
            "end_x": round(self.end_x, 1),
            "end_y": round(self.end_y, 1),
            "dist": round(self.carry_distance, 1),
            "xt": round(self.xt_gained, 4),
            "prog": self.progressive,
        }


@dataclass
class CarryXTMatchReport:
    home_total_xt: float = 0.0
    away_total_xt: float = 0.0
    home_carries: int = 0
    away_carries: int = 0
    home_progressive: int = 0
    away_progressive: int = 0
    home_avg_xt_per_carry: float = 0.0
    away_avg_xt_per_carry: float = 0.0
    carries: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "home_total_xt": round(self.home_total_xt, 4),
            "away_total_xt": round(self.away_total_xt, 4),
            "home_carries": self.home_carries,
            "away_carries": self.away_carries,
            "home_progressive": self.home_progressive,
            "away_progressive": self.away_progressive,
            "home_avg_xt": round(self.home_avg_xt_per_carry, 4),
            "away_avg_xt": round(self.away_avg_xt_per_carry, 4),
        }


def compute_carry_xt(
    events: list[dict[str, Any]],
    xt_model: ExpectedThreatModel | None = None,
    attacking_direction: str = "right",
) -> CarryXTMatchReport:
    """Compute carry-based expected threat from match events.

    Args:
        events: List of event dicts with type, team, start_x, start_y, end_x, end_y.
        xt_model: Optional pre-built ExpectedThreatModel. Builds one if not provided.
        attacking_direction: Direction the home team attacks ("right" or "left").

    Returns:
        CarryXTMatchReport with per-team carry xT stats.
    """
    if xt_model is None:
        xt_model = ExpectedThreatModel()
        xt_model.build_transition_matrix(events)

    carry_events = [ev for ev in events if ev.get("type") == "carry"]
    if not carry_events:
        return CarryXTMatchReport()

    home_total = 0.0
    away_total = 0.0
    home_count = 0
    away_count = 0
    home_prog = 0
    away_prog = 0
    carry_results: list[dict[str, Any]] = []

    for i, ev in enumerate(carry_events):
        sx = ev.get("start_x", 0.0)
        sy = ev.get("start_y", 34.0)
        ex = ev.get("end_x", 0.0)
        ey = ev.get("end_y", 34.0)
        team = ev.get("team", "home")
        ts = ev.get("timestamp", 0.0)

        xt_gained = xt_model.compute_action_xt(sx, sy, ex, ey)
        carry_distance = math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)

        if attacking_direction == "left":
            forward_distance = sx - ex if team == "home" else ex - sx
        else:
            forward_distance = ex - sx if team == "home" else sx - ex
        progressive = xt_gained > 0.0 or forward_distance > 5.0

        res = CarryXTResult(
            event_index=i,
            timestamp=ts,
            team=team,
            start_x=sx,
            start_y=sy,
            end_x=ex,
            end_y=ey,
            carry_distance=carry_distance,
            xt_gained=xt_gained,
            progressive=progressive,
        )
        carry_results.append(res.to_dict())

        if team == "home":
            home_total += xt_gained
            home_count += 1
            if progressive:
                home_prog += 1
        else:
            away_total += xt_gained
            away_count += 1
            if progressive:
                away_prog += 1

    return CarryXTMatchReport(
        home_total_xt=home_total,
        away_total_xt=away_total,
        home_carries=home_count,
        away_carries=away_count,
        home_progressive=home_prog,
        away_progressive=away_prog,
        home_avg_xt_per_carry=home_total / max(home_count, 1),
        away_avg_xt_per_carry=away_total / max(away_count, 1),
        carries=carry_results,
    )


def compute_carry_xt_from_tracking(
    frames: list[dict[str, Any]],
    events: list[dict[str, Any]],
    xt_model: ExpectedThreatModel | None = None,
    attacking_direction: str = "right",
) -> CarryXTMatchReport:
    """Compute carry xT from tracking data where carry events are detected.

    Args:
        frames: Tracking frames with player positions and ball position.
        events: Detected events (must include "carry" type events).
        xt_model: Optional pre-built ExpectedThreatModel.
        attacking_direction: Direction the home team attacks ("right" or "left").

    Returns:
        CarryXTMatchReport with per-team carry xT stats.
    """
    if xt_model is None:
        xt_model = ExpectedThreatModel()
        xt_model.build_transition_matrix(events)

    carry_events = [ev for ev in events if ev.get("type") == "carry"]
    if not carry_events:
        return CarryXTMatchReport()

    home_total = 0.0
    away_total = 0.0
    home_count = 0
    away_count = 0
    home_prog = 0
    away_prog = 0
    carry_results: list[dict[str, Any]] = []

    for i, ev in enumerate(carry_events):
        sx = ev.get("start_x", 0.0)
        sy = ev.get("start_y", 34.0)
        ex = ev.get("end_x", 0.0)
        ey = ev.get("end_y", 34.0)
        team = ev.get("team", "home")
        ts = ev.get("timestamp", 0.0)

        xt_gained = xt_model.compute_action_xt(sx, sy, ex, ey)
        carry_distance = math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)

        if attacking_direction == "left":
            forward_distance = sx - ex if team == "home" else ex - sx
        else:
            forward_distance = ex - sx if team == "home" else sx - ex
        progressive = xt_gained > 0.0 or forward_distance > 5.0

        res = CarryXTResult(
            event_index=i,
            timestamp=ts,
            team=team,
            start_x=sx,
            start_y=sy,
            end_x=ex,
            end_y=ey,
            carry_distance=carry_distance,
            xt_gained=xt_gained,
            progressive=progressive,
        )
        carry_results.append(res.to_dict())

        if team == "home":
            home_total += xt_gained
            home_count += 1
            if progressive:
                home_prog += 1
        else:
            away_total += xt_gained
            away_count += 1
            if progressive:
                away_prog += 1

    return CarryXTMatchReport(
        home_total_xt=home_total,
        away_total_xt=away_total,
        home_carries=home_count,
        away_carries=away_count,
        home_progressive=home_prog,
        away_progressive=away_prog,
        home_avg_xt_per_carry=home_total / max(home_count, 1),
        away_avg_xt_per_carry=away_total / max(away_count, 1),
        carries=carry_results,
    )
