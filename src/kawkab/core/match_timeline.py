"""Match timeline analysis — xG flow, event sequence, momentum."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TimelinePoint:
    minute: float
    home_xg: float = 0.0
    away_xg: float = 0.0
    home_cumulative: float = 0.0
    away_cumulative: float = 0.0
    event_type: str | None = None
    team: str | None = None
    description: str = ""


@dataclass
class XGFlowReport:
    points: list[dict[str, Any]] = field(default_factory=list)
    home_total: float = 0.0
    away_total: float = 0.0
    home_goals: int = 0
    away_goals: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": self.points,
            "home_total": round(self.home_total, 3),
            "away_total": round(self.away_total, 3),
            "home_goals": self.home_goals,
            "away_goals": self.away_goals,
        }


def compute_xg_timeline(
    events: list[dict[str, Any]],
    match_duration_minutes: float = 90.0,
) -> XGFlowReport:
    """Compute minute-by-minute cumulative xG timeline.

    Args:
        events: List of event dicts with shot events containing xg and is_goal.
        match_duration_minutes: Total match duration in minutes.

    Returns:
        XGFlowReport with per-minute cumulative xG data points.
    """
    home_cum = 0.0
    away_cum = 0.0
    home_goals = 0
    away_goals = 0

    points: list[dict[str, Any]] = []
    # Start at 0
    points.append(
        {
            "minute": 0,
            "home_xg": 0.0,
            "away_xg": 0.0,
            "home_cumulative": 0.0,
            "away_cumulative": 0.0,
            "event_type": None,
            "team": None,
            "description": "Kick-off",
        }
    )

    # Sort shots by timestamp and interleave
    shot_events = [e for e in events if e.get("type") == "shot"]
    shot_events.sort(key=lambda e: e.get("timestamp", 0))

    for ev in shot_events:
        minute = ev.get("timestamp", 0) / 60.0
        xg = ev.get("xg", 0)
        team = ev.get("team", "home")
        is_goal = ev.get("is_goal", False)

        if team == "home":
            home_cum += xg
            if is_goal:
                home_goals += 1
        else:
            away_cum += xg
            if is_goal:
                away_goals += 1

        desc = "⚽ GOAL!" if is_goal else f"Shot ({xg:.2f} xG)"
        points.append(
            {
                "minute": round(minute, 1),
                "home_xg": round(xg if team == "home" else 0, 3),
                "away_xg": round(xg if team == "away" else 0, 3),
                "home_cumulative": round(home_cum, 3),
                "away_cumulative": round(away_cum, 3),
                "event_type": "goal" if is_goal else "shot",
                "team": team,
                "description": desc,
            }
        )

    # End point
    points.append(
        {
            "minute": match_duration_minutes,
            "home_xg": 0.0,
            "away_xg": 0.0,
            "home_cumulative": round(home_cum, 3),
            "away_cumulative": round(away_cum, 3),
            "event_type": None,
            "team": None,
            "description": "Full-time",
        }
    )

    return XGFlowReport(
        points=points,
        home_total=home_cum,
        away_total=away_cum,
        home_goals=home_goals,
        away_goals=away_goals,
    )
