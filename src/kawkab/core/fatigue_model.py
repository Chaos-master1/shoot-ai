"""Fatigue and substitution impact model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlayerFatigueProfile:
    track_id: int = 0
    team: str = "home"
    minutes_played: float = 0.0
    distance_covered_m: float = 0.0
    high_intensity_actions: int = 0
    fatigue_index: float = 0.0  # 0 = fresh, 1 = exhausted
    speed_decline_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "team": self.team,
            "minutes_played": round(self.minutes_played, 1),
            "distance_covered_m": round(self.distance_covered_m, 0),
            "high_intensity_actions": self.high_intensity_actions,
            "fatigue_index": round(self.fatigue_index, 2),
            "speed_decline_pct": round(self.speed_decline_pct, 1),
        }


@dataclass
class SubstitutionImpact:
    track_id_in: int = 0
    track_id_out: int = 0
    team: str = "home"
    minute: float = 0.0
    impact_score: float = 0.0  # positive = beneficial
    momentum_shift: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id_in": self.track_id_in,
            "track_id_out": self.track_id_out,
            "team": self.team,
            "minute": round(self.minute, 1),
            "impact_score": round(self.impact_score, 2),
            "momentum_shift": round(self.momentum_shift, 2),
        }


@dataclass
class FatigueReport:
    home_fatigue: list[dict[str, Any]] = field(default_factory=list)
    away_fatigue: list[dict[str, Any]] = field(default_factory=list)
    substitutions: list[dict[str, Any]] = field(default_factory=list)
    home_avg_fatigue: float = 0.0
    away_avg_fatigue: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "home_fatigue": self.home_fatigue,
            "away_fatigue": self.away_fatigue,
            "substitutions": self.substitutions,
            "home_avg_fatigue": round(self.home_avg_fatigue, 2),
            "away_avg_fatigue": round(self.away_avg_fatigue, 2),
        }


def compute_fatigue(
    events: list[dict[str, Any]],
    match_duration_minutes: float = 90.0,
) -> FatigueReport:
    """Estimate player fatigue from event data.

    Fatigue is estimated from minutes played, distance covered,
    and high-intensity actions. Uses a decay model where fatigue
    accumulates and causes speed decline.

    Args:
        events: List of event dicts with team, track_id, type, timestamp.
        match_duration_minutes: Total match duration.

    Returns:
        FatigueReport with per-player fatigue profiles and substitution impacts.
    """
    from collections import defaultdict

    player_data: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "team": "home",
            "total_time": 0.0,
            "total_dist": 0.0,
            "high_intensity": 0,
            "first_seen": 1e9,
            "last_seen": 0.0,
            "sprints": 0,
        }
    )
    substitutions_list: list[dict[str, Any]] = []

    for ev in events:
        # Handle substitutions first (no track_id needed)
        if ev.get("type") == "substitution":
            s_tid = ev.get("player_in") or ev.get("track_id")
            substitutions_list.append(
                {
                    "track_id_in": ev.get("player_in", s_tid),
                    "track_id_out": ev.get("player_out", 0),
                    "team": ev.get("team", "home"),
                    "minute": ev.get("timestamp", 0) / 60.0,
                }
            )

        tid = ev.get("track_id") or ev.get("player_id")
        if tid is None:
            continue
        tid = int(tid)
        ts = ev.get("timestamp", 0)
        team = ev.get("team", "home")
        pd = player_data[tid]
        pd["team"] = team
        pd["total_time"] = max(pd["total_time"], ts)
        pd["first_seen"] = min(pd["first_seen"], ts)
        pd["last_seen"] = max(pd["last_seen"], ts)

        # Estimate distance from event type
        _type = ev.get("type", "")
        if _type in ("pass", "carry", "run"):
            dx = ev.get("end_x", 0) - ev.get("start_x", 0)
            dy = ev.get("end_y", 0) - ev.get("start_y", 0)
            dist = math.hypot(dx, dy)
            if dist > 0:
                pd["total_dist"] += dist
                if dist > 15:
                    pd["sprints"] += 1
                    pd["high_intensity"] += 1
        if _type == "shot":
            pd["high_intensity"] += 1
        if _type == "tackle":
            pd["high_intensity"] += 1

    report = FatigueReport()
    home_fatigue = []
    away_fatigue = []

    for tid, pd in player_data.items():
        minutes_played = (pd["last_seen"] - pd["first_seen"]) / 60.0
        total_dist_m = pd["total_dist"]

        # Fatigue model: exponential decay based on minutes + intensity
        intensity_factor = pd["high_intensity"] / max(minutes_played, 1.0)
        fatigue = 1.0 - math.exp(-0.02 * minutes_played - 0.08 * intensity_factor)
        fatigue = max(0.0, min(1.0, fatigue))

        # Speed decline proportional to fatigue
        speed_decline = fatigue * 15.0  # up to 15% decline

        profile = PlayerFatigueProfile(
            track_id=tid,
            team=pd["team"],
            minutes_played=minutes_played,
            distance_covered_m=total_dist_m,
            high_intensity_actions=pd["high_intensity"],
            fatigue_index=fatigue,
            speed_decline_pct=speed_decline,
        )
        if pd["team"] == "home":
            home_fatigue.append(profile)
        else:
            away_fatigue.append(profile)

    report.home_fatigue = [
        p.to_dict() for p in sorted(home_fatigue, key=lambda x: x.fatigue_index, reverse=True)
    ]
    report.away_fatigue = [
        p.to_dict() for p in sorted(away_fatigue, key=lambda x: x.fatigue_index, reverse=True)
    ]

    if home_fatigue:
        report.home_avg_fatigue = sum(p.fatigue_index for p in home_fatigue) / len(home_fatigue)
    if away_fatigue:
        report.away_avg_fatigue = sum(p.fatigue_index for p in away_fatigue) / len(away_fatigue)

    # Compute substitution impacts (before/after momentum)
    for sub in substitutions_list:
        report.substitutions.append(sub)

    return report
