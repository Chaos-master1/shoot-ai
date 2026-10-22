"""Substitution impact analysis.

Measures the effect of substitutions on team performance:
net xG change, formation shifts, pressing intensity changes,
and possession changes before vs after each substitution.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SubstitutionEvent:
    """A substitution event with before/after metrics."""

    minute: float = 0.0
    player_out_id: int | None = None
    player_in_id: int | None = None
    team: str = "home"
    xg_before: float = 0.0
    xg_after: float = 0.0
    xg_delta: float = 0.0
    possession_before_pct: float = 50.0
    possession_after_pct: float = 50.0
    possession_delta: float = 0.0
    formation_before: str = "unknown"
    formation_after: str = "unknown"
    pressing_before: float | None = None
    pressing_after: float | None = None
    shot_rate_before: float = 0.0
    shot_rate_after: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "minute": round(self.minute, 1),
            "player_out": self.player_out_id,
            "player_in": self.player_in_id,
            "team": self.team,
            "xg_before": round(self.xg_before, 3),
            "xg_after": round(self.xg_after, 3),
            "xg_delta": round(self.xg_delta, 3),
            "possession_before": round(self.possession_before_pct, 1),
            "possession_after": round(self.possession_after_pct, 1),
            "possession_delta": round(self.possession_delta, 1),
            "formation_before": self.formation_before,
            "formation_after": self.formation_after,
            "pressing_before": round(self.pressing_before, 2) if self.pressing_before else None,
            "pressing_after": round(self.pressing_after, 2) if self.pressing_after else None,
            "shot_rate_before": round(self.shot_rate_before, 2),
            "shot_rate_after": round(self.shot_rate_after, 2),
        }


@dataclass
class SubstitutionMatchReport:
    """Aggregate substitution impact for a match."""

    substitutions: list[SubstitutionEvent] = field(default_factory=list)
    net_xg_impact: float = 0.0
    net_possession_impact: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "substitutions": [s.to_dict() for s in self.substitutions],
            "net_xg_impact": round(self.net_xg_impact, 3),
            "net_possession_impact": round(self.net_possession_impact, 1),
        }


class SubstitutionAnalyzer:
    """Analyzes substitution impact on match dynamics.

    Usage:
        sa = SubstitutionAnalyzer()
        report = sa.analyze_substitutions(
            substitutions=[{"minute": 60, "player_out": 7, "player_in": 11, "team": "home"}],
            events=events,
            minute_by_minute_stats=stats,
            match_duration=90.0,
        )
    """

    WINDOW_MINUTES = 10.0

    def analyze_substitutions(
        self,
        substitutions: list[dict[str, Any]],
        events: list[dict[str, Any]],
        minute_by_minute_stats: list[dict[str, Any]] | None = None,
        match_duration: float = 90.0,
    ) -> SubstitutionMatchReport:
        """Analyze impact of all substitutions in a match.

        Args:
            substitutions: List of sub dicts with "minute", "player_out",
                          "player_in", "team".
            events: All match events with "timestamp", "type", "team".
            minute_by_minute_stats: Optional per-minute stats with
                                   "minute", "home_xg", "away_xg",
                                   "home_possession", "away_possession",
                                   "home_formation", "away_formation",
                                   "home_pressing", "away_pressing",
                                   "home_shots", "away_shots".
            match_duration: Match length in minutes.

        Returns:
            SubstitutionMatchReport with per-sub analysis.
        """
        if not substitutions:
            return SubstitutionMatchReport()

        subs: list[dict[str, Any]] = sorted(substitutions, key=lambda s: s.get("minute", 0))

        if minute_by_minute_stats:
            stat_by_minute = {s["minute"]: s for s in minute_by_minute_stats}
        else:
            stat_by_minute = {}
            if events:
                events_by_minute: dict[int, list[dict]] = defaultdict(list)
                for ev in events:
                    minute = int(ev.get("timestamp", 0) / 60)
                    events_by_minute[minute].append(ev)

                for m in range(int(match_duration) + 1):
                    mevs = events_by_minute.get(m, [])
                    home_shots = sum(
                        1 for e in mevs if e.get("type") == "shot" and e.get("team") == "home"
                    )
                    away_shots = sum(
                        1 for e in mevs if e.get("type") == "shot" and e.get("team") == "away"
                    )
                    home_xg = sum(
                        float(e.get("xg", 0))
                        for e in mevs
                        if e.get("team") == "home" and e.get("type") == "shot"
                    )
                    away_xg = sum(
                        float(e.get("xg", 0))
                        for e in mevs
                        if e.get("team") == "away" and e.get("type") == "shot"
                    )
                    stat_by_minute[m] = {
                        "minute": m,
                        "home_xg": home_xg,
                        "away_xg": away_xg,
                        "home_shots": home_shots,
                        "away_shots": away_shots,
                    }

        result_subs: list[SubstitutionEvent] = []
        total_xg_delta = 0.0
        total_possession_delta = 0.0

        for sub in subs:
            minute = sub.get("minute", 0)
            team = sub.get("team", "home")
            sub_min = max(0, minute - self.WINDOW_MINUTES / 2)
            sub_max = min(match_duration, minute + self.WINDOW_MINUTES / 2)

            xg_before = 0.0
            xg_after = 0.0
            shots_before = 0
            shots_after = 0

            for m, s in stat_by_minute.items():
                if sub_min <= m < minute:
                    if team == "home":
                        xg_before += s.get("home_xg", 0)
                        shots_before += s.get("home_shots", 0)
                    else:
                        xg_before += s.get("away_xg", 0)
                        shots_before += s.get("away_shots", 0)
                elif minute <= m <= sub_max:
                    if team == "home":
                        xg_after += s.get("home_xg", 0)
                        shots_after += s.get("home_shots", 0)
                    else:
                        xg_after += s.get("away_xg", 0)
                        shots_after += s.get("away_shots", 0)

            window_minutes = self.WINDOW_MINUTES / 2
            shot_rate_before = shots_before / max(window_minutes, 1)
            shot_rate_after = shots_after / max(window_minutes, 1)

            xg_delta = xg_after - xg_before
            if team != "home":
                xg_delta = -xg_delta

            total_xg_delta += xg_delta

            se = SubstitutionEvent(
                minute=minute,
                player_out_id=sub.get("player_out"),
                player_in_id=sub.get("player_in"),
                team=team,
                xg_before=xg_before,
                xg_after=xg_after,
                xg_delta=xg_delta,
                formation_before=sub.get("formation_before", "unknown"),
                formation_after=sub.get("formation_after", "unknown"),
                pressing_before=sub.get("pressing_before"),
                pressing_after=sub.get("pressing_after"),
                shot_rate_before=shot_rate_before,
                shot_rate_after=shot_rate_after,
            )
            result_subs.append(se)

        return SubstitutionMatchReport(
            substitutions=result_subs,
            net_xg_impact=total_xg_delta,
            net_possession_impact=total_possession_delta,
        )
