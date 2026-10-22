"""Expected Possession Value (EPV) — per-possession and per-player valuation.

EPV measures the expected goal contribution of each possession based on
field location, progression, and eventual outcome. All numpy-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kawkab.core.game_constants import GAME

PITCH_LENGTH = GAME.PITCH_LENGTH_M
PITCH_WIDTH = GAME.PITCH_WIDTH_M
X_ZONES = 16
Y_ZONES = 12
ZONE_WIDTH = PITCH_LENGTH / X_ZONES
ZONE_HEIGHT = PITCH_WIDTH / Y_ZONES


def _to_zone(x: float, y: float) -> tuple[int, int]:
    # Events without spatial data arrive as x/y=None (storage's json_extract
    # emits NULL when the event has no coordinates). Zone math on None
    # raised TypeError for every such possession — treat unlocated events
    # as pitch-center rather than crashing the whole match report.
    if x is None or y is None:
        x = PITCH_LENGTH / 2.0
        y = PITCH_WIDTH / 2.0
    zx = min(int(x / ZONE_WIDTH), X_ZONES - 1)
    zy = min(int(y / ZONE_HEIGHT), Y_ZONES - 1)
    return (zx, zy)


def _possession_switching_events() -> set[str]:
    return {
        "tackle",
        "interception",
        "clearance",
        "block",
        "ball_recovery",
        "dribble_past",
        "miscontrol",
        "foul",
        "own_goal",
    }


def _extract_possessions(
    events: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Split events into individual possessions.

    Returns:
        List of possessions, each being a list of events.
    """
    if not events:
        return []
    sorted_ev = sorted(events, key=lambda e: e.get("timestamp", 0))
    switching = _possession_switching_events()
    possessions: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = [sorted_ev[0]]
    current_team = sorted_ev[0].get("team", "home")

    for ev in sorted_ev[1:]:
        team = ev.get("team", current_team)
        ev_type = ev.get("type", "")
        is_switch = False
        if ev_type in switching or ev_type in ("pass", "carry", "shot") and team != current_team:
            is_switch = True

        if is_switch:
            possessions.append(current)
            current = [ev]
            current_team = team
        else:
            current.append(ev)

    if current:
        possessions.append(current)
    return possessions


# Zone-based possession value grid (expected goals per 100 possessions)
_ZONE_EPV_GRID: list[list[float]] = [
    [0.50, 0.80, 1.20, 1.20, 0.80, 0.50],  # row 0 — six-yard box
    [0.20, 0.35, 0.55, 0.55, 0.35, 0.20],  # row 1 — penalty box
    [0.08, 0.15, 0.25, 0.25, 0.15, 0.08],  # row 2 — penalty box edge
    [0.04, 0.08, 0.12, 0.12, 0.08, 0.04],  # row 3 — outside box
    [0.02, 0.04, 0.06, 0.06, 0.04, 0.02],  # row 4 — final third wide
]


@dataclass
class EPVResult:
    value: float = 0.0
    events: int = 0
    start_zone: str = ""
    end_zone: str = ""
    has_shot: bool = False
    is_goal: bool = False
    team: str = "home"

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": round(self.value, 4),
            "events": self.events,
            "start_zone": self.start_zone,
            "end_zone": self.end_zone,
            "has_shot": self.has_shot,
            "is_goal": self.is_goal,
            "team": self.team,
        }


@dataclass
class EPVReport:
    possessions: list[EPVResult] = field(default_factory=list)
    home_total: float = 0.0
    away_total: float = 0.0
    home_per_possession: float = 0.0
    away_per_possession: float = 0.0
    total_possessions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "home_total": round(self.home_total, 4),
            "away_total": round(self.away_total, 4),
            "home_per_possession": round(self.home_per_possession, 4),
            "away_per_possession": round(self.away_per_possession, 4),
            "total_possessions": self.total_possessions,
            "home_possessions": sum(1 for p in self.possessions if p.team == "home"),
            "away_possessions": sum(1 for p in self.possessions if p.team == "away"),
        }


class EPVModel:
    """Expected Possession Value model.

    Values each possession by tracking its field location, progression,
    and eventual outcome (shot/goal/turnover).
    """

    def __init__(self):
        self._epv_grid = np.array(_ZONE_EPV_GRID, dtype=np.float64)
        self._zone_value_cache: dict[tuple[float, float], float] = {}

    def _zone_value(self, x: float, y: float) -> float:
        cache_key = (x, y)
        cached = self._zone_value_cache.get(cache_key)
        if cached is not None:
            return cached
        zx, zy = _to_zone(x, y)
        # Reverse x: high x (near attacking goal) -> row 0 (highest EPV)
        # x=105 -> row 0, x=0 -> row 4
        reversed_x = X_ZONES - 1 - zx
        r = min(4, int(reversed_x / X_ZONES * 5))
        c = min(5, int(zy / Y_ZONES * 6))
        value = float(self._epv_grid[r, c])
        self._zone_value_cache[cache_key] = value
        return value

    def compute_possession_epv(
        self,
        possession: list[dict[str, Any]],
    ) -> EPVResult:
        if not possession:
            return EPVResult()

        team = possession[0].get("team", "home")

        # Coordinates may be present-but-NULL for events without spatial
        # data — fall back to the previous event's position / pitch
        # center instead of propagating None into the progress math
        # (end_x - start_x raised TypeError on real imported matches).
        def _coord(value: Any, fallback: float) -> float:
            return float(value) if value is not None else fallback

        start_x = _coord(possession[0].get("x"), 52.5)
        start_y = _coord(possession[0].get("y"), 34.0)
        last_ev = possession[-1]
        raw_end_x = last_ev.get("end_x")
        if raw_end_x is None:
            raw_end_x = last_ev.get("x")
        end_x = _coord(raw_end_x, start_x)
        raw_end_y = last_ev.get("end_y")
        if raw_end_y is None:
            raw_end_y = last_ev.get("y")
        end_y = _coord(raw_end_y, start_y)

        start_val = self._zone_value(start_x, start_y)

        # Check possession outcome
        has_shot = any(ev.get("type") == "shot" for ev in possession)
        is_goal = any(ev.get("type") == "shot" and ev.get("is_goal") for ev in possession)

        # EPV = starting zone value + progression bonus + outcome bonus
        # Progression: how much further forward the possession moved
        progress = max(0.0, end_x - start_x) / PITCH_LENGTH
        progress_bonus = 0.10 * progress

        # Outcome bonus
        if is_goal:
            outcome_bonus = 0.80
        elif has_shot:
            outcome_bonus = 0.30
        else:
            outcome_bonus = -0.05

        value = start_val + progress_bonus + outcome_bonus
        value = max(-0.5, min(1.5, value))

        return EPVResult(
            value=value,
            events=len(possession),
            start_zone=f"{_to_zone(start_x, start_y)[0]}_{_to_zone(start_x, start_y)[1]}",
            end_zone=f"{_to_zone(end_x, end_y)[0]}_{_to_zone(end_x, end_y)[1]}",
            has_shot=has_shot,
            is_goal=is_goal,
            team=team,
        )

    def compute_match_epv(
        self,
        events: list[dict[str, Any]],
    ) -> EPVReport:
        possessions = _extract_possessions(events)
        report = EPVReport()
        home_total = 0.0
        away_total = 0.0
        home_count = 0
        away_count = 0

        for poss in possessions:
            result = self.compute_possession_epv(poss)
            report.possessions.append(result)
            if result.team == "home":
                home_total += result.value
                home_count += 1
            else:
                away_total += result.value
                away_count += 1

        report.home_total = home_total
        report.away_total = away_total
        report.home_per_possession = home_total / max(home_count, 1)
        report.away_per_possession = away_total / max(away_count, 1)
        report.total_possessions = len(possessions)
        return report
