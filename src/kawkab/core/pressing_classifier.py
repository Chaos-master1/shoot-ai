"""Pressing System Classification — identifies a team's pressing approach.

Classifies pressing type (high block / mid block / low block),
detects man-oriented vs zonal pressing, and identifies trigger pressing
moments from event data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from kawkab.core.game_constants import GAME

PITCH_LENGTH = GAME.PITCH_LENGTH_M
PITCH_WIDTH = GAME.PITCH_WIDTH_M


def _avg_defensive_line_x(events: list[dict[str, Any]], team: str) -> float:
    """Estimate average defensive line x-coordinate from event data."""
    defending = [e for e in events if e.get("team") != team]
    positions: list[float] = []
    for e in defending:
        x = e.get("start_x") or e.get("x")
        if x is not None and isinstance(x, (int, float)):
            positions.append(float(x))
    if not positions:
        return PITCH_LENGTH * 0.35
    return sum(positions) / len(positions)


def _classify_block_type(avg_def_line_x: float) -> str:
    """Classify block type based on defensive line position."""
    if avg_def_line_x >= PITCH_LENGTH * 0.55:
        return "high_block"
    if avg_def_line_x >= PITCH_LENGTH * 0.35:
        return "mid_block"
    return "low_block"


def _detect_man_or_zonal(events: list[dict[str, Any]], team: str) -> str:
    """Heuristic to detect man-oriented vs zonal pressing."""
    defending = [e for e in events if e.get("team") != team]
    tackles = [e for e in defending if e.get("type") in ("tackle", "interception")]
    if len(tackles) < 5:
        return "unknown"
    spread_x = [e.get("start_x", 0) for e in tackles if e.get("start_x") is not None]
    spread_y = [e.get("start_y", 0) for e in tackles if e.get("start_y") is not None]
    if not spread_x or not spread_y:
        return "unknown"
    var_x = sum((x - sum(spread_x) / len(spread_x)) ** 2 for x in spread_x) / len(spread_x)
    var_y = sum((y - sum(spread_y) / len(spread_y)) ** 2 for y in spread_y) / len(spread_y)
    coverage = math.sqrt(var_x * var_y)
    # High variance across pitch = man-oriented tracking, low = zonal
    if coverage > PITCH_WIDTH * 0.15:
        return "man_oriented"
    return "zonal"


def _detect_trigger_pressing_moments(
    events: list[dict[str, Any]],
    team: str,
    trigger_window_s: float = 3.0,
) -> list[dict[str, Any]]:
    """Detect trigger pressing moments — specific opponent actions that trigger press."""
    triggers: list[dict[str, Any]] = []
    sorted_ev = sorted(events, key=lambda e: e.get("timestamp", 0.0))
    n = len(sorted_ev)
    for i in range(n):
        ev = sorted_ev[i]
        if ev.get("team") == team:
            continue
        etype = ev.get("type", "")
        is_trigger = etype in ("back_pass", "poor_control", "slow_pass")
        if not is_trigger:
            # Check if under high pressure (multiple defenders within threshold)
            is_trigger = bool(ev.get("under_pressure"))
        if is_trigger:
            # Look for defensive action within window
            ts = ev.get("timestamp", 0.0)
            pressed = False
            for j in range(i + 1, min(i + 10, n)):
                next_ev = sorted_ev[j]
                if next_ev.get("timestamp", 0.0) - ts > trigger_window_s:
                    break
                if next_ev.get("team") != team and next_ev.get("type") in (
                    "tackle",
                    "interception",
                    "foul",
                ):
                    pressed = True
                    triggers.append(
                        {
                            "trigger_time": round(ts, 1),
                            "trigger_event": etype,
                            "response_time_s": round(next_ev.get("timestamp", 0.0) - ts, 1),
                            "regained_possession": next_ev.get("type")
                            in ("tackle", "interception"),
                        }
                    )
                    break
            if not pressed:
                triggers.append(
                    {
                        "trigger_time": round(ts, 1),
                        "trigger_event": etype,
                        "response_time_s": None,
                        "regained_possession": False,
                    }
                )
    return triggers


@dataclass
class PressingSystemReport:
    team: str = "home"
    primary_block_type: str = "unknown"
    pressing_style: str = "unknown"  # man_oriented / zonal / unknown
    trigger_count: int = 0
    trigger_success_rate: float = 0.0
    avg_press_intensity: float = 0.0
    ppda: float = 0.0
    triggers: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "team": self.team,
            "primary_block_type": self.primary_block_type,
            "pressing_style": self.pressing_style,
            "trigger_count": self.trigger_count,
            "trigger_success_rate": round(self.trigger_success_rate, 2),
            "avg_press_intensity": round(self.avg_press_intensity, 2),
            "ppda": round(self.ppda, 1),
            "triggers": self.triggers[:20],
        }


def classify_pressing_system(
    events: list[dict[str, Any]], team: str = "home"
) -> PressingSystemReport:
    """Full pressing system classification for a team."""
    team_events = [e for e in events if e.get("team") == team]
    opp_events = [e for e in events if e.get("team") != team]

    # Block type
    avg_def_line = _avg_defensive_line_x(events, team)
    block_type = _classify_block_type(avg_def_line)

    # Man vs zonal
    style = _detect_man_or_zonal(events, team)

    # Trigger pressing
    triggers = _detect_trigger_pressing_moments(events, team)
    trigger_count = len(triggers)
    successful_triggers = sum(1 for t in triggers if t.get("regained_possession"))
    trigger_success_rate = (successful_triggers / trigger_count) if trigger_count > 0 else 0.0

    # PPDA (passes per defensive action)
    opp_passes = [e for e in opp_events if e.get("type") == "pass" and e.get("completed", True)]
    def_actions = [e for e in team_events if e.get("type") in ("tackle", "interception", "foul")]
    ppda = (len(opp_passes) / len(def_actions)) if def_actions else 0.0

    # Press intensity (defensive actions per minute)
    total_minutes = 0.0
    if events:
        ts_max = max(e.get("timestamp", 0) for e in events)
        ts_min = min(e.get("timestamp", 0) for e in events)
        total_minutes = max(ts_max - ts_min, 1.0) / 60.0
    intensity = (len(def_actions) / total_minutes) if total_minutes > 0 else 0.0

    return PressingSystemReport(
        team=team,
        primary_block_type=block_type,
        pressing_style=style,
        trigger_count=trigger_count,
        trigger_success_rate=trigger_success_rate,
        avg_press_intensity=intensity,
        ppda=ppda,
        triggers=triggers,
    )
