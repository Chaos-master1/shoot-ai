"""Pressing Trap Detection — identifies zones where a defending team
deliberately funnels opposition play to win the ball back.

Uses event-density analysis, trigger classification, and success-rate
computation — all via numpy + stdlib only.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kawkab.core.game_constants import GAME

PITCH_LENGTH = GAME.PITCH_LENGTH_M
PITCH_WIDTH = GAME.PITCH_WIDTH_M


# ── Zone definitions ──────────────────────────────────────────────────────

ZONE_DEFS: list[dict[str, Any]] = [
    {"name": "left_defensive", "x": (0.0, PITCH_LENGTH * 0.33), "y": (0.0, PITCH_WIDTH * 0.50)},
    {
        "name": "left_mid",
        "x": (PITCH_LENGTH * 0.33, PITCH_LENGTH * 0.67),
        "y": (0.0, PITCH_WIDTH * 0.50),
    },
    {
        "name": "left_attacking",
        "x": (PITCH_LENGTH * 0.67, PITCH_LENGTH),
        "y": (0.0, PITCH_WIDTH * 0.50),
    },
    {
        "name": "right_defensive",
        "x": (0.0, PITCH_LENGTH * 0.33),
        "y": (PITCH_WIDTH * 0.50, PITCH_WIDTH),
    },
    {
        "name": "right_mid",
        "x": (PITCH_LENGTH * 0.33, PITCH_LENGTH * 0.67),
        "y": (PITCH_WIDTH * 0.50, PITCH_WIDTH),
    },
    {
        "name": "right_attacking",
        "x": (PITCH_LENGTH * 0.67, PITCH_LENGTH),
        "y": (PITCH_WIDTH * 0.50, PITCH_WIDTH),
    },
    {
        "name": "central_defensive",
        "x": (0.0, PITCH_LENGTH * 0.33),
        "y": (
            PITCH_WIDTH * GAME.PRESSING_TRAP_ZONE_BOUNDARY_PCT[0],
            PITCH_WIDTH * GAME.PRESSING_TRAP_ZONE_BOUNDARY_PCT[1],
        ),
    },
    {
        "name": "central_mid",
        "x": (PITCH_LENGTH * 0.33, PITCH_LENGTH * 0.67),
        "y": (
            PITCH_WIDTH * GAME.PRESSING_TRAP_ZONE_BOUNDARY_PCT[0],
            PITCH_WIDTH * GAME.PRESSING_TRAP_ZONE_BOUNDARY_PCT[1],
        ),
    },
    {
        "name": "central_attacking",
        "x": (PITCH_LENGTH * 0.67, PITCH_LENGTH),
        "y": (
            PITCH_WIDTH * GAME.PRESSING_TRAP_ZONE_BOUNDARY_PCT[0],
            PITCH_WIDTH * GAME.PRESSING_TRAP_ZONE_BOUNDARY_PCT[1],
        ),
    },
]

# Overlap: central zones share some territory with wide zones.
# Wide zones get priority for classification (central is fallback).


@dataclass
class PressingTrap:
    zone_name: str
    zone_x_range: tuple[float, float]
    zone_y_range: tuple[float, float]
    trigger_event_type: str = ""
    defensive_actions_in_zone: int = 0
    opponent_passes_into_zone: int = 0
    regain_possession_count: int = 0
    success_rate: float = 0.0
    intensity: float = 0.0
    trap_rating: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone_name": self.zone_name,
            "zone_x_range": list(self.zone_x_range),
            "zone_y_range": list(self.zone_y_range),
            "trigger_event_type": self.trigger_event_type,
            "defensive_actions_in_zone": self.defensive_actions_in_zone,
            "opponent_passes_into_zone": self.opponent_passes_into_zone,
            "regain_possession_count": self.regain_possession_count,
            "success_rate": round(self.success_rate, 3),
            "intensity": round(self.intensity, 3),
            "trap_rating": round(self.trap_rating, 3),
        }


@dataclass
class PressingTrapReport:
    team: str
    traps: list[PressingTrap] = field(default_factory=list)
    total_traps: int = 0
    overall_success_rate: float = 0.0
    most_common_trigger: str = ""
    dangerous_zones: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "team": self.team,
            "traps": [t.to_dict() for t in self.traps],
            "total_traps": self.total_traps,
            "overall_success_rate": round(self.overall_success_rate, 3),
            "most_common_trigger": self.most_common_trigger,
            "dangerous_zones": self.dangerous_zones,
        }


# ── Helpers ───────────────────────────────────────────────────────────────


def _classify_trap_zone(
    x: float, y: float, pitch_length: float = PITCH_LENGTH, pitch_width: float = PITCH_WIDTH
) -> str:
    """Classify a position into one of 9 zone names.

    Wide zones take priority; central is fallback when the point falls
    in the middle 50 % of pitch width.
    """
    _ = pitch_width / 3.0

    # Determine longitudinal third
    if x < pitch_length * 0.33:
        x_part = "defensive"
    elif x < pitch_length * 0.67:
        x_part = "mid"
    else:
        x_part = "attacking"

    # Determine lateral lane
    left_bound, right_bound = GAME.PRESSING_TRAP_ZONE_BOUNDARY_PCT
    if y < pitch_width * left_bound:
        y_part = "left"
    elif y > pitch_width * right_bound:
        y_part = "right"
    else:
        # Fall back to central
        return f"central_{x_part}"

    return f"{y_part}_{x_part}"


def _compute_trap_rating(actions: int, passes_into: int, regains: int) -> float:
    """Compute overall trap quality rating (0–1).

    Combines raw counts into a single score:
      - Higher defensive action density → better
      - More opponent passes funnelled → better
      - Higher regain rate → significantly better
    """
    if actions == 0:
        return 0.0

    action_score = min(1.0, actions / 20.0)
    funnel_score = min(1.0, passes_into / max(1, actions) * 2.0)
    regain_rate = regains / max(1, actions)
    regain_score = regain_rate**0.6  # Diminishing returns beyond ~60%

    return round(0.25 * action_score + 0.30 * funnel_score + 0.45 * regain_score, 4)


def _find_trigger_events(
    events: list[dict],
    zone_name: str,
    team: str,
    pitch_length: float = PITCH_LENGTH,
    pitch_width: float = PITCH_WIDTH,
) -> tuple[str, int]:
    """Find what type of event typically triggers traps in this zone.

    Looks at events in the 5 seconds *before* each defensive action in the
    zone and classifies the most common preceding event type.
    """
    trigger_counts: dict[str, int] = defaultdict(int)
    sorted_ev = sorted(events, key=lambda e: e.get("timestamp", 0.0))
    _ = len(sorted_ev)

    for i, ev in enumerate(sorted_ev):
        if ev.get("type") not in ("tackle", "interception", "foul"):
            continue
        if ev.get("team") != team:
            continue
        x = ev.get("x", ev.get("start_x"))
        y = ev.get("y", ev.get("start_y"))
        if x is None or y is None:
            continue
        if _classify_trap_zone(float(x), float(y), pitch_length, pitch_width) != zone_name:
            continue

        ts = ev.get("timestamp", 0.0)
        window = 5.0

        # Scan backwards for the most recent opponent event
        for j in range(i - 1, max(-1, i - 60), -1):
            if j < 0:
                break
            prev = sorted_ev[j]
            if prev.get("team") == team:
                continue  # same team, not trigger
            pts = prev.get("timestamp", ts)
            if ts - pts > window:
                break

            ptype = prev.get("type", "")
            if ptype == "pass":
                # Sub-classify pass
                end_x = prev.get("end_x")
                end_y = prev.get("end_y")
                start_x = prev.get("start_x")
                if end_x is not None and start_x is not None:
                    dx = float(end_x) - float(start_x)
                    if abs(dx) < 5.0:
                        trigger_counts["backward_pass"] += 1
                    elif end_y is not None and (
                        float(end_y) < pitch_width * GAME.PRESSING_TRAP_ZONE_BOUNDARY_PCT[0]
                        or float(end_y) > pitch_width * GAME.PRESSING_TRAP_ZONE_BOUNDARY_PCT[1]
                    ):
                        trigger_counts["pass_to_wide"] += 1
                    else:
                        trigger_counts["pass_forward"] += 1
            elif ptype == "carry":
                trigger_counts["dribble_inside"] += 1
            elif ptype == "shot":
                trigger_counts["shot_blocked"] += 1
            else:
                trigger_counts[f"other_{ptype}"] += 1

            break  # Only the immediate preceding opponent event

    if not trigger_counts:
        return ("unknown", 0)

    best_trigger = max(trigger_counts, key=trigger_counts.get)  # type: ignore[arg-type,return-value]
    return (best_trigger, trigger_counts[best_trigger])


# ── Zone geometry helpers ─────────────────────────────────────────────────


def _x_mid(x_range: tuple[float, float]) -> float:
    return (x_range[0] + x_range[1]) / 2.0


def _y_mid(y_range: tuple[float, float]) -> float:
    return (y_range[0] + y_range[1]) / 2.0


def _in_zone(x: float, y: float, zone: dict[str, Any], pitch_width: float = PITCH_WIDTH) -> bool:
    """Check if (x, y) falls within a zone definition.

    Central zones are narrower (middle 50 % width); wide zones are
    left/right 50 % respectively.
    """
    xr = zone["x"]
    yr = zone["y"]
    if not (xr[0] <= x <= xr[1] and yr[0] <= y <= yr[1]):
        return False

    name: str = zone["name"]
    if name.startswith("central_"):
        # Central zones: only assign if strictly in the middle zone
        left_bound, right_bound = GAME.PRESSING_TRAP_ZONE_BOUNDARY_PCT
        return pitch_width * left_bound <= y <= pitch_width * right_bound
    return True


# ── Main detection ────────────────────────────────────────────────────────


def detect_pressing_traps(
    events: list[dict],
    team: str,
    pitch_length: float = PITCH_LENGTH,
    pitch_width: float = PITCH_WIDTH,
) -> PressingTrapReport:
    """Detect pressing traps from event data.

    Parameters
    ----------
    events : list[dict]
        Chronological list of events. Each dict should contain:
          - ``type`` : str — event type (pass, tackle, interception, …)
          - ``team`` : str — team identifier
          - ``timestamp`` : float — seconds from kick-off
          - ``x``, ``y`` or ``start_x``, ``start_y`` — pitch coordinates
          - (for passes) ``end_x``, ``end_y``
    team : str
        The defending / pressing team to analyse.
    pitch_length, pitch_width : float
        Pitch dimensions in metres (default 105 × 68).

    Returns
    -------
    PressingTrapReport
    """
    sorted_ev = sorted(events, key=lambda e: e.get("timestamp", 0.0))
    n = len(sorted_ev)

    # ── 1.  Count defensive actions & opponent passes per zone ──
    zone_actions: dict[str, int] = defaultdict(int)
    zone_opp_passes: dict[str, int] = defaultdict(int)
    zone_timestamps: dict[str, list[float]] = defaultdict(list)
    zone_regains: dict[str, int] = defaultdict(int)

    for i, ev in enumerate(sorted_ev):
        etype = ev.get("type", "")
        eteam = ev.get("team", "")

        if eteam != team or etype not in ("tackle", "interception", "foul"):
            continue

        x = ev.get("x", ev.get("start_x"))
        y = ev.get("y", ev.get("start_y"))
        if x is None or y is None:
            continue

        z = _classify_trap_zone(float(x), float(y), pitch_length, pitch_width)
        zone_actions[z] += 1
        ts = ev.get("timestamp", 0.0)
        zone_timestamps[z].append(ts)

        # ── 1b.  Count opponent passes into this zone in preceding 5 s ──
        opp_passes = 0
        for j in range(i - 1, max(-1, i - 60), -1):
            if j < 0:
                break
            prev = sorted_ev[j]
            if prev.get("team") == team:
                continue
            pts = prev.get("timestamp", ts)
            if ts - pts > 5.0:
                break
            if prev.get("type") == "pass":
                ex = prev.get("end_x")
                ey = prev.get("end_y")
                if (
                    ex is not None
                    and ey is not None
                    and _classify_trap_zone(float(ex), float(ey), pitch_length, pitch_width) == z
                ):
                    opp_passes += 1
        zone_opp_passes[z] += opp_passes

        # ── 1c.  Check if team regained possession within 3 events ──
        for k in range(i + 1, min(n, i + 8)):
            later = sorted_ev[k]
            if later.get("team") == team and later.get("type") in (
                "pass",
                "carry",
                "shot",
                "dribble",
            ):
                zone_regains[z] += 1
                break

    if not zone_actions:
        return PressingTrapReport(team=team)

    # ── 2.  Build trap objects ──
    traps: list[PressingTrap] = []
    for zd in ZONE_DEFS:
        zname = zd["name"]
        actions = zone_actions.get(zname, 0)
        if actions < 2:
            continue  # noise floor — at least 2 actions to be a trap

        opp_passes = zone_opp_passes.get(zname, 0)
        regains = zone_regains.get(zname, 0)
        success_rate = regains / max(1, actions)

        # Intensity: actions per minute based on time span
        tss = zone_timestamps.get(zname, [])
        if len(tss) >= 2:
            span_min = (max(tss) - min(tss)) / 60.0
            intensity = actions / max(0.1, span_min)
        else:
            intensity = float(actions)

        trap_rating = _compute_trap_rating(actions, opp_passes, regains)

        # Classify trigger
        trigger_type, _ = _find_trigger_events(
            sorted_ev,
            zname,
            team,
            pitch_length,
            pitch_width,
        )

        traps.append(
            PressingTrap(
                zone_name=zname,
                zone_x_range=zd["x"],
                zone_y_range=zd["y"],
                trigger_event_type=trigger_type,
                defensive_actions_in_zone=actions,
                opponent_passes_into_zone=opp_passes,
                regain_possession_count=regains,
                success_rate=success_rate,
                intensity=intensity,
                trap_rating=trap_rating,
            )
        )

    traps.sort(key=lambda t: t.trap_rating, reverse=True)

    # ── 3.  Aggregate report ──
    if not traps:
        return PressingTrapReport(team=team)

    all_rates = [t.success_rate for t in traps if t.defensive_actions_in_zone > 0]
    all_rates = [r for r in all_rates if np.isfinite(r)]
    overall_success = float(np.mean(all_rates)) if all_rates else 0.0

    trigger_counts: dict[str, int] = defaultdict(int)
    for t in traps:
        trigger_counts[t.trigger_event_type] += 1
    most_common = max(trigger_counts, key=trigger_counts.get) if trigger_counts else ""  # type: ignore[arg-type,return-value]

    # Dangerous zones: traps where success rate is below 30 %
    dangerous = [t.zone_name for t in traps if t.success_rate < 0.30]

    return PressingTrapReport(
        team=team,
        traps=traps,
        total_traps=len(traps),
        overall_success_rate=overall_success,
        most_common_trigger=most_common,
        dangerous_zones=dangerous,
    )
