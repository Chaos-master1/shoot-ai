"""Ball Recovery Analysis — classify and analyze ball recovery events.

All numpy-only, no pandas/scipy/sklearn.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from kawkab.core.game_constants import GAME

PITCH_LENGTH = GAME.PITCH_LENGTH_M
PITCH_WIDTH = GAME.PITCH_WIDTH_M
NUM_ZONES = 5
ZONE_WIDTH = PITCH_LENGTH / NUM_ZONES
ZONE_HEIGHT = PITCH_WIDTH / NUM_ZONES
RECOVERY_EVENT_TYPES = {"interception", "tackle", "loose_ball", "goal_kick", "clearance"}


def _to_zone(x: float, y: float) -> tuple[int, int]:
    # x/y=None (events without coordinates — dict.get(key, default) returns
    # None when the key exists with a NULL value) → pitch-center fallback,
    # same fix as epv._to_zone / pressing_efficiency / tactical_shape.
    if x is None or y is None:
        x = PITCH_LENGTH / 2.0
        y = PITCH_WIDTH / 2.0
    zx = min(int(x / ZONE_WIDTH), NUM_ZONES - 1)
    zy = min(int(y / ZONE_HEIGHT), NUM_ZONES - 1)
    return (zx, zy)


def _zone_key(zx: int, zy: int) -> str:
    return f"{zx}_{zy}"


class BallRecoveryAnalyzer:
    """Analyze ball recoveries: classification, location, efficiency."""

    def classify_recovery(
        self,
        recovery_event: dict[str, Any],
        previous_events: list[dict[str, Any]],
    ) -> tuple[str, float, float]:
        ev_type = recovery_event.get("type", "")
        # x/y may be absent or present-but-NULL (no coordinates). Guard with
        # an isinstance check (isfinite(None) raises TypeError, which is what
        # killed real-match reports) and bind to float so every return site
        # satisfies the (str, float, float) contract.
        x = recovery_event.get("x")
        y = recovery_event.get("y")
        x = (
            PITCH_LENGTH / 2
            if not isinstance(x, (int, float)) or not math.isfinite(x)
            else float(x)
        )
        y = PITCH_WIDTH / 2 if not isinstance(y, (int, float)) or not math.isfinite(y) else float(y)
        _ = recovery_event.get("team", "home")

        if ev_type == "interception":
            return ("interception", x, y)

        if ev_type == "tackle":
            return ("tackle", x, y)

        if ev_type in ("goal_kick",):
            return ("goal_kick", x, y)

        if ev_type == "clearance":
            return ("clearance", x, y)

        _ = {e.get("type", "") for e in previous_events[-5:]}
        if ev_type == "pass" and not recovery_event.get("completed", True):
            return ("loose_ball", x, y)

        if "loose_ball" in ev_type or ev_type == "fifty_fifty":
            return ("loose_ball", x, y)

        if ev_type == "ball_recovery":
            return ("loose_ball", x, y)

        return ("loose_ball", x, y)

    def compute_recovery_locations(
        self,
        events: list[dict[str, Any]],
        team: str,
    ) -> dict[str, int]:
        zones: dict[str, int] = defaultdict(int)
        for ev in events:
            ev_team = ev.get("team", "")
            ev_type = ev.get("type", "")
            if ev_team == team and ev_type in RECOVERY_EVENT_TYPES:
                x = ev.get("x", PITCH_LENGTH / 2)
                y = ev.get("y", PITCH_WIDTH / 2)
                zx, zy = _to_zone(x, y)
                zones[_zone_key(zx, zy)] += 1
        return dict(zones)

    def analyze_recoveries(
        self,
        events: list[dict[str, Any]],
        team: str,
    ) -> dict[str, Any]:
        recoveries = [
            e for e in events if e.get("team") == team and e.get("type") in RECOVERY_EVENT_TYPES
        ]
        total = len(recoveries)

        by_type: dict[str, int] = defaultdict(int)
        for ev in recoveries:
            prev_idx = events.index(ev)
            prev = events[max(0, prev_idx - 5) : prev_idx]
            rtype, _, _ = self.classify_recovery(ev, prev)
            by_type[rtype] += 1

        by_zone: dict[str, int] = defaultdict(int)
        for ev in recoveries:
            x = ev.get("x", PITCH_LENGTH / 2)
            y = ev.get("y", PITCH_WIDTH / 2)
            zx, zy = _to_zone(x, y)
            by_zone[_zone_key(zx, zy)] += 1

        leading_to_shot = 0
        leading_to_goal = 0
        total_time = 0.0
        shot_count = 0

        for i, ev in enumerate(events):
            if ev.get("team") != team or ev.get("type") not in RECOVERY_EVENT_TYPES:
                continue
            for j in range(i + 1, min(i + 6, len(events))):
                later = events[j]
                if later.get("type") == "shot":
                    leading_to_shot += 1
                    if later.get("is_goal"):
                        leading_to_goal += 1
                    dt = later.get("timestamp", 0) - ev.get("timestamp", 0)
                    if dt > 0:
                        total_time += dt
                        shot_count += 1
                    break

        avg_time_to_shot = total_time / max(shot_count, 1)

        return {
            "total_recoveries": total,
            "recoveries_by_type": dict(by_type),
            "recoveries_by_zone": dict(by_zone),
            "recoveries_leading_to_shot": leading_to_shot,
            "recoveries_leading_to_goal": leading_to_goal,
            "avg_time_to_shot_seconds": round(avg_time_to_shot, 2),
        }

    def detect_counter_press(
        self,
        event: dict[str, Any],
        events: list[dict[str, Any]],
    ) -> tuple[bool, float, str]:
        if event.get("type") not in RECOVERY_EVENT_TYPES:
            return (False, 0.0, "no_recovery")

        x = event.get("x", PITCH_LENGTH / 2)
        event_time = event.get("timestamp", 0)
        team = event.get("team", "home")

        is_attacking_half = x > PITCH_LENGTH * 0.5
        if not is_attacking_half:
            return (False, 0.0, "not_attacking_half")

        counter_team = "away" if team == "home" else "home"
        pressure_end = event_time + 2.0
        pressure_events = [
            e
            for e in events
            if e.get("timestamp", 0) > event_time
            and e.get("timestamp", 0) <= pressure_end
            and e.get("team") == counter_team
            and e.get("type") in ("tackle", "interception", "pressure", "foul")
        ]

        if not pressure_events:
            return (False, 0.0, "no_pressure")

        pressure_duration = pressure_events[-1].get("timestamp", event_time) - event_time
        last_pressure = pressure_events[-1]
        result = last_pressure.get("type", "pressure")

        return (True, round(pressure_duration, 2), result)

    def compute_recovery_efficiency(
        self,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        teams = {e.get("team", "") for e in events if e.get("team")}
        result: dict[str, Any] = {}

        for team in teams:
            team_events = [e for e in events if e.get("team") == team]
            recoveries = [e for e in team_events if e.get("type") in RECOVERY_EVENT_TYPES]
            total_rec = len(recoveries)

            timestamps = [e.get("timestamp", 0) for e in team_events]
            match_duration = max(timestamps) - min(timestamps) if len(timestamps) >= 2 else 90.0
            match_minutes = max(match_duration / 60.0, 1.0)

            recoveries_per_min = total_rec / match_minutes

            attacking_third_recoveries = sum(
                1 for e in recoveries if e.get("x", 0) > PITCH_LENGTH * (2.0 / 3.0)
            )
            attacking_third_pct = (attacking_third_recoveries / max(total_rec, 1)) * 100

            recovery_to_goal = 0
            for i, ev in enumerate(team_events):
                if ev.get("type") not in RECOVERY_EVENT_TYPES:
                    continue
                for j in range(i + 1, min(i + 6, len(team_events))):
                    later = team_events[j]
                    if later.get("type") == "shot" and later.get("is_goal"):
                        recovery_to_goal += 1
                        break

            conversion_rate = (recovery_to_goal / max(total_rec, 1)) * 100

            result[team] = {
                "recoveries": total_rec,
                "recoveries_per_minute": round(recoveries_per_min, 3),
                "attacking_third_recoveries": attacking_third_recoveries,
                "attacking_third_pct": round(attacking_third_pct, 1),
                "recovery_to_goal_conversion": round(conversion_rate, 2),
            }

        return result
