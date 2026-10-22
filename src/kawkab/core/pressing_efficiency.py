"""Pressing Efficiency — trap-to-shot conversion, high press efficiency,
and press recovery attack analysis.

All methods are numpy-only and use game_constants pitch dimensions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kawkab.core.game_constants import GAME

PITCH_LENGTH = getattr(GAME, "PITCH_LENGTH_M", 105.0)
PITCH_WIDTH = getattr(GAME, "PITCH_WIDTH_M", 68.0)
PRESSING_TRAP_ZONE_BOUNDARY_PCT = getattr(GAME, "PRESSING_TRAP_ZONE_BOUNDARY_PCT", (0.25, 0.75))


class PressingEfficiencyAnalyzer:
    """Analyze pressing trap efficiency — conversion to shots and goals."""

    @staticmethod
    def _is_trap_event(event: dict) -> bool:
        etype = event.get("type", "")
        return etype in ("tackle", "interception", "foul")

    @staticmethod
    def _is_shot_event(event: dict) -> bool:
        return event.get("type", "") == "shot"

    @staticmethod
    def _is_goal_event(event: dict) -> bool:
        return event.get("type", "") == "shot" and event.get("is_goal", False)

    def compute_trap_to_shot_rate(
        self, events: list[dict[str, Any]]
    ) -> dict[str, dict[str, float]]:
        """For each pressing trap, determine what % led to a shot within 5 events.

        Returns dict like {team: {traps, shots_from_traps, goals_from_traps,
                                  conversion_rate}}
        """
        sorted_ev = sorted(events, key=lambda e: e.get("timestamp", 0.0))
        n = len(sorted_ev)
        result: dict[str, dict[str, float]] = {}

        for team in ("home", "away"):
            team_events = [e for e in sorted_ev if e.get("team") == team]
            traps = 0
            shots_from_traps = 0
            goals_from_traps = 0
            used_shot_indices: set[int] = set()

            _ = np.array(
                [
                    [
                        # dict.get(k, default) returns None when the key
                        # exists with a NULL value (storage's json_extract
                        # emits x/y=None for events without spatial data) —
                        # explicit None-check, same fix as
                        # tactical_shape_analyzer._analyze_window.
                        float(e["x"] if e.get("x") is not None else e.get("start_x", 0)),
                        float(e["y"] if e.get("y") is not None else e.get("start_y", 0)),
                    ]
                    for e in team_events
                ],
                dtype=np.float64,
            )
            trap_mask = np.array([self._is_trap_event(e) for e in team_events], dtype=bool)
            trap_indices = np.where(trap_mask)[0]

            for ti in trap_indices:
                traps += 1
                idx_in_sorted = sorted_ev.index(team_events[ti])
                window = sorted_ev[idx_in_sorted + 1 : min(n, idx_in_sorted + 6)]
                for wi, we in enumerate(window):
                    actual_idx = idx_in_sorted + 1 + wi
                    if self._is_shot_event(we) and actual_idx not in used_shot_indices:
                        shots_from_traps += 1
                        used_shot_indices.add(actual_idx)
                        if self._is_goal_event(we):
                            goals_from_traps += 1
                        break

            result[team] = {
                "traps": float(traps),
                "shots_from_traps": float(shots_from_traps),
                "goals_from_traps": float(goals_from_traps),
                "conversion_rate": (shots_from_traps / traps if traps > 0 else 0.0),
            }

        return result

    def compute_trap_to_goal_rate(
        self, events: list[dict[str, Any]]
    ) -> dict[str, dict[str, float]]:
        """Similar to trap-to-shot but specifically for goals."""
        sorted_ev = sorted(events, key=lambda e: e.get("timestamp", 0.0))
        n = len(sorted_ev)
        result: dict[str, dict[str, float]] = {}

        for team in ("home", "away"):
            team_events = [e for e in sorted_ev if e.get("team") == team]
            traps = 0
            goals_from_traps = 0
            used_goal_indices: set[int] = set()

            for ev in team_events:
                if not self._is_trap_event(ev):
                    continue
                traps += 1
                idx = sorted_ev.index(ev)
                window = sorted_ev[idx + 1 : min(n, idx + 6)]
                for wi, we in enumerate(window):
                    actual_idx = idx + 1 + wi
                    if self._is_goal_event(we) and actual_idx not in used_goal_indices:
                        goals_from_traps += 1
                        used_goal_indices.add(actual_idx)
                        break

            result[team] = {
                "traps": float(traps),
                "goals_from_traps": float(goals_from_traps),
                "conversion_rate": (goals_from_traps / traps if traps > 0 else 0.0),
            }

        return result

    def analyze_high_press_efficiency(self, events: list[dict[str, Any]]) -> dict[str, float]:
        """High press efficiency index.

        Traps in attacking third / shots conceded after losing press.
        Higher ratio = better press.
        """
        attacking_third_x = PITCH_LENGTH * 2.0 / 3.0
        result: dict[str, float] = {}

        for team in ("home", "away"):
            traps_in_attacking = 0
            shots_after_press_loss = 0

            sorted_ev = sorted(events, key=lambda e: e.get("timestamp", 0.0))
            n = len(sorted_ev)

            for i, ev in enumerate(sorted_ev):
                if ev.get("team") != team:
                    continue
                if not self._is_trap_event(ev):
                    continue
                # dict.get(k, default) returns None when the key exists
                # with a NULL value (storage's json_extract emits x=None
                # for events without spatial data) -- explicit None-check,
                # same fix as compute_trap_to_shot_rate above.
                x = float(ev["x"] if ev.get("x") is not None else ev.get("start_x", 0))
                if x < attacking_third_x:
                    continue
                traps_in_attacking += 1

                # Check if opponent gets a shot within next 5 events
                opponent = "away" if team == "home" else "home"
                for j in range(i + 1, min(n, i + 6)):
                    later = sorted_ev[j]
                    if later.get("team") == opponent and self._is_shot_event(later):
                        shots_after_press_loss += 1
                        break

            efficiency = traps_in_attacking / max(shots_after_press_loss, 1)
            result[team] = round(efficiency, 4)

        return result

    def compute_press_recovery_attack(
        self, events: list[dict[str, Any]]
    ) -> dict[str, dict[str, float]]:
        """After a successful press recovery (ball won within 5s of
        opponent receiving), how often does it lead to a chance.

        Returns dict with {team: {recoveries, chances_created, xg_created,
                                  avg_time_to_shot}}
        """
        sorted_ev = sorted(events, key=lambda e: e.get("timestamp", 0.0))
        n = len(sorted_ev)
        result: dict[str, dict[str, float]] = {}

        for team in ("home", "away"):
            recoveries = 0
            chances_created = 0
            xg_created = 0.0
            times_to_shot: list[float] = []

            for i, ev in enumerate(sorted_ev):
                if ev.get("team") != team:
                    continue
                if not self._is_trap_event(ev):
                    continue

                # Check if ball was won within 5s of opponent touch
                # (simplified: next own-team action that is pass/carry/dribble)
                ts = float(ev.get("timestamp", 0.0))
                won_ball = False
                for j in range(i + 1, min(n, i + 8)):
                    later = sorted_ev[j]
                    if later.get("team") == team and later.get("type") in (
                        "pass",
                        "carry",
                        "dribble",
                    ):
                        later_ts = float(later.get("timestamp", ts))
                        if later_ts - ts <= 5.0:
                            won_ball = True
                            break

                if not won_ball:
                    continue

                recoveries += 1

                # Check if this leads to a chance (shot) within next 8 events
                for j in range(i + 1, min(n, i + 10)):
                    later = sorted_ev[j]
                    if later.get("team") == team and self._is_shot_event(later):
                        chances_created += 1
                        xg_created += float(later.get("xg", 0.0))
                        shot_ts = float(later.get("timestamp", ts))
                        times_to_shot.append(shot_ts - ts)
                        break

            result[team] = {
                "recoveries": float(recoveries),
                "chances_created": float(chances_created),
                "xg_created": round(xg_created, 4),
                "avg_time_to_shot": (
                    round(float(np.mean(times_to_shot)), 2) if times_to_shot else 0.0
                ),
            }

        return result
