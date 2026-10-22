"""Phase transition analysis — counter-attack detection and outcome tracking.

Detects counter-attack moments: when a team regains possession and
rapidly transitions to attack. Measures transition speed, duration,
and outcome (shot created, chance, or reset).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from kawkab.core.coords import FINAL_THIRD_X, STANDARD_PITCH, PitchConfig


@dataclass
class PhaseTransition:
    timestamp: float = 0.0
    team: str = "home"
    transition_type: str = "counter_attack"  # counter_attack, organized, gegenpress
    start_x: float = 0.0
    start_y: float = 34.0
    duration_s: float = 0.0
    speed_mps: float = 0.0
    outcome: str = "ongoing"  # shot, goal, turnover, foul, out_of_play
    ended_in_final_third: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 1),
            "team": self.team,
            "transition_type": self.transition_type,
            "start_x": round(self.start_x, 1),
            "duration_s": round(self.duration_s, 1),
            "speed_mps": round(self.speed_mps, 2),
            "outcome": self.outcome,
            "ended_in_final_third": self.ended_in_final_third,
        }


@dataclass
class TransitionReport:
    home_transitions: int = 0
    away_transitions: int = 0
    home_counter_attacks: int = 0
    away_counter_attacks: int = 0
    home_shot_conversion: float = 0.0
    away_shot_conversion: float = 0.0
    home_avg_speed_mps: float = 0.0
    away_avg_speed_mps: float = 0.0
    transitions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "home_transitions": self.home_transitions,
            "away_transitions": self.away_transitions,
            "home_counter_attacks": self.home_counter_attacks,
            "away_counter_attacks": self.away_counter_attacks,
            "home_shot_conversion": round(self.home_shot_conversion, 1),
            "away_shot_conversion": round(self.away_shot_conversion, 1),
            "home_avg_speed_mps": round(self.home_avg_speed_mps, 2),
            "away_avg_speed_mps": round(self.away_avg_speed_mps, 2),
            "total_transitions": self.home_transitions + self.away_transitions,
        }


def detect_transitions(
    events: list[dict[str, Any]],
    possession_changes: list[dict[str, Any]] | None = None,
    pitch: PitchConfig = STANDARD_PITCH,
) -> TransitionReport:
    """Detect phase transitions from possession changes and event sequences.

    A transition is triggered when possession changes. It's classified as:
    - counter_attack: ball moves >20m toward goal in <10s
    - gegenpress: transition starts in attacking half (start_x > 70m)
    - organized: anything else

    Args:
        events: List of event dicts sorted by timestamp.
        possession_changes: Optional pre-computed possession change markers.
        pitch_length: Pitch length in meters.

    Returns:
        TransitionReport with per-team statistics.
    """
    if not events:
        return TransitionReport()

    sorted_events = sorted(events, key=lambda e: e.get("timestamp", 0))
    final_third_x = FINAL_THIRD_X

    transitions: list[PhaseTransition] = []
    home_count = 0
    away_count = 0
    home_counter = 0
    away_counter = 0
    home_shots_from_transition = 0
    away_shots_from_transition = 0
    home_total_trans = 0
    away_total_trans = 0
    home_speed_sum = 0.0
    away_speed_sum = 0.0

    # Detect possession changes: consecutive events from different teams
    prev_team = None
    prev_ts = 0.0
    prev_x = 52.5
    prev_y = 34.0

    for ev in sorted_events:
        team = ev.get("team")
        ts = ev.get("timestamp", 0)
        etype = ev.get("type", "")
        if team not in ("home", "away"):
            continue

        # Possession change detected
        if prev_team is not None and team != prev_team:
            dx = ev.get("start_x", 52.5) - prev_x
            dy = ev.get("start_y", 34.0) - prev_y
            dt = max(0.01, ts - prev_ts)
            distance = math.sqrt(dx * dx + dy * dy)
            speed = distance / dt

            # Classify transition type
            if speed > 2.0 and distance > 15.0 and dt < 12.0:
                ttype = "counter_attack"
                if prev_x > final_third_x:
                    ttype = "gegenpress"
            else:
                ttype = "organized"

            start_x = prev_x
            ended_in_final_third = ev.get("end_x", 0) > final_third_x
            outcome = "ongoing"
            if etype == "shot":
                outcome = "goal" if ev.get("is_goal") else "shot"
            elif etype == "foul":
                outcome = "foul"
            elif not ev.get("completed", True):
                outcome = "turnover"

            trans = PhaseTransition(
                timestamp=ts,
                team=team,
                transition_type=ttype,
                start_x=start_x,
                start_y=prev_y,
                duration_s=dt,
                speed_mps=speed,
                outcome=outcome,
                ended_in_final_third=ended_in_final_third,
            )
            transitions.append(trans)

            if team == "home":
                home_count += 1
                home_total_trans += 1
                home_speed_sum += speed
                if ttype in ("counter_attack", "gegenpress"):
                    home_counter += 1
                if etype == "shot":
                    home_shots_from_transition += 1
            else:
                away_count += 1
                away_total_trans += 1
                away_speed_sum += speed
                if ttype in ("counter_attack", "gegenpress"):
                    away_counter += 1
                if etype == "shot":
                    away_shots_from_transition += 1

        prev_team = team
        prev_ts = ts
        prev_x = ev.get("start_x", 52.5)
        prev_y = ev.get("start_y", 34.0)

    return TransitionReport(
        home_transitions=home_count,
        away_transitions=away_count,
        home_counter_attacks=home_counter,
        away_counter_attacks=away_counter,
        home_shot_conversion=(home_shots_from_transition / max(home_total_trans, 1)) * 100.0,
        away_shot_conversion=(away_shots_from_transition / max(away_total_trans, 1)) * 100.0,
        home_avg_speed_mps=home_speed_sum / max(home_total_trans, 1),
        away_avg_speed_mps=away_speed_sum / max(away_total_trans, 1),
        transitions=[t.to_dict() for t in transitions],
    )
