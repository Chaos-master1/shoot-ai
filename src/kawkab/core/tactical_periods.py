"""Tactical period / game phase detection.

Splits a match into tactical phases: high press, low block,
settled possession, transition, etc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TacticalPhase:
    start_time: float
    end_time: float
    label: str  # "high_press", "low_block", "settled_possession", "transition", "unknown"
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start_time, 1),
            "end": round(self.end_time, 1),
            "label": self.label,
            "duration_s": round(self.duration_s, 1),
        }


@dataclass
class TacticalPeriodReport:
    phases: list[TacticalPhase] = field(default_factory=list)
    press_pct: float = 0.0
    low_block_pct: float = 0.0
    settled_possession_pct: float = 0.0
    transition_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "phases": [p.to_dict() for p in self.phases],
            "press_pct": round(self.press_pct, 1),
            "low_block_pct": round(self.low_block_pct, 1),
            "settled_possession_pct": round(self.settled_possession_pct, 1),
            "transition_pct": round(self.transition_pct, 1),
        }


def detect_tactical_periods(
    frame_data: list[dict[str, Any]],
    pitch_length: float = 105.0,
    min_phase_duration: float = 5.0,
) -> TacticalPeriodReport:
    """Detect tactical phases from frame-level tracking data.

    Args:
        frame_data: List of dicts with keys:
            - timestamp: float
            - possession: bool (True if home has possession)
            - home_positions: list of (x, y)
            - away_positions: list of (x, y)
            - ball_pos: (x, y) or None
        pitch_length: Pitch length in meters.
        min_phase_duration: Minimum phase duration in seconds.

    Returns:
        TacticalPeriodReport with phases and aggregate percentages.
    """
    if not frame_data:
        return TacticalPeriodReport()

    phases: list[tuple[float, str]] = []  # (timestamp, phase_label)

    window_size = max(
        1,
        int(
            30.0
            / max(1.0, frame_data[-1]["timestamp"] - frame_data[0]["timestamp"])
            * len(frame_data)
        ),
    )
    window_size = min(window_size, len(frame_data))

    for i, fdata in enumerate(frame_data):
        ts = fdata.get("timestamp", 0.0)
        possession = fdata.get("possession", True)

        home_pos = fdata.get("home_positions", [])
        away_pos = fdata.get("away_positions", [])

        # Compute defensive line height (avg x of defensive half)
        # When in possession, the AWAY team is defending (and vice versa)
        def_team_pos = away_pos if possession else home_pos

        def_line_x = 0.0
        if def_team_pos:
            def_line_x = sum(p[0] for p in def_team_pos) / len(def_team_pos)

        # Compute ball speed (rolling avg over small window)
        ball_speed = 0.0
        ball_pos = fdata.get("ball_pos")
        if ball_pos and i > 0:
            prev = frame_data[i - 1].get("ball_pos")
            if prev:
                dx = ball_pos[0] - prev[0]
                dy = ball_pos[1] - prev[1]
                dt = ts - frame_data[i - 1].get("timestamp", ts)
                ball_speed = math.sqrt(dx * dx + dy * dy) / max(dt, 0.01)

        # Classify phase
        if not possession:
            if def_line_x > pitch_length * 0.55:
                phase = "high_press"
            elif def_line_x < pitch_length * 0.3:
                phase = "low_block"
            elif ball_speed > 15.0:
                phase = "transition"
            else:
                phase = "settled_possession"
        else:
            phase = "transition" if ball_speed > 15.0 else "settled_possession"

        phases.append((ts, phase))

    # Merge consecutive same-phase segments, filter short phases
    merged: list[TacticalPhase] = []
    current_label = phases[0][1]
    current_start = phases[0][0]

    for ts, label in phases[1:]:
        if label == current_label:
            continue
        duration = ts - current_start
        if duration >= min_phase_duration:
            merged.append(
                TacticalPhase(
                    start_time=current_start,
                    end_time=ts,
                    label=current_label,
                    duration_s=duration,
                )
            )
        current_label = label
        current_start = ts

    # Last phase
    last_ts = frame_data[-1]["timestamp"]
    duration = last_ts - current_start
    if duration >= min_phase_duration:
        merged.append(
            TacticalPhase(
                start_time=current_start,
                end_time=last_ts,
                label=current_label,
                duration_s=duration,
            )
        )

    total_time = last_ts - frame_data[0]["timestamp"]
    if total_time <= 0:
        return TacticalPeriodReport(phases=merged)

    def _pct(label: str) -> float:
        return sum(p.duration_s for p in merged if p.label == label) / total_time * 100.0

    return TacticalPeriodReport(
        phases=merged,
        press_pct=_pct("high_press"),
        low_block_pct=_pct("low_block"),
        settled_possession_pct=_pct("settled_possession"),
        transition_pct=_pct("transition"),
    )
