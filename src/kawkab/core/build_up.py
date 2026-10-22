"""Build-Up Analysis — how a team builds from the back.

Analyses goal-kick patterns, line-breaking passes, pressure management,
and overall efficiency of moving the ball from the defensive third
to the final third.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BuildUpAction:
    event_index: int
    type: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    successful: bool
    under_pressure: bool
    zone: str
    defensive_line_bypassed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_index": self.event_index,
            "type": self.type,
            "start_x": round(self.start_x, 1),
            "start_y": round(self.start_y, 1),
            "end_x": round(self.end_x, 1),
            "end_y": round(self.end_y, 1),
            "successful": self.successful,
            "under_pressure": self.under_pressure,
            "zone": self.zone,
            "defensive_line_bypassed": self.defensive_line_bypassed,
        }


@dataclass
class BuildUpReport:
    team: str
    match_id: str
    goal_kick_patterns: dict = field(default_factory=dict)
    zone_exit_stats: dict[str, dict] = field(default_factory=dict)
    line_breaking_passes: list[dict] = field(default_factory=list)
    build_out_under_pressure: dict = field(default_factory=dict)
    build_up_efficiency: float = 0.0
    average_pass_sequence_length: float = 0.0
    build_up_actions: list[BuildUpAction] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "team": self.team,
            "match_id": self.match_id,
            "goal_kick_patterns": dict(self.goal_kick_patterns),
            "zone_exit_stats": dict(self.zone_exit_stats),
            "line_breaking_passes": self.line_breaking_passes,
            "build_out_under_pressure": dict(self.build_out_under_pressure),
            "build_up_efficiency": round(self.build_up_efficiency, 3),
            "average_pass_sequence_length": round(self.average_pass_sequence_length, 2),
            "build_up_actions": [a.to_dict() for a in self.build_up_actions],
        }

    def summary_text(self) -> str:
        lines = [f"Build-Up Report for {self.team} (Match: {self.match_id})"]
        gk = self.goal_kick_patterns
        lines.append(
            f"  Goal kicks — short: {gk.get('short', {}).get('attempts', 0)} "
            f"({gk.get('short', {}).get('success_pct', 0.0):.0f}%), "
            f"long: {gk.get('long', {}).get('attempts', 0)} "
            f"({gk.get('long', {}).get('success_pct', 0.0):.0f}%)"
        )
        lines.append(f"  Line-breaking passes: {len(self.line_breaking_passes)}")
        bp = self.build_out_under_pressure
        lines.append(
            f"  Build-out under pressure: {bp.get('attempts', 0)} attempts, "
            f"{bp.get('success_pct', 0.0):.0f}% successful"
        )
        lines.append(f"  Build-up efficiency: {self.build_up_efficiency:.1%}")
        lines.append(f"  Avg pass sequence length: {self.average_pass_sequence_length:.1f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SHORT_GK_MAX_DIST = 25.0
LONG_GK_MIN_DIST = 35.0
DEFENSIVE_THIRD_X_MAX = 34.0
MIDDLE_THIRD_X_MIN = 34.0
MIDDLE_THIRD_X_MAX = 68.0
FINAL_THIRD_X_MIN = 68.0
PRESSURE_DISTANCE_M = 3.0
PRESSURE_TIME_WINDOW_S = 2.0


def _classify_zone(x: float) -> str:
    if x < DEFENSIVE_THIRD_X_MAX:
        return "defensive_third"
    if x < FINAL_THIRD_X_MIN:
        return "middle_third"
    return "final_third"


def _passes_through_line(start_x: float, end_x: float, line_x: float) -> bool:
    """Return True if a pass from *start_x* to *end_x* crosses *line_x*."""
    return (start_x < line_x < end_x) or (end_x < line_x < start_x)


def _lines_bypassed(start_x: float, end_x: float) -> int:
    """Count defensive lines (at x=34, x=68) crossed by an action."""
    lines = [DEFENSIVE_THIRD_X_MAX, FINAL_THIRD_X_MIN]
    count = 0
    for lx in lines:
        if _passes_through_line(start_x, end_x, lx):
            count += 1
    return count


def _is_under_pressure(
    event: dict,
    events: list[dict],
    event_index: int,
) -> bool:
    """Determine if an event occurs under pressure.

    Checks for an explicit 'under_pressure' flag, nearby opponent positions,
    or a defender within PRESSURE_DISTANCE_M of the event start location
    in the preceding PRESSURE_TIME_WINDOW_S.
    """
    if event.get("under_pressure"):
        return True

    opp_positions = event.get("opponent_positions", [])
    if opp_positions:
        sx = event.get("start_x", 0.0)
        sy = event.get("start_y", 34.0)
        for opp in opp_positions:
            dx = sx - opp.get("x", sx)
            dy = sy - opp.get("y", sy)
            if math.hypot(dx, dy) <= PRESSURE_DISTANCE_M:
                return True

    # Check nearby events from the opponent within time window
    event_ts = event.get("timestamp", 0.0)
    for i in range(max(0, event_index - 5), event_index):
        prev = events[i]
        if prev.get("team") != event.get("team"):
            ts_diff = event_ts - prev.get("timestamp", event_ts)
            if 0 <= ts_diff <= PRESSURE_TIME_WINDOW_S:
                dx = event.get("start_x", 0.0) - prev.get("end_x", 0.0)
                dy = event.get("start_y", 34.0) - prev.get("end_y", 34.0)
                if math.hypot(dx, dy) <= PRESSURE_DISTANCE_M * 2:
                    return True

    return False


def _detect_build_up_sequences(
    team_events: list[dict],
    events: list[dict],
    pressure_events: list[dict] | None,
    defensive_third_x_max: float,
) -> tuple[list[BuildUpAction], list[list[dict]]]:
    """Detect build-up actions and sequences.

    Returns (actions, sequences) where each sequence is a list of event dicts
    that begin in the defensive third and form a build-up possession.
    """
    actions: list[BuildUpAction] = []
    sequences: list[list[dict]] = []
    current_seq: list[dict] = []

    for idx, ev in enumerate(events):
        if ev.get("team") not in (team_events[0].get("team") if team_events else None,):
            if current_seq:
                sequences.append(current_seq)
                current_seq = []
            continue

        etype = ev.get("type", "")
        if etype not in ("pass", "carry", "goal_kick", "dribble"):
            if current_seq and etype in ("shot",):
                current_seq.append(ev)
                sequences.append(current_seq)
                current_seq = []
            continue

        sx = ev.get("start_x", 0.0)
        sy = ev.get("start_y", 34.0)
        ex = ev.get("end_x", 0.0)
        ey = ev.get("end_y", 34.0)
        success = ev.get("completed", True)
        zone = _classify_zone(sx)
        under_press = _is_under_pressure(ev, events, idx)
        lines = _lines_bypassed(sx, ex)

        action = BuildUpAction(
            event_index=idx,
            type=etype,
            start_x=sx,
            start_y=sy,
            end_x=ex,
            end_y=ey,
            successful=success,
            under_pressure=under_press,
            zone=zone,
            defensive_line_bypassed=lines,
        )
        actions.append(action)

        if zone == "defensive_third":
            current_seq = [ev]
        elif current_seq:
            current_seq.append(ev)

    if current_seq:
        sequences.append(current_seq)

    return actions, sequences


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def analyze_build_up(
    team_events: list[dict],
    match_events: list[dict],
    team_id: str,
    pressure_events: list[dict] | None = None,
    defensive_third_x_max: float = 34.0,
) -> BuildUpReport:
    """Analyse how a team builds from the back.

    Parameters
    ----------
    team_events : list[dict]
        Events belonging to the analysed team.
    match_events : list[dict]
        Full match event list.
    team_id : str
        Team identifier (e.g. ``"home"``).
    pressure_events : list[dict], optional
        Pre-detected pressure events.
    defensive_third_x_max : float
        Maximum x-coordinate (from own goal line) considered defensive third.

    Returns
    -------
    BuildUpReport
    """
    if not team_events or not match_events:
        return BuildUpReport(
            team=team_id,
            match_id="",
            goal_kick_patterns={
                "short": {"attempts": 0, "success_pct": 0.0},
                "long": {"attempts": 0, "success_pct": 0.0},
            },
            zone_exit_stats={},
            build_out_under_pressure={"attempts": 0, "success_pct": 0.0, "avg_touch_time": 0.0},
        )

    match_id = match_events[0].get("match_id", "") if match_events else ""

    actions, sequences = _detect_build_up_sequences(
        team_events, match_events, pressure_events, defensive_third_x_max
    )

    # Goal-kick analysis
    gk_actions = [a for a in actions if a.type == "goal_kick"]
    short_gk = [a for a in gk_actions if a.end_x - a.start_x < SHORT_GK_MAX_DIST]
    long_gk = [a for a in gk_actions if a.end_x - a.start_x > LONG_GK_MIN_DIST]

    def _gk_stats(gk_list: list[BuildUpAction]) -> dict:
        attempts = len(gk_list)
        success = sum(1 for a in gk_list if a.successful)
        return {
            "attempts": attempts,
            "success_pct": round(success / attempts * 100, 1) if attempts else 0.0,
        }

    goal_kick_patterns = {
        "short": _gk_stats(short_gk),
        "long": _gk_stats(long_gk),
    }

    # Zone exit stats
    zone_data: dict[str, dict] = {
        "defensive_third": {"entries": 0, "exits": 0, "success_pct": 0.0},
        "middle_third": {"entries": 0, "exits": 0, "success_pct": 0.0},
    }
    for a in actions:
        if a.zone in zone_data:
            zone_data[a.zone]["entries"] += 1
        # Check if action exits this zone
        end_zone = _classify_zone(a.end_x)
        if a.zone != end_zone and a.zone in zone_data:
            zone_data[a.zone]["exits"] += 1

    for z in zone_data:
        e = zone_data[z]["entries"]
        x = zone_data[z]["exits"]
        zone_data[z]["success_pct"] = round(x / e * 100, 1) if e else 0.0

    # Line-breaking passes
    line_breaking_passes: list[dict] = []
    for a in actions:
        if (
            a.type in ("pass", "carry")
            and a.defensive_line_bypassed
            and a.defensive_line_bypassed > 0
        ):
            line_breaking_passes.append(a.to_dict())

    # Build-out under pressure
    pressured_actions = [a for a in actions if a.under_pressure and a.zone == "defensive_third"]
    build_out_under_pressure = {
        "attempts": len(pressured_actions),
        "success_pct": round(
            sum(1 for a in pressured_actions if a.successful) / len(pressured_actions) * 100, 1
        )
        if pressured_actions
        else 0.0,
        "avg_touch_time": 0.0,
    }

    # Build-up efficiency: proportion of defensive-possessions that reach final third
    def_reached_final = 0
    def_total = 0
    for seq in sequences:
        if not seq:
            continue
        first = seq[0]
        if _classify_zone(first.get("start_x", 0.0)) == "defensive_third":
            def_total += 1
            for ev in seq:
                if _classify_zone(ev.get("end_x", 0.0)) == "final_third":
                    def_reached_final += 1
                    break

    build_up_efficiency = def_reached_final / def_total if def_total else 0.0

    # Average pass sequence length
    pass_seq_lengths: list[int] = []
    for seq in sequences:
        passes = sum(1 for ev in seq if ev.get("type") == "pass")
        if passes > 0:
            pass_seq_lengths.append(passes)
    avg_seq_len = sum(pass_seq_lengths) / len(pass_seq_lengths) if pass_seq_lengths else 0.0

    return BuildUpReport(
        team=team_id,
        match_id=match_id,
        goal_kick_patterns=goal_kick_patterns,
        zone_exit_stats=zone_data,
        line_breaking_passes=line_breaking_passes,
        build_out_under_pressure=build_out_under_pressure,
        build_up_efficiency=build_up_efficiency,
        average_pass_sequence_length=avg_seq_len,
        build_up_actions=actions,
    )
