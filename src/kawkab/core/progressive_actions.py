"""Progressive Passes and Carries — ball-advancement toward opponent goal.

Identifies and quantifies passes and carries that move the ball significantly
toward the opponent's goal. Core metric in professional football analytics
(StatsBomb, Opta, Wyscout).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from kawkab.core.game_constants import GAME

PITCH_LENGTH = GAME.PITCH_LENGTH_M
PITCH_WIDTH = GAME.PITCH_WIDTH_M
GOAL_WIDTH = 7.32
GOAL_CENTER_Y = PITCH_WIDTH / 2.0
SIX_YARD_DEPTH = 5.5
SIX_YARD_HALF_WIDTH = 5.5 + GOAL_WIDTH / 2.0
PENALTY_AREA_DEPTH = 16.5
PENALTY_AREA_HALF_WIDTH = 16.5 + GOAL_WIDTH / 2.0
MIN_PROGRESSION_RATIO = GAME.PROGRESSIVE_MIN_PROGRESSION_RATIO
ATTACKING_THIRD_FRACTION = GAME.PROGRESSIVE_ATTACKING_THIRD_FRACTION
MIN_CARRY_PROGRESSION_M = GAME.PROGRESSIVE_MIN_CARRY_M
CORRIDOR_WIDTH = 2.0

# 15 pitch zones: 5 lengthwise bands x 3 widthwise bands
X_BANDS = [
    (0.0, 0.2, "Defensive"),
    (0.2, 0.4, "Defensive Mid"),
    (0.4, 0.6, "Middle"),
    (0.6, 0.8, "Attacking Mid"),
    (0.8, 1.0, "Attacking"),
]
Y_BANDS = [
    (0.0, 1 / 3, "Left"),
    (1 / 3, 2 / 3, "Center"),
    (2 / 3, 1.0, "Right"),
]


@dataclass
class ProgressiveAction:
    action_type: str  # "pass" or "carry"
    player_track_id: int
    team: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    distance_m: float
    progression_m: float
    is_progressive: bool
    zone_start: str
    zone_end: str
    opponent_bypassed: int = 0
    danger_rating: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "player_track_id": self.player_track_id,
            "team": self.team,
            "start_x": round(self.start_x, 1),
            "start_y": round(self.start_y, 1),
            "end_x": round(self.end_x, 1),
            "end_y": round(self.end_y, 1),
            "distance_m": round(self.distance_m, 1),
            "progression_m": round(self.progression_m, 1),
            "is_progressive": self.is_progressive,
            "zone_start": self.zone_start,
            "zone_end": self.zone_end,
            "opponent_bypassed": self.opponent_bypassed,
            "danger_rating": round(self.danger_rating, 2),
        }


@dataclass
class ProgressiveReport:
    team: str
    total_progressive_passes: int = 0
    total_progressive_carries: int = 0
    total_pass_progression_m: float = 0.0
    total_carry_progression_m: float = 0.0
    avg_pass_progression_m: float = 0.0
    avg_carry_progression_m: float = 0.0
    progressive_pass_rate: float = 0.0
    progressive_carry_rate: float = 0.0
    actions_by_zone: dict[str, int] = field(default_factory=dict)
    danger_actions: int = 0
    top_players: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "team": self.team,
            "total_progressive_passes": self.total_progressive_passes,
            "total_progressive_carries": self.total_progressive_carries,
            "total_pass_progression_m": round(self.total_pass_progression_m, 1),
            "total_carry_progression_m": round(self.total_carry_progression_m, 1),
            "avg_pass_progression_m": round(self.avg_pass_progression_m, 2),
            "avg_carry_progression_m": round(self.avg_carry_progression_m, 2),
            "progressive_pass_rate": round(self.progressive_pass_rate, 3),
            "progressive_carry_rate": round(self.progressive_carry_rate, 3),
            "actions_by_zone": dict(self.actions_by_zone),
            "danger_actions": self.danger_actions,
            "top_players": self.top_players,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_progressive_passes(
    events: list[dict[str, Any]],
    team: str,
    attacking_direction: int = 1,
) -> ProgressiveReport:
    """Analyze all passes and carries to find progressive actions.

    Args:
        events: List of event dicts. Each event should contain keys:
            type, player_track_id, team, start_x, start_y, end_x, end_y,
            distance (optional), opponent_positions (optional list of dicts
            with x/y for packing estimate).
        team: Team identifier to filter events for.
        attacking_direction: 1 if attacking right (toward +x),
            -1 if attacking left (toward -x).

    Returns:
        ProgressiveReport summarizing all progressive actions.
    """
    team_events = [e for e in events if e.get("team") == team]
    if not team_events:
        return ProgressiveReport(team=team)

    actions: list[ProgressiveAction] = []
    all_pass_count = 0
    all_carry_count = 0

    for ev in team_events:
        ev_type = ev.get("type", "")
        pid = int(ev.get("player_track_id", 0))
        sx = float(ev.get("start_x", 0.0))
        sy = float(ev.get("start_y", 0.0))
        ex = float(ev.get("end_x", 0.0))
        ey = float(ev.get("end_y", 0.0))
        dist = float(ev.get("distance", 0.0))
        opp_positions: list[dict[str, float]] | None = ev.get("opponent_positions")

        zs = _classify_zone(sx, sy)
        ze = _classify_zone(ex, ey)

        if ev_type == "pass":
            all_pass_count += 1
            prog = _is_progressive_pass(sx, ex, sy, ey, PITCH_LENGTH, attacking_direction)
            progression_m = (ex - sx) * attacking_direction
        elif ev_type == "carry":
            all_carry_count += 1
            prog = _is_progressive_carry(sx, ex, dist, attacking_direction, MIN_CARRY_PROGRESSION_M)
            progression_m = (ex - sx) * attacking_direction
        else:
            continue

        opp_bypassed = _count_opponents_behind_pass(
            sx,
            sy,
            ex,
            ey,
            opp_positions or [],
            attacking_direction,
        )
        danger = _compute_danger_rating(ex, ey, PITCH_LENGTH, PITCH_WIDTH, attacking_direction)

        action = ProgressiveAction(
            action_type=ev_type,
            player_track_id=pid,
            team=team,
            start_x=sx,
            start_y=sy,
            end_x=ex,
            end_y=ey,
            distance_m=dist,
            progression_m=progression_m,
            is_progressive=prog,
            zone_start=zs,
            zone_end=ze,
            opponent_bypassed=opp_bypassed,
            danger_rating=danger,
        )
        actions.append(action)

    progressive_actions = [a for a in actions if a.is_progressive]
    progressive_passes = [a for a in progressive_actions if a.action_type == "pass"]
    progressive_carries = [a for a in progressive_actions if a.action_type == "carry"]

    total_pass_prog = sum(a.progression_m for a in progressive_passes)
    total_carry_prog = sum(a.progression_m for a in progressive_carries)

    avg_pass_prog = total_pass_prog / len(progressive_passes) if progressive_passes else 0.0
    avg_carry_prog = total_carry_prog / len(progressive_carries) if progressive_carries else 0.0

    pass_rate = len(progressive_passes) / all_pass_count if all_pass_count else 0.0
    carry_rate = len(progressive_carries) / all_carry_count if all_carry_count else 0.0

    actions_by_zone: dict[str, int] = {}
    for a in progressive_actions:
        actions_by_zone[a.zone_start] = actions_by_zone.get(a.zone_start, 0) + 1

    danger_actions = sum(1 for a in progressive_actions if a.danger_rating > 0.7)

    # Top 3 players by total progression
    player_prog: dict[int, dict[str, Any]] = {}
    for a in progressive_actions:
        if a.player_track_id not in player_prog:
            player_prog[a.player_track_id] = {
                "player_track_id": a.player_track_id,
                "team": team,
                "total_progression_m": 0.0,
                "progressive_actions": 0,
            }
        player_prog[a.player_track_id]["total_progression_m"] += a.progression_m
        player_prog[a.player_track_id]["progressive_actions"] += 1

    sorted_players = sorted(
        player_prog.values(),
        key=lambda p: p["total_progression_m"],
        reverse=True,
    )
    for p in sorted_players:
        p["total_progression_m"] = round(p["total_progression_m"], 1)

    return ProgressiveReport(
        team=team,
        total_progressive_passes=len(progressive_passes),
        total_progressive_carries=len(progressive_carries),
        total_pass_progression_m=total_pass_prog,
        total_carry_progression_m=total_carry_prog,
        avg_pass_progression_m=avg_pass_prog,
        avg_carry_progression_m=avg_carry_prog,
        progressive_pass_rate=pass_rate,
        progressive_carry_rate=carry_rate,
        actions_by_zone=actions_by_zone,
        danger_actions=danger_actions,
        top_players=sorted_players[:3],
    )


# ---------------------------------------------------------------------------
# Helper functions (individually importable and testable)
# ---------------------------------------------------------------------------


def _is_progressive_pass(
    start_x: float,
    end_x: float,
    start_y: float,
    end_y: float,
    pitch_length: float = PITCH_LENGTH,
    attacking_direction: int = 1,
) -> bool:
    """Determine if a pass is progressive.

    A pass is progressive if:
    1. It moves the ball >=25% of the remaining distance toward opponent goal.
    2. The end point is in the attacking 60% of the pitch.

    Args:
        start_x, end_x: x-coordinates (0 = own goal line).
        start_y, end_y: y-coordinates (ignored for x-only check, reserved).
        pitch_length: Total pitch length in metres.
        attacking_direction: 1 = attacking right (+x), -1 = attacking left (-x).

    Returns:
        True if the pass meets progressive criteria.
    """
    if start_x == end_x:
        return False

    if attacking_direction == 1:
        remaining_before = pitch_length - start_x
        remaining_after = pitch_length - end_x
        attacking_threshold = pitch_length * (1 - ATTACKING_THIRD_FRACTION)  # 42.0
        in_attacking_third = end_x >= attacking_threshold
    else:
        remaining_before = start_x
        remaining_after = end_x
        attacking_threshold = pitch_length * ATTACKING_THIRD_FRACTION  # 63.0
        in_attacking_third = end_x <= attacking_threshold

    if remaining_before <= 0.0:
        return False

    reduction = (remaining_before - remaining_after) / remaining_before
    return reduction >= MIN_PROGRESSION_RATIO and in_attacking_third


def _is_progressive_carry(
    start_x: float,
    end_x: float,
    distance_m: float,
    attacking_direction: int = 1,
    min_progression: float = MIN_CARRY_PROGRESSION_M,
) -> bool:
    """Determine if a carry is progressive.

    A carry is progressive if it moves >=min_progression metres toward
    the opponent's goal.

    Args:
        start_x, end_x: x-coordinates.
        distance_m: Total distance carried (unused in default logic).
        attacking_direction: 1 = attacking right, -1 = attacking left.
        min_progression: Minimum net forward metres to qualify (default 5.0).

    Returns:
        True if the carry meets the progression threshold.
    """
    progression_m = (end_x - start_x) * attacking_direction
    return progression_m >= min_progression


def _compute_danger_rating(
    end_x: float,
    end_y: float,
    pitch_length: float = PITCH_LENGTH,
    pitch_width: float = PITCH_WIDTH,
    attacking_direction: int = 1,
) -> float:
    """Compute danger rating of where an action ends (0-1).

    Rating schema:
      1.0 — inside 6-yard box (5.5 m from goal line, central)
      0.8 — inside penalty area (16.5 m from goal line)
      0.7 — central area of final third (outside penalty area)
      0.5 — wide areas of final third
      0.3 — middle third
      0.1 — own half

    Args:
        end_x, end_y: Coordinates of the action endpoint.
        pitch_length, pitch_width: Pitch dimensions in metres.
        attacking_direction: 1 = attacking right, -1 = attacking left.

    Returns:
        Float danger rating in [0.1, 1.0].
    """
    goal_line = pitch_length if attacking_direction == 1 else 0.0
    dist_to_goal = abs(goal_line - end_x)
    cy = pitch_width / 2.0

    if dist_to_goal <= SIX_YARD_DEPTH and abs(end_y - cy) <= SIX_YARD_HALF_WIDTH:
        return 1.0

    if dist_to_goal <= PENALTY_AREA_DEPTH and abs(end_y - cy) <= PENALTY_AREA_HALF_WIDTH:
        return 0.8

    # Final third (last 35 m of pitch)
    final_third_depth = pitch_length / 3.0
    if dist_to_goal <= final_third_depth:
        if abs(end_y - cy) <= final_third_depth / 2.0:
            return 0.7
        return 0.5

    # Middle third (35-70 m from own goal; 35m to 70m bands)
    middle_third_depth = pitch_length / 3.0
    if dist_to_goal <= 2 * middle_third_depth:
        return 0.3

    return 0.1


def _classify_zone(
    x: float,
    y: float,
    pitch_length: float = PITCH_LENGTH,
    pitch_width: float = PITCH_WIDTH,
) -> str:
    """Classify a coordinate into one of 15 pitch zone names.

    Zones are 5 lengthwise (Defensive, Defensive Mid, Middle,
    Attacking Mid, Attacking) x 3 widthwise (Left, Center, Right).

    Args:
        x, y: Pitch coordinates in metres.
        pitch_length, pitch_width: Pitch dimensions.

    Returns:
        Human-readable zone name, e.g. "Attacking Center".
    """
    x_frac = max(0.0, min(0.999, x / pitch_length))
    y_frac = max(0.0, min(0.999, y / pitch_width))

    x_name: str = "Defensive"
    for lo, hi, name in X_BANDS:
        if lo <= x_frac < hi:
            x_name = name
            break

    y_name: str = "Left"
    for lo, hi, name in Y_BANDS:
        if lo <= y_frac < hi:
            y_name = name
            break

    return f"{x_name} {y_name}"


def _count_opponents_behind_pass(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    opponent_positions: list[dict[str, float]],
    attacking_direction: int = 1,
) -> int:
    """Estimate opponents bypassed using a packing-like line check.

    An opponent is considered bypassed if they are positioned between
    the start and end x-coordinate (in the attacking direction) and
    within CORRIDOR_WIDTH metres perpendicular distance of the
    pass/carry line.

    Args:
        start_x, start_y, end_x, end_y: Action endpoints.
        opponent_positions: List of dicts with "x" and "y" keys.
        attacking_direction: 1 = attacking right, -1 = attacking left.

    Returns:
        Count of opponents bypassed.
    """
    dx = end_x - start_x
    dy = end_y - start_y
    length = math.hypot(dx, dy)

    if length < 1e-6 or not opponent_positions:
        return 0

    count = 0
    for opp in opponent_positions:
        ox = float(opp.get("x", 0.0))
        oy = float(opp.get("y", 0.0))

        # Check opponent is between start and end in x (forward direction)
        if attacking_direction == 1:
            if not (start_x < ox < end_x or end_x < ox < start_x):
                continue
        else:
            if not (end_x < ox < start_x or start_x < ox < end_x):
                continue

        # Perpendicular distance from point to line segment
        t = max(0.0, min(1.0, ((ox - start_x) * dx + (oy - start_y) * dy) / (length * length)))
        proj_x = start_x + t * dx
        proj_y = start_y + t * dy
        perp_dist = math.hypot(ox - proj_x, oy - proj_y)

        if perp_dist <= CORRIDOR_WIDTH:
            count += 1

    return count
