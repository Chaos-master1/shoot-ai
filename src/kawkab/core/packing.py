"""Packing passes — opponent bypass count and territory penetration.

Measures how many opposition players each pass removes from the game,
based on the Impect packing metric methodology.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from kawkab.core.game_constants import GAME

PITCH_LENGTH = GAME.PITCH_LENGTH_M
PITCH_WIDTH = GAME.PITCH_WIDTH_M


@dataclass
class PackingResult:
    packing_count: int = 0
    territory_penetration: float = 0.0
    pass_length: float = 0.0
    is_progressive: bool = False
    direction: str = "forward"

    def to_dict(self) -> dict[str, Any]:
        return {
            "packing_count": self.packing_count,
            "territory_penetration": round(self.territory_penetration, 2),
            "pass_length": round(self.pass_length, 1),
            "is_progressive": self.is_progressive,
            "direction": self.direction,
        }


@dataclass
class PackingReport:
    total_packing: int = 0
    avg_packing: float = 0.0
    max_packing: int = 0
    total_passes: int = 0
    passes_with_packing: int = 0
    territory_penetration_gained: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_packing": self.total_packing,
            "avg_packing": round(self.avg_packing, 2),
            "max_packing": self.max_packing,
            "total_passes": self.total_passes,
            "passes_with_packing": self.passes_with_packing,
            "territory_penetration_gained": round(self.territory_penetration_gained, 2),
        }


def _is_behind_line(
    player_x: float,
    player_y: float,
    line_start_x: float,
    line_start_y: float,
    line_end_x: float,
    line_end_y: float,
    attacking_direction: int = 1,
) -> bool:
    """Check if a player is behind (on the defensive side of) the pass line.

    Args:
        attacking_direction: 1 if attacking right, -1 if attacking left.

    Returns:
        True if the player is packed (bypassed by the pass).
    """
    # Vector along the pass
    dx = line_end_x - line_start_x
    dy = line_end_y - line_start_y
    line_len = math.hypot(dx, dy)
    if line_len < 0.5:
        return False

    # Vector from pass start to player
    pdx = player_x - line_start_x
    pdy = player_y - line_start_y

    # Project player onto pass line (0.0 = start, 1.0 = end)
    t = (pdx * dx + pdy * dy) / (line_len**2)

    # Perpendicular distance from pass line
    perp_dist = abs(pdx * dy - pdy * dx) / line_len

    # A player is "packed" if:
    # 1. Within width corridor (perpendicular distance < ~3m)
    # 2. Behind the pass relative to attacking direction
    # 3. Between start and end or just past start
    corridor_width = 3.0
    if perp_dist > corridor_width:
        return False

    # Check if player is behind the pass (on defensive side of line)
    # For attacking_right: player_y means nothing — use x position
    # behind = the player's position is on the defensive side
    if attacking_direction > 0:
        # Attacking right: player is packed if they're behind (greater x than) the start
        # and between the start and end of the pass
        return player_x > line_start_x and t < 1.0
    else:
        return player_x < line_start_x and t < 1.0


def compute_packing(
    pass_event: dict[str, Any],
    opponent_positions: list[tuple[float, float]],
    attacking_direction: int = 1,
) -> PackingResult:
    """Compute packing count for a single pass.

    Args:
        pass_event: Dict with start_x, start_y, end_x, end_y.
        opponent_positions: List of (x, y) tuples for opposition players.
        attacking_direction: 1 for rightward attack, -1 for leftward.

    Returns:
        PackingResult with count and territory metrics.
    """
    sx = pass_event.get("start_x", 52.5)
    sy = pass_event.get("start_y", 34.0)
    ex = pass_event.get("end_x", 52.5)
    ey = pass_event.get("end_y", 34.0)

    pass_length = math.hypot(ex - sx, ey - sy)

    # Territory penetration: how far forward the pass goes
    territory = max(0.0, ex - sx) if attacking_direction > 0 else max(0.0, sx - ex)
    territory_pct = (territory / PITCH_LENGTH) * 100.0

    # Count packed opponents
    packing_count = 0
    for ox, oy in opponent_positions:
        if _is_behind_line(ox, oy, sx, sy, ex, ey, attacking_direction):
            packing_count += 1

    is_progressive = territory_pct > 5.0

    direction = "right" if attacking_direction > 0 else "left"

    return PackingResult(
        packing_count=packing_count,
        territory_penetration=territory_pct,
        pass_length=pass_length,
        is_progressive=is_progressive,
        direction=direction,
    )


def compute_match_packing(
    events: list[dict[str, Any]],
    team_attacks_right: bool = True,
) -> dict[str, PackingReport]:
    """Compute packing metrics for all passes in a match.

    Args:
        events: List of event dicts with position and team data.
        team_attacks_right: If True, home team attacks right in first half.

    Returns:
        Dict with "home" and "away" PackingReport.
    """
    from collections import defaultdict

    # Group events by team and timestamp for opponent position lookup
    team_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in events:
        team_events[ev.get("team", "home")].append(ev)

    # For each pass, estimate opponent positions from nearby events
    home_passes: list[PackingResult] = []
    away_passes: list[PackingResult] = []
    att_dir = 1 if team_attacks_right else -1

    sorted_ev = sorted(events, key=lambda e: e.get("timestamp", 0))
    n = len(sorted_ev)

    for i, ev in enumerate(sorted_ev):
        if ev.get("type") != "pass" or not ev.get("completed", True):
            continue

        team = ev.get("team", "home")
        _ = ev.get("start_x", 52.5)
        _ = ev.get("start_y", 34.0)

        # Estimate opponent positions from nearby events
        opp_positions: list[tuple[float, float]] = []
        # Look at events within 5 seconds for opponent players' positions
        ts = ev.get("timestamp", 0.0)
        window = 5.0

        for j in range(max(0, i - 20), min(n, i + 20)):
            other = sorted_ev[j]
            if other.get("type") not in ("pass", "carry", "tackle", "shot", "interception"):
                continue
            oteam = other.get("team", "home")
            if oteam == team:
                continue  # Same team, not opponent
            ox = other.get("x", other.get("start_x"))
            oy = other.get("y", other.get("start_y"))
            if ox is not None and oy is not None:
                ots = other.get("timestamp", ts)
                if abs(ots - ts) <= window:
                    opp_positions.append((float(ox), float(oy)))

        if not opp_positions:
            continue

        # Attacking direction depends on the team
        # Away team attacks in the opposite direction
        team_att_dir = att_dir if team == "home" else -att_dir

        result = compute_packing(ev, opp_positions, team_att_dir)

        if team == "home":
            home_passes.append(result)
        else:
            away_passes.append(result)

    def _build_report(results: list[PackingResult]) -> PackingReport:
        if not results:
            return PackingReport()
        total_packing = sum(r.packing_count for r in results)
        return PackingReport(
            total_packing=total_packing,
            avg_packing=total_packing / len(results),
            max_packing=max(r.packing_count for r in results),
            total_passes=len(results),
            passes_with_packing=sum(1 for r in results if r.packing_count > 0),
            territory_penetration_gained=sum(r.territory_penetration for r in results),
        )

    return {
        "home": _build_report(home_passes),
        "away": _build_report(away_passes),
    }
