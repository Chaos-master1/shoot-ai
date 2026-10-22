"""Through-ball detection and valuation.

Identifies passes that split defenders and play the ball into space
behind the defensive line, then values them using an xT grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kawkab.core.game_constants import GAME

PITCH_LENGTH = GAME.PITCH_LENGTH_M
PITCH_WIDTH = GAME.PITCH_WIDTH_M
PRESS_THRESHOLD = GAME.PRESS_THRESHOLD_M


@dataclass
class ThroughBall:
    pass_event: dict[str, Any] = field(default_factory=dict)
    xT_gained: float = 0.0
    receiver: int = 0
    split_defenders: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass": {
                "start": (self.pass_event.get("start_x"), self.pass_event.get("start_y")),
                "end": (self.pass_event.get("end_x"), self.pass_event.get("end_y")),
            },
            "xT_gained": round(self.xT_gained, 4),
            "receiver": self.receiver,
            "split_defenders": self.split_defenders,
        }


def _classify_through_ball(pass_event: dict[str, Any]) -> bool:
    """Check if a pass event is marked as a through ball."""
    return pass_event.get("pass_type") == "through_ball" or pass_event.get("type") == "through_ball"


def _compute_behind_defenders(
    pass_event: dict[str, Any],
    defender_positions: list[tuple[float, float, int]],
) -> tuple[bool, list[int]]:
    """Determine if pass goes behind defenders and which are split.

    A through ball splits defenders when the pass trajectory passes
    within PRESS_THRESHOLD of a defender and ends beyond the deepest defender.

    Args:
        pass_event: Dict with start_x, start_y, end_x, end_y.
        defender_positions: List of (x, y, track_id) for defending team.

    Returns:
        (is_through, split_defender_ids).
    """
    sx = pass_event.get("start_x", 0.0)
    sy = pass_event.get("start_y", 34.0)
    ex = pass_event.get("end_x", 0.0)
    ey = pass_event.get("end_y", 34.0)

    if not defender_positions:
        return False, []

    deepest_def_x = max(d[0] for d in defender_positions)
    if ex <= deepest_def_x:
        return False, []

    split_ids: list[int] = []
    for dx, dy, tid in defender_positions:
        t = 0.0
        closest_dist = float("inf")
        while t <= 1.0:
            px = sx + (ex - sx) * t
            py = sy + (ey - sy) * t
            dist = math.hypot(px - dx, py - dy)
            if dist < closest_dist:
                closest_dist = dist
            t += 0.05
        if closest_dist < PRESS_THRESHOLD * 2:
            split_ids.append(tid)

    return len(split_ids) >= 1, split_ids


def detect_through_balls(
    events: list[dict[str, Any]],
    defender_positions: dict[int, list[tuple[float, float, int]]] | None = None,
) -> list[ThroughBall]:
    """Detect through-ball passes in a list of events.

    Args:
        events: List of event dicts. Pass events with pass_type='through_ball'
            or type='through_ball' are detected.
        defender_positions: Optional dict mapping event index to defender
            positions at that moment.

    Returns:
        List of ThroughBall detections.
    """
    results: list[ThroughBall] = []
    for i, ev in enumerate(events):
        if not _classify_through_ball(ev):
            continue

        def_pos = []
        if defender_positions and i in defender_positions:
            def_pos = defender_positions[i]

        is_through, split_ids = _compute_behind_defenders(ev, def_pos)
        if not is_through:
            continue

        results.append(
            ThroughBall(
                pass_event=ev,
                xT_gained=0.0,
                receiver=ev.get("to_track_id", 0),
                split_defenders=split_ids,
            )
        )

    return results


def value_through_ball(
    through_ball: ThroughBall,
    xT_grid: np.ndarray,
    xT_rows: int = 16,
    xT_cols: int = 12,
) -> float:
    """Compute xT value gained by a through ball.

    The xT gained is the difference between the destination zone's
    xT value and the origin zone's xT value.

    Args:
        through_ball: ThroughBall object with pass_event.
        xT_grid: 2D numpy array of xT values per zone.
        xT_rows: Number of rows in xT grid.
        xT_cols: Number of columns in xT grid.

    Returns:
        xT value gained (non-negative).
    """
    ev = through_ball.pass_event
    sx = ev.get("start_x", 0.0)
    sy = ev.get("start_y", 34.0)
    ex = ev.get("end_x", 0.0)
    ey = ev.get("end_y", 34.0)

    def zone(val: float, dim: float, n: int) -> int:
        return min(n - 1, max(0, int(val / dim * n)))

    sz = (zone(sy, PITCH_WIDTH, xT_rows), zone(sx, PITCH_LENGTH, xT_cols))
    ez = (zone(ey, PITCH_WIDTH, xT_rows), zone(ex, PITCH_LENGTH, xT_cols))

    try:
        start_val = xT_grid[sz]
        end_val = xT_grid[ez]
    except IndexError:
        return 0.0

    gained = end_val - start_val
    return round(max(gained, 0.0), 4)
