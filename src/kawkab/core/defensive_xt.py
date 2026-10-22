"""Defensive xT — values defensive actions by xT prevented.

Each interception, tackle, clearance, and block is assigned an xT
value equal to the xT of the zone where the action occurred,
representing the scoring threat that was prevented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from kawkab.core.game_constants import GAME

PITCH_LENGTH = GAME.PITCH_LENGTH_M
PITCH_WIDTH = GAME.PITCH_WIDTH_M


@dataclass
class DefensiveAction:
    event_idx: int = 0
    event_type: str = ""
    team: str = ""
    xT_prevented: float = 0.0
    zone: tuple[int, int] = (0, 0)
    x: float = 0.0
    y: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "idx": self.event_idx,
            "type": self.event_type,
            "team": self.team,
            "xT_prevented": round(self.xT_prevented, 4),
            "zone": list(self.zone),
            "x": round(self.x, 1),
            "y": round(self.y, 1),
        }


DEFENSIVE_EVENT_TYPES = {"interception", "tackle", "clearance", "block"}


def compute_defensive_xt(
    events: list[dict[str, Any]],
    xT_grid: np.ndarray,
    xT_rows: int = 16,
    xT_cols: int = 12,
) -> list[DefensiveAction]:
    """Compute xT prevented by defensive actions.

    Each defensive event is assigned the xT value of the zone
    where it occurred. Higher xT values indicate more dangerous
    situations were prevented.

    Args:
        events: List of event dicts with 'type', 'team', 'start_x', 'start_y'.
        xT_grid: 2D numpy array of xT values per zone.
        xT_rows: Number of rows in xT grid.
        xT_cols: Number of columns in xT grid.

    Returns:
        List of DefensiveAction entries sorted by xT_prevented descending.
    """
    results: list[DefensiveAction] = []

    def zone(val: float, dim: float, n: int) -> int:
        return min(n - 1, max(0, int(val / dim * n)))

    for idx, ev in enumerate(events):
        etype = ev.get("type", "")
        if etype not in DEFENSIVE_EVENT_TYPES:
            continue

        ex = ev.get("start_x", 0.0)
        ey = ev.get("start_y", 34.0)

        if etype in ("interception", "block"):
            ex = ev.get("x", ex)
            ey = ev.get("y", ey)

        # Events without spatial data arrive as None (json_extract NULL) —
        # zone math on None raised TypeError. Skip unlocated defensive
        # events entirely: an xT-prevented value needs a location to mean
        # anything (a center-zone fallback would misattribute danger).
        if ex is None or ey is None:
            continue

        z = (zone(ey, PITCH_WIDTH, xT_rows), zone(ex, PITCH_LENGTH, xT_cols))

        try:
            xt_val = float(xT_grid[z])
        except (IndexError, TypeError):
            xt_val = 0.0

        results.append(
            DefensiveAction(
                event_idx=idx,
                event_type=etype,
                team=ev.get("team", ""),
                xT_prevented=xt_val,
                zone=z,
                x=ex,
                y=ey,
            )
        )

    results.sort(key=lambda a: a.xT_prevented, reverse=True)
    return results
