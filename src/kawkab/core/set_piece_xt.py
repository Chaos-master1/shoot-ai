"""Set piece xT — values set piece delivery zones.

Computes xT value for corners, free kicks, and throw-ins
based on their delivery destination zones.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kawkab.core.game_constants import GAME

PITCH_LENGTH = GAME.PITCH_LENGTH_M
PITCH_WIDTH = GAME.PITCH_WIDTH_M


SET_PIECE_TYPES = {"corner_kick", "free_kick", "throw_in"}


@dataclass
class SetPieceDeliveryZone:
    sp_type: str = ""
    zone: tuple[int, int] = (0, 0)
    count: int = 0
    avg_xT: float = 0.0
    total_xT: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.sp_type,
            "zone": list(self.zone),
            "count": self.count,
            "avg_xT": round(self.avg_xT, 4),
            "total_xT": round(self.total_xT, 4),
        }


@dataclass
class SetPieceXTReport:
    by_type: dict[str, list[SetPieceDeliveryZone]] = field(default_factory=dict)
    total_xT_by_type: dict[str, float] = field(default_factory=dict)
    most_dangerous_zone: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "by_type": {k: [z.to_dict() for z in v] for k, v in self.by_type.items()},
            "total_xT_by_type": {k: round(v, 4) for k, v in self.total_xT_by_type.items()},
            "most_dangerous_zone": self.most_dangerous_zone,
        }


def _zone(val: float, dim: float, n: int) -> int:
    return min(n - 1, max(0, int(val / dim * n)))


def compute_set_piece_xt(
    set_piece_events: list[dict[str, Any]],
    xT_grid: np.ndarray,
    xT_rows: int = 16,
    xT_cols: int = 12,
) -> SetPieceXTReport:
    """Compute xT value per delivery zone for set pieces.

    Separates analysis by set piece type (corner_kick, free_kick, throw_in)
    and identifies the most dangerous delivery zone.

    Args:
        set_piece_events: List of set piece event dicts with type, start_x,
            start_y, end_x, end_y.
        xT_grid: 2D numpy array of xT values per zone.
        xT_rows: Number of rows in xT grid.
        xT_cols: Number of columns in xT grid.

    Returns:
        SetPieceXTReport with per-type and per-zone breakdown.
    """
    by_type_data: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for ev in set_piece_events:
        etype = ev.get("type", "")
        if etype not in SET_PIECE_TYPES and etype not in ("corner", "free_kick", "throw_in"):
            continue

        norm_type = etype
        if norm_type == "corner":
            norm_type = "corner_kick"

        ex = ev.get("end_x", ev.get("start_x", PITCH_LENGTH))
        ey = ev.get("end_y", ev.get("start_y", PITCH_WIDTH / 2))

        z = (_zone(ey, PITCH_WIDTH, xT_rows), _zone(ex, PITCH_LENGTH, xT_cols))
        try:
            xt_val = float(xT_grid[z])
        except (IndexError, TypeError):
            xt_val = 0.0

        by_type_data[norm_type].append(
            {
                "zone": z,
                "xT": xt_val,
            }
        )

    report = SetPieceXTReport()
    most_dangerous: tuple[float, str, tuple[int, int]] = (0.0, "", (0, 0))

    for sp_type, entries in by_type_data.items():
        zone_counts: dict[tuple[int, int], dict[str, Any]] = {}
        for entry in entries:
            z = entry["zone"]
            if z not in zone_counts:
                zone_counts[z] = {"count": 0, "total_xT": 0.0}
            zone_counts[z]["count"] += 1
            zone_counts[z]["total_xT"] += entry["xT"]

        zones_list: list[SetPieceDeliveryZone] = []
        for z, data in zone_counts.items():
            avg_xt = data["total_xT"] / data["count"]
            dz = SetPieceDeliveryZone(
                sp_type=sp_type,
                zone=z,
                count=data["count"],
                avg_xT=round(avg_xt, 4),
                total_xT=round(data["total_xT"], 4),
            )
            zones_list.append(dz)

            if data["total_xT"] > most_dangerous[0]:
                most_dangerous = (data["total_xT"], sp_type, z)

        zones_list.sort(key=lambda x: x.total_xT, reverse=True)
        report.by_type[sp_type] = zones_list
        report.total_xT_by_type[sp_type] = round(sum(e["xT"] for e in entries), 4)

    report.most_dangerous_zone = {
        "type": most_dangerous[1],
        "zone": list(most_dangerous[2]),
        "total_xT": round(most_dangerous[0], 4),
    }

    return report
