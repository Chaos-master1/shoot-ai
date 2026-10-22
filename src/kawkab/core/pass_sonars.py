"""Pass sonars — per-player pass direction/type/distance visualization data."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class PassSonarSector:
    angle_center: float = 0.0
    angle_width: float = 30.0
    count: int = 0
    completed: int = 0
    avg_distance: float = 0.0
    avg_progress: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "angle_center": round(self.angle_center, 1),
            "angle_width": round(self.angle_width, 1),
            "count": self.count,
            "completed": self.completed,
            "accuracy": round(self.completed / max(self.count, 1), 2),
            "avg_distance": round(self.avg_distance, 1),
            "avg_progress": round(self.avg_progress, 1),
        }


def compute_pass_sonars(
    events: list[dict[str, Any]],
    sectors: int = 12,
) -> list[dict[str, Any]]:
    """Compute pass direction sonars per player.

    Divides 360° into `sectors` equal wedges and aggregates passes
    by the angle from player position to pass destination.

    Args:
        events: List of event dicts. Accepts both the producer shapes:
            typed in-memory dicts (type, track_id, start_x/y, end_x/y) and
            storage rows (event_type is normalized by the caller,
            from_track_id, metadata-extracted x/y, end_x/y, completed).
        sectors: Number of directional sectors (default 12 = 30° each).

    Returns:
        List of per-player sonar dicts: {track_id, team, sectors: [...]}
    """
    from collections import defaultdict

    def _num(event: dict[str, Any], *keys: str, default: float) -> float:
        """First non-None numeric value among `keys` (json_extract emits
        None for absent metadata keys -- don't coerce that to 0.0)."""
        for k in keys:
            v = event.get(k)
            if v is not None:
                return float(v)
        return default

    player_passes: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for ev in events:
        if ev.get("type") != "pass":
            continue
        # Storage rows attribute passes via from_track_id (and use
        # event_type for the type key); accept those shapes too, plus the
        # metadata-extracted x/y the real rows carry instead of start_x.
        tid = str(ev.get("player_id") or ev.get("track_id") or ev.get("from_track_id") or "?")
        sx = _num(ev, "start_x", "x", default=0.0)
        sy = _num(ev, "start_y", "y", default=34.0)
        ex = _num(ev, "end_x", default=sx + 1.0)
        ey = _num(ev, "end_y", default=sy)
        dx = ex - sx
        dy = ey - sy
        angle = math.degrees(math.atan2(dy, dx)) % 360
        dist = math.hypot(dx, dy)
        progress = abs(ex - sx)
        player_passes[tid].append(
            {
                "angle": angle,
                "dist": dist,
                "progress": progress,
                "completed": ev.get("completed", False),
                "team": ev.get("team", "home"),
            }
        )

    result = []
    for tid, passes in player_passes.items():
        sector_angle = 360.0 / sectors
        sector_data = {
            i: PassSonarSector(
                angle_center=i * sector_angle + sector_angle / 2,
                angle_width=sector_angle,
            )
            for i in range(sectors)
        }

        for p in passes:
            idx = int(p["angle"] / sector_angle) % sectors
            sd = sector_data[idx]
            sd.count += 1
            if p["completed"]:
                sd.completed += 1
            sd.avg_distance += p["dist"]
            sd.avg_progress += p["progress"]

        for sd in sector_data.values():
            if sd.count > 0:
                sd.avg_distance /= sd.count
                sd.avg_progress /= sd.count

        team = passes[0]["team"]
        result.append(
            {
                "track_id": tid,
                "team": team,
                "total_passes": len(passes),
                "sectors": sorted(
                    [s.to_dict() for s in sector_data.values()],
                    key=lambda x: x["angle_center"],
                ),
            }
        )

    return sorted(result, key=lambda x: x["total_passes"], reverse=True)
