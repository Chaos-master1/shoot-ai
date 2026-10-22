"""Pass flow analysis — computes pass origin/destination/volume for visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kawkab.core.coords import STANDARD_PITCH, PitchConfig
from kawkab.core.game_constants import GAME


@dataclass
class PassFlowLink:
    origin_x: float = 0.0
    origin_y: float = 34.0
    dest_x: float = 0.0
    dest_y: float = 34.0
    count: int = 0
    completed: int = 0
    avg_progress: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin_x": round(self.origin_x, 1),
            "origin_y": round(self.origin_y, 1),
            "dest_x": round(self.dest_x, 1),
            "dest_y": round(self.dest_y, 1),
            "count": self.count,
            "completed": self.completed,
            "accuracy": round(self.completed / max(self.count, 1), 2),
            "avg_progress": round(self.avg_progress, 1),
        }


def compute_pass_flow(
    events: list[dict[str, Any]],
    team: str = "home",
    pitch: PitchConfig = STANDARD_PITCH,
    grid_cells: int = GAME.PASS_FLOW_GRID_CELLS,
) -> list[dict[str, Any]]:
    """Compute pass flow between pitch zones for a team.

    Divides the pitch into grid_cells×grid_cells zones and aggregates
    passes between zone centers. Returns links with volume and accuracy.

    Args:
        events: List of event dicts.
        team: Team to analyze.
        pitch_length: Pitch length in meters.
        pitch_width: Pitch width in meters.
        grid_cells: Number of zones per dimension.

    Returns:
        List of pass flow link dicts sorted by count descending.
    """
    cell_w = pitch.length_m / grid_cells
    cell_h = pitch.width_m / grid_cells

    links: dict[tuple[int, int, int, int], PassFlowLink] = {}

    for ev in events:
        if ev.get("type") != "pass" or ev.get("team") != team:
            continue
        sx = ev.get("start_x", 0)
        sy = ev.get("start_y", pitch.width_m / 2)
        ex = ev.get("end_x", 0)
        ey = ev.get("end_y", pitch.width_m / 2)

        src_c = min(grid_cells - 1, max(0, int(sx / cell_w)))
        src_r = min(grid_cells - 1, max(0, int(sy / cell_h)))
        dst_c = min(grid_cells - 1, max(0, int(ex / cell_w)))
        dst_r = min(grid_cells - 1, max(0, int(ey / cell_h)))

        key = (src_c, src_r, dst_c, dst_r)
        if key not in links:
            links[key] = PassFlowLink(
                origin_x=(src_c + 0.5) * cell_w,
                origin_y=(src_r + 0.5) * cell_h,
                dest_x=(dst_c + 0.5) * cell_w,
                dest_y=(dst_r + 0.5) * cell_h,
            )
        link = links[key]
        link.count += 1
        if ev.get("completed", False):
            link.completed += 1
        link.avg_progress += ex - sx

    for link in links.values():
        link.avg_progress /= max(link.count, 1)

    return sorted(
        [v.to_dict() for v in links.values()],
        key=lambda x: x["count"],
        reverse=True,
    )
