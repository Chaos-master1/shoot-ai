"""Pressing cluster analysis using Voronoi-based spatial clustering.

Identifies spatial clusters of high pressing intensity,
measuring zone-level press frequency, success rate, and
opponent impact.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from kawkab.core.game_constants import GAME

PITCH_LENGTH = GAME.PITCH_LENGTH_M
PITCH_WIDTH = GAME.PITCH_WIDTH_M


@dataclass
class PressingCluster:
    zone: tuple[int, int] = (0, 0)
    center_x: float = 0.0
    center_y: float = 0.0
    intensity: float = 0.0
    event_count: int = 0
    success_rate: float = 0.0
    dominant_team: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone": list(self.zone),
            "center_x": round(self.center_x, 1),
            "center_y": round(self.center_y, 1),
            "intensity": round(self.intensity, 2),
            "event_count": self.event_count,
            "success_rate": round(self.success_rate, 3),
            "dominant_team": self.dominant_team,
        }


PRESS_EVENT_TYPES = {"tackle", "interception", "block", "pressure", "foul"}


def cluster_pressing_events(
    events: list[dict[str, Any]],
    player_tracks: dict[int, list[tuple[float, float, float]]] | None = None,
    grid_rows: int = 12,
    grid_cols: int = 8,
    min_events_per_cluster: int = 2,
) -> list[PressingCluster]:
    """Identify spatial clusters of high pressing intensity.

    Aggregates pressing events (tackles, interceptions, blocks, pressure, fouls)
    into a spatial grid and identifies clusters with above-average intensity.

    Args:
        events: List of event dicts with type, team, start_x, start_y.
        player_tracks: Optional dict mapping track_id to list of (x, y, timestamp).
            Not currently used but included for API compatibility.
        grid_rows: Number of rows in the spatial grid.
        grid_cols: Number of columns in the spatial grid.
        min_events_per_cluster: Minimum events to form a cluster.

    Returns:
        List of PressingCluster objects sorted by intensity descending.
    """
    grid_counts: dict[tuple[int, int], int] = defaultdict(int)
    grid_teams: dict[tuple[int, int], set[str]] = defaultdict(set)
    grid_success: dict[tuple[int, int], int] = defaultdict(int)

    def zone(val: float, dim: float, n: int) -> int:
        return min(n - 1, max(0, int(val / dim * n)))

    total_press_events = 0
    for ev in events:
        etype = ev.get("type", "")
        if etype not in PRESS_EVENT_TYPES:
            continue

        ex = ev.get("start_x", 0.0)
        ey = ev.get("start_y", PITCH_WIDTH / 2)

        z = (zone(ey, PITCH_WIDTH, grid_rows), zone(ex, PITCH_LENGTH, grid_cols))
        grid_counts[z] += 1
        grid_teams[z].add(ev.get("team", ""))

        if etype in ("tackle", "interception", "block"):
            grid_success[z] += 1

        total_press_events += 1

    if total_press_events == 0:
        return []

    avg_intensity = total_press_events / (grid_rows * grid_cols)

    clusters: list[PressingCluster] = []
    for z, count in grid_counts.items():
        if count < min_events_per_cluster:
            continue

        intensity = count / max(avg_intensity, 0.01)
        if intensity < 1.0:
            continue

        success = grid_success.get(z, 0)
        success_rate = success / count

        zy, zx = z
        center_x = (zx + 0.5) * PITCH_LENGTH / grid_cols
        center_y = (zy + 0.5) * PITCH_WIDTH / grid_rows

        teams_in_zone = grid_teams.get(z, set())
        dominant_team = ""
        if len(teams_in_zone) == 1:
            dominant_team = list(teams_in_zone)[0]
        elif len(teams_in_zone) > 1:
            home_count = sum(
                1
                for ev in events
                if ev.get("type") in PRESS_EVENT_TYPES
                and ev.get("team") == "home"
                and zone(ev.get("start_y", PITCH_WIDTH / 2), PITCH_WIDTH, grid_rows) == zy
                and zone(ev.get("start_x", 0.0), PITCH_LENGTH, grid_cols) == zx
            )
            away_count = count - home_count
            dominant_team = "home" if home_count >= away_count else "away"

        clusters.append(
            PressingCluster(
                zone=z,
                center_x=round(center_x, 1),
                center_y=round(center_y, 1),
                intensity=round(intensity, 2),
                event_count=count,
                success_rate=round(success_rate, 3),
                dominant_team=dominant_team,
            )
        )

    clusters.sort(key=lambda c: c.intensity, reverse=True)
    return clusters
