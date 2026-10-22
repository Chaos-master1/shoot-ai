"""Space control analysis using Voronoi tessellation.

Computes pitch control grids, space gained from passes, and
identifies hot zones where a team has dominant control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kawkab.core.game_constants import GAME

PITCH_LENGTH = GAME.PITCH_LENGTH_M
PITCH_WIDTH = GAME.PITCH_WIDTH_M


@dataclass
class SpaceControlReport:
    team: str = ""
    grid: list[list[float]] = field(default_factory=list)
    team_control_pcts: dict[str, float] = field(default_factory=dict)
    hot_zones: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "team": self.team,
            "grid": self.grid,
            "team_control_pcts": self.team_control_pcts,
            "hot_zones": self.hot_zones,
        }


def compute_pitch_control_grid(
    player_positions: list[tuple[float, float, int]],
    team_ids: list[int],
    grid_rows: int = 30,
    grid_cols: int = 46,
    pitch_length: float = PITCH_LENGTH,
    pitch_width: float = PITCH_WIDTH,
) -> tuple[np.ndarray, dict[int, float]]:
    """Compute Voronoi pitch control grid per team.

    Args:
        player_positions: List of (x, y, track_id) tuples for all players.
        team_ids: List of team ids aligned with player_positions (0 or 1).
        grid_rows: Number of rows in the output grid.
        grid_cols: Number of columns in the output grid.
        pitch_length: Pitch length in meters.
        pitch_width: Pitch width in meters.

    Returns:
        (control_grid, team_pcts) where control_grid is (grid_rows, grid_cols)
        with values 0 (team 0) or 1 (team 1), and team_pcts maps team_id -> control %.
    """
    if not player_positions:
        grid = np.full((grid_rows, grid_cols), -1, dtype=np.int32)
        return grid, {}

    gx = (np.arange(grid_cols) + 0.5) * pitch_length / grid_cols
    gy = (np.arange(grid_rows) + 0.5) * pitch_width / grid_rows

    coords = np.array([(p[0], p[1]) for p in player_positions], dtype=np.float64)
    _ = len(coords)

    dx = gx[np.newaxis, :, np.newaxis] - coords[np.newaxis, np.newaxis, :, 0]
    dy = gy[:, np.newaxis, np.newaxis] - coords[np.newaxis, np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy

    nearest = np.argmin(dist_sq, axis=2)

    team_arr = np.array(team_ids, dtype=np.int32)
    control_grid = team_arr[nearest]

    team_pcts: dict[int, float] = {}
    unique_teams = set(team_ids)
    total = grid_rows * grid_cols
    for tid in unique_teams:
        count = int(np.sum(control_grid == tid))
        team_pcts[tid] = round((count / total) * 100.0, 2)

    return control_grid, team_pcts


def compute_space_gained(
    pass_event: dict[str, Any],
    player_tracks: list[tuple[float, float, float, float, int]],
    grid_rows: int = 30,
    grid_cols: int = 46,
) -> float:
    """Compute space gained by a pass (change in controlled area before/after).

    Args:
        pass_event: Dict with start_x, start_y, end_x, end_y, team.
        player_tracks: List of (x_before, y_before, x_after, y_after, team_id).
        grid_rows: Grid rows for control estimation.
        grid_cols: Grid columns for control estimation.

    Returns:
        Change in controlled area percentage (positive = space gained).
    """
    before_positions = [(t[0], t[1], i) for i, t in enumerate(player_tracks)]
    after_positions = [(t[2], t[3], i) for i, t in enumerate(player_tracks)]
    team_ids = [t[4] for t in player_tracks]

    passing_team = 0 if pass_event.get("team", "home") == "home" else 1

    _, before_pcts = compute_pitch_control_grid(before_positions, team_ids, grid_rows, grid_cols)
    _, after_pcts = compute_pitch_control_grid(after_positions, team_ids, grid_rows, grid_cols)

    before = before_pcts.get(passing_team, 50.0)
    after = after_pcts.get(passing_team, 50.0)
    return round(after - before, 2)


def identify_hot_zones(
    control_grid: np.ndarray,
    team_id: int,
    min_area_pct: float = 5.0,
) -> list[dict[str, Any]]:
    """Identify contiguous zones where a team has dominant control.

    Args:
        control_grid: 2D array of team_id values per cell.
        team_id: Team to find hot zones for.
        min_area_pct: Minimum zone size as percentage of total grid.

    Returns:
        List of hot zone dicts with 'cells', 'center_x', 'center_y', 'area_pct'.
    """
    from scipy import ndimage

    mask = (control_grid == team_id).astype(np.int32)
    labeled, num_features = ndimage.label(mask)

    hot_zones: list[dict[str, Any]] = []
    total_cells = control_grid.size

    for feat_id in range(1, num_features + 1):
        cells = np.argwhere(labeled == feat_id)
        area_pct = (len(cells) / total_cells) * 100.0
        if area_pct < min_area_pct:
            continue
        center_x = float(np.mean(cells[:, 1]))
        center_y = float(np.mean(cells[:, 0]))
        hot_zones.append(
            {
                "cells": len(cells),
                "center_x": round(center_x, 1),
                "center_y": round(center_y, 1),
                "area_pct": round(area_pct, 2),
            }
        )

    return hot_zones
