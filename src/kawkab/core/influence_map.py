"""Player influence maps — Gaussian-based spatial influence computation."""

from __future__ import annotations

import numpy as np


def compute_player_influence(
    positions: list[tuple[float, float]],
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    grid_rows: int = 40,
    grid_cols: int = 60,
    influence_radius: float = 8.0,
) -> list[list[float]]:
    """Compute spatial influence map from a player's position history.

    Each position contributes a Gaussian blob. The map is normalized
    to [0, 1] showing where the player spent the most time/influence.

    Args:
        positions: List of (x, y) pitch coordinates.
        pitch_length: Pitch length in meters.
        pitch_width: Pitch width in meters.
        grid_rows: Grid resolution (rows).
        grid_cols: Grid resolution (cols).
        influence_radius: Gaussian sigma in meters.

    Returns:
        2D grid of influence values normalized to [0, 1].
    """
    if not positions:
        return [[0.0] * grid_cols for _ in range(grid_rows)]

    sigma2 = influence_radius * influence_radius
    gx = (np.arange(grid_cols) + 0.5) * pitch_length / grid_cols
    gy = (np.arange(grid_rows) + 0.5) * pitch_width / grid_rows
    pos = np.array(positions, dtype=np.float64)

    dx = gx[np.newaxis, :, np.newaxis] - pos[np.newaxis, np.newaxis, :, 0]
    dy = gy[:, np.newaxis, np.newaxis] - pos[np.newaxis, np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    density = np.sum(np.exp(-dist_sq / (2.0 * sigma2)), axis=2)
    norm = 1.0 / (2.0 * np.pi * sigma2)

    grid = density * norm
    max_val = float(np.max(grid))
    if max_val > 0:
        grid = np.minimum(1.0, grid / max_val)

    return grid.tolist()


def compute_team_influence_map(
    home_positions: dict[int, list[tuple[float, float]]],
    away_positions: dict[int, list[tuple[float, float]]],
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    grid_rows: int = 40,
    grid_cols: int = 60,
) -> dict[str, list[list[float]]]:
    """Compute aggregate influence maps per team.

    Args:
        home_positions: Dict of track_id -> positions for home team.
        away_positions: Dict of track_id -> positions for away team.
        pitch_length: Pitch length in meters.
        pitch_width: Pitch width in meters.
        grid_rows: Grid resolution (rows).
        grid_cols: Grid resolution (cols).

    Returns:
        Dict with "home" and "away" influence grids.
    """

    def _merge(team_positions):
        if not team_positions:
            return [[0.0] * grid_cols for _ in range(grid_rows)]
        all_pos = []
        for pos_list in team_positions.values():
            all_pos.extend(pos_list)
        if not all_pos:
            return [[0.0] * grid_cols for _ in range(grid_rows)]
        return compute_player_influence(all_pos, pitch_length, pitch_width, grid_rows, grid_cols)

    return {
        "home": _merge(home_positions),
        "away": _merge(away_positions),
    }
