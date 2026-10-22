"""Player position heatmaps using Gaussian KDE with fast binning + convolution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kawkab.core.game_constants import GAME


@dataclass
class HeatmapData:
    player_id: int
    player_name: str | None
    team: str
    grid: list[list[float]]
    min_density: float
    max_density: float


def _gaussian_kernel_1d(size: int, sigma: float) -> np.ndarray:
    """Create a 1D Gaussian kernel normalized to sum to 1."""
    ax = np.linspace(-(size // 2), size // 2, size)
    kernel = np.exp(-0.5 * (ax / sigma) ** 2)
    return kernel / kernel.sum()


def compute_player_heatmap(
    positions: list[tuple[float, float]],
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    grid_rows: int = GAME.HEATMAP_GRID_ROWS,
    grid_cols: int = GAME.HEATMAP_GRID_COLS,
    bandwidth: float = GAME.HEATMAP_KERNEL_SIZE,
) -> list[list[float]]:
    """Compute a 2D Gaussian KDE heatmap using binning + separable convolution.

    Strategy: bin positions to the grid, then apply a separable Gaussian blur.
    This is O(grid_cells) instead of O(grid_cells × positions) for large N.

    Args:
        positions: List of (x, y) pitch coordinates.
        pitch_length: Pitch length in meters.
        pitch_width: Pitch width in meters.
        grid_rows: Number of rows in output grid.
        grid_cols: Number of columns in output grid.
        bandwidth: Gaussian kernel bandwidth in meters.

    Returns:
        2D grid of density values normalized to [0, 1].
    """
    if not positions:
        return [[0.0] * grid_cols for _ in range(grid_rows)]

    pos = np.array(positions, dtype=np.float64)
    x = pos[:, 0]
    y = pos[:, 1]

    # Bin positions to grid cells
    col_idx = np.clip(np.floor(x / pitch_length * grid_cols).astype(np.intp), 0, grid_cols - 1)
    row_idx = np.clip(np.floor(y / pitch_width * grid_rows).astype(np.intp), 0, grid_rows - 1)

    # Build binned histogram
    grid = np.zeros((grid_rows, grid_cols), dtype=np.float64)
    np.add.at(grid, (row_idx, col_idx), 1.0)

    # Normalize by cell area to get density
    cell_area = (pitch_length / grid_cols) * (pitch_width / grid_rows)
    grid = grid / cell_area

    # Separable Gaussian blur
    sigma_cells = bandwidth * grid_cols / pitch_length
    ksize = max(3, int(sigma_cells * 4) | 1)  # odd size, ~4 sigma
    kernel_x = _gaussian_kernel_1d(ksize, sigma_cells)

    sigma_rows = bandwidth * grid_rows / pitch_width
    ksize_y = max(3, int(sigma_rows * 4) | 1)
    kernel_y = _gaussian_kernel_1d(ksize_y, sigma_rows)

    # Convolve along cols then rows (separable)
    grid = _convolve_1d(grid, kernel_x, axis=1)
    grid = _convolve_1d(grid, kernel_y, axis=0)

    # Normalize to [0, 1]
    max_val = float(np.max(grid))
    if max_val > 0:
        grid = np.minimum(1.0, grid / max_val)

    return grid.tolist()


def _convolve_1d(grid: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """1D convolution along axis with zero padding using np.convolve (C-level)."""
    n = grid.shape[axis]
    pad = len(kernel) // 2
    return np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode="full")[pad : pad + n], axis, grid
    )


def compute_team_heatmap(
    all_player_positions: dict[int, list[tuple[float, float]]],
    team: str,
    player_teams: dict[int, str],
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    grid_rows: int = GAME.HEATMAP_GRID_ROWS,
    grid_cols: int = GAME.HEATMAP_GRID_COLS,
    bandwidth: float = GAME.HEATMAP_KERNEL_SIZE,
) -> dict[int, HeatmapData]:
    """Compute per-player heatmaps for all players on a team.

    Args:
        all_player_positions: Dict mapping track_id to list of (x, y) positions.
        team: "home" or "away".
        player_teams: Dict mapping track_id to team.
        pitch_length: Pitch length in meters.
        pitch_width: Pitch width in meters.
        grid_rows: Number of rows in output grid.
        grid_cols: Number of columns in output grid.
        bandwidth: Gaussian kernel bandwidth in meters.

    Returns:
        Dict mapping track_id to HeatmapData.
    """
    result: dict[int, HeatmapData] = {}
    for tid, positions in all_player_positions.items():
        if player_teams.get(tid) != team:
            continue
        grid = compute_player_heatmap(
            positions, pitch_length, pitch_width, grid_rows, grid_cols, bandwidth
        )
        max_val = max(max(row) for row in grid) if grid else 0.0
        min_val = min(min(row) for row in grid) if grid else 0.0
        result[tid] = HeatmapData(
            player_id=tid,
            player_name=None,
            team=team,
            grid=grid,
            min_density=min_val,
            max_density=max_val,
        )
    return result
