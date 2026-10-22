"""Player position heat map generator from tracking data.

Generates per-player and per-team position density heat maps.
Uses 2D Gaussian KDE on tracked pixel positions.
Output: matplotlib figure or numpy array.

Usage:
    from kawkab.services.heatmap_generator import generate_heatmap
    heatmap_img = generate_heatmap(positions, img_w=1920, img_h=1080)
    cv2.imwrite("heatmap.png", heatmap_img)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("heatmap_generator")


def generate_heatmap(
    positions: list[tuple[float, float]],
    img_w: int = 1920,
    img_h: int = 1080,
    sigma_px: float = 30.0,
    threshold_pct: float = 5.0,
) -> np.ndarray:
    """Generate a 2D heat map image from position data.

    Args:
        positions: List of (x, y) pixel coordinates
        img_w, img_h: Output image dimensions
        sigma_px: Gaussian kernel standard deviation in pixels
        threshold_pct: Minimum density percentile to show (removes noise)

    Returns:
        H x W uint8 heatmap image (0-255)
    """
    if not positions:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    pts = np.array(positions, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(np.arange(img_w), np.arange(img_h))

    # Vectorized Gaussian KDE
    sigma2 = sigma_px**2
    density = np.zeros((img_h, img_w), dtype=np.float32)

    # Process in batches to avoid OOM
    batch_size = max(1, len(pts) // 10)
    for i in range(0, len(pts), batch_size):
        batch = pts[i : i + batch_size]
        for p in batch:
            dx = grid_x - p[0]
            dy = grid_y - p[1]
            density += np.exp(-(dx**2 + dy**2) / (2 * sigma2))

    # Normalize to 0-255
    if np.max(density) > 0:
        # Apply threshold (remove noise floor)
        threshold = np.percentile(density[density > 0], threshold_pct) if np.any(density > 0) else 0
        density = np.clip(density - threshold, 0, None)
        density = (density / np.max(density) * 255).astype(np.uint8)

    return density


def generate_team_heatmaps(
    player_positions: dict[int, list[tuple[float, float]]],
    team_assignment: dict[int, str],
    img_w: int = 1920,
    img_h: int = 1080,
) -> dict[str, np.ndarray]:
    """Generate per-team aggregated heat maps.

    Args:
        player_positions: {track_id: [(x, y), ...]}
        team_assignment: {track_id: "home" | "away" | "referee"}

    Returns:
        {"home": heatmap, "away": heatmap, "all": heatmap}
    """
    team_positions: dict[str, list[tuple[float, float]]] = {
        "home": [],
        "away": [],
        "referee": [],
    }
    for tid, positions in player_positions.items():
        team = team_assignment.get(tid, "referee")
        if team not in team_positions:
            team_positions[team] = []
        team_positions[team].extend(positions)

    result: dict[str, np.ndarray] = {}
    for team, positions in team_positions.items():
        if positions:
            result[team] = generate_heatmap(positions, img_w, img_h)
    # Combined
    all_pos = [p for pos_list in team_positions.values() for p in pos_list]
    if all_pos:
        result["all"] = generate_heatmap(all_pos, img_w, img_h)

    return result
