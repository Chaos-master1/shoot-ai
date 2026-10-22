"""Voronoi and velocity-weighted pitch control models.

Computes which areas of the pitch each team controls using Voronoi
tessellation or velocity-weighted soft assignment. Supports per-frame
and aggregate match-level statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kawkab.core.coordinate_validator import CoordinateValidator
from kawkab.core.perf_timing import timed


@dataclass
class PitchControlFrame:
    """Pitch control for a single frame."""

    timestamp: float
    home_control_pct: float = 0.0
    away_control_pct: float = 0.0
    disputed_pct: float = 0.0
    ball_zone_team: str | None = None
    home_grid: list[list[float]] = field(default_factory=list)
    away_grid: list[list[float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": round(self.timestamp, 1),
            "h": round(self.home_control_pct, 1),
            "a": round(self.away_control_pct, 1),
            "d": round(self.disputed_pct, 1),
            "bz": self.ball_zone_team,
        }


@dataclass
class MatchPitchControl:
    """Aggregate pitch control for a full match."""

    avg_home_control: float = 0.0
    avg_away_control: float = 0.0
    frames: list[PitchControlFrame] = field(default_factory=list)

    home_third_control: float = 0.0
    middle_third_control: float = 0.0
    away_third_control: float = 0.0

    ball_in_home_third: float = 0.0
    ball_in_middle_third: float = 0.0
    ball_in_away_third: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "avg_home_control": round(self.avg_home_control, 1),
            "avg_away_control": round(self.avg_away_control, 1),
            "home_third_control": round(self.home_third_control, 1),
            "middle_third_control": round(self.middle_third_control, 1),
            "away_third_control": round(self.away_third_control, 1),
            "ball_in_home_third": round(self.ball_in_home_third, 1),
            "ball_in_middle_third": round(self.ball_in_middle_third, 1),
            "ball_in_away_third": round(self.ball_in_away_third, 1),
        }


class VoronoiPitchControl:
    """Voronoi-based pitch control computation.

    Usage:
        pc = VoronoiPitchControl()
        frame_result = pc.compute_frame_control(home_positions, away_positions)
        match_result = pc.compute_match_control(frames_data)
    """

    def __init__(self, grid_rows: int = 30, grid_cols: int = 46) -> None:
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

    @timed()
    def compute_frame_control(
        self,
        home_positions: list[tuple[float, float]],
        away_positions: list[tuple[float, float]],
        ball_pos: tuple[float, float] | None = None,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
    ) -> PitchControlFrame:
        """Compute pitch control for a single frame using Voronoi tessellation.

        Args:
            home_positions: List of (x, y) positions for home team players.
            away_positions: List of (x, y) positions for away team players.
            ball_pos: Optional (x, y) ball position for ball-zone attribution.
            pitch_length: Pitch length in meters.
            pitch_width: Pitch width in meters.

        Returns:
            PitchControlFrame with control percentages and grid.
        """
        for px, py in home_positions + away_positions:
            CoordinateValidator.validate_point(px, py)
        if ball_pos is not None:
            CoordinateValidator.validate_point(*ball_pos)
        total = self.grid_rows * self.grid_cols

        all_players = home_positions + away_positions
        if not all_players:
            return PitchControlFrame(
                timestamp=0.0,
                home_control_pct=50.0,
                away_control_pct=50.0,
                disputed_pct=0.0,
            )

        gx = (np.arange(self.grid_cols) + 0.5) * pitch_length / self.grid_cols
        gy = (np.arange(self.grid_rows) + 0.5) * pitch_width / self.grid_rows
        player_arr = np.array(all_players, dtype=np.float64)
        n_home = len(home_positions)

        dx = gx[np.newaxis, :, np.newaxis] - player_arr[np.newaxis, np.newaxis, :, 0]
        dy = gy[:, np.newaxis, np.newaxis] - player_arr[np.newaxis, np.newaxis, :, 1]
        dist_sq = dx * dx + dy * dy

        nearest = np.argmin(dist_sq, axis=2)
        home_mask = nearest < n_home
        home_count = int(np.sum(home_mask))
        away_count = total - home_count

        home_grid = np.where(home_mask, 1.0, 0.0)
        away_grid = np.where(home_mask, 0.0, 1.0)

        home_pct = (home_count / total) * 100.0
        away_pct = (away_count / total) * 100.0

        ball_zone_team: str | None = None
        if ball_pos is not None and all_players:
            bx, by = ball_pos
            d2 = (bx - player_arr[:, 0]) ** 2 + (by - player_arr[:, 1]) ** 2
            nearest_idx = int(np.argmin(d2))
            ball_zone_team = "home" if nearest_idx < n_home else "away"

        return PitchControlFrame(
            timestamp=0.0,
            home_control_pct=round(home_pct, 2),
            away_control_pct=round(away_pct, 2),
            disputed_pct=round(100.0 - home_pct - away_pct, 2),
            ball_zone_team=ball_zone_team,
            home_grid=home_grid.tolist(),
            away_grid=away_grid.tolist(),
        )

    def compute_match_control(
        self,
        frames: list[dict[str, Any]],
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
    ) -> MatchPitchControl:
        """Compute aggregate pitch control across multiple frames.

        Args:
            frames: List of dicts with keys:
                - "timestamp": float
                - "home_positions": list of (x, y)
                - "away_positions": list of (x, y)
                - "ball_pos": (x, y) or None
            pitch_length: Pitch length in meters.
            pitch_width: Pitch width in meters.

        Returns:
            MatchPitchControl with aggregated stats.
        """
        if not frames:
            return MatchPitchControl()

        third_x = pitch_length / 3.0

        frame_results: list[PitchControlFrame] = []
        total_home = 0.0
        total_away = 0.0
        frame_count = 0

        home_def_third_sum = 0.0
        home_mid_third_sum = 0.0
        home_att_third_sum = 0.0
        ball_home = 0
        ball_mid = 0
        ball_away = 0
        ball_frames = 0

        for fdata in frames:
            ts = fdata.get("timestamp", 0.0)
            home_pos = fdata.get("home_positions", [])
            away_pos = fdata.get("away_positions", [])
            ball_pos = fdata.get("ball_pos")

            result = self.compute_frame_control(
                home_positions=home_pos,
                away_positions=away_pos,
                ball_pos=ball_pos,
                pitch_length=pitch_length,
                pitch_width=pitch_width,
            )
            result.timestamp = ts
            frame_results.append(result)
            total_home += result.home_control_pct
            total_away += result.away_control_pct
            frame_count += 1

            # per-third control
            hg = result.home_grid
            ag = result.away_grid
            if hg and ag:
                _ = pitch_length / self.grid_cols
                for zone_idx, _zone_name in enumerate([0, 1, 2]):
                    c_start = int(zone_idx * self.grid_cols / 3)
                    c_end = int((zone_idx + 1) * self.grid_cols / 3)
                    zone_home = sum(
                        hg[r][c] for r in range(self.grid_rows) for c in range(c_start, c_end)
                    )
                    zone_away = sum(
                        ag[r][c] for r in range(self.grid_rows) for c in range(c_start, c_end)
                    )
                    zone_total = zone_home + zone_away
                    zone_home_pct = (zone_home / zone_total * 100) if zone_total > 0 else 50.0
                    if zone_idx == 0:
                        home_def_third_sum += zone_home_pct
                    elif zone_idx == 1:
                        home_mid_third_sum += zone_home_pct
                    else:
                        home_att_third_sum += zone_home_pct

            if ball_pos is not None:
                ball_frames += 1
                bx, by = ball_pos
                if bx < third_x:
                    ball_home += 1
                elif bx < 2 * third_x:
                    ball_mid += 1
                else:
                    ball_away += 1

        return MatchPitchControl(
            avg_home_control=total_home / max(frame_count, 1),
            avg_away_control=total_away / max(frame_count, 1),
            frames=frame_results,
            home_third_control=home_att_third_sum / max(frame_count, 1),
            middle_third_control=home_mid_third_sum / max(frame_count, 1),
            away_third_control=home_def_third_sum / max(frame_count, 1),
            ball_in_home_third=(ball_home / max(ball_frames, 1)) * 100,
            ball_in_middle_third=(ball_mid / max(ball_frames, 1)) * 100,
            ball_in_away_third=(ball_away / max(ball_frames, 1)) * 100,
        )


class WeightedPitchControl:
    """Velocity-weighted pitch control model.

    Uses soft assignment based on player reachable area instead of hard Voronoi.
    Each player's influence radius is proportional to their speed: σ = v_max × τ.

    This is closer to professional models (e.g., StatsBomb, Tractable) than
    basic Voronoi tessellation.

    Args:
        grid_rows: Number of rows in control grid.
        grid_cols: Number of columns in control grid.
        time_horizon: Look-ahead time in seconds (default 2.0).
        default_speed: Default max player speed in m/s when velocity unavailable.
    """

    def __init__(
        self,
        grid_rows: int = 30,
        grid_cols: int = 46,
        time_horizon: float = 2.0,
        default_speed: float = 7.0,
    ) -> None:
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.time_horizon = time_horizon
        self.default_speed = default_speed

    def compute_frame_control(
        self,
        home_positions: list[tuple[float, float]],
        away_positions: list[tuple[float, float]],
        ball_pos: tuple[float, float] | None = None,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        home_speeds: list[float] | None = None,
        away_speeds: list[float] | None = None,
    ) -> PitchControlFrame:
        """Compute soft pitch control using velocity-weighted reachability.

        Args:
            home_positions: List of (x, y) positions for home team.
            away_positions: List of (x, y) positions for away team.
            ball_pos: Optional ball position.
            pitch_length: Pitch length in meters.
            pitch_width: Pitch width in meters.
            home_speeds: Per-player max speeds (m/s) for home team.
            away_speeds: Per-player max speeds (m/s) for away team.

        Returns:
            PitchControlFrame with continuous control values and grid.
        """
        _ = self.grid_rows * self.grid_cols

        all_players = home_positions + away_positions
        n_home = len(home_positions)
        if not all_players:
            return PitchControlFrame(
                timestamp=0.0,
                home_control_pct=50.0,
                away_control_pct=50.0,
                disputed_pct=0.0,
            )

        # Build per-player sigma
        home_speeds = home_speeds or [self.default_speed] * n_home
        away_speeds = away_speeds or [self.default_speed] * len(away_positions)
        all_sigmas = np.array(
            [max(s * self.time_horizon, 2.0) for s in list(home_speeds) + list(away_speeds)],
            dtype=np.float64,
        )

        gx = (np.arange(self.grid_cols) + 0.5) * pitch_length / self.grid_cols
        gy = (np.arange(self.grid_rows) + 0.5) * pitch_width / self.grid_rows
        player_arr = np.array(all_players, dtype=np.float64)

        dx = gx[np.newaxis, :, np.newaxis] - player_arr[np.newaxis, np.newaxis, :, 0]
        dy = gy[:, np.newaxis, np.newaxis] - player_arr[np.newaxis, np.newaxis, :, 1]
        dist_sq = dx * dx + dy * dy

        influence = np.exp(-dist_sq / (2.0 * all_sigmas[np.newaxis, np.newaxis, :] ** 2))

        home_influence = np.sum(influence[:, :, :n_home], axis=2)
        away_influence = np.sum(influence[:, :, n_home:], axis=2)
        total_influence = home_influence + away_influence

        mask = total_influence > 0
        home_grid = np.where(mask, home_influence / total_influence, 0.0)
        away_grid = np.where(mask, away_influence / total_influence, 0.0)

        home_pct = float(np.mean(home_grid)) * 100.0
        away_pct = float(np.mean(away_grid)) * 100.0

        ball_zone_team = None
        if ball_pos is not None and all_players:
            bx, by = ball_pos
            d2 = (bx - player_arr[:, 0]) ** 2 + (by - player_arr[:, 1]) ** 2
            influence_ball = np.exp(-d2 / (2.0 * all_sigmas**2))
            best_idx = int(np.argmax(influence_ball))
            ball_zone_team = "home" if best_idx < n_home else "away"

        return PitchControlFrame(
            timestamp=0.0,
            home_control_pct=round(home_pct, 2),
            away_control_pct=round(away_pct, 2),
            disputed_pct=round(100.0 - home_pct - away_pct, 2),
            ball_zone_team=ball_zone_team,
            home_grid=home_grid.tolist(),
            away_grid=away_grid.tolist(),
        )

    def compute_match_control(
        self,
        frames: list[dict[str, Any]],
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
    ) -> MatchPitchControl:
        """Compute aggregate weighted pitch control across frames.

        Args:
            frames: List of dicts with keys:
                - "timestamp": float
                - "home_positions": list of (x, y)
                - "away_positions": list of (x, y)
                - "ball_pos": (x, y) or None
                - "home_speeds": optional list of float (m/s) per home player
                - "away_speeds": optional list of float (m/s) per away player
            pitch_length: Pitch length in meters.
            pitch_width: Pitch width in meters.

        Returns:
            MatchPitchControl with aggregated stats.
        """
        if not frames:
            return MatchPitchControl()

        third_x = pitch_length / 3.0

        frame_results: list[PitchControlFrame] = []
        total_home = 0.0
        total_away = 0.0
        frame_count = 0

        home_def_third_sum = 0.0
        home_mid_third_sum = 0.0
        home_att_third_sum = 0.0
        ball_home = 0
        ball_mid = 0
        ball_away = 0
        ball_frames = 0

        for fdata in frames:
            ts = fdata.get("timestamp", 0.0)
            home_pos = fdata.get("home_positions", [])
            away_pos = fdata.get("away_positions", [])
            ball_pos = fdata.get("ball_pos")
            home_speeds = fdata.get("home_speeds")
            away_speeds = fdata.get("away_speeds")

            result = self.compute_frame_control(
                home_positions=home_pos,
                away_positions=away_pos,
                ball_pos=ball_pos,
                pitch_length=pitch_length,
                pitch_width=pitch_width,
                home_speeds=home_speeds,
                away_speeds=away_speeds,
            )
            result.timestamp = ts
            frame_results.append(result)
            total_home += result.home_control_pct
            total_away += result.away_control_pct
            frame_count += 1

            hg = result.home_grid
            ag = result.away_grid
            if hg and ag:
                for zone_idx in range(3):
                    c_start = int(zone_idx * self.grid_cols / 3)
                    c_end = int((zone_idx + 1) * self.grid_cols / 3)
                    zone_home = sum(
                        hg[r][c] for r in range(self.grid_rows) for c in range(c_start, c_end)
                    )
                    zone_away = sum(
                        ag[r][c] for r in range(self.grid_rows) for c in range(c_start, c_end)
                    )
                    zone_total = zone_home + zone_away
                    zone_home_pct = (zone_home / zone_total * 100) if zone_total > 0 else 50.0
                    if zone_idx == 0:
                        home_def_third_sum += zone_home_pct
                    elif zone_idx == 1:
                        home_mid_third_sum += zone_home_pct
                    else:
                        home_att_third_sum += zone_home_pct

            if ball_pos is not None:
                ball_frames += 1
                bx = ball_pos[0]
                if bx < third_x:
                    ball_home += 1
                elif bx < 2 * third_x:
                    ball_mid += 1
                else:
                    ball_away += 1

        return MatchPitchControl(
            avg_home_control=total_home / max(frame_count, 1),
            avg_away_control=total_away / max(frame_count, 1),
            frames=frame_results,
            home_third_control=home_att_third_sum / max(frame_count, 1),
            middle_third_control=home_mid_third_sum / max(frame_count, 1),
            away_third_control=home_def_third_sum / max(frame_count, 1),
            ball_in_home_third=(ball_home / max(ball_frames, 1)) * 100,
            ball_in_middle_third=(ball_mid / max(ball_frames, 1)) * 100,
            ball_in_away_third=(ball_away / max(ball_frames, 1)) * 100,
        )
