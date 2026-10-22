"""Physical load service - compute advanced physical metrics from tracking data.

Computes professional-grade physical load metrics:
- Sprint count and distance (runs > 25 km/h for > 1 second)
- High-intensity running (15-25 km/h)
- Accelerations (> 3 m/s²)
- Decelerations (< -3 m/s²)
- Total high-intensity distance
- Work-to-rest ratio
- Sprint profile (max, average, peak)
- Metabolic power estimate

These metrics are used by sports scientists and fitness coaches to manage
player load, prevent injury, and optimize conditioning.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any

from kawkab.core.logging import get_logger
from kawkab.services.cv_service import MatchTrackData

logger = get_logger(__name__)


@dataclass
class PhysicalLoadMetrics:
    """Physical load metrics for a single player."""

    track_id: int
    total_distance_m: float = 0.0
    walking_distance_m: float = 0.0  # < 7 km/h
    jogging_distance_m: float = 0.0  # 7-15 km/h
    high_intensity_distance_m: float = 0.0  # 15-25 km/h
    sprint_distance_m: float = 0.0  # > 25 km/h
    sprint_count: int = 0
    max_speed_kmh: float = 0.0
    avg_speed_kmh: float = 0.0
    acceleration_count: int = 0  # > 3 m/s²
    deceleration_count: int = 0  # < -3 m/s²
    peak_acceleration_mps2: float = 0.0
    peak_deceleration_mps2: float = 0.0
    high_intensity_bouts: int = 0
    work_rest_ratio: float = 0.0
    metabolic_power_estimate: float = 0.0  # arbitrary units, relative


class PhysicalLoadService:
    """Computes advanced physical load metrics from tracking data."""

    # Speed thresholds (km/h)
    WALKING_MAX = 7.0
    JOGGING_MAX = 15.0
    HIGH_INTENSITY_MAX = 25.0

    # Acceleration thresholds (m/s²)
    ACCELERATION_THRESHOLD = 3.0
    DECELERATION_THRESHOLD = -3.0

    def __init__(self) -> None:
        logger.info("PhysicalLoadService initialized")

    async def compute_physical_load(
        self,
        track_data: MatchTrackData,
        homography_matrix=None,
    ) -> dict[int, PhysicalLoadMetrics]:
        """Compute physical load metrics for all tracked players.

        Args:
            track_data: Match tracking data with per-frame positions
            homography_matrix: Optional pitch calibration for meter-based stats

        Returns:
            Dict mapping track_id to PhysicalLoadMetrics
        """
        logger.info("Computing physical load metrics...")

        # Extract per-player position trajectories
        player_trajectories: dict[int, list[tuple[float, float, float]]] = {}
        for frame in track_data.frames:
            for det in frame.detections:
                if det.class_name != "person" or det.track_id is None:
                    continue

                tid = det.track_id
                x1, y1, x2, y2 = det.bbox
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                if homography_matrix is not None:
                    with contextlib.suppress(Exception):
                        cx, cy = homography_matrix.pixel_to_pitch(cx, cy)

                if tid not in player_trajectories:
                    player_trajectories[tid] = []
                player_trajectories[tid].append((frame.timestamp, cx, cy))

        results: dict[int, PhysicalLoadMetrics] = {}

        for tid, traj in player_trajectories.items():
            if len(traj) < 2:
                continue

            metrics = self._analyze_trajectory(tid, traj)
            results[tid] = metrics

        logger.info(f"Physical load computed for {len(results)} players")
        return results

    def _analyze_trajectory(
        self, track_id: int, trajectory: list[tuple[float, float, float]]
    ) -> PhysicalLoadMetrics:
        """Analyze a single player's trajectory and compute physical load."""
        metrics = PhysicalLoadMetrics(track_id=track_id)

        speeds = []
        accelerations = []
        total_time = 0.0
        high_intensity_time = 0.0
        rest_time = 0.0
        in_sprint = False
        sprint_start_time = 0.0
        sprint_distance = 0.0

        for i in range(1, len(trajectory)):
            t0, x0, y0 = trajectory[i - 1]
            t1, x1, y1 = trajectory[i]
            dt = t1 - t0
            if dt <= 0:
                continue

            dx = x1 - x0
            dy = y1 - y0
            distance = math.sqrt(dx * dx + dy * dy)
            speed_mps = distance / dt
            speed_kmh = speed_mps * 3.6

            # Cap speed at human limit
            speed_kmh = min(speed_kmh, 40.0)
            speed_mps = speed_kmh / 3.6

            metrics.total_distance_m += distance
            speeds.append(speed_kmh)
            total_time += dt

            # Speed zone classification
            if speed_kmh < self.WALKING_MAX:
                metrics.walking_distance_m += distance
                rest_time += dt
            elif speed_kmh < self.JOGGING_MAX:
                metrics.jogging_distance_m += distance
            elif speed_kmh < self.HIGH_INTENSITY_MAX:
                metrics.high_intensity_distance_m += distance
                high_intensity_time += dt
            else:
                metrics.sprint_distance_m += distance
                high_intensity_time += dt

            # Sprint detection (continuous > 25 km/h for > 1 second)
            if speed_kmh >= self.HIGH_INTENSITY_MAX:
                if not in_sprint:
                    in_sprint = True
                    sprint_start_time = t0
                    sprint_distance = 0.0
                sprint_distance += distance
            else:
                if in_sprint:
                    sprint_duration = t0 - sprint_start_time
                    if sprint_duration >= 1.0:  # At least 1 second
                        metrics.sprint_count += 1
                    in_sprint = False

            # Acceleration / deceleration
            if i >= 2:
                t_prev, _, _ = trajectory[i - 2]
                dt_prev = t0 - t_prev
                if dt_prev > 0:
                    speed_prev_mps = self._compute_speed(trajectory[i - 2], trajectory[i - 1])
                    acceleration = (speed_mps - speed_prev_mps) / dt_prev
                    accelerations.append(acceleration)

                    if acceleration >= self.ACCELERATION_THRESHOLD:
                        metrics.acceleration_count += 1
                        metrics.peak_acceleration_mps2 = max(
                            metrics.peak_acceleration_mps2, acceleration
                        )
                    elif acceleration <= self.DECELERATION_THRESHOLD:
                        metrics.deceleration_count += 1
                        metrics.peak_deceleration_mps2 = min(
                            metrics.peak_deceleration_mps2, acceleration
                        )

        # Close any open sprint
        if in_sprint and trajectory:
            sprint_duration = trajectory[-1][0] - sprint_start_time
            if sprint_duration >= 1.0:
                metrics.sprint_count += 1

        # Summary stats
        if speeds:
            metrics.max_speed_kmh = max(speeds)
            metrics.avg_speed_kmh = sum(speeds) / len(speeds)

        # High intensity bouts (periods of >15 km/h)
        metrics.high_intensity_bouts = self._count_bouts(
            speeds, threshold=15.0, min_duration_frames=5
        )

        # Work-to-rest ratio
        if rest_time > 0:
            metrics.work_rest_ratio = round(high_intensity_time / rest_time, 2)
        else:
            metrics.work_rest_ratio = 999.0  # All work, no rest

        # Metabolic power estimate (simplified)
        # Based on di Prampero model: MP ≈ speed * (1 + 0.5 * (acceleration / g)²)
        # We use a simplified version: MP = sprint_distance * 2 + high_intensity_distance * 1.5 + jogging_distance
        metrics.metabolic_power_estimate = (
            metrics.sprint_distance_m * 2.0
            + metrics.high_intensity_distance_m * 1.5
            + metrics.jogging_distance_m * 1.0
            + metrics.walking_distance_m * 0.5
        )

        return metrics

    def _compute_speed(
        self, p1: tuple[float, float, float], p2: tuple[float, float, float]
    ) -> float:
        """Compute speed between two points in m/s."""
        dt = p2[0] - p1[0]
        if dt <= 0:
            return 0.0
        dx = p2[1] - p1[1]
        dy = p2[2] - p1[2]
        return math.sqrt(dx * dx + dy * dy) / dt

    def _count_bouts(self, speeds: list[float], threshold: float, min_duration_frames: int) -> int:
        """Count continuous periods above threshold speed."""
        bouts = 0
        in_bout = False
        bout_length = 0

        for speed in speeds:
            if speed >= threshold:
                if not in_bout:
                    in_bout = True
                    bout_length = 1
                else:
                    bout_length += 1
            else:
                if in_bout and bout_length >= min_duration_frames:
                    bouts += 1
                in_bout = False
                bout_length = 0

        if in_bout and bout_length >= min_duration_frames:
            bouts += 1

        return bouts

    async def compute_team_physical_summary(
        self, player_loads: dict[int, PhysicalLoadMetrics], player_teams: dict[int, str]
    ) -> dict[str, dict[str, Any]]:
        """Compute team-level physical summary."""
        team_data: dict[str, list[PhysicalLoadMetrics]] = {"home": [], "away": []}

        for tid, metrics in player_loads.items():
            team = player_teams.get(tid, "unknown")
            if team in team_data:
                team_data[team].append(metrics)

        summary = {}
        for team, players in team_data.items():
            if not players:
                continue

            n = len(players)
            summary[team] = {
                "players_analyzed": n,
                "avg_distance_m": round(sum(p.total_distance_m for p in players) / n, 1),
                "avg_sprint_distance_m": round(sum(p.sprint_distance_m for p in players) / n, 1),
                "avg_sprint_count": round(sum(p.sprint_count for p in players) / n, 1),
                "avg_max_speed_kmh": round(sum(p.max_speed_kmh for p in players) / n, 2),
                "avg_acceleration_count": round(sum(p.acceleration_count for p in players) / n, 1),
                "avg_deceleration_count": round(sum(p.deceleration_count for p in players) / n, 1),
                "total_sprints": sum(p.sprint_count for p in players),
                "total_distance_km": round(sum(p.total_distance_m for p in players) / 1000, 2),
                "work_rest_ratio": round(sum(p.work_rest_ratio for p in players) / n, 2),
                "metabolic_power_total": round(sum(p.metabolic_power_estimate for p in players), 1),
            }

        return summary
