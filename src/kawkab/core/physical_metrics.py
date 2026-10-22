"""Physical performance metrics from tracking data.

Computes professional-grade physical metrics:
- Distance covered (total, per minute, by speed zone)
- Sprint count and distance
- High-intensity runs (>5.5 m/s = 20 km/h)
- Acceleration/deceleration counts
- Metabolic power (estimated from instantaneous speed)
- Player load (accelerometer-derived workload estimate)

All numpy-only, no external dependencies beyond numpy.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Speed zones in m/s (from professional sports science conventions)
SPEED_ZONES = {
    "walking": (0.0, 1.7),  # 0-6 km/h
    "jogging": (1.7, 3.3),  # 6-12 km/h
    "running": (3.3, 5.5),  # 12-20 km/h
    "high_intensity": (5.5, 7.0),  # 20-25 km/h
    "sprinting": (7.0, float("inf")),  # >25 km/h
}

# Metabolic power constants (di Prampero et al.)
# Energy cost of horizontal running: ~3.6 J/kg/m
# Energy cost of acceleration: proportional to acceleration²
RUNNING_COST = 3.6  # J/kg/m
ACCEL_COST_FACTOR = 1.2  # J/kg per (m/s²)²


@dataclass
class PlayerPhysicalMetrics:
    """Physical metrics for a single player."""

    track_id: int = 0
    total_distance_m: float = 0.0
    distance_by_zone: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(SPEED_ZONES, 0.0)
    )
    max_speed_ms: float = 0.0
    avg_speed_ms: float = 0.0
    sprint_count: int = 0
    sprint_distance_m: float = 0.0
    high_intensity_runs: int = 0
    high_intensity_distance_m: float = 0.0
    acceleration_count: int = 0  # >3 m/s²
    deceleration_count: int = 0  # <-3 m/s²
    metabolic_power_avg_w_kg: float = 0.0
    metabolic_power_peak_w_kg: float = 0.0
    player_load: float = 0.0  # Arbitrary workload units
    distance_per_minute_m: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tid": self.track_id,
            "total_dist": round(self.total_distance_m, 1),
            "dist_by_zone": {k: round(v, 1) for k, v in self.distance_by_zone.items()},
            "max_speed": round(self.max_speed_ms * 3.6, 1),  # km/h
            "avg_speed": round(self.avg_speed_ms * 3.6, 1),
            "sprints": self.sprint_count,
            "sprint_dist": round(self.sprint_distance_m, 1),
            "hi_runs": self.high_intensity_runs,
            "hi_dist": round(self.high_intensity_distance_m, 1),
            "accels": self.acceleration_count,
            "decels": self.deceleration_count,
            "met_power_avg": round(self.metabolic_power_avg_w_kg, 1),
            "met_power_peak": round(self.metabolic_power_peak_w_kg, 1),
            "player_load": round(self.player_load, 1),
            "dist_per_min": round(self.distance_per_minute_m, 1),
        }


@dataclass
class TeamPhysicalReport:
    """Aggregate physical metrics for a team."""

    team: str = ""
    players: dict[int, PlayerPhysicalMetrics] = field(default_factory=dict)
    total_distance_m: float = 0.0
    avg_distance_per_player_m: float = 0.0
    total_sprints: int = 0
    total_high_intensity_runs: int = 0
    team_avg_speed_ms: float = 0.0
    team_peak_speed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "team": self.team,
            "players": {str(k): v.to_dict() for k, v in self.players.items()},
            "total_dist_km": round(self.total_distance_m / 1000, 2),
            "avg_dist_per_player_km": round(self.avg_distance_per_player_m / 1000, 2),
            "total_sprints": self.total_sprints,
            "total_hi_runs": self.total_high_intensity_runs,
            "team_avg_speed": round(self.team_avg_speed_ms * 3.6, 1),
            "team_peak_speed": round(self.team_peak_speed_ms * 3.6, 1),
        }


class PhysicalMetricsAnalyzer:
    """Computes physical performance metrics from tracking data.

    Analyzes player movement across tracking frames to compute
    distance, speed, acceleration, sprint, and metabolic metrics.

    Usage:
        pma = PhysicalMetricsAnalyzer()
        report = pma.analyze_team(frames_data, team="home")
    """

    SPRINT_THRESHOLD_MS = 7.0  # 25 km/h
    HIGH_INTENSITY_THRESHOLD_MS = 5.5  # 20 km/h
    ACCEL_THRESHOLD = 3.0  # m/s²

    def analyze_player(
        self,
        trajectory: list[tuple[float, float, float]],  # (timestamp, x, y)
        body_mass_kg: float = 75.0,
    ) -> PlayerPhysicalMetrics:
        """Compute physical metrics for a single player's trajectory.

        Args:
            trajectory: List of (timestamp, x, y) tuples sorted by time.
            body_mass_kg: Player body mass for metabolic power estimation.

        Returns:
            PlayerPhysicalMetrics with all computed values.
        """
        if len(trajectory) < 3:
            return PlayerPhysicalMetrics(track_id=0)

        # Extract arrays
        ts = np.array([p[0] for p in trajectory], dtype=np.float64)
        xs = np.array([p[1] for p in trajectory], dtype=np.float64)
        ys = np.array([p[2] for p in trajectory], dtype=np.float64)

        # Time deltas
        dt = np.diff(ts)
        dt = np.where(dt < 0.01, 0.01, dt)

        # Displacements and speeds
        dx = np.diff(xs)
        dy = np.diff(ys)
        dist = np.sqrt(dx**2 + dy**2)
        speeds = dist / dt  # m/s

        # Smooth speeds with 3-point moving average
        if len(speeds) >= 3:
            window = np.ones(3) / 3
            speeds_smooth = np.convolve(speeds, window, mode="same")
        else:
            speeds_smooth = speeds

        # Accelerations (from smoothed speeds)
        accels = np.diff(speeds_smooth) / dt[1:] if len(speeds_smooth) > 1 else np.array([0.0])

        # --- Metrics ---
        total_distance = float(np.sum(dist))
        duration = float(ts[-1] - ts[0])

        # Distance by speed zone
        dist_by_zone = dict.fromkeys(SPEED_ZONES, 0.0)
        for i, speed in enumerate(speeds_smooth):
            for zone_name, (lo, hi) in SPEED_ZONES.items():
                if lo <= speed < hi:
                    dist_by_zone[zone_name] += float(dist[i])
                    break

        max_speed = float(np.max(speeds_smooth))
        avg_speed = float(np.mean(speeds_smooth))

        # Sprint detection
        in_sprint = False
        sprint_count = 0
        for speed in speeds_smooth:
            if speed >= self.SPRINT_THRESHOLD_MS and not in_sprint:
                in_sprint = True
                sprint_count += 1
            elif speed < self.SPRINT_THRESHOLD_MS and in_sprint:
                in_sprint = False

        # Sprint distance
        sprint_mask = speeds_smooth >= self.SPRINT_THRESHOLD_MS
        sprint_dist = float(np.sum(dist[sprint_mask[: len(dist)]])) if np.any(sprint_mask) else 0.0

        # High intensity runs
        hi_mask = speeds_smooth >= self.HIGH_INTENSITY_THRESHOLD_MS
        hi_runs = 0
        in_hi = False
        for m in hi_mask:
            if m and not in_hi:
                hi_runs += 1
                in_hi = True
            elif not m:
                in_hi = False
        hi_dist = float(np.sum(dist[hi_mask[: len(dist)]])) if np.any(hi_mask) else 0.0

        # Acceleration/deceleration counts
        accel_count = int(np.sum(accels > self.ACCEL_THRESHOLD))
        decel_count = int(np.sum(accels < -self.ACCEL_THRESHOLD))

        # Metabolic power (di Prampero model)
        # P_met = Running_cost * speed + Accel_cost * |accel| * speed  (simplified)
        # Pad accels to match speeds_smooth length (accels has 1 fewer element)
        if len(accels) < len(speeds_smooth):
            accels_padded = np.pad(accels, (0, len(speeds_smooth) - len(accels)), mode="edge")
        else:
            accels_padded = accels
        met_powers = RUNNING_COST * speeds_smooth + ACCEL_COST_FACTOR * np.abs(accels_padded)

        met_power_avg = float(np.mean(met_powers)) if len(met_powers) > 0 else 0.0
        met_power_peak = float(np.max(met_powers)) if len(met_powers) > 0 else 0.0

        # Player load: sum of absolute speed differences
        if len(speeds_smooth) > 1:
            speed_diffs = np.diff(speeds_smooth)
            player_load = float(np.sum(np.abs(speed_diffs)))
        else:
            player_load = 0.0

        # Distance per minute
        dist_per_min = (total_distance / max(duration, 1)) * 60.0 if duration > 0 else 0.0

        metrics = PlayerPhysicalMetrics(track_id=0)
        metrics.total_distance_m = total_distance
        metrics.distance_by_zone = dist_by_zone
        metrics.max_speed_ms = max_speed
        metrics.avg_speed_ms = avg_speed
        metrics.sprint_count = sprint_count
        metrics.sprint_distance_m = sprint_dist
        metrics.high_intensity_runs = hi_runs
        metrics.high_intensity_distance_m = hi_dist
        metrics.acceleration_count = accel_count
        metrics.deceleration_count = decel_count
        metrics.metabolic_power_avg_w_kg = met_power_avg
        metrics.metabolic_power_peak_w_kg = met_power_peak
        metrics.player_load = player_load
        metrics.distance_per_minute_m = dist_per_min

        return metrics

    def analyze_team(
        self,
        frames: list[dict[str, Any]],
        team: str = "home",
        player_track_ids: dict[int, float] | None = None,
    ) -> TeamPhysicalReport:
        """Compute physical metrics for all players on a team.

        Args:
            frames: List of frame dicts with tracking data.
            team: "home" or "away".
            player_track_ids: Optional dict of track_id -> body_mass_kg.

        Returns:
            TeamPhysicalReport with per-player and aggregate metrics.
        """
        if not frames:
            return TeamPhysicalReport(team=team)

        # Extract per-player trajectories
        trajectories: dict[int, list[tuple[float, float, float]]] = defaultdict(list)

        for fdata in frames:
            ts = fdata.get("timestamp", 0.0)
            positions = fdata.get(f"{team}_positions", [])
            for item in positions:
                if len(item) >= 3:
                    x, y, tid = item[0], item[1], item[2]
                    trajectories[tid].append((ts, x, y))

        if not trajectories:
            return TeamPhysicalReport(team=team)

        # Compute per-player metrics
        players: dict[int, PlayerPhysicalMetrics] = {}
        for tid, traj in trajectories.items():
            if len(traj) < 3:
                continue
            mass = (player_track_ids or {}).get(tid, 75.0)
            metrics = self.analyze_player(traj, body_mass_kg=mass)
            metrics.track_id = tid
            players[tid] = metrics

        if not players:
            return TeamPhysicalReport(team=team)

        # Aggregate
        total_dist = sum(p.total_distance_m for p in players.values())
        total_sprints = sum(p.sprint_count for p in players.values())
        total_hi = sum(p.high_intensity_runs for p in players.values())
        avg_speed = sum(p.avg_speed_ms for p in players.values()) / len(players)
        peak_speed = max(p.max_speed_ms for p in players.values())

        return TeamPhysicalReport(
            team=team,
            players=players,
            total_distance_m=total_dist,
            avg_distance_per_player_m=total_dist / len(players),
            total_sprints=total_sprints,
            total_high_intensity_runs=total_hi,
            team_avg_speed_ms=avg_speed,
            team_peak_speed_ms=peak_speed,
        )
