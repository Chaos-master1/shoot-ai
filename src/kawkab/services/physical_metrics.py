"""Player physical metrics from tracking data.

Computes per-player:
  - Total distance run (m)
  - Average speed (km/h), max speed (km/h)
  - Sprint count (> 25 km/h), sprint distance
  - High intensity runs (> 20 km/h)
  - Jog/walk/sprint time breakdown
  - Distance per half

Requires homography-calibrated track positions (pixel → meter).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import dist

logger = logging.getLogger("physical_metrics")

SPRINT_THRESHOLD_KMH = 25.0
HIGH_INTENSITY_THRESHOLD_KMH = 20.0
JOG_THRESHOLD_KMH = 7.0
WALK_THRESHOLD_KMH = 3.0
WINDOW_SIZE = 5  # frames for speed smoothing


@dataclass
class PlayerPhysicalProfile:
    track_id: int
    total_distance_m: float = 0.0
    avg_speed_kmh: float = 0.0
    max_speed_kmh: float = 0.0
    sprint_count: int = 0
    sprint_distance_m: float = 0.0
    high_intensity_distance_m: float = 0.0
    jogging_distance_m: float = 0.0
    walking_distance_m: float = 0.0
    standing_time_s: float = 0.0
    total_active_time_s: float = 0.0
    avg_heart_rate_estimate: float = 0.0  # placeholder
    per_half_distance: list[float] = field(default_factory=lambda: [0.0, 0.0])
    per_half_time: list[float] = field(default_factory=lambda: [0.0, 0.0])


def compute_physical_metrics(
    track_positions: dict[int, list[tuple[int, float, float, float]]],
    fps: float = 25.0,
    half_duration_s: float = 2700.0,
) -> dict[int, PlayerPhysicalProfile]:
    """Compute physical metrics per player.

    Args:
        track_positions: {track_id: [(frame, x_m, y_m, timestamp)]}
        fps: video frame rate
        half_duration_s: duration of each half in seconds

    Returns:
        Dict mapping track_id → PlayerPhysicalProfile
    """
    profiles: dict[int, PlayerPhysicalProfile] = {}
    for tid, positions in track_positions.items():
        if len(positions) < 10:
            continue
        positions.sort(key=lambda p: p[0])
        speeds_kmh = []
        total_dist = 0.0
        sprint_count = 0
        sprint_dist = 0.0
        high_int_dist = 0.0
        jog_dist = 0.0
        walk_dist = 0.0
        active_time = 0.0
        in_sprint = False

        half_dists = [0.0, 0.0]
        half_times = [0.0, 0.0]

        total_time = positions[-1][3] - positions[0][3]

        for i in range(1, len(positions)):
            _, x1, y1, t1 = positions[i - 1]
            _, x2, y2, t2 = positions[i]
            dt = max(t2 - t1, 0.001)
            d = dist((x1, y1), (x2, y2))
            speed_ms = d / dt
            speed_kmh = speed_ms * 3.6
            speeds_kmh.append(speed_kmh)
            total_dist += d

            half_idx = 0 if t2 < half_duration_s else 1
            if half_idx < 2:
                half_dists[half_idx] += d
                half_times[half_idx] += dt

            if speed_kmh > SPRINT_THRESHOLD_KMH:
                sprint_dist += d
                high_int_dist += d
                if not in_sprint:
                    sprint_count += 1
                    in_sprint = True
            elif speed_kmh > HIGH_INTENSITY_THRESHOLD_KMH:
                high_int_dist += d
                in_sprint = False
            elif speed_kmh > JOG_THRESHOLD_KMH:
                jog_dist += d
                in_sprint = False
            elif speed_kmh > WALK_THRESHOLD_KMH:
                walk_dist += d
                in_sprint = False
            else:
                in_sprint = False

            if speed_kmh > JOG_THRESHOLD_KMH:
                active_time += dt

        avg_speed = sum(speeds_kmh) / max(len(speeds_kmh), 1)
        max_speed = max(speeds_kmh) if speeds_kmh else 0.0

        profiles[tid] = PlayerPhysicalProfile(
            track_id=tid,
            total_distance_m=round(total_dist, 1),
            avg_speed_kmh=round(avg_speed, 1),
            max_speed_kmh=round(max_speed, 1),
            sprint_count=sprint_count,
            sprint_distance_m=round(sprint_dist, 1),
            high_intensity_distance_m=round(high_int_dist, 1),
            jogging_distance_m=round(jog_dist, 1),
            walking_distance_m=round(walk_dist, 1),
            standing_time_s=round(max(0.0, total_time - active_time), 1),
            total_active_time_s=round(active_time, 1),
            per_half_distance=[round(d, 1) for d in half_dists],
            per_half_time=[round(t, 1) for t in half_times],
        )

    return profiles
