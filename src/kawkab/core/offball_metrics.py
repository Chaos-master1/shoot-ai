"""Off-ball player movement metrics.

Analyzes player movement without the ball: space creation runs,
defensive positioning quality, support run classification, and
movement efficiency.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OffBallPlayerMetrics:
    """Off-ball metrics for a single player."""

    track_id: int = 0
    total_distance_without_ball_m: float = 0.0
    high_speed_runs_without_ball: int = 0
    space_creation_runs: int = 0
    avg_defensive_distance_to_ball_m: float = 0.0
    avg_offensive_support_distance_m: float = 0.0
    movement_efficiency: float = 0.0
    decoy_runs: int = 0
    time_in_high_activity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tid": self.track_id,
            "dist_no_ball": round(self.total_distance_without_ball_m, 1),
            "high_speed_runs": self.high_speed_runs_without_ball,
            "space_creation": self.space_creation_runs,
            "avg_def_dist": round(self.avg_defensive_distance_to_ball_m, 1),
            "avg_support_dist": round(self.avg_offensive_support_distance_m, 1),
            "movement_eff": round(self.movement_efficiency, 2),
            "decoy_runs": self.decoy_runs,
            "high_activity_s": round(self.time_in_high_activity, 1),
        }


@dataclass
class OffBallMatchReport:
    """Aggregate off-ball metrics for a team."""

    team: str = ""
    players: dict[int, OffBallPlayerMetrics] = field(default_factory=dict)
    team_total_dist_no_ball_km: float = 0.0
    team_space_creation_runs: int = 0
    team_avg_efficiency: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "team": self.team,
            "players": {str(k): v.to_dict() for k, v in self.players.items()},
            "team_dist_no_ball_km": round(self.team_total_dist_no_ball_km, 2),
            "team_space_creation": self.team_space_creation_runs,
            "team_avg_efficiency": round(self.team_avg_efficiency, 2),
        }


class OffBallAnalyzer:
    """Analyzes off-ball player movement.

    Usage:
        oba = OffBallAnalyzer()
        report = oba.analyze_offball(frames_data, team="home")
    """

    HIGH_SPEED_THRESHOLD_MS = 5.5  # ~20 km/h
    SPACE_CREATION_SPEED_THRESHOLD_MS = 4.0
    SPACE_CREATION_DISTANCE_M = 5.0

    def analyze_offball(
        self,
        frames: list[dict[str, Any]],
        team: str = "home",
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
    ) -> OffBallMatchReport:
        """Compute off-ball metrics for all players on a team.

        Args:
            frames: List of dicts with "timestamp", "possession" (bool),
                    "home_positions" (list of (x,y, track_id)),
                    "away_positions", "ball_pos" (x,y) or None.
            team: "home" or "away".
            pitch_length: Pitch length in meters.
            pitch_width: Pitch width in meters.

        Returns:
            OffBallMatchReport with per-player and team stats.
        """
        if not frames:
            return OffBallMatchReport(team=team)

        player_positions: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
        possession_by_frame: list[bool] = []
        ball_positions: list[tuple[float, float] | None] = []

        for fdata in frames:
            ts = fdata.get("timestamp", 0.0)
            possession = fdata.get("possession", False)
            ball = fdata.get("ball_pos")
            ball_positions.append(ball)
            possession_by_frame.append(possession)

            positions = fdata.get(f"{team}_positions", [])
            for item in positions:
                if len(item) == 3:
                    x, y, tid = item
                elif len(item) == 2:
                    continue
                else:
                    continue
                player_positions[tid].append((ts, x, y))

        player_metrics: dict[int, OffBallPlayerMetrics] = {}

        for tid, positions in player_positions.items():
            if len(positions) < 5:
                continue

            total_dist = 0.0
            high_speed_runs = 0
            space_creation = 0
            decoy_runs = 0
            def_dist_sum = 0.0
            def_dist_count = 0
            support_dist_sum = 0.0
            support_dist_count = 0
            high_activity_frames = 0

            prev_ts = positions[0][0]
            prev_x = positions[0][1]
            prev_y = positions[0][2]

            for i in range(1, len(positions)):
                ts, x, y = positions[i]
                dt = ts - prev_ts
                if dt <= 0:
                    continue

                dx = x - prev_x
                dy = y - prev_y
                dist = math.sqrt(dx * dx + dy * dy)
                total_dist += dist

                speed = dist / dt

                if speed >= self.HIGH_SPEED_THRESHOLD_MS:
                    high_speed_runs += 1

                frame_idx = min(i, len(ball_positions) - 1)
                ball = ball_positions[frame_idx]
                possession = (
                    possession_by_frame[frame_idx]
                    if frame_idx < len(possession_by_frame)
                    else False
                )

                if ball and not possession:
                    def_dist = math.sqrt((x - ball[0]) ** 2 + (y - ball[1]) ** 2)
                    def_dist_sum += def_dist
                    def_dist_count += 1

                    if (
                        speed >= self.SPACE_CREATION_SPEED_THRESHOLD_MS
                        and dist >= self.SPACE_CREATION_DISTANCE_M
                    ):
                        # Moving away from ball while defending = space creation
                        prev_ball_dist = math.sqrt(
                            (prev_x - ball[0]) ** 2 + (prev_y - ball[1]) ** 2
                        )
                        curr_ball_dist = math.sqrt((x - ball[0]) ** 2 + (y - ball[1]) ** 2)
                        if curr_ball_dist > prev_ball_dist * 1.1:
                            space_creation += 1

                    if speed >= self.SPACE_CREATION_SPEED_THRESHOLD_MS:
                        high_activity_frames += 1

                if ball and possession:
                    support_dist = math.sqrt((x - ball[0]) ** 2 + (y - ball[1]) ** 2)
                    support_dist_sum += support_dist
                    support_dist_count += 1

                    # Decoy run: moving away from ball while teammate has it
                    if support_dist > 15.0 and speed >= self.SPACE_CREATION_SPEED_THRESHOLD_MS:
                        prev_ball_dist = math.sqrt(
                            (prev_x - ball[0]) ** 2 + (prev_y - ball[1]) ** 2
                        )
                        curr_ball_dist = math.sqrt((x - ball[0]) ** 2 + (y - ball[1]) ** 2)
                        if curr_ball_dist > prev_ball_dist * 1.05:
                            decoy_runs += 1

                prev_ts = ts
                prev_x = x
                prev_y = y

            avg_def_dist = def_dist_sum / max(def_dist_count, 1)
            avg_support_dist = support_dist_sum / max(support_dist_count, 1)
            n_frames = len(positions)
            movement_eff = 1.0 - (total_dist / max(n_frames, 1)) / 2.0 if n_frames > 0 else 0.0
            movement_eff = max(0.0, min(1.0, movement_eff))

            player_metrics[tid] = OffBallPlayerMetrics(
                track_id=tid,
                total_distance_without_ball_m=total_dist,
                high_speed_runs_without_ball=high_speed_runs,
                space_creation_runs=space_creation,
                avg_defensive_distance_to_ball_m=avg_def_dist,
                avg_offensive_support_distance_m=avg_support_dist,
                movement_efficiency=movement_eff,
                decoy_runs=decoy_runs,
                time_in_high_activity=(
                    (high_activity_frames / max(len(frames), 1))
                    * (frames[-1]["timestamp"] - frames[0]["timestamp"])
                )
                if len(frames) > 1
                else 0,
            )

        if not player_metrics:
            return OffBallMatchReport(team=team)

        total_dist_km = (
            sum(p.total_distance_without_ball_m for p in player_metrics.values()) / 1000.0
        )
        total_space = sum(p.space_creation_runs for p in player_metrics.values())
        avg_eff = sum(p.movement_efficiency for p in player_metrics.values()) / len(player_metrics)

        return OffBallMatchReport(
            team=team,
            players=player_metrics,
            team_total_dist_no_ball_km=total_dist_km,
            team_space_creation_runs=total_space,
            team_avg_efficiency=avg_eff,
        )
