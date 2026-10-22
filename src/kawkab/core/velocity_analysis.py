"""Acceleration / Velocity Analysis.

Computes player velocity, acceleration, sprint detection,
fatigue index, and per-team aggregates. All numpy-only.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class VelocityAnalyzer:
    def compute_player_velocity(
        self, player_trajectory: list[tuple[float, float, float]]
    ) -> dict[str, Any]:
        if len(player_trajectory) < 2:
            return {"velocities": [], "accelerations": [], "avg_speed": 0.0, "max_speed": 0.0}
        ts = np.array([p[0] for p in player_trajectory])
        xs = np.array([p[1] for p in player_trajectory])
        ys = np.array([p[2] for p in player_trajectory])
        dt = np.diff(ts)
        dt = np.where(dt < 0.01, 0.01, dt)
        dx = np.diff(xs)
        dy = np.diff(ys)
        speeds = np.sqrt(dx**2 + dy**2) / dt
        if len(speeds) >= 3:
            window = np.ones(3) / 3
            speeds_smooth = np.convolve(speeds, window, mode="same")
        else:
            speeds_smooth = speeds
        accelerations = (
            np.diff(speeds_smooth) / dt[1:] if len(speeds_smooth) > 1 else np.array([0.0])
        )
        return {
            "velocities": [round(float(v), 2) for v in speeds_smooth],
            "accelerations": [round(float(a), 2) for a in accelerations],
            "avg_speed": round(float(np.mean(speeds_smooth)), 2),
            "max_speed": round(float(np.max(speeds_smooth)), 2),
        }

    def analyze_sprints(
        self, velocity_profile: list[float], threshold: float = 7.0
    ) -> dict[str, Any]:
        if not velocity_profile:
            return {
                "sprint_count": 0,
                "avg_duration_s": 0.0,
                "max_speed": 0.0,
                "distance_covered_m": 0.0,
            }
        in_sprint = False
        sprint_start = 0
        sprints: list[float] = []
        for i, v in enumerate(velocity_profile):
            if v > threshold and not in_sprint:
                in_sprint = True
                sprint_start = i
            elif v <= threshold and in_sprint:
                in_sprint = False
                duration = i - sprint_start
                if duration > 0:
                    sprints.append(duration)
        if in_sprint:
            sprints.append(len(velocity_profile) - sprint_start)
        sprints = [s for s in sprints if s > 0]
        return {
            "sprint_count": len(sprints),
            "avg_duration_s": round(sum(sprints) / len(sprints), 2) if sprints else 0.0,
            "max_speed": round(max(velocity_profile), 2),
            "distance_covered_m": round(sum(velocity_profile), 1),
        }

    def compute_acceleration_zones(
        self, player_trajectory: list[tuple[float, float, float]]
    ) -> dict[str, Any]:
        result = self.compute_player_velocity(player_trajectory)
        accels = result.get("accelerations", [])
        high = sum(1 for a in accels if a > 3.0)
        moderate = sum(1 for a in accels if 1.0 <= a <= 3.0)
        low = sum(1 for a in accels if a < 1.0)
        return {"high_intensity": high, "moderate": moderate, "low": low, "total": len(accels)}

    def analyze_team_velocity(
        self, players_trajectories: dict[str | int, list[tuple[float, float, float]]]
    ) -> dict[str, Any]:
        if not players_trajectories:
            return {
                "avg_speed": 0.0,
                "max_speed": 0.0,
                "total_sprints": 0,
                "total_distance_m": 0.0,
                "acceleration_profile": {"high_intensity": 0, "moderate": 0, "low": 0},
            }
        all_speeds: list[float] = []
        total_sprints = 0
        total_distance = 0.0
        acc_profile: dict[str, int] = {"high_intensity": 0, "moderate": 0, "low": 0}
        for _pid, traj in players_trajectories.items():
            pv = self.compute_player_velocity(traj)
            all_speeds.extend(pv["velocities"])
            sprints = self.analyze_sprints(pv["velocities"])
            total_sprints += sprints["sprint_count"]
            total_distance += sprints["distance_covered_m"]
            zones = self.compute_acceleration_zones(traj)
            for k in acc_profile:
                acc_profile[k] += zones[k]
        avg_speed = sum(all_speeds) / len(all_speeds) if all_speeds else 0.0
        max_speed = max(all_speeds) if all_speeds else 0.0
        return {
            "avg_speed": round(avg_speed, 2),
            "max_speed": round(max_speed, 2),
            "total_sprints": total_sprints,
            "total_distance_m": round(total_distance, 1),
            "acceleration_profile": acc_profile,
        }

    def compute_fatigue_index(
        self, velocity_profile: list[float], window_minutes: int = 5
    ) -> dict[str, Any]:
        if len(velocity_profile) < 2:
            return {
                "fatigue_index": 0.0,
                "peak_window_speed": 0.0,
                "final_window_speed": 0.0,
                "fatigue_level": "none",
            }
        window_size = max(window_minutes, 1)
        n = len(velocity_profile)
        window_avgs: list[float] = []
        for i in range(0, n, window_size):
            chunk = velocity_profile[i : i + window_size]
            window_avgs.append(sum(chunk) / len(chunk))
        if len(window_avgs) < 2:
            return {
                "fatigue_index": 0.0,
                "peak_window_speed": float(window_avgs[0]),
                "final_window_speed": float(window_avgs[0]),
                "fatigue_level": "none",
            }
        peak = max(window_avgs)
        final = window_avgs[-1]
        fatigue_index = ((peak - final) / peak * 100) if peak > 0 else 0.0
        if fatigue_index > 20:
            level = "high"
        elif fatigue_index > 10:
            level = "moderate"
        else:
            level = "low"
        return {
            "fatigue_index": round(fatigue_index, 1),
            "peak_window_speed": round(peak, 2),
            "final_window_speed": round(final, 2),
            "fatigue_level": level,
        }

    def generate_velocity_report(
        self, events: list[dict[str, Any]], tracking_data: dict[str, Any]
    ) -> dict[str, Any]:
        if not tracking_data:
            return {"error": "No tracking data provided"}
        players_traj: dict[int | str, list[tuple[float, float, float]]] = {}
        for frame in tracking_data.get("frames", []):
            ts = frame.get("timestamp", 0)
            for det in frame.get("detections", []):
                if det.get("class_name") != "person":
                    continue
                tid = det.get("track_id")
                if tid is None:
                    continue
                if tid not in players_traj:
                    players_traj[tid] = []
                players_traj[tid].append((ts, det.get("x", 0), det.get("y", 0)))
        team_velocity = self.analyze_team_velocity(players_traj)
        player_reports: dict[str, Any] = {}
        for tid, traj in players_traj.items():
            pv = self.compute_player_velocity(traj)
            sprints = self.analyze_sprints(pv["velocities"])
            fatigue = self.compute_fatigue_index(pv["velocities"])
            zones = self.compute_acceleration_zones(traj)
            player_reports[str(tid)] = {
                "avg_speed": pv["avg_speed"],
                "max_speed": pv["max_speed"],
                "sprint_count": sprints["sprint_count"],
                "distance_covered_m": sprints["distance_covered_m"],
                "fatigue_index": fatigue["fatigue_index"],
                "fatigue_level": fatigue["fatigue_level"],
                "acceleration_zones": zones,
            }
        return {"team": team_velocity, "players": player_reports}
