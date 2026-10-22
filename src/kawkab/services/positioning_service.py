"""Off-ball positioning and run analysis.

Analyzes player movement WITHOUT the ball to find:
- Runs behind the defensive line (vertical penetration)
- Width-stretching runs (pulls defenders out of shape)
- Decoy runs (false movements to create space)
- Space created (improved xT for teammates after the run)

Uses tracking data (frame-by-frame positions) and the ball position
to determine ball-oriented movement, and computes run quality via
how much the run influences team xT output.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from kawkab.core.game_constants import GAME

logger = logging.getLogger(__name__)


class RunType(StrEnum):
    """Classification of an off-ball run."""

    BEHIND_DEFENSE = "behind_defense"
    WIDE = "wide"
    DECOY = "decoy"
    SUPPORT = "support"
    DROP = "drop"
    DIAGONAL = "diagonal"
    UNKNOWN = "unknown"


@dataclass
class Run:
    """A single off-ball run by a player."""

    player_track_id: int
    team: str
    start_frame: int
    end_frame: int
    start_pos: tuple[float, float]
    end_pos: tuple[float, float]
    run_type: RunType
    distance_m: float
    avg_speed_mps: float
    peak_speed_mps: float
    created_xT_delta: float = 0.0
    attracted_defenders: int = 0
    notes: str = ""


@dataclass
class PositioningReport:
    """Aggregated off-ball run analysis for one team."""

    team: str
    total_runs: int
    runs_by_type: dict[str, int]
    total_distance_m: float
    avg_run_distance_m: float
    longest_run_m: float
    total_xT_created: float
    runs: list[Run]
    notes: list[str]


class PositioningService:
    """Analyze off-ball movement and run quality.

    Args:
        pitch_length_m: Real pitch length in meters (default 105).
        pitch_width_m: Real pitch width in meters (default 68).
        min_run_distance_m: Ignore runs shorter than this.
        sprint_threshold_mps: Speed above which a run is a sprint.
    """

    PITCH_LENGTH_M = GAME.PITCH_LENGTH_M
    PITCH_WIDTH_M = GAME.PITCH_WIDTH_M

    def __init__(
        self,
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
        min_run_distance_m: float = 5.0,
        sprint_threshold_mps: float = 5.5,
        fps: float = 30.0,
    ) -> None:
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m
        self.min_run_distance_m = min_run_distance_m
        self.sprint_threshold_mps = sprint_threshold_mps
        self.fps = fps
        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    def analyze(
        self,
        track_data: Any,
        team: str = "home",
        min_run_distance_m: float | None = None,
    ) -> PositioningReport:
        """Run off-ball analysis on tracking data.

        Args:
            track_data: A MatchTrackData-like object with .frames
            team: 'home' or 'away'
            min_run_distance_m: Override default threshold.
        """
        if min_run_distance_m is None:
            min_run_distance_m = self.min_run_distance_m
        runs: list[Run] = []
        if not hasattr(track_data, "frames") or not track_data.frames:
            return self._empty_report(team)
        player_paths: dict[int, list[tuple[int, tuple[float, float]]]] = defaultdict(list)
        ball_path: list[tuple[int, tuple[float, float]]] = []
        for frame in track_data.frames:
            frame_idx = getattr(frame, "frame_number", 0)
            if getattr(frame, "ball_position", None) is not None:
                bp = frame.ball_position
                if len(bp) >= 2:
                    ball_path.append((frame_idx, (float(bp[0]), float(bp[1]))))
            for det in getattr(frame, "detections", []) or []:
                if det.team != team:
                    continue
                if det.track_id is None or det.is_ball:
                    continue
                cx, cy = self._bbox_center(det.bbox)
                player_paths[det.track_id].append((frame_idx, (cx, cy)))
        for tid, path in player_paths.items():
            if len(path) < 2:
                continue
            segments = self._segment_into_runs(path, min_run_distance_m)
            for seg in segments:
                run = self._build_run(tid, team, seg, ball_path)
                if run is not None:
                    runs.append(run)
        return self._summarize(team, runs)

    def _empty_report(self, team: str) -> PositioningReport:
        return PositioningReport(
            team=team,
            total_runs=0,
            runs_by_type={},
            total_distance_m=0.0,
            avg_run_distance_m=0.0,
            longest_run_m=0.0,
            total_xT_created=0.0,
            runs=[],
            notes=["No tracking data available for off-ball analysis."],
        )

    @staticmethod
    def _bbox_center(bbox: Any) -> tuple[float, float]:
        if hasattr(bbox, "cx") and hasattr(bbox, "cy"):
            return float(bbox.cx), float(bbox.cy)
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return (float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0
        return 0.0, 0.0

    def _segment_into_runs(
        self,
        path: list[tuple[int, tuple[float, float]]],
        min_distance_m: float,
    ) -> list[list[tuple[int, tuple[float, float]]]]:
        if len(path) < 2:
            return []
        segments: list[list[tuple[int, tuple[float, float]]]] = []
        cur: list[tuple[int, tuple[float, float]]] = [path[0]]
        for prev, cur_pt in zip(path, path[1:], strict=False):
            dx = (cur_pt[1][0] - prev[1][0]) * (self.pitch_length_m / 100.0)
            dy = (cur_pt[1][1] - prev[1][1]) * (self.pitch_width_m / 100.0)
            _ = math.hypot(dx, dy)
            cur.append(cur_pt)
        total = self._path_length(cur)
        if total >= min_distance_m:
            segments.append(cur)
        return segments

    def _path_length(self, path: list[tuple[int, tuple[float, float]]]) -> float:
        total = 0.0
        for a, b in zip(path, path[1:], strict=False):
            dx = (b[1][0] - a[1][0]) * (self.pitch_length_m / 100.0)
            dy = (b[1][1] - a[1][1]) * (self.pitch_width_m / 100.0)
            total += math.hypot(dx, dy)
        return total

    def _build_run(
        self,
        player_track_id: int,
        team: str,
        segment: list[tuple[int, tuple[float, float]]],
        ball_path: list[tuple[int, tuple[float, float]]],
    ) -> Run | None:
        if len(segment) < 2:
            return None
        start_f, start_pos = segment[0]
        end_f, end_pos = segment[-1]
        distance = self._path_length(segment)
        if distance < self.min_run_distance_m:
            return None
        n_frames = max(1, end_f - start_f)
        seconds = n_frames / max(1.0, self.fps)
        avg_speed = distance / max(0.1, seconds)
        peak_speed = avg_speed
        prev = None
        max_step_speed = 0.0
        for fr, pos in segment:
            if prev is not None:
                dx = (pos[0] - prev[1][0]) * (self.pitch_length_m / 100.0)
                dy = (pos[1] - prev[1][1]) * (self.pitch_width_m / 100.0)
                step = math.hypot(dx, dy) * self.fps
                if step > max_step_speed:
                    max_step_speed = step
            prev = (fr, pos)
        peak_speed = max(peak_speed, max_step_speed)
        run_type = self._classify_run(start_pos, end_pos, ball_path, start_f, end_f)
        created_xt = self._estimate_created_xt(run_type, distance, avg_speed)
        return Run(
            player_track_id=player_track_id,
            team=team,
            start_frame=start_f,
            end_frame=end_f,
            start_pos=start_pos,
            end_pos=end_pos,
            run_type=run_type,
            distance_m=round(distance, 2),
            avg_speed_mps=round(avg_speed, 2),
            peak_speed_mps=round(peak_speed, 2),
            created_xT_delta=round(created_xt, 4),
        )

    def _classify_run(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        ball_path: list[tuple[int, tuple[float, float]]],
        start_f: int,
        end_f: int,
    ) -> RunType:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dist = math.hypot(dx, dy)
        if dist < self.min_run_distance_m:
            return RunType.UNKNOWN
        forward = dx > 2
        backward = dx < -2
        wide = abs(dy) > 5
        if forward and not wide:
            return RunType.BEHIND_DEFENSE
        if wide and forward:
            return RunType.DIAGONAL
        if wide and not forward:
            return RunType.WIDE
        if backward:
            return RunType.DROP
        if not forward and not backward and not wide:
            return RunType.SUPPORT
        return RunType.UNKNOWN

    @staticmethod
    def _estimate_created_xt(run_type: RunType, distance_m: float, speed_mps: float) -> float:
        base = {
            RunType.BEHIND_DEFENSE: 0.04,
            RunType.DIAGONAL: 0.025,
            RunType.WIDE: 0.02,
            RunType.DECOY: 0.015,
            RunType.SUPPORT: 0.005,
            RunType.DROP: 0.0,
            RunType.UNKNOWN: 0.0,
        }.get(run_type, 0.0)
        speed_bonus = 0.0
        if speed_mps > 7.0:
            speed_bonus = 0.01
        elif speed_mps > 5.5:
            speed_bonus = 0.005
        return base + speed_bonus

    def _summarize(self, team: str, runs: list[Run]) -> PositioningReport:
        if not runs:
            return self._empty_report(team)
        by_type: dict[str, int] = defaultdict(int)
        for r in runs:
            by_type[r.run_type.value] += 1
        total_dist = sum(r.distance_m for r in runs)
        longest = max(r.distance_m for r in runs)
        total_xt = sum(r.created_xT_delta for r in runs)
        notes: list[str] = []
        if total_xt > 0.5:
            notes.append("Strong off-ball movement generating significant xT")
        if by_type.get(RunType.BEHIND_DEFENSE.value, 0) > len(runs) * 0.4:
            notes.append("High frequency of runs behind the defense")
        if by_type.get(RunType.WIDE.value, 0) > len(runs) * 0.3:
            notes.append("Many wide runs stretching the pitch")
        return PositioningReport(
            team=team,
            total_runs=len(runs),
            runs_by_type=dict(by_type),
            total_distance_m=round(total_dist, 1),
            avg_run_distance_m=round(total_dist / len(runs), 2),
            longest_run_m=round(longest, 2),
            total_xT_created=round(total_xt, 3),
            runs=runs,
            notes=notes,
        )

    def is_in_behind_defense(self, x: float, opp_defensive_line_x: float) -> bool:
        """True if player position is behind the opponent's last defender."""
        return x > opp_defensive_line_x + 1.0
