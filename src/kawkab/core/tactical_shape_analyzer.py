"""Tactical Shape Analysis — team shape classification from event data.

Detects attacking/defensive shapes (3-2-5, 4-4-2, diamond midfield),
computes support angles between players, and classifies shape per phase.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from kawkab.core.game_constants import GAME

PITCH_LENGTH = GAME.PITCH_LENGTH_M
PITCH_WIDTH = GAME.PITCH_WIDTH_M


def _classify_line_count(counts: list[int]) -> str:
    """Classify an outfield line distribution into a known attacking shape."""
    # Normalize to sum of 10 outfield players
    total = sum(counts)
    if total == 0:
        return "unknown"
    if len(counts) == 2:
        return "4-4-2" if counts[0] == 4 else "5-3-2" if counts[0] == 5 else "3-5-2"
    if len(counts) == 3:
        mapping = {
            (4, 3, 3): "4-3-3",
            (4, 4, 2): "4-4-2",
            (3, 4, 3): "3-4-3",
            (3, 5, 2): "3-5-2",
            (5, 3, 2): "5-3-2",
            (4, 2, 4): "4-2-4",
            (3, 2, 5): "3-2-5",
        }
        key = tuple(counts)
        if key in mapping:
            return mapping[key]
    if len(counts) == 4:
        if tuple(counts) == (4, 2, 3, 1):
            return "4-2-3-1"
        if tuple(counts) == (4, 1, 4, 1):
            return "4-1-4-1"
        if tuple(counts) == (3, 2, 4, 1):
            return "3-2-4-1"
        if tuple(counts) == (3, 2, 5, 0):
            return "3-2-5"
        if tuple(counts) == (2, 3, 5, 0):
            return "2-3-5"
        if tuple(counts) == (4, 1, 3, 2):
            return "4-1-3-2"
    return "unknown"


def _detect_diamond_midfield(
    positions: list[tuple[float, float]], x_threshold: float = 15.0
) -> bool:
    """Detect if midfield 4 form a diamond shape (1 deep, 2 wide, 1 advanced)."""
    if len(positions) < 4:
        return False
    # Sort by x coordinate
    sorted_by_x = sorted(positions, key=lambda p: p[0])
    # Take middle 4 as midfield candidates
    midfield = sorted_by_x[1:-1] if len(sorted_by_x) > 4 else sorted_by_x
    if len(midfield) < 4:
        return False
    # Find the deepest, most advanced, leftmost, rightmost
    xs = [p[0] for p in midfield]
    ys = [p[1] for p in midfield]
    deep = min(xs)
    adv = max(xs)
    left = min(ys)
    right = max(ys)
    # Diamond: central 2 are between deep/adv and left/right
    mid_x = (deep + adv) / 2
    mid_y = (left + right) / 2
    distances = [math.hypot(p[0] - mid_x, p[1] - mid_y) for p in midfield]
    avg_dist = sum(distances) / len(distances)
    width = right - left
    height = adv - deep
    # Diamond should have players at all 4 corners
    return width > x_threshold * 1.5 and height > x_threshold and avg_dist > x_threshold * 0.6


def _classify_attacking_shape(positions: list[tuple[float, float]]) -> str:
    """Classify attacking shape (3-2-5, 4-3-3, etc.) from player positions."""
    if len(positions) < 8:
        return "insufficient_players"
    # Sort by x (pitch progression), skip deepest player (presumed GK)
    sorted_pos = sorted(positions, key=lambda p: p[0])
    if len(sorted_pos) >= 11:
        sorted_pos = sorted_pos[1:]  # Remove GK
    # Cluster into lines using k-means-like x-gap detection
    gaps = []
    for i in range(1, len(sorted_pos)):
        gaps.append(sorted_pos[i][0] - sorted_pos[i - 1][0])
    if not gaps:
        return "unknown"
    mean_gap = sum(gaps) / len(gaps)
    # Find significant gaps (>= 1.5x mean)
    lines = []
    current_line = [sorted_pos[0]]
    for i in range(1, len(sorted_pos)):
        if gaps[i - 1] >= mean_gap * 1.5:
            lines.append(current_line)
            current_line = [sorted_pos[i]]
        else:
            current_line.append(sorted_pos[i])
    if current_line:
        lines.append(current_line)
    line_counts = [len(ln) for ln in lines]
    # Classify
    shape = _classify_line_count(line_counts)
    return shape


def _compute_support_angles(
    carrier_pos: tuple[float, float],
    teammate_positions: list[tuple[float, float]],
    angle_threshold: float = 60.0,
) -> list[dict[str, Any]]:
    """Compute support angles — which teammates offer passing options."""
    if not carrier_pos or not teammate_positions:
        return []
    cx, cy = carrier_pos
    supports = []
    for tx, ty in teammate_positions:
        if math.isclose(cx, tx) and math.isclose(cy, ty):
            continue
        dx = tx - cx
        dy = ty - cy
        dist = math.hypot(dx, dy)
        if dist < 1.0:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        supports.append(
            {
                "dx": round(dx, 1),
                "dy": round(dy, 1),
                "distance_m": round(dist, 1),
                "angle_deg": round(angle, 1),
                "is_forward": dx > 0,
            }
        )
    return supports


def _find_triangles_in_shape(
    positions: list[tuple[float, float]], max_dist: float = 20.0
) -> list[list[int]]:
    """Find all triangles formed by players within max_dist of each other."""
    n = len(positions)
    triangles = []
    for i in range(n):
        for j in range(i + 1, n):
            d_ij = math.hypot(positions[i][0] - positions[j][0], positions[i][1] - positions[j][1])
            if d_ij > max_dist:
                continue
            for k in range(j + 1, n):
                d_ik = math.hypot(
                    positions[i][0] - positions[k][0], positions[i][1] - positions[k][1]
                )
                d_jk = math.hypot(
                    positions[j][0] - positions[k][0], positions[j][1] - positions[k][1]
                )
                if d_ik <= max_dist and d_jk <= max_dist:
                    triangles.append([i, j, k])
    return triangles


@dataclass
class ShapeSnapshot:
    timestamp: float = 0.0
    attacking_shape: str = "unknown"
    defensive_shape: str = "unknown"
    has_diamond_midfield: bool = False
    attacking_line_count: int = 0
    defensive_line_count: int = 0
    triangle_count: int = 0
    is_attacking_phase: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": round(self.timestamp, 1),
            "att_shape": self.attacking_shape,
            "def_shape": self.defensive_shape,
            "diamond": self.has_diamond_midfield,
            "att_lines": self.attacking_line_count,
            "def_lines": self.defensive_line_count,
            "triangles": self.triangle_count,
            "attacking": self.is_attacking_phase,
        }


@dataclass
class ShapeReport:
    team: str = "home"
    primary_attacking_shape: str = "unknown"
    primary_defensive_shape: str = "unknown"
    shape_changes: int = 0
    diamond_midfield_pct: float = 0.0
    avg_support_angle_coverage: float = 0.0
    avg_triangles_per_frame: float = 0.0
    snapshots: list[ShapeSnapshot] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "team": self.team,
            "primary_attacking_shape": self.primary_attacking_shape,
            "primary_defensive_shape": self.primary_defensive_shape,
            "shape_changes": self.shape_changes,
            "diamond_midfield_pct": round(self.diamond_midfield_pct, 1),
            "avg_support_angle_coverage": round(self.avg_support_angle_coverage, 1),
            "avg_triangles_per_frame": round(self.avg_triangles_per_frame, 1),
            "snapshots": [s.to_dict() for s in self.snapshots],
        }


class TacticalShapeAnalyzer:
    """Analyze team shapes from event/frame data."""

    def analyze_shapes(
        self,
        events: list[dict[str, Any]],
        team: str = "home",
        pitch_length: float = PITCH_LENGTH,
        pitch_width: float = PITCH_WIDTH,
    ) -> ShapeReport:
        team_events = [e for e in events if e.get("team") == team]
        if not team_events:
            return ShapeReport(team=team)

        snapshots: list[ShapeSnapshot] = []
        ts_min = min(e.get("timestamp", 0) for e in team_events)
        ts_max = max(e.get("timestamp", 0) for e in team_events)
        _ = max(ts_max - ts_min, 1.0)

        # Group events into windows
        window_s = 30.0
        t = ts_min
        shape_history: list[str] = []

        while t < ts_max:
            window_events = [e for e in team_events if t <= e.get("timestamp", 0) < t + window_s]
            ss = self._analyze_window(window_events, t, team)
            snapshots.append(ss)
            shape_history.append(ss.attacking_shape)
            t += window_s

        if not snapshots:
            return ShapeReport(team=team)

        # Aggregate
        shape_counts: dict[str, int] = defaultdict(int)
        def_shape_counts: dict[str, int] = defaultdict(int)
        diamond_count = sum(1 for s in snapshots if s.has_diamond_midfield)
        triangle_total = sum(s.triangle_count for s in snapshots)

        for s in snapshots:
            shape_counts[s.attacking_shape] += 1
            def_shape_counts[s.defensive_shape] += 1

        # Count shape changes
        changes = 0
        for i in range(1, len(shape_history)):
            if shape_history[i] != shape_history[i - 1]:
                changes += 1

        primary_att = max(shape_counts, key=shape_counts.get) if shape_counts else "unknown"  # type: ignore[arg-type,return-value]
        primary_def = (
            max(def_shape_counts, key=def_shape_counts.get) if def_shape_counts else "unknown"  # type: ignore[arg-type,return-value]
        )

        return ShapeReport(
            team=team,
            primary_attacking_shape=primary_att,
            primary_defensive_shape=primary_def,
            shape_changes=changes,
            diamond_midfield_pct=(diamond_count / len(snapshots) * 100) if snapshots else 0.0,
            avg_triangles_per_frame=round(triangle_total / len(snapshots), 1) if snapshots else 0.0,
            snapshots=snapshots,
        )

    def _analyze_window(
        self,
        window_events: list[dict[str, Any]],
        timestamp: float,
        team: str,
    ) -> ShapeSnapshot:
        # Extract positions from events
        positions: list[tuple[float, float]] = []
        seen_track_ids: set[int] = set()
        for ev in window_events:
            tid = ev.get("from_track_id") or ev.get("player_track_id", 0)
            if tid and tid not in seen_track_ids:
                # ev.get("x", 0.5) does NOT fall back to 0.5 when "x" is
                # present with value None -- and get_match_events() always
                # includes the "x" key (via json_extract on metadata),
                # None whenever an event has no spatial metadata. That
                # made `None * PITCH_LENGTH` raise on any such event.
                x = ev.get("start_x")
                if x is None:
                    x_frac = ev.get("x")
                    x = (x_frac if x_frac is not None else 0.5) * PITCH_LENGTH
                y = ev.get("start_y")
                if y is None:
                    y_frac = ev.get("y")
                    y = (y_frac if y_frac is not None else 0.5) * PITCH_WIDTH
                if isinstance(x, float) and isinstance(y, float):
                    positions.append((x, y))
                    seen_track_ids.add(tid)

        if len(positions) < 4:
            return ShapeSnapshot(timestamp=timestamp)

        attacking_shape = _classify_attacking_shape(positions)
        diamond = False
        if len(positions) >= 8:
            diamond = _detect_diamond_midfield(positions)

        # Detect triangles
        triangles = _find_triangles_in_shape(positions, max_dist=25.0)

        return ShapeSnapshot(
            timestamp=timestamp,
            attacking_shape=attacking_shape,
            defensive_shape=attacking_shape,
            has_diamond_midfield=diamond,
            attacking_line_count=len({p[0] for p in positions}),
            triangle_count=len(triangles),
            is_attacking_phase=True,
        )
