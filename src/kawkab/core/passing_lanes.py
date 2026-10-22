"""Passing Lane Analysis — counting pass options, lane density, and blocking.

Identifies how many viable passing lanes exist for each pass event,
analyses lane density per team, detects blocked/intercepted passes,
and tracks how passing lanes change during possession sequences.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from kawkab.core.game_constants import GAME

PITCH_LENGTH = GAME.PITCH_LENGTH_M
PITCH_WIDTH = GAME.PITCH_WIDTH_M
MAX_LANE_DISTANCE = 40.0
MAX_LANE_ANGLE_DEG = 90.0
CORRIDOR_WIDTH = 2.0
PROGRESSIVE_LANE_THRESHOLD = 0.3

X_BANDS = [
    (0.0, 0.2, "Defensive"),
    (0.2, 0.4, "Defensive Mid"),
    (0.4, 0.6, "Middle"),
    (0.6, 0.8, "Attacking Mid"),
    (0.8, 1.0, "Attacking"),
]
Y_BANDS = [
    (0.0, 1 / 3, "Left"),
    (1 / 3, 2 / 3, "Center"),
    (2 / 3, 1.0, "Right"),
]


def _classify_zone(x: float, y: float) -> str:
    x_frac = max(0.0, min(0.999, x / PITCH_LENGTH))
    y_frac = max(0.0, min(0.999, y / PITCH_WIDTH))
    x_name = "Defensive"
    for lo, hi, name in X_BANDS:
        if lo <= x_frac < hi:
            x_name = name
            break
    y_name = "Left"
    for lo, hi, name in Y_BANDS:
        if lo <= y_frac < hi:
            y_name = name
            break
    return f"{x_name} {y_name}"


def _estimate_player_positions(
    events: list[dict[str, Any]],
    team: str,
    timestamp: float,
) -> list[dict[str, float]]:
    positions: list[dict[str, float]] = []
    for ev in events:
        ev_ts = float(ev.get("timestamp", 0))
        if abs(ev_ts - timestamp) > 5.0:
            continue
        if ev.get("team") != team:
            continue
        sx = ev.get("start_x")
        sy = ev.get("start_y")
        if sx is not None and sy is not None:
            positions.append({"x": float(sx), "y": float(sy)})
        ex = ev.get("end_x")
        ey = ev.get("end_y")
        if ex is not None and ey is not None:
            positions.append({"x": float(ex), "y": float(ey)})
    return positions


class PassingLaneAnalysis:
    def count_pass_options(
        self,
        event: dict[str, Any],
        all_events: list[dict[str, Any]],
        team_player_positions: list[dict[str, float]] | None = None,
    ) -> int:
        if event.get("type") != "pass":
            return 0
        sx = float(event.get("start_x", 0))
        sy = float(event.get("start_y", 0))
        team = event.get("team", "")
        if not team:
            return 0
        ts = float(event.get("timestamp", 0))
        if team_player_positions is None:
            team_player_positions = _estimate_player_positions(all_events, team, ts)
        count = 0
        for pos in team_player_positions:
            px = pos["x"]
            py = pos["y"]
            dx = px - sx
            dy = py - sy
            dist = math.hypot(dx, dy)
            if dist < 1.0 or dist > MAX_LANE_DISTANCE:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            if angle < -MAX_LANE_ANGLE_DEG or angle > MAX_LANE_ANGLE_DEG:
                continue
            blocked = False
            for opp_ev in all_events:
                if opp_ev.get("team") == team:
                    continue
                if opp_ev.get("type") not in ("pass", "carry", "defensive"):
                    continue
                ox = float(opp_ev.get("start_x", 0))
                oy = float(opp_ev.get("start_y", 0))
                odx = ox - sx
                ody = oy - sy
                opp_dist = math.hypot(odx, ody)
                if opp_dist < 1.0 or opp_dist > dist:
                    continue
                t_param = ((ox - sx) * dx + (oy - sy) * dy) / (dist * dist + 1e-9)
                if 0.0 < t_param < 1.0:
                    proj_x = sx + t_param * dx
                    proj_y = sy + t_param * dy
                    perp = math.hypot(ox - proj_x, oy - proj_y)
                    if perp <= CORRIDOR_WIDTH:
                        blocked = True
                        break
            if not blocked:
                count += 1
        return count

    def analyze_lane_density(
        self,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not events:
            return {}
        pass_events = [e for e in events if e.get("type") == "pass"]
        if not pass_events:
            return {}
        teams: dict[str, dict[str, Any]] = {}
        for ev in pass_events:
            team = ev.get("team", "")
            if not team:
                continue
            if team not in teams:
                teams[team] = {"options": [], "lanes_by_zone": defaultdict(list)}
            opts = self.count_pass_options(ev, events)
            teams[team]["options"].append(opts)
            zone = _classify_zone(
                float(ev.get("start_x", 52.5)),
                float(ev.get("start_y", 34.0)),
            )
            teams[team]["lanes_by_zone"][zone].append(opts)
        result: dict[str, Any] = {}
        for team, data in teams.items():
            opts = data["options"]
            avg_opts = sum(opts) / len(opts) if opts else 0.0
            lanes_by_zone = {}
            for zone, vals in data["lanes_by_zone"].items():
                lanes_by_zone[zone] = {
                    "avg": round(sum(vals) / len(vals), 2) if vals else 0.0,
                    "count": len(vals),
                }
            result[team] = {
                "avg_options": round(avg_opts, 2),
                "max_options": max(opts) if opts else 0,
                "min_options": min(opts) if opts else 0,
                "lanes_by_zone": lanes_by_zone,
            }
        return result

    def detect_lane_blocking(
        self,
        event: dict[str, Any],
        all_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if event.get("type") != "pass":
            return {"is_blocked": False, "blocker_distance": 0.0, "lane_zone": ""}
        sx = float(event.get("start_x", 0))
        sy = float(event.get("start_y", 0))
        ex = float(event.get("end_x", 0))
        ey = float(event.get("end_y", 0))
        team = event.get("team", "")
        dx = ex - sx
        dy = ey - sy
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return {"is_blocked": False, "blocker_distance": 0.0, "lane_zone": ""}
        min_blocker_dist = float("inf")
        for opp_ev in all_events:
            if opp_ev.get("team") == team:
                continue
            ox = float(opp_ev.get("start_x", 0))
            oy = float(opp_ev.get("start_y", 0))
            t_param = ((ox - sx) * dx + (oy - sy) * dy) / (length * length)
            if 0.0 < t_param < 1.0:
                proj_x = sx + t_param * dx
                proj_y = sy + t_param * dy
                perp = math.hypot(ox - proj_x, oy - proj_y)
                if perp <= CORRIDOR_WIDTH:
                    blocker_dist = math.hypot(ox - sx, oy - sy)
                    if blocker_dist < min_blocker_dist:
                        min_blocker_dist = blocker_dist
        is_blocked = min_blocker_dist < float("inf")
        lane_zone = _classify_zone((sx + ex) / 2.0, (sy + ey) / 2.0) if is_blocked else ""
        return {
            "is_blocked": is_blocked,
            "blocker_distance": round(min_blocker_dist, 2) if is_blocked else 0.0,
            "lane_zone": lane_zone,
        }

    def compute_progressive_lane_changes(
        self,
        events: list[dict[str, Any]],
    ) -> dict[str, int]:
        if not events:
            return {}
        pass_events = [e for e in events if e.get("type") == "pass"]
        if len(pass_events) < 2:
            return {}
        teams: dict[str, int] = defaultdict(int)
        for i in range(len(pass_events) - 1):
            ev = pass_events[i]
            next_ev = pass_events[i + 1]
            if ev.get("team") != next_ev.get("team"):
                continue
            team = ev.get("team", "")
            if not team:
                continue
            prev_opts = self.count_pass_options(ev, events)
            next_opts = self.count_pass_options(next_ev, events)
            increase = next_opts - prev_opts
            if increase > PROGRESSIVE_LANE_THRESHOLD:
                teams[team] += 1
        return dict(teams)
