"""Switch of Play Detection + Box Entries.

Detects switches of play (flank-to-flank passes covering >30 m total
with >20 m lateral movement) and box entries (passes or carries
entering the penalty area). All numpy-only.
"""

from __future__ import annotations

import math
from typing import Any

from kawkab.core.game_constants import GAME

PITCH_LENGTH = GAME.PITCH_LENGTH_M
PITCH_WIDTH = GAME.PITCH_WIDTH_M
GOAL_WIDTH = 7.32
GOAL_CENTER_Y = PITCH_WIDTH / 2.0
PENALTY_AREA_DEPTH = 16.5
PENALTY_AREA_X = PITCH_LENGTH - PENALTY_AREA_DEPTH
PENALTY_AREA_HALF_WIDTH = 16.5 + GOAL_WIDTH / 2.0
PENALTY_AREA_Y_LO = GOAL_CENTER_Y - PENALTY_AREA_HALF_WIDTH
PENALTY_AREA_Y_HI = GOAL_CENTER_Y + PENALTY_AREA_HALF_WIDTH

SWITCH_MIN_LATERAL_M = 20.0
SWITCH_MIN_TOTAL_M = 30.0
BOX_X_THRESHOLD = 102.0
BOX_Y_LO = 30.5
BOX_Y_HI = 37.5


def _classify_zone(x: float, y: float) -> str:
    x_frac = max(0.0, min(0.999, x / PITCH_LENGTH))
    y_frac = max(0.0, min(0.999, y / PITCH_WIDTH))
    if x_frac < 0.33:
        x_name = "Defensive"
    elif x_frac < 0.66:
        x_name = "Middle"
    else:
        x_name = "Attacking"
    if y_frac < 0.33:
        y_name = "Left"
    elif y_frac < 0.66:
        y_name = "Center"
    else:
        y_name = "Right"
    return f"{x_name} {y_name}"


class SwitchOfPlayDetector:
    def detect_switch_of_play(
        self,
        event: dict[str, Any],
        pitch_length: float = PITCH_LENGTH,
    ) -> dict[str, Any]:
        if event.get("type") != "pass":
            return {
                "is_switch": False,
                "lateral_distance_m": 0.0,
                "total_distance_m": 0.0,
                "recipient_zone": "",
            }
        sx = float(event.get("start_x", 0))
        sy = float(event.get("start_y", 0))
        ex = float(event.get("end_x", 0))
        ey = float(event.get("end_y", 0))
        lateral_dist = abs(ey - sy)
        total_dist = math.hypot(ex - sx, ey - sy)
        is_switch = lateral_dist >= SWITCH_MIN_LATERAL_M and total_dist >= SWITCH_MIN_TOTAL_M
        zone = ""
        if is_switch:
            zone = _classify_zone(ex, ey)
        return {
            "is_switch": is_switch,
            "lateral_distance_m": round(lateral_dist, 2),
            "total_distance_m": round(total_dist, 2),
            "recipient_zone": zone,
        }

    def analyze_switches(
        self,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not events:
            return {
                "switch_count": {"home": 0, "away": 0},
                "completion_rate": {"home": 0.0, "away": 0.0},
                "switches_leading_to_chances": {"home": 0, "away": 0},
                "avg_lateral_distance_m": {"home": 0.0, "away": 0.0},
                "preferred_direction": {"home": "", "away": ""},
            }
        pass_events = [e for e in events if e.get("type") == "pass"]
        teams: dict[str, dict[str, Any]] = {}
        for ev in pass_events:
            team = ev.get("team", "")
            if not team:
                continue
            if team not in teams:
                teams[team] = {
                    "switches": [],
                    "completed": 0,
                    "total": 0,
                    "lateral_dists": [],
                    "directions": [],
                }
            result = self.detect_switch_of_play(ev)
            if not result["is_switch"]:
                continue
            teams[team]["total"] += 1
            if ev.get("completed", False):
                teams[team]["completed"] += 1
            teams[team]["switches"].append(ev)
            teams[team]["lateral_dists"].append(result["lateral_distance_m"])
            direction = (
                "right" if float(ev.get("end_y", 0)) > float(ev.get("start_y", 0)) else "left"
            )
            teams[team]["directions"].append(direction)
        output: dict[str, Any] = {}
        for team, data in teams.items():
            _ = [e for e in events if e.get("type") == "shot"]
            leading_to_chances = 0
            for sw in data["switches"]:
                sw_idx = pass_events.index(sw) if sw in pass_events else -1
                if sw_idx >= 0:
                    for candidate in pass_events[sw_idx + 1 : sw_idx + 4]:
                        if candidate.get("type") == "shot" and candidate.get("team") == team:
                            leading_to_chances += 1
                            break
            total_sw = data["total"]
            completed = data["completed"]
            lateral_dists = data["lateral_dists"]
            directions = data["directions"]
            preferred = ""
            if directions:
                right_cnt = sum(1 for d in directions if d == "right")
                left_cnt = sum(1 for d in directions if d == "left")
                preferred = "right" if right_cnt >= left_cnt else "left"
            output[team] = {
                "switch_count": total_sw,
                "completion_rate": round(completed / total_sw, 3) if total_sw else 0.0,
                "switches_leading_to_chances": leading_to_chances,
                "avg_lateral_distance_m": round(sum(lateral_dists) / len(lateral_dists), 2)
                if lateral_dists
                else 0.0,
                "preferred_direction": preferred,
            }
        for side in ("home", "away"):
            if side not in output:
                output[side] = {
                    "switch_count": 0,
                    "completion_rate": 0.0,
                    "switches_leading_to_chances": 0,
                    "avg_lateral_distance_m": 0.0,
                    "preferred_direction": "",
                }
        return output

    def detect_box_entries(
        self,
        event: dict[str, Any],
    ) -> dict[str, Any]:
        ev_type = event.get("type", "")
        if ev_type not in ("pass", "carry", "dribble"):
            return {"is_entry": False, "entry_type": "", "zone_x": 0.0, "zone_y": 0.0}
        ex = float(event.get("end_x", 0))
        ey = float(event.get("end_y", 0))
        is_entry = ex >= BOX_X_THRESHOLD and BOX_Y_LO <= ey <= BOX_Y_HI
        return {
            "is_entry": is_entry,
            "entry_type": ev_type if is_entry else "",
            "zone_x": round(ex, 2) if is_entry else 0.0,
            "zone_y": round(ey, 2) if is_entry else 0.0,
        }

    def analyze_box_entries(
        self,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not events:
            return {
                "total_entries": {"home": 0, "away": 0},
                "entries_via_pass": {"home": 0, "away": 0},
                "entries_via_carry": {"home": 0, "away": 0},
                "entries_leading_to_shots": {"home": 0, "away": 0},
                "entries_leading_to_goals": {"home": 0, "away": 0},
                "entry_zones": {"home": [], "away": []},
            }
        teams: dict[str, dict[str, Any]] = {}
        for ev in events:
            team = ev.get("team", "")
            if not team:
                continue
            if team not in teams:
                teams[team] = {
                    "entries": [],
                    "via_pass": 0,
                    "via_carry": 0,
                    "entries_leading_to_shots": 0,
                    "entries_leading_to_goals": 0,
                    "entry_zones": [],
                }
            result = self.detect_box_entries(ev)
            if not result["is_entry"]:
                continue
            teams[team]["entries"].append(ev)
            entry_type = result["entry_type"]
            if entry_type == "pass":
                teams[team]["via_pass"] += 1
            elif entry_type in ("carry", "dribble"):
                teams[team]["via_carry"] += 1
            teams[team]["entry_zones"].append((result["zone_x"], result["zone_y"]))
        _ = [e for e in events if e.get("type") == "shot"]
        for team, data in teams.items():
            for entry in data["entries"]:
                entry_idx = events.index(entry)
                for candidate in events[entry_idx + 1 : entry_idx + 4]:
                    if candidate.get("type") == "shot" and candidate.get("team") == team:
                        data["entries_leading_to_shots"] += 1
                        if candidate.get("is_goal"):
                            data["entries_leading_to_goals"] += 1
                        break
        output: dict[str, Any] = {}
        for team, data in teams.items():
            output[team] = {
                "total_entries": len(data["entries"]),
                "entries_via_pass": data["via_pass"],
                "entries_via_carry": data["via_carry"],
                "entries_leading_to_shots": data["entries_leading_to_shots"],
                "entries_leading_to_goals": data["entries_leading_to_goals"],
                "entry_zones": data["entry_zones"],
            }
        for side in ("home", "away"):
            if side not in output:
                output[side] = {
                    "total_entries": 0,
                    "entries_via_pass": 0,
                    "entries_via_carry": 0,
                    "entries_leading_to_shots": 0,
                    "entries_leading_to_goals": 0,
                    "entry_zones": [],
                }
        return output
