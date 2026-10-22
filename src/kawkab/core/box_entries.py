"""Box Touches / Penalty Area Entries analysis.

Detects touches in the penalty area, entries into the box,
and computes effectiveness metrics. All numpy-only.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from kawkab.core.game_constants import GAME

PITCH_LENGTH = GAME.PITCH_LENGTH_M
PITCH_WIDTH = GAME.PITCH_WIDTH_M
PENALTY_AREA_START_X = 102.0
PENALTY_AREA_END_X = 105.0
PENALTY_AREA_START_Y = 30.5
PENALTY_AREA_END_Y = 37.5


def _in_penalty_area(x: float, y: float) -> bool:
    return (
        PENALTY_AREA_START_X <= x <= PENALTY_AREA_END_X
        and PENALTY_AREA_START_Y <= y <= PENALTY_AREA_END_Y
    )


def _entry_zone(x: float) -> str:
    mid = (PENALTY_AREA_START_Y + PENALTY_AREA_END_Y) / 2
    if x < mid - 2:
        return "left"
    if x > mid + 2:
        return "right"
    return "center"


class BoxEntryAnalyzer:
    def detect_box_touch(self, event: dict[str, Any]) -> dict[str, Any]:
        ev_type = event.get("type", "")
        if ev_type not in ("pass", "receive", "dribble", "shot"):
            return {
                "is_touch": False,
                "player_id": None,
                "team": "",
                "touch_type": "",
                "zone_x": 0.0,
                "zone_y": 0.0,
            }
        ex = float(event.get("end_x", 0))
        ey = float(event.get("end_y", 0))
        is_touch = _in_penalty_area(ex, ey)
        return {
            "is_touch": is_touch,
            "player_id": event.get("from_track_id") or event.get("track_id"),
            "team": event.get("team", ""),
            "touch_type": ev_type if is_touch else "",
            "zone_x": round(ex, 2) if is_touch else 0.0,
            "zone_y": round(ey, 2) if is_touch else 0.0,
        }

    def detect_penalty_area_entry(self, event: dict[str, Any]) -> dict[str, Any]:
        ev_type = event.get("type", "")
        if ev_type not in ("pass", "carry", "dribble"):
            return {"is_entry": False, "entry_type": "", "player_id": None, "entry_zone": ""}
        sx = float(event.get("start_x", 0))
        sy = float(event.get("start_y", 0))
        ex = float(event.get("end_x", 0))
        ey = float(event.get("end_y", 0))
        start_in = _in_penalty_area(sx, sy)
        end_in = _in_penalty_area(ex, ey)
        is_entry = (not start_in) and end_in
        return {
            "is_entry": is_entry,
            "entry_type": ev_type if is_entry else "",
            "player_id": event.get("from_track_id") if is_entry else None,
            "entry_zone": _entry_zone(ex) if is_entry else "",
        }

    def analyze_box_touches(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        teams: dict[str, dict[str, Any]] = {}
        for ev in events:
            team = ev.get("team", "")
            if not team:
                continue
            if team not in teams:
                teams[team] = {
                    "touches": [],
                    "touch_types": defaultdict(int),
                    "players": defaultdict(int),
                    "shots_from_touches": 0,
                    "goals_from_touches": 0,
                    "distances": [],
                }
            result = self.detect_box_touch(ev)
            if not result["is_touch"]:
                continue
            teams[team]["touches"].append(ev)
            teams[team]["touch_types"][result["touch_type"]] += 1
            if result["player_id"] is not None:
                teams[team]["players"][result["player_id"]] += 1
        output: dict[str, Any] = {}
        for team, data in teams.items():
            touches = data["touches"]
            for ev in touches:
                idx = events.index(ev)
                for candidate in events[idx + 1 : idx + 4]:
                    if candidate.get("type") == "shot" and candidate.get("team") == team:
                        data["shots_from_touches"] += 1
                        if candidate.get("is_goal"):
                            data["goals_from_touches"] += 1
                        break
            avg_dist = 0.0
            if touches:
                total_dist = 0.0
                for ev in touches:
                    total_dist += PENALTY_AREA_END_X - float(ev.get("end_x", PENALTY_AREA_END_X))
                avg_dist = total_dist / len(touches)
            output[team] = {
                "total_touches": len(touches),
                "touches_per_player": {
                    str(k): v for k, v in sorted(data["players"].items(), key=lambda x: -x[1])
                },
                "touches_by_type": dict(data["touch_types"]),
                "touches_leading_to_shots": data["shots_from_touches"],
                "touches_leading_to_goals": data["goals_from_touches"],
                "avg_distance_from_goal_m": round(avg_dist, 2),
            }
        for side in ("home", "away"):
            if side not in output:
                output[side] = {
                    "total_touches": 0,
                    "touches_per_player": {},
                    "touches_by_type": {},
                    "touches_leading_to_shots": 0,
                    "touches_leading_to_goals": 0,
                    "avg_distance_from_goal_m": 0.0,
                }
        return output

    def analyze_box_entries(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        teams: dict[str, dict[str, Any]] = {}
        for ev in events:
            team = ev.get("team", "")
            if not team:
                continue
            if team not in teams:
                teams[team] = {
                    "entries": [],
                    "entry_types": defaultdict(int),
                    "entry_zones": defaultdict(int),
                    "entries_to_shots": 0,
                    "entries_to_goals": 0,
                }
            result = self.detect_penalty_area_entry(ev)
            if not result["is_entry"]:
                continue
            teams[team]["entries"].append(ev)
            teams[team]["entry_types"][result["entry_type"]] += 1
            teams[team]["entry_zones"][result["entry_zone"]] += 1
        output: dict[str, Any] = {}
        for team, data in teams.items():
            for ev in data["entries"]:
                idx = events.index(ev)
                for candidate in events[idx + 1 : idx + 4]:
                    if candidate.get("type") == "shot" and candidate.get("team") == team:
                        data["entries_to_shots"] += 1
                        if candidate.get("is_goal"):
                            data["entries_to_goals"] += 1
                        break
            total = sum(data["entry_zones"].values()) or 1
            output[team] = {
                "total_entries": len(data["entries"]),
                "entries_via_pass": data["entry_types"].get("pass", 0),
                "entries_via_carry": data["entry_types"].get("carry", 0)
                + data["entry_types"].get("dribble", 0),
                "entries_leading_to_shots": data["entries_to_shots"],
                "entries_leading_to_goals": data["entries_to_goals"],
                "preferred_entry_zone": max(data["entry_zones"], key=data["entry_zones"].get)
                if data["entry_zones"]
                else "",
                "entry_zones_pct": {
                    k: round(v / total * 100, 1) for k, v in data["entry_zones"].items()
                },
            }
        for side in ("home", "away"):
            if side not in output:
                output[side] = {
                    "total_entries": 0,
                    "entries_via_pass": 0,
                    "entries_via_carry": 0,
                    "entries_leading_to_shots": 0,
                    "entries_leading_to_goals": 0,
                    "preferred_entry_zone": "",
                    "entry_zones_pct": {},
                }
        return output

    def compute_effectiveness(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        box_touch_analysis = self.analyze_box_touches(events)
        entry_analysis = self.analyze_box_entries(events)
        output: dict[str, Any] = {}
        for team in ("home", "away"):
            bt = box_touch_analysis.get(team, {})
            ea = entry_analysis.get(team, {})
            touch_to_shot = (
                (bt["touches_leading_to_shots"] / bt["total_touches"] * 100)
                if bt.get("total_touches", 0) > 0
                else 0.0
            )
            touch_to_goal = (
                (bt["touches_leading_to_goals"] / bt["total_touches"] * 100)
                if bt.get("total_touches", 0) > 0
                else 0.0
            )
            entry_to_goal = (
                (ea["entries_leading_to_goals"] / ea["total_entries"] * 100)
                if ea.get("total_entries", 0) > 0
                else 0.0
            )
            output[team] = {
                "box_touch_to_shot_pct": round(touch_to_shot, 1),
                "box_touch_to_goal_pct": round(touch_to_goal, 1),
                "entry_to_goal_pct": round(entry_to_goal, 1),
            }
        return output
