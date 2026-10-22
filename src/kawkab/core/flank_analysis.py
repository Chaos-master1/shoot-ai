"""Flank Preference Analysis.

Analyzes build-up side preference, attacking flank preference,
flank effectiveness, and switch-of-play detection. All numpy-only.
"""

from __future__ import annotations

from typing import Any

from kawkab.core.game_constants import GAME

PITCH_LENGTH = GAME.PITCH_LENGTH_M
PITCH_WIDTH = GAME.PITCH_WIDTH_M
GOAL_CENTER_Y = PITCH_WIDTH / 2.0


def classify_zone(x: float, pitch_length: float = PITCH_LENGTH) -> str:
    third = pitch_length / 3
    if x < third:
        return "left"
    if x <= 2 * third:
        return "center"
    return "right"


class FlankAnalyzer:
    def analyze_build_up_side(self, events: list[dict[str, Any]], team: str) -> dict[str, Any]:
        team_passes = [e for e in events if e.get("team") == team and e.get("type") == "pass"]
        build_up_passes = [
            e
            for e in team_passes
            if float(e.get("start_x", PITCH_LENGTH / 2)) < PITCH_LENGTH * 0.33
        ]
        if not build_up_passes:
            return {
                "left_pct": 0,
                "center_pct": 0,
                "right_pct": 0,
                "dominant_side": "center",
                "total_actions": 0,
            }
        sides: list[str] = []
        for e in build_up_passes:
            ex = float(e.get("end_x", PITCH_LENGTH / 2))
            if ex > float(e.get("start_x", 0)):
                sides.append(classify_zone(ex))
        if not sides:
            return {
                "left_pct": 0,
                "center_pct": 0,
                "right_pct": 0,
                "dominant_side": "center",
                "total_actions": 0,
            }
        total = len(sides)
        left_c = sum(1 for s in sides if s == "left")
        center_c = sum(1 for s in sides if s == "center")
        right_c = sum(1 for s in sides if s == "right")
        dominant = max(
            {"left": left_c, "center": center_c, "right": right_c},
            key=lambda k: {"left": left_c, "center": center_c, "right": right_c}[k],
        )
        return {
            "left_pct": round(left_c / total * 100, 1),
            "center_pct": round(center_c / total * 100, 1),
            "right_pct": round(right_c / total * 100, 1),
            "dominant_side": dominant,
            "total_actions": total,
        }

    def analyze_attack_side(self, events: list[dict[str, Any]], team: str) -> dict[str, Any]:
        team_passes = [e for e in events if e.get("team") == team and e.get("type") == "pass"]
        attack_passes = [e for e in team_passes if float(e.get("start_x", 0)) > PITCH_LENGTH * 0.66]
        if not attack_passes:
            return {
                "left_pct": 0,
                "center_pct": 0,
                "right_pct": 0,
                "dominant_side": "center",
                "total_actions": 0,
            }
        sides = [
            classify_zone(float(e.get("end_y", GOAL_CENTER_Y)), PITCH_WIDTH) for e in attack_passes
        ]
        total = len(sides)
        left_c = sum(1 for s in sides if s == "left")
        center_c = sum(1 for s in sides if s == "center")
        right_c = sum(1 for s in sides if s == "right")
        dominant = max(
            {"left": left_c, "center": center_c, "right": right_c},
            key=lambda k: {"left": left_c, "center": center_c, "right": right_c}[k],
        )
        return {
            "left_pct": round(left_c / total * 100, 1),
            "center_pct": round(center_c / total * 100, 1),
            "right_pct": round(right_c / total * 100, 1),
            "dominant_side": dominant,
            "total_actions": total,
        }

    def compute_flank_effectiveness(
        self, events: list[dict[str, Any]], team: str
    ) -> dict[str, Any]:
        team_events = [e for e in events if e.get("team") == team]
        flanks: dict[str, dict[str, Any]] = {
            "left": {
                "passes": 0,
                "crosses": 0,
                "crosses_completed": 0,
                "shots": 0,
                "xg": 0.0,
                "goals": 0,
            },
            "center": {
                "passes": 0,
                "crosses": 0,
                "crosses_completed": 0,
                "shots": 0,
                "xg": 0.0,
                "goals": 0,
            },
            "right": {
                "passes": 0,
                "crosses": 0,
                "crosses_completed": 0,
                "shots": 0,
                "xg": 0.0,
                "goals": 0,
            },
        }
        for ev in team_events:
            sx = float(ev.get("start_x", PITCH_LENGTH / 2))
            flank = classify_zone(sx)
            if flank not in flanks:
                continue
            et = ev.get("type")
            if et == "pass":
                flanks[flank]["passes"] += 1
                if ev.get("completed"):
                    ex = float(ev.get("end_x", 0))
                    if ex >= PITCH_LENGTH * 0.8:
                        flanks[flank]["crosses"] += 1
                        flanks[flank]["crosses_completed"] += 1
            elif et == "cross":
                flanks[flank]["crosses"] += 1
                if ev.get("completed"):
                    flanks[flank]["crosses_completed"] += 1
            elif et == "shot":
                flanks[flank]["shots"] += 1
                flanks[flank]["xg"] += ev.get("xg", 0)
                if ev.get("is_goal"):
                    flanks[flank]["goals"] += 1
        output: dict[str, Any] = {}
        for fname, fdata in flanks.items():
            cross_completion = (
                (fdata["crosses_completed"] / fdata["crosses"] * 100) if fdata["crosses"] else 0.0
            )
            output[fname] = {
                "passes": fdata["passes"],
                "shots_created": fdata["shots"],
                "xg_created": round(fdata["xg"], 3),
                "goals_scored": fdata["goals"],
                "crosses_attempted": fdata["crosses"],
                "cross_completion_pct": round(cross_completion, 1),
            }
        output["team"] = team
        return output

    def detect_flank_switches(self, events: list[dict[str, Any]], team: str) -> dict[str, Any]:
        team_passes = [e for e in events if e.get("team") == team and e.get("type") == "pass"]
        switches: list[dict[str, Any]] = []
        for e in team_passes:
            _ = float(e.get("start_x", 0))
            sy = float(e.get("start_y", GOAL_CENTER_Y))
            _ = float(e.get("end_x", 0))
            ey = float(e.get("end_y", GOAL_CENTER_Y))
            start_flank = classify_zone(sy, PITCH_WIDTH)
            end_flank = classify_zone(ey, PITCH_WIDTH)
            if start_flank != end_flank and start_flank != "center" and end_flank != "center":
                switches.append(e)
        completed = sum(1 for s in switches if s.get("completed", True))
        total = len(switches)
        completion_rate = (completed / total * 100) if total else 0.0
        leading_to_chances = 0
        for sw in switches:
            idx = team_passes.index(sw)
            for candidate in team_passes[idx + 1 : idx + 4]:
                if candidate.get("type") == "shot" and candidate.get("team") == team:
                    leading_to_chances += 1
                    break
        return {
            "team": team,
            "switch_count": total,
            "completion_rate_pct": round(completion_rate, 1),
            "pct_leading_to_chances": round(leading_to_chances / total * 100, 1) if total else 0.0,
        }

    def generate_flank_report(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for team in {e.get("team", "") for e in events if e.get("team")}:
            output[team] = {
                "build_up": self.analyze_build_up_side(events, team),
                "attack_side": self.analyze_attack_side(events, team),
                "flank_effectiveness": self.compute_flank_effectiveness(events, team),
                "flank_switches": self.detect_flank_switches(events, team),
            }
        for side in ("home", "away"):
            if side not in output:
                output[side] = {
                    "build_up": {
                        "left_pct": 0,
                        "center_pct": 0,
                        "right_pct": 0,
                        "dominant_side": "center",
                        "total_actions": 0,
                    },
                    "attack_side": {
                        "left_pct": 0,
                        "center_pct": 0,
                        "right_pct": 0,
                        "dominant_side": "center",
                        "total_actions": 0,
                    },
                    "flank_effectiveness": {"team": side},
                    "flank_switches": {
                        "team": side,
                        "switch_count": 0,
                        "completion_rate_pct": 0.0,
                        "pct_leading_to_chances": 0.0,
                    },
                }
        return output
