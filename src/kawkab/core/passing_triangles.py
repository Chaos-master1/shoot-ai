"""Passing Triangles / Third-Man Combinations detection.

Detects triangular pass sequences, third-man combinations,
and builds passing triangle network analysis. All numpy-only.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from kawkab.core.game_constants import GAME

PITCH_LENGTH = GAME.PITCH_LENGTH_M
PITCH_WIDTH = GAME.PITCH_WIDTH_M
TIME_WINDOW_S = 30.0


def _calc_triangle_area(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> float:
    return abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0


def _classify_zone_from_x(x: float) -> str:
    if x < PITCH_LENGTH * 0.33:
        return "defensive"
    if x < PITCH_LENGTH * 0.66:
        return "mid"
    return "attacking"


def _zone_hex(x: float, y: float) -> str:
    zx = min(int(x / (PITCH_LENGTH / 3)), 2)
    zy = min(int(y / (PITCH_WIDTH / 3)), 2)
    names = {0: "left", 1: "center", 2: "right"}
    return f"{names[zy]}_{zx}"


class PassingTriangleAnalyzer:
    def detect_passing_triangles(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not events:
            return []
        passes = [e for e in events if e.get("type") == "pass" and e.get("completed", True)]
        passes.sort(key=lambda e: e.get("timestamp", 0))
        triangles: list[dict[str, Any]] = []
        n = len(passes)
        for i in range(n):
            for j in range(i + 1, n):
                pj = passes[j]
                if pj.get("timestamp", 0) - passes[i].get("timestamp", 0) > TIME_WINDOW_S:
                    break
                for k in range(j + 1, n):
                    pk = passes[k]
                    if pk.get("timestamp", 0) - passes[i].get("timestamp", 0) > TIME_WINDOW_S:
                        break
                    # track_ids may be None (unlinked events — imports or
                    # untracked passes); sorted([None, ...]) raised
                    # TypeError. A triangle needs three REAL players.
                    players = {
                        p
                        for p in (
                            passes[i].get("from_track_id"),
                            passes[i].get("to_track_id"),
                            passes[j].get("from_track_id"),
                            passes[j].get("to_track_id"),
                            pk.get("from_track_id"),
                            pk.get("to_track_id"),
                        )
                        if p is not None
                    }
                    if len(players) != 3:
                        continue
                    player_a, player_b, player_c = sorted(players)
                    ex = (
                        passes[i].get("end_x", 0) + passes[j].get("end_x", 0) + pk.get("end_x", 0)
                    ) / 3
                    _ = (
                        passes[i].get("end_y", 0) + passes[j].get("end_y", 0) + pk.get("end_y", 0)
                    ) / 3
                    sx1, sy1 = passes[i].get("start_x", 0), passes[i].get("start_y", 0)
                    sx2, sy2 = passes[j].get("start_x", 0), passes[j].get("start_y", 0)
                    sx3, sy3 = pk.get("start_x", 0), pk.get("start_y", 0)
                    area = _calc_triangle_area(sx1, sy1, sx2, sy2, sx3, sy3)
                    triangles.append(
                        {
                            "player_a": player_a,
                            "player_b": player_b,
                            "player_c": player_c,
                            "pass_count": 3,
                            "area_sqm": round(area, 2),
                            "zone": _classify_zone_from_x(ex),
                        }
                    )
        return triangles

    def detect_third_man_combinations(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not events:
            return []
        passes = [e for e in events if e.get("type") == "pass"]
        passes.sort(key=lambda e: e.get("timestamp", 0))
        results: list[dict[str, Any]] = []
        n = len(passes)
        for i in range(n - 2):
            p1 = passes[i]
            p2 = passes[i + 1]
            p3 = passes[i + 2]
            if p2.get("timestamp", 0) - p1.get("timestamp", 0) > 5:
                continue
            p1_from = p1.get("from_track_id")
            p1_to = p1.get("to_track_id")
            p2_from = p2.get("from_track_id")
            p2_to = p2.get("to_track_id")
            if not (p1_to == p2_from and p2_to == p1_from):
                continue
            p3_from = p3.get("from_track_id")
            p3_to = p3.get("to_track_id")
            if p3_from != p1_from:
                continue
            end_x = float(p3.get("end_x", 0))
            start_x = float(p1.get("start_x", 0))
            is_progressive = (end_x - start_x) > 20
            leads_to_shot = False
            for ev in events:
                if (
                    ev.get("type") == "shot"
                    and ev.get("team") == p1.get("team")
                    and 0 < ev.get("timestamp", 0) - p3.get("timestamp", 0) <= 10
                ):
                    leads_to_shot = True
                    break
            results.append(
                {
                    "three_players": [p1_from, p1_to, p3_to],
                    "is_progressive": is_progressive,
                    "leads_to_shot": leads_to_shot,
                }
            )
        return results

    def analyze_triangle_network(self, events: list[dict[str, Any]], team: str) -> dict[str, Any]:
        team_events = [e for e in events if e.get("team") == team]
        triangles = self.detect_passing_triangles(team_events)
        if not triangles:
            return {
                "team": team,
                "triangle_count": 0,
                "triangles": [],
                "most_common_zone": "none",
                "avg_area_sqm": 0,
            }
        counts: dict[str, int] = defaultdict(int)
        zone_counts: dict[str, int] = defaultdict(int)
        total_area = 0.0
        for t in triangles:
            key = f"{t['player_a']}-{t['player_b']}-{t['player_c']}"
            counts[key] += 1
            zone_counts[t["zone"]] += 1
            total_area += t["area_sqm"]
        most_common_zone = max(zone_counts, key=zone_counts.get)  # type: ignore[arg-type,return-value]
        return {
            "team": team,
            "triangle_count": len(triangles),
            "unique_triangles": len(counts),
            "triangles": [
                {"players": k, "count": v} for k, v in sorted(counts.items(), key=lambda x: -x[1])
            ],
            "avg_area_sqm": round(total_area / len(triangles), 2) if triangles else 0,
            "most_common_zone": most_common_zone,
        }

    def compute_triangle_efficiency(
        self, events: list[dict[str, Any]], team: str
    ) -> dict[str, Any]:
        team_events = [e for e in events if e.get("team") == team]
        triangles = self.detect_passing_triangles(team_events)
        total_triangles = len(triangles)
        duration = max((e.get("timestamp", 0) for e in team_events), default=0)
        minutes = duration / 60.0
        triangles_per_90 = (total_triangles / minutes * 90) if minutes > 0 else 0
        passes = [e for e in team_events if e.get("type") == "pass"]
        total_passes_team = len(passes)
        completion_rate = (
            (total_triangles * 3 / total_passes_team * 100) if total_passes_team else 0
        )
        shots_after_triangle = 0
        goals_after_triangle = 0
        for t in triangles:
            last_ts = max(
                e.get("timestamp", 0)
                for e in team_events
                if e.get("type") == "pass"
                and e.get("from_track_id") in (t["player_a"], t["player_b"], t["player_c"])
            )
            for ev in team_events:
                if ev.get("type") == "shot" and 0 < ev.get("timestamp", 0) - last_ts <= 10:
                    shots_after_triangle += 1
                    if ev.get("is_goal"):
                        goals_after_triangle += 1
                    break
        return {
            "team": team,
            "triangles_per_90": round(triangles_per_90, 2),
            "triangle_completion_rate_pct": round(completion_rate, 1),
            "triangles_leading_to_shots": shots_after_triangle,
            "triangles_leading_to_goals": goals_after_triangle,
        }

    def get_triangle_heatmap(self, events: list[dict[str, Any]], team: str) -> dict[str, Any]:
        team_events = [e for e in events if e.get("team") == team]
        triangles = self.detect_passing_triangles(team_events)
        zones = [
            "deep_left",
            "deep_center",
            "deep_right",
            "final_third_left",
            "final_third_center",
            "final_third_right",
        ]
        zone_counts: dict[str, int] = dict.fromkeys(zones, 0)
        for _t in triangles:
            z = _zone_hex(
                sum(e.get("end_x", 0) for e in team_events if e.get("type") == "pass")
                / max(len(team_events), 1),
                sum(e.get("end_y", 0) for e in team_events if e.get("type") == "pass")
                / max(len(team_events), 1),
            )
            zone_counts[z] = zone_counts.get(z, 0) + 1
        return {"team": team, "zone_counts": zone_counts, "total_triangles": len(triangles)}
