"""Corner Kick xG Model — delivery danger rating, xG attribution,
efficiency rates, and delivery zone classification.

All methods are numpy-only.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from kawkab.core.game_constants import GAME

PITCH_LENGTH = getattr(GAME, "PITCH_LENGTH_M", 105.0)
PITCH_WIDTH = getattr(GAME, "PITCH_WIDTH_M", 68.0)

DELIVERY_ZONE_RATINGS: dict[str, float] = {
    "near-post": 0.08,
    "far-post": 0.10,
    "short": 0.04,
    "edge-of-box": 0.03,
    "deep": 0.02,
}


def _classify_delivery_zone(end_x: float, end_y: float) -> str:
    """Classify corner delivery endpoint into one of 5 zones."""
    if end_y < 20:
        return "short"
    if end_y > 48:
        return "deep"
    if end_x > 102:
        if 30.5 <= end_y <= 37.5:
            return "near-post"
        return "far-post"
    if end_x <= 102 and 20 <= end_y <= 48:
        return "edge-of-box"
    return "far-post"


class CornerKickXgModel:
    """Corner kick xG model with delivery danger ratings and efficiency."""

    @staticmethod
    def compute_corner_danger_rating(corner_event: dict[str, Any]) -> float:
        """Compute 0-1 danger rating for a corner delivery.

        Factors: delivery zone (base rating), delivery type
        (inswinging +0.02, outswinging -0.01), headed (x1.1).
        """
        end_x = float(
            corner_event.get(
                "end_x",
                corner_event.get("x", corner_event.get("start_x", 104)),
            )
        )
        end_y = float(
            corner_event.get(
                "end_y",
                corner_event.get("y", corner_event.get("start_y", 34)),
            )
        )

        zone = _classify_delivery_zone(end_x, end_y)
        base = DELIVERY_ZONE_RATINGS.get(zone, 0.03)

        delivery_type = corner_event.get("delivery_type", "").lower()
        if "inswing" in delivery_type:
            base += 0.02
        elif "outswing" in delivery_type:
            base -= 0.01

        body_part = corner_event.get("body_part", "").lower()
        if body_part == "head":
            base *= 1.1

        return round(min(1.0, max(0.0, base)), 4)

    @staticmethod
    def compute_corner_xg(events: list[dict[str, Any]]) -> dict[str, float]:
        """Total xG from corner kicks = sum of shot xG values where
        shot is preceded by a corner within 3 events.

        Returns dict with total_xg, home_xg, away_xg.
        """
        sorted_ev = sorted(events, key=lambda e: e.get("timestamp", 0.0))
        n = len(sorted_ev)
        total = 0.0
        home = 0.0
        away = 0.0

        corner_indices = [i for i, e in enumerate(sorted_ev) if e.get("type") == "corner_kick"]

        for ci in corner_indices:
            window = sorted_ev[ci + 1 : min(n, ci + 4)]
            for we in window:
                if we.get("type") == "shot":
                    xg = float(we.get("xg", 0.0))
                    total += xg
                    team = we.get("team", "")
                    if team == "home":
                        home += xg
                    else:
                        away += xg
                    break

        return {
            "total_xg": round(total, 4),
            "home_xg": round(home, 4),
            "away_xg": round(away, 4),
        }

    @staticmethod
    def compute_corner_efficiency(events: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        """Total corners → shot conversion, goal conversion, xG per corner."""
        sorted_ev = sorted(events, key=lambda e: e.get("timestamp", 0.0))
        n = len(sorted_ev)
        result: dict[str, dict[str, float]] = {}

        for team in ("home", "away"):
            team_corners = [
                i
                for i, e in enumerate(sorted_ev)
                if e.get("type") == "corner_kick" and e.get("team") == team
            ]
            total_corners = len(team_corners)
            shots = 0
            goals = 0
            xg_total = 0.0

            for ci in team_corners:
                window = sorted_ev[ci + 1 : min(n, ci + 4)]
                for we in window:
                    if we.get("type") == "shot" and we.get("team") == team:
                        shots += 1
                        xg_total += float(we.get("xg", 0.0))
                        if we.get("is_goal"):
                            goals += 1
                        break

            result[team] = {
                "corners": float(total_corners),
                "shots": float(shots),
                "goals": float(goals),
                "total_xg": round(xg_total, 4),
                "shot_conversion": (shots / total_corners if total_corners > 0 else 0.0),
                "goal_conversion": (goals / total_corners if total_corners > 0 else 0.0),
                "xg_per_corner": (xg_total / total_corners if total_corners > 0 else 0.0),
            }

        return result

    @staticmethod
    def analyze_delivery_zones(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Classify each corner into 5 delivery zones.

        Returns dict with zone breakdown per team and overall:
        {team: {zone_name: {count, xg, shots, goals}}}
        """
        sorted_ev = sorted(events, key=lambda e: e.get("timestamp", 0.0))
        n = len(sorted_ev)
        zones: dict[str, dict[str, dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: {"count": 0.0, "xg": 0.0, "shots": 0.0, "goals": 0.0})
        )

        corner_indices = [i for i, e in enumerate(sorted_ev) if e.get("type") == "corner_kick"]

        for ci in corner_indices:
            ev = sorted_ev[ci]
            team = ev.get("team", "")

            end_x = float(
                ev.get(
                    "end_x",
                    ev.get("x", ev.get("start_x", 104)),
                )
            )
            end_y = float(
                ev.get(
                    "end_y",
                    ev.get("y", ev.get("start_y", 34)),
                )
            )
            zone = _classify_delivery_zone(end_x, end_y)
            zones[team][zone]["count"] += 1.0

            window = sorted_ev[ci + 1 : min(n, ci + 4)]
            for we in window:
                if we.get("type") == "shot" and we.get("team") == team:
                    zones[team][zone]["shots"] += 1.0
                    zones[team][zone]["xg"] += float(we.get("xg", 0.0))
                    if we.get("is_goal"):
                        zones[team][zone]["goals"] += 1.0
                    break

        # Build output with rounded values
        result: dict[str, dict[str, Any]] = {}
        for team, zone_data in zones.items():
            result[team] = {
                zn: {
                    "count": int(v["count"]),
                    "xg": round(v["xg"], 4),
                    "shots": int(v["shots"]),
                    "goals": int(v["goals"]),
                }
                for zn, v in sorted(zone_data.items())
            }

        return result
