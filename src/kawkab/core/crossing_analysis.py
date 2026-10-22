"""Crossing analysis — cross type classification, danger rating, heatmap.

All numpy-only. Supports open-play and corner-kick crosses.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kawkab.core.game_constants import GAME

PITCH_LENGTH = GAME.PITCH_LENGTH_M
PITCH_WIDTH = GAME.PITCH_WIDTH_M

CROSS_TYPES = ("early", "driven", "floated", "pulled_back", "byline")

ZONE_LABELS = [
    "near_post_6yd",
    "far_post_6yd",
    "near_post_18yd",
    "far_post_18yd",
    "edge_of_box",
    "deep",
]


def _six_yard_x(end_x: float) -> float:
    return PITCH_LENGTH - 5.5


def _penalty_area_x(end_x: float) -> float:
    return PITCH_LENGTH - 16.5


def _zone_label(end_x: float, end_y: float) -> str:
    near_post = end_y < PITCH_WIDTH / 2
    in_six = end_x >= _six_yard_x(end_x)
    in_penalty = end_x >= _penalty_area_x(end_x)

    if in_six:
        return "near_post_6yd" if near_post else "far_post_6yd"
    if in_penalty:
        return "near_post_18yd" if near_post else "far_post_18yd"
    if end_x >= PITCH_LENGTH * 0.7:
        return "edge_of_box"
    return "deep"


@dataclass
class CrossResult:
    cross_type: str = ""
    danger_rating: float = 0.0
    headed_shot_created: bool = False
    goal_created: bool = False
    is_corner: bool = False
    start_x: float = 0.0
    start_y: float = 0.0
    end_x: float = 0.0
    end_y: float = 0.0
    zone: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "cross_type": self.cross_type,
            "danger_rating": round(self.danger_rating, 3),
            "headed_shot_created": self.headed_shot_created,
            "goal_created": self.goal_created,
            "is_corner": self.is_corner,
            "start_x": round(self.start_x, 1),
            "start_y": round(self.start_y, 1),
            "end_x": round(self.end_x, 1),
            "end_y": round(self.end_y, 1),
            "zone": self.zone,
        }


@dataclass
class CrossingReport:
    total_crosses: int = 0
    crosses_by_type: dict[str, int] = field(default_factory=dict)
    completion_rate: float = 0.0
    avg_danger_rating: float = 0.0
    headed_shots_created: int = 0
    goals_created: int = 0
    zone_heatmap: dict[str, int] = field(default_factory=dict)
    corner_crosses: int = 0
    crosses: list[CrossResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_crosses": self.total_crosses,
            "crosses_by_type": self.crosses_by_type,
            "completion_rate": round(self.completion_rate, 3),
            "avg_danger_rating": round(self.avg_danger_rating, 3),
            "headed_shots_created": self.headed_shots_created,
            "goals_created": self.goals_created,
            "zone_heatmap": self.zone_heatmap,
            "corner_crosses": self.corner_crosses,
            "crosses": [c.to_dict() for c in self.crosses],
        }


class CrossingAnalysis:
    """Analyse crossing events — type classification, danger, zone heatmap."""

    def classify_cross(self, event: dict[str, Any]) -> str:
        start_x = float(event.get("start_x", event.get("x", PITCH_LENGTH / 2)))
        end_x = float(event.get("end_x", start_x))
        height = event.get("height", event.get("metadata", {}).get("height", ""))
        if isinstance(height, str):
            height = height.lower()

        if start_x > 95:
            return "byline"
        if start_x < 40:
            return "early"
        if height == "low":
            return "driven"
        if height == "high":
            return "floated"
        if end_x < start_x:
            return "pulled_back"
        return "driven"

    def compute_cross_danger_rating(self, cross_event: dict[str, Any]) -> float:
        cross_type = self.classify_cross(cross_event)
        cx = float(cross_event.get("end_x", cross_event.get("start_x", PITCH_LENGTH / 2)))
        cy = float(cross_event.get("end_y", cross_event.get("start_y", PITCH_WIDTH / 2)))

        type_weights = {
            "pulled_back": 0.9,
            "early": 0.7,
            "byline": 0.8,
            "driven": 0.5,
            "floated": 0.4,
        }
        type_w = type_weights.get(cross_type, 0.5)

        prox = (cx / PITCH_LENGTH) * 0.4
        central = (1.0 - abs(cy - PITCH_WIDTH / 2) / (PITCH_WIDTH / 2)) * 0.2

        headed = 0.0
        if cross_event.get("metadata"):
            meta = cross_event["metadata"]
            if isinstance(meta, dict) and meta.get("headed_shot_created"):
                headed = 0.2

        rating = type_w + prox + central + headed
        return float(np.clip(rating, 0.0, 1.0))

    def analyze_crosses(self, events: list[dict[str, Any]]) -> CrossingReport:
        crosses = [e for e in events if e.get("type") == "cross"]
        if not crosses:
            return CrossingReport()

        results: list[CrossResult] = []
        type_counts: dict[str, int] = defaultdict(int)
        total_danger = 0.0
        headed_count = 0
        goal_count = 0
        corner_count = 0
        zone_hm: dict[str, int] = defaultdict(int)

        for c in crosses:
            ct = self.classify_cross(c)
            dr = self.compute_cross_danger_rating(c)
            sx = float(c.get("start_x", c.get("x", PITCH_LENGTH / 2)))
            sy = float(c.get("start_y", c.get("y", PITCH_WIDTH / 2)))
            ex = float(c.get("end_x", sx))
            ey = float(c.get("end_y", sy))

            meta = c.get("metadata", {})
            is_corner = bool(c.get("corner", meta.get("corner", False)))

            headed = False
            goal_created = False
            if isinstance(meta, dict):
                headed = meta.get("headed_shot_created", False)
                goal_created = meta.get("goal_created", False)

            zone = _zone_label(ex, ey)

            res = CrossResult(
                cross_type=ct,
                danger_rating=dr,
                headed_shot_created=headed,
                goal_created=goal_created,
                is_corner=is_corner,
                start_x=sx,
                start_y=sy,
                end_x=ex,
                end_y=ey,
                zone=zone,
            )
            results.append(res)
            type_counts[ct] += 1
            total_danger += dr
            zone_hm[zone] += 1
            if headed:
                headed_count += 1
            if goal_created:
                goal_count += 1
            if is_corner:
                corner_count += 1

        n = len(crosses)
        return CrossingReport(
            total_crosses=n,
            crosses_by_type=dict(type_counts),
            completion_rate=sum(1 for r in results if r.danger_rating > 0.4) / max(n, 1),
            avg_danger_rating=total_danger / max(n, 1),
            headed_shots_created=headed_count,
            goals_created=goal_count,
            zone_heatmap=dict(zone_hm),
            corner_crosses=corner_count,
            crosses=results,
        )

    def compute_cross_zone_heatmap(self, events: list[dict[str, Any]]) -> dict[str, int]:
        hm: dict[str, int] = defaultdict(int)
        for e in events:
            if e.get("type") != "cross":
                continue
            ex = float(e.get("end_x", e.get("start_x", PITCH_LENGTH / 2)))
            ey = float(e.get("end_y", e.get("start_y", PITCH_WIDTH / 2)))
            zone = _zone_label(ex, ey)
            hm[zone] += 1
        return dict(hm)
