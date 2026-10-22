"""Finishing Analysis — decomposes finishing skill into shot quality tiers,
streak detection, and placement skill."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

DEFAULT_TIER_THRESHOLDS: dict[str, tuple[float, float | None]] = {
    "big_chance": (0.35, None),
    "half_chance": (0.10, 0.35),
    "low_chance": (0.0, 0.10),
}


@dataclass
class FinishingReport:
    player_id: str
    total_goals: int
    total_xg: float
    finishing_delta: float
    shot_tiers: dict
    streak_data: dict
    placement_skill: float
    conversion_rate: float
    xg_per_shot: float


def _tier_for_xg(xg: float, tier_thresholds: dict[str, tuple[float, float | None]]) -> str:
    for tier_name, (lo, hi) in tier_thresholds.items():
        if hi is None:
            if xg >= lo:
                return tier_name
        else:
            if lo <= xg < hi:
                return tier_name
    return "low_chance"


def _detect_streaks(shots: list[dict[str, Any]], window: int) -> dict:
    hot = False
    cold = False
    hot_shots_detail: list[dict] = []
    cold_shots_detail: list[dict] = []

    for i in range(len(shots)):
        chunk = shots[max(0, i - window + 1) : i + 1]
        if len(chunk) < window:
            continue
        goals = sum(1 for s in chunk if s.get("goal"))
        goals_in_5 = sum(1 for s in chunk[-5:] if s.get("goal"))
        total_xg_chunk = sum(s.get("xG", 0.0) for s in chunk)
        actual_goals = sum(1 for s in chunk if s.get("goal"))

        # Hot: >=3 goals from <=5 shots where actual > xG by 0.5+
        if goals_in_5 >= 3:
            xg_chunk = sum(s.get("xG", 0.0) for s in chunk[-5:])
            if actual_goals - xg_chunk >= 0.5:
                hot = True
                hot_shots_detail = chunk[-5:]

        # Cold: >=5 shots no goals where combined xG > 1.0
        if goals == 0 and total_xg_chunk > 1.0 and len(chunk) >= 5:
            cold = True
            cold_shots_detail = chunk

    return {
        "hot_streak": hot,
        "cold_streak": cold,
        "hot_shots_detail": hot_shots_detail,
        "cold_shots_detail": cold_shots_detail,
        "window": window,
    }


def _compute_placement_skill(shots: list[dict[str, Any]]) -> float:
    placements = []
    for s in shots:
        px = s.get("placement_x")
        py = s.get("placement_y")
        if px is not None and py is not None:
            goal_w = 7.32
            goal_h = 2.44
            dist_center = math.sqrt((px - 0.0) ** 2 + (py - 0.0) ** 2)
            max_dist = math.sqrt((goal_w / 2) ** 2 + goal_h**2)
            if max_dist > 0:
                norm = dist_center / max_dist
                placements.append(norm)
    if not placements:
        return 0.0
    raw = sum(placements) / len(placements)
    return max(-1.0, min(1.0, raw * 2.0 - 0.5))


def analyze_finishing(
    player_id: str,
    shots: list[dict],
    tier_thresholds: dict | None = None,
    streak_window: int = 5,
) -> FinishingReport:
    if tier_thresholds is None:
        tier_thresholds = DEFAULT_TIER_THRESHOLDS

    total_xg = sum(s.get("xG", 0.0) for s in shots)
    total_goals = sum(1 for s in shots if s.get("goal"))
    finishing_delta = total_goals - total_xg
    n_shots = len(shots)
    conversion_rate = total_goals / n_shots if n_shots > 0 else 0.0
    xg_per_shot = total_xg / n_shots if n_shots > 0 else 0.0

    tiers: dict[str, dict] = {}
    for tier_name in tier_thresholds:
        tiers[tier_name] = {"goals": 0, "xg": 0.0, "shots": 0}

    for s in shots:
        xg = s.get("xG", 0.0)
        tn = _tier_for_xg(xg, tier_thresholds)
        tiers[tn]["shots"] += 1
        tiers[tn]["xg"] += xg
        if s.get("goal"):
            tiers[tn]["goals"] += 1

    streak_data = _detect_streaks(shots, streak_window)
    placement_skill = _compute_placement_skill(shots)

    return FinishingReport(
        player_id=player_id,
        total_goals=total_goals,
        total_xg=round(total_xg, 4),
        finishing_delta=round(finishing_delta, 4),
        shot_tiers=tiers,
        streak_data=streak_data,
        placement_skill=round(placement_skill, 4),
        conversion_rate=round(conversion_rate, 4),
        xg_per_shot=round(xg_per_shot, 4),
    )
