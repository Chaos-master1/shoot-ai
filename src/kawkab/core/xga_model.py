"""Expected Goals Against (xGA) — measures defensive shot quality allowed.

xGA sums the xG values of all shots a team concedes, quantifying how
dangerous the opposition's chances were. All numpy-only.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kawkab.core.game_constants import GAME

PITCH_LENGTH = GAME.PITCH_LENGTH_M
PITCH_WIDTH = GAME.PITCH_WIDTH_M


def _to_zone(x: float, y: float, nx: int, ny: int) -> tuple[int, int]:
    zx = min(int(x / PITCH_LENGTH * nx), nx - 1)
    zy = min(int(y / PITCH_WIDTH * ny), ny - 1)
    return (zx, zy)


def _extract_xg(event: dict[str, Any]) -> float:
    return float(event.get("xg", event.get("xG", 0.0)))


def _get_shot_type(event: dict[str, Any]) -> str:
    body = event.get("body_part", event.get("metadata", {}).get("body_part", ""))
    if not body:
        body = event.get("shot_type", "open_play")
    return body


def _get_situation(event: dict[str, Any]) -> str:
    meta = event.get("metadata", {})
    if isinstance(meta, str):
        return "open_play"
    st = meta.get("set_piece", "") or event.get("set_piece", "")
    if st in ("penalty",):
        return "penalty"
    if st in ("counter_attack", "counter"):
        return "counter_attack"
    if st:
        return "set_piece"
    return "open_play"


@dataclass
class XGAReport:
    total_xga: float = 0.0
    shots_faced: int = 0
    actual_goals_conceded: int = 0
    save_pct: float = 0.0
    goals_prevented: float = 0.0
    zone_breakdown: dict[str, float] = field(default_factory=dict)
    type_breakdown: dict[str, float] = field(default_factory=dict)
    situation_breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_xga": round(self.total_xga, 3),
            "shots_faced": self.shots_faced,
            "actual_goals_conceded": self.actual_goals_conceded,
            "save_pct": round(self.save_pct, 3),
            "goals_prevented": round(self.goals_prevented, 3),
            "zone_breakdown": {k: round(v, 3) for k, v in self.zone_breakdown.items()},
            "type_breakdown": {k: round(v, 3) for k, v in self.type_breakdown.items()},
            "situation_breakdown": {k: round(v, 3) for k, v in self.situation_breakdown.items()},
        }


class ExpectedGoalsAgainstModel:
    """Measures the quality of shots a team concedes (xGA).

    Args:
        zone_grid: (nx, ny) for the zone breakdown grid. Default (5, 4).
    """

    def __init__(self, zone_grid: tuple[int, int] = (5, 4)):
        self.zone_grid = zone_grid

    def compute_xga(self, events: list[dict[str, Any]], team: str) -> float:
        shots = [e for e in events if e.get("type") == "shot" and e.get("team", "") != team]
        return float(np.sum([_extract_xg(s) for s in shots]))

    def compute_xga_by_zone(self, events: list[dict[str, Any]], team: str) -> dict[str, float]:
        nx, ny = self.zone_grid
        zone_totals: dict[str, float] = defaultdict(float)
        for e in events:
            if e.get("type") != "shot" or e.get("team", "") == team:
                continue
            x = float(e.get("start_x", e.get("x", PITCH_LENGTH / 2)))
            y = float(e.get("start_y", e.get("y", PITCH_WIDTH / 2)))
            zx, zy = _to_zone(x, y, nx, ny)
            key = f"{zx}_{zy}"
            zone_totals[key] += _extract_xg(e)
        return dict(zone_totals)

    def compute_xga_by_type(
        self, events: list[dict[str, Any]], team: str
    ) -> tuple[dict[str, float], dict[str, float]]:
        type_totals: dict[str, float] = defaultdict(float)
        sit_totals: dict[str, float] = defaultdict(float)
        for e in events:
            if e.get("type") != "shot" or e.get("team", "") == team:
                continue
            xg = _extract_xg(e)
            st = _get_shot_type(e)
            sit = _get_situation(e)
            type_totals[st] += xg
            sit_totals[sit] += xg
        return dict(type_totals), dict(sit_totals)

    def compute_xga_save_pct(self, events: list[dict[str, Any]], team: str) -> float:
        shots = [e for e in events if e.get("type") == "shot" and e.get("team", "") != team]
        total_xga = float(np.sum([_extract_xg(s) for s in shots]))
        goals_conceded = sum(1 for s in shots if s.get("is_goal"))
        if total_xga == 0:
            return 0.0
        return max(0.0, 1.0 - (goals_conceded / total_xga))

    def compute_full_report(self, events: list[dict[str, Any]], team: str) -> XGAReport:
        shots = [e for e in events if e.get("type") == "shot" and e.get("team", "") != team]
        total_xga = float(np.sum([_extract_xg(s) for s in shots]))
        goals_conceded = sum(1 for s in shots if s.get("is_goal"))
        save_pct = 0.0
        if total_xga > 0:
            save_pct = max(0.0, 1.0 - (goals_conceded / total_xga))
        goals_prevented = total_xga - goals_conceded
        zone_bd = self.compute_xga_by_zone(events, team)
        type_bd, sit_bd = self.compute_xga_by_type(events, team)

        return XGAReport(
            total_xga=total_xga,
            shots_faced=len(shots),
            actual_goals_conceded=goals_conceded,
            save_pct=save_pct,
            goals_prevented=goals_prevented,
            zone_breakdown=zone_bd,
            type_breakdown=type_bd,
            situation_breakdown=sit_bd,
        )
