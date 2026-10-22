"""Goalkeeper analytics service.

Analyzes goalkeeper-specific metrics:
- Save rate and save quality
- Expected Goals on Target (xGOT) faced
- Distribution accuracy (short, medium, long)
- Sweeper-keeper actions (interceptions outside box)
- Claiming crosses
- Distribution zones

This is a high-impact service for amateur teams where the goalkeeper
is often the most decisive player.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from kawkab.core.game_constants import GAME
from kawkab.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GoalkeeperAction:
    action_type: str
    minute: int
    second: int
    team: str
    player_track_id: int | None = None
    outcome: str = "unknown"
    quality: float = 0.0
    x: float | None = None
    y: float | None = None
    description: str = ""


@dataclass
class GoalkeeperStats:
    """Aggregated goalkeeper stats."""

    team: str
    saves: int = 0
    goals_conceded: int = 0
    shots_faced: int = 0
    save_rate: float = 0.0
    goals_prevented_xgot: float = 0.0
    xgot_per_shot: float = 0.0
    crosses_claimed: int = 0
    crosses_punched: int = 0
    crosses_missed: int = 0
    sweep_actions: int = 0
    short_distribution_attempts: int = 0
    short_distribution_successful: int = 0
    long_distribution_attempts: int = 0
    long_distribution_successful: int = 0
    avg_distribution_distance: float = 0.0
    clean_sheet: bool = False
    action_breakdown: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class GoalkeeperService:
    """Analyze goalkeeper performance.

    All methods take a list of goalkeeper actions and a list of shots
    faced, and compute comprehensive stats.
    """

    PITCH_LENGTH = GAME.PITCH_LENGTH_M
    PITCH_WIDTH = GAME.PITCH_WIDTH_M
    GK_AREA_DEPTH = 5.5
    SHORT_DIST_THRESHOLD = 25.0
    LONG_DIST_THRESHOLD = 50.0

    def __init__(self) -> None:
        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    def compute_xgot_simple(
        self, shot_x: float, shot_y: float, body_part: str = "foot", one_on_one: bool = False
    ) -> float:
        """Estimate xGOT (expected goals on target) for a shot on target.

        Simplified model: distance to goal, angle, body part, one-on-one bonus.
        Real xGOT models use ML trained on millions of shots; this is a
        reasonable approximation for amateur analysis.
        """
        goal_x = self.PITCH_LENGTH
        _ = 34 - 3.66
        _ = 34 + 3.66
        if shot_x > goal_x:
            shot_x = self.PITCH_LENGTH - shot_x
            goal_x = self.PITCH_LENGTH
        dist = math.hypot(goal_x - shot_x, shot_y - 34)
        angle_rad = 0.0
        if dist > 0:
            angle_rad = math.atan2(7.32, dist)
        angle_factor = math.sin(angle_rad) * 0.5
        dist_factor = 1.0 / (1.0 + dist / 20.0)
        base_xgot = (angle_factor + dist_factor * 0.5) * 0.7
        if body_part == "head":
            base_xgot *= 0.5
        elif body_part in {"left_foot", "right_foot"}:
            base_xgot *= 0.95
        if one_on_one:
            base_xgot = min(0.85, base_xgot * 1.3)
        return max(0.0, min(1.0, base_xgot))

    def compute_stats(
        self,
        team: str,
        actions: list[GoalkeeperAction],
        shots_faced: list[dict[str, Any]],
        clean_sheet: bool = False,
    ) -> GoalkeeperStats:
        """Compute goalkeeper stats from actions and shots faced."""
        stats = GoalkeeperStats(team=team, clean_sheet=clean_sheet)
        team_actions = [a for a in actions if a.team == team]
        if not shots_faced and not team_actions and not clean_sheet:
            return stats
        stats.shots_faced = len(shots_faced)
        stats.goals_conceded = sum(1 for s in shots_faced if s.get("outcome") == "goal")
        if stats.shots_faced > 0:
            saves = sum(1 for s in shots_faced if s.get("outcome") in {"save", "goal_kick"})
            stats.saves = saves
            stats.save_rate = round(saves / stats.shots_faced, 3)
        total_xgot = 0.0
        goals_xgot = 0.0
        for s in shots_faced:
            x = float(s.get("x", 88))
            y = float(s.get("y", 34))
            body_part = s.get("body_part", "foot")
            one_on_one = s.get("one_on_one", False)
            xgot = self.compute_xgot_simple(x, y, body_part, one_on_one)
            total_xgot += xgot
            if s.get("outcome") == "goal":
                goals_xgot += xgot
        stats.xgot_per_shot = round(total_xgot / max(1, stats.shots_faced), 3)
        stats.goals_prevented_xgot = round(total_xgot - goals_xgot, 3)
        dist_distances: list[float] = []
        for a in team_actions:
            action_type = a.action_type
            stats.action_breakdown[action_type] = stats.action_breakdown.get(action_type, 0) + 1
            if action_type == "save_cross":
                stats.crosses_claimed += 1
            elif action_type == "punch_cross":
                stats.crosses_punched += 1
            elif action_type == "miss_cross":
                stats.crosses_missed += 1
            elif action_type == "sweep":
                stats.sweep_actions += 1
            elif action_type == "short_dist":
                stats.short_distribution_attempts += 1
                if a.outcome == "complete":
                    stats.short_distribution_successful += 1
                if a.x is not None:
                    dist_distances.append(a.x)
            elif action_type == "long_dist":
                stats.long_distribution_attempts += 1
                if a.outcome == "complete":
                    stats.long_distribution_successful += 1
                if a.x is not None:
                    dist_distances.append(a.x)
        if dist_distances:
            stats.avg_distribution_distance = round(sum(dist_distances) / len(dist_distances), 1)
        notes = self._generate_notes(stats)
        stats.notes = notes
        return stats

    def _generate_notes(self, stats: GoalkeeperStats) -> list[str]:
        notes: list[str] = []
        if stats.clean_sheet:
            notes.append("Clean sheet — no goals conceded")
        if stats.save_rate >= 0.7:
            notes.append(f"Excellent save rate: {stats.save_rate * 100:.0f}%")
        elif stats.save_rate >= 0.5:
            notes.append(f"Good save rate: {stats.save_rate * 100:.0f}%")
        elif stats.shots_faced > 0:
            notes.append(f"Low save rate: {stats.save_rate * 100:.0f}% — coaching review needed")
        if stats.clean_sheet:
            notes.append("Clean sheet — no goals conceded")
        if stats.shots_faced > 0 and stats.xgot_per_shot < 0.15:
            notes.append("GK faced low-quality shots — defense is doing well")
        elif stats.shots_faced > 0 and stats.xgot_per_shot > 0.4:
            notes.append("GK faced high-quality shots — opponent had dangerous chances")
        if stats.short_distribution_attempts > 0:
            short_pct = stats.short_distribution_successful / stats.short_distribution_attempts
            if short_pct < 0.5:
                notes.append(f"Short distribution accuracy low: {short_pct * 100:.0f}%")
        if stats.long_distribution_attempts > 0:
            long_pct = stats.long_distribution_successful / stats.long_distribution_attempts
            if long_pct < 0.3:
                notes.append(f"Long distribution accuracy very low: {long_pct * 100:.0f}%")
        if stats.sweep_actions > 0:
            notes.append(f"Active sweeper: {stats.sweep_actions} actions outside box")
        if stats.crosses_missed > stats.crosses_claimed:
            notes.append("Struggles with cross claiming — needs aerial work")
        if not notes:
            notes.append("No significant GK activity")
        return notes

    def is_in_gk_area(self, x: float, y: float, defending_x: float = 0) -> bool:
        """Check if position is in the goalkeeper area."""
        if defending_x < 52.5:
            return x < self.GK_AREA_DEPTH and 30.34 < y < 37.66
        return x > self.PITCH_LENGTH - self.GK_AREA_DEPTH and 30.34 < y < 37.66
