"""Substitution impact analysis service.

Analyzes how each substitution affected match dynamics:
- xG delta before/after
- Possession delta
- Defensive stability
- Player ratings
- Tactical formation change
- Win probability impact (if trained model available)

Useful for:
- Coach reviewing halftime adjustments
- Player rotation analysis
- Building a "who to sub next" decision tool
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SubstitutionEvent:
    """A single substitution event."""

    minute: int
    second: int
    team: str
    player_off_track_id: int | None = None
    player_off_name: str | None = None
    player_on_track_id: int | None = None
    player_on_name: str | None = None
    formation_before: str | None = None
    formation_after: str | None = None
    position_changed: bool = False


@dataclass
class SubstitutionImpact:
    """Computed impact of one substitution."""

    substitution: SubstitutionEvent
    window_minutes: int
    xg_delta: float
    possession_delta: float
    shots_delta: int
    corners_delta: int
    goals_for: int
    goals_against: int
    rating: float  # -1.0 (very bad) to 1.0 (very good)
    verdict: str  # 'positive', 'negative', 'neutral', 'tactical'
    notes: list[str] = field(default_factory=list)


@dataclass
class SubstitutionReport:
    """Full substitution report for a match."""

    team: str
    impacts: list[SubstitutionImpact]
    best_sub: SubstitutionImpact | None
    worst_sub: SubstitutionImpact | None
    total_impact: float
    tactical_changes: int
    formation_changes: int
    avg_impact: float


class SubstitutionService:
    """Analyze the impact of each substitution in a match.

    Compares match state in a window before the sub vs after the sub.
    Default window: 10 minutes (5 min before, 5 min after).
    """

    DEFAULT_WINDOW_MIN = 10

    def __init__(self, window_minutes: int = DEFAULT_WINDOW_MIN) -> None:
        self.window_minutes = window_minutes
        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    def analyze(
        self,
        team: str,
        subs: list[SubstitutionEvent],
        events: list[dict[str, Any]],
    ) -> SubstitutionReport:
        """Analyze all subs for one team."""
        impacts: list[SubstitutionImpact] = []
        team_subs = [s for s in subs if s.team == team]
        for sub in team_subs:
            impact = self._analyze_single_sub(sub, events)
            impacts.append(impact)
        if not impacts:
            return SubstitutionReport(
                team=team,
                impacts=[],
                best_sub=None,
                worst_sub=None,
                total_impact=0.0,
                tactical_changes=0,
                formation_changes=0,
                avg_impact=0.0,
            )
        best = max(impacts, key=lambda i: i.rating)
        worst = min(impacts, key=lambda i: i.rating)
        total = sum(i.rating for i in impacts)
        avg = total / len(impacts)
        tactical = sum(1 for s in team_subs if s.position_changed)
        formation = sum(
            1
            for s in team_subs
            if s.formation_before is not None
            and s.formation_after is not None
            and s.formation_before != s.formation_after
        )
        return SubstitutionReport(
            team=team,
            impacts=impacts,
            best_sub=best,
            worst_sub=worst,
            total_impact=total,
            tactical_changes=tactical,
            formation_changes=formation,
            avg_impact=round(avg, 3),
        )

    def _analyze_single_sub(
        self,
        sub: SubstitutionEvent,
        events: list[dict[str, Any]],
    ) -> SubstitutionImpact:
        """Analyze one substitution's impact."""
        t0 = sub.minute
        win = self.window_minutes
        pre_start = max(0, t0 - win)
        post_end = t0 + win
        pre_events = [e for e in events if pre_start <= e.get("minute", 0) < t0]
        post_events = [e for e in events if t0 <= e.get("minute", 0) < post_end]
        pre_xg = sum(e.get("xg", 0) for e in pre_events)
        post_xg = sum(e.get("xg", 0) for e in post_events)
        pre_pos = self._team_possession(pre_events, sub.team)
        post_pos = self._team_possession(post_events, sub.team)
        pre_shots = sum(
            1 for e in pre_events if e.get("type") == "shot" and e.get("team") == sub.team
        )
        post_shots = sum(
            1 for e in post_events if e.get("type") == "shot" and e.get("team") == sub.team
        )
        pre_corners = sum(
            1
            for e in pre_events
            if e.get("type") in {"corner", "shot"} and e.get("team") == sub.team
        )
        post_corners = sum(
            1
            for e in post_events
            if e.get("type") in {"corner", "shot"} and e.get("team") == sub.team
        )
        goals_for = sum(
            1 for e in post_events if e.get("type") == "goal" and e.get("team") == sub.team
        )
        goals_against = sum(
            1 for e in post_events if e.get("type") == "goal" and e.get("team") != sub.team
        )
        xg_delta = post_xg - pre_xg
        pos_delta = post_pos - pre_pos
        shots_delta = post_shots - pre_shots
        corners_delta = post_corners - pre_corners
        rating = self._compute_rating(
            xg_delta=xg_delta,
            pos_delta=pos_delta,
            shots_delta=shots_delta,
            corners_delta=corners_delta,
            goals_for=goals_for,
            goals_against=goals_against,
        )
        verdict = self._verdict(rating, goals_for, goals_against, sub)
        notes: list[str] = []
        if goals_for > 0:
            notes.append(f"Team scored {goals_for} goal(s) after sub")
        if goals_against > 0:
            notes.append(f"Team conceded {goals_against} goal(s) after sub")
        if xg_delta > 0.2:
            notes.append(f"xG improved by {xg_delta:.2f}")
        if xg_delta < -0.2:
            notes.append(f"xG dropped by {abs(xg_delta):.2f}")
        if (
            sub.formation_before
            and sub.formation_after
            and sub.formation_before != sub.formation_after
        ):
            notes.append(f"Formation change: {sub.formation_before} → {sub.formation_after}")
        if not notes:
            notes.append("No significant impact")
        return SubstitutionImpact(
            substitution=sub,
            window_minutes=win,
            xg_delta=round(xg_delta, 3),
            possession_delta=round(pos_delta, 3),
            shots_delta=shots_delta,
            corners_delta=corners_delta,
            goals_for=goals_for,
            goals_against=goals_against,
            rating=round(rating, 3),
            verdict=verdict,
            notes=notes,
        )

    def _team_possession(self, events: list[dict[str, Any]], team: str) -> float:
        """Estimate team possession % from pass events."""
        passes = [e for e in events if e.get("type") == "pass"]
        if not passes:
            return 50.0
        team_passes = [p for p in passes if p.get("team") == team]
        completed = sum(1 for p in team_passes if p.get("completed", False))
        total_completed = sum(1 for p in passes if p.get("completed", False))
        if total_completed == 0:
            return 50.0
        return completed / total_completed * 100.0

    def _compute_rating(
        self,
        xg_delta: float,
        pos_delta: float,
        shots_delta: int,
        corners_delta: int,
        goals_for: int,
        goals_against: int,
    ) -> float:
        """Compute sub rating in [-1, 1]."""
        score = 0.0
        score += max(-0.5, min(0.5, xg_delta))
        score += max(-0.3, min(0.3, pos_delta / 100.0))
        score += max(-0.2, min(0.2, shots_delta * 0.1))
        score += max(-0.1, min(0.1, corners_delta * 0.05))
        score += goals_for * 0.5
        score -= goals_against * 0.4
        return max(-1.0, min(1.0, score))

    def _verdict(
        self, rating: float, goals_for: int, goals_against: int, sub: SubstitutionEvent
    ) -> str:
        """Classify sub outcome."""
        if rating >= 0.4:
            return "positive"
        if rating <= -0.4:
            return "negative"
        if sub.formation_before != sub.formation_after:
            return "tactical"
        return "neutral"

    def compare_sub_pairs(
        self, impacts: list[SubstitutionImpact]
    ) -> list[tuple[SubstitutionImpact, SubstitutionImpact, float]]:
        """Find pairs of subs with similar ratings; returns the diff.

        Useful for comparing two halves' subs to find the most impactful.
        """
        results: list[tuple[SubstitutionImpact, SubstitutionImpact, float]] = []
        for i in range(len(impacts)):
            for j in range(i + 1, len(impacts)):
                a, b = impacts[i], impacts[j]
                diff = abs(a.rating - b.rating)
                results.append((a, b, diff))
        results.sort(key=lambda r: r[2], reverse=True)
        return results[:5]
