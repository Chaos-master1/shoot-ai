"""Player development tracking over multiple matches.

Builds per-player trends across a season from individual match stats:
- Pass completion rate
- Distance covered / sprint count
- xG per 90 / xT per 90
- Pressure actions per match
- Formation position adherence
- Injury/workload risk

Detects:
- Improvement (positive slope over rolling window)
- Regression (negative slope)
- Stagnation (flat trend)
- Volatility (high standard deviation)
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)


class TrendDirection(StrEnum):
    """Direction of a player's trend over a rolling window."""

    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    VOLATILE = "volatile"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class PlayerMatchStat:
    """A single match's stats for one player."""

    match_id: int
    match_date: str
    minutes_played: int
    passes_attempted: int = 0
    passes_completed: int = 0
    distance_m: float = 0.0
    sprints: int = 0
    xg: float = 0.0
    xt: float = 0.0
    pressure_actions: int = 0
    goals: int = 0
    assists: int = 0
    touches: int = 0


@dataclass
class PlayerTrend:
    """Trend summary for one metric."""

    metric: str
    direction: TrendDirection
    slope_per_match: float
    rolling_avg: float
    rolling_std: float
    n_matches: int
    last_value: float
    best_value: float
    worst_value: float


@dataclass
class PlayerDevelopmentReport:
    """Full development report for a player."""

    player_id: int
    player_name: str
    position: str
    matches_played: int
    trends: list[PlayerTrend]
    overall_trend: TrendDirection
    strengths: list[str]
    areas_to_improve: list[str]
    notes: list[str]


class PlayerDevelopmentService:
    """Track per-player development across multiple matches.

    Args:
        min_matches_for_trend: Minimum matches required to compute a trend.
        rolling_window: How many recent matches to use for rolling stats.
        improvement_threshold: Slope magnitude to count as improving/declining.
    """

    def __init__(
        self,
        min_matches_for_trend: int = 3,
        rolling_window: int = 5,
        improvement_threshold: float = 0.05,
    ) -> None:
        self.min_matches_for_trend = min_matches_for_trend
        self.rolling_window = rolling_window
        self.improvement_threshold = improvement_threshold
        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    def analyze(
        self,
        player_id: int,
        player_name: str,
        position: str,
        history: list[PlayerMatchStat],
    ) -> PlayerDevelopmentReport:
        """Compute development trends for one player."""
        if len(history) < self.min_matches_for_trend:
            return PlayerDevelopmentReport(
                player_id=player_id,
                player_name=player_name,
                position=position,
                matches_played=len(history),
                trends=[],
                overall_trend=TrendDirection.INSUFFICIENT_DATA,
                strengths=[],
                areas_to_improve=[],
                notes=[f"Need at least {self.min_matches_for_trend} matches, got {len(history)}."],
            )
        history_sorted = sorted(history, key=lambda h: h.match_date)
        metrics = [
            "pass_completion",
            "distance_per_90",
            "sprints_per_90",
            "xg_per_90",
            "xt_per_90",
            "pressure_per_90",
        ]
        trends: list[PlayerTrend] = []
        for metric in metrics:
            values = [self._extract_metric(m, metric) for m in history_sorted]
            t = self._compute_trend(metric, values)
            trends.append(t)
        overall = self._aggregate_trend(trends)
        strengths = self._identify_strengths(trends, history_sorted)
        improvements = self._identify_improvements(trends, history_sorted)
        notes = self._generate_notes(overall, trends, history_sorted)
        return PlayerDevelopmentReport(
            player_id=player_id,
            player_name=player_name,
            position=position,
            matches_played=len(history_sorted),
            trends=trends,
            overall_trend=overall,
            strengths=strengths,
            areas_to_improve=improvements,
            notes=notes,
        )

    @staticmethod
    def _per_90(value: float, minutes: int) -> float:
        if minutes <= 0:
            return 0.0
        return value * 90.0 / minutes

    def _extract_metric(self, m: PlayerMatchStat, metric: str) -> float:
        if metric == "pass_completion":
            return m.passes_completed / m.passes_attempted if m.passes_attempted > 0 else 0.0
        if metric == "distance_per_90":
            return self._per_90(m.distance_m, m.minutes_played)
        if metric == "sprints_per_90":
            return self._per_90(m.sprints, m.minutes_played)
        if metric == "xg_per_90":
            return self._per_90(m.xg, m.minutes_played)
        if metric == "xt_per_90":
            return self._per_90(m.xt, m.minutes_played)
        if metric == "pressure_per_90":
            return self._per_90(m.pressure_actions, m.minutes_played)
        return 0.0

    def _compute_trend(self, metric: str, values: list[float]) -> PlayerTrend:
        n = len(values)
        if n < self.min_matches_for_trend:
            return PlayerTrend(
                metric=metric,
                direction=TrendDirection.INSUFFICIENT_DATA,
                slope_per_match=0.0,
                rolling_avg=0.0,
                rolling_std=0.0,
                n_matches=n,
                last_value=values[-1] if values else 0.0,
                best_value=max(values) if values else 0.0,
                worst_value=min(values) if values else 0.0,
            )
        x = list(range(n))
        y = values
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=False))
        den = sum((xi - mean_x) ** 2 for xi in x)
        slope = num / den if den > 0 else 0.0
        window = values[-self.rolling_window :]
        rolling_avg = sum(window) / len(window)
        rolling_std = statistics.pstdev(window) if len(window) > 1 else 0.0
        cv = rolling_std / abs(rolling_avg) if rolling_avg != 0 else 0.0
        if cv > 0.3 and n >= self.rolling_window:
            direction = TrendDirection.VOLATILE
        elif slope > self.improvement_threshold:
            direction = TrendDirection.IMPROVING
        elif slope < -self.improvement_threshold:
            direction = TrendDirection.DECLINING
        else:
            direction = TrendDirection.STABLE
        return PlayerTrend(
            metric=metric,
            direction=direction,
            slope_per_match=round(slope, 4),
            rolling_avg=round(rolling_avg, 3),
            rolling_std=round(rolling_std, 3),
            n_matches=n,
            last_value=round(values[-1], 3),
            best_value=round(max(values), 3),
            worst_value=round(min(values), 3),
        )

    def _aggregate_trend(self, trends: list[PlayerTrend]) -> TrendDirection:
        valid = [t for t in trends if t.direction != TrendDirection.INSUFFICIENT_DATA]
        if not valid:
            return TrendDirection.INSUFFICIENT_DATA
        counts: dict[TrendDirection, int] = {
            TrendDirection.IMPROVING: 0,
            TrendDirection.DECLINING: 0,
            TrendDirection.STABLE: 0,
            TrendDirection.VOLATILE: 0,
        }
        for t in valid:
            counts[t.direction] = counts.get(t.direction, 0) + 1
        if counts[TrendDirection.VOLATILE] >= len(valid) / 2:
            return TrendDirection.VOLATILE
        if counts[TrendDirection.IMPROVING] > counts[TrendDirection.DECLINING]:
            return TrendDirection.IMPROVING
        if counts[TrendDirection.DECLINING] > counts[TrendDirection.IMPROVING]:
            return TrendDirection.DECLINING
        return TrendDirection.STABLE

    def _identify_strengths(
        self, trends: list[PlayerTrend], history: list[PlayerMatchStat]
    ) -> list[str]:
        strengths: list[str] = []
        for t in trends:
            if (
                t.direction in (TrendDirection.IMPROVING, TrendDirection.STABLE)
                and t.rolling_avg > 0
            ):
                strengths.append(
                    f"{t.metric}: avg {t.rolling_avg:.2f} (last {t.n_matches} matches)"
                )
        return strengths[:5]

    def _identify_improvements(
        self, trends: list[PlayerTrend], history: list[PlayerMatchStat]
    ) -> list[str]:
        improvements: list[str] = []
        for t in trends:
            if t.direction == TrendDirection.DECLINING:
                improvements.append(
                    f"{t.metric}: declining slope {t.slope_per_match:.3f} per match"
                )
            elif t.direction == TrendDirection.VOLATILE:
                improvements.append(f"{t.metric}: high variance (std={t.rolling_std:.2f})")
        return improvements[:5]

    def _generate_notes(
        self,
        overall: TrendDirection,
        trends: list[PlayerTrend],
        history: list[PlayerMatchStat],
    ) -> list[str]:
        notes: list[str] = []
        if overall == TrendDirection.IMPROVING:
            notes.append("Player is on an upward trajectory across multiple metrics.")
        elif overall == TrendDirection.DECLINING:
            notes.append("Performance is declining — consider load management or tactical review.")
        elif overall == TrendDirection.VOLATILE:
            notes.append("Inconsistent performance — investigate match-by-match context.")
        if history and (history[-1].minutes_played < 45):
            notes.append("Last match had limited minutes; trends may be noisy.")
        return notes
