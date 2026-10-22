"""Periodization analysis for multi-week training/match planning.

Aggregates player development and workload data across multiple weeks
to provide macro-level planning insight:

- Weekly load totals with trend
- Taper detection (planned load reduction)
- Fixture congestion (matches per week)
- Recovery index
- Peaking detection (load + performance both high)
- Microcycle / mesocycle / macrocycle boundaries
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class CyclePhase(StrEnum):
    """Where a week sits in a training cycle."""

    PREPARATION = "preparation"
    BUILD = "build"
    PEAK = "peak"
    TAPER = "taper"
    RECOVERY = "recovery"
    COMPETITION = "competition"
    UNKNOWN = "unknown"


class CongestionLevel(StrEnum):
    """How packed the match schedule is."""

    LIGHT = "light"
    NORMAL = "normal"
    CONGESTED = "congested"
    OVERLOADED = "overloaded"


@dataclass
class WeekSummary:
    """Aggregated stats for one week."""

    week_start: str
    matches: int
    training_sessions: int
    total_minutes: int
    total_distance_m: float
    total_load: float
    avg_rpe: float
    congestion: CongestionLevel
    phase: CyclePhase
    is_recovery: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class PeriodizationReport:
    """Multi-week periodization analysis for a player."""

    player_id: int
    player_name: str
    weeks: list[WeekSummary]
    total_weeks: int
    avg_weekly_load: float
    load_trend: str
    peak_weeks: list[str]
    taper_weeks: list[str]
    congestion_weeks: list[str]
    macro_recommendations: list[str]
    notes: list[str]


class PeriodizationService:
    """Analyze multi-week load, congestion, and periodization.

    Args:
        congested_matches_threshold: Matches per week considered congested.
        overloaded_matches_threshold: Matches per week considered overloaded.
        taper_load_drop_pct: How much the load must drop week-over-week
            to count as a taper (default 0.20 = 20%).
    """

    def __init__(
        self,
        congested_matches_threshold: int = 2,
        overloaded_matches_threshold: int = 3,
        taper_load_drop_pct: float = 0.20,
    ) -> None:
        self.congested_matches_threshold = congested_matches_threshold
        self.overloaded_matches_threshold = overloaded_matches_threshold
        self.taper_load_drop_pct = taper_load_drop_pct
        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    def analyze(
        self,
        player_id: int,
        player_name: str,
        history: list[dict[str, Any]],
    ) -> PeriodizationReport:
        """Build a periodization report.

        Args:
            player_id: Player ID.
            player_name: Player display name.
            history: List of records, each with keys:
              - week_start: ISO date of Monday
              - date: ISO date
              - source: 'match' or 'training'
              - duration_min: int
              - rpe: float (optional, default 0)
              - distance_m: float (optional)
        """
        if not history:
            return self._empty_report(player_id, player_name)
        by_week: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in history:
            wk = r.get("week_start")
            if not wk:
                from datetime import date as _date

                d = _date.fromisoformat(r["date"])
                wk = d.isoformat()
            by_week[wk].append(r)
        weeks_sorted = sorted(by_week.keys())
        summaries: list[WeekSummary] = []
        for wk in weeks_sorted:
            week_records = by_week[wk]
            summary = self._build_week_summary(wk, week_records)
            if weeks_sorted:
                prev_loads = [self._calc_total_load(by_week[w]) for w in weeks_sorted if w < wk]
                if prev_loads:
                    prev_load = prev_loads[-1]
                    if prev_load > 0:
                        drop = (prev_load - summary.total_load) / prev_load
                        if drop >= self.taper_load_drop_pct:
                            summary.phase = CyclePhase.TAPER
                            summary.notes.append(f"taper detected (load drop {drop:.0%})")
            summaries.append(summary)
        peak_weeks = [
            w.week_start for w in summaries if w.phase in (CyclePhase.PEAK, CyclePhase.COMPETITION)
        ]
        taper_weeks = [w.week_start for w in summaries if w.phase == CyclePhase.TAPER]
        congestion_weeks = [
            w.week_start
            for w in summaries
            if w.congestion in (CongestionLevel.CONGESTED, CongestionLevel.OVERLOADED)
        ]
        avg_load = statistics.mean(w.total_load for w in summaries) if summaries else 0.0
        load_trend = self._compute_load_trend(summaries)
        recs = self._build_macro_recommendations(summaries, congestion_weeks, taper_weeks)
        notes = self._build_notes(summaries, load_trend)
        return PeriodizationReport(
            player_id=player_id,
            player_name=player_name,
            weeks=summaries,
            total_weeks=len(summaries),
            avg_weekly_load=round(avg_load, 1),
            load_trend=load_trend,
            peak_weeks=peak_weeks,
            taper_weeks=taper_weeks,
            congestion_weeks=congestion_weeks,
            macro_recommendations=recs,
            notes=notes,
        )

    def _empty_report(self, player_id: int, player_name: str) -> PeriodizationReport:
        return PeriodizationReport(
            player_id=player_id,
            player_name=player_name,
            weeks=[],
            total_weeks=0,
            avg_weekly_load=0.0,
            load_trend="unknown",
            peak_weeks=[],
            taper_weeks=[],
            congestion_weeks=[],
            macro_recommendations=["No history available."],
            notes=[],
        )

    def _build_week_summary(self, week_start: str, records: list[dict[str, Any]]) -> WeekSummary:
        matches = sum(1 for r in records if r.get("source") == "match")
        training = sum(1 for r in records if r.get("source") == "training")
        total_min = sum(int(r.get("duration_min", 0)) for r in records)
        total_dist = sum(float(r.get("distance_m", 0.0)) for r in records)
        rpes = [float(r.get("rpe", 0.0)) for r in records if r.get("rpe", 0.0) > 0]
        avg_rpe = statistics.mean(rpes) if rpes else 0.0
        total_load = self._calc_total_load(records)
        congestion = self._classify_congestion(matches)
        phase = self._classify_phase(matches, total_min, total_load, training)
        is_recovery = matches == 0 and total_min < 120
        notes: list[str] = []
        if matches >= self.overloaded_matches_threshold:
            notes.append(f"overloaded fixture week ({matches} matches)")
        if is_recovery:
            notes.append("recovery week (no matches, low total minutes)")
        return WeekSummary(
            week_start=week_start,
            matches=matches,
            training_sessions=training,
            total_minutes=total_min,
            total_distance_m=round(total_dist, 1),
            total_load=round(total_load, 1),
            avg_rpe=round(avg_rpe, 2),
            congestion=congestion,
            phase=phase,
            is_recovery=is_recovery,
            notes=notes,
        )

    @staticmethod
    def _calc_total_load(records: list[dict[str, Any]]) -> float:
        total = 0.0
        for r in records:
            rpe = float(r.get("rpe", 0.0))
            duration = int(r.get("duration_min", 0))
            if rpe > 0:
                total += rpe * duration
            else:
                base = 7.0 if r.get("source") == "match" else 4.0
                total += base * duration
        return total

    def _classify_congestion(self, matches: int) -> CongestionLevel:
        if matches >= self.overloaded_matches_threshold:
            return CongestionLevel.OVERLOADED
        if matches >= self.congested_matches_threshold:
            return CongestionLevel.CONGESTED
        if matches == 1:
            return CongestionLevel.NORMAL
        return CongestionLevel.LIGHT

    @staticmethod
    def _classify_phase(
        matches: int, total_min: int, total_load: float, training: int
    ) -> CyclePhase:
        if matches == 0 and total_min < 180:
            return CyclePhase.RECOVERY
        if matches >= 2:
            return CyclePhase.COMPETITION
        if training >= 4 and total_load > 2000:
            return CyclePhase.PEAK
        if training >= 3 and total_load > 1500:
            return CyclePhase.BUILD
        if training > 0:
            return CyclePhase.PREPARATION
        return CyclePhase.UNKNOWN

    @staticmethod
    def _compute_load_trend(weeks: list[WeekSummary]) -> str:
        if len(weeks) < 2:
            return "insufficient_data"
        loads = [w.total_load for w in weeks]
        n = len(loads)
        x = list(range(n))
        mean_x = sum(x) / n
        mean_y = sum(loads) / n
        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, loads, strict=False))
        den = sum((xi - mean_x) ** 2 for xi in x)
        slope = num / den if den > 0 else 0.0
        if slope > 50:
            return "increasing"
        if slope < -50:
            return "decreasing"
        return "stable"

    def _build_macro_recommendations(
        self,
        weeks: list[WeekSummary],
        congestion_weeks: list[str],
        taper_weeks: list[str],
    ) -> list[str]:
        recs: list[str] = []
        if len(congestion_weeks) > len(weeks) * 0.3:
            recs.append("Schedule a recovery microcycle after each congested week")
        if not taper_weeks and len(weeks) >= 4:
            recs.append("No taper detected — plan a deload before next peak")
        if len(weeks) >= 2:
            increasing = sum(1 for w in weeks if w.phase == CyclePhase.BUILD)
            if increasing > len(weeks) * 0.5:
                recs.append("Build phase dominance — schedule peak/taper soon")
        if not recs:
            recs.append("Periodization looks balanced")
        return recs

    @staticmethod
    def _build_notes(weeks: list[WeekSummary], load_trend: str) -> list[str]:
        notes: list[str] = []
        if load_trend == "increasing":
            notes.append("Weekly load is trending upward — monitor for overtraining")
        elif load_trend == "decreasing":
            notes.append("Weekly load is trending downward — possible taper or off-season")
        return notes

    def classify_macrocycle(self, weeks: list[WeekSummary]) -> dict[str, Any]:
        """Classify a sequence of weeks into a macrocycle structure.

        Returns a dict with phase counts and cycle length recommendation.
        """
        phase_counts: dict[str, int] = defaultdict(int)
        for w in weeks:
            phase_counts[w.phase.value] += 1
        build = phase_counts.get(CyclePhase.BUILD.value, 0)
        peak = phase_counts.get(CyclePhase.PEAK.value, 0) + phase_counts.get(
            CyclePhase.COMPETITION.value, 0
        )
        recovery = phase_counts.get(CyclePhase.RECOVERY.value, 0)
        taper = phase_counts.get(CyclePhase.TAPER.value, 0)
        if build >= 3 and peak >= 1 and recovery >= 1:
            cycle_type = "well-structured"
        elif build >= 4 and (peak + recovery) < 2:
            cycle_type = "build-heavy"
        elif peak >= 3 and build < 2:
            cycle_type = "competition-heavy"
        else:
            cycle_type = "irregular"
        return {
            "cycle_type": cycle_type,
            "phase_counts": dict(phase_counts),
            "build_weeks": build,
            "peak_weeks": peak,
            "recovery_weeks": recovery,
            "taper_weeks": taper,
            "recommendation": self._macrocycle_recommendation(cycle_type, phase_counts),
        }

    @staticmethod
    def _macrocycle_recommendation(cycle_type: str, phase_counts: dict[str, int]) -> str:
        if cycle_type == "well-structured":
            return "Macrocycle is balanced — maintain current plan"
        if cycle_type == "build-heavy":
            return "Too much build without peak/recovery — add a competition peak and recovery week"
        if cycle_type == "competition-heavy":
            return "Too many competition weeks — add training blocks between peaks"
        return "Macrocycle structure needs review with the coaching staff"
