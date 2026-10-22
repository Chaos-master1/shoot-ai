"""Workload and injury-risk monitoring.

Computes the Acute:Chronic Workload Ratio (ACWR) and related metrics
from a player's training and match load over time. Used by sports
scientists to flag fatigue and injury risk.

References (open literature):
- Hulin et al. (2014): ACWR > 1.5 associated with elevated injury risk
- Gabbett (2016): The "sweet spot" of 0.8 - 1.3
- Banister TRIMP: heart-rate based training impulse
- Foster's session-RPE: sRPE = rating * minutes
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)


class RiskLevel(StrEnum):
    """Categorical injury-risk band."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    INSUFFICIENT_DATA = "insufficient_data"


class WorkloadSource(StrEnum):
    """Type of workload record."""

    MATCH = "match"
    TRAINING = "training"


@dataclass
class WorkloadRecord:
    """A single day of workload for a player."""

    date: str
    source: WorkloadSource
    duration_min: int
    rpe: float = 0.0
    distance_m: float = 0.0
    sprints: int = 0
    high_intensity_m: float = 0.0
    notes: str = ""


@dataclass
class WorkloadReport:
    """Aggregated workload report for a player."""

    player_id: int
    player_name: str
    acute_load: float
    chronic_load: float
    acwr: float
    risk_level: RiskLevel
    acute_7d_minutes: int
    acute_28d_minutes: int
    acute_7d_distance: float
    acute_28d_distance: float
    monotony: float
    strain: float
    fitness_fatigue_trend: list[tuple[str, float]]
    flags: list[str]
    recommendations: list[str]


class WorkloadService:
    """Compute ACWR, monotony, strain, and risk levels.

    Args:
        acute_window_days: Length of acute (recent) window (default 7).
        chronic_window_days: Length of chronic (baseline) window (default 28).
        high_acwr_threshold: ACWR above this is high risk.
        very_high_acwr_threshold: ACWR above this is very high risk.
    """

    ACUTE_DAYS = 7
    CHRONIC_DAYS = 28
    SWEET_SPOT_LOW = 0.8
    SWEET_SPOT_HIGH = 1.3

    def __init__(
        self,
        acute_window_days: int = 7,
        chronic_window_days: int = 28,
        high_acwr_threshold: float = 1.5,
        very_high_acwr_threshold: float = 2.0,
    ) -> None:
        self.acute_window_days = acute_window_days
        self.chronic_window_days = chronic_window_days
        self.high_acwr_threshold = high_acwr_threshold
        self.very_high_acwr_threshold = very_high_acwr_threshold
        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    def analyze(
        self,
        player_id: int,
        player_name: str,
        history: list[WorkloadRecord],
        reference_date: str | None = None,
    ) -> WorkloadReport:
        """Compute workload report.

        Args:
            player_id: Internal player ID.
            player_name: Display name.
            history: Chronological list of workload records.
            reference_date: ISO date to compute "today" against. Defaults
                to the latest record's date.
        """
        if not history:
            return self._empty_report(player_id, player_name)
        sorted_h = sorted(history, key=lambda r: r.date)
        if reference_date is None:
            reference_date = sorted_h[-1].date
        acute = [
            r
            for r in sorted_h
            if self._days_between(r.date, reference_date) <= self.acute_window_days
        ]
        chronic = [
            r
            for r in sorted_h
            if self._days_between(r.date, reference_date) <= self.chronic_window_days
        ]
        acute_load = sum(self._session_load(r) for r in acute)
        chronic_raw = sum(self._session_load(r) for r in chronic)
        chronic_load = chronic_raw / 4.0 if chronic_raw > 0 else 0.0
        acwr = acute_load / chronic_load if chronic_load > 0 else 0.0
        risk = self._classify_risk(acwr, len(acute))
        daily_loads = self._daily_loads(chronic, reference_date)
        monotony = self._compute_monotony(daily_loads)
        weekly_total = sum(self._session_load(r) for r in acute)
        strain = weekly_total * monotony
        flags = self._build_flags(acwr, risk, monotony, acute, chronic)
        recs = self._build_recommendations(risk, flags, acwr)
        return WorkloadReport(
            player_id=player_id,
            player_name=player_name,
            acute_load=round(acute_load, 1),
            chronic_load=round(chronic_load, 1),
            acwr=round(acwr, 3),
            risk_level=risk,
            acute_7d_minutes=sum(r.duration_min for r in acute),
            acute_28d_minutes=sum(r.duration_min for r in chronic),
            acute_7d_distance=round(sum(r.distance_m for r in acute), 1),
            acute_28d_distance=round(sum(r.distance_m for r in chronic), 1),
            monotony=round(monotony, 3),
            strain=round(strain, 1),
            fitness_fatigue_trend=daily_loads,
            flags=flags,
            recommendations=recs,
        )

    def _empty_report(self, player_id: int, player_name: str) -> WorkloadReport:
        return WorkloadReport(
            player_id=player_id,
            player_name=player_name,
            acute_load=0.0,
            chronic_load=0.0,
            acwr=0.0,
            risk_level=RiskLevel.INSUFFICIENT_DATA,
            acute_7d_minutes=0,
            acute_28d_minutes=0,
            acute_7d_distance=0.0,
            acute_28d_distance=0.0,
            monotony=0.0,
            strain=0.0,
            fitness_fatigue_trend=[],
            flags=[],
            recommendations=["No workload data — start logging sessions."],
        )

    @staticmethod
    def _session_load(record: WorkloadRecord) -> float:
        if record.rpe > 0:
            return record.rpe * record.duration_min
        if record.source == WorkloadSource.MATCH:
            return 7.0 * record.duration_min
        return 4.0 * record.duration_min

    @staticmethod
    def _days_between(d1: str, d2: str) -> int:
        try:
            from datetime import date

            a = date.fromisoformat(d1)
            b = date.fromisoformat(d2)
            return abs((b - a).days)
        except (ValueError, TypeError):
            return 0

    def _daily_loads(
        self, records: list[WorkloadRecord], reference_date: str
    ) -> list[tuple[str, float]]:
        daily: dict[str, float] = {}
        for r in records:
            if self._days_between(r.date, reference_date) > self.chronic_window_days:
                continue
            daily[r.date] = daily.get(r.date, 0.0) + self._session_load(r)
        return sorted(daily.items(), key=lambda x: x[0])

    def _compute_monotony(self, daily_loads: list[tuple[str, float]]) -> float:
        if len(daily_loads) < 2:
            return 0.0
        values = [v for _, v in daily_loads]
        mean = sum(values) / len(values)
        if mean == 0:
            return 0.0
        std = statistics.pstdev(values)
        return mean / std if std > 0 else 0.0

    def _classify_risk(self, acwr: float, n_acute: int) -> RiskLevel:
        if n_acute == 0:
            return RiskLevel.INSUFFICIENT_DATA
        if acwr >= self.very_high_acwr_threshold:
            return RiskLevel.VERY_HIGH
        if acwr >= self.high_acwr_threshold:
            return RiskLevel.HIGH
        if acwr < self.SWEET_SPOT_LOW:
            return RiskLevel.MODERATE
        return RiskLevel.LOW

    def _build_flags(
        self,
        acwr: float,
        risk: RiskLevel,
        monotony: float,
        acute: list[WorkloadRecord],
        chronic: list[WorkloadRecord],
    ) -> list[str]:
        flags: list[str] = []
        if acwr >= self.high_acwr_threshold:
            flags.append(f"ACWR {acwr:.2f} exceeds {self.high_acwr_threshold} threshold")
        if acwr < self.SWEET_SPOT_LOW and acwr > 0:
            flags.append(f"ACWR {acwr:.2f} below sweet spot (under-training)")
        if monotony > 2.0:
            flags.append(f"High monotony {monotony:.2f} — repetitive training load")
        if len(acute) > len(chronic) * 0.6:
            flags.append("More than 60% of chronic window is acute — short baseline")
        return flags

    def _build_recommendations(
        self,
        risk: RiskLevel,
        flags: list[str],
        acwr: float,
    ) -> list[str]:
        recs: list[str] = []
        if risk == RiskLevel.VERY_HIGH:
            recs.append("Reduce training intensity for 48-72h; consider a rest day.")
        elif risk == RiskLevel.HIGH:
            recs.append("Consider a deload session; monitor for fatigue symptoms.")
        elif risk == RiskLevel.LOW and acwr > 0:
            recs.append("Load is in the sweet spot — maintain current plan.")
        if any("monotony" in f for f in flags):
            recs.append("Vary training stimulus to reduce monotony injury risk.")
        if not recs:
            recs.append("Continue current workload plan.")
        return recs
