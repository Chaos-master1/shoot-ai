"""Testing battery interpretation (transformation Phase C).

The battery itself (CMJ, 10m/30m sprints, yoyo) is recorded through
``save_testing_result``; this service answers "what does a result
mean?" — honestly:

- **Percentiles come from the club's own recorded distribution** of the
  same test type, not from published norms for other populations.
- Results at the tails of a thin distribution (n < 8) are flagged
  insufficient, because a percentile from 4 data points is not evidence.
- Where a test type has no recorded history at all, the result is
  returned uninterpreted rather than guessed at.
- Comparisons are against the player's own history (trend) as well as
  the squad distribution, since individual trajectories matter more
  than one-off cross-sections.
"""

from __future__ import annotations

from typing import Any

from kawkab.core.logging import get_logger

logger = get_logger(__name__)

# Test types where LOWER is better. Everything else is treated as
# higher-is-better (CMJ height, yoyo distance...).
_LOWER_IS_BETTER = {"sprint_10m", "sprint_30m", "sprint_40m", "time_trial", "heart_rate_recovery"}


def _r1(v: float) -> float:
    return round(v, 1)


class TestingBatteryService:
    """Interpret physical testing results against the club's own data."""

    def __init__(self, storage_service: Any) -> None:
        self._storage = storage_service

    @staticmethod
    def _percentile(value: float, values: list[float], lower_is_better: bool) -> float:
        """Share of the distribution the value beats (0-100)."""
        if lower_is_better:
            beats = sum(1 for v in values if v > value)
        else:
            beats = sum(1 for v in values if v < value)
        return 100.0 * beats / len(values)

    async def interpret_result(
        self,
        player_id: int,
        test_type: str,
        value: float,
        test_date: str | None = None,
    ) -> dict[str, Any]:
        """Interpret one result: squad percentile + personal trend.

        Returns a dict that always tells the truth about how much
        evidence stands behind the interpretation.
        """
        lower_is_better = test_type in _LOWER_IS_BETTER
        out: dict[str, Any] = {
            "player_id": player_id,
            "test_type": test_type,
            "value": value,
            "direction": "lower_is_better" if lower_is_better else "higher_is_better",
            "squad_percentile": None,
            "distribution_size": 0,
            "interpretation": "no_data",
            "note": "",
            "provenance": {"normative": False, "source": "club_recorded_results"},
        }

        # Personal trend first: a player's own trajectory is evidence
        # independent of the squad distribution, and must survive the
        # thin-distribution early return below.
        try:
            history = await self._storage.get_player_testing_history(player_id, test_type)
        except Exception as e:
            logger.warning(f"testing history unavailable for player {player_id}: {e}")
            history = []
        prior = [
            float(h["value"])
            for h in history
            if h.get("value") is not None
            and (test_date is None or str(h.get("test_date")) < str(test_date))
        ]
        if prior:
            last = prior[0]  # history is ordered desc
            if last != 0:
                delta_pct = 100.0 * (value - last) / abs(last)
                improved = delta_pct > 0 if not lower_is_better else delta_pct < 0
                out["personal_trend"] = {
                    "previous_value": _r1(last),
                    "delta_pct": _r1(delta_pct),
                    "direction": "improved" if improved else "declined",
                }

        # Squad distribution for this test type (club-recorded only)
        try:
            dist = await self._storage.get_testing_distribution(test_type)
        except Exception as e:
            logger.warning(f"testing distribution unavailable for {test_type}: {e}")
            dist = []
            out["note"] = "distribution lookup failed; result stored uninterpreted"
        values = [float(d["value"]) for d in dist if d.get("value") is not None]

        if len(values) < 8:
            out["distribution_size"] = len(values)
            out["interpretation"] = "insufficient_distribution"
            if not out["note"]:
                out["note"] = (
                    f"only {len(values)} recorded result(s) for {test_type}; "
                    "a percentile from this is not evidence — record more testing"
                )
            return out

        out["distribution_size"] = len(values)
        out["squad_percentile"] = _r1(self._percentile(value, values, lower_is_better))
        out["interpretation"] = "interpreted"
        out["note"] = (
            f"percentile against {len(values)} club-recorded {test_type} results — "
            "descriptive of this squad, not a norm for other populations"
        )
        return out

    async def battery_summary(self, player_id: int) -> dict[str, Any]:
        """All recorded test types for a player with latest-vs-trend view."""
        try:
            history = await self._storage.get_player_testing_history(player_id)
        except Exception as e:
            logger.warning(f"testing history unavailable for player {player_id}: {e}")
            history = []
        by_type: dict[str, list[dict[str, Any]]] = {}
        for row in history:
            by_type.setdefault(str(row.get("test_type")), []).append(row)
        tests = []
        for ttype, rows in sorted(by_type.items()):
            values = [float(r["value"]) for r in rows if r.get("value") is not None]
            lower_better = ttype in _LOWER_IS_BETTER
            trend = "no_trend"
            if len(values) >= 2:
                delta = 100.0 * (values[0] - values[1]) / abs(values[1]) if values[1] else 0.0
                improved = delta > 0 if not lower_better else delta < 0
                trend = "improved" if improved else "declined"
            tests.append(
                {
                    "test_type": ttype,
                    "latest": values[0] if values else None,
                    "recordings": len(values),
                    "trend": trend,
                }
            )
        return {
            "player_id": player_id,
            "tests": tests,
            "provenance": {"source": "club_recorded_results"},
        }
