"""Isokinetic strength data import and management service.

Stores and analyses strength test results including NordBoard, ForceFrame,
isokinetic dynamometry, and manual entries. Supports limb symmetry index
(LSI) computation and normative reference lookups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StrengthTestRecord:
    date: str
    player_id: str
    test_type: str = "manual"
    metric_name: str = ""
    metric_value: float = 0.0
    unit: str = "N"
    limb: str = "both"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "player_id": self.player_id,
            "test_type": self.test_type,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "unit": self.unit,
            "limb": self.limb,
            "notes": self.notes,
        }


# Normative reference values (position, test_type) -> (mean, std, unit)
# Based on published literature for professional footballers.
_STRENGTH_NORMS: dict[tuple[str, str], tuple[float, float, str]] = {
    ("defender", "nordboard"): (425, 55, "N"),
    ("defender", "isokinetic_60"): (2.8, 0.4, "Nm/kg"),
    ("defender", "isokinetic_300"): (1.6, 0.3, "Nm/kg"),
    ("midfielder", "nordboard"): (400, 50, "N"),
    ("midfielder", "isokinetic_60"): (2.6, 0.4, "Nm/kg"),
    ("midfielder", "isokinetic_300"): (1.5, 0.3, "Nm/kg"),
    ("forward", "nordboard"): (410, 52, "N"),
    ("forward", "isokinetic_60"): (2.7, 0.4, "Nm/kg"),
    ("forward", "isokinetic_300"): (1.55, 0.3, "Nm/kg"),
    ("goalkeeper", "nordboard"): (440, 60, "N"),
    ("goalkeeper", "isokinetic_60"): (2.9, 0.5, "Nm/kg"),
    ("goalkeeper", "isokinetic_300"): (1.7, 0.35, "Nm/kg"),
}


class StrengthDataService:
    def __init__(self) -> None:
        self._records: list[StrengthTestRecord] = []

    def record_test(self, test_data: StrengthTestRecord) -> int:
        """Save a strength test result. Returns record index (ID)."""
        self._records.append(test_data)
        idx = len(self._records) - 1
        logger.info(
            f"Strength test recorded: player={test_data.player_id} "
            f"test={test_data.test_type} metric={test_data.metric_name} "
            f"value={test_data.metric_value} {test_data.unit}"
        )
        return idx

    def get_player_history(self, player_id: str, limit: int = 20) -> list[StrengthTestRecord]:
        """Return recent test records for a player, newest first."""
        player_records = [r for r in self._records if r.player_id == player_id]
        sorted_records = sorted(
            player_records,
            key=lambda r: r.date,
            reverse=True,
        )
        return sorted_records[:limit]

    def get_limb_symmetry_index(self, player_id: str) -> dict[str, Any] | None:
        """Compute limb symmetry index (LSI) for the most recent bilateral test.

        LSI = (weaker / stronger) × 100. Returns None when no bilateral test
        with left/right values is available.
        """
        bilateral = [
            r for r in self._records if r.player_id == player_id and r.limb in ("left", "right")
        ]
        if not bilateral:
            return None

        # Group by metric_name + test_type, pick most recent pair
        from collections import defaultdict

        groups: dict[tuple[str, str], dict[str, StrengthTestRecord]] = defaultdict(dict)
        for r in bilateral:
            key = (r.metric_name, r.test_type)
            existing = groups[key].get(r.limb)
            if existing is None or r.date > existing.date:
                groups[key][r.limb] = r

        results: list[dict[str, Any]] = []
        for (metric, test_type), limbs in groups.items():
            left = limbs.get("left")
            right = limbs.get("right")
            if left is None or right is None:
                continue
            stronger = max(left.metric_value, right.metric_value)
            weaker = min(left.metric_value, right.metric_value)
            lsi = (weaker / stronger * 100) if stronger > 0 else 100.0
            results.append(
                {
                    "metric_name": metric,
                    "test_type": test_type,
                    "left_value": left.metric_value,
                    "left_unit": left.unit,
                    "right_value": right.metric_value,
                    "right_unit": right.unit,
                    "lsi_pct": round(lsi, 1),
                    "date": max(left.date, right.date),
                }
            )

        if not results:
            return None

        return {
            "player_id": player_id,
            "results": sorted(results, key=lambda r: r["date"], reverse=True),
        }

    def get_strength_norms(self, position: str, test_type: str) -> dict[str, Any] | None:
        """Return normative reference values for a position + test type.

        Returns dict with ``mean``, ``std``, ``unit``, or None if unknown.
        """
        key = (position.lower(), test_type.lower())
        norm = _STRENGTH_NORMS.get(key)
        if norm is None:
            logger.debug(f"No strength norms for position={position}, test_type={test_type}")
            return None
        mean, std, unit = norm
        return {
            "position": position,
            "test_type": test_type,
            "mean": mean,
            "std": std,
            "unit": unit,
        }

    def get_all(self) -> list[StrengthTestRecord]:
        """Return all stored records (useful for testing / inspection)."""
        return list(self._records)

    def clear(self) -> None:
        """Clear all in-memory records (useful for testing)."""
        self._records.clear()
