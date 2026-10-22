"""Sleep and recovery data import/integration service.

Provides sleep record management from Oura Ring, WHOOP band, and manual entry,
plus composite recovery scoring based on sleep quality, HRV, and training load.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SleepRecord:
    date: str
    bedtime: str = ""
    wake_time: str = ""
    total_sleep_h: float = 0.0
    deep_sleep_pct: float = 0.0
    rem_sleep_pct: float = 0.0
    hrv_rmssd: float | None = None
    resting_hr: float | None = None
    source: str = "manual"


class SleepRecoveryService:
    def __init__(self) -> None:
        self._records: dict[str, list[SleepRecord]] = {}  # player_id -> records

    def import_oura_json(self, path: str | Path) -> list[SleepRecord]:
        """Parse Oura Ring export JSON (v2 API format).

        Expects a list of objects with ``sleep`` entries containing
        ``bedtime_start``, ``bedtime_end``, ``total_sleep_duration``,
        ``deep_sleep_duration``, ``rem_sleep_duration``, ``hrv_rmssd``,
        and ``resting_heart_rate``.
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"Oura JSON not found: {path}")
            return []

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to parse Oura JSON {path}: {e}")
            return []

        if isinstance(data, dict):
            data = data.get("sleep", [])

        records: list[SleepRecord] = []
        for entry in data:
            try:
                total_ms = entry.get("total_sleep_duration", 0) or 0
                deep_ms = entry.get("deep_sleep_duration", 0) or 0
                rem_ms = entry.get("rem_sleep_duration", 0) or 0
                total_h = total_ms / 3_600_000 if total_ms else 0.0
                deep_pct = (deep_ms / total_ms * 100) if total_ms else 0.0
                rem_pct = (rem_ms / total_ms * 100) if total_ms else 0.0

                sleep_date = entry.get("day", "")
                if not sleep_date:
                    sleep_date = entry.get("bedtime_start", "")[:10]

                records.append(
                    SleepRecord(
                        date=sleep_date,
                        bedtime=entry.get("bedtime_start", ""),
                        wake_time=entry.get("bedtime_end", ""),
                        total_sleep_h=round(total_h, 1),
                        deep_sleep_pct=round(deep_pct, 1),
                        rem_sleep_pct=round(rem_pct, 1),
                        hrv_rmssd=entry.get("hrv_rmssd"),
                        resting_hr=entry.get("resting_heart_rate"),
                        source="oura",
                    )
                )
            except (TypeError, ValueError) as e:
                logger.warning(f"Skipping malformed Oura entry: {e}")
                continue

        logger.info(f"Imported {len(records)} sleep records from Oura JSON")
        return records

    def import_whoop_csv(self, path: str | Path) -> list[SleepRecord]:
        """Parse WHOOP band export CSV.

        Expected columns (case-insensitive): date, bedtime, wake_time,
        total_sleep_h, deep_sleep_pct, rem_sleep_pct, hrv_rmssd, resting_hr.
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"WHOOP CSV not found: {path}")
            return []

        records: list[SleepRecord] = []
        try:
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    return []
                col_map = {h.strip().lower(): h for h in reader.fieldnames}
                for row in reader:
                    try:
                        # `row` bound as a default: consumed within this
                        # iteration, but default-arg binding removes the
                        # late-binding trap (B023).
                        def _val(key: str, cast: Callable[..., Any] = str, row=row) -> Any:
                            raw = row.get(col_map.get(key, ""), "").strip()
                            if not raw:
                                return None
                            try:
                                return cast(raw)
                            except (ValueError, TypeError):
                                return None

                        hrv = _val("hrv_rmssd", float)
                        rhr = _val("resting_hr", float)
                        date_val = _val("date", str) or ""

                        records.append(
                            SleepRecord(
                                date=date_val,
                                bedtime=_val("bedtime", str) or "",
                                wake_time=_val("wake_time", str) or "",
                                total_sleep_h=_val("total_sleep_h", float) or 0.0,
                                deep_sleep_pct=_val("deep_sleep_pct", float) or 0.0,
                                rem_sleep_pct=_val("rem_sleep_pct", float) or 0.0,
                                hrv_rmssd=hrv,
                                resting_hr=rhr,
                                source="whoop",
                            )
                        )
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Skipping malformed WHOOP row: {e}")
                        continue
        except OSError as e:
            logger.error(f"Failed to read WHOOP CSV {path}: {e}")

        logger.info(f"Imported {len(records)} sleep records from WHOOP CSV")
        return records

    def get_sleep_history(self, player_id: str, days: int = 30) -> list[SleepRecord]:
        """Return recent sleep records for a player, newest first."""
        records = self._records.get(player_id, [])
        if not records:
            return []
        cutoff = datetime.now() - timedelta(days=days)
        filtered = []
        for r in records:
            parsed_date = _parse_date(r.date) if r.date else None
            if parsed_date is not None and parsed_date >= cutoff.date():
                filtered.append(r)
        return sorted(filtered, key=lambda r: r.date, reverse=True)

    def get_recovery_score(self, player_id: str) -> int | None:
        """Composite recovery score 0-100 based on sleep + HRV.

        Factors:
          - Sleep duration (target 7-9 h)    → up to 40 pts
          - Deep sleep % (target >=20%)       → up to 20 pts
          - HRV RMSSD (higher = better)       → up to 25 pts
          - Resting HR (lower = better)       → up to 15 pts
        Returns None when no data is available.
        """
        records = self.get_sleep_history(player_id, days=7)
        if not records:
            return None

        avg_sleep = sum(r.total_sleep_h for r in records if r.total_sleep_h > 0) / max(
            sum(1 for r in records if r.total_sleep_h > 0), 1
        )
        avg_deep = sum(r.deep_sleep_pct for r in records if r.deep_sleep_pct > 0) / max(
            sum(1 for r in records if r.deep_sleep_pct > 0), 1
        )

        hrv_vals = [r.hrv_rmssd for r in records if r.hrv_rmssd is not None]
        avg_hrv = sum(hrv_vals) / len(hrv_vals) if hrv_vals else 0.0

        hr_vals = [r.resting_hr for r in records if r.resting_hr is not None]
        avg_rhr = sum(hr_vals) / len(hr_vals) if hr_vals else 60.0

        # Sleep duration score (40 pts)
        if avg_sleep >= 7.0:
            sleep_pts = 40.0
        elif avg_sleep >= 6.0:
            sleep_pts = 25.0 + (avg_sleep - 6.0) * 15.0
        elif avg_sleep >= 5.0:
            sleep_pts = 10.0 + (avg_sleep - 5.0) * 15.0
        else:
            sleep_pts = max(0.0, avg_sleep * 2.0)

        # Deep sleep score (20 pts)
        deep_pts = min(20.0, avg_deep * 1.0) if avg_deep >= 10 else max(0.0, avg_deep * 1.5)

        # HRV RMSSD score (25 pts)
        hrv_pts = min(25.0, avg_hrv * 0.5) if avg_hrv else 12.5

        # Resting HR score (15 pts) — lower is better
        if avg_rhr <= 50:
            rhr_pts = 15.0
        elif avg_rhr <= 60:
            rhr_pts = 12.0 + (60 - avg_rhr) * 0.3
        elif avg_rhr <= 70:
            rhr_pts = 6.0 + (70 - avg_rhr) * 0.6
        else:
            rhr_pts = max(0.0, 6.0 - (avg_rhr - 70) * 0.3)

        total = round(sleep_pts + deep_pts + hrv_pts + rhr_pts)
        return max(0, min(100, total))

    def generate_recommendation(self, player_id: str) -> str | None:
        """Generate a text recommendation based on current recovery status."""
        score = self.get_recovery_score(player_id)
        if score is None:
            return None

        records = self.get_sleep_history(player_id, days=3)
        if not records:
            return None

        latest = records[0]
        parts: list[str] = []

        if score >= 80:
            parts.append(
                "Recovery is excellent — player is ready for full training and match load."
            )
        elif score >= 60:
            parts.append("Moderate recovery — consider lighter training load or an extra rest day.")
        elif score >= 40:
            parts.append("Poor recovery — prioritise sleep hygiene and active recovery today.")
        else:
            parts.append("Critical recovery deficit — recommend rest day and medical assessment.")

        if latest.total_sleep_h < 6.0:
            parts.append(f"Sleep duration ({latest.total_sleep_h}h) is below the 7-9h target.")
        if latest.deep_sleep_pct < 15.0:
            parts.append("Deep sleep percentage is low; consider sleep hygiene improvements.")

        if latest.hrv_rmssd is not None and latest.hrv_rmssd < 20:
            parts.append("HRV is suppressed — may indicate accumulated fatigue or overreaching.")

        return " ".join(parts)

    def add_record(self, player_id: str, record: SleepRecord) -> None:
        """Add a sleep record for a player (in-memory)."""
        if player_id not in self._records:
            self._records[player_id] = []
        self._records[player_id].append(record)

    def clear(self) -> None:
        """Clear all in-memory records (useful for testing)."""
        self._records.clear()


def _parse_date(date_str: str) -> datetime | None:
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(date_str[:10], fmt)
        except (ValueError, TypeError):
            continue
    return None
