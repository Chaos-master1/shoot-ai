"""Polar heart-rate CSV parser.

Polar Team Pro and Polar H10/H9 chest straps export session data as CSV.
Two layouts are seen in the wild:

1. **Per-sample HR CSV** — ``Time,HR (bpm),Speed (m/s),...`` rows with a
   ``---`` separator before the data section (Polar Flow export).
2. **Plain CSV** — ``timestamp_s,hr[,speed]`` numeric rows with no header.

This parser handles both. The ``---`` separator and a leading ``Time`` header
identify layout (1); otherwise we treat it as layout (2).

Polar Team Pro's full session export (with GPS + accelerometry) is handled by
the dedicated ``polar_pro.py`` parser (cycle C5); this module is for the
portable HR-only export that any Polar device can produce.
"""

from __future__ import annotations

import csv
from datetime import datetime

from kawkab.core.logging import get_logger
from kawkab.services.wearables.base import BaseWearableParser
from kawkab.services.wearables.models import WearableDataPoint, WearableSession

logger = get_logger(__name__)


class PolarHrCsvParser(BaseWearableParser):
    """Parses Polar HR CSV exports (per-sample or plain) into a WearableSession."""

    device_type = "polar"
    supported_extensions = (".csv",)

    def parse(self, file_path: str) -> WearableSession:
        path = self._check_file(file_path)
        session = WearableSession(device_type=self.device_type)

        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        data_start, header = self._find_data_start(rows)

        # column-name → index map (lowercased) when a header row exists
        col_index: dict[str, int] = {}
        if header:
            for i, name in enumerate(header):
                col_index[name.lower().strip()] = i

        for row in rows[data_start:]:
            if not row or all(not c.strip() for c in row):
                continue
            dp = self._row_to_datapoint(row, col_index)
            if dp is not None:
                session.data.append(dp)

        if session.data:
            session.metadata["row_count"] = len(session.data)
            session.metadata["source_format"] = "polar_hr_csv"
            session.metadata["source_columns"] = header or []
            session.finalize()

        logger.info(
            f"PolarHrCsvParser: parsed {len(session.data)} points "
            f"({session.sample_rate_hz} Hz, {session.duration_s:.0f}s) from {path.name}"
        )
        return session

    # -- internals ---------------------------------------------------------

    def _find_data_start(self, rows: list[list[str]]) -> tuple[int, list[str] | None]:
        """Locate the first data row, returning (data_start_index, header_or_None)."""
        for i, row in enumerate(rows):
            if not row:
                continue
            first = (row[0] or "").strip() if row else ""
            if first == "---":
                # Polar Flow style: header before this line, data after
                header = rows[i - 1] if i > 0 else None
                return i + 1, header
            if first.lower() in ("time", "timestamp", "t"):
                return i + 1, row
        # No separator, no header → assume data from row 0
        return 0, None

    def _row_to_datapoint(
        self, row: list[str], col_index: dict[str, int]
    ) -> WearableDataPoint | None:
        if not row:
            return None

        # Resolve columns: by header name if present, else by position
        ts = self._get(row, col_index, ("time", "timestamp", "t"), 0)
        hr = self._get(row, col_index, ("hr (bpm)", "hr", "heart rate (bpm)", "heart rate"), 1)

        ts_val = self._parse_timestamp(ts)
        if ts_val is None:
            return None

        dp = WearableDataPoint(timestamp_s=ts_val)
        dp.heart_rate_bpm = self._parse_float(hr)
        # Speed: by header name if present, else positional index 2 if the row
        # has a third column (plain numeric ``ts,hr,speed`` layout).
        speed_val = self._get(row, col_index, ("speed (m/s)", "speed", "velocity"), None)
        if speed_val is None and len(row) >= 3:
            speed_val = row[2].strip()
        dp.speed_ms = self._parse_float(speed_val)
        return dp

    @staticmethod
    def _get(
        row: list[str],
        col_index: dict[str, int],
        names: tuple[str, ...],
        positional_fallback: int | None,
    ) -> str | None:
        for name in names:
            if name in col_index and col_index[name] < len(row):
                return row[col_index[name]].strip()
        if positional_fallback is not None and positional_fallback < len(row):
            return row[positional_fallback].strip()
        return None

    @staticmethod
    def _parse_timestamp(val: str | None) -> float | None:
        if not val:
            return None
        val = val.strip()
        # HH:MM:SS form (Polar Flow)
        parts = val.split(":")
        if len(parts) == 3:
            try:
                return float(int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2]))
            except ValueError:
                return None
        try:
            return float(val)
        except ValueError:
            pass
        # ISO timestamp form
        try:
            return datetime.fromisoformat(val.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None


__all__ = ["PolarHrCsvParser"]
