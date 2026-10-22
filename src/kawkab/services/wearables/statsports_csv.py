"""STATSports Sonra CSV aggregate parser.

STATSports Sonra exports two kinds of data:

1. **Raw 10 Hz sensor data** — GPX or binary (not this parser).
2. **Aggregate CSV** — one row per athlete × session (or per drill), with
   summary metrics. This is the format coaches actually export from Sonra.

Typical Sonra CSV columns (customisable in Sonra, but these are the defaults):

    ``Player Name, Position, Team, Period, Session Type,
    Total Distance (m), High Speed Running (m), Sprint Distance (m),
    Max Speed (m/s), High Speed Distance >5.5m/s (m), High Intensity Distance (m),
    Accelerations, Decelerations, Total Accels + Decels,
    Explosive Runs, HR Avg, HR Max,
    Impacts (2G+), Impacts (4G+), Impacts (6G+),
    Metres Per Minute, PlayerLoad, Body Load``

Key speed bands (per STATSports docs):
- Walking: 0–3 m/s
- Jogging: 3–5 m/s
- Running: 5–5.5 m/s
- High Speed Running (HSR): 5.5–7 m/s
- Sprint: >7 m/s

This parser extracts one ``WearableDataPoint`` per row where the aggregate
metrics are stored in ``extras``. It also produces a separate "aggregate
summary" in ``session.metadata`` for quick dashboard rendering.

Reference: https://elitesupport.statsports.com/hc/en-us/articles/13494616698141-Export-Data
"""

from __future__ import annotations

import csv

from kawkab.core.logging import get_logger
from kawkab.services.wearables.base import BaseWearableParser
from kawkab.services.wearables.models import WearableDataPoint, WearableSession

logger = get_logger(__name__)

# Aliases per logical field, lowercase. Sonra allows custom column names
# but the defaults below cover the standard export template.
_SONRA_ALIASES: dict[str, tuple[str, ...]] = {
    "player_name": ("player name", "player", "athlete", "name"),
    "position": ("position", "player position"),
    "team": ("team", "squad"),
    "period": ("period", "drill", "session period"),
    "session_type": ("session type", "activity"),
    "total_distance": ("total distance (m)", "total distance", "distance"),
    "hsr_distance": (
        "high speed running (m)",
        "high speed running",
        "hsr (m)",
        "hsr",
        "high speed distance >5.5m/s (m)",
    ),
    "sprint_distance": ("sprint distance (m)", "sprint distance", "sprint (m)"),
    "max_speed": ("max speed (m/s)", "max speed", "top speed"),
    "high_intensity_distance": (
        "high intensity distance (m)",
        "high intensity distance",
        "hid",
    ),
    "accelerations": ("accelerations", "accels", "accelerations (>3 m/s²)"),
    "decelerations": ("decelerations", "decs", "decelerations (<-3 m/s²)"),
    "total_accels_decs": ("total accels + decels", "total accelerations + decelerations"),
    "explosive_runs": ("explosive runs", "explosive sprints"),
    "hr_avg": ("hr avg", "average heart rate", "avg hr", "avg heart rate (bpm)"),
    "hr_max": ("hr max", "max heart rate", "max hr", "max heart rate (bpm)"),
    "impacts_2g": ("impacts (2g+)", "impacts 2g+"),
    "impacts_4g": ("impacts (4g+)", "impacts 4g+"),
    "impacts_6g": ("impacts (6g+)", "impacts 6g+"),
    "metres_per_minute": ("metres per minute", "m/min", "distance per min"),
    "playerload": ("playerload", "player load", "body load"),
}


class StatsportsCsvParser(BaseWearableParser):
    """Parses STATSports Sonra aggregate CSV exports.

    Each row becomes one ``WearableDataPoint`` with aggregate metrics in
    ``extras``. The timestamp is derived from the row index (1-indexed, at
    1-second intervals) because Sonra aggregate CSVs don't carry per-second
    timestamps — they carry summary totals.

    For multi-row files (one per drill), timestamp represents the drill's
    ordinal position within the session.
    """

    device_type = "statsports"
    supported_extensions = (".csv",)

    def parse(self, file_path: str) -> WearableSession:
        path = self._check_file(file_path)
        session = WearableSession(device_type=self.device_type)

        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            raw_headers = list(reader.fieldnames or [])
            lowered = [h.lower().strip() for h in raw_headers]

            col = self._resolve_columns(lowered, raw_headers)

            for idx, row in enumerate(reader):
                dp = self._row_to_datapoint(row, col, idx)
                if dp is not None:
                    session.data.append(dp)
                    # Capture athlete name from first row
                    if not session.athlete_name and col.get("player_name"):
                        val = row.get(col["player_name"])
                        if val:
                            session.athlete_name = val.strip()

        if session.data:
            session.metadata["is_sonra_aggregate"] = True
            session.metadata["row_count"] = len(session.data)
            session.metadata["source_columns"] = lowered
            # Copy first row's aggregate values into session metadata for
            # quick dashboard rendering (single-row session summary).
            if session.data[0].extras:
                for k, v in session.data[0].extras.items():
                    if k.startswith("sonra_"):
                        session.metadata[k] = v
            session.finalize()

        logger.info(
            f"StatsportsCsvParser: parsed {len(session.data)} aggregate rows from {path.name}"
        )
        return session

    # -- internals ---------------------------------------------------------

    def _row_to_datapoint(
        self, row: dict, col: dict[str, str | None], row_idx: int
    ) -> WearableDataPoint | None:
        # Timestamp: aggregate CSVs have no per-second timestamps. Use row
        # index as a synthetic timestamp (1-indexed, 1s intervals).
        dp = WearableDataPoint(timestamp_s=float(row_idx))

        # Map every Sonra column to extras with "sonra_" prefix
        for field_name, _aliases in _SONRA_ALIASES.items():
            header = col.get(field_name)
            if header is None:
                continue
            val = row.get(header)
            if val is None or val.strip() == "":
                continue
            float_val = self._parse_float(val.strip())
            if float_val is not None:
                dp.extras[f"sonra_{field_name}"] = float_val
            else:
                dp.extras[f"sonra_{field_name}"] = val.strip()

        # Also store total distance in the main field for to_dict compatibility
        td = (
            self._parse_float(row.get(col["total_distance"])) if col.get("total_distance") else None
        )
        if td is not None:
            dp.distance_m = td

        # Max speed in the main field
        ms = self._parse_float(row.get(col["max_speed"])) if col.get("max_speed") else None
        if ms is not None:
            dp.speed_ms = ms

        return dp

    @staticmethod
    def _resolve_columns(lowered: list[str], raw_headers: list[str]) -> dict[str, str | None]:
        col: dict[str, str | None] = {}
        for field_name, aliases in _SONRA_ALIASES.items():
            for alias in aliases:
                if alias in lowered:
                    col[field_name] = raw_headers[lowered.index(alias)]
                    break
            else:
                col[field_name] = None
        return col


__all__ = ["StatsportsCsvParser"]
