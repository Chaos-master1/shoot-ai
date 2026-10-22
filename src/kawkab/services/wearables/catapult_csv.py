"""Catapult OpenField CSV parser.

Catapult OpenField exports two flavours of CSV that we care about:

1. **10 Hz sensor export** — one row per 0.1 s sample with raw GPS + inertial
   channels. Produced via *OpenField Console → Export → Sensor Export*.
   Reference: https://support.catapultsports.com/hc/en-us/articles/360001427755

   Typical columns (case varies across OpenField versions):
       ``Athlete Id, Athlete, Date, Timestamp (s), Latitude, Longitude,
       Speed (m/s), Acceleration (m/s²), Heart Rate (bpm), Distance (m),
       Cadence (rpm), Body Load, Metabolic Power, ...``

2. **Catapult Training Report (CTR)** — aggregated per-period summary with
   one row per athlete × period × drill. Reference:
   https://support.catapultsports.com/hc/en-us/articles/9506893619599

This parser targets flavour **(1)** (the high-rate raw stream), because it is
the one suitable for time-sync with video/tracking and for computing
PlayerLoad / HSR / sprint distance ourselves. CTR rows are detected and
forwarded to the summary path with a metadata flag.

The parser is header-driven: it lowercases + strips every column name and
looks up each field by a list of known aliases. This makes it robust to the
column-name drift between OpenField versions (e.g. ``Speed (m/s)`` vs
``speed`` vs ``Velocity (m/s)``).
"""

from __future__ import annotations

import csv

from kawkab.core.logging import get_logger
from kawkab.services.wearables.base import BaseWearableParser
from kawkab.services.wearables.models import WearableDataPoint, WearableSession

logger = get_logger(__name__)

# Aliases per logical field, lowercase. Extend as new OpenField versions add
# or rename columns. Order matters only for readability, not for matching.
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": ("timestamp (s)", "timestamp", "time (s)", "time", "t"),
    "latitude": ("latitude", "lat"),
    "longitude": ("longitude", "lon", "long"),
    "speed": ("speed (m/s)", "speed", "velocity (m/s)", "velocity"),
    "acceleration": ("acceleration (m/s²)", "acceleration (m/s^2)", "acceleration", "acc"),
    "heart_rate": ("heart rate (bpm)", "heart rate", "hr (bpm)", "hr"),
    "distance": ("distance (m)", "distance", "odometer"),
    "cadence": ("cadence (rpm)", "cadence"),
    "altitude": ("altitude (m)", "altitude", "elevation (m)", "elevation"),
    "power": ("power (w)", "power", "metabolic power (w)", "metabolic power"),
    "body_load": ("body load", "playerload", "player load"),
    "athlete_id": ("athlete id", "athlete_id", "player id"),
    "athlete_name": ("athlete", "athlete name", "player name", "player"),
    "device_serial": ("device id", "serial", "unit id", "device serial"),
}

# Heuristic threshold for distinguishing 10Hz sensor export from a CTR.
# CTRs have < 1000 rows for a normal session (one per period); sensor exports
# have tens of thousands (one per 0.1s).
_SENSOR_EXPORT_MIN_ROWS = 500


class CatapultCsvParser(BaseWearableParser):
    """Parses Catapult OpenField sensor-export CSV into a WearableSession."""

    device_type = "catapult"
    supported_extensions = (".csv",)

    def parse(self, file_path: str) -> WearableSession:
        path = self._check_file(file_path)
        session = WearableSession(device_type=self.device_type)

        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            raw_headers = list(reader.fieldnames or [])
            # Lowercased for alias matching, but we look rows up by the ORIGINAL
            # header case (that's what csv.DictReader uses as keys).
            lowered = [h.lower().strip() for h in raw_headers]

            # Build a lookup: logical field → ORIGINAL header (or None)
            col = self._resolve_columns(lowered, raw_headers)

            if not col.get("timestamp"):
                raise ValueError(
                    "Catapult CSV missing a timestamp column "
                    f"(looked for {self._aliases('timestamp')}); headers={lowered}"
                )

            for row in reader:
                dp = self._row_to_datapoint(row, col)
                if dp is None:
                    continue
                session.data.append(dp)
                # Capture athlete/device metadata from the first row that has it.
                if not session.athlete_id and col.get("athlete_id"):
                    val = row.get(col["athlete_id"])
                    if val:
                        session.athlete_id = str(val).strip()
                if not session.athlete_name and col.get("athlete_name"):
                    val = row.get(col["athlete_name"])
                    if val:
                        session.athlete_name = str(val).strip()
                if not session.device_serial and col.get("device_serial"):
                    val = row.get(col["device_serial"])
                    if val:
                        session.device_serial = str(val).strip()

        if session.data:
            session.metadata["is_sensor_export"] = len(session.data) >= _SENSOR_EXPORT_MIN_ROWS
            session.metadata["row_count"] = len(session.data)
            session.metadata["source_columns"] = lowered
            session.finalize()

        logger.info(
            f"CatapultCsvParser: parsed {len(session.data)} points "
            f"({session.sample_rate_hz} Hz, {session.duration_s:.0f}s) from {path.name}"
        )
        return session

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _aliases(field: str) -> tuple[str, ...]:
        return _FIELD_ALIASES[field]

    def _resolve_columns(self, lowered: list[str], raw_headers: list[str]) -> dict[str, str | None]:
        """Return {logical_field: ORIGINAL_header_or_None} for the given file.

        ``lowered`` is the lowercase form used for alias matching; ``raw_headers``
        is the original-case form csv.DictReader will use as row keys.
        """
        col: dict[str, str | None] = {}
        for field, aliases in _FIELD_ALIASES.items():
            for alias in aliases:
                if alias in lowered:
                    col[field] = raw_headers[lowered.index(alias)]
                    break
            else:
                col[field] = None
        return col

    def _row_to_datapoint(self, row: dict, col: dict[str, str | None]) -> WearableDataPoint | None:
        ts_raw = self._cell(row, col.get("timestamp"))
        if ts_raw is None:
            return None
        ts = self._parse_float(ts_raw)
        if ts is None:
            return None

        dp = WearableDataPoint(timestamp_s=ts)
        dp.latitude = self._parse_float(self._cell(row, col.get("latitude")))
        dp.longitude = self._parse_float(self._cell(row, col.get("longitude")))
        dp.speed_ms = self._parse_float(self._cell(row, col.get("speed")))
        dp.acceleration_ms2 = self._parse_float(self._cell(row, col.get("acceleration")))
        dp.heart_rate_bpm = self._parse_float(self._cell(row, col.get("heart_rate")))
        dp.distance_m = self._parse_float(self._cell(row, col.get("distance")))
        dp.cadence_rpm = self._parse_float(self._cell(row, col.get("cadence")))
        dp.altitude_m = self._parse_float(self._cell(row, col.get("altitude")))
        dp.power_w = self._parse_float(self._cell(row, col.get("power")))

        body_load = self._parse_float(self._cell(row, col.get("body_load")))
        if body_load is not None:
            dp.extras["body_load"] = body_load

        return dp

    @staticmethod
    def _cell(row: dict, key: str | None):
        if key is None:
            return None
        return row.get(key)


__all__ = ["CatapultCsvParser"]
