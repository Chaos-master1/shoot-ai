"""FIT binary file parser.

Flexible and Interoperable Data Transport (FIT) is Garmin / Catapult / Polar's
binary format for wearable sensor data. This parser uses the ``fitdecode``
library (pure Python, no compiled deps) to decode FIT files and extract the
channels relevant to football analytics.

Key FIT messages:
- **record** — one per sample (1 Hz for GPS watches, 10–100 Hz for dedicated
  GPS vests). Carries timestamp, position, speed, heart rate, cadence, distance,
  power, altitude.
- **session** — per-session summary. Carries total distance, total time,
  calories, avg/max HR, avg/max speed. Useful for quick session-level stats.
- **device_info** — device serial number, manufacturer, product name.
- **file_id** — time created, manufacturer, product, serial number.

The parser extracts **record** messages into ``WearableDataPoint`` objects,
captures **session** and **device_info** metadata, and delegates to ``fitdecode``
for the heavy lifting (CRC validation, field definitions, timestamp decoding).

Reference: FIT SDK https://developer.garmin.com/fit/protocol/
"""

from __future__ import annotations

from datetime import datetime

from kawkab.core.logging import get_logger
from kawkab.services.wearables.base import BaseWearableParser
from kawkab.services.wearables.models import WearableDataPoint, WearableSession

logger = get_logger(__name__)

# FIT field names we care about, mapped to our unified field names.
# The FIT spec uses fixed global field numbers; fitdecode resolves them
# to descriptive names for us.
_FIT_RECORD_FIELDS: dict[str, str] = {
    "timestamp": "timestamp",
    "position_lat": "latitude",
    "position_long": "longitude",
    "enhanced_altitude": "altitude",
    "altitude": "altitude",  # fallback if enhanced not present
    "heart_rate": "heart_rate_bpm",
    "speed": "speed_ms",
    "distance": "distance_m",
    "cadence": "cadence_rpm",
    "power": "power_w",
    "accumulated_power": "power_w",  # some devices use this instead
}

_FIT_SESSION_FIELDS: dict[str, str] = {
    "total_distance": "total_distance_m",
    "total_timer_time": "duration_s",
    "avg_heart_rate": "avg_hr",
    "max_heart_rate": "max_hr",
    "avg_speed": "avg_speed_ms",
    "max_speed": "max_speed_ms",
    "total_calories": "total_calories",
}

_FIT_DEVICE_FIELDS: dict[str, str] = {
    "serial_number": "serial",
    "manufacturer": "manufacturer",
    "product": "product_name",
}


class FitParser(BaseWearableParser):
    """Parses FIT (.fit) files into a WearableSession using fitdecode."""

    device_type = "fit"
    supported_extensions = (".fit",)

    def parse(self, file_path: str) -> WearableSession:
        self._check_file(file_path)
        session = WearableSession(device_type=self.device_type)

        try:
            import fitdecode  # type: ignore[import-untyped]
        except ImportError:
            logger.error("fitdecode not installed. Install with: pip install fitdecode")
            raise

        session_start_dt: datetime | None = None

        with fitdecode.FitReader(file_path) as fit:
            for msg in fit:
                if not isinstance(msg, fitdecode.FitDataMessage):
                    continue

                msg_name = msg.name

                # -- per-sample record messages --
                if msg_name == "record":
                    dp = self._record_to_datapoint(msg)
                    if dp is not None:
                        # Capture session start from the first record's timestamp
                        if session_start_dt is None and dp.extras.get("_dt"):
                            session_start_dt = dp.extras.pop("_dt")
                        elif "_dt" in dp.extras:
                            dp.extras.pop("_dt", None)
                        session.data.append(dp)

                # -- session summary --
                elif msg_name == "session":
                    self._apply_session_metadata(session, msg)

                # -- device info --
                elif msg_name == "device_info":
                    self._apply_device_metadata(session, msg)

                # -- file_id (fallback for serial) --
                elif msg_name == "file_id":
                    self._apply_fileid_metadata(session, msg)

        if session_start_dt is not None:
            session.start_time = session_start_dt

        if session.data:
            session.metadata["row_count"] = len(session.data)
            session.metadata["source_format"] = "fit"
            session.finalize()

        logger.info(
            f"FitParser: parsed {len(session.data)} points "
            f"({session.sample_rate_hz} Hz, {session.duration_s:.0f}s) from {file_path}"
        )
        return session

    # -- internals ---------------------------------------------------------

    def _record_to_datapoint(self, msg) -> WearableDataPoint | None:  # noqa: ANN001
        """Convert a FIT record message to a WearableDataPoint."""
        ts_val = self._get_value(msg, "timestamp")
        if ts_val is None:
            return None

        # fitdecode returns timestamps as datetime objects (UTC-aware)
        timestamp_s: float
        dt: datetime | None = None
        if isinstance(ts_val, datetime):
            dt = ts_val
            timestamp_s = dt.timestamp()
        elif isinstance(ts_val, (int, float)):
            timestamp_s = float(ts_val)
        else:
            return None

        dp = WearableDataPoint(timestamp_s=timestamp_s)
        if dt:
            dp.extras["_dt"] = dt

        # Position: FIT stores lat/lon as semicircles (int32). fitdecode
        # converts to degrees for us.
        lat = self._get_value(msg, "position_lat")
        lon = self._get_value(msg, "position_long")
        if lat is not None:
            dp.latitude = float(lat)
        if lon is not None:
            dp.longitude = float(lon)

        dp.altitude_m = self._get_float(msg, "enhanced_altitude") or self._get_float(
            msg, "altitude"
        )
        dp.heart_rate_bpm = self._get_float(msg, "heart_rate")
        dp.speed_ms = self._get_float(msg, "speed")
        dp.distance_m = self._get_float(msg, "distance")
        dp.cadence_rpm = self._get_float(msg, "cadence")
        dp.power_w = self._get_float(msg, "power")

        # Accumulated power (some devices use this instead of instantaneous)
        if dp.power_w is None:
            dp.power_w = self._get_float(msg, "accumulated_power")

        return dp

    @staticmethod
    def _apply_session_metadata(session: WearableSession, msg) -> None:  # noqa: ANN001
        """Apply session summary fields to the session metadata dict."""
        for fit_field, meta_key in _FIT_SESSION_FIELDS.items():
            val = msg.get_value(fit_field)
            if val is not None:
                session.metadata[meta_key] = (
                    float(val) if isinstance(val, (int, float)) else str(val)
                )

        # Also set start_time from session timestamp if not already captured
        if session.start_time is None:
            ts = msg.get_value("start_time")
            if isinstance(ts, datetime):
                session.start_time = ts

    @staticmethod
    def _apply_device_metadata(session: WearableSession, msg) -> None:  # noqa: ANN001
        if not session.device_serial:
            sn = msg.get_value("serial_number")
            if sn is not None:
                session.device_serial = str(sn)
        if not session.metadata.get("manufacturer"):
            mfr = msg.get_value("manufacturer")
            if mfr is not None:
                session.metadata["manufacturer"] = str(mfr)

    @staticmethod
    def _apply_fileid_metadata(session: WearableSession, msg) -> None:  # noqa: ANN001
        if not session.device_serial:
            sn = msg.get_value("serial_number")
            if sn is not None:
                session.device_serial = str(sn)
        # Athlete name from "file_creator" if available
        if not session.athlete_name:
            creator = msg.get_value("file_creator")
            if creator is not None:
                session.athlete_name = str(creator)

    # -- helpers for fitdecode's get_value --------------------------------

    @staticmethod
    def _get_value(msg, field_name: str):  # noqa: ANN001
        """Get a raw value from a FIT message field, returning None on missing."""
        try:
            if msg.has_field(field_name):
                return msg.get_value(field_name)
        except Exception:
            pass
        return None

    @staticmethod
    def _get_float(msg, field_name: str) -> float | None:  # noqa: ANN001
        val = FitParser._get_value(msg, field_name)
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None


__all__ = ["FitParser"]
