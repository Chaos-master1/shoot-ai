"""STATSports Sonra GPX parser.

STATSports Apex / Apex Athlete Series devices export position data as GPX 1.1
files (https://www.topografix.com/GPX/1/1/), with vendor extensions carrying
the non-GPS channels (speed, heart rate). Sonra's primary export format is
CSV/XML (see :mod:`kawkab.services.wearables.statsports_csv`), but GPX is
supported as a portable interchange.

GPX structure::

    <gpx>
      <trk>
        <trkseg>
          <trkpt lat="..." lon="...">
            <ele>...</ele>
            <time>2026-07-05T10:00:00Z</time>
            <extensions>
              <gpxtpx:TrackPointExtension>
                <gpxtpx:hr>145</gpxtpx:hr>
              </gpxtpx:TrackPointExtension>
              <speed-extension:speed>5.4</speed-extension:speed>
            </extensions>
          </trkpt>
          ...
        </trkseg>
      </trk>
    </gpx>

We resolve namespaced children defensively (with and without the namespace
prefix) because Sonra's namespace URIs have drifted across firmware versions.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime

from kawkab.core.logging import get_logger
from kawkab.services.wearables.base import BaseWearableParser
from kawkab.services.wearables.models import WearableDataPoint, WearableSession

logger = get_logger(__name__)

_GPX_NS = "http://www.topografix.com/GPX/1/1"
_TRACKPOINT_EXT_NS = "http://www.garmin.com/xmlschemas/TrackPointExtension/v1"
_SPEED_EXT_NS = "http://www.garmin.com/xmlschemas/SpeedExtension/v1"


class StatsportsGpxParser(BaseWearableParser):
    """Parses STATSports Sonra GPX exports into a WearableSession."""

    device_type = "statsports"
    supported_extensions = (".gpx",)

    def parse(self, file_path: str) -> WearableSession:
        path = self._check_file(file_path)
        session = WearableSession(device_type=self.device_type)

        tree = ET.parse(str(path))
        root = tree.getroot()

        for trkpt in root.iter(f"{{{_GPX_NS}}}trkpt"):
            dp = self._trkpt_to_datapoint(trkpt)
            if dp is not None:
                if session.start_time is None and dp.extras.get("_dt"):
                    session.start_time = dp.extras.pop("_dt")
                elif "_dt" in dp.extras:
                    dp.extras.pop("_dt", None)
                session.data.append(dp)

        if session.data:
            session.metadata["row_count"] = len(session.data)
            session.metadata["source_format"] = "gpx"
            session.finalize()

        logger.info(
            f"StatsportsGpxParser: parsed {len(session.data)} points "
            f"({session.sample_rate_hz} Hz, {session.duration_s:.0f}s) from {path.name}"
        )
        return session

    # -- internals ---------------------------------------------------------

    def _trkpt_to_datapoint(self, trkpt: ET.Element) -> WearableDataPoint | None:
        lat = self._parse_float(trkpt.get("lat"))
        lon = self._parse_float(trkpt.get("lon"))
        if lat is None and lon is None:
            return None

        dp = WearableDataPoint(latitude=lat, longitude=lon)

        # elevation
        ele = self._find(trkpt, "ele")
        if ele is not None and ele.text:
            dp.altitude_m = self._parse_float(ele.text)

        # time → timestamp_s + session start capture
        time_el = self._find(trkpt, "time")
        if time_el is not None and time_el.text:
            dt = self._parse_dt(time_el.text)
            if dt is not None:
                dp.timestamp_s = dt.timestamp()
                dp.extras["_dt"] = dt  # lifted into session.start_time by parse()

        # extensions (HR, speed). HR lives under TrackPointExtension; speed
        # may live directly under extensions OR under TrackPointExtension.
        # Use recursive iter so we find them regardless of nesting depth.
        ext = self._find(trkpt, "extensions")
        if ext is not None:
            hr = self._find_recursive_prefixed(ext, "hr", [_TRACKPOINT_EXT_NS])
            if hr is not None and hr.text:
                dp.heart_rate_bpm = self._parse_float(hr.text)
            spd = self._find_recursive_prefixed(ext, "speed", [_SPEED_EXT_NS, _TRACKPOINT_EXT_NS])
            if spd is not None and spd.text:
                dp.speed_ms = self._parse_float(spd.text)

        return dp

    @staticmethod
    def _find(parent: ET.Element, tag: str) -> ET.Element | None:
        """Find a child by tag, trying both bare and namespaced forms."""
        # Bare (no namespace) — common in Sonra exports
        el = parent.find(tag)
        if el is not None:
            return el
        # Any namespace
        return parent.find(f"{{*}}{tag}")

    @staticmethod
    def _find_prefixed(parent: ET.Element, tag: str, namespaces: list[str]) -> ET.Element | None:
        for ns in namespaces:
            el = parent.find(f"{{{ns}}}{tag}")
            if el is not None:
                return el
        # bare fallback
        return parent.find(tag)

    @staticmethod
    def _find_recursive_prefixed(
        parent: ET.Element, tag: str, namespaces: list[str]
    ) -> ET.Element | None:
        """Recursively search for ``tag`` under ``parent`` across namespaces.

        Handles elements nested at arbitrary depth (e.g. ``gpxtpx:hr`` living
        inside ``gpxtpx:TrackPointExtension`` inside ``extensions``).
        """
        # Try each namespace recursively via iter()
        for ns in namespaces:
            for el in parent.iter(f"{{{ns}}}{tag}"):
                return el
        # Bare tag, recursive
        for el in parent.iter(tag):
            return el
        return None

    @staticmethod
    def _parse_dt(text: str) -> datetime | None:
        text = text.strip()
        if not text:
            return None
        try:
            # ISO 8601 with trailing Z (GPX spec)
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            pass
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None


__all__ = ["StatsportsGpxParser"]
