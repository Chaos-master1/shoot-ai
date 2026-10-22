"""Training Center XML (TCX) parser.

TCX is Garmin / Polar's XML format for GPS + sensor data. It carries per-trackpoint
position, heart rate, cadence, speed, distance, and (optionally) watts — the same
channels as FIT, but in a human-readable XML envelope.

Structure::

    <TrainingCenterDatabase>
      <Activities>
        <Activity Sport="Running">
          <Id>2026-07-05T10:00:00Z</Id>
          <Lap StartTime="...">
            <TotalDistanceMeters>...</TotalDistanceMeters>
            <TotalTimeSeconds>...</TotalTimeSeconds>
            <Track>
              <Trackpoint>
                <Time>2026-07-05T10:00:00Z</Time>
                <Position><LatitudeDegrees>53.43</LatitudeDegrees><LongitudeDegrees>-2.96</LongitudeDegrees></Position>
                <AltitudeMeters>10</AltitudeMeters>
                <DistanceMeters>5.0</DistanceMeters>
                <HeartRateBpm><Value>145</Value></HeartRateBpm>
                <Cadence>80</Cadence>
                <Extensions><TPX><Speed>5.4</Speed></TPX></Extensions>
              </Trackpoint>
            </Track>
          </Lap>
        </Activity>
      </Activities>
    </TrainingCenterDatabase>

Namespaces vary between Garmin, Polar, and third-party converters. This parser
searches both namespaced and bare forms of every child element.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime

from kawkab.core.logging import get_logger
from kawkab.services.wearables.base import BaseWearableParser
from kawkab.services.wearables.models import WearableDataPoint, WearableSession

logger = get_logger(__name__)

# Common namespace URIs seen in TCX files
_TCX_NS = "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"
_NS5_NS = "http://www.garmin.com/xmlschemas/ActivityExtension/v2"
_TPX_NS = "http://www.garmin.com/xmlschemas/ActivityExtension/v2"


class TcxParser(BaseWearableParser):
    """Parses TCX (.tcx) files into a WearableSession."""

    device_type = "tcx"
    supported_extensions = (".tcx", ".xml")

    def parse(self, file_path: str) -> WearableSession:
        path = self._check_file(file_path)
        session = WearableSession(device_type=self.device_type)

        tree = ET.parse(str(path))
        root = tree.getroot()

        # Extract sport type
        activity = self._find_any(root, "Activity")
        if activity is not None:
            sport = activity.get("Sport", "")
            session.metadata["sport"] = sport

        # Extract session start time from <Id> or first <Lap StartTime>
        activity_id = self._find_any(root, "Id")
        if activity_id is not None and activity_id.text:
            session.start_time = self._parse_dt(activity_id.text)

        # Iterate all Trackpoints across all Laps
        for tp in root.iter():
            # Match any element whose local name is "Trackpoint"
            local = self._local_tag(tp.tag)
            if local == "Trackpoint":
                dp = self._trackpoint_to_datapoint(tp)
                if dp is not None:
                    if session.start_time is None and dp.extras.get("_dt"):
                        session.start_time = dp.extras.pop("_dt")
                    elif "_dt" in dp.extras:
                        dp.extras.pop("_dt", None)
                    session.data.append(dp)

        # Lap-level summaries for metadata
        for lap in self._iter_local(root, "Lap"):
            start = lap.get("StartTime")
            if start:
                session.metadata["lap_start"] = start
            td = self._find_child_text(lap, "TotalDistanceMeters")
            if td:
                session.metadata["lap_distance_m"] = self._parse_float(td)
            tt = self._find_child_text(lap, "TotalTimeSeconds")
            if tt:
                session.metadata["lap_duration_s"] = self._parse_float(tt)

        if session.data:
            session.metadata["row_count"] = len(session.data)
            session.metadata["source_format"] = "tcx"
            session.finalize()

        logger.info(
            f"TcxParser: parsed {len(session.data)} points "
            f"({session.sample_rate_hz} Hz, {session.duration_s:.0f}s) from {path.name}"
        )
        return session

    # -- internals ---------------------------------------------------------

    def _trackpoint_to_datapoint(self, tp: ET.Element) -> WearableDataPoint | None:
        dp = WearableDataPoint()

        # timestamp
        time_el = self._find_child(tp, "Time")
        if time_el is not None and time_el.text:
            dt = self._parse_dt(time_el.text)
            if dt is None:
                return None
            dp.timestamp_s = dt.timestamp()
            dp.extras["_dt"] = dt

        # position
        pos = self._find_child(tp, "Position")
        if pos is not None:
            lat = self._find_child_text(pos, "LatitudeDegrees")
            lon = self._find_child_text(pos, "LongitudeDegrees")
            if lat is not None:
                dp.latitude = self._parse_float(lat)
            if lon is not None:
                dp.longitude = self._parse_float(lon)

        # altitude
        alt = self._find_child_text(tp, "AltitudeMeters")
        if alt is not None:
            dp.altitude_m = self._parse_float(alt)

        # distance
        dist = self._find_child_text(tp, "DistanceMeters")
        if dist is not None:
            dp.distance_m = self._parse_float(dist)

        # heart rate (nested under HeartRateBpm > Value)
        hr_el = self._find_child(tp, "HeartRateBpm")
        if hr_el is not None:
            val_el = self._find_child(hr_el, "Value")
            if val_el is not None and val_el.text:
                dp.heart_rate_bpm = self._parse_float(val_el.text)
            elif hr_el.text:
                dp.heart_rate_bpm = self._parse_float(hr_el.text)

        # cadence (direct child of Trackpoint)
        cad = self._find_child_text(tp, "Cadence")
        if cad is not None:
            dp.cadence_rpm = self._parse_float(cad)

        # speed from extensions (TPX > Speed)
        ext = self._find_child(tp, "Extensions")
        if ext is not None:
            for child in ext.iter():
                local = self._local_tag(child.tag)
                if local == "Speed" and child.text:
                    dp.speed_ms = self._parse_float(child.text)

        return dp

    # -- XML helpers (defensive namespace handling) --------------------------

    @staticmethod
    def _local_tag(tag: str) -> str:
        """Strip namespace: '{http://...}Trackpoint' → 'Trackpoint'."""
        if "}" in tag:
            return tag.split("}", 1)[1]
        return tag

    @classmethod
    def _find_child(cls, parent: ET.Element, local_name: str) -> ET.Element | None:
        """Find direct child by local tag name, namespace-agnostic."""
        for child in parent:
            if cls._local_tag(child.tag) == local_name:
                return child
        return None

    @classmethod
    def _find_child_text(cls, parent: ET.Element, local_name: str) -> str | None:
        el = cls._find_child(parent, local_name)
        return el.text.strip() if el is not None and el.text else None

    @classmethod
    def _find_any(cls, root: ET.Element, local_name: str) -> ET.Element | None:
        """Find first element anywhere in the tree with this local name."""
        for el in root.iter():
            if cls._local_tag(el.tag) == local_name:
                return el
        return None

    @classmethod
    def _iter_local(cls, root: ET.Element, local_name: str):
        """Yield all descendants whose local tag matches."""
        for el in root.iter():
            if cls._local_tag(el.tag) == local_name:
                yield el

    @staticmethod
    def _parse_dt(text: str) -> datetime | None:
        text = text.strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            pass
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None


__all__ = ["TcxParser"]
