"""Wearables data models — vendor-neutral session and data-point representations.

All vendor parsers (Catapult, STATSports, Polar, FIT, TCX, GPX) convert their
proprietary formats into the unified ``WearableDataPoint`` / ``WearableSession``
types defined here. Downstream code (metric computer, fusion layer, storage)
only ever deals with these unified types, never with vendor-specific schemas.

A ``WearableDataPoint`` is a single sample at one timestamp. A
``WearableSession`` is the full session for one athlete. Sessions are
deliberately per-athlete: a team training session produces N sessions, one per
player, mirroring how Catapult OpenField and STATSports Sonra export data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WearableDataPoint:
    """A single raw sample from a wearable at one timestamp.

    All fields except ``timestamp_s`` are optional because no single vendor
    populates every channel (e.g. Catapult has PlayerLoad, Polar HR CSV has
    only heart rate, FIT has lat/lon + speed + HR + cadence).
    """

    timestamp_s: float = 0.0
    heart_rate_bpm: float | None = None
    speed_ms: float | None = None
    distance_m: float | None = None
    acceleration_ms2: float | None = None
    latitude: float | None = None
    longitude: float | None = None
    altitude_m: float | None = None
    cadence_rpm: float | None = None
    power_w: float | None = None
    # Vendors-specific extras that don't fit the unified channels above
    # (e.g. Catapult "Body Load", STATSports "PlayerLoad", FIT "left_right_balance").
    extras: dict = field(default_factory=dict)


@dataclass
class WearableSession:
    """A full wearable session for one athlete.

    Attributes:
        device_type: vendor key (``catapult``, ``statsports``, ``polar``,
            ``fit``, ``tcx``, ``gpx``, ``wimu`` ...).
        device_serial: device serial number if the source file exposes it.
        athlete_id: stable athlete identifier if known (else ``""``).
        athlete_name: display name if known (else ``""``).
        start_time: session start as a datetime (UTC-aware when the source
            provides a timezone, naive otherwise).
        duration_s: wall-clock duration of the session in seconds.
        sample_rate_hz: native sampling rate of the dominant channel, when
            known (Catapult 10 Hz, STATSports 10 Hz, FIT 1 Hz, etc.). ``0.0``
            means unknown.
        data: ordered list of ``WearableDataPoint`` sorted by timestamp.
        metadata: vendor-specific bag (session id, drill labels, team id,
            pitch id, raw column headers, parser version, ...).
    """

    device_type: str = ""
    device_serial: str = ""
    athlete_id: str = ""
    athlete_name: str = ""
    start_time: datetime | None = None
    duration_s: float = 0.0
    sample_rate_hz: float = 0.0
    data: list[WearableDataPoint] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def finalize(self) -> None:
        """Recompute derived session fields (duration, sample rate) from data.

        Call this once at the end of every parser, after all data points have
        been appended. Idempotent.
        """
        if not self.data:
            self.duration_s = 0.0
            self.sample_rate_hz = 0.0
            return
        ts = [d.timestamp_s for d in self.data]
        first, last = ts[0], ts[-1]
        self.duration_s = max(0.0, last - first)
        span = last - first
        if span > 0 and len(self.data) > 1:
            self.sample_rate_hz = round((len(self.data) - 1) / span, 3)

    def to_dict(self) -> dict:
        """Bridge-friendly summary dict (no raw data — that goes via storage)."""
        hr = [d.heart_rate_bpm for d in self.data if d.heart_rate_bpm is not None]
        spd = [d.speed_ms for d in self.data if d.speed_ms is not None]
        dist = [d.distance_m for d in self.data if d.distance_m is not None]
        return {
            "device_type": self.device_type,
            "device_serial": self.device_serial,
            "athlete_id": self.athlete_id,
            "athlete_name": self.athlete_name,
            "start_time": self.start_time.isoformat() if self.start_time else "",
            "duration_s": round(self.duration_s, 1),
            "sample_rate_hz": self.sample_rate_hz,
            "point_count": len(self.data),
            "avg_hr": round(float(np.mean(hr)), 1) if hr else 0.0,
            "max_hr": round(float(np.max(hr)), 1) if hr else 0.0,
            "min_hr": round(float(np.min(hr)), 1) if hr else 0.0,
            "total_distance_m": round(float(np.sum(dist)), 1) if dist else 0.0,
            "max_speed_ms": round(float(np.max(spd)), 2) if spd else 0.0,
            "avg_speed_ms": round(float(np.mean(spd)), 2) if spd else 0.0,
        }


__all__ = ["WearableDataPoint", "WearableSession"]
