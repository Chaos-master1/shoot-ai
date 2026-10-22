"""GPS/IMU data importers for professional tracking vendors.

Supports:
  - Catapult Vector (CSV export)
  - STATSports Apex (CSV export)
  - Kinexon (JSON export)

All importers normalize to a common sample dict:
    {
        "timestamp": float (seconds since session start),
        "lat": float | None,
        "lon": float | None,
        "speed_ms": float | None,
        "acceleration": float | None,
        "accel_x": float | None,
        "accel_y": float | None,
        "accel_z": float | None,
        "heart_rate": int | None,
        "distance": float | None,
        "player_load": float | None,
        "metabolic_power": float | None,
        "speed_zone": int | None,
        "x_m": float | None,
        "y_m": float | None,
    }
"""

from __future__ import annotations

import contextlib
import csv
import json
import logging
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

logger = logging.getLogger("gps_import")

# Speed zone thresholds (m/s)
SPEED_ZONE_THRESHOLDS = [
    (1, "walking", 0.0, 1.7),
    (2, "jogging", 1.7, 3.3),
    (3, "running", 3.3, 5.5),
    (4, "high_intensity", 5.5, 7.0),
    (5, "sprinting", 7.0, float("inf")),
]


def _speed_zone(speed_ms: float | None) -> int | None:
    if speed_ms is None:
        return None
    for zone_id, _name, lo, hi in SPEED_ZONE_THRESHOLDS:
        if lo <= speed_ms < hi:
            return zone_id
    return 5  # sprinting (catch-all for over 7.0)


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _safe_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def _parse_timestamp_seconds(ts_str: str, start_time: datetime | None = None) -> float:
    """Parse timestamp string to seconds since start_time."""
    ts_str = ts_str.strip()
    try:
        # Try HH:MM:SS.sss format
        parts = ts_str.split(":")
        if len(parts) == 3:
            h, m, s = parts
            total_s = int(h) * 3600 + int(m) * 60 + float(s)
            return total_s
        elif len(parts) == 2:
            m, s = parts
            total_s = int(m) * 60 + float(s)
            return total_s
    except (ValueError, IndexError):
        pass
    try:
        # Try ISO format
        dt = datetime.fromisoformat(ts_str)
        if start_time:
            return (dt - start_time).total_seconds()
        return dt.timestamp()
    except (ValueError, TypeError):
        pass
    try:
        return float(ts_str)
    except (ValueError, TypeError):
        return 0.0


# ─── Catapult CSV ──────────────────────────────────────────────────────────


def parse_catapult_csv(content: str | bytes) -> list[dict[str, Any]]:
    """Parse Catapult Vector CSV export to normalized sample list.

    Catapult CSV typically has columns:
        Time, Speed (m/s), Accel X, Accel Y, Accel Z, Heart Rate,
        Distance (m), Player Load, Metabolic Power (W/kg), Lat, Lon
    """
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    reader = csv.DictReader(StringIO(content))
    samples: list[dict[str, Any]] = []
    start_time = None

    for row in reader:
        ts = _parse_timestamp_seconds(row.get("Time", "0"))
        if start_time is None:
            start_time = ts
        speed = _safe_float(row.get("Speed (m/s)") or row.get("Speed") or row.get("speed"))
        hr = _safe_int(row.get("Heart Rate") or row.get("Heart_Rate") or row.get("HR"))
        dist = _safe_float(row.get("Distance (m)") or row.get("Distance") or row.get("distance"))
        p_load = _safe_float(
            row.get("Player Load") or row.get("Player_Load") or row.get("PlayerLoad")
        )
        met_power = _safe_float(
            row.get("Metabolic Power (W/kg)")
            or row.get("Metabolic Power")
            or row.get("MetabolicPower")
        )
        lat = _safe_float(row.get("Lat") or row.get("lat") or row.get("Latitude"))
        lon = _safe_float(row.get("Lon") or row.get("lon") or row.get("Longitude"))

        sample = {
            "timestamp": round(ts, 3),
            "lat": lat,
            "lon": lon,
            "speed_ms": speed,
            "acceleration": None,
            "accel_x": _safe_float(row.get("Accel X") or row.get("Accel_X") or row.get("accel_x")),
            "accel_y": _safe_float(row.get("Accel Y") or row.get("Accel_Y") or row.get("accel_y")),
            "accel_z": _safe_float(row.get("Accel Z") or row.get("Accel_Z") or row.get("accel_z")),
            "heart_rate": hr,
            "distance": dist,
            "player_load": p_load,
            "metabolic_power": met_power,
            "speed_zone": _speed_zone(speed),
            "x_m": None,
            "y_m": None,
        }
        samples.append(sample)

    if not samples:
        logger.warning("Catapult CSV parser produced 0 samples")
    else:
        logger.info(f"Parsed {len(samples)} Catapult samples")
    return samples


# ─── STATSports CSV ────────────────────────────────────────────────────────


def parse_statsports_csv(content: str | bytes) -> list[dict[str, Any]]:
    """Parse STATSports Apex CSV export to normalized sample list.

    STATSports Apex columns:
        Time (s), Speed (km/h), Accel X (g), Accel Y (g), Accel Z (g),
        HR (bpm), Distance (m), Player Load, Metabolic Power
    """
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    reader = csv.DictReader(StringIO(content))
    samples: list[dict[str, Any]] = []

    for row in reader:
        ts = _parse_timestamp_seconds(row.get("Time (s)") or row.get("Time") or "0")
        speed_kmh = _safe_float(row.get("Speed (km/h)") or row.get("Speed") or row.get("speed"))
        speed_ms = round(speed_kmh / 3.6, 3) if speed_kmh is not None else None
        hr = _safe_int(
            row.get("HR (bpm)") or row.get("HR") or row.get("hr") or row.get("Heart Rate")
        )
        dist = _safe_float(row.get("Distance (m)") or row.get("Distance") or row.get("distance"))

        accel_x = _safe_float(row.get("Accel X (g)") or row.get("Accel X") or row.get("accel_x"))
        accel_y = _safe_float(row.get("Accel Y (g)") or row.get("Accel Y") or row.get("accel_y"))
        accel_z = _safe_float(row.get("Accel Z (g)") or row.get("Accel Z") or row.get("accel_z"))

        if accel_x is not None:
            accel_x *= 9.81  # convert g to m/s²
        if accel_y is not None:
            accel_y *= 9.81
        if accel_z is not None:
            accel_z *= 9.81

        # Compute net acceleration
        if accel_x is not None and accel_y is not None and accel_z is not None:
            net_accel = round((accel_x**2 + accel_y**2 + accel_z**2) ** 0.5, 3)
        elif speed_ms is not None:
            net_accel = None  # Can't compute from speed alone without time delta
        else:
            net_accel = None

        p_load = _safe_float(
            row.get("Player Load") or row.get("Player_Load") or row.get("PlayerLoad")
        )
        met_power = _safe_float(
            row.get("Metabolic Power") or row.get("MetabolicPower") or row.get("metabolic_power")
        )

        sample = {
            "timestamp": round(ts, 3),
            "lat": None,
            "lon": None,
            "speed_ms": speed_ms,
            "acceleration": net_accel,
            "accel_x": accel_x,
            "accel_y": accel_y,
            "accel_z": accel_z,
            "heart_rate": hr,
            "distance": dist,
            "player_load": p_load,
            "metabolic_power": met_power,
            "speed_zone": _speed_zone(speed_ms),
            "x_m": None,
            "y_m": None,
        }
        samples.append(sample)

    if not samples:
        logger.warning("STATSports CSV parser produced 0 samples")
    else:
        logger.info(f"Parsed {len(samples)} STATSports samples")
    return samples


# ─── Kinexon JSON ──────────────────────────────────────────────────────────


def parse_kinexon_json(content: str | bytes) -> list[dict[str, Any]]:
    """Parse Kinexon JSON export to normalized sample list.

    Kinexon JSON format:
    {
        "sessions": [{
            "start_time": "...",
            "samples": [{
                "timestamp": "HH:MM:SS.sss",
                "position": {"x": float, "y": float},
                "speed": float,
                "acceleration": float,
                "heart_rate": int,
                "player_load": float
            }]
        }]
    }
    """
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    data = json.loads(content)
    samples: list[dict[str, Any]] = []

    sessions = data.get("sessions", [data])  # allow single object or list
    for session in sessions:
        start_time = None
        with contextlib.suppress(ValueError, TypeError):
            start_time = datetime.fromisoformat(session.get("start_time", ""))

        for raw in session.get("samples", []):
            ts = raw.get("timestamp", 0)
            ts_s = _parse_timestamp_seconds(ts, start_time) if isinstance(ts, str) else float(ts)

            pos = raw.get("position") or {}
            speed = _safe_float(raw.get("speed"))
            accel = _safe_float(raw.get("acceleration"))
            hr = _safe_int(raw.get("heart_rate"))
            p_load = _safe_float(raw.get("player_load"))

            sample = {
                "timestamp": round(ts_s, 3),
                "lat": _safe_float(raw.get("lat") or raw.get("latitude")),
                "lon": _safe_float(raw.get("lon") or raw.get("longitude")),
                "speed_ms": speed,
                "acceleration": accel,
                "accel_x": None,
                "accel_y": None,
                "accel_z": None,
                "heart_rate": hr,
                "distance": _safe_float(raw.get("distance")),
                "player_load": p_load,
                "metabolic_power": _safe_float(raw.get("metabolic_power")),
                "speed_zone": _speed_zone(speed),
                "x_m": _safe_float(pos.get("x")),
                "y_m": _safe_float(pos.get("y")),
            }
            samples.append(sample)

    if not samples:
        logger.warning("Kinexon JSON parser produced 0 samples")
    else:
        logger.info(f"Parsed {len(samples)} Kinexon samples")
    return samples


# ─── Auto-detect ───────────────────────────────────────────────────────────


def import_gps_file(path: str | Path) -> list[dict[str, Any]]:
    """Auto-detect and import a GPS data file.

    Detects format by file extension and content patterns.

    Returns:
        List of normalized sample dicts.
    """
    path = Path(path)
    content = path.read_bytes()

    ext = path.suffix.lower()
    if ext == ".csv":
        head = content[:4096].decode("utf-8", errors="replace").lower()
        if "accelerometer" in head or "accel x" in head or "player load" in head:
            return parse_catapult_csv(content)
        elif "speed (km/h)" in head or "hr (bpm)" in head or "accel x (g)" in head:
            return parse_statsports_csv(content)
        # Try both
        catapult = parse_catapult_csv(content)
        if len(catapult) > 0:
            return catapult
        return parse_statsports_csv(content)
    elif ext == ".json":
        return parse_kinexon_json(content)
    else:
        raise ValueError(f"Unsupported GPS file format: {ext}")


def compute_session_summary(
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute summary statistics from a list of normalized GPS samples."""
    if not samples:
        return {}

    speeds = [s["speed_ms"] for s in samples if s["speed_ms"] is not None]
    distances = [s["distance"] for s in samples if s["distance"] is not None]
    heart_rates = [s["heart_rate"] for s in samples if s["heart_rate"] is not None]
    loads = [s["player_load"] for s in samples if s["player_load"] is not None]
    accels = [s["acceleration"] for s in samples if s["acceleration"] is not None]

    total_dist = sum(distances) if distances else 0.0
    duration = (
        max(s["timestamp"] for s in samples) - min(s["timestamp"] for s in samples)
        if len(samples) > 1
        else 0.0
    )

    # Distance by speed zone
    dist_by_zone: dict[str, float] = {}
    for s in samples:
        zone = s["speed_zone"]
        d = s["distance"] or 0.0
        dist_by_zone[zone] = dist_by_zone.get(zone, 0.0) + d

    # Sprint count (>25 km/h = 6.94 m/s)
    sprint_count = sum(1 for s in speeds if s >= 6.94) if speeds else 0

    # High intensity runs (>20 km/h = 5.56 m/s)
    hi_count = sum(1 for s in speeds if s >= 5.56) if speeds else 0

    # Acceleration/deceleration counts (>3 m/s²)
    accel_count = sum(1 for a in accels if a > 3.0) if accels else 0
    decel_count = sum(1 for a in accels if a < -3.0) if accels else 0

    return {
        "total_distance_m": round(total_dist, 1),
        "duration_s": round(duration, 1),
        "max_speed_kmh": round(max(speeds) * 3.6, 1) if speeds else 0.0,
        "avg_speed_kmh": round((sum(speeds) / len(speeds)) * 3.6, 1) if speeds else 0.0,
        "max_heart_rate": max(heart_rates) if heart_rates else None,
        "avg_heart_rate": round(sum(heart_rates) / len(heart_rates), 1) if heart_rates else None,
        "total_player_load": round(sum(loads), 1) if loads else 0.0,
        "sprint_count": sprint_count,
        "hi_run_count": hi_count,
        "acceleration_count": accel_count,
        "deceleration_count": decel_count,
        "distance_by_zone": {str(k): round(v, 1) for k, v in dist_by_zone.items()},
        "sample_count": len(samples),
    }
