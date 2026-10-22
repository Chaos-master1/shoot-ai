"""Vendor auto-detection.

Given a file path, pick the right parser by sniffing extension + content.
Order of preference when the extension is ambiguous (e.g. ``.csv`` is shared
by Catapult, STATSports Sonra, and Polar):

1. STATSports Sonra CSV exports always contain the literal ``"Player Name"``
   or ``"Total Distance"`` column (very distinctive).
2. Polar HR CSV exports start with ``Time,HR`` or contain a ``---`` separator.
3. Catapult OpenField sensor CSV exports contain ``"Timestamp (s)"`` and
   ``"Speed (m/s)"`` columns.
4. Otherwise fall back to the first parser that accepts the extension.

Detection only reads the first ~4 KB of the file to stay fast on large
session files (which can be several MB).
"""

from __future__ import annotations

from os.path import splitext

from kawkab.core.logging import get_logger
from kawkab.services.wearables.base import BaseWearableParser
from kawkab.services.wearables.catapult_csv import CatapultCsvParser
from kawkab.services.wearables.fit_parser import FitParser
from kawkab.services.wearables.polar_hr import PolarHrCsvParser
from kawkab.services.wearables.statsports_csv import StatsportsCsvParser
from kawkab.services.wearables.statsports_gpx import StatsportsGpxParser
from kawkab.services.wearables.tcx_parser import TcxParser

logger = get_logger(__name__)

_SNIFF_BYTES = 4096


def detect_parser(file_path: str) -> BaseWearableParser | None:
    """Pick a parser for ``file_path`` or return None if unknown format."""
    ext = splitext(file_path)[1].lower()

    if ext == ".gpx":
        return StatsportsGpxParser()
    if ext == ".fit":
        return FitParser()
    if ext == ".tcx":
        return TcxParser()
    if ext == ".csv":
        return _detect_csv(file_path)
    # TCX files sometimes have .xml extension
    if ext == ".xml":
        return TcxParser()

    logger.warning(f"auto-detect: unsupported extension {ext!r} for {file_path}")
    return None


def _detect_csv(file_path: str) -> BaseWearableParser | None:
    """Sniff the header to distinguish Catapult / Sonra / Polar CSV."""
    try:
        with open(file_path, encoding="utf-8-sig", errors="ignore") as f:
            head = f.read(_SNIFF_BYTES).lower()
    except OSError as e:
        logger.error(f"auto-detect: cannot read {file_path}: {e}")
        return None

    # Sonra distinctive columns
    if "player name" in head and "total distance" in head:
        logger.info("auto-detect: STATSports Sonra aggregate CSV")
        return StatsportsCsvParser()

    # Polar HR distinctive
    if "hr (bpm)" in head and ("---" in head or head.startswith("time,")):
        return PolarHrCsvParser()
    if head.startswith("time") and "hr" in head.split("\n", 1)[0]:
        return PolarHrCsvParser()

    # Catapult OpenField distinctive
    if "timestamp (s)" in head or "speed (m/s)" in head:
        return CatapultCsvParser()

    # Generic CSV — best guess: Catapult-style (most permissive alias table).
    logger.info("auto-detect: unrecognised CSV header; defaulting to Catapult parser")
    return CatapultCsvParser()


__all__ = ["detect_parser"]
