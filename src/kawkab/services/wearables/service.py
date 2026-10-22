"""WearableImportService — top-level facade for ingesting wearable data.

This is the entry point the bridge and CLI use. It:

1. Auto-detects the vendor/format from the file.
2. Dispatches to the right parser.
3. Returns JSON (back-compat with the original API) so the existing bridge
   surface keeps working.

Future cycles (C2–C5) add more parsers; this facade and the bridge don't
change. Future cycles (C10+) add a ``save()`` path that persists the parsed
:class:`~kawkab.services.wearables.models.WearableSession` to the new
``wearables_sessions`` table.
"""

from __future__ import annotations

import inspect
import json

from kawkab.core.logging import get_logger
from kawkab.services.wearables.auto import detect_parser
from kawkab.services.wearables.models import WearableSession

logger = get_logger(__name__)


class WearableImportService:
    """High-level facade over the vendor parsers.

    Preserves the original method names (``import_catapult_csv``,
    ``import_statsports_gpx``, ``import_polar_hr_csv``, ``import_auto``) so
    existing callers — and any future reflection-based bridge wiring — keep
    working. Each returns a JSON string for back-compat.

    The new preferred entry point is :meth:`import_session`, which returns a
    structured :class:`~kawkab.services.wearables.models.WearableSession`
    object instead of JSON, and is what the storage layer will consume.
    """

    def import_catapult_csv(self, file_path: str) -> str:
        from kawkab.services.wearables.catapult_csv import CatapultCsvParser

        return self._run(CatapultCsvParser(), file_path)

    def import_statsports_gpx(self, file_path: str) -> str:
        from kawkab.services.wearables.statsports_gpx import StatsportsGpxParser

        return self._run(StatsportsGpxParser(), file_path)

    def import_polar_hr_csv(self, file_path: str) -> str:
        from kawkab.services.wearables.polar_hr import PolarHrCsvParser

        return self._run(PolarHrCsvParser(), file_path)

    def import_auto(self, file_path: str) -> str:
        parser = detect_parser(file_path)
        if parser is None:
            return json.dumps({"error": f"Unsupported file format: {file_path}"})
        return self._run(parser, file_path)

    # -- persistence ------------------------------------------------------

    async def save_session(
        self, session: WearableSession, storage_service=None, match_id: int = 0
    ) -> dict:
        """Persist a parsed WearableSession to the database.

        Requires migration 020 (wearable_sessions table). If ``storage_service``
        is not provided the method returns a dict suitable for insertion.
        Accepts both async (StorageService/PG adapter) and sync (test stubs)
        ``save_wearable_session`` implementations.
        """
        if storage_service is not None:
            try:
                d = session.to_dict()
                row = {
                    "match_id": match_id or None,
                    "athlete_id": session.athlete_id,
                    "athlete_name": session.athlete_name,
                    "device_type": session.device_type,
                    "device_serial": session.device_serial,
                    "start_time": (
                        session.start_time.isoformat() if session.start_time is not None else ""
                    ),
                    "duration_s": d["duration_s"],
                    "sample_rate_hz": d["sample_rate_hz"],
                    "avg_hr": d["avg_hr"],
                    "max_hr": d["max_hr"],
                    "min_hr": d["min_hr"],
                    "total_distance_m": d["total_distance_m"],
                    "max_speed_ms": d["max_speed_ms"],
                    "avg_speed_ms": d["avg_speed_ms"],
                    "player_load": d.get("player_load"),
                    "body_load": d.get("body_load"),
                    "high_speed_running_m": d.get("high_speed_running_m"),
                    "sprint_distance_m": d.get("sprint_distance_m"),
                    "accelerations": d.get("accelerations"),
                    "decelerations": d.get("decelerations"),
                    "point_count": d["point_count"],
                    "metadata_json": json.dumps(session.metadata),
                }
                result = storage_service.save_wearable_session(row)
                if inspect.isawaitable(result):
                    result = await result
                return {"ok": True, "session_id": result}
            except Exception as e:
                logger.error(f"save_session failed: {e}")
                return {"error": str(e)}
        return {"error": "No storage_service provided"}

    # -- structured entry point -------------------------------------------

    def import_session(self, file_path: str) -> WearableSession | None:
        """Parse a file into a structured WearableSession (or None on failure).

        This is the preferred path for new code (storage, fusion, metrics).
        Use the ``import_*`` methods only when you need the legacy JSON shape.
        """
        parser = detect_parser(file_path)
        if parser is None:
            logger.error(f"import_session: no parser for {file_path}")
            return None
        try:
            return parser.parse(file_path)
        except Exception as e:
            logger.error(f"import_session failed on {file_path}: {e}")
            return None

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _run(parser, file_path: str) -> str:
        try:
            session = parser.parse(file_path)
            return json.dumps({"session": session.to_dict(), "ok": True})
        except FileNotFoundError as e:
            logger.error(f"wearable import: {e}")
            return json.dumps({"error": str(e)})
        except Exception as e:
            logger.error(f"wearable import failed on {file_path}: {e}")
            return json.dumps({"error": str(e)})


__all__ = ["WearableImportService"]
