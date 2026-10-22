"""Data import service — import event data from local files (CSV, JSON, StatsBomb)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


class DataImportService:
    """Import events from CSV, JSON, or StatsBomb JSON files."""

    def __init__(self, storage_service=None, coordinate_validator=None):
        self._storage = storage_service
        self._validator = coordinate_validator

    def import_file(self, file_path: str, match_id: str) -> dict:
        ext = Path(file_path).suffix.lower()
        if ext == ".csv":
            return self._import_csv(file_path, match_id)
        elif ext == ".json":
            return self._import_json(file_path, match_id)
        else:
            raise ValueError(f"Unsupported format: {ext}")

    def _import_csv(self, file_path: str, match_id: str) -> dict:
        events = []
        errors = []
        with open(file_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                try:
                    event = self._csv_row_to_event(row, match_id)
                    if event:
                        events.append(event)
                except Exception as e:
                    errors.append(f"Row {i + 2}: {e}")

        imported = 0
        skipped = 0
        if events and self._storage:
            imported = self._storage.save_events_bulk(match_id, events)
        skipped = len(events) - imported

        return {
            "imported_count": imported,
            "skipped_count": skipped,
            "errors": errors[:20],
            "total_errors": len(errors),
        }

    def _import_json(self, file_path: str, match_id: str) -> dict:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        events = []
        errors = []

        if isinstance(data, list):
            for i, item in enumerate(data):
                try:
                    event = self._json_item_to_event(item, match_id)
                    if event:
                        events.append(event)
                except Exception as e:
                    errors.append(f"Item {i + 1}: {e}")
        elif isinstance(data, dict):
            raw_events = data.get("events", data.get("data", [])) or []
            for i, item in enumerate(raw_events):
                try:
                    event = self._statsbomb_to_event(item, match_id)
                    if event:
                        events.append(event)
                except Exception as e:
                    errors.append(f"StatsBomb event {i + 1}: {e}")

        imported = 0
        if events and self._storage:
            imported = self._storage.save_events_bulk(match_id, events)
        skipped = len(events) - imported
        return {
            "imported_count": imported,
            "skipped_count": skipped,
            "errors": errors[:20],
            "total_errors": len(errors),
        }

    def _csv_row_to_event(self, row: dict, match_id: str) -> dict | None:
        row = {k.strip(): v for k, v in row.items()}
        event_type = row.get("type", "").strip().lower()
        if not event_type:
            return None

        event = {
            "type": event_type,
            "timestamp": float(row.get("timestamp", 0)),
            "team": row.get("team", "unknown"),
            "from_track_id": int(row["from_track_id"])
            if row.get("from_track_id", "").strip()
            else None,
            "to_track_id": int(row["to_track_id"]) if row.get("to_track_id", "").strip() else None,
            "player_name": row.get("player_name", ""),
            "completed": row.get("completed", "true").strip().lower() in ("true", "1", "yes"),
            "confidence": float(row.get("confidence", 1.0)),
            "period": int(row.get("period", 1)),
        }

        for coord in ("x", "y", "end_x", "end_y", "start_x", "start_y"):
            raw = row.get(coord, "").strip()
            if raw:
                event[coord] = float(raw)

        if row.get("xg", "").strip():
            event["xg"] = float(row["xg"])
        if row.get("xa", "").strip():
            event["xa"] = float(row["xa"])
        if row.get("xt", "").strip():
            event["xt"] = float(row["xt"])

        if self._validator:
            for coord in ("x", "y", "end_x", "end_y", "start_x", "start_y"):
                if coord in event:
                    if coord in ("x", "end_x", "start_x"):
                        event[coord] = self._validator.clamp_x(event[coord])
                    else:
                        event[coord] = self._validator.clamp_y(event[coord])

        return event

    def _json_item_to_event(self, item: dict, match_id: str) -> dict | None:
        event_type = str(item.get("type", "")).strip().lower()
        if not event_type:
            return None

        event = {
            "type": event_type,
            "timestamp": float(item.get("timestamp", 0)),
            "team": item.get("team", "unknown"),
            "from_track_id": item.get("from_track_id")
            or item.get("player_id")
            or item.get("track_id"),
            "to_track_id": item.get("to_track_id"),
            "player_name": item.get("player_name", ""),
            "completed": item.get("completed", True),
            "confidence": float(item.get("confidence", 1.0)),
            "period": int(item.get("period", 1)),
        }

        for coord in ("x", "y", "end_x", "end_y", "start_x", "start_y"):
            if coord in item and item[coord] is not None:
                event[coord] = float(item[coord])

        for metric in ("xg", "xa", "xt"):
            if metric in item and item[metric] is not None:
                event[metric] = float(item[metric])

        if self._validator:
            for coord in ("x", "y", "end_x", "end_y", "start_x", "start_y"):
                if coord in event:
                    if coord in ("x", "end_x", "start_x"):
                        event[coord] = self._validator.clamp_x(event[coord])
                    else:
                        event[coord] = self._validator.clamp_y(event[coord])

        return event

    @staticmethod
    def _parse_sb_timestamp(item: dict) -> float:
        raw = item.get("timestamp")
        if raw and isinstance(raw, str) and ":" in raw:
            parts = raw.split(":")
            try:
                h, m = int(parts[0]), int(parts[1])
                s = float(parts[2]) if len(parts) > 2 else 0.0
                return h * 3600 + m * 60 + s
            except (ValueError, IndexError):
                pass
        try:
            return float(item.get("timestamp", 0) or item.get("minute", 0) * 60)
        except (TypeError, ValueError):
            return float(item.get("minute", 0) * 60)

    def _statsbomb_to_event(self, item: dict, match_id: str) -> dict | None:
        type_info = item.get("type", {})
        if isinstance(type_info, dict):
            sb_type = type_info.get("name", "").lower()
        else:
            sb_type = str(type_info).lower()

        type_map = {
            "pass": "pass",
            "shot": "shot",
            "goal": "goal",
            "tackle": "tackle",
            "interception": "interception",
            "carry": "carry",
            "dribble": "dribble",
            "foul committed": "foul",
            "foul": "foul",
            "corner kick": "corner",
            "free kick": "free_kick",
            "substitution": "substitution",
            "clearance": "clearance",
            "offside": "offside",
            "save": "save",
            "card": "yellow_card",
        }
        event_type = type_map.get(sb_type, sb_type)

        location = item.get("location") or []
        start_x = float(location[0]) if len(location) > 0 else None
        start_y = float(location[1]) if len(location) > 1 else None

        end_x, end_y = None, None
        pass_info = item.get("pass") or {}
        end_loc = pass_info.get("end_location") or []
        if len(end_loc) > 0:
            end_x = float(end_loc[0])
        if len(end_loc) > 1:
            end_y = float(end_loc[1])

        shot_info = item.get("shot") or {}
        xg = shot_info.get("statsbomb_xg") or None

        team_info = item.get("team") or {}
        team_name = (
            team_info.get("name") if isinstance(team_info, dict) else item.get("team", "unknown")
        )

        player_info = item.get("player") or {}
        player_name = ""
        if isinstance(player_info, dict):
            player_name = player_info.get("name", "")
        player_id = player_info.get("id") if isinstance(player_info, dict) else None

        event = {
            "type": event_type,
            "timestamp": self._parse_sb_timestamp(item),
            "team": team_name,
            "from_track_id": player_id or item.get("from_track_id"),
            "to_track_id": item.get("to_track_id"),
            "player_name": player_name,
            "completed": True,
            "confidence": 1.0,
            "period": int(item.get("period", 1)),
            "x": start_x,
            "y": start_y,
            "end_x": end_x,
            "end_y": end_y,
        }

        if xg is not None:
            event["xg"] = float(xg)

        if self._validator:
            for coord in ("x", "y", "end_x", "end_y"):
                if coord in event and event[coord] is not None:
                    if coord in ("x", "end_x"):
                        event[coord] = self._validator.clamp_x(event[coord])
                    else:
                        event[coord] = self._validator.clamp_y(event[coord])

        return event

    def detect_format(self, file_path: str) -> str:
        with open(file_path, encoding="utf-8-sig") as f:
            first_chunk = f.read(4096)

        if not first_chunk.strip():
            raise ValueError("Empty file")

        stripped = first_chunk.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                try:
                    data = json.loads(first_chunk)
                except json.JSONDecodeError:
                    return "csv"
            if isinstance(data, dict):
                if "events" in data or "match" in data:
                    return "statsbomb_json"
                return "generic_json"
            if isinstance(data, list):
                if data and isinstance(data[0], dict):
                    first = data[0]
                    if "type" in first and isinstance(first.get("type"), dict):
                        return "statsbomb_json"
                    return "generic_json"
                return "generic_json"
            return "generic_json"

        return "csv"
