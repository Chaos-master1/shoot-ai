"""Data export service - export match data in professional formats.

Supports:
1. CSV export - match events, player stats, team stats
2. JSON export - full analysis results, structured data
3. StatsBomb-compatible JSON - industry standard event data format
4. SPADL-compatible - socceraction format for action valuation

Professional analysts need to move data between tools. This service
makes Kawkab AI data portable.
"""

from __future__ import annotations

import contextlib
import csv
import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from kawkab.core.logging import get_logger
from kawkab.core.paths import get_paths

logger = get_logger(__name__)


class DataExportService:
    """Exports match data in various professional formats."""

    def __init__(self) -> None:
        self._db_path = get_paths().database
        self._exports_dir = get_paths().exports
        self._conn: sqlite3.Connection | None = None
        logger.info(f"DataExportService: exports_dir={self._exports_dir}")

    def _sanitize_name(self, name: str) -> str:
        """Sanitize user-derived name strings for safe filesystem use."""
        sanitized = name.replace(" ", "_")
        sanitized = sanitized.replace("/", "-")
        sanitized = sanitized.replace("\\", "-")
        sanitized = sanitized.replace("..", "")
        return sanitized.strip("._- ")

    def _resolve_export_path(self, *segments: str) -> Path:
        """Build and validate a path under exports_dir, preventing traversal."""
        resolved = self._exports_dir.resolve()
        for seg in segments:
            safe = self._sanitize_name(seg)
            resolved = resolved / safe
        resolved = resolved.resolve()
        if not str(resolved).startswith(str(self._exports_dir.resolve())):
            raise ValueError(f"Path traversal detected: {resolved}")
        return resolved

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    async def export_match_csv(
        self, match_id: int, include_events: bool = True, include_players: bool = True
    ) -> Path:
        """Export match data as CSV files (events.csv, players.csv, summary.csv).

        Returns:
            Path to directory containing CSV files.
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM matches WHERE id = ?", (match_id,))
        match_row = cursor.fetchone()
        if not match_row:
            raise ValueError(f"Match {match_id} not found")

        match_name = self._sanitize_name(match_row["name"])
        export_dir = self._resolve_export_path(
            f"match_{match_id}_{match_name}_{datetime.now().strftime('%Y%m%d')}"
        )
        export_dir.mkdir(parents=True, exist_ok=True)

        # Summary CSV
        summary_path = export_dir / "summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "match_id",
                    "name",
                    "home_team",
                    "away_team",
                    "match_date",
                    "duration_seconds",
                    "fps",
                    "total_frames",
                ]
            )
            writer.writerow(
                [
                    match_row["id"],
                    match_row["name"],
                    match_row["home_team"],
                    match_row["away_team"],
                    match_row["match_date"],
                    match_row["duration_seconds"],
                    match_row["fps"],
                    match_row["total_frames"],
                ]
            )

        # Events CSV
        if include_events:
            cursor.execute(
                "SELECT * FROM events WHERE match_id = ? ORDER BY timestamp",
                (match_id,),
            )
            events = cursor.fetchall()
            events_path = export_dir / "events.csv"
            with open(events_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "event_id",
                        "event_type",
                        "timestamp",
                        "from_track_id",
                        "to_track_id",
                        "team",
                        "completed",
                        "confidence",
                        "metadata",
                    ]
                )
                for e in events:
                    writer.writerow(
                        [
                            e["id"],
                            e["event_type"],
                            e["timestamp"],
                            e["from_track_id"],
                            e["to_track_id"],
                            e["team"],
                            e["completed"],
                            e["confidence"],
                            e["metadata"],
                        ]
                    )

        # Players CSV
        if include_players:
            cursor.execute("SELECT * FROM players WHERE match_id = ?", (match_id,))
            players = cursor.fetchall()
            players_path = export_dir / "players.csv"
            with open(players_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "player_id",
                        "track_id",
                        "jersey_number",
                        "name",
                        "team",
                        "position",
                        "distance_covered_m",
                        "max_speed_kmh",
                        "avg_speed_kmh",
                        "passes_attempted",
                        "passes_completed",
                        "shots",
                        "tackles",
                    ]
                )
                for p in players:
                    writer.writerow(
                        [
                            p["id"],
                            p["track_id"],
                            p["jersey_number"],
                            p["name"],
                            p["team"],
                            p["position"],
                            p["distance_covered_m"],
                            p["max_speed_kmh"],
                            p["avg_speed_kmh"],
                            p["passes_attempted"],
                            p["passes_completed"],
                            p["shots"],
                            p["tackles"],
                        ]
                    )

        logger.info(f"Exported match {match_id} CSV to {export_dir}")
        return export_dir

    async def export_match_json(self, match_id: int) -> Path:
        """Export full match analysis as a single structured JSON file.

        Returns:
            Path to the JSON file.
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM matches WHERE id = ?", (match_id,))
        match_row = cursor.fetchone()
        if not match_row:
            raise ValueError(f"Match {match_id} not found")

        match_name = self._sanitize_name(match_row["name"])
        export_path = self._resolve_export_path(
            f"match_{match_id}_{match_name}_{datetime.now().strftime('%Y%m%d')}.json"
        )

        cursor.execute("SELECT * FROM analysis_results WHERE match_id = ?", (match_id,))
        analysis_row = cursor.fetchone()

        cursor.execute("SELECT * FROM players WHERE match_id = ?", (match_id,))
        players = [dict(p) for p in cursor.fetchall()]

        cursor.execute("SELECT * FROM events WHERE match_id = ? ORDER BY timestamp", (match_id,))
        events = [dict(e) for e in cursor.fetchall()]

        for e in events:
            try:
                e["metadata"] = json.loads(e.get("metadata", "{}"))
            except Exception:
                e["metadata"] = {}

        data = {
            "match": dict(match_row),
            "analysis": dict(analysis_row) if analysis_row else {},
            "players": players,
            "events": events,
            "exported_at": datetime.now().isoformat(),
            "export_version": "1.0",
            "source": "Kawkab AI",
        }

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Exported match {match_id} JSON to {export_path}")
        return export_path

    async def export_statsbomb_compatible(self, match_id: int) -> Path:
        """Export match events in a StatsBomb-compatible JSON format.

        StatsBomb format is the industry standard for football event data.
        This format allows importing into tools like socceraction, mplsoccer,
        and other analytics libraries.
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM matches WHERE id = ?", (match_id,))
        match_row = cursor.fetchone()
        if not match_row:
            raise ValueError(f"Match {match_id} not found")

        match_name = self._sanitize_name(match_row["name"])
        export_path = self._resolve_export_path(
            f"statsbomb_{match_id}_{match_name}_{datetime.now().strftime('%Y%m%d')}.json"
        )

        cursor.execute(
            "SELECT * FROM events WHERE match_id = ? ORDER BY timestamp",
            (match_id,),
        )
        events = cursor.fetchall()

        statsbomb_events: list[dict[str, Any]] = []
        for e in events:
            event_type = e["event_type"]

            # Map Kawkab event types to StatsBomb type IDs
            type_id, type_name = self._map_event_type(event_type)

            # Parse metadata JSON once
            try:
                meta = json.loads(e["metadata"] or "{}")
            except Exception:
                meta = {}

            # Period detection: use explicit field or compute from timestamp
            period = meta.get("period")
            if period is None:
                with contextlib.suppress(KeyError, IndexError, TypeError):
                    period = e["period"]
            if period is None:
                ts = float(e["timestamp"])
                period = 1 if ts < 2700 else 2

            sb_event = {
                "id": e["id"],
                "match_id": match_id,
                "index": len(statsbomb_events) + 1,
                "period": period,
                "timestamp": e["timestamp"],
                "minute": int(e["timestamp"] // 60),
                "second": int(e["timestamp"] % 60),
                "type": {"id": type_id, "name": type_name},
                "team": {"id": 1 if e["team"] == "home" else 2, "name": e["team"]},
                "player": {"id": e["from_track_id"], "name": f"Player {e['from_track_id']}"},
                "position": {"id": 1, "name": "Unknown"},
                "possession_team": {"id": 1 if e["team"] == "home" else 2, "name": e["team"]},
                "possession": len(statsbomb_events) + 1,
                "play_pattern": {"id": 1, "name": "Regular Play"},
            }

            if event_type == "pass":
                # Compute pass length and angle from positions
                start_x = meta.get("start_x")
                start_y = meta.get("start_y")
                end_x = meta.get("end_x")
                end_y = meta.get("end_y")
                if (
                    start_x is not None
                    and start_y is not None
                    and end_x is not None
                    and end_y is not None
                ):
                    dx = float(end_x) - float(start_x)
                    dy = float(end_y) - float(start_y)
                    length = math.hypot(dx, dy)
                    angle = math.degrees(math.atan2(dy, dx))
                else:
                    length = 0.0
                    angle = 0.0
                sb_event["pass"] = {
                    "recipient": {"id": e["to_track_id"], "name": f"Player {e['to_track_id']}"},
                    "outcome": {
                        "id": 15 if e["completed"] else 9,
                        "name": "Complete" if e["completed"] else "Incomplete",
                    },
                    "length": round(length, 1),
                    "angle": round(angle, 1),
                    "height": {"id": 1, "name": "Ground Pass"},
                }
            elif event_type == "shot":
                # Shot outcome mapping
                if meta.get("is_goal"):
                    outcome_id, outcome_name = 97, "Goal"
                elif meta.get("saved") or meta.get("is_saved"):
                    outcome_id, outcome_name = 95, "Saved"
                elif meta.get("blocked") or meta.get("is_blocked"):
                    outcome_id, outcome_name = 96, "Blocked"
                elif meta.get("off_target") or meta.get("missed"):
                    outcome_id, outcome_name = 94, "Missed"
                else:
                    outcome_id, outcome_name = 94, "Missed"
                # xG from event.xg or event.x_g, not distance
                xg_val = meta.get("xg") or meta.get("x_g") or 0.0
                try:
                    xg_val = float(xg_val)
                except (TypeError, ValueError):
                    xg_val = 0.0
                sb_event["shot"] = {
                    "outcome": {"id": outcome_id, "name": outcome_name},
                    "xG": round(xg_val, 4),
                    "key_pass_id": None,
                }

            statsbomb_events.append(sb_event)

        data = {
            "match_id": match_id,
            "match_name": match_row["name"],
            "home_team": match_row["home_team"],
            "away_team": match_row["away_team"],
            "events": statsbomb_events,
            "metadata": {
                "source": "Kawkab AI",
                "export_version": "1.0",
                "exported_at": datetime.now().isoformat(),
            },
        }

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Exported StatsBomb-compatible JSON to {export_path}")
        return export_path

    def _map_event_type(self, event_type: str) -> tuple[int, str]:
        """Map Kawkab event types to StatsBomb type IDs."""
        mapping = {
            "pass": (30, "Pass"),
            "shot": (16, "Shot"),
            "tackle": (70, "Tackle"),
            "interception": (49, "Interception"),
            "carry": (43, "Carry"),
            "duel": (4, "Duel"),
            "clearance": (22, "Clearance"),
            "foul": (21, "Foul Committed"),
            "offside": (72, "Offside"),
            "goalkeeper": (23, "Goal Keeper"),
            "ball_recovery": (42, "Ball Recovery"),
            "dribble": (14, "Dribble"),
            "goal": (16, "Goal"),
            "corner": (6, "Corner Kick"),
            "free_kick": (66, "Free Kick"),
            "throw_in": (52, "Throw In"),
        }
        return mapping.get(event_type, (1, "Unknown"))

    async def export_season_csv(self, season_id: int) -> Path:
        """Export all matches in a season as a single CSV."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, name, match_date, home_team, away_team, score_home, score_away,
                   duration_seconds, match_type
            FROM matches
            WHERE season_id = ?
            ORDER BY match_date ASC
            """,
            (season_id,),
        )
        matches = cursor.fetchall()

        cursor.execute("SELECT name FROM seasons WHERE id = ?", (season_id,))
        season_row = cursor.fetchone()
        season_name = season_row["name"] if season_row else f"Season_{season_id}"

        export_path = self._resolve_export_path(
            f"season_{season_id}_{self._sanitize_name(season_name)}_{datetime.now().strftime('%Y%m%d')}.csv"
        )

        with open(export_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "match_id",
                    "name",
                    "date",
                    "home_team",
                    "away_team",
                    "score_home",
                    "score_away",
                    "duration_seconds",
                    "match_type",
                ]
            )
            for m in matches:
                writer.writerow(
                    [
                        m["id"],
                        m["name"],
                        m["match_date"],
                        m["home_team"],
                        m["away_team"],
                        m["score_home"],
                        m["score_away"],
                        m["duration_seconds"],
                        m["match_type"],
                    ]
                )

        logger.info(f"Exported season {season_id} CSV to {export_path}")
        return export_path

    async def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
