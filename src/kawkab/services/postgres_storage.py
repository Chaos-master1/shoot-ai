"""PostgreSQL adapter for StorageService — asyncpg-based parallel to SQLite.

Full implementation covering all 43 tables from the SQLite schema.
Enables dual-mode operation: SQLite for local single-user, PostgreSQL for club deployment.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kawkab.core.logging import get_logger
from kawkab.services.storage.base import parse_metadata_json
from kawkab.services.storage_errors import (
    StorageDuplicateError,
    StorageNotInitializedError,
    StorageWriteError,
    is_duplicate_violation,
)

logger = get_logger(__name__)


class PostgresStorageAdapter:
    """Full PostgreSQL adapter mirroring ALL StorageService async methods.

    Contract (coding-tags cluster first; the rest migrates cluster by
    cluster): "pool not available" raises StorageNotInitialized instead of
    feeding silent 0/False/[] — the old fallback once made a Postgres
    deployment return empty data for every call. Rejected input keeps the
    legacy falsy return; database failures raise.
    Only raises NotImplementedError for SQLite-specific features (.backup() API).
    """

    _COLUMN_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

    def __init__(self, dsn: str | None = None):
        self._dsn = dsn
        self._pool = None
        self._available = False

    # ── Core connection methods ─────────────────────────────────────────────

    async def initialize(self):
        if self._pool:
            return  # idempotent: initialize() is safe to call repeatedly
        if not self._dsn:
            self._dsn = os.environ.get("KAWKAB_DB_URL")
        if not self._dsn:
            logger.info("PostgreSQL: no DSN, skipping initialization")
            return
        try:
            import asyncpg

            self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)
            self._available = True
            await self._apply_schema()
        except Exception as exc:
            self._available = False
            logger.warning("PostgreSQL init failed: %s", exc)

    async def _apply_schema(self):
        """Apply the full schema from pg_schema.sql if available, else inline DDL."""
        schema_path = Path(__file__).resolve().parent.parent / "migrations" / "pg_schema.sql"
        if schema_path.exists():
            sql = schema_path.read_text(encoding="utf-8")
            async with self._pool.acquire() as conn:
                await conn.execute(sql)
            logger.info("PostgreSQL schema applied from pg_schema.sql")
        else:
            async with self._pool.acquire() as conn:
                await conn.execute(self._SCHEMA_SQL)
            logger.info("PostgreSQL schema applied from inline DDL")
        # CREATE TABLE IF NOT EXISTS matches(...) above is a no-op against
        # an already-existing matches table from before match ownership
        # was added -- ADD COLUMN IF NOT EXISTS (Postgres-only syntax)
        # covers that upgrade path, matching the same pattern used for
        # cloud/database.py's users.token_version.
        async with self._pool.acquire() as conn:
            await conn.execute("ALTER TABLE matches ADD COLUMN IF NOT EXISTS owner_id INTEGER")
            await conn.execute("ALTER TABLE matches ADD COLUMN IF NOT EXISTS team_id INTEGER")
            await conn.execute(
                "ALTER TABLE matches ADD COLUMN IF NOT EXISTS is_shared INTEGER DEFAULT 0"
            )

    async def close(self):
        if self._pool:
            await self._pool.close()
            self._pool = None

    @property
    def available(self) -> bool:
        return self._available

    async def fetch(self, query: str, *args) -> list[dict]:
        if not self._pool:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(r) for r in rows]

    async def fetchrow(self, query: str, *args) -> dict | None:
        if not self._pool:
            return None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None

    async def execute(self, query: str, *args) -> str:
        if not self._pool:
            return "0"
        async with self._pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def executemany(self, query: str, args: list[tuple]) -> None:
        if not self._pool:
            return
        async with self._pool.acquire() as conn:
            await conn.executemany(query, args)

    @staticmethod
    def _sanitize_column_name(name: str) -> str | None:
        cleaned = re.sub(r"[^a-zA-Z0-9_]", "", name)
        if cleaned and (cleaned[0].isalpha() or cleaned.startswith("_")):
            return cleaned
        return None

    # ── Match CRUD ──────────────────────────────────────────────────────────

    async def save_match(
        self, name: str, video_path: str, home_team: str = "", away_team: str = ""
    ) -> int:
        if not self._pool:
            raise StorageNotInitializedError("save_match")
        async with self._pool.acquire() as conn:
            home_id = await self._ensure_team_id(conn, home_team) if home_team else None
            away_id = await self._ensure_team_id(conn, away_team) if away_team else None
            row = await conn.fetchrow(
                """INSERT INTO matches (name, video_path, home_team, away_team, home_team_id, away_team_id)
                   VALUES ($1, $2, $3, $4, $5, $6) RETURNING id""",
                name,
                video_path,
                home_team,
                away_team,
                home_id,
                away_id,
            )
            return row["id"] if row else 0

    async def _ensure_team_id(self, conn, name: str) -> int | None:
        row = await conn.fetchrow("SELECT id FROM teams WHERE name = $1", name)
        if row:
            return row["id"]
        row = await conn.fetchrow(
            "INSERT INTO teams (name, short_name) VALUES ($1, $2) ON CONFLICT (name) DO UPDATE SET name=EXCLUDED.name RETURNING id",
            name,
            name[:3].upper(),
        )
        return row["id"] if row else None

    async def get_all_matches(self) -> list[dict]:
        if not self._pool:
            raise StorageNotInitializedError("get_all_matches")
        return await self.fetch(
            """SELECT id, name, video_path, home_team, away_team, match_date,
                      duration_seconds, analyzed_at, created_at,
                      api_match_id, competition_code,
                      bzzoiro_home_team_id, bzzoiro_away_team_id,
                      bzzoiro_event_id, bzzoiro_league_id,
                      apifb_home_team_id, apifb_away_team_id,
                      apifb_fixture_id, apifb_league_id
               FROM matches
               WHERE (is_deleted IS NULL OR is_deleted=0)
               ORDER BY created_at DESC"""
        )

    async def get_match(self, match_id: int) -> dict | None:
        if not self._pool:
            raise StorageNotInitializedError("get_match")
        return await self.fetchrow(
            """SELECT id, name, video_path, duration_seconds AS duration, fps, total_frames,
                      home_team_id, away_team_id, score_home, score_away,
                      season_id, match_date, match_type, home_team, away_team, created_at
               FROM matches
               WHERE id = $1 AND (is_deleted IS NULL OR is_deleted=0)""",
            match_id,
        )

    async def hard_delete_match(self, match_id: int) -> bool:
        if not self._pool:
            raise StorageNotInitializedError("hard_delete_match")
        async with self._pool.acquire() as conn:
            r = await conn.execute("DELETE FROM matches WHERE id = $1", match_id)
            return r != "DELETE 0"

    async def restore_match(self, match_id: int) -> bool:
        if not self._pool:
            raise StorageNotInitializedError("restore_match")
        async with self._pool.acquire() as conn:
            r = await conn.execute(
                "UPDATE matches SET is_deleted=0, deleted_at=NULL WHERE id = $1",
                match_id,
            )
            return r != "UPDATE 0"

    async def update_match_analysis(
        self,
        match_id: int,
        analysis_data: Any = None,
        duration: float | None = None,
        fps: float | None = None,
        total_frames: int | None = None,
    ) -> bool:
        """Update match analysis fields. Accepts both dict (PG) and scalar (SQLite) signatures."""
        if not self._pool:
            raise StorageNotInitializedError("update_match_analysis")
        async with self._pool.acquire() as conn:
            if isinstance(analysis_data, dict) and duration is None:
                r = await conn.execute(
                    "UPDATE matches SET analysis_json = $1, updated_at = NOW() WHERE id = $2",
                    json.dumps(analysis_data, default=str),
                    match_id,
                )
            else:
                r = await conn.execute(
                    "UPDATE matches SET duration_seconds = $1, fps = $2, total_frames = $3, analyzed_at = NOW() WHERE id = $4",
                    duration or 0,
                    fps or 0,
                    total_frames or 0,
                    match_id,
                )
            return r != "UPDATE 0"

    async def update_match_teams(self, match_id: int, home_team: str, away_team: str) -> bool:
        if not self._pool:
            raise StorageNotInitializedError("update_match_teams")
        async with self._pool.acquire() as conn:
            r = await conn.execute(
                "UPDATE matches SET home_team = $1, away_team = $2, updated_at = NOW() WHERE id = $3",
                home_team,
                away_team,
                match_id,
            )
            return r != "UPDATE 0"

    async def update_match_football_data(
        self,
        match_id: int,
        data: Any = None,
        api_match_id: int | None = None,
        competition_code: str | None = None,
        football_data_home_team_id: int | None = None,
        football_data_away_team_id: int | None = None,
    ) -> bool:
        if not self._pool:
            raise StorageNotInitializedError("update_match_football_data")
        async with self._pool.acquire() as conn:
            if isinstance(data, dict) and api_match_id is None:
                r = await conn.execute(
                    "UPDATE matches SET football_data_json = $1, updated_at = NOW() WHERE id = $2",
                    json.dumps(data, default=str),
                    match_id,
                )
                return r != "UPDATE 0"
            sets = []
            args = []
            if api_match_id is not None:
                sets.append(f"api_match_id = ${len(args) + 1}")
                args.append(api_match_id)
            if competition_code is not None:
                sets.append(f"competition_code = ${len(args) + 1}")
                args.append(competition_code)
            if football_data_home_team_id is not None:
                sets.append(f"football_data_home_team_id = ${len(args) + 1}")
                args.append(football_data_home_team_id)
            if football_data_away_team_id is not None:
                sets.append(f"football_data_away_team_id = ${len(args) + 1}")
                args.append(football_data_away_team_id)
            if not sets:
                return False
            args.append(match_id)
            r = await conn.execute(
                f"UPDATE matches SET {', '.join(sets)}, updated_at = NOW() WHERE id = ${len(args)}",
                *args,
            )
            return r != "UPDATE 0"

    async def update_match_apifootball(
        self,
        match_id: int,
        data: Any = None,
        apifb_home_team_id: int | None = None,
        apifb_away_team_id: int | None = None,
        apifb_fixture_id: int | None = None,
        apifb_league_id: int | None = None,
        apifb_season: int | None = None,
    ) -> bool:
        if not self._pool:
            raise StorageNotInitializedError("update_match_apifootball")
        async with self._pool.acquire() as conn:
            if isinstance(data, dict) and apifb_home_team_id is None:
                r = await conn.execute(
                    "UPDATE matches SET apifootball_json = $1, updated_at = NOW() WHERE id = $2",
                    json.dumps(data, default=str),
                    match_id,
                )
                return r != "UPDATE 0"
            sets = []
            args = []
            for name, val in [
                ("apifb_home_team_id", apifb_home_team_id),
                ("apifb_away_team_id", apifb_away_team_id),
                ("apifb_fixture_id", apifb_fixture_id),
                ("apifb_league_id", apifb_league_id),
                ("apifb_season", apifb_season),
            ]:
                if val is not None:
                    sets.append(f"{name} = ${len(args) + 1}")
                    args.append(val)
            if not sets:
                return False
            args.append(match_id)
            r = await conn.execute(
                f"UPDATE matches SET {', '.join(sets)}, updated_at = NOW() WHERE id = ${len(args)}",
                *args,
            )
            return r != "UPDATE 0"

    async def update_match_bzzoiro(
        self,
        match_id: int,
        data: Any = None,
        bzzoiro_home_team_id: int | None = None,
        bzzoiro_away_team_id: int | None = None,
        bzzoiro_event_id: int | None = None,
        bzzoiro_league_id: int | None = None,
        bzzoiro_competition_code: str | None = None,
        prediction_data: str | None = None,
    ) -> bool:
        if not self._pool:
            raise StorageNotInitializedError("update_match_bzzoiro")
        async with self._pool.acquire() as conn:
            if isinstance(data, dict) and bzzoiro_home_team_id is None:
                r = await conn.execute(
                    "UPDATE matches SET bzzoiro_json = $1, updated_at = NOW() WHERE id = $2",
                    json.dumps(data, default=str),
                    match_id,
                )
                return r != "UPDATE 0"
            sets = []
            args = []
            for name, val in [
                ("bzzoiro_home_team_id", bzzoiro_home_team_id),
                ("bzzoiro_away_team_id", bzzoiro_away_team_id),
                ("bzzoiro_event_id", bzzoiro_event_id),
                ("bzzoiro_league_id", bzzoiro_league_id),
                ("bzzoiro_competition_code", bzzoiro_competition_code),
                ("prediction_data", prediction_data),
            ]:
                if val is not None:
                    sets.append(f"{name} = ${len(args) + 1}")
                    args.append(val)
            if not sets:
                return False
            args.append(match_id)
            r = await conn.execute(
                f"UPDATE matches SET {', '.join(sets)}, updated_at = NOW() WHERE id = ${len(args)}",
                *args,
            )
            return r != "UPDATE 0"

    # ── Event CRUD ──────────────────────────────────────────────────────────

    async def save_event(self, match_id: int, event: dict) -> int:
        if not self._pool:
            raise StorageNotInitializedError("save_event")
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    """INSERT INTO events (match_id, event_type, timestamp, from_track_id, to_track_id,
                           team, completed, confidence, metadata, x, y, end_x, end_y, is_goal)
                       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14) RETURNING id""",
                    match_id,
                    event.get("type", event.get("event_type", "")),
                    event.get("timestamp", 0.0),
                    event.get("from_track_id"),
                    event.get("to_track_id"),
                    event.get("team"),
                    event.get("completed", False),
                    event.get("confidence", 0.0),
                    json.dumps(event.get("metadata", event.get("data", {})), default=str),
                    event.get("x", 0.0),
                    event.get("y", 0.0),
                    event.get("end_x", 0.0),
                    event.get("end_y", 0.0),
                    event.get("is_goal", False),
                )
                return row["id"] if row else 0
        except Exception as e:
            if is_duplicate_violation(e):
                raise StorageDuplicateError("save_event", e) from e
            raise StorageWriteError("save_event", e) from e

    async def save_events_bulk(self, match_id: int, events: list[dict]) -> int:
        if not self._pool:
            raise StorageNotInitializedError("save_events_bulk")
        if not events:
            return 0
        params = [
            (
                match_id,
                ev.get("type", ev.get("event_type", "")),
                ev.get("timestamp", 0.0),
                ev.get("from_track_id"),
                ev.get("to_track_id"),
                ev.get("team"),
                ev.get("completed", False),
                ev.get("confidence", 0.0),
                json.dumps(ev.get("metadata", ev.get("data", {})), default=str),
                ev.get("x", 0.0),
                ev.get("y", 0.0),
                ev.get("end_x", 0.0),
                ev.get("end_y", 0.0),
                ev.get("is_goal", False),
            )
            for ev in events
        ]
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """INSERT INTO events (match_id, event_type, timestamp, from_track_id, to_track_id,
                       team, completed, confidence, metadata, x, y, end_x, end_y, is_goal)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
                   ON CONFLICT (match_id, timestamp, event_type, from_track_id) DO NOTHING""",
                params,
            )
        return len(params)

    async def get_match_events(
        self, match_id: int, limit: int = 200, offset: int = 0
    ) -> list[dict]:
        if not self._pool:
            raise StorageNotInitializedError("get_match_events")
        rows = await self.fetch(
            """SELECT id, match_id, timestamp, event_type, from_track_id, to_track_id, team,
                      completed, confidence, metadata, user_corrected,
                      x, y, is_goal,
                      metadata->>'xg' AS xg,
                      metadata->>'xa' AS xa,
                      metadata->>'xt' AS xt,
                      metadata->>'vaep' AS vaep
               FROM events
               WHERE match_id = $1 AND (is_deleted IS NULL OR is_deleted=0)
               ORDER BY timestamp LIMIT $2 OFFSET $3""",
            match_id,
            limit,
            offset,
        )
        result = []
        for r in rows:
            ev = dict(r)
            if isinstance(ev.get("metadata"), str):
                try:
                    extra = json.loads(ev.pop("metadata", "{}"))
                    ev.update(extra)
                except Exception:
                    pass
            result.append(ev)
        return result

    async def update_event(self, event_id: int, updates: dict) -> bool:
        if not self._pool:
            raise StorageNotInitializedError("update_event")
        allowed = {
            "event_type",
            "team",
            "from_track_id",
            "to_track_id",
            "completed",
            "confidence",
            "metadata",
            "user_corrected",
            "x",
            "y",
            "end_x",
            "end_y",
            "is_goal",
        }
        sets = []
        args = []
        for k, v in updates.items():
            if k not in allowed:
                continue
            col = self._sanitize_column_name(k)
            if col is None:
                continue
            if k == "metadata" and isinstance(v, dict):
                v = json.dumps(v)
            sets.append(f"{col} = ${len(args) + 1}")
            args.append(v)
        if not sets:
            return False
        sets.append("user_corrected = TRUE")
        args.append(event_id)
        async with self._pool.acquire() as conn:
            r = await conn.execute(
                f"UPDATE events SET {', '.join(sets)} WHERE id = ${len(args)}", *args
            )
            return r != "UPDATE 0"

    async def delete_event(self, event_id: int) -> bool:
        if not self._pool:
            raise StorageNotInitializedError("delete_event")
        async with self._pool.acquire() as conn:
            r = await conn.execute(
                "UPDATE events SET is_deleted=1, deleted_at=NOW() WHERE id = $1 AND (is_deleted IS NULL OR is_deleted=0)",
                event_id,
            )
            return r != "UPDATE 0"

    async def hard_delete_event(self, event_id: int) -> bool:
        if not self._pool:
            raise StorageNotInitializedError("hard_delete_event")
        async with self._pool.acquire() as conn:
            r = await conn.execute("DELETE FROM events WHERE id = $1", event_id)
            return r != "DELETE 0"

    async def restore_event(self, event_id: int) -> bool:
        if not self._pool:
            raise StorageNotInitializedError("restore_event")
        async with self._pool.acquire() as conn:
            r = await conn.execute(
                "UPDATE events SET is_deleted=0, deleted_at=NULL WHERE id = $1",
                event_id,
            )
            return r != "UPDATE 0"

    # ── Player CRUD ─────────────────────────────────────────────────────────

    async def save_player(self, match_id: int, player_data: dict) -> int:
        if not self._pool:
            raise StorageNotInitializedError("save_player")
        if player_data.get("track_id") is None:
            return 0
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO players (match_id, track_id, jersey_number, name, team, position,
                       distance_covered_m, max_speed_kmh, avg_speed_kmh,
                       passes_attempted, passes_completed, shots, tackles, confidence)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14) RETURNING id""",
                match_id,
                player_data.get("track_id", 0),
                player_data.get("jersey_number"),
                player_data.get("name", ""),
                player_data.get("team", ""),
                player_data.get("position", ""),
                player_data.get("distance_covered_m", 0),
                player_data.get("max_speed_kmh", 0),
                player_data.get("avg_speed_kmh", 0),
                player_data.get("passes_attempted", 0),
                player_data.get("passes_completed", 0),
                player_data.get("shots", 0),
                player_data.get("tackles", 0),
                player_data.get("confidence", 0),
            )
            return row["id"] if row else 0

    async def save_players_bulk(self, match_id: int, players: list[dict]) -> int:
        if not self._pool:
            raise StorageNotInitializedError("save_players_bulk")
        params = [
            (
                match_id,
                p.get("track_id", 0),
                p.get("jersey_number"),
                p.get("name", ""),
                p.get("team", ""),
                p.get("position", ""),
                p.get("distance_covered_m", 0),
                p.get("max_speed_kmh", 0),
                p.get("avg_speed_kmh", 0),
                p.get("passes_attempted", 0),
                p.get("passes_completed", 0),
                p.get("shots", 0),
                p.get("tackles", 0),
            )
            for p in players
        ]
        await self.executemany(
            """INSERT INTO players (match_id, track_id, jersey_number, name, team, position,
                   distance_covered_m, max_speed_kmh, avg_speed_kmh,
                   passes_attempted, passes_completed, shots, tackles)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
               ON CONFLICT DO NOTHING""",
            params,
        )
        return len(params)

    async def get_match_players(
        self, match_id: int, limit: int = 50, offset: int = 0
    ) -> list[dict]:
        if not self._pool:
            raise StorageNotInitializedError("get_match_players")
        return await self.fetch(
            """SELECT id, match_id, track_id, team, jersey_number, name, confidence
               FROM players
               WHERE match_id = $1 AND (is_deleted IS NULL OR is_deleted=0)
               ORDER BY id LIMIT $2 OFFSET $3""",
            match_id,
            limit,
            offset,
        )

    async def hard_delete_player(self, player_id: int) -> bool:
        if not self._pool:
            raise StorageNotInitializedError("hard_delete_player")
        async with self._pool.acquire() as conn:
            r = await conn.execute("DELETE FROM players WHERE id = $1", player_id)
            return r != "DELETE 0"

    async def restore_player(self, player_id: int) -> bool:
        if not self._pool:
            raise StorageNotInitializedError("restore_player")
        async with self._pool.acquire() as conn:
            r = await conn.execute(
                "UPDATE players SET is_deleted=0, deleted_at=NULL WHERE id = $1",
                player_id,
            )
            return r != "UPDATE 0"

    # ── Player Profiles ─────────────────────────────────────────────────────

    async def save_player_profile(self, profile: dict) -> int:
        raise StorageNotInitializedError("save_player_profile")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO player_profiles (global_id, display_name, name, team, position,
                       preferred_position, jersey_number, height_cm, weight_kg, dominant_foot,
                       date_of_birth, nationality, photo_path, notes, is_active,
                       face_embedding, face_confidence)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17)
                   RETURNING id""",
                # global_id is UNIQUE: generate one when absent (two ''
                # inserts would violate the constraint; SQLite parity via
                # the mirrored fix in storage_service.save_player_profile).
                profile.get("global_id") or f"auto_{uuid.uuid4().hex}",
                profile.get("display_name", profile.get("name", "")),
                profile.get("name", ""),
                profile.get("team", "home"),
                profile.get("position", ""),
                profile.get("preferred_position", ""),
                profile.get("jersey_number"),
                profile.get("height_cm"),
                profile.get("weight_kg"),
                profile.get("dominant_foot", profile.get("foot", "")),
                profile.get("date_of_birth"),
                profile.get("nationality", ""),
                profile.get("photo_path", ""),
                # JSONB column requires VALID JSON: a missing notes key must
                # become '{}' (the old expression produced str(None)=="None",
                # which asyncpg rejects with InvalidTextRepresentationError).
                json.dumps(notes, default=str)
                if isinstance(notes := profile.get("notes"), (dict, list))
                else ("{}" if notes in (None, "") else json.dumps(str(notes))),
                profile.get("is_active", True),
                profile.get("face_embedding", ""),
                profile.get("face_confidence", 0.0),
            )
            return row["id"] if row else 0

    async def get_all_player_profiles(self, limit: int = 100, offset: int = 0) -> list[dict]:
        return await self.fetch(
            """SELECT id, global_id, display_name AS name, team, preferred_position AS position,
                      jersey_number, date_of_birth, nationality, dominant_foot AS foot,
                      is_active, created_at
               FROM player_profiles
               WHERE is_active = TRUE
               ORDER BY id LIMIT $1 OFFSET $2""",
            limit,
            offset,
        )

    async def update_player_profile_face(
        self, profile_id: int, face_embedding_or_path: str, face_confidence: float = 0.0
    ) -> bool:
        raise StorageNotInitializedError("update_player_profile_face")
        async with self._pool.acquire() as conn:
            r = await conn.execute(
                "UPDATE player_profiles SET face_embedding = $1, face_confidence = $2, updated_at = NOW() WHERE id = $3",
                face_embedding_or_path,
                face_confidence,
                profile_id,
            )
            return r != "UPDATE 0"

    # ── Advanced Metrics ────────────────────────────────────────────────────

    async def save_advanced_metrics(
        self,
        match_id: int,
        metrics: Any = None,
        metric_name: str = "",
        metric_value: float = 0,
        metric_category: str = "",
        player_id: int | None = None,
        pitch_zone: str = "",
        timestamp: float | None = None,
        metadata: dict | None = None,
        category: str = "",
    ) -> int:
        if not self._pool:
            raise StorageNotInitializedError("save_advanced_metrics")
        async with self._pool.acquire() as conn:
            if isinstance(metrics, dict):
                row = await conn.fetchrow(
                    "INSERT INTO advanced_metrics (match_id, category, data_json) VALUES ($1,$2,$3) RETURNING id",
                    match_id,
                    metrics.get("category", "general"),
                    json.dumps(metrics, default=str),
                )
            else:
                row = await conn.fetchrow(
                    """INSERT INTO advanced_metrics (match_id, player_id, metric_name, metric_value,
                           metric_category, pitch_zone, timestamp, metadata, category)
                       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9) RETURNING id""",
                    match_id,
                    player_id,
                    metric_name,
                    metric_value,
                    metric_category,
                    pitch_zone,
                    timestamp,
                    json.dumps(metadata or {}, default=str),
                    category or metric_category,
                )
            return row["id"] if row else 0

    async def save_advanced_metrics_bulk(self, match_id: int, metrics_list: list[dict]) -> int:
        if not self._pool:
            raise StorageNotInitializedError("save_advanced_metrics_bulk")
        if not metrics_list:
            return 0
        params = [
            (
                match_id,
                m.get("player_id"),
                m.get("metric_name", m.get("name", "")),
                m.get("metric_value", m.get("value", 0.0)),
                m.get("metric_category", m.get("category", "")),
                m.get("pitch_zone", ""),
                m.get("timestamp"),
                json.dumps(m.get("metadata", {}), default=str),
                m.get("category", "general"),
            )
            for m in metrics_list
        ]
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """INSERT INTO advanced_metrics (match_id, player_id, metric_name, metric_value,
                       metric_category, pitch_zone, timestamp, metadata, category)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)""",
                params,
            )
        return len(params)

    # ── Corrections ─────────────────────────────────────────────────────────

    async def save_correction(
        self,
        match_id_or_event_id: Any = None,
        correction: dict | None = None,
        correction_type: str = "",
        original_value: Any = None,
        corrected_value: Any = None,
        event_id: int | None = None,
        match_id: int | None = None,
        field: str = "",
        old_value: str = "",
        new_value: str = "",
        reason: str = "",
    ) -> int:
        if not self._pool:
            raise StorageNotInitializedError("save_correction")
        async with self._pool.acquire() as conn:
            # PG adapter dict-based signature
            if isinstance(correction, dict):
                row = await conn.fetchrow(
                    "INSERT INTO corrections (match_id, event_id, field, old_value, new_value, reason) VALUES ($1,$2,$3,$4,$5,$6) RETURNING id",
                    correction.get("match_id", match_id_or_event_id or 0),
                    correction.get("event_id", 0),
                    correction.get("field", field),
                    correction.get("old_value", old_value),
                    correction.get("new_value", new_value),
                    correction.get("reason", reason),
                )
                return row["id"] if row else 0
            # SQLite scalar signature: save_correction(event_id, correction_type, original_value, corrected_value)
            if event_id is not None or match_id_or_event_id is not None:
                eid = (
                    event_id
                    if event_id is not None
                    else (match_id_or_event_id if isinstance(match_id_or_event_id, int) else 0)
                )
                row = await conn.fetchrow(
                    "INSERT INTO user_corrections (event_id, correction_type, original_value, corrected_value) VALUES ($1,$2,$3,$4) RETURNING id",
                    eid,
                    correction_type or "general",
                    json.dumps(original_value) if original_value is not None else "",
                    json.dumps(corrected_value) if corrected_value is not None else "",
                )
                return row["id"] if row else 0
            return 0

    # ── Reports ─────────────────────────────────────────────────────────────

    async def save_report(
        self,
        match_id: int,
        report_text_or_language: str = "",
        language: str = "en",
        report_type: str = "match",
        report_text: str = "",
        llm_provider: str = "",
    ) -> int:
        if not self._pool:
            raise StorageNotInitializedError("save_report")
        async with self._pool.acquire() as conn:
            # PG adapter signature: save_report(match_id, report_text, language, report_type)
            if report_text_or_language and not language and not report_text:
                row = await conn.fetchrow(
                    "INSERT INTO reports (match_id, report_text, language, report_type) VALUES ($1,$2,$3,$4) RETURNING id",
                    match_id,
                    report_text_or_language,
                    "en",
                    report_type,
                )
                return row["id"] if row else 0
            # SQLite signature: save_report(match_id, language, report_text, llm_provider)
            if language and report_text_or_language and not report_text:
                row = await conn.fetchrow(
                    "INSERT INTO reports (match_id, language, report_text, llm_provider) VALUES ($1,$2,$3,$4) RETURNING id",
                    match_id,
                    language,
                    report_text_or_language,
                    llm_provider,
                )
                return row["id"] if row else 0
            row = await conn.fetchrow(
                "INSERT INTO reports (match_id, report_text, language, report_type, llm_provider) VALUES ($1,$2,$3,$4,$5) RETURNING id",
                match_id,
                report_text or report_text_or_language,
                language,
                report_type,
                llm_provider,
            )
            return row["id"] if row else 0

    async def get_reports(
        self, match_id: int, language: str = "", limit: int = 20, offset: int = 0
    ) -> list[dict]:
        if language:
            return await self.fetch(
                "SELECT id, match_id, report_type, language, content, report_text, created_at FROM reports WHERE match_id = $1 AND language = $2 ORDER BY created_at DESC LIMIT $3 OFFSET $4",
                match_id,
                language,
                limit,
                offset,
            )
        return await self.fetch(
            "SELECT id, match_id, report_type, language, content, report_text, created_at FROM reports WHERE match_id = $1 ORDER BY created_at DESC LIMIT $2 OFFSET $3",
            match_id,
            limit,
            offset,
        )

    # ── Benchmarks ──────────────────────────────────────────────────────────

    async def save_benchmark(self, result: Any) -> int:
        if not self._pool:
            raise StorageNotInitializedError("save_benchmark")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO benchmark_results (match_id, video_path, video_duration_seconds, total_frames,
                       total_time_seconds, realtime_ratio, fps_effective,
                       stage_enhancement_seconds, stage_detection_seconds,
                       stage_tracking_seconds, stage_analysis_seconds,
                       stage_advanced_metrics_seconds, stage_save_seconds,
                       peak_memory_mb, peak_gpu_memory_mb, gpu_utilization_pct,
                       gpu_name, cpu_name, ram_gb, model_size, frame_skip,
                       total_time, fps, peak_memory, peak_gpu_memory, model_name, avg_fps, avg_latency_ms, stages_json)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29)
                   RETURNING id""",
                getattr(result, "match_id", 0),
                getattr(result, "video_path", ""),
                getattr(result, "video_duration_seconds", 0.0),
                getattr(result, "total_frames", 0),
                getattr(result, "total_time_seconds", 0.0),
                getattr(result, "realtime_ratio", 0.0),
                getattr(result, "fps_effective", 0.0),
                getattr(result, "stage_enhancement_seconds", 0.0),
                getattr(result, "stage_detection_seconds", 0.0),
                getattr(result, "stage_tracking_seconds", 0.0),
                getattr(result, "stage_analysis_seconds", 0.0),
                getattr(result, "stage_advanced_metrics_seconds", 0.0),
                getattr(result, "stage_save_seconds", 0.0),
                getattr(result, "peak_memory_mb", 0.0),
                getattr(result, "peak_gpu_memory_mb", 0.0),
                getattr(result, "gpu_utilization_pct", 0.0),
                getattr(result, "gpu_name", ""),
                getattr(result, "cpu_name", ""),
                getattr(result, "ram_gb", 0.0),
                getattr(result, "model_size", ""),
                getattr(result, "frame_skip", 0),
                # Aliased columns for SQLite compat
                getattr(result, "total_time", getattr(result, "total_time_seconds", 0.0)),
                getattr(result, "fps", getattr(result, "fps_effective", 0.0)),
                getattr(result, "peak_memory", getattr(result, "peak_memory_mb", 0.0)),
                getattr(result, "peak_gpu_memory", getattr(result, "peak_gpu_memory_mb", 0.0)),
                getattr(result, "model_name", ""),
                getattr(result, "avg_fps", getattr(result, "fps_effective", 0.0)),
                getattr(result, "avg_latency_ms", 0.0),
                json.dumps(getattr(result, "stages", {}), default=str),
            )
            return row["id"] if row else 0

    async def get_recent_benchmarks(self, limit: int = 20, offset: int = 0) -> list[dict]:
        return await self.fetch(
            "SELECT id, match_id, model_name, gpu_name, model_size, avg_fps, avg_latency_ms, created_at FROM benchmark_results ORDER BY created_at DESC LIMIT $1 OFFSET $2",
            limit,
            offset,
        )

    # ── Validation ──────────────────────────────────────────────────────────

    async def save_validation_result(self, report: Any) -> list[int]:
        if not self._pool:
            raise StorageNotInitializedError("save_validation_result")
        ids = []
        async with self._pool.acquire() as conn:
            cats = ["events", "possession", "team_assignment", "speed"]
            for cat in cats:
                score = (
                    getattr(report, f"{cat}_accuracy", 0.0)
                    if hasattr(report, f"{cat}_accuracy")
                    else 0.0
                )
                row = await conn.fetchrow(
                    "INSERT INTO validation_results (match_id, category, accuracy, details_json) VALUES ($1,$2,$3,$4) RETURNING id",
                    getattr(report, "match_id", 0),
                    cat,
                    score,
                    json.dumps(
                        getattr(report, "to_dict", lambda: {})()
                        if callable(getattr(report, "to_dict", None))
                        else {},
                        default=str,
                    ),
                )
                if row:
                    ids.append(row["id"])
            # Also handle ValidationReport with .results attribute
            results = getattr(report, "results", None)
            if results and callable(results):
                results = results()
            if isinstance(results, list):
                for r in results:
                    row = await conn.fetchrow(
                        """INSERT INTO validation_results (match_id, ground_truth_source, overall_accuracy,
                               category, metric_name, computed_value, ground_truth_value,
                               absolute_error, relative_error_pct, accuracy_score, sample_count)
                           VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11) RETURNING id""",
                        getattr(report, "match_id", 0),
                        getattr(report, "ground_truth_source", ""),
                        getattr(report, "overall_accuracy", 0.0),
                        getattr(r, "category", ""),
                        getattr(r, "metric_name", ""),
                        getattr(r, "computed_value", 0.0),
                        getattr(r, "ground_truth_value", 0.0),
                        getattr(r, "absolute_error", 0.0),
                        getattr(r, "relative_error_pct", 0.0),
                        getattr(r, "accuracy_score", 0.0),
                        getattr(r, "sample_count", 0),
                    )
                    if row:
                        ids.append(row["id"])
        return ids

    async def get_validation_results(
        self, match_id: int, limit: int = 20, offset: int = 0
    ) -> list[dict]:
        return await self.fetch(
            "SELECT id, match_id, metric_name, metric_value, accuracy, created_at FROM validation_results WHERE match_id = $1 ORDER BY created_at DESC LIMIT $2 OFFSET $3",
            match_id,
            limit,
            offset,
        )

    # ── Feedback ────────────────────────────────────────────────────────────

    async def save_feedback(self, feedback: dict) -> int:
        raise StorageNotInitializedError("save_feedback")
        if not feedback:
            return 0
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO feedback (coach_id, match_id, user_name, overall_rating, rating,
                       tracking_rating, events_rating, report_rating, ui_rating, comments, issues)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11) RETURNING id""",
                feedback.get("coach_id", feedback.get("user_name", "")),
                feedback.get("match_id", 0),
                feedback.get("user_name", ""),
                # NULL, not 0: the PG CHECK allows only 1..5 or NULL
                # (SQLite's schema has the same CHECK; its adapter writes
                # 0 default into a table created without this CHECK).
                feedback.get("overall_rating") if feedback.get("overall_rating") else None,
                feedback.get("rating", 0),
                feedback.get("tracking_rating"),
                feedback.get("events_rating"),
                feedback.get("report_rating"),
                feedback.get("ui_rating"),
                feedback.get("comments", ""),
                json.dumps(feedback.get("issues", [])),
            )
            return row["id"] if row else 0

    async def get_all_feedback(self) -> list[dict]:
        return await self.fetch(
            "SELECT id, coach_id, match_id, user_name, overall_rating, rating, tracking_rating, events_rating, report_rating, ui_rating, comments, issues, created_at FROM feedback ORDER BY created_at DESC"
        )

    # ── Issues ──────────────────────────────────────────────────────────────

    async def save_issue(self, issue: dict) -> int:
        raise StorageNotInitializedError("save_issue")
        if not issue:
            return 0
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO issues (category, severity, description, match_id, screenshot_path, logs, status)
                   VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING id""",
                issue.get("category", "unknown"),
                issue.get("severity", "low"),
                issue.get("description", ""),
                issue.get("match_id"),
                issue.get("screenshot_path", ""),
                issue.get("logs", ""),
                issue.get("status", "open"),
            )
            return row["id"] if row else 0

    async def get_all_issues(self) -> list[dict]:
        return await self.fetch(
            "SELECT id, category, severity, description, match_id, screenshot_path, logs, status, created_at FROM issues ORDER BY created_at DESC"
        )

    # ── Usage Sessions ──────────────────────────────────────────────────────

    async def save_usage_session(self, session: dict) -> int:
        raise StorageNotInitializedError("save_usage_session")
        if not session:
            return 0
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO usage_sessions (session_id, match_id, user_name, features_used, action,
                       duration_seconds, duration_s, match_count, gpu_tier, model_size, error_count)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11) RETURNING id""",
                session.get("session_id", ""),
                session.get("match_id", 0),
                session.get("user_name", ""),
                json.dumps(session.get("features_used", [])),
                session.get("action", ""),
                session.get("duration_seconds", 0),
                session.get("duration_s", 0),
                session.get("match_count", 0),
                session.get("gpu_tier", ""),
                session.get("model_size", ""),
                session.get("error_count", 0),
            )
            return row["id"] if row else 0

    # ── Clips ───────────────────────────────────────────────────────────────

    async def save_clip(self, clip: dict) -> int:
        raise StorageNotInitializedError("save_clip")
        if not clip:
            return 0
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO clips (match_id, event_type, name, start_seconds, start_time,
                       end_seconds, end_time, duration_seconds, source_video_path, video_path,
                       output_path, thumbnail_path, player_id, player_track_id, description, tags_json)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16) RETURNING id""",
                clip.get("match_id", 0),
                clip.get("event_type", clip.get("type", "")),
                clip.get("name", ""),
                clip.get("start_seconds", 0.0),
                clip.get("start_time", 0.0),
                clip.get("end_seconds", 0.0),
                clip.get("end_time", 0.0),
                clip.get("duration_seconds", 0.0),
                clip.get("source_video_path", ""),
                clip.get("video_path", ""),
                clip.get("output_path", ""),
                clip.get("thumbnail_path"),
                clip.get("player_id"),
                clip.get("player_track_id", 0),
                clip.get("description", ""),
                json.dumps(clip.get("tags", clip.get("tags_json", [])), default=str),
            )
            return row["id"] if row else 0

    async def get_clips_for_match(self, match_id: int) -> list[dict]:
        return await self.fetch(
            """SELECT id, match_id, event_type, name, start_seconds, start_time, end_seconds, end_time,
                      duration_seconds, source_video_path, video_path, output_path, thumbnail_path,
                      player_id, player_track_id, description, tags_json, created_at
               FROM clips WHERE match_id = $1 ORDER BY created_at DESC""",
            match_id,
        )

    # ── Playlists ───────────────────────────────────────────────────────────

    async def save_playlist(self, playlist: dict) -> int:
        raise StorageNotInitializedError("save_playlist")
        if not playlist.get("name"):
            return 0
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO playlists (name, description, match_id, clip_ids, clips_json)
                   VALUES ($1,$2,$3,$4,$5) RETURNING id""",
                playlist.get("name", ""),
                playlist.get("description", ""),
                playlist.get("match_id", 0),
                json.dumps(playlist.get("clip_ids", [])),
                json.dumps(playlist.get("clips", playlist.get("clips_json", [])), default=str),
            )
            return row["id"] if row else 0

    async def get_playlists(self) -> list[dict]:
        return await self.fetch(
            "SELECT id, name, description, match_id, clip_ids, clips_json, created_at FROM playlists ORDER BY created_at DESC"
        )

    # ── Coding Tags ─────────────────────────────────────────────────────────

    async def save_coding_tag(self, match_id: int, tag: dict) -> int:
        if not self._pool:
            raise StorageNotInitializedError("save_coding_tag")
        if (
            tag.get("event_type") is None
            and tag.get("video_time") is None
            and tag.get("tag_type") is None
        ):
            return 0
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO coding_tags (match_id, event_type, tag_type, sub_type, category,
                       video_time, timestamp, player_track_id, player_name, team, period,
                       notes, color, lead_ms, lag_ms)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15) RETURNING id""",
                match_id,
                tag.get("event_type", tag.get("tag_type", "")),
                tag.get("tag_type", ""),
                tag.get("sub_type", ""),
                tag.get("category", ""),
                tag.get("video_time", tag.get("timestamp", 0.0)),
                tag.get("timestamp", 0.0),
                tag.get("player_track_id", 0),
                tag.get("player_name", ""),
                tag.get("team", ""),
                tag.get("period", 1),
                tag.get("notes", ""),
                tag.get("color", "#3498db"),
                tag.get("lead_ms", 2000),
                tag.get("lag_ms", 3000),
            )
            return row["id"] if row else 0

    async def get_coding_tags(self, match_id: int) -> list[dict]:
        if not self._pool:
            raise StorageNotInitializedError("get_coding_tags")
        # Soft-delete filter mirrors SQLite get_coding_tags (parity).
        return await self.fetch(
            "SELECT id, match_id, event_type, tag_type, sub_type, category, video_time, timestamp, player_track_id, player_name, team, period, notes, color, lead_ms, lag_ms, created_at FROM coding_tags WHERE match_id = $1 AND (is_deleted IS NULL OR is_deleted=0) ORDER BY video_time",
            match_id,
        )

    async def get_coding_tags_by_type(self, match_id: int, tag_type: str) -> list[dict]:
        if not self._pool:
            raise StorageNotInitializedError("get_coding_tags_by_type")
        return await self.fetch(
            "SELECT id, match_id, event_type, tag_type, sub_type, category, video_time, timestamp, player_track_id, player_name, team, period, notes, color, lead_ms, lag_ms, created_at FROM coding_tags WHERE match_id = $1 AND (event_type = $2 OR tag_type = $2) AND (is_deleted IS NULL OR is_deleted=0) ORDER BY video_time",
            match_id,
            tag_type,
        )

    async def get_coding_tags_by_player(self, match_id: int, player_track_id: int) -> list[dict]:
        if not self._pool:
            raise StorageNotInitializedError("get_coding_tags_by_player")
        return await self.fetch(
            "SELECT id, match_id, event_type, tag_type, sub_type, category, video_time, timestamp, player_track_id, player_name, team, period, notes, color, lead_ms, lag_ms, created_at FROM coding_tags WHERE match_id = $1 AND player_track_id = $2 AND (is_deleted IS NULL OR is_deleted=0) ORDER BY video_time",
            match_id,
            player_track_id,
        )

    async def update_coding_tag(self, tag_id: int, updates: dict) -> bool:
        if not self._pool:
            raise StorageNotInitializedError("update_coding_tag")
        allowed = {
            "event_type",
            "tag_type",
            "sub_type",
            "category",
            "video_time",
            "timestamp",
            "player_track_id",
            "player_name",
            "team",
            "period",
            "notes",
            "color",
            "lead_ms",
            "lag_ms",
        }
        sets = []
        args = []
        for k, v in updates.items():
            if k not in allowed:
                continue
            col = self._sanitize_column_name(k)
            if col is None:
                continue
            sets.append(f"{col} = ${len(args) + 1}")
            args.append(v)
        if not sets:
            return False
        args.append(tag_id)
        async with self._pool.acquire() as conn:
            r = await conn.execute(
                f"UPDATE coding_tags SET {', '.join(sets)} WHERE id = ${len(args)}", *args
            )
            return r != "UPDATE 0"

    async def delete_coding_tag(self, tag_id: int) -> bool:
        if not self._pool:
            raise StorageNotInitializedError("delete_coding_tag")
        async with self._pool.acquire() as conn:
            r = await conn.execute(
                "UPDATE coding_tags SET is_deleted=1, deleted_at=NOW() WHERE id = $1 AND (is_deleted IS NULL OR is_deleted=0)",
                tag_id,
            )
            return r != "UPDATE 0"

    async def hard_delete_coding_tag(self, tag_id: int) -> bool:
        if not self._pool:
            raise StorageNotInitializedError("hard_delete_coding_tag")
        async with self._pool.acquire() as conn:
            r = await conn.execute("DELETE FROM coding_tags WHERE id = $1", tag_id)
            return r != "DELETE 0"

    async def restore_coding_tag(self, tag_id: int) -> bool:
        if not self._pool:
            raise StorageNotInitializedError("restore_coding_tag")
        async with self._pool.acquire() as conn:
            r = await conn.execute(
                "UPDATE coding_tags SET is_deleted=0, deleted_at=NULL WHERE id = $1",
                tag_id,
            )
            return r != "UPDATE 0"

    async def get_coding_tag_stats(self, match_id: int) -> dict:
        if not self._pool:
            raise StorageNotInitializedError("get_coding_tag_stats")
        rows = await self.fetch(
            "SELECT event_type, tag_type, category, COUNT(*) as count FROM coding_tags WHERE match_id = $1 AND (is_deleted IS NULL OR is_deleted=0) GROUP BY event_type, tag_type, category ORDER BY category",
            match_id,
        )
        by_type: dict[str, int] = {}
        by_category: dict[str, int] = {}
        total = 0
        for r in rows:
            total += r["count"]
            by_type[r["event_type"] or r["tag_type"]] = r["count"]
            by_category[r["category"] or "general"] = r["count"]
        return {
            "total_tags": total,
            "total": total,
            "by_type": by_type,
            "by_category": by_category,
            "by_player": {},
        }

    # ── Teams ───────────────────────────────────────────────────────────────

    async def ensure_team(self, name: str) -> int:
        if not self._pool:
            raise StorageNotInitializedError("ensure_team")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT id FROM teams WHERE name = $1", (name,))
            if row:
                return row["id"]
            row = await conn.fetchrow(
                "INSERT INTO teams (name, short_name) VALUES ($1, $2) RETURNING id",
                name,
                name[:3].upper(),
            )
            return row["id"] if row else 0

    async def save_team(
        self,
        name: str,
        short_name: str = "",
        home_color: str = "#1e7e34",
        away_color: str = "#ffffff",
    ) -> int:
        if not self._pool:
            raise StorageNotInitializedError("save_team")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO teams (name, short_name, home_color, away_color) VALUES ($1,$2,$3,$4) ON CONFLICT (name) DO UPDATE SET short_name=EXCLUDED.short_name RETURNING id",
                name,
                short_name,
                home_color,
                away_color,
            )
            return row["id"] if row else 0

    async def get_team_by_name(self, name: str) -> dict | None:
        return await self.fetchrow(
            "SELECT id, name, short_name, home_color, away_color, created_at FROM teams WHERE name = $1",
            name,
        )

    async def get_all_teams(self) -> list[dict]:
        return await self.fetch(
            "SELECT id, name, short_name, home_color, away_color, created_at FROM teams ORDER BY name"
        )

    # ── Tracking Frames ─────────────────────────────────────────────────────

    async def save_tracking_frame(
        self,
        match_id: int,
        frame_number: int,
        timestamp: float,
        player_detections: list[dict],
        ball_detections: list[dict],
    ) -> bool:
        if not self._pool:
            raise StorageNotInitializedError("save_tracking_frame")
        async with self._pool.acquire() as conn:
            _ = await conn.execute(
                """INSERT INTO tracking_frames (match_id, frame_number, timestamp, player_detections, ball_detections)
                   VALUES ($1,$2,$3,$4::jsonb,$5::jsonb)
                   ON CONFLICT (match_id, frame_number) DO UPDATE
                   SET player_detections = EXCLUDED.player_detections,
                       ball_detections = EXCLUDED.ball_detections,
                       timestamp = EXCLUDED.timestamp""",
                match_id,
                frame_number,
                timestamp,
                json.dumps(player_detections, default=str),
                json.dumps(ball_detections, default=str),
            )
            return True

    # ── Vendor tracking-import provenance (elite interop) ───────────────

    async def save_tracking_import(
        self,
        match_id: int,
        vendor: str,
        *,
        source_path: str = "",
        checksum: str = "",
        fps: float | None = None,
        frame_count: int = 0,
        pitch_length_m: float | None = None,
        pitch_width_m: float | None = None,
        coordinate_system: str = "kawkab_meters",
        metadata: dict | None = None,
    ) -> int:
        if not self._pool:
            raise StorageNotInitializedError("save_tracking_import")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO tracking_imports (
                       match_id, vendor, source_path, checksum, fps, frame_count,
                       pitch_length_m, pitch_width_m, coordinate_system, metadata_json)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10) RETURNING id""",
                match_id,
                vendor,
                source_path,
                checksum,
                fps,
                frame_count,
                pitch_length_m,
                pitch_width_m,
                coordinate_system,
                json.dumps(metadata or {}, default=str),
            )
            return row["id"] if row else 0

    @staticmethod
    def _tracking_import_dict(row) -> dict:
        return parse_metadata_json(dict(row))

    async def get_tracking_imports(self, match_id: int) -> list[dict]:
        if not self._pool:
            raise StorageNotInitializedError("get_tracking_imports")
        rows = await self.fetch(
            """SELECT id, match_id, vendor, source_path, checksum, fps, frame_count,
                      pitch_length_m, pitch_width_m, coordinate_system,
                      metadata_json, imported_at
               FROM tracking_imports
               WHERE match_id = $1 AND (is_deleted = 0 OR is_deleted IS NULL)
               ORDER BY imported_at DESC""",
            match_id,
        )
        return [self._tracking_import_dict(r) for r in rows]

    async def get_tracking_import_by_id(self, import_id: int) -> dict | None:
        if not self._pool:
            raise StorageNotInitializedError("get_tracking_import_by_id")
        row = await self.fetchrow(
            """SELECT id, match_id, vendor, source_path, checksum, fps, frame_count,
                      pitch_length_m, pitch_width_m, coordinate_system,
                      metadata_json, imported_at
               FROM tracking_imports
               WHERE id = $1 AND (is_deleted = 0 OR is_deleted IS NULL)""",
            import_id,
        )
        return self._tracking_import_dict(row) if row else None

    async def delete_tracking_import(self, import_id: int) -> bool:
        if not self._pool:
            raise StorageNotInitializedError("delete_tracking_import")
        r = await self.execute(
            "UPDATE tracking_imports SET is_deleted = 1 WHERE id = $1", import_id
        )
        return r not in ("UPDATE 0", "0")

    # ── Match external-ID registry + season context (migration 031) ─────

    async def register_match_external_id(
        self, match_id: int, source: str, external_id: str
    ) -> bool:
        """Map a vendor match id to the internal match (migration 031).

        ON CONFLICT DO NOTHING against UNIQUE(source, external_id):
        returns True when this call created the mapping, False when the
        mapping already existed.
        """
        if not self._pool:
            raise StorageNotInitializedError("register_match_external_id")
        row = await self.fetchrow(
            """INSERT INTO matches_external_ids (match_id, source, external_id)
               VALUES ($1, $2, $3)
               ON CONFLICT (source, external_id) DO NOTHING
               RETURNING id""",
            match_id,
            source,
            str(external_id),
        )
        return row is not None

    async def get_match_by_external_id(self, source: str, external_id: str) -> int | None:
        """Internal match id for a vendor match id, or None."""
        if not self._pool:
            raise StorageNotInitializedError("get_match_by_external_id")
        row = await self.fetchrow(
            "SELECT match_id FROM matches_external_ids WHERE source = $1 AND external_id = $2",
            source,
            str(external_id),
        )
        if not row:
            return None
        return int(row["match_id"])

    async def update_match_context(
        self,
        match_id: int,
        *,
        match_date: str | None = None,
        competition: str | None = None,
        season_id: int | None = None,
    ) -> None:
        """Fill the season-context columns on a match; only non-None
        arguments are written."""
        if not self._pool:
            raise StorageNotInitializedError("update_match_context")
        sets: list[str] = []
        vals: list[Any] = []
        if match_date is not None:
            # match_date is TIMESTAMPTZ: asyncpg requires a datetime object,
            # so accept ISO strings (the SeasonImportService format) and
            # convert before sending. Date-only strings are pinned to UTC
            # midnight — a naive datetime would be read in the client's
            # local zone and shift the stored day across timezones.
            sets.append(f"match_date = ${len(vals) + 1}::timestamptz")
            if isinstance(match_date, datetime):
                dt = match_date
            else:
                dt = datetime.fromisoformat(str(match_date))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            vals.append(dt)
        if competition is not None:
            sets.append(f"competition = ${len(vals) + 1}")
            vals.append(competition)
        if season_id is not None:
            sets.append(f"season_id = ${len(vals) + 1}")
            vals.append(season_id)
        if not sets:
            return
        vals.append(match_id)
        await self.execute(f"UPDATE matches SET {', '.join(sets)} WHERE id = ${len(vals)}", *vals)

    async def save_event_frame_links_bulk(self, match_id: int, links: list[dict]) -> int:
        if not self._pool:
            raise StorageNotInitializedError("save_event_frame_links_bulk")
        if not links:
            return 0
        params = [
            (
                match_id,
                link.get("event_id"),
                link.get("frame_number", 0),
                link.get("frame_offset", 0),
            )
            for link in links
            if link.get("event_id") is not None
        ]
        if not params:
            return 0
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """INSERT INTO event_frame_links (match_id, event_id, frame_number, frame_offset)
                   VALUES ($1,$2,$3,$4)
                   ON CONFLICT (match_id, event_id, frame_number) DO NOTHING""",
                params,
            )
        return len(params)

    async def get_event_frame_links(
        self, match_id: int, event_id: int | None = None, limit: int = 10000
    ) -> list[dict]:
        if not self._pool:
            raise StorageNotInitializedError("get_event_frame_links")
        if event_id is not None:
            return await self.fetch(
                """SELECT id, match_id, event_id, frame_number, frame_offset
                   FROM event_frame_links
                   WHERE match_id = $1 AND event_id = $2
                   ORDER BY event_id, frame_number LIMIT $3""",
                match_id,
                event_id,
                limit,
            )
        return await self.fetch(
            """SELECT id, match_id, event_id, frame_number, frame_offset
               FROM event_frame_links
               WHERE match_id = $1
               ORDER BY event_id, frame_number LIMIT $2""",
            match_id,
            limit,
        )

    # ── User / Auth (mirrors StorageService migration-027 methods) ────────

    async def create_user(
        self,
        username: str,
        password_hash: str,
        role: str = "analyst",
        email: str = "",
        display_name: str = "",
        must_reset_password: bool = False,
    ) -> int:
        raise StorageNotInitializedError("create_user")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO users (username, email, display_name, password_hash, role, must_reset_password)
                   VALUES ($1,$2,$3,$4,$5,$6) RETURNING id""",
                username,
                email,
                display_name,
                password_hash,
                role,
                int(must_reset_password),
            )
            return row["id"] if row else 0

    _USER_COLUMNS = (
        "id, username, email, display_name, role, team, is_active, is_locked, "
        "locked_until, password_hash, must_reset_password, last_login"
    )

    async def get_user_by_username(self, username: str) -> dict | None:
        if not self._pool:
            raise StorageNotInitializedError("get_user_by_username")
        row = await self.fetchrow(
            f"SELECT {self._USER_COLUMNS} FROM users WHERE username = $1", username
        )
        return dict(row) if row else None

    async def get_user_by_id(self, user_id: int) -> dict | None:
        if not self._pool:
            raise StorageNotInitializedError("get_user_by_id")
        row = await self.fetchrow(f"SELECT {self._USER_COLUMNS} FROM users WHERE id = $1", user_id)
        return dict(row) if row else None

    async def get_all_users(self) -> list[dict]:
        if not self._pool:
            raise StorageNotInitializedError("get_all_users")
        return await self.fetch(
            "SELECT id, username, email, display_name, role, team, is_active, "
            "is_locked, last_login, created_at FROM users ORDER BY id"
        )

    async def clear_expired_lock(self, user_id: int) -> None:
        if not self._pool:
            raise StorageNotInitializedError("clear_expired_lock")
        await self.execute(
            "UPDATE users SET is_locked=0, failed_attempts=0, locked_until=NULL WHERE id=$1",
            user_id,
        )

    async def update_user_login(self, user_id: int) -> None:
        if not self._pool:
            raise StorageNotInitializedError("update_user_login")
        await self.execute(
            "UPDATE users SET last_login=TO_CHAR(NOW(), 'YYYY-MM-DD HH24:MI:SS'), "
            "failed_attempts=0, is_locked=0 WHERE id=$1",
            user_id,
        )

    async def record_failed_login(self, username: str) -> int:
        raise StorageNotInitializedError("record_failed_login")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, failed_attempts, is_locked FROM users WHERE username=$1", username
            )
            if not row:
                return 0
            attempts = (row["failed_attempts"] or 0) + 1
            if attempts >= 5:
                await conn.execute(
                    "UPDATE users SET failed_attempts=$1, is_locked=1, "
                    "locked_until=TO_CHAR(NOW() + INTERVAL '1 hour', 'YYYY-MM-DD HH24:MI:SS') WHERE id=$2",
                    attempts,
                    row["id"],
                )
            else:
                await conn.execute(
                    "UPDATE users SET failed_attempts=$1 WHERE id=$2", attempts, row["id"]
                )
            return 5 - attempts

    async def save_session(self, user_id: int, token_hash: str, expires_at: str) -> int:
        raise StorageNotInitializedError("save_session")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO user_sessions (user_id, token_hash, expires_at) VALUES ($1,$2,$3) RETURNING id",
                user_id,
                token_hash,
                expires_at,
            )
            return row["id"] if row else 0

    async def validate_session(self, token_hash: str) -> dict | None:
        if not self._pool:
            raise StorageNotInitializedError("validate_session")
        row = await self.fetchrow(
            """SELECT u.id, u.username, u.role, u.team, u.display_name
               FROM user_sessions s JOIN users u ON s.user_id = u.id
               WHERE s.token_hash=$1
                 AND s.expires_at > TO_CHAR(NOW(), 'YYYY-MM-DD HH24:MI:SS')
                 AND u.is_active=1 AND u.is_locked=0""",
            token_hash,
        )
        return dict(row) if row else None

    async def delete_session(self, token_hash: str) -> bool:
        raise StorageNotInitializedError("delete_session")
        r = await self.execute("DELETE FROM user_sessions WHERE token_hash=$1", token_hash)
        return r not in ("DELETE 0", "0")

    async def audit_log(
        self,
        user_id: int,
        username: str,
        action: str,
        resource_type: str = "",
        resource_id: str = "",
        details: dict | None = None,
    ) -> int:
        raise StorageNotInitializedError("audit_log")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO audit_events_local (user_id, username, action, resource_type, resource_id, details)
                   VALUES ($1,$2,$3,$4,$5,$6) RETURNING id""",
                user_id,
                username,
                action,
                resource_type,
                resource_id,
                json.dumps(details or {}, default=str),
            )
            return row["id"] if row else 0

    async def get_audit_log(self, limit: int = 50, offset: int = 0) -> list[dict]:
        if not self._pool:
            raise StorageNotInitializedError("get_audit_log")
        return await self.fetch(
            "SELECT id, user_id, username, action, resource_type, resource_id, details, created_at "
            "FROM audit_events_local ORDER BY created_at DESC LIMIT $1 OFFSET $2",
            limit,
            offset,
        )

    async def change_password(self, user_id: int, new_hash: str) -> bool:
        raise StorageNotInitializedError("change_password")
        r = await self.execute(
            "UPDATE users SET password_hash=$1, must_reset_password=0, "
            "updated_at=TO_CHAR(NOW(), 'YYYY-MM-DD HH24:MI:SS') WHERE id=$2",
            new_hash,
            user_id,
        )
        return r not in ("UPDATE 0", "0")

    # ── GPS / Physical (mirrors StorageService migration-026 methods) ─────

    async def save_gps_session(
        self,
        match_id: int,
        player_id: int,
        session_type: str,
        vendor: str,
    ) -> int:
        raise StorageNotInitializedError("save_gps_session")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO gps_sessions (match_id, player_id, session_type, vendor) VALUES ($1,$2,$3,$4) RETURNING id",
                match_id,
                player_id,
                session_type,
                vendor,
            )
            return row["id"] if row else 0

    async def update_gps_session_stats(self, session_id: int, summary: dict) -> None:
        if not self._pool:
            raise StorageNotInitializedError("update_gps_session_stats")
        await self.execute(
            """UPDATE gps_sessions SET duration_seconds=$1, total_distance_m=$2,
               max_speed_kmh=$3, avg_speed_kmh=$4, player_load=$5 WHERE id=$6""",
            summary.get("duration_s"),
            summary.get("total_distance_m"),
            summary.get("max_speed_kmh"),
            summary.get("avg_speed_kmh"),
            summary.get("total_player_load"),
            session_id,
        )

    async def save_gps_samples_bulk(self, session_id: int, samples: list[dict]) -> int:
        if not self._pool:
            raise StorageNotInitializedError("save_gps_samples_bulk")
        if not samples:
            return 0
        params = [
            (
                session_id,
                s.get("timestamp", 0.0),
                s.get("lat"),
                s.get("lon"),
                s.get("speed_ms"),
                s.get("acceleration"),
                s.get("accel_x"),
                s.get("accel_y"),
                s.get("accel_z"),
                s.get("heart_rate"),
                s.get("distance"),
                s.get("player_load"),
                s.get("metabolic_power"),
                s.get("speed_zone"),
                s.get("x_m"),
                s.get("y_m"),
            )
            for s in samples
        ]
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """INSERT INTO gps_samples (session_id, timestamp, lat, lon, speed_ms,
                   acceleration, accel_x, accel_y, accel_z, heart_rate, distance,
                   player_load, metabolic_power, speed_zone, x_m, y_m)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)""",
                params,
            )
        return len(params)

    async def get_gps_sessions(self, match_id: int) -> list[dict]:
        if not self._pool:
            raise StorageNotInitializedError("get_gps_sessions")
        return await self.fetch(
            """SELECT id, player_id, session_type, vendor, start_time, end_time,
               duration_seconds, total_distance_m, max_speed_kmh, avg_speed_kmh,
               player_load FROM gps_sessions
               WHERE match_id=$1 AND (is_deleted IS NULL OR is_deleted=0)
               ORDER BY id""",
            match_id,
        )

    async def get_gps_samples(self, session_id: int) -> list[dict]:
        if not self._pool:
            raise StorageNotInitializedError("get_gps_samples")
        return await self.fetch(
            """SELECT timestamp, speed_ms, acceleration, heart_rate, distance,
               player_load, metabolic_power, speed_zone, x_m, y_m
               FROM gps_samples WHERE session_id=$1 ORDER BY timestamp""",
            session_id,
        )

    async def save_acwr(
        self,
        player_id: int,
        date: str,
        acute: float,
        chronic: float,
        acwr: float,
    ) -> int:
        raise StorageNotInitializedError("save_acwr")
        cat = "normal"
        if acwr > 1.5:
            cat = "very_high"
        elif acwr > 1.3:
            cat = "high"
        elif acwr < 0.8:
            cat = "low"
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO acwr_daily (player_id, date, acute_load_7d, chronic_load_28d, acwr, load_category)
                   VALUES ($1,$2,$3,$4,$5,$6)
                   ON CONFLICT (player_id, date) DO UPDATE SET
                     acute_load_7d=EXCLUDED.acute_load_7d,
                     chronic_load_28d=EXCLUDED.chronic_load_28d,
                     acwr=EXCLUDED.acwr, load_category=EXCLUDED.load_category
                   RETURNING id""",
                player_id,
                date,
                acute,
                chronic,
                acwr,
                cat,
            )
            return row["id"] if row else 0

    async def get_player_acwr(self, player_id: int, limit: int = 30) -> list[dict]:
        if not self._pool:
            raise StorageNotInitializedError("get_player_acwr")
        return await self.fetch(
            """SELECT date, acute_load_7d, chronic_load_28d, acwr, load_category
               FROM acwr_daily WHERE player_id=$1 ORDER BY date DESC LIMIT $2""",
            player_id,
            limit,
        )

    async def get_player_gps_summary(self, player_id: int, limit: int = 10) -> list[dict]:
        if not self._pool:
            raise StorageNotInitializedError("get_player_gps_summary")
        return await self.fetch(
            """SELECT id, session_type, vendor, start_time, duration_seconds,
               total_distance_m, max_speed_kmh, avg_speed_kmh, player_load
               FROM gps_sessions WHERE player_id=$1 ORDER BY id DESC LIMIT $2""",
            player_id,
            limit,
        )

    async def get_squad_injury_report(self, team_id: int) -> dict:
        """Active-injury report for players linked to matches of *team_id*.

        Mirrors StorageService.get_squad_injury_report: resolves team ->
        player_profile ids via matches + player_match_links, then reuses
        InjuryTrackerService for the aggregation.
        """
        empty: dict[str, Any] = {
            "total_active": 0,
            "injuries": [],
            "by_severity": {},
            "by_body_part": {},
            "high_risk_count": 0,
            "high_risk_injuries": [],
            "report_date": datetime.now().isoformat(),
        }
        if not self._pool:
            raise StorageNotInitializedError("get_squad_injury_report")
        rows = await self.fetch(
            """SELECT DISTINCT pml.player_id FROM player_match_links pml
               JOIN matches m ON m.id = pml.match_id
               WHERE m.home_team_id = $1 OR m.away_team_id = $1""",
            team_id,
        )
        player_ids = [r["player_id"] for r in rows]
        if not player_ids:
            return empty
        from kawkab.services.injury_tracker import InjuryTrackerService

        tracker = InjuryTrackerService(None)  # pool-backed queries below
        active = []
        placeholders = ",".join(f"${i + 1}" for i in range(len(player_ids)))
        injury_rows = await self.fetch(
            f"SELECT * FROM injuries WHERE status IN ('active','chronic') "
            f"AND player_id IN ({placeholders}) ORDER BY date_injured DESC",
            *player_ids,
        )
        for r in injury_rows:
            active.append(dict(r))
        total = len(active)
        by_severity: dict[str, int] = {}
        by_body_part: dict[str, int] = {}
        high_risk: list[dict] = []
        for inj in active:
            sev = inj.get("severity", "unknown")
            by_severity[sev] = by_severity.get(sev, 0) + 1
            bp = inj.get("body_part", "unknown")
            by_body_part[bp] = by_body_part.get(bp, 0) + 1
            risk = inj.get("injury_risk_score", 0)
            if risk and risk > 0.5:
                high_risk.append(inj)
        del tracker  # aggregation done inline; service kept for schema parity
        return {
            "total_active": total,
            "injuries": active,
            "by_severity": by_severity,
            "by_body_part": by_body_part,
            "high_risk_count": len(high_risk),
            "high_risk_injuries": high_risk,
            "report_date": datetime.now().isoformat(),
        }

    async def save_tracking_frames_bulk(self, match_id: int, frames: list[dict]) -> int:
        if not self._pool:
            raise StorageNotInitializedError("save_tracking_frames_bulk")
        if not frames:
            return 0
        params = []
        for f in frames:
            try:
                params.append(
                    (
                        match_id,
                        f.get("frame_number", 0),
                        f.get("timestamp", 0.0),
                        json.dumps(f.get("player_detections", []), default=str),
                        json.dumps(f.get("ball_detections", []), default=str),
                    )
                )
            except Exception as e:
                logger.warning(
                    "save_tracking_frames_bulk frame %s failed: %s", f.get("frame_number"), e
                )
        if not params:
            return 0
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """INSERT INTO tracking_frames (match_id, frame_number, timestamp, player_detections, ball_detections)
                   VALUES ($1,$2,$3,$4::jsonb,$5::jsonb)
                   ON CONFLICT (match_id, frame_number) DO UPDATE
                   SET player_detections = EXCLUDED.player_detections,
                       ball_detections = EXCLUDED.ball_detections,
                       timestamp = EXCLUDED.timestamp""",
                params,
            )
        return len(params)

    async def get_tracking_frames(
        self, match_id: int, start_frame: int = 0, end_frame: int | None = None, limit: int = 1000
    ) -> list[dict]:
        if end_frame is not None:
            rows = await self.fetch(
                "SELECT id, match_id, frame_number, timestamp, player_detections, ball_detections, created_at FROM tracking_frames WHERE match_id = $1 AND frame_number >= $2 AND frame_number <= $3 ORDER BY frame_number LIMIT $4",
                match_id,
                start_frame,
                end_frame,
                limit,
            )
        else:
            rows = await self.fetch(
                "SELECT id, match_id, frame_number, timestamp, player_detections, ball_detections, created_at FROM tracking_frames WHERE match_id = $1 AND frame_number >= $2 ORDER BY frame_number LIMIT $3",
                match_id,
                start_frame,
                limit,
            )
        result = []
        for r in rows:
            d = dict(r)
            if isinstance(d.get("player_detections"), str):
                d["player_detections"] = json.loads(d["player_detections"])
            if isinstance(d.get("ball_detections"), str):
                d["ball_detections"] = json.loads(d["ball_detections"])
            result.append(d)
        return result

    async def get_tracking_frame_count(self, match_id: int) -> int:
        if not self._pool:
            raise StorageNotInitializedError("get_tracking_frame_count")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COUNT(*) AS cnt FROM tracking_frames WHERE match_id = $1",
                match_id,
            )
            return row["cnt"] if row else 0

    async def delete_tracking_frames(self, match_id: int) -> bool:
        if not self._pool:
            raise StorageNotInitializedError("delete_tracking_frames")
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM tracking_frames WHERE match_id = $1", match_id)
            return True

    # ── Encryption Keys ─────────────────────────────────────────────────────

    async def get_encryption_key(self, key_name: str = "medical_v1") -> str | None:
        if not self._pool:
            raise StorageNotInitializedError("get_encryption_key")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT key_value FROM encryption_keys WHERE key_name = $1",
                key_name,
            )
            return row["key_value"] if row else None

    async def rotate_encryption_key(self, key_name: str = "medical_v1") -> str | None:
        raise StorageNotInitializedError("rotate_encryption_key")
        import os as _os

        new_key = _os.urandom(32).hex()
        async with self._pool.acquire() as conn:
            r = await conn.execute(
                "UPDATE encryption_keys SET key_value = $1, rotated_at = NOW() WHERE key_name = $2",
                new_key,
                key_name,
            )
            if r != "UPDATE 0":
                from kawkab.core.encryption import init_fernet

                init_fernet(new_key)
                return new_key
            return None

    # ── Backup / Restore (pg_dump/pg_restore) ──────────────────────────────

    def backup(self) -> str:
        """PostgreSQL backup via pg_dump. Raises NotImplementedError if CLI tools unavailable."""
        if not self._dsn:
            return ""
        try:
            from kawkab.core.paths import get_paths

            backup_dir = get_paths().appdata / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = str(backup_dir / f"kawkab_pg_backup_{timestamp}.dump")
            subprocess.run(
                ["pg_dump", "--no-owner", "--format=custom", "--file", backup_path, self._dsn],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("PostgreSQL backup saved to %s", backup_path)
            return backup_path
        except FileNotFoundError:
            logger.error("pg_dump not found; install PostgreSQL client tools")
            return ""
        except subprocess.CalledProcessError as e:
            logger.error("pg_dump failed: %s", e.stderr)
            return ""
        except Exception as e:
            logger.error("backup failed: %s", e)
            return ""

    def restore(self, backup_path: str) -> bool:
        """Restore PostgreSQL from a pg_dump custom-format dump."""
        if not backup_path or not self._dsn:
            return False
        if not Path(backup_path).exists():
            logger.error("restore: backup file not found: %s", backup_path)
            return False
        try:
            subprocess.run(
                ["pg_restore", "--no-owner", "--clean", "--dbname", self._dsn, backup_path],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("PostgreSQL restored from %s", backup_path)
            return True
        except FileNotFoundError:
            logger.error("pg_restore not found; install PostgreSQL client tools")
            return False
        except subprocess.CalledProcessError as e:
            logger.error("pg_restore failed: %s", e.stderr)
            return False
        except Exception as e:
            logger.error("restore failed: %s", e)
            return False

    def auto_backup(self) -> str:
        result = self.backup()
        if result:
            logger.info("Auto-backup completed")
        else:
            logger.warning("Auto-backup skipped or failed")
        return result

    # ── Seasons ─────────────────────────────────────────────────────────────

    async def save_season(
        self,
        name: str,
        team_name: str = "",
        competition: str = "",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> int:
        raise StorageNotInitializedError("save_season")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO seasons (name, team_name, competition, start_date, end_date) VALUES ($1,$2,$3,$4,$5) RETURNING id",
                name,
                team_name,
                competition,
                start_date,
                end_date,
            )
            return row["id"] if row else 0

    _SEASON_COLUMNS = "id, name, start_date, end_date, created_at, updated_at"

    async def get_all_seasons(self) -> list[dict]:
        return await self.fetch(
            f"SELECT {self._SEASON_COLUMNS} FROM seasons ORDER BY start_date DESC"
        )

    async def get_season(self, season_id: int) -> dict | None:
        return await self.fetchrow(
            f"SELECT {self._SEASON_COLUMNS} FROM seasons WHERE id = $1", season_id
        )

    # ── Player Match Links ──────────────────────────────────────────────────

    async def save_player_match_link(
        self,
        player_id: int,
        match_id: int,
        track_id: int | None = None,
        confidence: float = 0.0,
        is_verified: bool = False,
    ) -> int:
        raise StorageNotInitializedError("save_player_match_link")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO player_match_links (player_id, match_id, track_id, confidence, is_verified) VALUES ($1,$2,$3,$4,$5) ON CONFLICT (player_id, match_id) DO UPDATE SET track_id=EXCLUDED.track_id, confidence=EXCLUDED.confidence RETURNING id",
                player_id,
                match_id,
                track_id,
                confidence,
                is_verified,
            )
            return row["id"] if row else 0

    async def get_player_match_links(self, match_id: int) -> list[dict]:
        return await self.fetch(
            "SELECT id, player_id, match_id, track_id, confidence, is_verified, created_at "
            "FROM player_match_links WHERE match_id = $1 ORDER BY track_id",
            match_id,
        )

    # ── Match Comparisons ───────────────────────────────────────────────────

    async def save_match_comparison(
        self,
        name: str,
        match_id_1: int,
        match_id_2: int,
        comparison_type: str = "",
        focus_areas: list | None = None,
        notes: str = "",
    ) -> int:
        raise StorageNotInitializedError("save_match_comparison")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO match_comparisons (name, match_id_1, match_id_2, comparison_type, focus_areas, notes) VALUES ($1,$2,$3,$4,$5,$6) RETURNING id",
                name,
                match_id_1,
                match_id_2,
                comparison_type,
                json.dumps(focus_areas or []),
                notes,
            )
            return row["id"] if row else 0

    async def get_match_comparisons(self, match_id: int) -> list[dict]:
        return await self.fetch(
            "SELECT id, name, match_id_1, match_id_2, comparison_type, focus_areas, notes, created_at "
            "FROM match_comparisons WHERE match_id_1 = $1 OR match_id_2 = $1 ORDER BY created_at DESC",
            match_id,
        )

    # ── Analysis Quality ────────────────────────────────────────────────────

    async def save_analysis_quality(
        self,
        match_id: int,
        overall_score: float | None = None,
        tracking_score: float | None = None,
        event_detection_score: float | None = None,
        homography_score: float | None = None,
        team_assignment_score: float | None = None,
        issues: list | None = None,
        warnings: list | None = None,
    ) -> int:
        raise StorageNotInitializedError("save_analysis_quality")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO analysis_quality (match_id, overall_score, tracking_score, event_detection_score, homography_score, team_assignment_score, issues, warnings) VALUES ($1,$2,$3,$4,$5,$6,$7,$8) RETURNING id",
                match_id,
                overall_score,
                tracking_score,
                event_detection_score,
                homography_score,
                team_assignment_score,
                json.dumps(issues or []),
                json.dumps(warnings or []),
            )
            return row["id"] if row else 0

    async def get_analysis_quality(self, match_id: int) -> list[dict]:
        return await self.fetch(
            "SELECT id, match_id, overall_score, tracking_score, event_detection_score, "
            "homography_score, team_assignment_score, issues, warnings, created_at "
            "FROM analysis_quality WHERE match_id = $1 ORDER BY created_at DESC",
            match_id,
        )

    # ── Exports ─────────────────────────────────────────────────────────────

    async def save_export(
        self,
        export_type: str,
        format: str,
        match_id: int | None = None,
        season_id: int | None = None,
        file_path: str = "",
        file_size_bytes: int | None = None,
    ) -> int:
        raise StorageNotInitializedError("save_export")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO exports (match_id, season_id, export_type, format, file_path, file_size_bytes) VALUES ($1,$2,$3,$4,$5,$6) RETURNING id",
                match_id,
                season_id,
                export_type,
                format,
                file_path,
                file_size_bytes,
            )
            return row["id"] if row else 0

    # ── Batch Jobs ──────────────────────────────────────────────────────────

    async def save_batch_job(
        self, name: str, match_ids: list | None = None, options: dict | None = None
    ) -> int:
        raise StorageNotInitializedError("save_batch_job")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO batch_jobs (name, match_ids, options) VALUES ($1,$2,$3) RETURNING id",
                name,
                json.dumps(match_ids or []),
                json.dumps(options or {}),
            )
            return row["id"] if row else 0

    async def update_batch_job_status(
        self,
        job_id: int,
        status: str,
        completed_matches: int | None = None,
        failed_matches: int | None = None,
        error_message: str | None = None,
    ) -> bool:
        raise StorageNotInitializedError("update_batch_job_status")
        sets = ["status = $2"]
        args = [job_id, status]
        idx = 3
        if completed_matches is not None:
            sets.append(f"completed_matches = ${idx}")
            args.append(completed_matches)
            idx += 1
        if failed_matches is not None:
            sets.append(f"failed_matches = ${idx}")
            args.append(failed_matches)
            idx += 1
        if error_message is not None:
            sets.append(f"error_message = ${idx}")
            args.append(error_message)
            idx += 1
        if status in ("completed", "failed"):
            sets.append("completed_at = NOW()")
        args.append(job_id)
        async with self._pool.acquire() as conn:
            r = await conn.execute(
                f"UPDATE batch_jobs SET {', '.join(sets)} WHERE id = $1",
                *args,
            )
            return r != "UPDATE 0"

    _BATCH_JOB_COLUMNS = (
        "id, name, status, total_matches, completed_matches, failed_matches, "
        "match_ids, options, started_at, completed_at, error_message, created_at"
    )

    async def get_batch_jobs(self, status: str | None = None) -> list[dict]:
        if status:
            return await self.fetch(
                f"SELECT {self._BATCH_JOB_COLUMNS} FROM batch_jobs WHERE status = $1 ORDER BY created_at DESC",
                status,
            )
        return await self.fetch(
            f"SELECT {self._BATCH_JOB_COLUMNS} FROM batch_jobs ORDER BY created_at DESC"
        )

    # ── Caches (football_data_cache, external_data_cache) ───────────────────

    async def set_cache(
        self, cache_key: str, data: str, expires_at: float, table: str = "football_data_cache"
    ) -> bool:
        raise StorageNotInitializedError("set_cache")
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"INSERT INTO {table} (cache_key, data, expires_at) VALUES ($1,$2,$3) ON CONFLICT (cache_key) DO UPDATE SET data=EXCLUDED.data, expires_at=EXCLUDED.expires_at",
                cache_key,
                data,
                expires_at,
            )
            return True

    async def get_cache(self, cache_key: str, table: str = "football_data_cache") -> str | None:
        if not self._pool:
            raise StorageNotInitializedError("get_cache")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT data FROM {table} WHERE cache_key = $1 AND expires_at > extract(epoch from NOW())",
                cache_key,
            )
            return row["data"] if row else None

    # ── Match Weather ───────────────────────────────────────────────────────

    async def save_match_weather(self, match_id: int, weather: dict) -> int:
        raise StorageNotInitializedError("save_match_weather")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO match_weather (match_id, latitude, longitude, temperature_c, feels_like_c,
                       precipitation_mm, wind_speed_kmh, wind_direction_deg, humidity_pct,
                       cloud_cover_pct, conditions, pitch_state, source, recorded_at)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14) RETURNING id""",
                match_id,
                weather.get("latitude"),
                weather.get("longitude"),
                weather.get("temperature_c"),
                weather.get("feels_like_c"),
                weather.get("precipitation_mm"),
                weather.get("wind_speed_kmh"),
                weather.get("wind_direction_deg"),
                weather.get("humidity_pct"),
                weather.get("cloud_cover_pct"),
                weather.get("conditions"),
                weather.get("pitch_state"),
                weather.get("source"),
                weather.get("recorded_at"),
            )
            return row["id"] if row else 0

    async def get_match_weather(self, match_id: int) -> dict | None:
        return await self.fetchrow(
            "SELECT id, match_id, latitude, longitude, temperature_c, feels_like_c, "
            "precipitation_mm, wind_speed_kmh, wind_direction_deg, humidity_pct, "
            "cloud_cover_pct, conditions, pitch_state, source, recorded_at "
            "FROM match_weather WHERE match_id = $1 ORDER BY id DESC LIMIT 1",
            match_id,
        )

    # ── Card Events ─────────────────────────────────────────────────────────

    async def save_card_event(self, match_id: int, card: dict) -> int:
        raise StorageNotInitializedError("save_card_event")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO card_events (match_id, player_track_id, player_name, card_type, minute, second, detection_source, confidence, description) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9) RETURNING id",
                match_id,
                card.get("player_track_id"),
                card.get("player_name", ""),
                card["card_type"],
                card.get("minute", 0),
                card.get("second", 0),
                card.get("detection_source"),
                card.get("confidence"),
                card.get("description", ""),
            )
            return row["id"] if row else 0

    async def get_card_events(self, match_id: int) -> list[dict]:
        return await self.fetch(
            "SELECT id, match_id, player_track_id, player_name, card_type, minute, second, "
            "detection_source, confidence, description "
            "FROM card_events WHERE match_id = $1 ORDER BY minute, second",
            match_id,
        )

    # ── Psychology Events ───────────────────────────────────────────────────

    async def save_psychology_event(self, match_id: int, event: dict) -> int:
        raise StorageNotInitializedError("save_psychology_event")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO psychology_events (match_id, event_type, minute, second, team, description, severity, data_json) VALUES ($1,$2,$3,$4,$5,$6,$7,$8) RETURNING id",
                match_id,
                event["event_type"],
                event.get("minute", 0),
                event.get("second", 0),
                event.get("team", ""),
                event.get("description", ""),
                event.get("severity"),
                json.dumps(event.get("data_json", event.get("data", {})), default=str),
            )
            return row["id"] if row else 0

    async def get_psychology_events(self, match_id: int) -> list[dict]:
        return await self.fetch(
            "SELECT id, match_id, event_type, minute, second, team, description, "
            "severity, data_json "
            "FROM psychology_events WHERE match_id = $1 ORDER BY minute, second",
            match_id,
        )

    # ── Player Shortlist ────────────────────────────────────────────────────

    async def save_shortlist_entry(self, entry: dict) -> int:
        raise StorageNotInitializedError("save_shortlist_entry")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO player_shortlist (player_id, player_name, position, team, league,
                       priority, status, notes, scout_rating, estimated_value, age, nationality)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12) RETURNING id""",
                entry["player_id"],
                entry["player_name"],
                entry.get("position", ""),
                entry.get("team", ""),
                entry.get("league", ""),
                entry.get("priority", "medium"),
                entry.get("status", "scouted"),
                entry.get("notes", ""),
                entry.get("scout_rating", 0.0),
                entry.get("estimated_value"),
                entry.get("age"),
                entry.get("nationality", ""),
            )
            return row["id"] if row else 0

    async def update_shortlist_entry(self, entry_id: int, updates: dict) -> bool:
        raise StorageNotInitializedError("update_shortlist_entry")
        allowed = {"priority", "status", "notes", "scout_rating", "estimated_value"}
        sets = ["last_updated = NOW()"]
        args: list[Any] = []
        for k, v in updates.items():
            if k not in allowed:
                continue
            col = self._sanitize_column_name(k)
            if col is None:
                continue
            sets.append(f"{col} = ${len(args) + 1}")
            args.append(v)
        if len(sets) == 1:
            return False
        args.append(entry_id)
        async with self._pool.acquire() as conn:
            r = await conn.execute(
                f"UPDATE player_shortlist SET {', '.join(sets)} WHERE id = ${len(args)}", *args
            )
            return r != "UPDATE 0"

    async def get_shortlist(
        self, status: str | None = None, priority: str | None = None
    ) -> list[dict]:
        conditions = []
        args: list[Any] = []
        if status:
            conditions.append(f"status = ${len(args) + 1}")
            args.append(status)
        if priority:
            conditions.append(f"priority = ${len(args) + 1}")
            args.append(priority)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        return await self.fetch(
            "SELECT id, player_id, player_name, position, team, league, added_date, "
            "priority, status, notes, scout_rating, estimated_value, age, nationality, "
            f"last_updated FROM player_shortlist{where} ORDER BY last_updated DESC",
            *args,
        )

    async def delete_shortlist_entry(self, entry_id: int) -> bool:
        raise StorageNotInitializedError("delete_shortlist_entry")
        async with self._pool.acquire() as conn:
            r = await conn.execute("DELETE FROM player_shortlist WHERE id = $1", entry_id)
            return r != "DELETE 0"

    # ── Player Contracts ────────────────────────────────────────────────────

    async def save_contract(self, contract: dict) -> int:
        raise StorageNotInitializedError("save_contract")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO player_contracts (player_profile_id, player_name, contract_type,
                       start_date, end_date, club_option_years, player_option_years,
                       release_clause_millions, wage_weekly_pounds, agent_name, notes)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11) RETURNING id""",
                contract["player_profile_id"],
                contract["player_name"],
                contract.get("contract_type", "permanent"),
                contract["start_date"],
                contract["end_date"],
                contract.get("club_option_years", 0),
                contract.get("player_option_years", 0),
                contract.get("release_clause_millions"),
                contract.get("wage_weekly_pounds"),
                contract.get("agent_name", ""),
                contract.get("notes", ""),
            )
            return row["id"] if row else 0

    _CONTRACT_COLUMNS = (
        "id, player_profile_id, player_name, contract_type, start_date, end_date, "
        "club_option_years, player_option_years, release_clause_millions, "
        "wage_weekly_pounds, agent_name, notes, last_updated"
    )

    async def get_contracts(self, profile_id: int | None = None) -> list[dict]:
        if profile_id:
            return await self.fetch(
                f"SELECT {self._CONTRACT_COLUMNS} FROM player_contracts WHERE player_profile_id = $1 ORDER BY end_date DESC",
                profile_id,
            )
        return await self.fetch(
            f"SELECT {self._CONTRACT_COLUMNS} FROM player_contracts ORDER BY end_date ASC"
        )

    async def get_contracts_expiring_soon(self, days: int = 90) -> list[dict]:
        if not self._pool:
            raise StorageNotInitializedError("get_contracts_expiring_soon")
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT {self._CONTRACT_COLUMNS} FROM player_contracts WHERE end_date <= CURRENT_DATE + $1::integer AND end_date >= CURRENT_DATE ORDER BY end_date ASC",
                days,
            )
            return [dict(r) for r in rows]

    # ── Collaboration Tables ────────────────────────────────────────────────

    async def save_collab_user(self, username: str, role: str = "analyst") -> int:
        raise StorageNotInitializedError("save_collab_user")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO collab_users (username, role) VALUES ($1,$2) ON CONFLICT (username) DO UPDATE SET role=EXCLUDED.role RETURNING id",
                username,
                role,
            )
            return row["id"] if row else 0

    async def get_collab_users(self) -> list[dict]:
        return await self.fetch(
            "SELECT id, username, role, created_at FROM collab_users ORDER BY username"
        )

    async def save_collab_comment(
        self, match_id: int, text: str, user_id: int = 0, username: str = "", event_id: int = 0
    ) -> int:
        raise StorageNotInitializedError("save_collab_comment")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO collab_comments (match_id, event_id, user_id, username, text) VALUES ($1,$2,$3,$4,$5) RETURNING id",
                match_id,
                event_id,
                user_id,
                username,
                text,
            )
            return row["id"] if row else 0

    async def get_collab_comments(self, match_id: int) -> list[dict]:
        return await self.fetch(
            "SELECT id, match_id, event_id, user_id, username, text, created_at "
            "FROM collab_comments WHERE match_id = $1 ORDER BY created_at ASC",
            match_id,
        )

    async def save_collab_mention(
        self, username: str, from_user: str, text: str, match_id: int = 0, event_id: int = 0
    ) -> int:
        raise StorageNotInitializedError("save_collab_mention")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO collab_mentions (username, from_user, text, match_id, event_id) VALUES ($1,$2,$3,$4,$5) RETURNING id",
                username,
                from_user,
                text,
                match_id,
                event_id,
            )
            return row["id"] if row else 0

    async def get_collab_mentions(self, username: str) -> list[dict]:
        return await self.fetch(
            "SELECT id, username, from_user, text, match_id, event_id, read, created_at "
            "FROM collab_mentions WHERE username = $1 ORDER BY created_at DESC",
            username,
        )

    async def mark_mention_read(self, mention_id: int) -> bool:
        raise StorageNotInitializedError("mark_mention_read")
        async with self._pool.acquire() as conn:
            r = await conn.execute(
                "UPDATE collab_mentions SET read = 1 WHERE id = $1",
                mention_id,
            )
            return r != "UPDATE 0"

    # ── Wearable Sessions ───────────────────────────────────────────────────

    async def save_wearable_session(self, session: dict) -> int:
        raise StorageNotInitializedError("save_wearable_session")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO wearable_sessions (match_id, athlete_id, athlete_name, device_type,
                       device_serial, start_time, duration_s, sample_rate_hz,
                       avg_hr, max_hr, min_hr, total_distance_m, max_speed_ms, avg_speed_ms,
                       player_load, body_load, high_speed_running_m, sprint_distance_m,
                       accelerations, decelerations, point_count, metadata_json)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22) RETURNING id""",
                session.get("match_id"),
                session.get("athlete_id", ""),
                session.get("athlete_name", ""),
                session["device_type"],
                session.get("device_serial", ""),
                session.get("start_time"),
                session.get("duration_s", 0.0),
                session.get("sample_rate_hz", 0.0),
                session.get("avg_hr"),
                session.get("max_hr"),
                session.get("min_hr"),
                session.get("total_distance_m", 0.0),
                session.get("max_speed_ms"),
                session.get("avg_speed_ms"),
                session.get("player_load"),
                session.get("body_load"),
                session.get("high_speed_running_m", 0.0),
                session.get("sprint_distance_m", 0.0),
                session.get("accelerations", 0),
                session.get("decelerations", 0),
                session.get("point_count", 0),
                json.dumps(session.get("metadata_json", session.get("metadata", {})), default=str),
            )
            return row["id"] if row else 0

    _WEARABLE_SESSION_COLUMNS = (
        "id, match_id, athlete_id, athlete_name, device_type, device_serial, start_time, "
        "duration_s, sample_rate_hz, avg_hr, max_hr, min_hr, total_distance_m, max_speed_ms, "
        "avg_speed_ms, player_load, body_load, high_speed_running_m, sprint_distance_m, "
        "accelerations, decelerations, point_count, metadata_json, created_at, updated_at"
    )

    async def get_wearable_sessions(self, match_id: int | None = None) -> list[dict]:
        if match_id:
            return await self.fetch(
                f"SELECT {self._WEARABLE_SESSION_COLUMNS} FROM wearable_sessions WHERE match_id = $1 ORDER BY created_at DESC",
                match_id,
            )
        return await self.fetch(
            f"SELECT {self._WEARABLE_SESSION_COLUMNS} FROM wearable_sessions ORDER BY created_at DESC"
        )

    # ── Medical Tables (Injuries, Rehab, Concussion, History) ───────────────

    async def save_injury(self, injury: dict) -> int:
        raise StorageNotInitializedError("save_injury")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO injuries (player_id, match_id, injury_type, body_part, severity,
                       mechanism, date_injured, date_recovered, status, notes)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10) RETURNING id""",
                injury["player_id"],
                injury.get("match_id"),
                injury["injury_type"],
                injury["body_part"],
                injury.get("severity", "minor"),
                injury.get("mechanism", ""),
                injury["date_injured"],
                injury.get("date_recovered"),
                injury.get("status", "active"),
                injury.get("notes", ""),
            )
            return row["id"] if row else 0

    async def get_injuries(
        self, player_id: int | None = None, status: str | None = None
    ) -> list[dict]:
        conditions = []
        args: list[Any] = []
        if player_id is not None:
            conditions.append(f"player_id = ${len(args) + 1}")
            args.append(player_id)
        if status:
            conditions.append(f"status = ${len(args) + 1}")
            args.append(status)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        return await self.fetch(
            "SELECT id, player_id, match_id, injury_type, body_part, severity, mechanism, "
            "date_injured, date_recovered, status, notes, created_at, updated_at "
            f"FROM injuries{where} ORDER BY date_injured DESC",
            *args,
        )

    async def update_injury(self, injury_id: int, updates: dict) -> bool:
        raise StorageNotInitializedError("update_injury")
        allowed = {"severity", "status", "notes", "date_recovered", "mechanism"}
        sets = ["updated_at = NOW()"]
        args: list[Any] = []
        for k, v in updates.items():
            if k not in allowed:
                continue
            col = self._sanitize_column_name(k)
            if col is None:
                continue
            sets.append(f"{col} = ${len(args) + 1}")
            args.append(v)
        if len(sets) == 1:
            return False
        args.append(injury_id)
        async with self._pool.acquire() as conn:
            r = await conn.execute(
                f"UPDATE injuries SET {', '.join(sets)} WHERE id = ${len(args)}", *args
            )
            return r != "UPDATE 0"

    async def save_rehab_plan(self, plan: dict) -> int:
        raise StorageNotInitializedError("save_rehab_plan")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO rehab_plans (injury_id, phase, start_date, target_end_date,
                       actual_end_date, milestones, protocols, status, notes)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9) RETURNING id""",
                plan["injury_id"],
                plan.get("phase", "initial"),
                plan["start_date"],
                plan.get("target_end_date"),
                plan.get("actual_end_date"),
                json.dumps(plan.get("milestones", [])),
                plan.get("protocols", ""),
                plan.get("status", "active"),
                plan.get("notes", ""),
            )
            return row["id"] if row else 0

    async def get_rehab_plans(self, injury_id: int) -> list[dict]:
        return await self.fetch(
            "SELECT id, injury_id, phase, start_date, target_end_date, actual_end_date, "
            "milestones, protocols, status, notes, created_at "
            "FROM rehab_plans WHERE injury_id = $1 ORDER BY start_date ASC",
            injury_id,
        )

    async def save_concussion_assessment(self, assessment: dict) -> int:
        raise StorageNotInitializedError("save_concussion_assessment")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO concussion_assessments (player_id, match_id, assessment_date,
                       assessment_type, symptoms_score, cognitive_score, balance_score,
                       clearance_status, cleared_by, notes)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10) RETURNING id""",
                assessment["player_id"],
                assessment.get("match_id"),
                assessment["assessment_date"],
                assessment.get("assessment_type", "scat5"),
                assessment.get("symptoms_score", 0),
                assessment.get("cognitive_score", 0),
                assessment.get("balance_score", 0),
                assessment.get("clearance_status", "not_cleared"),
                assessment.get("cleared_by", ""),
                assessment.get("notes", ""),
            )
            return row["id"] if row else 0

    async def get_concussion_assessments(self, player_id: int) -> list[dict]:
        return await self.fetch(
            "SELECT id, player_id, match_id, assessment_date, assessment_type, symptoms_score, "
            "cognitive_score, balance_score, clearance_status, cleared_by, notes, created_at "
            "FROM concussion_assessments WHERE player_id = $1 ORDER BY assessment_date DESC",
            player_id,
        )

    async def save_medical_history(self, entry: dict) -> int:
        raise StorageNotInitializedError("save_medical_history")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO medical_history (player_id, condition_type, diagnosis, diagnosis_date, status, severity, notes) VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING id",
                entry["player_id"],
                entry["condition_type"],
                entry["diagnosis"],
                entry["diagnosis_date"],
                entry.get("status", "active"),
                entry.get("severity", "moderate"),
                entry.get("notes", ""),
            )
            return row["id"] if row else 0

    async def get_medical_history(self, player_id: int) -> list[dict]:
        return await self.fetch(
            "SELECT id, player_id, condition_type, diagnosis, diagnosis_date, status, "
            "severity, notes, created_at "
            "FROM medical_history WHERE player_id = $1 ORDER BY diagnosis_date DESC",
            player_id,
        )

    # ── Audit Events ────────────────────────────────────────────────────────

    async def save_audit_event(
        self,
        action: str,
        entity_type: str = "",
        entity_id: str | None = None,
        details: dict | None = None,
        user_name: str = "local",
    ) -> int:
        raise StorageNotInitializedError("save_audit_event")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO audit_events (action, entity_type, entity_id, details_json, user_name) VALUES ($1,$2,$3,$4,$5) RETURNING id",
                action,
                entity_type,
                entity_id,
                json.dumps(details or {}, default=str),
                user_name,
            )
            return row["id"] if row else 0

    async def get_audit_events(
        self,
        action: str | None = None,
        entity_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        conditions = []
        args: list[Any] = []
        if action:
            conditions.append(f"action = ${len(args) + 1}")
            args.append(action)
        if entity_type:
            conditions.append(f"entity_type = ${len(args) + 1}")
            args.append(entity_type)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        return await self.fetch(
            "SELECT id, action, entity_type, entity_id, details_json, user_name, created_at "
            f"FROM audit_events{where} ORDER BY created_at DESC LIMIT ${len(args) + 1} OFFSET ${len(args) + 2}",
            *args,
            limit,
            offset,
        )

    # ── Settings ────────────────────────────────────────────────────────────

    async def get_setting(self, key: str) -> str | None:
        if not self._pool:
            raise StorageNotInitializedError("get_setting")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT value FROM settings WHERE key = $1",
                key,
            )
            return row["value"] if row else None

    async def set_setting(self, key: str, value: str) -> bool:
        raise StorageNotInitializedError("set_setting")
        async with self._pool.acquire() as conn:
            _ = await conn.execute(
                "INSERT INTO settings (key, value) VALUES ($1, $2) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()",
                key,
                value,
            )
            return True

    # ── Schema Version ──────────────────────────────────────────────────────

    async def get_schema_version(self) -> int | None:
        if not self._pool:
            raise StorageNotInitializedError("get_schema_version")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT MAX(version) as version FROM schema_version",
            )
            return row["version"] if row else None

    async def set_schema_version(self, version: int) -> bool:
        raise StorageNotInitializedError("set_schema_version")
        async with self._pool.acquire() as conn:
            _ = await conn.execute(
                "INSERT INTO schema_version (version) VALUES ($1) ON CONFLICT (version) DO NOTHING",
                version,
            )
            return True

    # ── Inline DDL fallback (used when pg_schema.sql not found) ─────────────

    _SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS matches (
        id SERIAL PRIMARY KEY, name TEXT NOT NULL, video_path TEXT DEFAULT '',
        home_team TEXT DEFAULT '', away_team TEXT DEFAULT '',
        match_date TIMESTAMPTZ, duration_seconds REAL, fps REAL, total_frames INTEGER,
        home_team_id INTEGER REFERENCES teams(id) ON DELETE SET NULL,
        away_team_id INTEGER REFERENCES teams(id) ON DELETE SET NULL,
        season_id INTEGER REFERENCES seasons(id) ON DELETE SET NULL,
        competition TEXT DEFAULT '', round TEXT DEFAULT '', opponent TEXT DEFAULT '',
        score_home INTEGER, score_away INTEGER, match_type TEXT DEFAULT 'unknown',
        api_match_id INTEGER, competition_code TEXT DEFAULT '',
        football_data_home_team_id INTEGER, football_data_away_team_id INTEGER,
        bzzoiro_home_team_id INTEGER, bzzoiro_away_team_id INTEGER,
        bzzoiro_event_id INTEGER, bzzoiro_league_id INTEGER,
        bzzoiro_competition_code TEXT DEFAULT '', prediction_data TEXT DEFAULT '',
        apifb_home_team_id INTEGER, apifb_away_team_id INTEGER,
        apifb_fixture_id INTEGER, apifb_league_id INTEGER, apifb_season INTEGER,
        analysis_json JSONB DEFAULT '{}', football_data_json JSONB DEFAULT '{}',
        apifootball_json JSONB DEFAULT '{}', bzzoiro_json JSONB DEFAULT '{}',
        is_deleted INTEGER DEFAULT 0, deleted_at TIMESTAMPTZ, deleted_by TEXT DEFAULT '',
        owner_id INTEGER, team_id INTEGER, is_shared INTEGER DEFAULT 0,
        created_at TIMESTAMPTZ DEFAULT NOW(), updated_at TIMESTAMPTZ DEFAULT NOW(), analyzed_at TIMESTAMPTZ
    );
    CREATE TABLE IF NOT EXISTS players (
        id SERIAL PRIMARY KEY, match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
        track_id INTEGER NOT NULL DEFAULT 0, jersey_number INTEGER, name TEXT DEFAULT '',
        team TEXT DEFAULT '', position TEXT DEFAULT '',
        distance_covered_m REAL DEFAULT 0, max_speed_kmh REAL DEFAULT 0,
        avg_speed_kmh REAL DEFAULT 0, passes_attempted INTEGER DEFAULT 0,
        passes_completed INTEGER DEFAULT 0, shots INTEGER DEFAULT 0, tackles INTEGER DEFAULT 0,
        confidence REAL DEFAULT 0,
        is_deleted INTEGER DEFAULT 0, deleted_at TIMESTAMPTZ, deleted_by TEXT DEFAULT ''
    );
    CREATE TABLE IF NOT EXISTS events (
        id SERIAL PRIMARY KEY, match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
        event_type TEXT NOT NULL DEFAULT '', timestamp DOUBLE PRECISION NOT NULL DEFAULT 0,
        from_track_id INTEGER DEFAULT 0, to_track_id INTEGER, team TEXT DEFAULT '',
        completed BOOLEAN DEFAULT FALSE, confidence REAL DEFAULT 0, metadata JSONB DEFAULT '{}',
        user_corrected BOOLEAN DEFAULT FALSE,
        x DOUBLE PRECISION DEFAULT 0, y DOUBLE PRECISION DEFAULT 0,
        end_x DOUBLE PRECISION DEFAULT 0, end_y DOUBLE PRECISION DEFAULT 0,
        is_goal BOOLEAN DEFAULT FALSE, data JSONB DEFAULT '{}',
        is_deleted INTEGER DEFAULT 0, deleted_at TIMESTAMPTZ, deleted_by TEXT DEFAULT ''
    );
    CREATE TABLE IF NOT EXISTS reports (
        id SERIAL PRIMARY KEY, match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
        language TEXT NOT NULL DEFAULT 'en', report_text TEXT NOT NULL DEFAULT '',
        report_type TEXT DEFAULT 'match', llm_provider TEXT DEFAULT '',
        content TEXT DEFAULT '', created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS corrections (
        id SERIAL PRIMARY KEY, match_id INTEGER REFERENCES matches(id) ON DELETE CASCADE,
        event_id INTEGER DEFAULT 0, field TEXT DEFAULT '',
        old_value TEXT DEFAULT '', new_value TEXT DEFAULT '', reason TEXT DEFAULT ''
    );
    CREATE TABLE IF NOT EXISTS benchmark_results (
        id SERIAL PRIMARY KEY, match_id INTEGER REFERENCES matches(id) ON DELETE SET NULL,
        video_path TEXT DEFAULT '', video_duration_seconds REAL, total_frames INTEGER,
        total_time_seconds REAL, total_time REAL DEFAULT 0, realtime_ratio REAL,
        fps_effective REAL, fps REAL DEFAULT 0,
        stage_enhancement_seconds REAL, stage_detection_seconds REAL,
        stage_tracking_seconds REAL, stage_analysis_seconds REAL,
        stage_advanced_metrics_seconds REAL, stage_save_seconds REAL,
        peak_memory_mb REAL, peak_memory REAL DEFAULT 0,
        peak_gpu_memory_mb REAL, peak_gpu_memory REAL DEFAULT 0,
        gpu_utilization_pct REAL, gpu_name TEXT DEFAULT '', cpu_name TEXT DEFAULT '',
        ram_gb REAL, model_size TEXT DEFAULT '', model_name TEXT DEFAULT '',
        frame_skip INTEGER DEFAULT 0, avg_fps REAL DEFAULT 0, avg_latency_ms REAL DEFAULT 0,
        stages_json JSONB DEFAULT '{}', created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS validation_results (
        id SERIAL PRIMARY KEY, match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
        ground_truth_source TEXT DEFAULT '', overall_accuracy REAL, category TEXT DEFAULT '',
        metric_name TEXT NOT NULL DEFAULT '', computed_value REAL,
        ground_truth_value REAL, absolute_error REAL, relative_error_pct REAL,
        accuracy_score REAL, sample_count INTEGER, accuracy REAL DEFAULT 0,
        details_json JSONB DEFAULT '{}', created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS feedback (
        id SERIAL PRIMARY KEY, coach_id TEXT NOT NULL DEFAULT '',
        match_id INTEGER REFERENCES matches(id) ON DELETE SET NULL,
        user_name TEXT DEFAULT '', overall_rating INTEGER, rating INTEGER DEFAULT 0,
        tracking_rating INTEGER, events_rating INTEGER, report_rating INTEGER,
        ui_rating INTEGER, comments TEXT DEFAULT '', issues JSONB DEFAULT '[]',
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS issues (
        id SERIAL PRIMARY KEY, category TEXT NOT NULL DEFAULT 'other',
        severity TEXT NOT NULL DEFAULT 'low', description TEXT NOT NULL DEFAULT '',
        match_id INTEGER REFERENCES matches(id) ON DELETE SET NULL,
        screenshot_path TEXT DEFAULT '', logs TEXT DEFAULT '',
        status TEXT DEFAULT 'open', created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS usage_sessions (
        id SERIAL PRIMARY KEY, session_id TEXT UNIQUE NOT NULL DEFAULT '',
        match_id INTEGER REFERENCES matches(id) ON DELETE SET NULL, user_name TEXT DEFAULT '',
        features_used JSONB NOT NULL DEFAULT '[]', action TEXT DEFAULT '',
        duration_seconds REAL NOT NULL DEFAULT 0, duration_s REAL DEFAULT 0,
        match_count INTEGER DEFAULT 0, gpu_tier TEXT DEFAULT '', model_size TEXT DEFAULT '',
        error_count INTEGER DEFAULT 0, created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS clips (
        id SERIAL PRIMARY KEY, match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
        event_type TEXT NOT NULL DEFAULT '', name TEXT DEFAULT '',
        start_seconds REAL NOT NULL DEFAULT 0, start_time DOUBLE PRECISION DEFAULT 0,
        end_seconds REAL NOT NULL DEFAULT 0, end_time DOUBLE PRECISION DEFAULT 0,
        duration_seconds REAL NOT NULL DEFAULT 0,
        source_video_path TEXT NOT NULL DEFAULT '', video_path TEXT DEFAULT '',
        output_path TEXT NOT NULL DEFAULT '', thumbnail_path TEXT DEFAULT '',
        player_id INTEGER, player_track_id INTEGER DEFAULT 0,
        description TEXT DEFAULT '', tags_json JSONB DEFAULT '[]',
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS playlists (
        id SERIAL PRIMARY KEY, name TEXT NOT NULL DEFAULT '', description TEXT DEFAULT '',
        match_id INTEGER REFERENCES matches(id) ON DELETE CASCADE,
        clip_ids JSONB NOT NULL DEFAULT '[]', clips_json JSONB DEFAULT '[]',
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS player_profiles (
        id SERIAL PRIMARY KEY, global_id TEXT UNIQUE NOT NULL DEFAULT '',
        display_name TEXT DEFAULT '', name TEXT DEFAULT '', team TEXT DEFAULT '',
        position TEXT DEFAULT '', preferred_position TEXT DEFAULT '',
        jersey_number INTEGER DEFAULT 0, height_cm INTEGER, weight_kg INTEGER,
        dominant_foot TEXT DEFAULT '', date_of_birth TIMESTAMPTZ,
        nationality TEXT DEFAULT '', photo_path TEXT DEFAULT '', notes JSONB DEFAULT '{}',
        is_active BOOLEAN DEFAULT TRUE, face_embedding TEXT DEFAULT '', face_confidence REAL DEFAULT 0,
        football_data_person_id INTEGER, football_data_team_id INTEGER,
        bzzoiro_person_id INTEGER, bzzoiro_team_id INTEGER,
        apifb_person_id INTEGER, apifb_team_id INTEGER,
        contract_end_date TEXT DEFAULT '', contract_type TEXT DEFAULT 'permanent',
        wage_weekly_pounds REAL,
        created_at TIMESTAMPTZ DEFAULT NOW(), updated_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS advanced_metrics (
        id SERIAL PRIMARY KEY, match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
        player_id INTEGER REFERENCES player_profiles(id) ON DELETE SET NULL,
        metric_name TEXT NOT NULL DEFAULT '', metric_value REAL,
        category TEXT DEFAULT '', metric_category TEXT DEFAULT '',
        pitch_zone TEXT DEFAULT '', timestamp REAL,
        data_json JSONB DEFAULT '{}', metadata JSONB DEFAULT '{}',
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS coding_tags (
        id SERIAL PRIMARY KEY, match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
        event_type TEXT NOT NULL DEFAULT '', tag_type TEXT DEFAULT '', sub_type TEXT DEFAULT '',
        category TEXT DEFAULT '', video_time REAL NOT NULL DEFAULT 0,
        timestamp DOUBLE PRECISION DEFAULT 0, player_track_id INTEGER DEFAULT 0,
        player_name TEXT DEFAULT '', team TEXT DEFAULT '', period INTEGER DEFAULT 1,
        notes TEXT DEFAULT '', color TEXT DEFAULT '#3498db',
        lead_ms INTEGER DEFAULT 2000, lag_ms INTEGER DEFAULT 3000,
        is_deleted INTEGER DEFAULT 0, deleted_at TIMESTAMPTZ, deleted_by TEXT DEFAULT '',
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS teams (
        id SERIAL PRIMARY KEY, name TEXT UNIQUE NOT NULL,
        short_name TEXT DEFAULT '', home_color TEXT DEFAULT '#1e7e34',
        away_color TEXT DEFAULT '#ffffff', created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS tracking_frames (
        id SERIAL PRIMARY KEY, match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
        frame_number INTEGER NOT NULL, timestamp DOUBLE PRECISION NOT NULL DEFAULT 0,
        player_detections JSONB DEFAULT '[]', ball_detections JSONB DEFAULT '[]',
        created_at TIMESTAMPTZ DEFAULT NOW(), UNIQUE(match_id, frame_number)
    );
    CREATE TABLE IF NOT EXISTS encryption_keys (
        id SERIAL PRIMARY KEY, key_name TEXT UNIQUE NOT NULL,
        key_value TEXT NOT NULL, created_at TIMESTAMPTZ DEFAULT NOW(), rotated_at TIMESTAMPTZ
    );
    CREATE TABLE IF NOT EXISTS seasons (
        id SERIAL PRIMARY KEY, name TEXT NOT NULL, team_name TEXT DEFAULT '',
        competition TEXT DEFAULT '', start_date TIMESTAMPTZ, end_date TIMESTAMPTZ,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS player_match_links (
        id SERIAL PRIMARY KEY, player_id INTEGER NOT NULL REFERENCES player_profiles(id) ON DELETE CASCADE,
        match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
        track_id INTEGER, confidence REAL DEFAULT 0, is_verified BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMPTZ DEFAULT NOW(), UNIQUE(player_id, match_id)
    );
    CREATE TABLE IF NOT EXISTS match_comparisons (
        id SERIAL PRIMARY KEY, name TEXT DEFAULT '',
        match_id_1 INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
        match_id_2 INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
        comparison_type TEXT DEFAULT '', focus_areas JSONB DEFAULT '[]',
        notes TEXT DEFAULT '', created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS analysis_quality (
        id SERIAL PRIMARY KEY, match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
        overall_score REAL, tracking_score REAL, event_detection_score REAL,
        homography_score REAL, team_assignment_score REAL,
        issues JSONB DEFAULT '[]', warnings JSONB DEFAULT '[]',
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS exports (
        id SERIAL PRIMARY KEY, match_id INTEGER REFERENCES matches(id) ON DELETE SET NULL,
        season_id INTEGER REFERENCES seasons(id) ON DELETE SET NULL,
        export_type TEXT NOT NULL DEFAULT '', format TEXT NOT NULL DEFAULT '',
        file_path TEXT DEFAULT '', file_size_bytes INTEGER,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS batch_jobs (
        id SERIAL PRIMARY KEY, name TEXT NOT NULL, status TEXT DEFAULT 'pending',
        total_matches INTEGER DEFAULT 0, completed_matches INTEGER DEFAULT 0,
        failed_matches INTEGER DEFAULT 0, match_ids JSONB DEFAULT '[]',
        options JSONB DEFAULT '{}', started_at TIMESTAMPTZ, completed_at TIMESTAMPTZ,
        error_message TEXT DEFAULT '', created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS analysis_results (
        id SERIAL PRIMARY KEY, match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
        possession_home REAL, possession_away REAL,
        passes_home INTEGER, passes_away INTEGER,
        shots_home INTEGER, shots_away INTEGER,
        confidence_overall REAL, full_data JSONB DEFAULT '{}',
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS user_corrections (
        id SERIAL PRIMARY KEY, event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
        correction_type TEXT NOT NULL DEFAULT '', original_value TEXT DEFAULT '',
        corrected_value TEXT DEFAULT '', created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY, value TEXT DEFAULT '', updated_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY, applied_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS football_data_cache (
        cache_key TEXT PRIMARY KEY, data TEXT NOT NULL,
        expires_at DOUBLE PRECISION NOT NULL, created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS external_data_cache (
        cache_key TEXT PRIMARY KEY, data TEXT NOT NULL,
        expires_at DOUBLE PRECISION NOT NULL, created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS audit_events (
        id SERIAL PRIMARY KEY, action TEXT NOT NULL, entity_type TEXT NOT NULL DEFAULT '',
        entity_id TEXT, details_json JSONB DEFAULT '{}', user_name TEXT NOT NULL DEFAULT 'local',
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS match_weather (
        id SERIAL PRIMARY KEY, match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
        latitude REAL, longitude REAL, temperature_c REAL, feels_like_c REAL,
        precipitation_mm REAL, wind_speed_kmh REAL, wind_direction_deg REAL,
        humidity_pct REAL, cloud_cover_pct REAL, conditions TEXT DEFAULT '',
        pitch_state TEXT DEFAULT '', source TEXT DEFAULT '', recorded_at TIMESTAMPTZ
    );
    CREATE TABLE IF NOT EXISTS card_events (
        id SERIAL PRIMARY KEY, match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
        player_track_id INTEGER, player_name TEXT DEFAULT '', card_type TEXT NOT NULL,
        minute INTEGER NOT NULL DEFAULT 0, second INTEGER DEFAULT 0,
        detection_source TEXT DEFAULT '', confidence REAL, description TEXT DEFAULT ''
    );
    CREATE TABLE IF NOT EXISTS psychology_events (
        id SERIAL PRIMARY KEY, match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
        event_type TEXT NOT NULL, minute INTEGER NOT NULL DEFAULT 0,
        second INTEGER DEFAULT 0, team TEXT DEFAULT '', description TEXT DEFAULT '',
        severity REAL, data_json JSONB DEFAULT '{}'
    );
    CREATE TABLE IF NOT EXISTS player_shortlist (
        id SERIAL PRIMARY KEY, player_id TEXT NOT NULL, player_name TEXT NOT NULL,
        position TEXT DEFAULT '', team TEXT DEFAULT '', league TEXT DEFAULT '',
        added_date TIMESTAMPTZ DEFAULT NOW(),
        priority TEXT NOT NULL DEFAULT 'medium' CHECK(priority IN ('low','medium','high','urgent')),
        status TEXT NOT NULL DEFAULT 'scouted' CHECK(status IN ('scouted','shortlisted','contacted','trial','signed','rejected','archived')),
        notes TEXT DEFAULT '', scout_rating REAL DEFAULT 0 CHECK(scout_rating >= 0 AND scout_rating <= 10),
        estimated_value REAL, age INTEGER, nationality TEXT DEFAULT '',
        last_updated TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS player_contracts (
        id SERIAL PRIMARY KEY, player_profile_id INTEGER NOT NULL REFERENCES player_profiles(id) ON DELETE CASCADE,
        player_name TEXT NOT NULL,
        contract_type TEXT NOT NULL DEFAULT 'permanent' CHECK(contract_type IN ('permanent','loan','youth','scholar','trial')),
        start_date TEXT NOT NULL, end_date TEXT NOT NULL,
        club_option_years INTEGER DEFAULT 0, player_option_years INTEGER DEFAULT 0,
        release_clause_millions REAL, wage_weekly_pounds REAL,
        agent_name TEXT DEFAULT '', notes TEXT DEFAULT '',
        last_updated TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS collab_users (
        id SERIAL PRIMARY KEY, username TEXT UNIQUE NOT NULL,
        role TEXT DEFAULT 'analyst', created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS collab_comments (
        id SERIAL PRIMARY KEY, match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
        event_id INTEGER DEFAULT 0, user_id INTEGER DEFAULT 0,
        username TEXT DEFAULT '', text TEXT NOT NULL,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS collab_mentions (
        id SERIAL PRIMARY KEY, username TEXT NOT NULL, from_user TEXT NOT NULL,
        text TEXT NOT NULL, match_id INTEGER DEFAULT 0, event_id INTEGER DEFAULT 0,
        read INTEGER DEFAULT 0, created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS wearable_sessions (
        id SERIAL PRIMARY KEY, match_id INTEGER REFERENCES matches(id) ON DELETE SET NULL,
        athlete_id TEXT DEFAULT '', athlete_name TEXT DEFAULT '',
        device_type TEXT NOT NULL, device_serial TEXT DEFAULT '',
        start_time TEXT DEFAULT '', duration_s REAL DEFAULT 0,
        sample_rate_hz REAL DEFAULT 0, avg_hr REAL, max_hr REAL, min_hr REAL,
        total_distance_m REAL DEFAULT 0, max_speed_ms REAL, avg_speed_ms REAL,
        player_load REAL, body_load REAL, high_speed_running_m REAL DEFAULT 0,
        sprint_distance_m REAL DEFAULT 0, accelerations INTEGER DEFAULT 0,
        decelerations INTEGER DEFAULT 0, point_count INTEGER DEFAULT 0,
        metadata_json JSONB DEFAULT '{}',
        created_at TIMESTAMPTZ DEFAULT NOW(), updated_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS injuries (
        id SERIAL PRIMARY KEY, player_id INTEGER NOT NULL REFERENCES player_profiles(id) ON DELETE CASCADE,
        match_id INTEGER REFERENCES matches(id) ON DELETE SET NULL,
        injury_type TEXT NOT NULL, body_part TEXT NOT NULL,
        severity TEXT NOT NULL DEFAULT 'minor', mechanism TEXT DEFAULT '',
        date_injured TEXT NOT NULL, date_recovered TEXT,
        status TEXT NOT NULL DEFAULT 'active', notes TEXT DEFAULT '',
        created_at TIMESTAMPTZ DEFAULT NOW(), updated_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS rehab_plans (
        id SERIAL PRIMARY KEY, injury_id INTEGER NOT NULL REFERENCES injuries(id) ON DELETE CASCADE,
        phase TEXT NOT NULL DEFAULT 'initial', start_date TEXT NOT NULL,
        target_end_date TEXT, actual_end_date TEXT, milestones JSONB DEFAULT '[]',
        protocols TEXT DEFAULT '', status TEXT NOT NULL DEFAULT 'active',
        notes TEXT DEFAULT '', created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS concussion_assessments (
        id SERIAL PRIMARY KEY, player_id INTEGER NOT NULL REFERENCES player_profiles(id) ON DELETE CASCADE,
        match_id INTEGER REFERENCES matches(id) ON DELETE SET NULL,
        assessment_date TEXT NOT NULL, assessment_type TEXT NOT NULL DEFAULT 'scat5',
        symptoms_score INTEGER DEFAULT 0, cognitive_score INTEGER DEFAULT 0,
        balance_score INTEGER DEFAULT 0, clearance_status TEXT DEFAULT 'not_cleared',
        cleared_by TEXT DEFAULT '', notes TEXT DEFAULT '',
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS medical_history (
        id SERIAL PRIMARY KEY, player_id INTEGER NOT NULL REFERENCES player_profiles(id) ON DELETE CASCADE,
        condition_type TEXT NOT NULL, diagnosis TEXT NOT NULL,
        diagnosis_date TEXT NOT NULL, status TEXT DEFAULT 'active',
        severity TEXT DEFAULT 'moderate', notes TEXT DEFAULT '',
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    """  # noqa: E501


# ================================================================
# Training OS cluster (migration 032) — explicit Postgres stubs.
#
# These tables and their full behavior currently exist only in the
# SQLite StorageService (services/training_storage.py mixin). Stubs are
# deliberate, not an omission: an inherited no-op would silently
# "succeed" (write nothing, return None) in cloud mode and corrupt the
# training loop's data guarantees. Every method here fails loudly, and
# grep 'HONEST-UNIMPLEMENTED-PG' finds all of them.
# ================================================================


def _make_honest_stub(name: str):
    """Build a fail-loud coroutine method with a stable, greppable error."""
    # HONEST-UNIMPLEMENTED-PG factory — see block comment above.

    async def _stub(*_args: Any, **_kwargs: Any) -> Any:
        raise StorageNotInitializedError(f"{name}: not yet implemented on Postgres")

    _stub.__name__ = name
    _stub.__qualname__ = f"PostgresStorageAdapter.{name}"
    return _stub


for _name in [
    "save_game_model",
    "get_active_game_model",
    "get_active_game_model_for_match",
    "get_game_model_history",
    "save_training_plan",
    "update_training_plan_payload",
    "set_training_plan_status",
    "get_training_plan",
    "get_training_plans_for_match",
    "list_training_plans",
    "get_plan_versions",
    "save_training_session",
    "set_session_status",
    "get_training_sessions",
    "get_training_session",
    "add_session_drill",
    "set_drill_execution",
    "get_session_drills",
    "save_attendance",
    "get_session_attendance",
    "save_session_rpe",
    "get_session_rpe",
    "get_player_rpe_history",
    "save_wellness",
    "get_player_wellness",
    "get_squad_wellness_latest",
    "save_drill_feedback",
    "get_drill_feedback",
    "save_testing_result",
    "get_player_testing_history",
    "get_testing_distribution",
    "save_idp_goal",
    "update_idp_goal",
    "get_player_idp",
    "save_nutrition_log",
    "get_nutrition_logs",
    "save_psych_checkin",
    "get_psych_checkins",
    "get_flagged_psych_checkins",
    "save_ritual",
    "complete_ritual",
    "get_rituals",
    "set_medical_clearance",
    "get_latest_clearance",
    "get_clearances_by_status",
    "save_minutes_entry",
    "get_player_minutes_history",
    "get_squad_minutes_summary",
    "get_player_profile",
    "save_program_document",
    "get_current_program_document",
    "list_program_documents",
    "save_evidence_record",
    "get_evidence_records",
]:
    # HONEST-UNIMPLEMENTED-PG: fail-loud stub (see block comment above).
    setattr(
        PostgresStorageAdapter,
        _name,
        _make_honest_stub(_name),
    )
