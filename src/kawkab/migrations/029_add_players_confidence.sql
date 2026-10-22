-- Migration 029: add players.confidence
--
-- StorageService.get_match_players() has always SELECTed a `confidence`
-- column that no SQLite migration ever created -- 001_initial.sql's players
-- table never had one, and nothing since added it. Every call raised
-- sqlite3.OperationalError: no such column: confidence, on every real
-- SQLite deployment. Postgres was never affected: pg_schema.sql's players
-- table already declares confidence REAL DEFAULT 0. Nothing populates this
-- column yet (save_player/save_players_bulk don't set it), so it stays
-- NULL until a real per-player detection-confidence source exists -- this
-- migration only stops the crash, it doesn't fabricate a value.

ALTER TABLE players ADD COLUMN confidence REAL DEFAULT 0.0;
