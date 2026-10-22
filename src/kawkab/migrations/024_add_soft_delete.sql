-- Migration 024: Add soft-delete columns to key tables

ALTER TABLE matches ADD COLUMN is_deleted INTEGER DEFAULT 0;
ALTER TABLE matches ADD COLUMN deleted_at TEXT;
ALTER TABLE matches ADD COLUMN deleted_by TEXT;

ALTER TABLE events ADD COLUMN is_deleted INTEGER DEFAULT 0;
ALTER TABLE events ADD COLUMN deleted_at TEXT;
ALTER TABLE events ADD COLUMN deleted_by TEXT;

ALTER TABLE players ADD COLUMN is_deleted INTEGER DEFAULT 0;
ALTER TABLE players ADD COLUMN deleted_at TEXT;
ALTER TABLE players ADD COLUMN deleted_by TEXT;

ALTER TABLE coding_tags ADD COLUMN is_deleted INTEGER DEFAULT 0;
ALTER TABLE coding_tags ADD COLUMN deleted_at TEXT;
ALTER TABLE coding_tags ADD COLUMN deleted_by TEXT;

CREATE INDEX IF NOT EXISTS idx_matches_deleted ON matches(is_deleted);
CREATE INDEX IF NOT EXISTS idx_events_deleted ON events(is_deleted);
CREATE INDEX IF NOT EXISTS idx_players_deleted ON players(is_deleted);
CREATE INDEX IF NOT EXISTS idx_coding_tags_deleted ON coding_tags(is_deleted);
