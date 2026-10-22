-- Migration 031: Match external-ID registry (season-scale vendor imports)
--
-- Elite clubs import a whole season (hundreds of matches) from a vendor
-- feed. Re-running an import must not duplicate matches: this table maps
-- a vendor's own match identifier (StatsBomb match id / file stem, Opta
-- fixture id, ...) to the internal matches.id, with a UNIQUE constraint
-- that makes dedup a lookup instead of a guess.
--
-- Companion indexes on matches.competition / matches.season_id support
-- the season-dashboard queries the bulk import feeds.

CREATE TABLE IF NOT EXISTS matches_external_ids (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    source TEXT NOT NULL,          -- 'statsbomb' | 'opta' | 'wyscout' | 'skillcorner' | ...
    external_id TEXT NOT NULL,     -- vendor's own match identifier, as text
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source, external_id)
);

CREATE INDEX IF NOT EXISTS idx_matches_external_ids_match ON matches_external_ids(match_id);

-- Season/squad-level dashboards filter on these two columns for every
-- match-list render, and bulk imports write them on every match row.
CREATE INDEX IF NOT EXISTS idx_matches_competition ON matches(competition);
CREATE INDEX IF NOT EXISTS idx_matches_season ON matches(season_id);
