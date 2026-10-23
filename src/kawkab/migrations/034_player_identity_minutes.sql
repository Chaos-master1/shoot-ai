-- Migration 034: player identity + minutes played
--
-- The team benchmark compares team-matches; a position-matched player
-- benchmark needs to know WHICH player and HOW LONG they were on.
-- `players` already had track_id (per-match, synthetic) and position, but
-- no stable source identity and no minutes. The importer registered only
-- players who touched the ball and guessed minutes from nothing.
--
-- external_id is the source's player id as text (StatsBomb's integer id
-- for imported matches; '' for tracking-only matches with no identity).
-- It is deliberately not a foreign key: different providers use different
-- namespaces, and this column is provenance, not a join key.
--
-- minutes_played is regulation/extra time (90/120), not wall-clock
-- stoppage time. 0 means "not established", which the player benchmark
-- reports as unmeasurable rather than treating as a 0-minute appearance.

ALTER TABLE players ADD COLUMN external_id TEXT DEFAULT '';

ALTER TABLE players ADD COLUMN minutes_played REAL DEFAULT 0.0;

CREATE INDEX IF NOT EXISTS idx_players_external_id ON players(external_id);
