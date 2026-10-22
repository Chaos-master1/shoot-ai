-- Migration 028: Add match ownership for cloud/multi-user deployments
--
-- matches previously had no owner/team concept at all -- every row was
-- readable by any authenticated API v1 caller regardless of who created
-- it. owner_id/team_id are plain integers with no FK constraint: in
-- SQLite mode the cloud auth DB (users/teams) and this DB (matches) are
-- two separate files, so a cross-database foreign key isn't possible;
-- ownership is validated at the application layer instead (see
-- api_v1.py's _check_match_access). NULL owner_id means "no owner
-- assigned" (e.g. every match created via the existing desktop pipeline,
-- which has no per-user concept at all) and is treated as visible to
-- all authenticated users -- this migration does not retroactively
-- restrict access to data that was previously unrestricted.

ALTER TABLE matches ADD COLUMN owner_id INTEGER;
ALTER TABLE matches ADD COLUMN team_id INTEGER;
ALTER TABLE matches ADD COLUMN is_shared INTEGER DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_matches_owner ON matches(owner_id);
CREATE INDEX IF NOT EXISTS idx_matches_team ON matches(team_id);
