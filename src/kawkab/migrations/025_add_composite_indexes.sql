-- Migration 025: Add missing composite indexes for common query patterns
-- Addresses: MED-3 from professional readiness audit

-- events: filter by (match_id, event_type) for timeline filtering
CREATE INDEX IF NOT EXISTS idx_events_match_type ON events(match_id, event_type);

-- events: filter by (match_id, timestamp) for time-range queries
CREATE INDEX IF NOT EXISTS idx_events_match_time ON events(match_id, timestamp);

-- events: filter by team for per-team analysis
CREATE INDEX IF NOT EXISTS idx_events_team ON events(team);

-- players: lookup by (match_id, track_id) for per-player queries
CREATE INDEX IF NOT EXISTS idx_players_match_track ON players(match_id, track_id);

-- players: filter by team for squad views
CREATE INDEX IF NOT EXISTS idx_players_team ON players(team);

-- player_profiles: active players per team
CREATE INDEX IF NOT EXISTS idx_profiles_team_active ON player_profiles(team, is_active);

-- video_clips: filter by (match_id, event_type) for clip browsing
CREATE INDEX IF NOT EXISTS idx_clips_match_type ON video_clips(match_id, event_type);

-- reports: per-match per-language reports
CREATE INDEX IF NOT EXISTS idx_reports_match_lang ON reports(match_id, language);

-- coding_tags: composite type+time for timeline queries
CREATE INDEX IF NOT EXISTS idx_coding_tags_match_type_time ON coding_tags(match_id, event_type, video_time);
