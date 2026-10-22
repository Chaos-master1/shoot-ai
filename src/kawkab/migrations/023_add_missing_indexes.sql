-- Migration 023: Add missing indexes on player_profiles

CREATE INDEX IF NOT EXISTS idx_profiles_team ON player_profiles(team);
CREATE INDEX IF NOT EXISTS idx_profiles_global_id ON player_profiles(global_id);
