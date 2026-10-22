-- pg_schema.sql — Full PostgreSQL schema for Kawkab AI (43 tables)
-- Mirrors all migrations 001–025 with proper PG types, FKs, indexes, RLS.
-- Apply: psql -U postgres -d kawkab -f pg_schema.sql

BEGIN;

-- Extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ── updated_at trigger function ─────────────────────────────────────────
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ── 0. Independent tables (no FK deps) ─────────────────────────────────
CREATE TABLE IF NOT EXISTS teams (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    short_name      TEXT DEFAULT '',
    logo_url        TEXT DEFAULT '',
    country         TEXT DEFAULT '',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS seasons (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    start_date      DATE,
    end_date        DATE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── 1. matches ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS matches (
    id                  SERIAL PRIMARY KEY,
    name                TEXT NOT NULL,
    video_path          TEXT NOT NULL DEFAULT '',
    home_team           TEXT DEFAULT '',
    away_team           TEXT DEFAULT '',
    match_date          TIMESTAMPTZ,
    duration_seconds    REAL,
    fps                 REAL,
    total_frames        INTEGER,
    home_team_id        INTEGER,
    away_team_id        INTEGER,
    season_id           INTEGER,
    competition         TEXT DEFAULT '',
    round               TEXT DEFAULT '',
    opponent            TEXT DEFAULT '',
    score_home          INTEGER,
    score_away          INTEGER,
    match_type          TEXT DEFAULT 'unknown',
    api_match_id        INTEGER,
    competition_code    TEXT DEFAULT '',
    football_data_home_team_id INTEGER,
    football_data_away_team_id INTEGER,
    bzzoiro_home_team_id       INTEGER,
    bzzoiro_away_team_id       INTEGER,
    bzzoiro_event_id           INTEGER,
    bzzoiro_league_id          INTEGER,
    bzzoiro_competition_code   TEXT DEFAULT '',
    prediction_data     TEXT DEFAULT '',
    apifb_home_team_id  INTEGER,
    apifb_away_team_id  INTEGER,
    apifb_fixture_id    INTEGER,
    apifb_league_id     INTEGER,
    apifb_season        INTEGER,
    analysis_json       JSONB DEFAULT '{}',
    football_data_json  JSONB DEFAULT '{}',
    apifootball_json    JSONB DEFAULT '{}',
    bzzoiro_json        JSONB DEFAULT '{}',
    is_deleted          INTEGER DEFAULT 0,
    deleted_at          TIMESTAMPTZ,
    deleted_by          TEXT DEFAULT '',
    -- No FK to users/teams: in SQLite mode the cloud auth DB and this DB
    -- are separate files, so ownership is validated at the application
    -- layer (api_v1.py) rather than enforced by the schema -- see
    -- migrations/028_add_match_ownership.sql for the SQLite side and the
    -- full rationale.
    owner_id            INTEGER,
    team_id             INTEGER,
    is_shared           INTEGER DEFAULT 0,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW(),
    analyzed_at         TIMESTAMPTZ
);

DROP TRIGGER IF EXISTS trg_matches_updated_at ON matches;
CREATE TRIGGER trg_matches_updated_at
    BEFORE UPDATE ON matches FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ── 2. players ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS players (
    id                  SERIAL PRIMARY KEY,
    match_id            INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    track_id            INTEGER NOT NULL DEFAULT 0,
    jersey_number       INTEGER,
    name                TEXT DEFAULT '',
    team                TEXT DEFAULT '',
    position            TEXT DEFAULT '',
    distance_covered_m  REAL DEFAULT 0,
    max_speed_kmh       REAL DEFAULT 0,
    avg_speed_kmh       REAL DEFAULT 0,
    passes_attempted    INTEGER DEFAULT 0,
    passes_completed    INTEGER DEFAULT 0,
    shots               INTEGER DEFAULT 0,
    tackles             INTEGER DEFAULT 0,
    confidence          REAL DEFAULT 0,
    is_deleted          INTEGER DEFAULT 0,
    deleted_at          TIMESTAMPTZ,
    deleted_by          TEXT DEFAULT ''
);

-- ── 3. events ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS events (
    id                  SERIAL PRIMARY KEY,
    match_id            INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    event_type          TEXT NOT NULL DEFAULT '',
    timestamp           DOUBLE PRECISION NOT NULL DEFAULT 0,
    from_track_id       INTEGER DEFAULT 0,
    to_track_id         INTEGER,
    team                TEXT DEFAULT '',
    completed           BOOLEAN DEFAULT FALSE,
    confidence          REAL DEFAULT 0,
    metadata            JSONB DEFAULT '{}',
    user_corrected      BOOLEAN DEFAULT FALSE,
    x                   DOUBLE PRECISION DEFAULT 0,
    y                   DOUBLE PRECISION DEFAULT 0,
    end_x               DOUBLE PRECISION DEFAULT 0,
    end_y               DOUBLE PRECISION DEFAULT 0,
    is_goal             BOOLEAN DEFAULT FALSE,
    data                JSONB DEFAULT '{}',
    is_deleted          INTEGER DEFAULT 0,
    deleted_at          TIMESTAMPTZ,
    deleted_by          TEXT DEFAULT ''
);

-- ── 4. analysis_results ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS analysis_results (
    id                  SERIAL PRIMARY KEY,
    match_id            INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    possession_home     REAL,
    possession_away     REAL,
    passes_home         INTEGER,
    passes_away         INTEGER,
    shots_home          INTEGER,
    shots_away          INTEGER,
    confidence_overall  REAL,
    full_data           JSONB DEFAULT '{}',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── 5. reports ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS reports (
    id                  SERIAL PRIMARY KEY,
    match_id            INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    language            TEXT NOT NULL DEFAULT 'en',
    report_text         TEXT NOT NULL DEFAULT '',
    report_type         TEXT DEFAULT 'match',
    llm_provider        TEXT DEFAULT '',
    content             TEXT DEFAULT '',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── 6. user_corrections ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS user_corrections (
    id                  SERIAL PRIMARY KEY,
    event_id            INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    correction_type     TEXT NOT NULL DEFAULT '',
    original_value      TEXT DEFAULT '',
    corrected_value     TEXT DEFAULT '',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── 7. settings ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS settings (
    key                 TEXT PRIMARY KEY,
    value               TEXT DEFAULT '',
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── 8. schema_version ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS schema_version (
    version             INTEGER PRIMARY KEY,
    applied_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── 9. seasons ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS seasons (
    id                  SERIAL PRIMARY KEY,
    name                TEXT NOT NULL,
    team_name           TEXT DEFAULT '',
    competition         TEXT DEFAULT '',
    start_date          TIMESTAMPTZ,
    end_date            TIMESTAMPTZ,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── 10. player_profiles ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS player_profiles (
    id                  SERIAL PRIMARY KEY,
    global_id           TEXT UNIQUE NOT NULL DEFAULT '',
    display_name        TEXT DEFAULT '',
    name                TEXT DEFAULT '',
    team                TEXT DEFAULT '',
    position            TEXT DEFAULT '',
    preferred_position  TEXT DEFAULT '',
    jersey_number       INTEGER DEFAULT 0,
    height_cm           INTEGER,
    weight_kg           INTEGER,
    dominant_foot       TEXT DEFAULT '',
    date_of_birth       TIMESTAMPTZ,
    nationality         TEXT DEFAULT '',
    photo_path          TEXT DEFAULT '',
    notes               JSONB DEFAULT '{}',
    is_active           BOOLEAN DEFAULT TRUE,
    face_embedding      TEXT DEFAULT '',
    face_confidence     REAL DEFAULT 0,
    football_data_person_id INTEGER,
    football_data_team_id   INTEGER,
    bzzoiro_person_id   INTEGER,
    bzzoiro_team_id     INTEGER,
    apifb_person_id     INTEGER,
    apifb_team_id       INTEGER,
    contract_end_date   TEXT DEFAULT '',
    contract_type       TEXT DEFAULT 'permanent',
    wage_weekly_pounds  REAL,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

DROP TRIGGER IF EXISTS trg_player_profiles_updated_at ON player_profiles;
CREATE TRIGGER trg_player_profiles_updated_at
    BEFORE UPDATE ON player_profiles FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ── 11. player_match_links ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS player_match_links (
    id                  SERIAL PRIMARY KEY,
    player_id           INTEGER NOT NULL REFERENCES player_profiles(id) ON DELETE CASCADE,
    match_id            INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    track_id            INTEGER,
    confidence          REAL DEFAULT 0,
    is_verified         BOOLEAN DEFAULT FALSE,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(player_id, match_id)
);

-- ── 12. advanced_metrics ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS advanced_metrics (
    id                  SERIAL PRIMARY KEY,
    match_id            INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    player_id           INTEGER REFERENCES player_profiles(id) ON DELETE SET NULL,
    metric_name         TEXT NOT NULL DEFAULT '',
    metric_value        REAL,
    category            TEXT DEFAULT '',
    metric_category     TEXT DEFAULT '',
    pitch_zone          TEXT DEFAULT '',
    timestamp           REAL,
    data_json           JSONB DEFAULT '{}',
    metadata            JSONB DEFAULT '{}',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── 13. match_comparisons ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS match_comparisons (
    id                  SERIAL PRIMARY KEY,
    name                TEXT DEFAULT '',
    match_id_1          INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    match_id_2          INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    comparison_type     TEXT DEFAULT '',
    focus_areas         JSONB DEFAULT '[]',
    notes               TEXT DEFAULT '',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── 14. analysis_quality ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS analysis_quality (
    id                  SERIAL PRIMARY KEY,
    match_id            INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    overall_score       REAL,
    tracking_score      REAL,
    event_detection_score REAL,
    homography_score    REAL,
    team_assignment_score REAL,
    issues              JSONB DEFAULT '[]',
    warnings            JSONB DEFAULT '[]',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── 15. exports ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS exports (
    id                  SERIAL PRIMARY KEY,
    match_id            INTEGER REFERENCES matches(id) ON DELETE SET NULL,
    season_id           INTEGER REFERENCES seasons(id) ON DELETE SET NULL,
    export_type         TEXT NOT NULL DEFAULT '',
    format              TEXT NOT NULL DEFAULT '',
    file_path           TEXT DEFAULT '',
    file_size_bytes     INTEGER,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── 16. benchmark_results ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS benchmark_results (
    id                  SERIAL PRIMARY KEY,
    match_id            INTEGER REFERENCES matches(id) ON DELETE SET NULL,
    video_path          TEXT DEFAULT '',
    video_duration_seconds REAL,
    total_frames        INTEGER,
    total_time_seconds  REAL,
    total_time          REAL DEFAULT 0,
    realtime_ratio      REAL,
    fps_effective       REAL,
    fps                 REAL DEFAULT 0,
    stage_enhancement_seconds REAL,
    stage_detection_seconds  REAL,
    stage_tracking_seconds   REAL,
    stage_analysis_seconds   REAL,
    stage_advanced_metrics_seconds REAL,
    stage_save_seconds  REAL,
    peak_memory_mb      REAL,
    peak_memory         REAL DEFAULT 0,
    peak_gpu_memory_mb  REAL,
    peak_gpu_memory     REAL DEFAULT 0,
    gpu_utilization_pct REAL,
    gpu_name            TEXT DEFAULT '',
    cpu_name            TEXT DEFAULT '',
    ram_gb              REAL,
    model_size          TEXT DEFAULT '',
    model_name          TEXT DEFAULT '',
    frame_skip          INTEGER DEFAULT 0,
    avg_fps             REAL DEFAULT 0,
    avg_latency_ms      REAL DEFAULT 0,
    stages_json         JSONB DEFAULT '{}',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── 17. validation_results ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS validation_results (
    id                  SERIAL PRIMARY KEY,
    match_id            INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    ground_truth_source TEXT DEFAULT '',
    overall_accuracy    REAL,
    category            TEXT DEFAULT '',
    metric_name         TEXT NOT NULL DEFAULT '',
    computed_value      REAL,
    ground_truth_value  REAL,
    absolute_error      REAL,
    relative_error_pct  REAL,
    accuracy_score      REAL,
    sample_count        INTEGER,
    accuracy            REAL DEFAULT 0,
    details_json        JSONB DEFAULT '{}',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── 18. batch_jobs ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS batch_jobs (
    id                  SERIAL PRIMARY KEY,
    name                TEXT NOT NULL,
    status              TEXT DEFAULT 'pending',
    total_matches       INTEGER DEFAULT 0,
    completed_matches   INTEGER DEFAULT 0,
    failed_matches      INTEGER DEFAULT 0,
    match_ids           JSONB DEFAULT '[]',
    options             JSONB DEFAULT '{}',
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    error_message       TEXT DEFAULT '',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── 19. feedback (coach_feedback) ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS feedback (
    id                  SERIAL PRIMARY KEY,
    coach_id            TEXT NOT NULL DEFAULT '',
    match_id            INTEGER REFERENCES matches(id) ON DELETE SET NULL,
    user_name           TEXT DEFAULT '',
    overall_rating      INTEGER CHECK(overall_rating BETWEEN 1 AND 5),
    rating              INTEGER DEFAULT 0,
    tracking_rating     INTEGER CHECK(tracking_rating BETWEEN 1 AND 5),
    events_rating       INTEGER CHECK(events_rating BETWEEN 1 AND 5),
    report_rating       INTEGER CHECK(report_rating BETWEEN 1 AND 5),
    ui_rating           INTEGER CHECK(ui_rating BETWEEN 1 AND 5),
    comments            TEXT DEFAULT '',
    issues              JSONB DEFAULT '[]',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── 20. issues (issue_reports) ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS issues (
    id                  SERIAL PRIMARY KEY,
    category            TEXT NOT NULL DEFAULT 'other',
    severity            TEXT NOT NULL DEFAULT 'low',
    description         TEXT NOT NULL DEFAULT '',
    match_id            INTEGER REFERENCES matches(id) ON DELETE SET NULL,
    screenshot_path     TEXT DEFAULT '',
    logs                TEXT DEFAULT '',
    status              TEXT DEFAULT 'open',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── 21. usage_sessions ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS usage_sessions (
    id                  SERIAL PRIMARY KEY,
    session_id          TEXT UNIQUE NOT NULL DEFAULT '',
    match_id            INTEGER REFERENCES matches(id) ON DELETE SET NULL,
    user_name           TEXT DEFAULT '',
    features_used       JSONB NOT NULL DEFAULT '[]',
    action              TEXT DEFAULT '',
    duration_seconds    REAL NOT NULL DEFAULT 0,
    duration_s          REAL DEFAULT 0,
    match_count         INTEGER DEFAULT 0,
    gpu_tier            TEXT DEFAULT '',
    model_size          TEXT DEFAULT '',
    error_count         INTEGER DEFAULT 0,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── 22. clips (video_clips) ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS clips (
    id                  SERIAL PRIMARY KEY,
    match_id            INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    event_type          TEXT NOT NULL DEFAULT '',
    name                TEXT DEFAULT '',
    start_seconds       REAL NOT NULL DEFAULT 0,
    start_time          DOUBLE PRECISION DEFAULT 0,
    end_seconds         REAL NOT NULL DEFAULT 0,
    end_time            DOUBLE PRECISION DEFAULT 0,
    duration_seconds    REAL NOT NULL DEFAULT 0,
    source_video_path   TEXT NOT NULL DEFAULT '',
    video_path          TEXT DEFAULT '',
    output_path         TEXT NOT NULL DEFAULT '',
    thumbnail_path      TEXT DEFAULT '',
    player_id           INTEGER,
    player_track_id     INTEGER DEFAULT 0,
    description         TEXT DEFAULT '',
    tags_json           JSONB DEFAULT '[]',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── 23. playlists (clip_playlists) ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS playlists (
    id                  SERIAL PRIMARY KEY,
    name                TEXT NOT NULL DEFAULT '',
    description         TEXT DEFAULT '',
    match_id            INTEGER REFERENCES matches(id) ON DELETE CASCADE,
    clip_ids            JSONB NOT NULL DEFAULT '[]',
    clips_json          JSONB DEFAULT '[]',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── 24. football_data_cache ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS football_data_cache (
    cache_key           TEXT PRIMARY KEY,
    data                TEXT NOT NULL,
    expires_at          DOUBLE PRECISION NOT NULL,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── 25. external_data_cache ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS external_data_cache (
    cache_key           TEXT PRIMARY KEY,
    data                TEXT NOT NULL,
    expires_at          DOUBLE PRECISION NOT NULL,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── 26. audit_events ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_events (
    id                  SERIAL PRIMARY KEY,
    action              TEXT NOT NULL,
    entity_type         TEXT NOT NULL DEFAULT '',
    entity_id           TEXT,
    details_json        JSONB DEFAULT '{}',
    user_name           TEXT NOT NULL DEFAULT 'local',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── 27. match_weather ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS match_weather (
    id                  SERIAL PRIMARY KEY,
    match_id            INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    latitude            REAL,
    longitude           REAL,
    temperature_c       REAL,
    feels_like_c        REAL,
    precipitation_mm    REAL,
    wind_speed_kmh      REAL,
    wind_direction_deg  REAL,
    humidity_pct        REAL,
    cloud_cover_pct     REAL,
    conditions          TEXT DEFAULT '',
    pitch_state         TEXT DEFAULT '',
    source              TEXT DEFAULT '',
    recorded_at         TIMESTAMPTZ
);

-- ── 28. card_events ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS card_events (
    id                  SERIAL PRIMARY KEY,
    match_id            INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    player_track_id     INTEGER,
    player_name         TEXT DEFAULT '',
    card_type           TEXT NOT NULL,
    minute              INTEGER NOT NULL DEFAULT 0,
    second              INTEGER DEFAULT 0,
    detection_source    TEXT DEFAULT '',
    confidence          REAL,
    description         TEXT DEFAULT ''
);

-- ── 29. psychology_events ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS psychology_events (
    id                  SERIAL PRIMARY KEY,
    match_id            INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    event_type          TEXT NOT NULL,
    minute              INTEGER NOT NULL DEFAULT 0,
    second              INTEGER DEFAULT 0,
    team                TEXT DEFAULT '',
    description         TEXT DEFAULT '',
    severity            REAL,
    data_json           JSONB DEFAULT '{}'
);

-- ── 30. player_shortlist ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS player_shortlist (
    id                  SERIAL PRIMARY KEY,
    player_id           TEXT NOT NULL,
    player_name         TEXT NOT NULL,
    position            TEXT DEFAULT '',
    team                TEXT DEFAULT '',
    league              TEXT DEFAULT '',
    added_date          TIMESTAMPTZ DEFAULT NOW(),
    priority            TEXT NOT NULL DEFAULT 'medium' CHECK(priority IN ('low','medium','high','urgent')),
    status              TEXT NOT NULL DEFAULT 'scouted' CHECK(status IN ('scouted','shortlisted','contacted','trial','signed','rejected','archived')),
    notes               TEXT DEFAULT '',
    scout_rating        REAL DEFAULT 0 CHECK(scout_rating >= 0 AND scout_rating <= 10),
    estimated_value     REAL,
    age                 INTEGER,
    nationality         TEXT DEFAULT '',
    last_updated        TIMESTAMPTZ DEFAULT NOW()
);

-- ── 31. player_contracts ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS player_contracts (
    id                  SERIAL PRIMARY KEY,
    player_profile_id   INTEGER NOT NULL REFERENCES player_profiles(id) ON DELETE CASCADE,
    player_name         TEXT NOT NULL,
    contract_type       TEXT NOT NULL DEFAULT 'permanent' CHECK(contract_type IN ('permanent','loan','youth','scholar','trial')),
    start_date          TEXT NOT NULL,
    end_date            TEXT NOT NULL,
    club_option_years   INTEGER DEFAULT 0,
    player_option_years INTEGER DEFAULT 0,
    release_clause_millions REAL,
    wage_weekly_pounds  REAL,
    agent_name          TEXT DEFAULT '',
    notes               TEXT DEFAULT '',
    last_updated        TIMESTAMPTZ DEFAULT NOW()
);

-- ── 32. coding_tags ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS coding_tags (
    id                  SERIAL PRIMARY KEY,
    match_id            INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    event_type          TEXT NOT NULL DEFAULT '',
    tag_type            TEXT DEFAULT '',
    sub_type            TEXT DEFAULT '',
    category            TEXT DEFAULT '',
    video_time          REAL NOT NULL DEFAULT 0,
    timestamp           DOUBLE PRECISION DEFAULT 0,
    player_track_id     INTEGER DEFAULT 0,
    player_name         TEXT DEFAULT '',
    team                TEXT DEFAULT '',
    period              INTEGER DEFAULT 1,
    notes               TEXT DEFAULT '',
    color               TEXT DEFAULT '#3498db',
    lead_ms             INTEGER DEFAULT 2000,
    lag_ms              INTEGER DEFAULT 3000,
    is_deleted          INTEGER DEFAULT 0,
    deleted_at          TIMESTAMPTZ,
    deleted_by          TEXT DEFAULT '',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── 33. collab_users ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS collab_users (
    id                  SERIAL PRIMARY KEY,
    username            TEXT UNIQUE NOT NULL,
    role                TEXT DEFAULT 'analyst',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── 34. collab_comments ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS collab_comments (
    id                  SERIAL PRIMARY KEY,
    match_id            INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    event_id            INTEGER DEFAULT 0,
    user_id             INTEGER DEFAULT 0,
    username            TEXT DEFAULT '',
    text                TEXT NOT NULL,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── 35. collab_mentions ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS collab_mentions (
    id                  SERIAL PRIMARY KEY,
    username            TEXT NOT NULL,
    from_user           TEXT NOT NULL,
    text                TEXT NOT NULL,
    match_id            INTEGER DEFAULT 0,
    event_id            INTEGER DEFAULT 0,
    read                INTEGER DEFAULT 0,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── 36. wearable_sessions ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS wearable_sessions (
    id                  SERIAL PRIMARY KEY,
    match_id            INTEGER REFERENCES matches(id) ON DELETE SET NULL,
    athlete_id          TEXT DEFAULT '',
    athlete_name        TEXT DEFAULT '',
    device_type         TEXT NOT NULL,
    device_serial       TEXT DEFAULT '',
    start_time          TEXT DEFAULT '',
    duration_s          REAL DEFAULT 0,
    sample_rate_hz      REAL DEFAULT 0,
    avg_hr              REAL,
    max_hr              REAL,
    min_hr              REAL,
    total_distance_m    REAL DEFAULT 0,
    max_speed_ms        REAL,
    avg_speed_ms        REAL,
    player_load         REAL,
    body_load           REAL,
    high_speed_running_m REAL DEFAULT 0,
    sprint_distance_m   REAL DEFAULT 0,
    accelerations       INTEGER DEFAULT 0,
    decelerations       INTEGER DEFAULT 0,
    point_count         INTEGER DEFAULT 0,
    metadata_json       JSONB DEFAULT '{}',
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

DROP TRIGGER IF EXISTS trg_wearable_sessions_updated_at ON wearable_sessions;
CREATE TRIGGER trg_wearable_sessions_updated_at
    BEFORE UPDATE ON wearable_sessions FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ── 37. injuries ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS injuries (
    id                  SERIAL PRIMARY KEY,
    player_id           INTEGER NOT NULL REFERENCES player_profiles(id) ON DELETE CASCADE,
    match_id            INTEGER REFERENCES matches(id) ON DELETE SET NULL,
    injury_type         TEXT NOT NULL,
    body_part           TEXT NOT NULL,
    severity            TEXT NOT NULL DEFAULT 'minor',
    mechanism           TEXT DEFAULT '',
    date_injured        TEXT NOT NULL,
    date_recovered      TEXT,
    status              TEXT NOT NULL DEFAULT 'active',
    notes               TEXT DEFAULT '',
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

DROP TRIGGER IF EXISTS trg_injuries_updated_at ON injuries;
CREATE TRIGGER trg_injuries_updated_at
    BEFORE UPDATE ON injuries FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ── 38. rehab_plans ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS rehab_plans (
    id                  SERIAL PRIMARY KEY,
    injury_id           INTEGER NOT NULL REFERENCES injuries(id) ON DELETE CASCADE,
    phase               TEXT NOT NULL DEFAULT 'initial',
    start_date          TEXT NOT NULL,
    target_end_date     TEXT,
    actual_end_date     TEXT,
    milestones          JSONB DEFAULT '[]',
    protocols           TEXT DEFAULT '',
    status              TEXT NOT NULL DEFAULT 'active',
    notes               TEXT DEFAULT '',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── 39. concussion_assessments ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS concussion_assessments (
    id                  SERIAL PRIMARY KEY,
    player_id           INTEGER NOT NULL REFERENCES player_profiles(id) ON DELETE CASCADE,
    match_id            INTEGER REFERENCES matches(id) ON DELETE SET NULL,
    assessment_date     TEXT NOT NULL,
    assessment_type     TEXT NOT NULL DEFAULT 'scat5',
    symptoms_score      INTEGER DEFAULT 0,
    cognitive_score     INTEGER DEFAULT 0,
    balance_score       INTEGER DEFAULT 0,
    clearance_status    TEXT DEFAULT 'not_cleared',
    cleared_by          TEXT DEFAULT '',
    notes               TEXT DEFAULT '',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── 40. medical_history ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS medical_history (
    id                  SERIAL PRIMARY KEY,
    player_id           INTEGER NOT NULL REFERENCES player_profiles(id) ON DELETE CASCADE,
    condition_type      TEXT NOT NULL,
    diagnosis           TEXT NOT NULL,
    diagnosis_date      TEXT NOT NULL,
    status              TEXT DEFAULT 'active',
    severity            TEXT DEFAULT 'moderate',
    notes               TEXT DEFAULT '',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── 41. encryption_keys ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS encryption_keys (
    id                  SERIAL PRIMARY KEY,
    key_name            TEXT UNIQUE NOT NULL,
    key_value           TEXT NOT NULL,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    rotated_at          TIMESTAMPTZ
);

-- Seed medical encryption key
INSERT INTO encryption_keys (key_name, key_value)
SELECT 'medical_v1', encode(gen_random_bytes(32), 'hex')
WHERE NOT EXISTS (SELECT 1 FROM encryption_keys WHERE key_name = 'medical_v1');

-- ── 42. tracking_frames ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tracking_frames (
    id                  SERIAL PRIMARY KEY,
    match_id            INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    frame_number        INTEGER NOT NULL,
    timestamp           DOUBLE PRECISION NOT NULL DEFAULT 0,
    player_detections   JSONB DEFAULT '[]',
    ball_detections     JSONB DEFAULT '[]',
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(match_id, frame_number)
);

-- ── INDICES ─────────────────────────────────────────────────────────────

-- matches
CREATE INDEX IF NOT EXISTS idx_matches_season ON matches(season_id);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date);
CREATE INDEX IF NOT EXISTS idx_matches_home_team ON matches(home_team_id);
CREATE INDEX IF NOT EXISTS idx_matches_away_team ON matches(away_team_id);
CREATE INDEX IF NOT EXISTS idx_matches_deleted ON matches(is_deleted);

-- events
CREATE INDEX IF NOT EXISTS idx_events_match ON events(match_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
-- UNIQUE mirrors SQLite migration 015: save_events_bulk's ON CONFLICT
-- (match_id, timestamp, event_type, from_track_id) requires it.
CREATE UNIQUE INDEX IF NOT EXISTS idx_events_dedup ON events(match_id, timestamp, event_type, from_track_id);
CREATE INDEX IF NOT EXISTS idx_events_deleted ON events(is_deleted);
CREATE INDEX IF NOT EXISTS idx_events_match_type ON events(match_id, event_type);
CREATE INDEX IF NOT EXISTS idx_events_match_time ON events(match_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_events_team ON events(team);

-- players
CREATE INDEX IF NOT EXISTS idx_players_match ON players(match_id);
CREATE INDEX IF NOT EXISTS idx_players_deleted ON players(is_deleted);
CREATE INDEX IF NOT EXISTS idx_players_match_track ON players(match_id, track_id);
CREATE INDEX IF NOT EXISTS idx_players_team ON players(team);

-- player_profiles
CREATE INDEX IF NOT EXISTS idx_profiles_team ON player_profiles(team);
CREATE INDEX IF NOT EXISTS idx_profiles_global_id ON player_profiles(global_id);
CREATE INDEX IF NOT EXISTS idx_profiles_team_active ON player_profiles(team, is_active);

-- player_match_links
CREATE INDEX IF NOT EXISTS idx_player_match_links_player ON player_match_links(player_id);
CREATE INDEX IF NOT EXISTS idx_player_match_links_match ON player_match_links(match_id);

-- advanced_metrics
CREATE INDEX IF NOT EXISTS idx_advanced_metrics_match ON advanced_metrics(match_id);
CREATE INDEX IF NOT EXISTS idx_advanced_metrics_player ON advanced_metrics(player_id);

-- match_comparisons
CREATE INDEX IF NOT EXISTS idx_match_comparisons_m1 ON match_comparisons(match_id_1);
CREATE INDEX IF NOT EXISTS idx_match_comparisons_m2 ON match_comparisons(match_id_2);

-- analysis_quality
CREATE INDEX IF NOT EXISTS idx_quality_match ON analysis_quality(match_id);

-- benchmark_results
CREATE INDEX IF NOT EXISTS idx_benchmark_created ON benchmark_results(created_at);
CREATE INDEX IF NOT EXISTS idx_benchmark_gpu ON benchmark_results(gpu_name);
CREATE INDEX IF NOT EXISTS idx_benchmark_model ON benchmark_results(model_size);

-- validation_results
CREATE INDEX IF NOT EXISTS idx_validation_match ON validation_results(match_id);
CREATE INDEX IF NOT EXISTS idx_validation_category ON validation_results(category);

-- batch_jobs
CREATE INDEX IF NOT EXISTS idx_batch_status ON batch_jobs(status);
CREATE INDEX IF NOT EXISTS idx_batch_created ON batch_jobs(created_at);

-- feedback
CREATE INDEX IF NOT EXISTS idx_feedback_coach ON feedback(coach_id);
CREATE INDEX IF NOT EXISTS idx_feedback_match ON feedback(match_id);
CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback(created_at);

-- issues
CREATE INDEX IF NOT EXISTS idx_issues_category ON issues(category);
CREATE INDEX IF NOT EXISTS idx_issues_severity ON issues(severity);
CREATE INDEX IF NOT EXISTS idx_issue_reports_created ON issues(created_at);

-- usage_sessions
CREATE INDEX IF NOT EXISTS idx_sessions_created ON usage_sessions(created_at);

-- clips
CREATE INDEX IF NOT EXISTS idx_clips_match ON clips(match_id);
CREATE INDEX IF NOT EXISTS idx_clips_event ON clips(event_type);
CREATE INDEX IF NOT EXISTS idx_clips_match_type ON clips(match_id, event_type);

-- reports
CREATE INDEX IF NOT EXISTS idx_reports_match_lang ON reports(match_id, language);

-- audit_events
CREATE INDEX IF NOT EXISTS idx_audit_events_action ON audit_events(action);
CREATE INDEX IF NOT EXISTS idx_audit_events_type ON audit_events(entity_type);
CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON audit_events(created_at);

-- match_weather
CREATE INDEX IF NOT EXISTS idx_match_weather_match_id ON match_weather(match_id);

-- card_events
CREATE INDEX IF NOT EXISTS idx_card_events_match_id ON card_events(match_id);

-- psychology_events
CREATE INDEX IF NOT EXISTS idx_psychology_events_match_id ON psychology_events(match_id);

-- shortlist
CREATE INDEX IF NOT EXISTS idx_shortlist_status ON player_shortlist(status);
CREATE INDEX IF NOT EXISTS idx_shortlist_priority ON player_shortlist(priority);
CREATE INDEX IF NOT EXISTS idx_shortlist_player ON player_shortlist(player_id);

-- contracts
CREATE INDEX IF NOT EXISTS idx_contracts_end_date ON player_contracts(end_date);
CREATE INDEX IF NOT EXISTS idx_contracts_type ON player_contracts(contract_type);
CREATE INDEX IF NOT EXISTS idx_contracts_profile ON player_contracts(player_profile_id);

-- coding_tags
CREATE INDEX IF NOT EXISTS idx_coding_tags_match ON coding_tags(match_id);
CREATE INDEX IF NOT EXISTS idx_coding_tags_type ON coding_tags(event_type);
CREATE INDEX IF NOT EXISTS idx_coding_tags_player ON coding_tags(player_track_id);
CREATE INDEX IF NOT EXISTS idx_coding_tags_time ON coding_tags(match_id, video_time);
CREATE INDEX IF NOT EXISTS idx_coding_tags_deleted ON coding_tags(is_deleted);
CREATE INDEX IF NOT EXISTS idx_coding_tags_match_type_time ON coding_tags(match_id, event_type, video_time);

-- collab
CREATE INDEX IF NOT EXISTS idx_collab_mentions_username ON collab_mentions(username);
CREATE INDEX IF NOT EXISTS idx_collab_comments_match ON collab_comments(match_id);

-- wearable
CREATE INDEX IF NOT EXISTS idx_wearable_sessions_match ON wearable_sessions(match_id);
CREATE INDEX IF NOT EXISTS idx_wearable_sessions_device ON wearable_sessions(device_type);
CREATE INDEX IF NOT EXISTS idx_wearable_sessions_athlete ON wearable_sessions(athlete_id);

-- injuries
CREATE INDEX IF NOT EXISTS idx_injuries_player ON injuries(player_id);
CREATE INDEX IF NOT EXISTS idx_injuries_status ON injuries(status);

-- rehab
CREATE INDEX IF NOT EXISTS idx_rehab_injury ON rehab_plans(injury_id);

-- concussion
CREATE INDEX IF NOT EXISTS idx_concussion_player ON concussion_assessments(player_id);

-- medical_history
CREATE INDEX IF NOT EXISTS idx_medhist_player ON medical_history(player_id);

-- tracking_frames
CREATE INDEX IF NOT EXISTS idx_tracking_frames_match ON tracking_frames(match_id);
CREATE INDEX IF NOT EXISTS idx_tracking_frames_range ON tracking_frames(match_id, frame_number);

-- user_corrections
CREATE INDEX IF NOT EXISTS idx_user_corrections_event ON user_corrections(event_id);

-- ── Row-Level Security ──────────────────────────────────────────────────
-- Enable RLS on multi-team tables; policies assume a `tenant_team` session variable
-- or JWT claim. By default all rows visible; production should scope by team.

ALTER TABLE IF EXISTS matches ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS events ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS players ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS player_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS coding_tags ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS player_shortlist ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS player_contracts ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS injuries ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS concussion_assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS medical_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS wearable_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS tracking_frames ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS collab_comments ENABLE ROW LEVEL SECURITY;

-- Default: authenticated users see all rows.
-- Replace with team-scoped policies in production:
--   CREATE POLICY team_isolation ON matches
--     FOR ALL USING (
--       home_team = current_setting('app.tenant_team') OR
--       away_team = current_setting('app.tenant_team')
--     );

-- 'CREATE POLICY IF NOT EXISTS' is not valid Postgres syntax; the
-- idempotent pattern is DROP IF EXISTS then CREATE, run in one DO block.
DO $$
DECLARE t text;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'matches', 'events', 'players', 'player_profiles', 'reports',
        'coding_tags', 'player_shortlist', 'player_contracts', 'injuries',
        'concussion_assessments', 'medical_history', 'wearable_sessions',
        'tracking_frames', 'collab_comments'
    ]
    LOOP
        EXECUTE format('DROP POLICY IF EXISTS all_access ON %I', t);
        EXECUTE format('CREATE POLICY all_access ON %I FOR ALL USING (true)', t);
    END LOOP;
END
$$;

-- ── 43. tracking_imports (elite interop: vendor tracking provenance) ────
CREATE TABLE IF NOT EXISTS tracking_imports (
    id SERIAL PRIMARY KEY,
    match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    vendor TEXT NOT NULL,
    source_path TEXT NOT NULL DEFAULT '',
    checksum TEXT NOT NULL DEFAULT '',
    fps DOUBLE PRECISION,
    frame_count INTEGER NOT NULL DEFAULT 0,
    pitch_length_m DOUBLE PRECISION,
    pitch_width_m DOUBLE PRECISION,
    coordinate_system TEXT NOT NULL DEFAULT 'kawkab_meters',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    imported_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_deleted INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_tracking_imports_match ON tracking_imports(match_id);
CREATE INDEX IF NOT EXISTS idx_tracking_imports_checksum ON tracking_imports(match_id, vendor, checksum);

-- ── 44. event_frame_links (event<->frame alignment bridge) ─────────────
CREATE TABLE IF NOT EXISTS event_frame_links (
    id SERIAL PRIMARY KEY,
    match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    frame_number INTEGER NOT NULL,
    frame_offset INTEGER NOT NULL DEFAULT 0,
    UNIQUE(match_id, event_id, frame_number)
);

CREATE INDEX IF NOT EXISTS idx_event_frame_links_match ON event_frame_links(match_id);
CREATE INDEX IF NOT EXISTS idx_event_frame_links_event ON event_frame_links(event_id);

-- ── 45. users / sessions / local audit (mirrors SQLite migration 027) ──
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT,
    display_name TEXT DEFAULT '',
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'analyst',
    team TEXT DEFAULT '',
    is_active INTEGER DEFAULT 1,
    is_locked INTEGER DEFAULT 0,
    failed_attempts INTEGER DEFAULT 0,
    locked_until TEXT,
    must_reset_password INTEGER DEFAULT 0,
    last_login TEXT,
    created_at TEXT DEFAULT (TO_CHAR(NOW(), 'YYYY-MM-DD HH24:MI:SS')),
    updated_at TEXT DEFAULT (TO_CHAR(NOW(), 'YYYY-MM-DD HH24:MI:SS'))
);

CREATE TABLE IF NOT EXISTS user_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    ip_address TEXT,
    user_agent TEXT,
    created_at TEXT DEFAULT (TO_CHAR(NOW(), 'YYYY-MM-DD HH24:MI:SS'))
);

CREATE INDEX IF NOT EXISTS idx_pg_sessions_user ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_pg_sessions_token ON user_sessions(token_hash);

CREATE TABLE IF NOT EXISTS audit_events_local (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    username TEXT,
    action TEXT NOT NULL,
    resource_type TEXT,
    resource_id TEXT,
    details TEXT DEFAULT '{}',
    ip_address TEXT,
    created_at TEXT DEFAULT (TO_CHAR(NOW(), 'YYYY-MM-DD HH24:MI:SS'))
);

CREATE INDEX IF NOT EXISTS idx_pg_audit_user ON audit_events_local(user_id);
CREATE INDEX IF NOT EXISTS idx_pg_audit_action ON audit_events_local(action);
CREATE INDEX IF NOT EXISTS idx_pg_audit_time ON audit_events_local(created_at);

-- ── 46. GPS / ACWR (mirrors SQLite migration 026) ──────────────────────
CREATE TABLE IF NOT EXISTS gps_sessions (
    id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES matches(id) ON DELETE CASCADE,
    player_id INTEGER REFERENCES players(id) ON DELETE CASCADE,
    session_type TEXT NOT NULL DEFAULT 'match',
    vendor TEXT NOT NULL DEFAULT 'catapult',
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    duration_seconds DOUBLE PRECISION,
    total_distance_m DOUBLE PRECISION,
    max_speed_kmh DOUBLE PRECISION,
    avg_speed_kmh DOUBLE PRECISION,
    player_load DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS gps_samples (
    id SERIAL PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES gps_sessions(id) ON DELETE CASCADE,
    timestamp DOUBLE PRECISION NOT NULL,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    speed_ms DOUBLE PRECISION,
    acceleration DOUBLE PRECISION,
    accel_x DOUBLE PRECISION,
    accel_y DOUBLE PRECISION,
    accel_z DOUBLE PRECISION,
    heart_rate INTEGER,
    distance DOUBLE PRECISION,
    player_load DOUBLE PRECISION,
    metabolic_power DOUBLE PRECISION,
    speed_zone INTEGER,
    x_m DOUBLE PRECISION,
    y_m DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_pg_gps_samples_session ON gps_samples(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_pg_gps_sessions_match ON gps_sessions(match_id);
CREATE INDEX IF NOT EXISTS idx_pg_gps_sessions_player ON gps_sessions(player_id);

CREATE TABLE IF NOT EXISTS acwr_daily (
    id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    acute_load_7d DOUBLE PRECISION,
    chronic_load_28d DOUBLE PRECISION,
    acwr DOUBLE PRECISION,
    load_category TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(player_id, date)
);

CREATE INDEX IF NOT EXISTS idx_pg_acwr_player_date ON acwr_daily(player_id, date);

-- Migration 031: match external-ID registry (season-scale vendor imports).
-- UNIQUE(source, external_id) makes vendor re-import dedup a lookup.
CREATE TABLE IF NOT EXISTS matches_external_ids (
    id SERIAL PRIMARY KEY,
    match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    external_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source, external_id)
);

CREATE INDEX IF NOT EXISTS idx_matches_external_ids_match ON matches_external_ids(match_id);
CREATE INDEX IF NOT EXISTS idx_matches_competition ON matches(competition);
CREATE INDEX IF NOT EXISTS idx_matches_season ON matches(season_id);

-- ── 33. program_documents (Phase D operating program) ───────────────────
-- Versioned club documents: weekly_rhythm, raci_matrix, kpi_tree.
-- Mirrors migration 033; see that file for the honesty-contract notes.

CREATE TABLE IF NOT EXISTS program_documents (
    id SERIAL PRIMARY KEY,
    doc_type TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    version INTEGER NOT NULL DEFAULT 1,
    is_current BOOLEAN NOT NULL DEFAULT TRUE,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_by TEXT DEFAULT '',
    supersedes_id INTEGER REFERENCES program_documents(id),
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_program_docs_type
    ON program_documents(doc_type, name, is_current);

-- ── 34. evidence_records (trust layer: groundedness gate) ──────────────
CREATE TABLE IF NOT EXISTS evidence_records (
    id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES matches(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    label TEXT DEFAULT '',
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_evidence_match ON evidence_records(match_id);

COMMIT;
