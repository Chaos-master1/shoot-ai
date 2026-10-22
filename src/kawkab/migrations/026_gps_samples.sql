-- Migration 026: GPS/Physical data pipeline
-- Tables for importing and storing GPS tracking data from Catapult, STATSports, Kinexon

CREATE TABLE IF NOT EXISTS gps_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER REFERENCES matches(id) ON DELETE CASCADE,
    player_id INTEGER REFERENCES players(id) ON DELETE CASCADE,
    session_type TEXT NOT NULL DEFAULT 'match',  -- match, training, recovery
    vendor TEXT NOT NULL DEFAULT 'catapult',     -- catapult, statSports, kinexon
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    duration_seconds REAL,
    total_distance_m REAL,
    max_speed_kmh REAL,
    avg_speed_kmh REAL,
    player_load REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS gps_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES gps_sessions(id) ON DELETE CASCADE,
    timestamp REAL NOT NULL,
    lat REAL,
    lon REAL,
    speed_ms REAL,
    acceleration REAL,
    accel_x REAL,
    accel_y REAL,
    accel_z REAL,
    heart_rate INTEGER,
    distance REAL,
    player_load REAL,
    metabolic_power REAL,
    speed_zone INTEGER,  -- 1=walking, 2=jogging, 3=running, 4=high_intensity, 5=sprinting
    x_m REAL,
    y_m REAL
);

CREATE INDEX IF NOT EXISTS idx_gps_samples_session ON gps_samples(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_gps_sessions_match ON gps_sessions(match_id);
CREATE INDEX IF NOT EXISTS idx_gps_sessions_player ON gps_sessions(player_id);

-- Acute:Chronic Workload Ratio tracking
CREATE TABLE IF NOT EXISTS acwr_daily (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER REFERENCES players(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    acute_load_7d REAL,
    chronic_load_28d REAL,
    acwr REAL,
    load_category TEXT,  -- low, normal, high, very_high
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(player_id, date)
);

CREATE INDEX IF NOT EXISTS idx_acwr_player_date ON acwr_daily(player_id, date);
