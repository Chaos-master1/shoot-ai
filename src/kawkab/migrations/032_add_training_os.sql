-- Migration 032: Training Operating System
-- The closed loop: game model → plans → sessions → drills → execution → re-measure.
-- Honesty-contract era migration: every table is written only through
-- StorageService methods that raise on failure (never swallow).

-- ── The club's playing philosophy (Tactical Periodization anchor) ──────
CREATE TABLE IF NOT EXISTS game_model (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER REFERENCES teams(id),
    version INTEGER NOT NULL DEFAULT 1,
    is_active INTEGER NOT NULL DEFAULT 1,
    in_possession TEXT DEFAULT '',
    out_of_possession TEXT DEFAULT '',
    transition_attack TEXT DEFAULT '',
    transition_defence TEXT DEFAULT '',
    set_pieces_off TEXT DEFAULT '',
    set_pieces_def TEXT DEFAULT '',
    player_roles TEXT DEFAULT '[]',        -- JSON: [{role, responsibilities, non_negotiables}]
    non_negotiables TEXT DEFAULT '[]',     -- JSON: [str]
    language TEXT DEFAULT 'en',
    created_by TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now')),
    notes TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_game_model_team ON game_model(team_id, is_active);

-- ── Training plans (persisted, versioned, assignable) ──────────────────
CREATE TABLE IF NOT EXISTS training_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER REFERENCES matches(id),
    title TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'draft',       -- draft | active | completed | cancelled
    duration_weeks INTEGER NOT NULL DEFAULT 4,
    priority_diagnoses TEXT DEFAULT '[]',       -- JSON: [{rule_id, rule_name, confidence}]
    payload TEXT NOT NULL DEFAULT '{}',         -- JSON: full TrainingPlan export
    source TEXT NOT NULL DEFAULT 'reasoning_engine',  -- provenance: reasoning_engine | manual
    created_by TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_training_plans_match ON training_plans(match_id);
CREATE INDEX IF NOT EXISTS idx_training_plans_status ON training_plans(status);

CREATE TABLE IF NOT EXISTS plan_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id INTEGER NOT NULL REFERENCES training_plans(id) ON DELETE CASCADE,
    version INTEGER NOT NULL DEFAULT 1,
    change_note TEXT DEFAULT '',
    payload TEXT NOT NULL DEFAULT '{}',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_plan_versions_plan ON plan_versions(plan_id);

-- ── Actual training sessions (the morphocycle in practice) ─────────────
CREATE TABLE IF NOT EXISTS training_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id INTEGER REFERENCES training_plans(id) ON DELETE SET NULL,
    match_id INTEGER REFERENCES matches(id),
    session_date TEXT NOT NULL,
    md_offset TEXT DEFAULT '',                  -- MD+1, MD-3, MD-2, MD-1, MD (game day) or ''
    session_type TEXT NOT NULL DEFAULT 'tactical',  -- recovery | tactical | physical | activation | set_piece | test | other
    theme TEXT DEFAULT '',
    duration_min INTEGER DEFAULT 90,
    intensity TEXT DEFAULT 'medium',
    status TEXT NOT NULL DEFAULT 'planned',     -- planned | completed | cancelled
    notes TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_sessions_date ON training_sessions(session_date);
CREATE INDEX IF NOT EXISTS idx_sessions_plan ON training_sessions(plan_id);

CREATE TABLE IF NOT EXISTS session_drills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES training_sessions(id) ON DELETE CASCADE,
    drill_id TEXT NOT NULL,                     -- KB drill id (e.g. rondo_4v2)
    order_index INTEGER NOT NULL DEFAULT 0,
    duration_min INTEGER DEFAULT 15,
    executed INTEGER NOT NULL DEFAULT 0,        -- 0 planned / 1 executed / 2 modified / 3 skipped
    executed_as TEXT DEFAULT '',                -- what actually happened (if modified)
    coach_note TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_session_drills_session ON session_drills(session_id);
CREATE INDEX IF NOT EXISTS idx_session_drills_drill ON session_drills(drill_id);

-- ── Attendance + internal load (the daily ritual capture) ──────────────
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES training_sessions(id) ON DELETE CASCADE,
    player_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'present',     -- present | absent | injured | ill | excused | late
    minutes_participated INTEGER DEFAULT 0,
    note TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_attendance_session ON attendance(session_id);
CREATE INDEX IF NOT EXISTS idx_attendance_player ON attendance(player_id);

CREATE TABLE IF NOT EXISTS session_rpe (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES training_sessions(id) ON DELETE CASCADE,
    player_id INTEGER NOT NULL,
    rpe REAL NOT NULL DEFAULT 0.0,              -- Foster 0-10 Borg CR10
    minutes_played INTEGER DEFAULT 0,
    load REAL DEFAULT 0.0,                      -- sRPE = rpe * minutes (stored for audit)
    recorded_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_session_rpe_session ON session_rpe(session_id);
CREATE INDEX IF NOT EXISTS idx_session_rpe_player ON session_rpe(player_id);

CREATE TABLE IF NOT EXISTS wellness (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    record_date TEXT NOT NULL,
    sleep_quality INTEGER DEFAULT 3,            -- 1-5 Hooper index
    fatigue INTEGER DEFAULT 3,                  -- 1-5 (5 = very fresh)
    soreness INTEGER DEFAULT 3,                 -- 1-5 (5 = no soreness)
    stress INTEGER DEFAULT 3,                   -- 1-5 (5 = no stress)
    mood INTEGER DEFAULT 3,                     -- 1-5
    wellness_score REAL DEFAULT 0.0,            -- normalized mean (computed on write)
    source TEXT DEFAULT 'manual',               -- manual | import
    recorded_at TEXT DEFAULT (datetime('now')),
    UNIQUE(player_id, record_date)
);

CREATE INDEX IF NOT EXISTS idx_wellness_player_date ON wellness(player_id, record_date);

-- ── Drill feedback loop (did the drill work?) ──────────────────────────
CREATE TABLE IF NOT EXISTS drill_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    drill_id TEXT NOT NULL,
    session_id INTEGER REFERENCES training_sessions(id) ON DELETE SET NULL,
    player_scope TEXT DEFAULT 'squad',          -- squad | group | player:<id>
    effectiveness INTEGER DEFAULT 0,            -- 1-5 coach rating
    observations TEXT DEFAULT '',
    targeted_metric TEXT DEFAULT '',            -- the diagnosis metric this drill addressed
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_drill_feedback_drill ON drill_feedback(drill_id);

-- ── Fitness testing battery ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS testing_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    test_date TEXT NOT NULL,
    test_type TEXT NOT NULL,                    -- yoyo_ir1 | yoyo_ir2 | cmj | sprint_10m | sprint_30m | fifa11plus_compliance | other
    value REAL NOT NULL DEFAULT 0.0,
    unit TEXT DEFAULT '',
    percentile TEXT DEFAULT '',                 -- JSON: {peer_group: pct} computed later
    notes TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_testing_player ON testing_results(player_id, test_type);

-- ── Individual Development Plans (youth + senior) ──────────────────────
CREATE TABLE IF NOT EXISTS idp_goals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    season TEXT DEFAULT '',
    category TEXT NOT NULL DEFAULT 'technical', -- technical | tactical | physical | mental | social
    goal_text TEXT NOT NULL DEFAULT '',
    target_metric TEXT DEFAULT '',
    baseline_value REAL,
    current_value REAL,
    target_date TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',      -- active | achieved | revised | dropped
    review_notes TEXT DEFAULT '[]',             -- JSON: [{date, note, rating}]
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_idp_player ON idp_goals(player_id, status);

-- ── Nutrition + psychology (guided protocols) ──────────────────────────
CREATE TABLE IF NOT EXISTS nutrition_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    log_date TEXT NOT NULL,
    meal_type TEXT DEFAULT 'match_day',         -- match_day | md_minus_1 | recovery | daily
    hydration_score INTEGER DEFAULT 3,          -- 1-5
    fueling_score INTEGER DEFAULT 3,            -- 1-5 adherence to the fueling plan
    notes TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(player_id, log_date, meal_type)
);

CREATE TABLE IF NOT EXISTS psych_checkins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    checkin_date TEXT NOT NULL,
    confidence INTEGER DEFAULT 3,               -- 1-5
    focus INTEGER DEFAULT 3,                    -- 1-5
    motivation INTEGER DEFAULT 3,               -- 1-5
    anxiety INTEGER DEFAULT 3,                  -- 1-5 (5 = low anxiety)
    notes TEXT DEFAULT '',
    flag_for_followup INTEGER NOT NULL DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(player_id, checkin_date)
);

-- ── Staff rituals (the operating program) ──────────────────────────────
CREATE TABLE IF NOT EXISTS staff_rituals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ritual_type TEXT NOT NULL,                  -- wellness_huddle | post_match_review | staff_sync | md1_brief | analyst_prematch | analyst_postmatch | briefing_prep
    scheduled_for TEXT NOT NULL,                -- ISO date
    completed_at TEXT DEFAULT '',
    completed_by TEXT DEFAULT '',
    checklist_state TEXT DEFAULT '[]',          -- JSON: [{item, done}]
    notes TEXT DEFAULT '',
    UNIQUE(ritual_type, scheduled_for)
);

-- ── Medical clearance (gates selection; hard block) ────────────────────
CREATE TABLE IF NOT EXISTS medical_clearances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'fit',         -- fit | limited | unavailable
    reason TEXT DEFAULT '',
    source TEXT DEFAULT 'manual',               -- manual | concussion_protocol | injury_tracker
    cleared_by TEXT DEFAULT '',                 -- staff member (role-gated write)
    effective_until TEXT DEFAULT '',            -- '' = until changed
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_clearance_player ON medical_clearances(player_id);

-- ── Minutes management (game time as a development resource) ───────────
CREATE TABLE IF NOT EXISTS minutes_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    match_id INTEGER NOT NULL REFERENCES matches(id),
    minutes_played INTEGER NOT NULL DEFAULT 0,
    started INTEGER NOT NULL DEFAULT 0,
    age_phase TEXT DEFAULT '',                  -- foundation | youth_development | professional_development | senior
    bio_band TEXT DEFAULT '',                   -- bio-banding band estimate when known
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_minutes_player ON minutes_log(player_id);
CREATE INDEX IF NOT EXISTS idx_minutes_match ON minutes_log(match_id);

-- ── SCAT6 upgrade columns (concussion_assessments from migration 021) ──
-- NOTE: assessment_type already exists on concussion_assessments from
-- migration 021 (default 'scat5'); only red_flags is new here.
ALTER TABLE concussion_assessments ADD COLUMN red_flags TEXT DEFAULT '[]';

INSERT OR REPLACE INTO schema_version (version) VALUES (32);
