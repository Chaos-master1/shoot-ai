-- Migration 030: Vendor tracking-import provenance (elite interop path)
--
-- Per-match metadata about *external* tracking feeds (SkillCorner, EPTS,
-- Metrica, ...) imported with zero video capture. Frame positions
-- themselves live in the existing `tracking_frames` table (migration 022,
-- already shared by the video pipeline via save_tracking_frames_bulk /
-- get_tracking_frames in both storage backends) — this migration only
-- adds the provenance row a vendor import writes once per match, plus
-- the event<->frame alignment bridge used to run tracking-based models
-- (VAEP, pitch control, pressing) on vendor positional data.

CREATE TABLE IF NOT EXISTS tracking_imports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    vendor TEXT NOT NULL,                        -- skillcorner | epts | metrica | opta | wyscout | ...
    source_path TEXT NOT NULL DEFAULT '',        -- file (or dir) the import came from
    checksum TEXT NOT NULL DEFAULT '',           -- sha256 of the source file (dedup / idempotence)
    fps REAL,
    frame_count INTEGER NOT NULL DEFAULT 0,
    pitch_length_m REAL,
    pitch_width_m REAL,
    coordinate_system TEXT NOT NULL DEFAULT 'kawkab_meters',  -- normalized on import
    metadata_json TEXT NOT NULL DEFAULT '{}',    -- vendor-specific extras (teams, periods, ...)
    imported_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_deleted INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_tracking_imports_match ON tracking_imports(match_id);
CREATE INDEX IF NOT EXISTS idx_tracking_imports_checksum ON tracking_imports(match_id, vendor, checksum);

-- Event<->frame alignment: maps an imported/simulated event timestamp to
-- the tracking frame(s) that bracket it, so VAEP / pitch control / EPV can
-- pull the 22-player positional snapshot around each event without
-- re-deriving the index every run.
CREATE TABLE IF NOT EXISTS event_frame_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    frame_number INTEGER NOT NULL,
    frame_offset INTEGER NOT NULL DEFAULT 0,     -- +N frames after the event timestamp
    UNIQUE(match_id, event_id, frame_number)
);

CREATE INDEX IF NOT EXISTS idx_event_frame_links_match ON event_frame_links(match_id);
CREATE INDEX IF NOT EXISTS idx_event_frame_links_event ON event_frame_links(event_id);
