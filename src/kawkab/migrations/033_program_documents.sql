-- Migration 033: Program documents + evidence registry (Phase D).
--
-- The operating program (weekly rhythm, roles/RACI, KPI tree) is club
-- IP: it must be versioned like the game model, not hardcoded in the
-- service layer. One generic, versioned document table keeps the three
-- document kinds (weekly_rhythm, raci_matrix, kpi_tree) uniformly
-- auditable: exactly one current version per (type, name), superseded
-- versions retained for history.
--
-- The evidence_records table is the trust layer's registry: every
-- LLM-surfaced claim must be checkable against a stored evidence
-- record, so the records must outlive the process that created them.
--
-- Honesty-contract era migration: written only through StorageService
-- methods that raise on failure.

CREATE TABLE IF NOT EXISTS program_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_type TEXT NOT NULL,               -- weekly_rhythm | raci_matrix | kpi_tree
    name TEXT NOT NULL DEFAULT '',        -- e.g. "in-season morphocycle week"
    version INTEGER NOT NULL DEFAULT 1,
    is_current INTEGER NOT NULL DEFAULT 1,
    payload TEXT NOT NULL DEFAULT '{}',   -- JSON document body
    created_by TEXT DEFAULT '',
    supersedes_id INTEGER REFERENCES program_documents(id),
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_program_docs_type
    ON program_documents(doc_type, name, is_current);

-- ── Evidence registry (trust layer: groundedness gate) ───────────────
CREATE TABLE IF NOT EXISTS evidence_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER REFERENCES matches(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,                   -- event_summary | metric_snapshot | review_claim
    label TEXT DEFAULT '',                -- human-readable handle ("E1", "goals 2026-09-19")
    payload TEXT NOT NULL DEFAULT '{}',   -- JSON evidence body
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_evidence_match ON evidence_records(match_id);

INSERT OR REPLACE INTO schema_version (version) VALUES (33);
