"""Tests for event review bridge methods: get_unreviewed_events, get_detection_summary, submit_event_correction."""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.services.storage_service import StorageService
from kawkab.ui.bridge_handlers.bridge_analysis import AnalysisHandler

EVENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    short_name TEXT
);
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL, video_path TEXT NOT NULL,
    home_team TEXT, away_team TEXT,
    home_team_id INTEGER, away_team_id INTEGER,
    match_date TEXT, duration_seconds REAL,
    fps REAL, total_frames INTEGER,
    is_deleted INTEGER DEFAULT 0,
    analyzed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    owner_id INTEGER, team_id INTEGER, is_shared INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    timestamp REAL NOT NULL,
    from_track_id INTEGER,
    to_track_id INTEGER,
    team TEXT,
    completed BOOLEAN,
    confidence REAL,
    metadata TEXT,
    user_corrected BOOLEAN DEFAULT 0,
    is_deleted INTEGER DEFAULT 0,
    deleted_at TEXT,
    deleted_by TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS user_corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL,
    correction_type TEXT NOT NULL,
    original_value TEXT DEFAULT '{}',
    corrected_value TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


@pytest.fixture
def svc():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        s = StorageService()
        s._db_path = db_path
        s._conn = sqlite3.connect(str(db_path))
        s._conn.row_factory = sqlite3.Row
        s._conn.executescript(EVENTS_TABLE_SQL)
        s._conn.commit()
        yield s
        if s._conn:
            s._conn.close()


@pytest.fixture
def handler(svc):
    mock_bridge = MagicMock()
    services = {"storage_service": svc}
    h = AnalysisHandler(mock_bridge, services)
    return h


async def _mid(svc):
    return await svc.save_match("Test Match", "/v/test.mp4", "Home", "Away")


@pytest.mark.asyncio
async def test_get_unreviewed_events_empty(handler, svc):
    mid = await _mid(svc)
    result = json.loads(await handler.get_unreviewed_events(mid))
    assert result["success"] is True
    assert result["events"] == []
    assert result["total"] == 0


@pytest.mark.asyncio
async def test_get_unreviewed_events_filters_corrected(handler, svc):
    mid = await _mid(svc)
    svc._conn.execute(
        "INSERT INTO events (match_id, timestamp, event_type, confidence, user_corrected) VALUES (?,?,?,?,?)",
        (mid, 10.0, "pass", 0.3, 1),
    )
    svc._conn.commit()
    result = json.loads(await handler.get_unreviewed_events(mid))
    assert result["total"] == 0


@pytest.mark.asyncio
async def test_get_unreviewed_events_shows_low_confidence(handler, svc):
    mid = await _mid(svc)
    svc._conn.execute(
        "INSERT INTO events (match_id, timestamp, event_type, confidence) VALUES (?,?,?,?)",
        (mid, 10.0, "pass", 0.25),
    )
    svc._conn.execute(
        "INSERT INTO events (match_id, timestamp, event_type, confidence) VALUES (?,?,?,?)",
        (mid, 20.0, "shot", 0.85),
    )
    svc._conn.commit()
    result = json.loads(await handler.get_unreviewed_events(mid))
    assert result["total"] == 1
    assert result["events"][0]["event_type"] == "pass"


@pytest.mark.asyncio
async def test_get_unreviewed_events_sorted_by_confidence(handler, svc):
    mid = await _mid(svc)
    svc._conn.execute(
        "INSERT INTO events (match_id, timestamp, event_type, confidence) VALUES (?,?,?,?)",
        (mid, 10.0, "pass", 0.6),
    )
    svc._conn.execute(
        "INSERT INTO events (match_id, timestamp, event_type, confidence) VALUES (?,?,?,?)",
        (mid, 20.0, "shot", 0.2),
    )
    svc._conn.commit()
    result = json.loads(await handler.get_unreviewed_events(mid))
    assert result["total"] == 2
    assert result["events"][0]["event_type"] == "shot"
    assert result["events"][1]["event_type"] == "pass"


@pytest.mark.asyncio
async def test_get_unreviewed_events_min_max_confidence(handler, svc):
    mid = await _mid(svc)
    svc._conn.execute(
        "INSERT INTO events (match_id, timestamp, event_type, confidence) VALUES (?,?,?,?)",
        (mid, 10.0, "pass", 0.1),
    )
    svc._conn.execute(
        "INSERT INTO events (match_id, timestamp, event_type, confidence) VALUES (?,?,?,?)",
        (mid, 20.0, "shot", 0.4),
    )
    svc._conn.execute(
        "INSERT INTO events (match_id, timestamp, event_type, confidence) VALUES (?,?,?,?)",
        (mid, 30.0, "tackle", 0.8),
    )
    svc._conn.commit()
    result = json.loads(
        await handler.get_unreviewed_events(mid, min_confidence=0.2, max_confidence=0.5)
    )
    assert result["total"] == 1
    assert result["events"][0]["event_type"] == "shot"


@pytest.mark.asyncio
async def test_get_unreviewed_events_parses_metadata(handler, svc):
    mid = await _mid(svc)
    meta = json.dumps({"start_x": 0.5, "end_y": 0.8})
    svc._conn.execute(
        "INSERT INTO events (match_id, timestamp, event_type, confidence, metadata) VALUES (?,?,?,?,?)",
        (mid, 10.0, "pass", 0.3, meta),
    )
    svc._conn.commit()
    result = json.loads(await handler.get_unreviewed_events(mid))
    assert result["total"] == 1
    meta_parsed = result["events"][0].get("_meta", {})
    assert meta_parsed.get("start_x") == 0.5


@pytest.mark.asyncio
async def test_get_detection_summary_empty(handler, svc):
    mid = await _mid(svc)
    result = json.loads(await handler.get_detection_summary(mid))
    assert result["success"] is True
    assert result["total"] == 0
    assert result["by_type"] == {}


@pytest.mark.asyncio
async def test_get_detection_summary_counts_by_type(handler, svc):
    mid = await _mid(svc)
    svc._conn.execute(
        "INSERT INTO events (match_id, timestamp, event_type, confidence) VALUES (?,?,?,?)",
        (mid, 10.0, "pass", 0.5),
    )
    svc._conn.execute(
        "INSERT INTO events (match_id, timestamp, event_type, confidence) VALUES (?,?,?,?)",
        (mid, 20.0, "pass", 0.6),
    )
    svc._conn.execute(
        "INSERT INTO events (match_id, timestamp, event_type, confidence) VALUES (?,?,?,?)",
        (mid, 30.0, "shot", 0.8),
    )
    svc._conn.commit()
    result = json.loads(await handler.get_detection_summary(mid))
    assert result["total"] == 3
    assert result["by_type"]["pass"]["count"] == 2
    assert result["by_type"]["shot"]["count"] == 1


@pytest.mark.asyncio
async def test_get_detection_summary_avg_confidence(handler, svc):
    mid = await _mid(svc)
    svc._conn.execute(
        "INSERT INTO events (match_id, timestamp, event_type, confidence) VALUES (?,?,?,?)",
        (mid, 10.0, "pass", 0.4),
    )
    svc._conn.execute(
        "INSERT INTO events (match_id, timestamp, event_type, confidence) VALUES (?,?,?,?)",
        (mid, 20.0, "pass", 0.6),
    )
    svc._conn.commit()
    result = json.loads(await handler.get_detection_summary(mid))
    assert result["by_type"]["pass"]["avg_confidence"] == 0.5


@pytest.mark.asyncio
async def test_get_detection_summary_corrected_count(handler, svc):
    mid = await _mid(svc)
    svc._conn.execute(
        "INSERT INTO events (match_id, timestamp, event_type, confidence, user_corrected) VALUES (?,?,?,?,?)",
        (mid, 10.0, "pass", 0.5, 1),
    )
    svc._conn.execute(
        "INSERT INTO events (match_id, timestamp, event_type, confidence) VALUES (?,?,?,?)",
        (mid, 20.0, "pass", 0.5),
    )
    svc._conn.commit()
    result = json.loads(await handler.get_detection_summary(mid))
    assert result["corrected"] == 1
    assert result["unreviewed"] == 1
    assert result["by_type"]["pass"]["corrected"] == 1


@pytest.mark.asyncio
async def test_submit_correction_confirm(handler, svc):
    mid = await _mid(svc)
    svc._conn.execute(
        "INSERT INTO events (match_id, timestamp, event_type, confidence) VALUES (?,?,?,?)",
        (mid, 10.0, "pass", 0.3),
    )
    svc._conn.commit()
    event_id = svc._conn.execute("SELECT id FROM events LIMIT 1").fetchone()[0]

    result = json.loads(await handler.submit_event_correction(mid, event_id, "confirm", ""))
    assert result["success"] is True
    assert result["action"] == "confirmed"

    row = svc._conn.execute("SELECT user_corrected FROM events WHERE id=?", (event_id,)).fetchone()
    assert row[0] == 1


@pytest.mark.asyncio
async def test_submit_correction_reject(handler, svc):
    mid = await _mid(svc)
    svc._conn.execute(
        "INSERT INTO events (match_id, timestamp, event_type, confidence) VALUES (?,?,?,?)",
        (mid, 10.0, "pass", 0.3),
    )
    svc._conn.commit()
    event_id = svc._conn.execute("SELECT id FROM events LIMIT 1").fetchone()[0]

    result = json.loads(await handler.submit_event_correction(mid, event_id, "reject", ""))
    assert result["success"] is True
    assert result["action"] == "deleted"

    row = svc._conn.execute("SELECT is_deleted FROM events WHERE id=?", (event_id,)).fetchone()
    assert row[0] == 1


@pytest.mark.asyncio
async def test_submit_correction_edit_type(handler, svc):
    mid = await _mid(svc)
    svc._conn.execute(
        "INSERT INTO events (match_id, timestamp, event_type, team, confidence) VALUES (?,?,?,?,?)",
        (mid, 10.0, "pass", "home", 0.3),
    )
    svc._conn.commit()
    event_id = svc._conn.execute("SELECT id FROM events LIMIT 1").fetchone()[0]

    corrections = json.dumps({"event_type": "shot", "team": "away"})
    result = json.loads(await handler.submit_event_correction(mid, event_id, "edit", corrections))
    assert result["success"] is True
    assert result["action"] == "edited"

    row = svc._conn.execute(
        "SELECT event_type, team, user_corrected FROM events WHERE id=?", (event_id,)
    ).fetchone()
    assert row[0] == "shot"
    assert row[1] == "away"
    assert row[2] == 1


@pytest.mark.asyncio
async def test_submit_correction_edit_saves_correction_record(handler, svc):
    mid = await _mid(svc)
    svc._conn.execute(
        "INSERT INTO events (match_id, timestamp, event_type, team, confidence) VALUES (?,?,?,?,?)",
        (mid, 10.0, "pass", "home", 0.3),
    )
    svc._conn.commit()
    event_id = svc._conn.execute("SELECT id FROM events LIMIT 1").fetchone()[0]

    corrections = json.dumps({"event_type": "cross"})
    result = json.loads(await handler.submit_event_correction(mid, event_id, "edit", corrections))
    assert result["success"] is True

    corr_row = svc._conn.execute(
        "SELECT * FROM user_corrections WHERE event_id=?", (event_id,)
    ).fetchone()
    assert corr_row is not None
    assert corr_row["correction_type"] == "edit"
    orig = json.loads(corr_row["original_value"])
    assert orig["event_type"] == "pass"


@pytest.mark.asyncio
async def test_submit_correction_invalid_action(handler, svc):
    mid = await _mid(svc)
    result = json.loads(await handler.submit_event_correction(mid, 1, "invalid_action", ""))
    assert "error" in result


@pytest.mark.asyncio
async def test_submit_correction_nonexistent_event(handler, svc):
    mid = await _mid(svc)
    result = json.loads(await handler.submit_event_correction(mid, 999, "confirm", ""))
    assert result["success"] is False
