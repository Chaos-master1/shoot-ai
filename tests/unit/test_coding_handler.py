"""Tests for CodingHandler bridge methods."""

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

from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()

# Storage service uses the standard loader
_storage_mod = load_service_module("coding_storage_test", "storage_service.py")
StorageService = _storage_mod.StorageService

# Coding handler from ui/bridge_handlers/
from kawkab.ui.bridge_handlers.bridge_coding import CodingHandler

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    short_name TEXT,
    home_color TEXT DEFAULT '#1e7e34',
    away_color TEXT DEFAULT '#ffffff',
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    video_path TEXT NOT NULL,
    home_team TEXT, away_team TEXT,
    match_date TEXT, duration_seconds REAL,
    fps REAL, total_frames INTEGER,
    analyzed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    home_team_id INTEGER REFERENCES teams(id),
    away_team_id INTEGER REFERENCES teams(id),
    score_home INTEGER, score_away INTEGER,
    season_id INTEGER, match_type TEXT DEFAULT 'unknown',
    owner_id INTEGER, team_id INTEGER, is_shared INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS coding_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    sub_type TEXT DEFAULT '',
    video_time REAL NOT NULL,
    player_track_id INTEGER DEFAULT 0,
    player_name TEXT DEFAULT '',
    team TEXT DEFAULT '',
    period INTEGER DEFAULT 1,
    notes TEXT DEFAULT '',
    lead_ms INTEGER DEFAULT 2000,
    lag_ms INTEGER DEFAULT 3000,
    is_deleted INTEGER DEFAULT 0,
    deleted_at TEXT,
    deleted_by TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY AUTOINCREMENT, match_id INTEGER NOT NULL,
    track_id INTEGER NOT NULL, jersey_number INTEGER, name TEXT,
    team TEXT, position TEXT, distance_covered_m REAL DEFAULT 0,
    max_speed_kmh REAL DEFAULT 0, avg_speed_kmh REAL DEFAULT 0,
    passes_attempted INTEGER DEFAULT 0, passes_completed INTEGER DEFAULT 0,
    shots INTEGER DEFAULT 0, tackles INTEGER DEFAULT 0,
    confidence REAL DEFAULT 0.0,
    is_deleted INTEGER DEFAULT 0,
    deleted_at TEXT,
    deleted_by TEXT
);
"""


@pytest.fixture
def svc():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        s = StorageService()
        s._pg = None  # Force SQLite mode even if KAWKAB_DB_URL is set
        s._use_postgres = False
        s._db_path = db_path
        s._conn = sqlite3.connect(str(db_path))
        s._conn.row_factory = sqlite3.Row
        s._conn.executescript(CREATE_SQL)
        s._conn.commit()
        yield s
        if s._conn:
            s._conn.close()


@pytest.fixture
def handler(svc):
    mock_bridge = MagicMock()
    services = {"storage_service": svc}
    h = CodingHandler(mock_bridge, services)
    return h


async def _mid(svc):
    return await svc.save_match("Test", "/v/test.mp4", "Home", "Away")


@pytest.mark.asyncio
async def test_save_tag(handler, svc):
    mid = await _mid(svc)
    tag = {"event_type": "pass", "video_time": 30.0, "player_name": "Messi", "team": "home"}
    result = await handler.save_tag(mid, json.dumps(tag))
    data = json.loads(result)
    assert data["success"] is True
    assert data["tag_id"] > 0


@pytest.mark.asyncio
async def test_get_tags(handler, svc):
    mid = await _mid(svc)
    tag = {"event_type": "shot", "video_time": 45.0, "player_name": "Ronaldo"}
    await handler.save_tag(mid, json.dumps(tag))

    result = await handler.get_tags(mid)
    data = json.loads(result)
    assert data["success"] is True
    assert len(data["tags"]) == 1
    assert data["tags"][0]["event_type"] == "shot"


@pytest.mark.asyncio
async def test_get_tags_empty(handler, svc):
    mid = await _mid(svc)
    result = await handler.get_tags(mid)
    data = json.loads(result)
    assert data["success"] is True
    assert data["tags"] == []


@pytest.mark.asyncio
async def test_update_tag(handler, svc):
    mid = await _mid(svc)
    tag = {"event_type": "pass", "video_time": 10.0}
    save_result = json.loads(await handler.save_tag(mid, json.dumps(tag)))
    tag_id = save_result["tag_id"]

    result = await handler.update_tag(
        tag_id, json.dumps({"event_type": "shot", "notes": "Updated"})
    )
    data = json.loads(result)
    assert data["success"] is True

    get_result = json.loads(await handler.get_tags(mid))
    assert get_result["tags"][0]["event_type"] == "shot"


@pytest.mark.asyncio
async def test_delete_tag(handler, svc):
    mid = await _mid(svc)
    tag = {"event_type": "pass", "video_time": 10.0}
    save_result = json.loads(await handler.save_tag(mid, json.dumps(tag)))
    tag_id = save_result["tag_id"]

    result = json.loads(await handler.delete_tag(tag_id))
    assert result["success"] is True

    get_result = json.loads(await handler.get_tags(mid))
    assert get_result["tags"] == []


@pytest.mark.asyncio
async def test_delete_nonexistent_tag(handler):
    result = json.loads(await handler.delete_tag(999))
    assert result["success"] is False


@pytest.mark.asyncio
async def test_get_tag_stats(handler, svc):
    mid = await _mid(svc)
    await handler.save_tag(mid, json.dumps({"event_type": "pass", "video_time": 10.0}))
    await handler.save_tag(mid, json.dumps({"event_type": "shot", "video_time": 20.0}))

    result = json.loads(await handler.get_tag_stats(mid))
    assert result["success"] is True
    assert result["stats"]["total"] == 2


@pytest.mark.asyncio
async def test_get_tags_by_type(handler, svc):
    mid = await _mid(svc)
    await handler.save_tag(mid, json.dumps({"event_type": "pass", "video_time": 10.0}))
    await handler.save_tag(mid, json.dumps({"event_type": "pass", "video_time": 15.0}))
    await handler.save_tag(mid, json.dumps({"event_type": "shot", "video_time": 20.0}))

    result = json.loads(await handler.get_tags_by_type(mid, "pass"))
    assert result["success"] is True
    assert len(result["tags"]) == 2


@pytest.mark.asyncio
async def test_get_tags_by_player(handler, svc):
    mid = await _mid(svc)
    await handler.save_tag(
        mid, json.dumps({"event_type": "pass", "video_time": 10.0, "player_track_id": 1})
    )
    await handler.save_tag(
        mid, json.dumps({"event_type": "shot", "video_time": 20.0, "player_track_id": 1})
    )
    await handler.save_tag(
        mid, json.dumps({"event_type": "tackle", "video_time": 30.0, "player_track_id": 2})
    )

    result = json.loads(await handler.get_tags_by_player(mid, 1))
    assert result["success"] is True
    assert len(result["tags"]) == 2


@pytest.mark.asyncio
async def test_get_match_players_simple(handler, svc):
    mid = await _mid(svc)
    await svc.save_player(
        mid, {"track_id": 1, "name": "Messi", "jersey_number": 10, "team": "home"}
    )
    await svc.save_player(
        mid, {"track_id": 2, "name": "Ronaldo", "jersey_number": 7, "team": "away"}
    )

    result = json.loads(await handler.get_match_players_simple(mid))
    assert result["success"] is True
    assert len(result["players"]) == 2
    assert result["players"][0]["name"] == "Messi"


@pytest.mark.asyncio
async def test_default_tag_templates(handler):
    result = json.loads(await handler.get_default_tag_templates())
    assert result["success"] is True
    templates = result["templates"]
    assert "categories" in templates
    assert len(templates["categories"]) > 0
    assert templates["categories"][0]["id"] == "attack"
    assert len(templates["categories"][0]["buttons"]) > 0
    assert "shortcut" in templates["categories"][0]["buttons"][0]


@pytest.mark.asyncio
async def test_extract_tag_clip_no_match(handler):
    result = json.loads(await handler.extract_tag_clip(999, 1))
    assert "error" in result


@pytest.mark.asyncio
async def test_invalid_json_handling(handler, svc):
    mid = await _mid(svc)
    result = json.loads(await handler.save_tag(mid, "not-json"))
    assert "error" in result

    result = json.loads(await handler.update_tag(1, "not-json"))
    assert "error" in result
