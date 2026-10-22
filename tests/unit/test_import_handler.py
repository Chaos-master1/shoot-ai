"""Tests for the ImportHandler (bridge import slots).

Mirrors test_season_analytics_handler.py conventions: real SQLite storage
with the real migration chain; handlers instantiated directly with a
bridge stub carrying a recording signal.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS = PROJECT_ROOT / "data" / "statsbomb_corpus"


class _RecordingSignal:
    """Stands in for the bridge's importProgress Signal."""

    def __init__(self) -> None:
        self.emissions: list[tuple[float, str]] = []

    def emit(self, n: float, file: str) -> None:
        self.emissions.append((n, file))


class _StubBridge:
    def __init__(self) -> None:
        self.importProgress = _RecordingSignal()


@pytest.fixture()
def store(tmp_path):
    from kawkab.core.migration_manager import MigrationManager
    from kawkab.services.storage_service import StorageService

    db = tmp_path / "import_handler.db"
    MigrationManager(db, PROJECT_ROOT / "src" / "kawkab" / "migrations").migrate()
    svc = StorageService()
    svc._db_path = db
    svc._conn = sqlite3.connect(str(db))
    svc._conn.row_factory = sqlite3.Row
    return svc


@pytest.fixture()
def handler(store):
    from kawkab.ui.bridge_handlers.bridge_import import ImportHandler

    return ImportHandler(bridge=_StubBridge(), services={"storage_service": store})


def _corpus_dir() -> Path | None:
    if not CORPUS.is_dir():
        return None
    event_files = [p for p in sorted(CORPUS.glob("*.json")) if not p.name.endswith(".meta.json")]
    return CORPUS if event_files else None


class TestImportSeasonDirectory:
    def test_happy_path_with_progress(self, store, handler, tmp_path):
        src = _corpus_dir()
        if src is None:
            pytest.skip("statsbomb corpus not on this machine")
        # A 2-file scratch directory keeps the test fast
        files = [p for p in sorted(src.glob("*.json")) if not p.name.endswith(".meta.json")][:2]
        work = tmp_path / "season"
        work.mkdir()
        for f in files:
            (work / f.name).write_bytes(f.read_bytes())

        out = asyncio.run(handler.import_season_directory(str(work), "Test Cup"))
        r = json.loads(out)
        assert r["success"] is True
        assert r["imported"] == 2
        assert r["failed"] == 0
        # Progress fired once per imported file, through the stub signal
        emissions = handler._bridge.importProgress.emissions
        assert len(emissions) == 2
        assert emissions[0][0] == 1.0 and emissions[1][0] == 2.0

    def test_idempotent_rerun(self, store, handler, tmp_path):
        src = _corpus_dir()
        if src is None:
            pytest.skip("statsbomb corpus not on this machine")
        files = [p for p in sorted(src.glob("*.json")) if not p.name.endswith(".meta.json")][:1]
        work = tmp_path / "season2"
        work.mkdir()
        for f in files:
            (work / f.name).write_bytes(f.read_bytes())

        first = json.loads(asyncio.run(handler.import_season_directory(str(work))))
        second = json.loads(asyncio.run(handler.import_season_directory(str(work))))
        assert first["imported"] == 1
        assert second["imported"] == 0
        assert second["skipped_already"] == 1

    def test_bad_directory_is_an_error_not_a_crash(self, handler):
        out = asyncio.run(handler.import_season_directory("/nonexistent/dir"))
        r = json.loads(out)
        assert r["success"] is False
        assert r["error"]


class TestImportTrackingFile:
    def test_skillcorner_roundtrip(self, store, handler, tmp_path, monkeypatch):
        data = {
            "fps": 10,
            "players": [
                {"track_id": 1, "name": "A", "side": "home"},
                {"track_id": 2, "name": "B", "side": "away"},
            ],
            "frames": [
                {
                    "frame_id": i,
                    "time": i * 100,
                    "period": 1,
                    "players": [
                        {"track_id": 1, "x": -0.8, "y": 0.1},
                        {"track_id": 2, "x": 0.7, "y": -0.2},
                    ],
                    "ball": {"x": 0.0, "y": 0.0, "z": 0.0},
                }
                for i in range(3)
            ],
        }
        f = tmp_path / "sc.json"
        f.write_text(json.dumps(data), encoding="utf-8")

        # The handler now validates vendor paths (allowlist dir). The
        # stubbed paths singleton points at /tmp/kawkab_test; copy the
        # feed there so validation passes.
        from kawkab.core.paths import get_paths

        allowed = get_paths().documents / "vendor_test"
        allowed.mkdir(parents=True, exist_ok=True)
        f_allowed = allowed / "sc.json"
        f_allowed.write_text(json.dumps(data), encoding="utf-8")
        out = asyncio.run(handler.import_tracking_file(str(f_allowed)))
        r = json.loads(out)
        assert r["success"] is True
        assert r["frames_imported"] == 3
        assert r["players_registered"] == 2
        assert r["quality"]["frames"] == 3

    def test_missing_file_returns_error_json(self, handler):
        out = asyncio.run(handler.import_tracking_file("/nonexistent/feed.json"))
        r = json.loads(out)
        assert r["success"] is False
        assert r["error"]


class TestImportEventFile:
    def test_unreadable_file_returns_error_json(self, handler, tmp_path):
        f = tmp_path / "broken.xml"
        f.write_text("<not-xml", encoding="utf-8")
        out = asyncio.run(handler.import_event_file(str(f)))
        r = json.loads(out)
        assert r["success"] is False
        assert r["error"]
