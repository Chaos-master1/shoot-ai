"""Tests for the elite-readiness API endpoints (Phases 1/3):
    POST /api/v1/matches/import/tracking
    POST /api/v1/matches/import/events
    GET  /api/v1/audit/events

Follows test_api_v1.py's conventions: temp cloud DB, real JWT tokens,
role-gated access checks.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("KAWKAB_JWT_SECRET", "test-secret-for-testing-purposes-only")
os.environ["KAWKAB_CLOUD_DB"] = os.path.join(tempfile.gettempdir(), "kawkab_test_vendor_api.db")
os.environ["KAWKAB_RATE_LIMIT_DISABLE"] = "1"
# Vendor feed files must live under the SecurityValidator allowlist dir
# (Documents/KawkabAI); XDG_DOCUMENTS_DIR redirects that for this test run.
os.environ["XDG_DOCUMENTS_DIR"] = tempfile.mkdtemp(prefix="kawkab_vendor_docs_")
os.environ.pop("KAWKAB_DB_URL", None)

from fastapi.testclient import TestClient  # noqa: E402

from kawkab.cloud.auth import create_access_token  # noqa: E402
from kawkab.cloud.database import get_cloud_db  # noqa: E402
from kawkab.cloud.server import app  # noqa: E402

client = TestClient(app)


def _ensure_storage_ready():
    """Point _get_storage()'s singleton at a migrated scratch DB.

    Mirrors test_api_v1.py's shared helper: the singleton is per-process,
    never initialized elsewhere, and TestClient runs requests on its own
    thread — hence check_same_thread=False.
    """
    import sqlite3
    from pathlib import Path

    from kawkab.api.api_v1 import _get_storage
    from kawkab.core.migration_manager import MigrationManager

    svc = _get_storage()
    if svc._conn is None:
        scratch_db = (
            Path(tempfile.gettempdir()) / f"kawkab_test_vendor_api_storage_{os.getpid()}.db"
        )
        MigrationManager(scratch_db, Path("src/kawkab/migrations")).migrate()
        svc._db_path = scratch_db
        svc._conn = sqlite3.connect(str(scratch_db), check_same_thread=False)
        svc._conn.row_factory = sqlite3.Row


_ensure_storage_ready()


def _ensure_user(email, username, role):
    db = get_cloud_db()
    row = db.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
    if not row:
        db.execute(
            "INSERT OR IGNORE INTO users (username, email, password_hash, display_name, role) VALUES (?, ?, ?, ?, ?)",
            (username, email, "hash", f"{role.title()} Tester", role),
        )
        db.commit()
        row = db.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
    if row is None:
        return {}
    token = create_access_token(row["id"], role=role)
    return {"Authorization": f"Bearer {token}"}


def _analyst_headers():
    return _ensure_user("vendor_analyst@test.com", "vendortest", "analyst")


def _viewer_headers():
    return _ensure_user("vendor_viewer@test.com", "vendorviewer", "viewer")


@pytest.fixture(scope="module")
def skillcorner_file(tmp_path_factory):
    from kawkab.core.paths import get_paths

    data = {
        "fps": 25,
        "players": [
            {"track_id": 1, "name": "A", "side": "home", "position": "GK"},
            {"track_id": 2, "name": "B", "side": "away", "position": "ST"},
        ],
        "frames": [
            {
                "frame_id": 0,
                "time": 0,
                "period": 1,
                "players": [
                    {"track_id": 1, "x": -0.5, "y": 0.0},
                    {"track_id": 2, "x": 0.5, "y": 0.1},
                ],
                "ball": {"x": 0.0, "y": 0.0, "z": 0.0},
            },
            {
                "frame_id": 1,
                "time": 40,
                "period": 1,
                "players": [
                    {"track_id": 1, "x": -0.51, "y": 0.0},
                    {"track_id": 2, "x": 0.49, "y": 0.1},
                ],
                "ball": {"x": 0.01, "y": 0.0, "z": 0.0},
            },
        ],
    }
    feed_dir = Path(get_paths().documents) / "feeds"
    feed_dir.mkdir(parents=True, exist_ok=True)
    path = feed_dir / "sc.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def wyscout_file(tmp_path_factory):
    from kawkab.core.paths import get_paths

    data = {
        "match": {"matchId": "w1", "homeTeam": {"name": "H"}, "awayTeam": {"name": "A2"}},
        "events": [
            {
                "id": "1",
                "matchId": "w1",
                "teamId": "1",
                "playerId": "10",
                "eventName": "Passes",
                "minute": 5,
                "second": 0,
                "x": 50.0,
                "y": 50.0,
                "endX": 60.0,
                "endY": 50.0,
                "tags": [],
            },
        ],
    }
    feed_dir = Path(get_paths().documents) / "feeds"
    feed_dir.mkdir(parents=True, exist_ok=True)
    path = feed_dir / "wy.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestTrackingImportEndpoint:
    def test_requires_auth(self, skillcorner_file):
        resp = client.post(
            "/api/v1/matches/import/tracking",
            json={
                "file_path": str(skillcorner_file),
            },
        )
        assert resp.status_code == 401

    def test_requires_analyst_role(self, skillcorner_file):
        resp = client.post(
            "/api/v1/matches/import/tracking",
            headers=_viewer_headers(),
            json={"file_path": str(skillcorner_file)},
        )
        assert resp.status_code == 403

    def test_missing_file_outside_allowlist_400(self):
        """A path outside the Documents/KawkabAI allowlist is rejected as
        traversal (400) BEFORE existence is checked — the security check
        deliberately wins, so no oracle about the filesystem leaks."""
        resp = client.post(
            "/api/v1/matches/import/tracking",
            headers=_analyst_headers(),
            json={"file_path": "/nonexistent/nope.json"},
        )
        assert resp.status_code == 400

    def test_missing_file_inside_allowlist_404(self):
        from kawkab.core.paths import get_paths

        missing = Path(get_paths().documents) / "feeds" / "definitely_missing.json"
        resp = client.post(
            "/api/v1/matches/import/tracking",
            headers=_analyst_headers(),
            json={"file_path": str(missing)},
        )
        assert resp.status_code == 404

    def test_successful_import(self, skillcorner_file):
        resp = client.post(
            "/api/v1/matches/import/tracking",
            headers=_analyst_headers(),
            json={
                "file_path": str(skillcorner_file),
                "vendor": "skillcorner",
                "home_team": "API Home",
                "away_team": "API Away",
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["success"] is True
        assert data["vendor"] == "skillcorner"
        assert data["frames_imported"] == 2
        assert data["players_registered"] == 2
        assert data["checksum"]

    def test_bad_vendor_400(self, skillcorner_file):
        resp = client.post(
            "/api/v1/matches/import/tracking",
            headers=_analyst_headers(),
            json={"file_path": str(skillcorner_file), "vendor": "tracab"},
        )
        assert resp.status_code == 400


class TestEventImportEndpoint:
    def test_requires_auth(self, wyscout_file):
        resp = client.post(
            "/api/v1/matches/import/events",
            json={
                "vendor": "wyscout",
                "file_path": str(wyscout_file),
            },
        )
        assert resp.status_code == 401

    def test_unsupported_vendor_400(self, wyscout_file):
        resp = client.post(
            "/api/v1/matches/import/events",
            headers=_analyst_headers(),
            json={"vendor": "statsbomb", "file_path": str(wyscout_file)},
        )
        assert resp.status_code == 400

    def test_successful_wyscout_import(self, wyscout_file):
        resp = client.post(
            "/api/v1/matches/import/events",
            headers=_analyst_headers(),
            json={"vendor": "wyscout", "file_path": str(wyscout_file)},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["success"] is True
        assert data["vendor"] == "wyscout"
        assert data["events_imported"] == 1


class TestSeasonImportEndpoint:
    """POST /matches/import/statsbomb/season — the bulk season ingest."""

    @pytest.fixture(scope="class")
    def season_setup(self):
        """A fresh per-run directory inside the documents allowlist holding
        one real StatsBomb event file + one non-event JSON.

        Per-run uniqueness matters: the documents dir persists in /tmp
        across runs, so a fixed directory name would accumulate event
        files from earlier runs and each would import (distinct stems)."""
        import shutil
        import time

        from kawkab.core.paths import get_paths

        corpus = Path(__file__).resolve().parents[2] / "data" / "statsbomb_corpus"
        files = sorted(corpus.glob("*.json"))
        if not files:
            pytest.skip("statsbomb corpus not on this machine")

        sdir = Path(get_paths().documents) / f"season_test_{os.getpid()}_{int(time.time() * 1000)}"
        sdir.mkdir(parents=True, exist_ok=True)
        stem = f"{int(time.time())}"
        (sdir / f"{stem}.json").write_text(files[0].read_text(encoding="utf-8"), encoding="utf-8")
        (sdir / "lineups.json").write_text("{}", encoding="utf-8")
        yield sdir, stem
        shutil.rmtree(sdir, ignore_errors=True)

    def test_requires_auth(self, season_setup):
        sdir, _ = season_setup
        resp = client.post("/api/v1/matches/import/statsbomb/season", json={"directory": str(sdir)})
        assert resp.status_code == 401

    def test_requires_analyst_role(self, season_setup):
        sdir, _ = season_setup
        resp = client.post(
            "/api/v1/matches/import/statsbomb/season",
            headers=_viewer_headers(),
            json={"directory": str(sdir)},
        )
        assert resp.status_code == 403

    def test_directory_outside_allowlist_400(self):
        """Traversal wins before existence, same discipline as the file
        endpoints — no filesystem oracle."""
        resp = client.post(
            "/api/v1/matches/import/statsbomb/season",
            headers=_analyst_headers(),
            json={"directory": "/nonexistent/season"},
        )
        assert resp.status_code == 400

    def test_successful_season_import_and_audit(self, season_setup):
        sdir, stem = season_setup
        resp = client.post(
            "/api/v1/matches/import/statsbomb/season",
            headers=_analyst_headers(),
            json={"directory": str(sdir), "competition": "Test League"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["success"] is True
        assert data["source"] == "statsbomb"
        assert data["imported"] == 1
        assert data["skipped_not_events"] == 1
        assert data["matches"][0]["match_id"] > 0
        assert data["matches"][0]["competition"] == "Test League"

        # The import is audited under the season action
        audit = client.get(
            "/api/v1/audit/events", headers=_analyst_headers(), params={"action": "season.imported"}
        ).json()
        assert any(stem in str(e) for e in audit["events"])


class TestStatsBombSingleImportValidation:
    """Regression: the single-file StatsBomb endpoint must accept its own
    .json feeds (it previously ran them through validate_video_path, which
    rejects every non-video extension)."""

    def test_accepts_json_file_path(self):
        import time

        from kawkab.core.paths import get_paths

        corpus = Path(__file__).resolve().parents[2] / "data" / "statsbomb_corpus"
        files = sorted(corpus.glob("*.json"))
        if not files:
            pytest.skip("statsbomb corpus not on this machine")

        feeds = Path(get_paths().documents) / "feeds"
        feeds.mkdir(parents=True, exist_ok=True)
        path = feeds / f"api_single_{os.getpid()}_{int(time.time() * 1000)}.json"
        path.write_text(files[0].read_text(encoding="utf-8"), encoding="utf-8")
        try:
            resp = client.post(
                "/api/v1/matches/import/statsbomb",
                headers=_analyst_headers(),
                json={"file_path": str(path)},
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["success"] is True
        finally:
            path.unlink(missing_ok=True)

    def test_analyst_can_read_audit_trail(self, skillcorner_file):
        # Generate an audited event first
        client.post(
            "/api/v1/matches/import/tracking",
            headers=_analyst_headers(),
            json={
                "file_path": str(skillcorner_file),
                "vendor": "skillcorner",
                "home_team": "Audit Home",
                "away_team": "Audit Away",
                "match_name": f"Audit trail match {id(skillcorner_file)}",
            },
        )
        resp = client.get("/api/v1/audit/events", headers=_analyst_headers())
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["success"] is True
        assert isinstance(data["events"], list)
        assert data["total"] == len(data["events"])
        # The tracking import above logged match.imported events
        assert any(e["action"] == "match.imported" for e in data["events"])

    def test_action_filter(self):
        resp = client.get(
            "/api/v1/audit/events", headers=_analyst_headers(), params={"action": "match.imported"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert all(e["action"] == "match.imported" for e in data["events"])
