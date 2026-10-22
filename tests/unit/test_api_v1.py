"""Tests for REST API v1 endpoints."""

from __future__ import annotations

import os
import tempfile

from fastapi.testclient import TestClient

# Use isolated temp DB for testing
os.environ.setdefault("KAWKAB_JWT_SECRET", "test-secret-for-testing-purposes-only")
os.environ["KAWKAB_CLOUD_DB"] = os.path.join(tempfile.gettempdir(), "kawkab_test_api_v1.db")
os.environ["KAWKAB_RATE_LIMIT_DISABLE"] = "1"
os.environ.pop("KAWKAB_DB_URL", None)  # tests use SQLite for analytics

from kawkab.cloud.auth import create_access_token
from kawkab.cloud.database import get_cloud_db
from kawkab.cloud.server import app

client = TestClient(app)


def _ensure_user(email, username, role):
    """Get or create a test user, return auth headers."""
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


def _admin_headers():
    return _ensure_user("admin@test.com", "admintest", "admin")


def _analyst_headers():
    return _ensure_user("analyst@test.com", "analysttest", "analyst")


class TestApiHealth:
    def test_api_health(self):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_api_version(self):
        resp = client.get("/api/v1/health")
        assert resp.json()["api_version"] == "v1"

    def test_server_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestApiMatches:
    def test_list_matches(self):
        # Hot reads now raise on an uninitialized store (honesty contract),
        # so these tests must run against a real scratch DB instead of
        # silently passing against a never-initialized singleton.
        _ensure_storage_ready()
        resp = client.get("/api/v1/matches", headers=_analyst_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data or "matches" in data
        assert "total" in data

    def test_get_match_not_found(self):
        resp = client.get("/api/v1/matches/999999", headers=_analyst_headers())
        assert resp.status_code == 404

    def test_get_match_players_not_found(self):
        # get_match_players() previously never checked the match itself
        # existed before listing its players -- a nonexistent match_id
        # silently returned 200 with an empty list, inconsistent with
        # test_get_match_not_found just above (same fake id, correctly
        # 404). Fixed as part of adding match-ownership checks (every
        # match-scoped route now fetches and validates the match first).
        resp = client.get("/api/v1/matches/999999/players", headers=_analyst_headers())
        assert resp.status_code == 404


class TestApiModelComparison:
    def test_compare_models_empty(self):
        resp = client.post(
            "/api/v1/model-comparison?n_folds=0", json=[], headers=_analyst_headers()
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data

    def test_compare_models_with_shots(self):
        shots = [
            {"xg_heuristic": 0.5, "is_goal": True, "distance_m": 5.0, "angle_deg": 0.0},
            {"xg_heuristic": 0.1, "is_goal": False, "distance_m": 20.0, "angle_deg": 30.0},
            {"xg_heuristic": 0.05, "is_goal": False, "distance_m": 30.0, "angle_deg": 45.0},
            {"xg_heuristic": 0.3, "is_goal": True, "distance_m": 12.0, "angle_deg": 10.0},
        ]
        resp = client.post(
            "/api/v1/model-comparison?n_folds=2", json=shots, headers=_analyst_headers()
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["models"]) >= 1


class TestApiWebhooks:
    def test_create_webhook(self):
        resp = client.post(
            "/api/v1/webhooks",
            headers=_admin_headers(),
            json={
                "url": "https://example.com/hook",
                "events": ["match.analyzed"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["url"] == "https://example.com/hook"

    def test_list_webhooks(self):
        resp = client.get("/api/v1/webhooks", headers=_admin_headers())
        assert resp.status_code == 200

    def test_delete_webhook(self):
        resp = client.delete("/api/v1/webhooks/1", headers=_admin_headers())
        assert resp.status_code == 200


class TestApiMonitoring:
    def test_monitoring_dashboard(self):
        resp = client.get("/api/v1/monitoring/dashboard", headers=_admin_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert "total_evaluations" in data

    def test_drift_alerts(self):
        resp = client.get("/api/v1/monitoring/drift", headers=_admin_headers())
        assert resp.status_code == 200
        assert "alerts" in resp.json()


class TestApiRecruitment:
    def test_search_players(self):
        resp = client.post(
            "/api/v1/recruitment/search",
            headers=_analyst_headers(),
            json={
                "position": "forward",
                "min_age": 20,
                "max_age": 30,
            },
        )
        assert resp.status_code == 200

    def test_transfer_fee(self):
        resp = client.get("/api/v1/recruitment/transfer-fee/Mbappe", headers=_analyst_headers())
        assert resp.status_code == 200
        assert "estimated_fee" in resp.json()

    def test_shortlist(self):
        resp = client.get("/api/v1/recruitment/shortlist", headers=_analyst_headers())
        assert resp.status_code == 200


class TestApiGamePlan:
    def test_game_plan(self):
        # get_game_plan() previously never checked the match existed
        # before generating a plan for it -- this test relied on match
        # id 1 happening to already exist in whatever ambient local DB
        # _get_storage() resolves to, which is what let it pass before
        # match-ownership checks required a real match record. Create one
        # explicitly instead of assuming pre-existing state.
        import asyncio
        import sqlite3
        from pathlib import Path

        from kawkab.api.api_v1 import _get_storage
        from kawkab.core.migration_manager import MigrationManager

        svc = _get_storage()
        if svc._conn is None:
            # _get_storage()'s module-level singleton is never initialized
            # anywhere else in this test file either -- every other test
            # that reads via it (e.g. test_list_matches) has been passing
            # against an empty, uninitialized store, not real data.
            # Point it at a scratch DB rather than the real ~/KawkabAI
            # database this process would otherwise share with an actual
            # desktop app install. Migrated and connected manually (not
            # via svc.initialize()) with check_same_thread=False: this
            # connection is created here on the test's own thread, but
            # TestClient runs the app -- and every subsequent request
            # using this same singleton -- on its own dedicated thread.
            scratch_db = (
                Path(tempfile.gettempdir()) / f"kawkab_test_api_v1_storage_{os.getpid()}.db"
            )
            MigrationManager(scratch_db, Path("src/kawkab/migrations")).migrate()
            svc._db_path = scratch_db
            svc._conn = sqlite3.connect(str(scratch_db), check_same_thread=False)
            svc._conn.row_factory = sqlite3.Row
        match_id = asyncio.run(
            svc.save_match(name="Game Plan Test Match", video_path="gameplan_test.mp4")
        )
        assert match_id != 0, "save_match returned 0 -- storage not initialized?"

        resp = client.get(f"/api/v1/game-plan/{match_id}/vs/Barcelona", headers=_analyst_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert "opponent" in data
        assert data["opponent"] == "Barcelona"


def _ensure_storage_ready():
    """Shared scratch-DB singleton init, mirroring TestApiGamePlan
    .test_game_plan above (see its comment for why this is needed --
    _get_storage()'s singleton is never reset between test classes in
    this file, so whichever test initializes it first wins for the rest
    of the process)."""
    import sqlite3
    from pathlib import Path

    from kawkab.api.api_v1 import _get_storage
    from kawkab.core.migration_manager import MigrationManager

    svc = _get_storage()
    if svc._conn is None:
        scratch_db = Path(tempfile.gettempdir()) / f"kawkab_test_api_v1_storage_{os.getpid()}.db"
        MigrationManager(scratch_db, Path("src/kawkab/migrations")).migrate()
        svc._db_path = scratch_db
        svc._conn = sqlite3.connect(str(scratch_db), check_same_thread=False)
        svc._conn.row_factory = sqlite3.Row
    return svc


class TestApiTacticalPressingReport:
    """get_tactical_shapes/get_pressing/get_match_report each called a
    nonexistent class or method (PressingClassifier, TacticalReportGenerator,
    TacticalShapeAnalyzer.analyze instead of .analyze_shapes) -- an instant
    500 (AttributeError/ImportError) on every call, before the routes' own
    match-ownership/auth checks even mattered."""

    def _match_with_events(self):
        import asyncio

        svc = _ensure_storage_ready()
        match_id = asyncio.run(
            svc.save_match(name="Tactical Test Match", video_path="tactical_test.mp4")
        )
        events = [
            {
                "type": "pass",
                "timestamp": 1.0,
                "team": "home",
                "from_track_id": 1,
                "to_track_id": 2,
                "completed": True,
            },
            {
                "type": "pass",
                "timestamp": 5.0,
                "team": "home",
                "from_track_id": 2,
                "to_track_id": 3,
                "completed": True,
            },
            {
                "type": "shot",
                "timestamp": 10.0,
                "team": "home",
                "from_track_id": 3,
                "completed": True,
            },
            {
                "type": "tackle",
                "timestamp": 15.0,
                "team": "away",
                "from_track_id": 8,
                "completed": True,
            },
            {
                "type": "pass",
                "timestamp": 20.0,
                "team": "away",
                "from_track_id": 8,
                "to_track_id": 9,
                "completed": True,
            },
        ]
        for e in events:
            asyncio.run(svc.save_event(match_id, e))
        return match_id

    def test_tactical_shapes_no_longer_500s(self):
        match_id = self._match_with_events()
        resp = client.get(
            f"/api/v1/matches/{match_id}/analysis/tactical-shapes", headers=_analyst_headers()
        )
        assert resp.status_code == 200
        data = resp.json()
        assert set(data["shapes"].keys()) == {"home", "away"}

    def test_pressing_no_longer_500s(self):
        match_id = self._match_with_events()
        resp = client.get(
            f"/api/v1/matches/{match_id}/analysis/pressing", headers=_analyst_headers()
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["home_ppda"], float)
        assert isinstance(data["pressing_triggers"], int)

    def test_match_report_no_longer_500s(self):
        match_id = self._match_with_events()
        resp = client.get(f"/api/v1/matches/{match_id}/analysis/report", headers=_analyst_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert data["match_id"] == match_id
        assert isinstance(data["tactical_observations"], list)


class TestApiPlayerFitnessAndInjuryRisk:
    """get_player_fitness/get_player_injury_risk called methods that don't
    exist anywhere (PhysicalLoadService.compute_load, WorkloadService
    .compute_acwr, InjuryRiskPredictor.predict_risk) -- an instant 500 on
    every call. Fixed to use the real, GPS-import-backed data (gps_sessions/
    acwr_daily) instead, since PhysicalLoadService/WorkloadService need raw
    per-frame tracking / season-long history this endpoint doesn't have."""

    def _match_with_player(self):
        import asyncio

        svc = _ensure_storage_ready()
        match_id = asyncio.run(
            svc.save_match(name="Fitness Test Match", video_path="fitness_test.mp4")
        )
        asyncio.run(
            svc.save_players_bulk(
                match_id,
                [
                    {
                        "track_id": 501,
                        "name": "Test Player",
                        "team": "home",
                        "position": "FWD",
                        "jersey_number": 9,
                    },
                ],
            )
        )
        return match_id

    def test_fitness_without_gps_data_is_honest_not_500(self):
        match_id = self._match_with_player()
        resp = client.get(
            f"/api/v1/players/501/fitness?match_id={match_id}", headers=_admin_headers()
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["player_name"] == "Test Player"
        assert data["total_distance"] == 0.0
        assert data["workload_score"] == 0.0

    def test_fitness_with_real_gps_and_acwr_data(self):
        import asyncio

        match_id = self._match_with_player()
        svc = _ensure_storage_ready()
        sid = asyncio.run(svc.save_gps_session(match_id, 501, "match", "catapult"))
        asyncio.run(
            svc.update_gps_session_stats(sid, {"total_distance_m": 10800.0, "max_speed_kmh": 31.2})
        )
        asyncio.run(svc.save_acwr(501, "2026-08-18", 5000.0, 4500.0, 1.11))

        resp = client.get(
            f"/api/v1/players/501/fitness?match_id={match_id}", headers=_admin_headers()
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_distance"] == 10800.0
        assert data["max_speed"] == 31.2
        assert data["workload_score"] == 1.11

    def test_injury_risk_without_data_reports_data_unavailable(self):
        resp = client.get("/api/v1/players/999888/injury-risk", headers=_admin_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert data["data_available"] is False

    def test_injury_risk_with_real_acwr_data(self):
        import asyncio

        svc = _ensure_storage_ready()
        asyncio.run(svc.save_acwr(777, "2026-08-18", 6000.0, 3000.0, 2.0))
        resp = client.get("/api/v1/players/777/injury-risk", headers=_admin_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert data["data_available"] is True
        assert data["acwr"] == 2.0
        assert "risk_score" in data


class TestApiSquadInjuryReport:
    """get_squad_injury_report previously constructed a bare StorageService()
    (bypassing the module's own _get_storage() singleton) and called a
    method that didn't exist on the class at all."""

    def test_squad_injury_report_no_longer_500s(self):
        resp = client.get("/api/v1/squad/999777/injury-report", headers=_admin_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_active"] == 0


class TestApiSeason:
    def test_season_summary(self):
        resp = client.get("/api/v1/season/summary", headers=_analyst_headers())
        assert resp.status_code == 200


class TestStorageError503:
    """Honest storage failures surface as 503 storage_unavailable, not 500
    tracebacks or silent empty lists (storage_errors.py contract)."""

    def test_not_initialized_maps_to_503(self, monkeypatch):
        from kawkab.api import api_v1
        from kawkab.services.storage_errors import StorageNotInitializedError

        async def boom(*args, **kwargs):
            raise StorageNotInitializedError("get_all_matches")

        monkeypatch.setattr(api_v1, "_storage", None, raising=False)
        monkeypatch.setattr(
            type(api_v1._get_storage()),
            "get_all_matches",
            lambda self: boom(),
        )
        resp = client.get("/api/v1/matches", headers=_analyst_headers())
        assert resp.status_code == 503
        data = resp.json()
        assert data["error"] == "storage_unavailable"
        assert data["operation"] == "get_all_matches"

    def test_read_failure_maps_to_503(self, monkeypatch):
        from kawkab.api import api_v1
        from kawkab.services.storage_errors import StorageReadError

        async def boom(*args, **kwargs):
            raise StorageReadError("get_all_matches", RuntimeError("disk gone"))

        monkeypatch.setattr(
            type(api_v1._get_storage()),
            "get_all_matches",
            lambda self: boom(),
        )
        resp = client.get("/api/v1/matches", headers=_analyst_headers())
        assert resp.status_code == 503
        assert "disk gone" in resp.json()["detail"]
