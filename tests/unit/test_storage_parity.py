"""Postgres/SQLite storage parity tests (Phase 2c, elite-readiness roadmap).

The Postgres adapter (postgres_storage.py) is a hand-written parallel of
StorageService, not a drop-in twin — CLAUDE.md's "storage backend
divergence" known gap. These tests pin the *contract* both backends must
satisfy: same method surface, same result shapes, same return types,
same fallback behavior when no database is connected. Running the same
assertion helper against both adapters means a divergence introduced on
one side fails here, not in production at a club.

The SQLite side runs against a REAL migrated scratch DB (real migration
chain 001-030, same pattern as test_statsbomb_import_service.py). The
Postgres side runs against the adapter's documented no-pool fallback
behavior (no live Postgres in unit CI — the pool paths need a real
server, exercised by the integration job when one exists).

Method surface equality is the core regression guard: for every public
async storage method the SQLite service exposes (minus documented
SQLite-only methods like backup()), the Postgres adapter must expose the
same name — that is the contract StatsBomb/vendor import services and
the API layer rely on when the backend is swapped.
"""

from __future__ import annotations

import asyncio
import inspect
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def sqlite_storage(tmp_path):
    """Real SQLite storage with the REAL 001-030 migration chain."""
    from kawkab.core.migration_manager import MigrationManager
    from kawkab.services.storage_service import StorageService

    scratch_db = tmp_path / "test_parity.db"
    migrations_dir = PROJECT_ROOT / "src" / "kawkab" / "migrations"
    MigrationManager(scratch_db, migrations_dir).migrate()
    svc = StorageService()
    svc._db_path = scratch_db
    svc._conn = sqlite3.connect(str(scratch_db))
    svc._conn.row_factory = sqlite3.Row
    return svc


@pytest.fixture()
def pg_adapter():
    from kawkab.services.postgres_storage import PostgresStorageAdapter

    return PostgresStorageAdapter()  # no DSN -> not available -> fallback paths


# ── Method-surface parity (the divergence guard) ───────────────────────────

# Documented exceptions: SQLite-specific backup API, connection plumbing,
# and the frame/loop helper StorageService uses for __getattr__ delegation.
_SQLITE_ONLY = {
    "backup",
    "initialize",
    "close",
    "_get_conn",
    "_ensure_initialized",
    "_sanitize_column_name",
    "_log_error",
}


def _public_async_methods(cls) -> set[str]:
    out = set()
    for name, member in inspect.getmembers(cls):
        if name.startswith("_"):
            continue
        if inspect.iscoroutinefunction(member):
            out.add(name)
    return out


class TestMethodSurfaceParity:
    def test_migration_030_tables_exist_in_sqlite(self, sqlite_storage):
        """The vendor-provenance tables must be present after migration."""
        cur = sqlite_storage._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('tracking_imports', 'event_frame_links')"
        )
        names = {row["name"] for row in cur.fetchall()}
        assert names == {"tracking_imports", "event_frame_links"}

    def test_tracking_import_methods_on_both_backends(self, sqlite_storage, pg_adapter):
        for method in (
            "save_tracking_import",
            "get_tracking_imports",
            "get_tracking_import_by_id",
            "delete_tracking_import",
            "save_event_frame_links_bulk",
            "get_event_frame_links",
            "save_tracking_frames_bulk",
            "get_tracking_frames",
        ):
            assert hasattr(sqlite_storage, method), f"SQLite missing {method}"
            assert hasattr(pg_adapter, method), f"Postgres missing {method}"

    def test_import_service_contract_methods_on_both_backends(self, sqlite_storage, pg_adapter):
        """Methods the StatsBomb/vendor import services call — must exist on
        both backends or vendor imports break the moment a club runs Postgres."""
        for method in (
            "save_match",
            "save_player",
            "save_event",
            "save_advanced_metrics",
            "get_match_events",
            "get_tracking_imports",
        ):
            assert hasattr(sqlite_storage, method), f"SQLite missing {method}"
            assert hasattr(pg_adapter, method), f"Postgres missing {method}"

    def test_no_orphaned_sqlite_async_methods(self, sqlite_storage, pg_adapter):
        """Every public async StorageService method (minus documented
        SQLite-only ones) must exist on the Postgres adapter."""
        sqlite_methods = _public_async_methods(type(sqlite_storage)) - _SQLITE_ONLY
        pg_methods = _public_async_methods(type(pg_adapter))
        missing = sqlite_methods - pg_methods
        assert not missing, (
            f"Postgres adapter is missing {len(missing)} methods that exist on "
            f"StorageService: {sorted(missing)[:20]}"
        )


# ── Behavioral parity on shared fallback semantics ─────────────────────────


class TestNoConnectionFallbackParity:
    """Reads fail soft on no-connection (documented convention, still
    migrating cluster-by-cluster); writes in converted clusters raise
    StorageNotInitialized — silent 0 on a write hid failures for months
    (see storage_errors.py)."""

    def test_sqlite_save_match_no_conn_raises(self, tmp_path):
        from kawkab.services.storage_errors import StorageNotInitializedError
        from kawkab.services.storage_service import StorageService

        svc = StorageService()
        svc._conn = None
        with pytest.raises(StorageNotInitializedError):
            asyncio.run(svc.save_match("x", ""))

    def test_pg_save_match_no_pool_raises(self, pg_adapter):
        from kawkab.services.storage_errors import StorageNotInitializedError

        with pytest.raises(StorageNotInitializedError):
            asyncio.run(pg_adapter.save_match("x", ""))

    def test_sqlite_get_tracking_imports_no_conn_raises(self, tmp_path):
        from kawkab.services.storage_errors import StorageNotInitializedError
        from kawkab.services.storage_service import StorageService

        svc = StorageService()
        svc._conn = None
        with pytest.raises(StorageNotInitializedError):
            asyncio.run(svc.get_tracking_imports(1))

    def test_pg_get_tracking_imports_no_pool_raises(self, pg_adapter):
        from kawkab.services.storage_errors import StorageNotInitializedError

        with pytest.raises(StorageNotInitializedError):
            asyncio.run(pg_adapter.get_tracking_imports(1))

    def test_pg_save_tracking_import_no_pool_raises(self, pg_adapter):
        from kawkab.services.storage_errors import StorageNotInitializedError

        # Converted tracking cluster: a silent 0 here made a failed import
        # provenance row indistinguishable from "saved".
        with pytest.raises(StorageNotInitializedError):
            asyncio.run(pg_adapter.save_tracking_import(1, "skillcorner"))

    def test_pg_get_tracking_import_by_id_no_pool_raises(self, pg_adapter):
        from kawkab.services.storage_errors import StorageNotInitializedError

        with pytest.raises(StorageNotInitializedError):
            asyncio.run(pg_adapter.get_tracking_import_by_id(1))

    def test_pg_delete_tracking_import_no_pool_raises(self, pg_adapter):
        from kawkab.services.storage_errors import StorageNotInitializedError

        with pytest.raises(StorageNotInitializedError):
            asyncio.run(pg_adapter.delete_tracking_import(1))

    def test_pg_event_frame_links_no_pool(self, pg_adapter):
        from kawkab.services.storage_errors import StorageNotInitializedError

        # Reads converted in the read batch — both directions honest now.
        with pytest.raises(StorageNotInitializedError):
            asyncio.run(pg_adapter.save_event_frame_links_bulk(1, [{"event_id": 1}]))
        with pytest.raises(StorageNotInitializedError):
            asyncio.run(pg_adapter.get_event_frame_links(1))


# ── Round-trip parity: the same operations against the real SQLite DB ──────


class TestTrackingImportRoundTrip:
    def test_tracking_import_full_round_trip(self, sqlite_storage):
        """save_match -> save_tracking_import -> get_tracking_imports ->
        delete, exercising migration 030's real tables."""
        match_id = asyncio.run(
            sqlite_storage.save_match(
                "SkillCorner Fixture", "", home_team="Alpha", away_team="Beta"
            )
        )
        assert match_id > 0

        import_row = asyncio.run(
            sqlite_storage.save_tracking_import(
                match_id,
                "skillcorner",
                source_path="/tmp/fake.json",
                checksum="deadbeef",
                fps=25.0,
                frame_count=100,
                pitch_length_m=105.0,
                pitch_width_m=68.0,
                metadata={"periods": [1, 2]},
            )
        )
        assert import_row > 0

        rows = asyncio.run(sqlite_storage.get_tracking_imports(match_id))
        assert len(rows) == 1
        row = rows[0]
        assert row["vendor"] == "skillcorner"
        assert row["checksum"] == "deadbeef"
        assert row["fps"] == pytest.approx(25.0)
        assert row["frame_count"] == 100
        assert row["metadata"] == {"periods": [1, 2]}

        by_id = asyncio.run(sqlite_storage.get_tracking_import_by_id(import_row))
        assert by_id is not None and by_id["id"] == import_row

        assert asyncio.run(sqlite_storage.delete_tracking_import(import_row)) is True
        assert asyncio.run(sqlite_storage.get_tracking_imports(match_id)) == []

    def test_event_frame_links_round_trip_with_dedup(self, sqlite_storage):
        match_id = asyncio.run(sqlite_storage.save_match("L", ""))
        # events table needs a real event row (FK)
        event_id = asyncio.run(
            sqlite_storage.save_event(
                match_id,
                {
                    "type": "pass",
                    "timestamp": 1.0,
                    "team": "home",
                },
            )
        )
        assert event_id > 0

        links = [
            {"event_id": event_id, "frame_number": 5, "frame_offset": 0},
            {"event_id": event_id, "frame_number": 6, "frame_offset": 1},
            {"event_id": event_id, "frame_number": 5, "frame_offset": 0},
        ]  # dup
        written = asyncio.run(sqlite_storage.save_event_frame_links_bulk(match_id, links))
        assert written == 3  # executemany row count; the UNIQUE dup is ignored

        rows = asyncio.run(sqlite_storage.get_event_frame_links(match_id))
        assert len(rows) == 2
        assert {r["frame_number"] for r in rows} == {5, 6}

        single = asyncio.run(sqlite_storage.get_event_frame_links(match_id, event_id=event_id))
        assert len(single) == 2

    def test_vendor_tracking_frames_round_trip(self, sqlite_storage):
        """Vendor frames persist through the SAME plumbing the video
        pipeline uses, and read back with parsed detection lists."""
        match_id = asyncio.run(sqlite_storage.save_match("T", ""))
        frames = [
            {
                "frame_number": i,
                "timestamp": i / 25.0,
                "player_detections": [
                    {"track_id": 1, "x": 10.0 + i, "y": 20.0, "speed": 1.5},
                    {"track_id": 2, "x": 30.0, "y": 40.0 + i, "speed": 2.0},
                ],
                "ball_detections": [{"track_id": 999, "x": 50.0, "y": 34.0, "z": 0.1}],
            }
            for i in range(5)
        ]
        saved = asyncio.run(sqlite_storage.save_tracking_frames_bulk(match_id, frames))
        assert saved == 5

        got = asyncio.run(sqlite_storage.get_tracking_frames(match_id, limit=100))
        assert len(got) == 5
        assert got[0]["player_detections"][0]["x"] == pytest.approx(10.0)
        assert got[4]["ball_detections"][0]["y"] == pytest.approx(34.0)
        assert asyncio.run(sqlite_storage.get_tracking_frame_count(match_id)) == 5


# ── Postgres-mode initialize contract ────────────────────────────────────────
# StorageService.initialize() in Postgres mode used to early-return WITHOUT
# calling the adapter's initialize(), so the pool was never created and every
# delegated call hit the no-pool fallback: Postgres deployments silently
# returned empty results. These tests pin the fixed contract.


class TestPostgresModeInitializeContract:
    """StorageService(dsn) + initialize() must actually bring the pool up."""

    @pytest.mark.asyncio
    async def test_pg_mode_initialize_calls_adapter_initialize(self):
        """The regression: initialize() in Postgres mode must delegate."""
        from kawkab.services.storage_service import StorageService

        svc = StorageService(dsn="postgresql://user:pass@localhost:5432/kawkab_test")
        assert svc._use_postgres is True
        assert svc._pg is not None

        calls = []

        async def fake_init():
            calls.append(True)

        svc._pg.initialize = fake_init
        await svc.initialize()
        assert calls, "StorageService.initialize() must call _pg.initialize() in PG mode"

    @pytest.mark.asyncio
    async def test_pg_adapter_initialize_is_idempotent(self):
        """Calling initialize() twice must not create two pools."""

        from kawkab.services import postgres_storage

        adapter = postgres_storage.PostgresStorageAdapter(
            dsn="postgresql://user:pass@localhost:5432/kawkab_test"
        )
        created = []

        class _FakePool:
            async def close(self):
                created.append("closed")

        async def fake_create_pool(dsn, min_size=2, max_size=10):
            created.append(dsn)
            return _FakePool()

        import asyncpg  # noqa: F401 -- proves the dep import path works

        orig_create = None
        try:
            import asyncpg as _asyncpg

            orig_create = _asyncpg.create_pool
            _asyncpg.create_pool = fake_create_pool
            await adapter.initialize()
            await adapter.initialize()  # second call: must early-return
        finally:
            if orig_create is not None:
                import asyncpg as _asyncpg

                _asyncpg.create_pool = orig_create
        assert created.count("postgresql://user:pass@localhost:5432/kawkab_test") == 1, (
            "initialize() must be idempotent — one pool per adapter lifetime"
        )

    @pytest.mark.asyncio
    async def test_pg_adapter_initialize_without_dsn_stays_unavailable(self):
        from kawkab.services import postgres_storage

        adapter = postgres_storage.PostgresStorageAdapter(dsn=None)
        # Ensure no ambient KAWKAB_DB_URL leaks into the unit test
        import os

        orig = os.environ.pop("KAWKAB_DB_URL", None)
        try:
            await adapter.initialize()
        finally:
            if orig is not None:
                os.environ["KAWKAB_DB_URL"] = orig
        assert adapter._pool is None
        assert adapter._available is False

    def test_sqlite_mode_dsn_omitted_uses_sqlite_path(self):
        from kawkab.services.storage_service import StorageService

        svc = StorageService()
        assert svc._use_postgres is False
        assert svc._db_path is not None


# ── Contract-storage parity (migration 017) ─────────────────────────────────


class TestContractStorageParity:
    """Contracts storage exists on BOTH backends (migration 017 table +
    the Settings contracts panel read path)."""

    def test_contract_methods_on_both_backends(self, sqlite_storage, pg_adapter):
        for method in ("save_contract", "get_contracts", "get_contracts_expiring_soon"):
            assert hasattr(sqlite_storage, method), f"SQLite missing {method}"
            assert hasattr(pg_adapter, method), f"Postgres missing {method}"

    @pytest.mark.asyncio
    async def test_contract_round_trip_on_real_migrated_sqlite(self, sqlite_storage):
        from datetime import date, timedelta

        cid = await sqlite_storage.save_contract(
            {
                "player_profile_id": 1,
                "player_name": "Parity Pro",
                "contract_type": "loan",
                "start_date": "2025-07-01",
                "end_date": (date.today() + timedelta(days=45)).isoformat(),
                "wage_weekly_pounds": 40.0,
            }
        )
        assert cid > 0
        contracts = await sqlite_storage.get_contracts()
        assert any(
            c["player_name"] == "Parity Pro" and c["contract_type"] == "loan" for c in contracts
        )
        expiring = await sqlite_storage.get_contracts_expiring_soon(90)
        assert any(c["player_name"] == "Parity Pro" for c in expiring)

    def test_pg_contract_methods_fail_soft_without_pool(self, pg_adapter):
        """No live Postgres in unit CI: delegated calls must fail soft."""
        assert asyncio.iscoroutinefunction(pg_adapter.get_contracts)
        assert asyncio.iscoroutinefunction(pg_adapter.save_contract)
        assert asyncio.iscoroutinefunction(pg_adapter.get_contracts_expiring_soon)
