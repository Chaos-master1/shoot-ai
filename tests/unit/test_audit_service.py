"""Tests for AuditService — structured audit logging."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()
_mod = load_service_module("audit_test", "audit_service.py")
AuditService = _mod.AuditService


@pytest.fixture
def db_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            action TEXT NOT NULL,
            entity_type TEXT NOT NULL DEFAULT '',
            entity_id TEXT DEFAULT NULL,
            details_json TEXT DEFAULT '{}',
            user TEXT NOT NULL DEFAULT 'local'
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_audit_events_action
        ON audit_events(action)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_audit_events_type
        ON audit_events(entity_type)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp
        ON audit_events(timestamp)
    """)
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def storage_mock(tmp_path):
    """A real StorageService on a migrated scratch DB — AuditService now
    goes through the typed audit_log/get_audit_log methods (which also
    work on Postgres), not raw _conn access."""
    from pathlib import Path

    from kawkab.core.migration_manager import MigrationManager
    from kawkab.services.storage_service import StorageService

    svc = StorageService()
    svc._use_postgres = False
    svc._pg = None
    svc._db_path = tmp_path / "audit_test.db"
    svc._conn = sqlite3.connect(str(svc._db_path))
    svc._conn.row_factory = sqlite3.Row
    MigrationManager(svc._db_path, Path("src/kawkab/migrations")).migrate()
    return svc


@pytest.fixture
def audit_service(storage_mock):
    return AuditService(storage_service=storage_mock)


class TestAuditService:
    def test_log_event_creates_record(self, audit_service, storage_mock):
        row_id = audit_service.log_event(
            "analysis.completed", "match", "42", {"xg": 1.5, "shots": 10}, "coach@example.com"
        )
        assert row_id > 0
        cursor = storage_mock._conn.cursor()
        cursor.execute("SELECT * FROM audit_events_local WHERE id = ?", (row_id,))
        row = dict(cursor.fetchone())
        assert row["action"] == "analysis.completed"
        assert row["resource_type"] == "match"
        assert row["resource_id"] == "42"
        assert json.loads(row["details"])["xg"] == 1.5
        assert row["username"] == "coach@example.com"

    def test_get_events_returns_all(self, audit_service):
        for i in range(3):
            audit_service.log_event("event.created", "event", str(i))
        events = audit_service.get_events()
        assert len(events) == 3

    def test_get_events_filter_by_action(self, audit_service):
        audit_service.log_event("analysis.started", "match", "1")
        audit_service.log_event("analysis.completed", "match", "1")
        audit_service.log_event("event.created", "event", "2")
        events = audit_service.get_events(action="analysis.started")
        assert len(events) == 1
        assert events[0]["action"] == "analysis.started"

    def test_get_events_filter_by_type(self, audit_service):
        audit_service.log_event("analysis.started", "match", "1")
        audit_service.log_event("export.csv", "match", "1")
        events = audit_service.get_events(entity_type="match")
        assert len(events) == 2

    def test_get_events_pagination(self, audit_service):
        for i in range(5):
            audit_service.log_event("event.created", "event", str(i))
        events = audit_service.get_events(limit=2, offset=0)
        assert len(events) == 2

    def test_empty_db_returns_empty(self, audit_service):
        events = audit_service.get_events()
        assert events == []

    def test_get_stats_computes_correctly(self, audit_service):
        audit_service.log_event("analysis.completed", "match", "1")
        audit_service.log_event("analysis.completed", "match", "2")
        audit_service.log_event("export.csv", "match", "1")
        stats = audit_service.get_stats()
        assert stats["total_events"] == 3
        assert stats["by_action"]["analysis.completed"] == 2
        assert stats["by_action"]["export.csv"] == 1
        assert stats["by_type"]["match"] == 3

    def test_get_stats_empty_db(self, audit_service):
        stats = audit_service.get_stats()
        assert stats["total_events"] == 0
        assert stats["events_last_24h"] == 0
        assert stats["by_action"] == {}
        assert stats["by_type"] == {}

    def test_log_event_no_storage_returns_zero(self):
        svc = AuditService(storage_service=None)
        assert svc.log_event("test.action", "test") == 0

    def test_get_events_no_storage_returns_empty(self):
        svc = AuditService(storage_service=None)
        assert svc.get_events() == []

    def test_get_stats_no_storage_returns_zeros(self):
        svc = AuditService(storage_service=None)
        stats = svc.get_stats()
        assert stats["total_events"] == 0

    def test_log_event_closed_connection_returns_zero(self):
        from kawkab.services.storage_service import StorageService

        svc_raw = StorageService()
        svc_raw._conn = None
        svc = AuditService(storage_service=svc_raw)
        # Not-initialized storage raises honestly; log_event surfaces that
        # as 0 so callers can detect a non-persisted audit event.
        assert svc.log_event("test.action", "test") == 0

    def test_multiple_actions_appear_in_stats(self, audit_service):
        actions = [
            ("analysis.started", "match"),
            ("analysis.completed", "match"),
            ("analysis.failed", "match"),
            ("export.csv", "match"),
            ("export.json", "match"),
            ("feedback.submitted", "feedback"),
            ("config.changed", "config"),
        ]
        for action, etype in actions:
            audit_service.log_event(action, etype)
        stats = audit_service.get_stats()
        assert stats["total_events"] == 7
        for action, _ in actions:
            assert stats["by_action"].get(action) == 1
