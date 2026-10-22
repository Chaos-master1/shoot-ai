"""Tests for Shortlist Service."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()

ShortlistService = load_service_module(
    "kawkab.services.shortlist_service", "shortlist_service.py"
).ShortlistService


@pytest.fixture
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS player_shortlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT NOT NULL,
            player_name TEXT NOT NULL,
            position TEXT DEFAULT '',
            team TEXT DEFAULT '',
            league TEXT DEFAULT '',
            added_date TEXT NOT NULL DEFAULT (datetime('now')),
            priority TEXT NOT NULL DEFAULT 'medium' CHECK(priority IN ('low','medium','high','urgent')),
            status TEXT NOT NULL DEFAULT 'scouted' CHECK(status IN ('scouted','shortlisted','contacted','trial','signed','rejected','archived')),
            notes TEXT DEFAULT '',
            scout_rating REAL DEFAULT 0.0 CHECK(scout_rating >= 0 AND scout_rating <= 10),
            estimated_value REAL DEFAULT NULL,
            age INTEGER DEFAULT NULL,
            nationality TEXT DEFAULT '',
            last_updated TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)
    return conn


@pytest.fixture
def svc(db: sqlite3.Connection) -> ShortlistService:  # type: ignore
    s = ShortlistService()
    s.set_connection(db)
    return s


class TestAddPlayer:
    def test_add_creates_entry(self, svc: ShortlistService) -> None:
        eid = svc.add_to_shortlist("p1", "Player One", position="FWD", team="Team A", league="PL")
        assert eid > 0

    def test_duplicate_player_returns_existing(self, svc: ShortlistService) -> None:
        eid1 = svc.add_to_shortlist("p1", "Player One")
        eid2 = svc.add_to_shortlist("p1", "Player One")
        assert eid1 == eid2

    def test_add_with_all_fields(self, svc: ShortlistService) -> None:
        eid = svc.add_to_shortlist(
            "p2",
            "Player Two",
            position="MID",
            team="Team B",
            league="La Liga",
            priority="high",
            notes="Watch list",
            scout_rating=8.5,
            age=24,
            nationality="Spain",
            estimated_value=15.0,
        )
        assert eid > 0
        entry = svc.get_player_on_shortlist("p2")
        assert entry is not None
        assert entry["scout_rating"] == 8.5
        assert entry["age"] == 24
        assert entry["nationality"] == "Spain"

    def test_empty_storage_returns_graceful_defaults(self) -> None:
        s = ShortlistService()
        assert s.add_to_shortlist("p1", "P1") == 0
        assert s.get_shortlist() == []
        assert s.get_shortlist_stats() == {
            "total": 0,
            "by_status": {},
            "by_priority": {},
            "by_position": {},
        }
        assert s.get_player_on_shortlist("p1") is None


class TestUpdateStatus:
    def test_update_status_works(self, svc: ShortlistService) -> None:
        eid = svc.add_to_shortlist("p1", "P1")
        assert svc.update_status(eid, "contacted") is True
        entry = svc.get_player_on_shortlist("p1")
        assert entry["status"] == "contacted"

    def test_invalid_status_returns_false(self, svc: ShortlistService) -> None:
        eid = svc.add_to_shortlist("p1", "P1")
        assert svc.update_status(eid, "invalid") is False

    def test_remove_archives(self, svc: ShortlistService) -> None:
        eid = svc.add_to_shortlist("p1", "P1")
        assert svc.remove_player(eid) is True
        assert svc.get_player_on_shortlist("p1") is None


class TestUpdatePriority:
    def test_update_priority_works(self, svc: ShortlistService) -> None:
        eid = svc.add_to_shortlist("p1", "P1", priority="low")
        assert svc.update_priority(eid, "urgent") is True
        entries = svc.get_shortlist(priority="urgent")
        assert len(entries) == 1

    def test_invalid_priority_returns_false(self, svc: ShortlistService) -> None:
        eid = svc.add_to_shortlist("p1", "P1")
        assert svc.update_priority(eid, "critical") is False


class TestGetShortlist:
    def test_filter_by_status(self, svc: ShortlistService) -> None:
        svc.add_to_shortlist("p1", "P1")
        svc.add_to_shortlist("p2", "P2")
        eid3 = svc.add_to_shortlist("p3", "P3")
        svc.update_status(eid3, "signed")
        entries = svc.get_shortlist(status="signed")
        assert len(entries) == 1
        assert entries[0]["player_id"] == "p3"

    def test_filter_by_position(self, svc: ShortlistService) -> None:
        svc.add_to_shortlist("p1", "P1", position="FWD")
        svc.add_to_shortlist("p2", "P2", position="MID")
        entries = svc.get_shortlist(position="MID")
        assert len(entries) == 1


class TestShortlistStats:
    def test_stats_computed_correctly(self, svc: ShortlistService) -> None:
        svc.add_to_shortlist("p1", "P1", position="FWD", priority="high")
        svc.add_to_shortlist("p2", "P2", position="MID", priority="medium")
        svc.add_to_shortlist("p3", "P3", position="FWD", priority="high")
        stats = svc.get_shortlist_stats()
        assert stats["total"] == 3
        assert stats["by_priority"]["high"] == 2
        assert stats["by_priority"]["medium"] == 1
        assert stats["by_position"]["FWD"] == 2
        assert stats["by_position"]["MID"] == 1
