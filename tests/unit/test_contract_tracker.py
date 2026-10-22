"""Tests for Contract Tracker Service."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()

ContractTracker = load_service_module(
    "kawkab.services.contract_tracker", "contract_tracker.py"
).ContractTracker


@pytest.fixture
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS player_contracts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_profile_id INTEGER NOT NULL,
            player_name TEXT NOT NULL,
            contract_type TEXT NOT NULL DEFAULT 'permanent',
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            club_option_years INTEGER DEFAULT 0,
            player_option_years INTEGER DEFAULT 0,
            release_clause_millions REAL DEFAULT NULL,
            wage_weekly_pounds REAL DEFAULT NULL,
            agent_name TEXT DEFAULT '',
            notes TEXT DEFAULT '',
            last_updated TEXT NOT NULL DEFAULT '2025-01-01'
        );
        CREATE TABLE IF NOT EXISTS player_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            global_id TEXT UNIQUE NOT NULL,
            display_name TEXT
        );
    """)
    return conn


@pytest.fixture
def svc(db: sqlite3.Connection) -> ContractTracker:
    s = ContractTracker()
    s.set_connection(db)
    return s


class TestAddContract:
    def test_add_creates_record(self, svc: ContractTracker) -> None:
        cid = svc.add_contract(1, "Player One", "permanent", "2024-01-01", "2027-06-30")
        assert cid > 0

    def test_add_with_all_fields(self, svc: ContractTracker) -> None:
        cid = svc.add_contract(
            1,
            "Player One",
            "permanent",
            "2024-01-01",
            "2027-06-30",
            club_option_years=1,
            release_clause_millions=50.0,
            wage_weekly_pounds=100000,
            agent_name="John Doe",
        )
        assert cid > 0

    def test_empty_storage_returns_zero(self) -> None:
        t = ContractTracker()
        assert t.add_contract(1, "P1", "permanent", "2024-01-01", "2027-06-30") == 0
        assert t.get_expiring_contracts() == []
        assert t.get_expired_contracts() == []
        summary = t.get_squad_contract_summary()
        assert summary["total_contracts"] == 0
        assert t.get_contract_alerts() == []


class TestExpiredExpiring:
    def test_non_expiring_not_returned(self, svc: ContractTracker) -> None:
        svc.add_contract(1, "Player A", "permanent", "2020-01-01", "2029-12-31")
        exp = svc.get_expiring_contracts(within_months=6)
        assert len(exp) == 0

    def test_expired_contracts_found(self, svc: ContractTracker) -> None:
        svc.add_contract(1, "Player A", "permanent", "2020-01-01", "2020-06-30")
        exp = svc.get_expired_contracts()
        assert len(exp) >= 1


class TestContractSummary:
    def test_squad_summary_correct(self, svc: ContractTracker) -> None:
        svc.add_contract(1, "P1", "permanent", "2024-01-01", "2027-06-30", wage_weekly_pounds=50000)
        svc.add_contract(2, "P2", "loan", "2024-07-01", "2025-06-30", wage_weekly_pounds=30000)
        s = svc.get_squad_contract_summary()
        assert s["total_contracts"] == 2
        assert s["by_type"]["permanent"] == 1
        assert s["by_type"]["loan"] == 1
        assert s["total_wage_bill"] == 80000


class TestContractAlerts:
    def test_player_option_alerts(self, svc: ContractTracker) -> None:
        svc.add_contract(
            1, "With Option", "permanent", "2024-01-01", "2027-06-30", player_option_years=2
        )
        alerts = svc.get_contract_alerts()
        opts = [a for a in alerts if a["type"] == "player_option"]
        assert len(opts) >= 1


class TestUpdateContract:
    def test_update_works(self, svc: ContractTracker) -> None:
        cid = svc.add_contract(1, "P1", "permanent", "2024-01-01", "2027-06-30")
        assert svc.update_contract(cid, wage_weekly_pounds=200000) is True

    def test_update_invalid_field_returns_false(self, svc: ContractTracker) -> None:
        cid = svc.add_contract(1, "P1", "permanent", "2024-01-01", "2027-06-30")
        assert svc.update_contract(cid, invalid_field=123) is False
