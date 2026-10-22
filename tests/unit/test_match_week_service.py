"""MatchWeekService: the loop's stage vocabulary over persisted facts.

No delta math is tested here -- that belongs to InterventionEfficacyService
(tests/unit/test_match_week_contract.py). This file pins the mapping from
persisted state to the stage the UI renders, including the honest states.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import install_kawkab_stubs

install_kawkab_stubs()

_REAL_KB_ROOT = (Path(__file__).resolve().parents[2] / "src" / "kawkab" / "knowledge").resolve()


@pytest.fixture()
def storage(tmp_path):
    from kawkab.core.migration_manager import MigrationManager
    from kawkab.services.storage_service import StorageService

    db = tmp_path / "week.db"
    MigrationManager(db, Path("src/kawkab/migrations")).migrate()
    svc = StorageService.__new__(StorageService)
    svc._conn = sqlite3.connect(str(db))
    svc._conn.row_factory = sqlite3.Row
    svc._use_postgres = False
    svc._db_path = db
    yield svc
    svc._conn.close()


@pytest.fixture(autouse=True)
def _real_kb(monkeypatch):
    import kawkab.services.knowledge_service as ks_mod

    class _RealPaths:
        knowledge_base = _REAL_KB_ROOT

    monkeypatch.setattr(ks_mod, "get_paths", lambda: _RealPaths())


async def _kb():
    from kawkab.services.knowledge_service import KnowledgeService

    kb = KnowledgeService()
    await kb.initialize()
    return kb


def _add_match(storage, mid: int) -> None:
    storage._conn.execute(
        """INSERT INTO matches (id, name, video_path, home_team, away_team, match_date, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            mid,
            f"M{mid}",
            f"/tmp/m{mid}.mp4",
            "Kawkab",
            f"Opp{mid}",
            f"2026-08-{10 + mid}",
            "2026-08-15",
        ),
    )


def _add_events(storage, mid: int, turnovers: int) -> None:
    for k in range(turnovers):
        storage._conn.execute(
            """INSERT INTO events (match_id, timestamp, event_type, team, completed, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (mid, k * 60.0, "turnover", "home", 0, json.dumps({"zone": "defensive_third"})),
        )
    for k in range(10):
        storage._conn.execute(
            """INSERT INTO events (match_id, timestamp, event_type, team, completed, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (mid, 600.0 + k, "pass", "home", 0, json.dumps({"zone": "defensive_third"})),
        )
    storage._conn.commit()


async def _seed_plan(storage) -> int:
    for mid, n in [(1, 7), (2, 6), (3, 8)]:
        _add_match(storage, mid)
        _add_events(storage, mid, n)
    return await storage.save_training_plan(
        match_id=3,
        title="Plan",
        payload={"based_on_match_ids": [3], "weeks": [], "priority_addressed": ["errors"]},
        priority_diagnoses=[
            {"rule_id": "defensive_third_errors_v1", "rule_name": "errors", "confidence": 0.8}
        ],
    )


@pytest.mark.asyncio
async def test_no_events_is_no_analysis(storage):
    from kawkab.services.match_week_service import MatchWeekService

    _add_match(storage, 1)
    state = await MatchWeekService(storage, await _kb()).state(1)
    assert state.stage == "no_analysis"
    assert "no analysis events" in state.note


@pytest.mark.asyncio
async def test_events_without_plan_is_no_plan(storage):
    from kawkab.services.match_week_service import MatchWeekService

    _add_match(storage, 1)
    _add_events(storage, 1, 7)
    state = await MatchWeekService(storage, await _kb()).state(1)
    assert state.stage == "no_plan"
    assert state.plan is None
    assert state.provenance["calibrated"] is None  # no homography service injected


@pytest.mark.asyncio
async def test_plan_with_empty_after_window_awaits_matches(storage):
    from kawkab.services.match_week_service import MatchWeekService

    await _seed_plan(storage)
    state = await MatchWeekService(storage, await _kb()).state(3)

    assert state.stage == "awaiting_matches"
    assert "0/2" in state.note
    assert state.plan["id"] > 0
    assert len(state.diagnoses) == 1
    assert state.provenance["based_on_match_ids"] == [3]
    assert state.provenance["after_window_complete"] is False


@pytest.mark.asyncio
async def test_complete_window_is_measured(storage):
    from kawkab.services.match_week_service import MatchWeekService

    await _seed_plan(storage)
    for mid, n in [(4, 2), (5, 3)]:
        _add_match(storage, mid)
        _add_events(storage, mid, n)

    state = await MatchWeekService(storage, await _kb()).state(3)

    assert state.stage == "measured"
    assert "1 of 1" in state.note
    rule = state.recheck["rules"][0]
    assert rule["state"] == "measured"
    assert rule["improved"] is True


@pytest.mark.asyncio
async def test_slot_returns_the_state_as_json(storage):
    from kawkab.ui.bridge_handlers.bridge_training import TrainingHandler

    _add_match(storage, 1)
    _add_events(storage, 1, 7)
    handler = TrainingHandler(
        None,
        {"storage_service": storage, "knowledge_service": await _kb()},
    )
    payload = json.loads(await handler.get_match_week_state("1"))

    assert payload["success"] is True
    assert payload["stage"] == "no_plan"
    assert payload["match_id"] == 1
