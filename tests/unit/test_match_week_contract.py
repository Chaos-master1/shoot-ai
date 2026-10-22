"""Seam tests for the Match Week loop.

These cross the boundary that used to be mocked away:

    generate_training_plan -> training_plans row -> measure_intervention

Before this file existed, the efficacy tests seeded a *hand-written*
plan payload containing ``based_on_match_ids`` and ``priority_diagnoses``.
The real generator writes the diagnoses to a column and never puts them
in the payload, so every generated plan fell into the "plan lacks ..."
error path and no test noticed.

The fixture uses real migrated SQLite, the real knowledge base, the real
reasoning engine and the real handler.
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

# The defensive-third turnover rule fires at >= 6 turnovers per match.
# Before-window: 7, 6, 8 (all fire). After-window: 2, 3 (improved).
_BEFORE = {1: 7, 2: 6, 3: 8}
_AFTER = {4: 2, 5: 3}


@pytest.fixture()
def storage(tmp_path):
    from kawkab.core.migration_manager import MigrationManager
    from kawkab.services.storage_service import StorageService

    db = tmp_path / "match_week.db"
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


def _add_event(storage, mid: int, event_type: str, timestamp: float, **meta) -> None:
    storage._conn.execute(
        """INSERT INTO events (match_id, timestamp, event_type, team, completed, metadata)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (mid, timestamp, event_type, "home", int(meta.get("completed", 0)), json.dumps(meta)),
    )


def _seed(storage, after: int = 2) -> None:
    for mid in (1, 2, 3, 4, 5):
        _add_match(storage, mid)
    for mid, count in _BEFORE.items():
        for k in range(count):
            _add_event(storage, mid, "turnover", k * 60.0, zone="defensive_third")
        for k in range(10):
            _add_event(storage, mid, "pass", 600.0 + k, zone="defensive_third", completed=0)
    for mid in list(_AFTER)[:after]:
        for k in range(_AFTER[mid]):
            _add_event(storage, mid, "turnover", k * 60.0, zone="defensive_third")
        for k in range(10):
            _add_event(storage, mid, "pass", 600.0 + k, zone="defensive_third", completed=0)
    storage._conn.commit()


async def _generate_plan(storage, match_id: int = 3) -> dict:
    from kawkab.ui.bridge_handlers.bridge_analysis import AnalysisHandler

    handler = AnalysisHandler(None, {"storage_service": storage})
    raw = await handler.generate_training_plan(str(match_id))
    return json.loads(raw)


@pytest.mark.asyncio
async def test_generated_plan_is_measurable(storage):
    """The whole loop: real generator -> persisted row -> efficacy."""
    _seed(storage)
    generated = await _generate_plan(storage)
    assert generated["success"] is True
    assert generated["diagnosis_count"] >= 1
    assert generated["persisted"] is True

    from kawkab.services.intervention_efficacy_service import InterventionEfficacyService

    report = await InterventionEfficacyService(storage, await _kb()).measure_intervention(
        generated["plan_id"]
    )

    assert "error" not in report, report.get("error")
    assert report["based_on_match_ids"] == [3]
    assert report["before_window"] == [1, 2]
    assert report["after_window"] == [4, 5]
    assert report["after_window_complete"] is True

    rule = next(r for r in report["rules"] if r["rule_id"] == "defensive_third_errors_v1")
    assert rule["state"] == "measured"
    assert rule["mean_before"] == pytest.approx(6.5)  # 7 and 6
    assert rule["mean_after"] == pytest.approx(2.5)  # 2 and 3
    assert rule["delta"] == pytest.approx(-4.0)
    assert rule["improved"] is True


@pytest.mark.asyncio
async def test_observation_equals_the_diagnosis_criterion(storage):
    """The measured quantity is the one the diagnosis used, not a proxy."""
    from kawkab.services.reasoning.inputs import (
        build_match_analysis,
        normalize_storage_events,
    )
    from kawkab.services.reasoning.metrics import extract_metric
    from kawkab.services.reasoning_service import ReasoningService

    _seed(storage)
    kb = await _kb()
    reasoner = ReasoningService(kb)

    rows = await storage.get_match_events(3, limit=5000)
    events = normalize_storage_events(rows)
    analysis = build_match_analysis(3, await storage.get_match(3) or {}, events)

    report = await reasoner.diagnose_match(analysis, events=events)
    diagnosis = next(d for d in report.diagnoses if d.rule_id == "defensive_third_errors_v1")
    criterion = next(
        c
        for h in diagnosis.hypothesis_evaluation
        for c in h["criteria"]
        if c["metric"] == "turnovers_in_defensive_third"
    )

    observation = await reasoner.observe_rule("defensive_third_errors_v1", analysis, events)
    resolved = extract_metric(
        "turnovers_in_defensive_third", observation["evidence"], observation["event_stats"]
    )

    assert criterion["observed"] == resolved == 8.0


@pytest.mark.asyncio
async def test_no_after_events_is_insufficient_not_zero(storage):
    """A missing observation must never be reported as a zero delta."""
    _seed(storage, after=2)
    storage._conn.execute("DELETE FROM events WHERE match_id = 5")
    storage._conn.commit()

    generated = await _generate_plan(storage)
    from kawkab.services.intervention_efficacy_service import InterventionEfficacyService

    report = await InterventionEfficacyService(storage, await _kb()).measure_intervention(
        generated["plan_id"]
    )
    rule = next(r for r in report["rules"] if r["rule_id"] == "defensive_third_errors_v1")

    assert rule["state"] == "insufficient_sample"
    assert rule["after"][1]["value"] is None
    assert rule["after"][1]["reason"] == "match has no stored events"
    assert "delta" not in rule


@pytest.mark.asyncio
async def test_plan_without_diagnoses_returns_an_explicit_error(storage):
    _seed(storage)
    plan_id = await storage.save_training_plan(
        match_id=3,
        title="Empty",
        payload={"based_on_match_ids": [3]},
        priority_diagnoses=[],
    )

    from kawkab.services.intervention_efficacy_service import InterventionEfficacyService

    report = await InterventionEfficacyService(storage, await _kb()).measure_intervention(plan_id)

    assert "error" in report
    assert "priority_diagnoses" in report["error"]
