from __future__ import annotations

import sqlite3
from datetime import datetime

import pytest

from kawkab.services.concussion_protocol import (
    STAGE_DESCRIPTIONS,
    ConcussionProtocolService,
    SCAT5Assessment,
)
from kawkab.services.injury_tracker import (
    InjuryRecord,
    InjuryTrackerService,
)
from kawkab.services.rehab_service import RehabService


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS injuries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            match_id INTEGER,
            injury_type TEXT NOT NULL,
            body_part TEXT NOT NULL,
            severity TEXT NOT NULL DEFAULT 'minor',
            mechanism TEXT DEFAULT '',
            date_injured TEXT NOT NULL,
            date_recovered TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            notes TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS rehab_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            injury_id INTEGER NOT NULL REFERENCES injuries(id),
            phase TEXT NOT NULL DEFAULT 'initial',
            start_date TEXT NOT NULL,
            target_end_date TEXT,
            actual_end_date TEXT,
            milestones TEXT DEFAULT '[]',
            protocols TEXT DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active',
            notes TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS concussion_assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            match_id INTEGER,
            assessment_date TEXT DEFAULT (datetime('now')),
            assessment_type TEXT NOT NULL DEFAULT 'scat5',
            symptoms_score INTEGER DEFAULT 0,
            cognitive_score INTEGER DEFAULT 0,
            balance_score INTEGER DEFAULT 0,
            clearance_status TEXT DEFAULT 'not_cleared',
            cleared_by TEXT DEFAULT '',
            notes TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now'))
        );
    """)
    yield conn
    conn.close()


class TestInjuryTracker:
    def test_record_injury(self, db):
        svc = InjuryTrackerService(db)
        rec = InjuryRecord(
            player_id=1, injury_type="Hamstring Strain", body_part="thigh", severity="moderate"
        )
        injury_id = svc.record_injury(rec)
        assert injury_id > 0

    def test_get_player_injuries(self, db):
        svc = InjuryTrackerService(db)
        svc.record_injury(
            InjuryRecord(player_id=1, injury_type="ACL", body_part="knee", severity="severe")
        )
        svc.record_injury(
            InjuryRecord(
                player_id=1, injury_type="Ankle Sprain", body_part="ankle", severity="moderate"
            )
        )
        injuries = svc.get_player_injuries(1)
        assert len(injuries) == 2

    def test_get_player_injuries_empty(self, db):
        svc = InjuryTrackerService(db)
        assert svc.get_player_injuries(99) == []

    def test_get_active_injuries(self, db):
        svc = InjuryTrackerService(db)
        svc.record_injury(
            InjuryRecord(player_id=1, injury_type="A", body_part="knee", status="active")
        )
        svc.record_injury(
            InjuryRecord(player_id=2, injury_type="B", body_part="thigh", status="recovered")
        )
        active = svc.get_active_injuries([1, 2])
        assert len(active) == 1
        assert active[0]["injury_type"] == "A"

    def test_update_injury_status(self, db):
        svc = InjuryTrackerService(db)
        iid = svc.record_injury(InjuryRecord(player_id=1, injury_type="Sprain", body_part="ankle"))
        svc.update_injury_status(iid, "recovered")
        injuries = svc.get_player_injuries(1)
        assert injuries[0]["status"] == "recovered"
        assert injuries[0]["date_recovered"] != ""

    def test_squad_injury_report(self, db):
        svc = InjuryTrackerService(db)
        svc.record_injury(
            InjuryRecord(player_id=1, injury_type="A", body_part="knee", severity="severe")
        )
        svc.record_injury(
            InjuryRecord(player_id=2, injury_type="B", body_part="thigh", severity="moderate")
        )
        report = svc.get_squad_injury_report([1, 2])
        assert report["total_active"] == 2
        assert report["by_severity"]["severe"] == 1
        assert report["by_severity"]["moderate"] == 1

    def test_get_injury_stats(self, db):
        svc = InjuryTrackerService(db)
        svc.record_injury(InjuryRecord(player_id=1, injury_type="Hamstring", body_part="thigh"))
        svc.record_injury(InjuryRecord(player_id=1, injury_type="Hamstring", body_part="thigh"))
        svc.record_injury(InjuryRecord(player_id=1, injury_type="Ankle", body_part="ankle"))
        stats = svc.get_injury_stats(1)
        assert stats["total_injuries"] == 3
        assert stats["by_type"]["Hamstring"] == 2

    def test_injury_record_days_since_injury(self):
        rec = InjuryRecord(
            player_id=1,
            injury_type="Test",
            body_part="knee",
            date_injured=datetime.now().isoformat(),
        )
        assert rec.days_since_injury() == 0

    def test_injury_record_estimated_recovery_days(self):
        rec = InjuryRecord(player_id=1, injury_type="ACL", body_part="knee", severity="severe")
        assert rec.estimated_recovery_days() == 180


class TestRehabService:
    def test_create_plan(self, db):
        svc = InjuryTrackerService(db)
        rehab = RehabService(db)
        iid = svc.record_injury(InjuryRecord(player_id=1, injury_type="ACL", body_part="knee"))
        plan = rehab.create_plan(iid)
        assert plan.id > 0
        assert plan.phase == "initial"

    def test_get_plan(self, db):
        svc = InjuryTrackerService(db)
        rehab = RehabService(db)
        iid = svc.record_injury(InjuryRecord(player_id=1, injury_type="ACL", body_part="knee"))
        plan = rehab.create_plan(iid)
        fetched = rehab.get_plan(plan.id)
        assert fetched is not None
        assert fetched["phase"] == "initial"

    def test_get_plan_not_found(self, db):
        rehab = RehabService(db)
        assert rehab.get_plan(999) is None

    def test_get_plan_by_injury(self, db):
        svc = InjuryTrackerService(db)
        rehab = RehabService(db)
        iid = svc.record_injury(InjuryRecord(player_id=1, injury_type="ACL", body_part="knee"))
        rehab.create_plan(iid)
        fetched = rehab.get_plan_by_injury(iid)
        assert fetched is not None
        assert fetched["injury_id"] == iid

    def test_advance_phase(self, db):
        svc = InjuryTrackerService(db)
        rehab = RehabService(db)
        iid = svc.record_injury(InjuryRecord(player_id=1, injury_type="ACL", body_part="knee"))
        plan = rehab.create_plan(iid)
        advanced = rehab.advance_phase(plan.id)
        assert advanced["phase"] == "protective"
        advanced2 = rehab.advance_phase(plan.id)
        assert advanced2["phase"] == "restorative"

    def test_complete_milestone(self, db):
        svc = InjuryTrackerService(db)
        rehab = RehabService(db)
        iid = svc.record_injury(InjuryRecord(player_id=1, injury_type="ACL", body_part="knee"))
        plan = rehab.create_plan(iid)
        ms_before = len(plan.milestones)
        rehab.complete_milestone(plan.id, plan.milestones[0])
        updated = rehab.get_plan(plan.id)
        assert len(updated["milestones"]) == ms_before - 1

    def test_close_plan(self, db):
        svc = InjuryTrackerService(db)
        rehab = RehabService(db)
        iid = svc.record_injury(InjuryRecord(player_id=1, injury_type="ACL", body_part="knee"))
        plan = rehab.create_plan(iid)
        rehab.close_plan(plan.id)
        fetched = rehab.get_plan(plan.id)
        assert fetched["status"] == "completed"

    def test_get_active_plans(self, db):
        svc = InjuryTrackerService(db)
        rehab = RehabService(db)
        iid = svc.record_injury(InjuryRecord(player_id=1, injury_type="ACL", body_part="knee"))
        rehab.create_plan(iid)
        plans = rehab.get_active_plans(1)
        assert len(plans) == 1

    def test_advance_phase_beyond_last(self, db):
        svc = InjuryTrackerService(db)
        rehab = RehabService(db)
        iid = svc.record_injury(InjuryRecord(player_id=1, injury_type="ACL", body_part="knee"))
        plan = rehab.create_plan(iid)
        for _ in range(6):
            rehab.advance_phase(plan.id)
        plan_data = rehab.get_plan(plan.id)
        assert plan_data["phase"] == "maintenance"


class TestConcussionProtocol:
    def test_record_assessment(self, db):
        svc = ConcussionProtocolService(db)
        assessment = SCAT5Assessment(
            player_id=1, symptoms_score=5, cognitive_score=2, balance_score=1
        )
        aid = svc.record_assessment(assessment)
        assert aid > 0

    def test_get_assessments(self, db):
        svc = ConcussionProtocolService(db)
        svc.record_assessment(SCAT5Assessment(player_id=1, symptoms_score=3))
        svc.record_assessment(SCAT5Assessment(player_id=1, symptoms_score=1))
        assessments = svc.get_assessments(1)
        assert len(assessments) == 2

    def test_advance_clearance(self, db):
        svc = ConcussionProtocolService(db)
        aid = svc.record_assessment(SCAT5Assessment(player_id=1))
        advanced = svc.advance_clearance(aid, "Dr. Smith")
        assert advanced["clearance_status"] == "stage_1"

    def test_advance_through_all_stages(self, db):
        svc = ConcussionProtocolService(db)
        aid = svc.record_assessment(SCAT5Assessment(player_id=1))
        for _ in range(6):
            svc.advance_clearance(aid, "Dr. Smith")
        status = svc.get_clearance_status(1)
        assert status["clearance_status"] == "full_cleared"

    def test_clearance_status_no_assessment(self, db):
        svc = ConcussionProtocolService(db)
        status = svc.get_clearance_status(99)
        assert status["has_assessment"] is False
        assert status["clearance_status"] == "not_cleared"

    def test_return_to_play_readiness_cleared(self, db):
        svc = ConcussionProtocolService(db)
        aid = svc.record_assessment(SCAT5Assessment(player_id=1))
        for _ in range(6):
            svc.advance_clearance(aid, "Dr. Smith")
        readiness = svc.check_return_to_play_readiness(1)
        assert readiness["ready"] is True

    def test_return_to_play_readiness_not_assessed(self, db):
        svc = ConcussionProtocolService(db)
        readiness = svc.check_return_to_play_readiness(99)
        assert readiness["ready"] is False

    def test_get_stage_protocol(self, db):
        svc = ConcussionProtocolService(db)
        stage = svc.get_stage_protocol("stage_1")
        assert stage["description"] == STAGE_DESCRIPTIONS["stage_1"]

    def test_scat5_total_score(self):
        a = SCAT5Assessment(player_id=1, symptoms_score=5, cognitive_score=3, balance_score=2)
        assert a.total_score() == 10

    def test_scat5_is_symptomatic(self):
        a = SCAT5Assessment(player_id=1, symptoms_score=5)
        assert a.is_symptomatic() is True
        a2 = SCAT5Assessment(player_id=1, symptoms_score=0)
        assert a2.is_symptomatic() is False
