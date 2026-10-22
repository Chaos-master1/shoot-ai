"""Phase D workstream tests over real (migrated) SQLite.

Pins the four final workstreams of the transformation:

- AcademyService: EPPP phase classification from real DOBs, bio-banding
  through the real Mirwald estimator (and honest unbanded for females —
  the male-only equation), minutes guidance with small-sample honesty,
  safeguarding holds as hard blocks.
- OppositionService: real dossier over stored matches (goal attribution
  via player rows), no_data for unknown opponents, kloppy no-provider
  honesty, set-play library structure.
- OperatingProgramService: versioned program documents, ritual
  instantiation (idempotent), KPI registry validation, honest no_data
  KPI snapshot.
- Trust layer: intervention efficacy windows (measured /
  insufficient_sample / unmeasurable / rule_not_found) and the LLM
  groundedness gate (evidence registry, uncited claims refused,
  fabricated numbers flagged).
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

    db = tmp_path / "phase_d.db"
    mgr = MigrationManager(db, Path("src/kawkab/migrations"))
    mgr.migrate()
    svc = StorageService.__new__(StorageService)
    svc._conn = sqlite3.connect(str(db))
    svc._conn.row_factory = sqlite3.Row
    svc._use_postgres = False
    svc._db_path = db
    yield svc
    svc._conn.close()


def _add_profile(storage, name: str, dob: str | None) -> int:
    cur = storage._conn.execute(
        """INSERT INTO player_profiles (global_id, display_name, date_of_birth, created_at)
           VALUES (?, ?, ?, '2026-09-01')""",
        (f"g_{name.replace(' ', '_')}", name, dob),
    )
    storage._conn.commit()
    return int(cur.lastrowid)


def _add_match(storage, mid: int, home: str, away: str, date_str: str = "2026-08-10") -> None:
    storage._conn.execute(
        """INSERT INTO matches (id, name, video_path, home_team, away_team, match_date, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (mid, f"M{mid}", f"/tmp/m{mid}.mp4", home, away, date_str, "2026-08-15"),
    )
    storage._conn.commit()


def _add_event(
    storage,
    mid: int,
    etype: str,
    ts: float,
    team: str = "home",
    track_id: int | None = None,
    completed: int = 1,
    x: float = 50.0,
    y: float = 50.0,
) -> None:
    storage._conn.execute(
        """INSERT INTO events (match_id, timestamp, event_type, from_track_id, team, completed, confidence, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (mid, ts, etype, track_id, team, completed, 1.0, json.dumps({"x": x, "y": y})),
    )
    storage._conn.commit()


# ----------------------------------------------------------------------
# D1: Academy
# ----------------------------------------------------------------------


class TestAcademy:
    @pytest.mark.asyncio
    async def test_squad_phases_classify_from_dob(self, storage):
        from kawkab.services.academy_service import AcademyService

        _add_profile(storage, "Kid", "2017-05-01")  # ~9 -> foundation
        _add_profile(storage, "Youth", "2011-05-01")  # ~15 -> youth_development
        _add_profile(storage, "ProDev", "2007-05-01")  # ~19 -> professional_development
        _add_profile(storage, "Senior", "1995-05-01")  # ~31 -> senior
        _add_profile(storage, "NoDob", None)

        out = await AcademyService(storage).squad_phases(ref_date="2026-09-20")
        assert out["counts"]["foundation"] == 1
        assert out["counts"]["youth_development"] == 1
        assert out["counts"]["professional_development"] == 1
        assert out["counts"]["senior"] == 1
        assert out["counts"]["unknown"] == 1
        assert out["provenance"]["profiles_without_dob"] == 1
        assert "Elite Player Performance Plan" in out["provenance"]["eipp_reference"]

    @pytest.mark.asyncio
    async def test_bio_banding_bands_males_and_refuses_females(self, storage):
        from kawkab.services.academy_service import AcademyService

        svc = AcademyService(storage)
        # Real Mirwald male input (pre-PHV: negative offset).
        male = {
            "player_id": 1,
            "name": "Youngster",
            "age_years": 12.0,
            "standing_height_cm": 150.0,
            "sitting_height_cm": 75.0,
            "sex": "male",
        }
        out = await svc.bio_banded_groups([male])
        assert any(g["player_id"] == 1 for band in out["groups"].values() for g in band)
        # Female inputs cannot be estimated (unverified coefficients):
        female = dict(male, player_id=2, sex="female")
        out2 = await svc.bio_banded_groups([female])
        assert out2["unbanded"], "female input must land in unbanded, not a guessed band"
        assert "not verifiable" in str(out2["unbanded"][0].get("reason", ""))

    @pytest.mark.asyncio
    async def test_minutes_management_small_sample_honesty(self, storage):
        from kawkab.services.academy_service import SMALL_SAMPLE_MATCHES, AcademyService

        pid = _add_profile(storage, "Winger", "2011-05-01")  # ~15 -> youth development
        mid = 501
        _add_match(storage, mid, "Kawkab", "Rivals")
        # One logged match: far below the small-sample threshold.
        await storage.save_minutes_entry(
            player_id=pid,
            match_id=mid,
            minutes_played=90,
            started=True,
            age_phase="youth_development",
        )
        out = await AcademyService(storage).minutes_management(pid)
        assert out["sample_size"] == "small_sample"
        assert out["matches_logged"] < SMALL_SAMPLE_MATCHES
        assert "directional only" in out["provenance"]["honesty_note"]
        assert out["minutes_by_phase"]["youth_development"]["minutes"] == 90
        assert out["phase"] == "youth_development"

    @pytest.mark.asyncio
    async def test_safeguarding_hold_is_hard_block(self, storage):
        from kawkab.services.academy_service import AcademyService

        held = _add_profile(storage, "Held", "2009-05-01")
        free = _add_profile(storage, "Free", "2009-06-01")
        await storage.set_medical_clearance(
            player_id=held,
            status="unavailable",
            cleared_by="safeguarding officer",
            reason="open welfare case",
            source="safeguarding",
        )
        out = await AcademyService(storage).academy_selection_view([held, free])
        by_id = {p["player_id"]: p for p in out["players"]}
        assert by_id[held]["safeguarding_hold"] is True
        assert by_id[held]["selection"] == "blocked"
        assert by_id[free]["safeguarding_hold"] is False
        assert out["provenance"]["hard_blocks"] == 1


# ----------------------------------------------------------------------
# D2: Opposition intelligence
# ----------------------------------------------------------------------


class TestOpposition:
    @pytest.mark.asyncio
    async def test_unknown_opponent_is_no_data_never_template(self, storage):
        from kawkab.services.opposition_service import OppositionService

        out = await OppositionService(storage, None).build_dossier("Never Seen FC")
        assert out["state"] == "no_data"
        assert out["provider_available"] is False
        assert "no stored matches" in out["honesty_note"]

    @pytest.mark.asyncio
    async def test_dossier_from_real_stored_matches(self, storage):
        from kawkab.services.opposition_service import OppositionService

        # Two matches vs Rivals; Rivals concede 3 goals total, their
        # striker (track 9) scores twice; our completed-pass share is high.
        _add_match(storage, 1, "Kawkab", "Rivals")
        _add_match(storage, 2, "Rivals", "Kawkab")
        await storage.save_players_bulk(
            1, [{"track_id": 9, "name": "Opp Striker", "team": "away", "jersey_number": 9}]
        )
        _add_event(storage, 1, "goal", 10.0, team="away", track_id=9)
        _add_event(storage, 1, "goal", 20.0, team="away", track_id=9)
        _add_event(storage, 1, "goal", 30.0, team="home", track_id=7)
        _add_event(storage, 2, "goal", 10.0, team="away", track_id=7)  # Kawkab away goal
        for k in range(30):
            _add_event(storage, 1, "pass", 100.0 + k, team="home")
        for k in range(10):
            _add_event(storage, 1, "pass", 200.0 + k, team="away")

        out = await OppositionService(storage, None).build_dossier("Rivals")
        assert out["state"] == "ok"
        assert out["provider_available"] is True
        assert out["provenance"]["matches_analyzed"] == 2
        assert out["provenance"]["sample_size"] == "small_sample"
        # Possession is completed-pass share and labeled as such:
        assert "completed-pass share" in out["provenance"]["honesty_note"]
        # Scoreline "prediction" carries its naive-average disclaimer:
        assert "not a model forecast" in out["provenance"]["scoreline_note"]

    def test_kloppy_probe_honesty_contract(self):
        """The probe reports reality in either direction — never a middle state."""
        from kawkab.services.opposition_service import _probe_kloppy

        status = _probe_kloppy()
        assert status["provider"] == "kloppy"
        assert "statsbomb" in status["supported_formats"]
        try:
            import kloppy  # noqa: F401

            available = True
        except ImportError:
            available = False
        assert status["provider_available"] is available
        if available:
            assert status["version"]
            assert "no vendor account" in status["honesty_note"]
        else:
            assert "not installed" in status["reason"]

    @pytest.mark.asyncio
    async def test_set_play_library_structure(self, storage, monkeypatch):
        import kawkab.services.knowledge_service as ks_mod
        from kawkab.services.knowledge_service import KnowledgeService
        from kawkab.services.opposition_service import OppositionService

        class _RealPaths:
            knowledge_base = _REAL_KB_ROOT

        monkeypatch.setattr(ks_mod, "get_paths", lambda: _RealPaths())
        kb = KnowledgeService()
        await kb.initialize()

        out = await OppositionService(storage, kb).set_play_library()
        assert "drills" in out and "rules" in out
        for d in out["drills"]:
            assert (
                "set_piece" in str(d["category"]).lower() or "aerial" in str(d["category"]).lower()
            )


# ----------------------------------------------------------------------
# D3: Operating program
# ----------------------------------------------------------------------


class TestOperatingProgram:
    @pytest.mark.asyncio
    async def test_weekly_rhythm_is_versioned(self, storage):
        from kawkab.services.operating_program_service import OperatingProgramService

        svc = OperatingProgramService(storage)
        week = [{"day_label": "MD+1", "rituals": [{"ritual_type": "post_match_review"}]}]
        v1 = await svc.publish_weekly_rhythm(
            week, name="in-season morphocycle week", created_by="head coach"
        )
        assert v1["success"] and v1["days"] == 1
        # Same name republishes as a new version; audit trail preserved:
        v2 = await svc.publish_weekly_rhythm(
            week, name="in-season morphocycle week", created_by="head coach"
        )
        assert v2["document_id"] != v1["document_id"]
        prog = await svc.current_program()
        assert prog["weekly_rhythm"]["version"] == 2

    @pytest.mark.asyncio
    async def test_ritual_instantiation_idempotent(self, storage):
        from kawkab.services.operating_program_service import OperatingProgramService

        svc = OperatingProgramService(storage)
        week = [
            {"day_label": "MD+1", "rituals": [{"ritual_type": "post_match_review"}]},
            {
                "day_label": "MD-3",
                "rituals": [{"ritual_type": "staff_sync"}, {"ritual_type": "wellness_huddle"}],
            },
        ]
        await svc.publish_weekly_rhythm(week)
        r1 = await svc.instantiate_week_rituals("2026-09-21", completed_by="staff")
        assert r1["success"] and r1["rituals_scheduled"] == 3
        r2 = await svc.instantiate_week_rituals("2026-09-21")
        assert r2["success"] and r2["rituals_scheduled"] == 3  # upsert, not duplicate

    @pytest.mark.asyncio
    async def test_kpi_registry_rejects_unknown_metrics(self, storage):
        from kawkab.services.operating_program_service import OperatingProgramService

        out = await OperatingProgramService(storage).publish_kpi_tree(
            [
                {
                    "key": "x",
                    "label": "X",
                    "metric": "vibes_index",
                    "target": 1,
                    "direction": "higher",
                }
            ]
        )
        assert "error" in out and "supported_metrics" in out

    @pytest.mark.asyncio
    async def test_kpi_snapshot_honest_no_data(self, storage):
        from kawkab.services.operating_program_service import OperatingProgramService

        svc = OperatingProgramService(storage)
        await svc.publish_kpi_tree(
            [
                {
                    "key": "avail",
                    "label": "Availability",
                    "metric": "availability_pct",
                    "target": 95,
                    "direction": "higher",
                }
            ]
        )
        snap = await svc.kpi_snapshot()
        kpi = snap["kpis"][0]
        assert kpi["state"] == "no_data"  # empty club: no data, not zero
        assert kpi["basis"]

    @pytest.mark.asyncio
    async def test_kpi_snapshot_measures_availability(self, storage):
        from kawkab.services.operating_program_service import OperatingProgramService

        fit = _add_profile(storage, "Fit", "2000-01-01")
        out = _add_profile(storage, "Out", "2000-02-01")
        await storage.set_medical_clearance(player_id=fit, status="fit", source="medical")
        await storage.set_medical_clearance(player_id=out, status="unavailable", source="medical")
        svc = OperatingProgramService(storage)
        await svc.publish_kpi_tree(
            [
                {
                    "key": "avail",
                    "label": "Availability",
                    "metric": "availability_pct",
                    "target": 95,
                    "direction": "higher",
                }
            ]
        )
        snap = await svc.kpi_snapshot()
        kpi = snap["kpis"][0]
        assert kpi["state"] == "ok"
        assert kpi["value"] == 50.0
        assert "1 fit of 2 cleared profiles" in kpi["basis"]


# ----------------------------------------------------------------------
# D4a: Intervention efficacy
# ----------------------------------------------------------------------


def _seed_turnover_plan(storage, after_matches: int):
    """Plan built on matches 1-3; N+1/N+2 = matches 4/5 (if present)."""
    for mid in (1, 2, 3, 4, 5)[: 3 + after_matches]:
        _add_match(storage, mid, "Kawkab", f"Opp{mid}", f"2026-08-{10 + mid}")
    for mid, n_to in [(1, 5), (2, 4), (3, 5), (4, 1), (5, 2)]:
        for k in range(n_to):
            _add_event(storage, mid, "turnover", k * 60.0, x=10.0, y=5.0)
    for mid in (1, 2, 3, 4, 5)[: 3 + after_matches]:
        for k in range(2):  # activity so windows are populated
            _add_event(storage, mid, "pass", 300.0 + k)
    payload = {
        "based_on_match_ids": [1, 2, 3],
        "priority_diagnoses": [
            {
                "rule_id": "defensive_third_errors_v1",
                "rule_name": "defensive third errors",
                "confidence": 0.8,
            }
        ],
    }
    cur = storage._conn.execute(
        """INSERT INTO training_plans (match_id, title, status, duration_weeks, priority_diagnoses, payload, source, created_by)
           VALUES (3, 'Plan N', 'active', 1, ?, ?, 'reasoning_engine', 'coach')""",
        (json.dumps(payload["priority_diagnoses"]), json.dumps(payload)),
    )
    storage._conn.commit()
    return int(cur.lastrowid)


class TestInterventionEfficacy:
    @pytest.fixture(autouse=True)
    def _real_kb(self, monkeypatch):
        import kawkab.services.knowledge_service as ks_mod

        class _RealPaths:
            knowledge_base = _REAL_KB_ROOT

        monkeypatch.setattr(ks_mod, "get_paths", lambda: _RealPaths())

    async def _kb(self):
        from kawkab.services.knowledge_service import KnowledgeService

        kb = KnowledgeService()
        await kb.initialize()
        return kb

    @pytest.mark.asyncio
    async def test_measured_delta_two_after_matches(self, storage):
        from kawkab.services.intervention_efficacy_service import InterventionEfficacyService

        plan_id = _seed_turnover_plan(storage, after_matches=2)
        kb = await self._kb()
        rep = await InterventionEfficacyService(storage, kb).measure_intervention(plan_id)
        r = rep["rules"][0]
        assert rep["after_window_complete"] is True
        assert rep["after_window"] == [4, 5]
        assert r["state"] == "measured"
        assert r["metric"] == "turnovers_in_defensive_third"
        assert r["direction"] == "lower_after_is_better"
        assert r["mean_before"] == 4.5 and r["mean_after"] == 1.5
        assert r["delta"] == -3.0 and r["improved"] is True
        assert "not a causal verdict" in r["note"]

    @pytest.mark.asyncio
    async def test_one_after_match_is_insufficient_sample(self, storage):
        from kawkab.services.intervention_efficacy_service import InterventionEfficacyService

        plan_id = _seed_turnover_plan(storage, after_matches=1)
        kb = await self._kb()
        rep = await InterventionEfficacyService(storage, kb).measure_intervention(plan_id)
        r = rep["rules"][0]
        assert r["state"] == "insufficient_sample"
        assert rep["after_window_complete"] is False
        # The delta may be observed but must not be dressed as evidence:
        assert "noise, not evidence" in r["note"]
        assert r["delta_observed_but_not_conclusive"] == 1.0 - 4.5  # match 4 only

    @pytest.mark.asyncio
    async def test_tracking_level_metric_is_unmeasurable(self, storage):
        from kawkab.services.intervention_efficacy_service import InterventionEfficacyService

        plan_id = _seed_turnover_plan(storage, after_matches=2)
        storage._conn.execute(
            "UPDATE training_plans SET payload = ? WHERE id = ?",
            (
                json.dumps(
                    {
                        "based_on_match_ids": [1, 2, 3],
                        "priority_diagnoses": [
                            {
                                "rule_id": "goals_conceded_left_channel_v1",
                                "rule_name": "left channel",
                                "confidence": 0.7,
                            }
                        ],
                    }
                ),
                plan_id,
            ),
        )
        storage._conn.commit()
        kb = await self._kb()
        rep = await InterventionEfficacyService(storage, kb).measure_intervention(plan_id)
        r = rep["rules"][0]
        assert r["state"] == "unmeasurable"
        assert "'no data' is not 'no change'" in r["note"]

    @pytest.mark.asyncio
    async def test_unknown_rule_and_incomplete_plan(self, storage):
        from kawkab.services.intervention_efficacy_service import InterventionEfficacyService

        plan_id = _seed_turnover_plan(storage, after_matches=2)
        kb = await self._kb()
        svc = InterventionEfficacyService(storage, kb)
        storage._conn.execute(
            "UPDATE training_plans SET payload = ? WHERE id = ?",
            (
                json.dumps(
                    {
                        "based_on_match_ids": [1, 2, 3],
                        "priority_diagnoses": [{"rule_id": "no_such_rule", "rule_name": "ghost"}],
                    }
                ),
                plan_id,
            ),
        )
        storage._conn.commit()
        rep = await svc.measure_intervention(plan_id)
        assert rep["rules"][0]["state"] == "rule_not_found"

        bad = await svc.measure_intervention(plan_id + 1000)
        assert "error" in bad  # missing plan


# ----------------------------------------------------------------------
# D4b: Groundedness gate
# ----------------------------------------------------------------------


class TestGroundedness:
    @pytest.mark.asyncio
    async def test_evidence_registry_records_real_counts(self, storage):
        from kawkab.services.groundedness_service import GroundednessService

        _add_match(storage, 1, "Kawkab", "Rivals")
        _add_event(storage, 1, "goal", 10.0, team="home", track_id=7)
        _add_event(storage, 1, "yellow_card", 40.0, team="away", track_id=9)
        _add_event(storage, 1, "shot", 50.0, team="home")
        _add_event(storage, 1, "shot", 60.0, team="away")
        _add_event(storage, 1, "pass", 70.0, team="home")

        out = await GroundednessService(storage).register_match_evidence(1)
        assert out["evidence_id"] > 0
        p = out["payload"]
        assert p["event_counts"] == {"goal": 1, "yellow_card": 1, "shot": 2, "pass": 1}
        assert p["goal_count"] == 1
        assert p["shots_by_team"]["home"] == 1 and p["shots_by_team"]["away"] == 1
        assert "no external or inferred data" in p["source"]

    @pytest.mark.asyncio
    async def test_uncited_claims_do_not_ship(self, storage):
        from kawkab.services.groundedness_service import GroundednessService

        gate = await GroundednessService(storage).gate_claims("We dominated. 3 goals scored.", [])
        assert gate["status"] == "ungrounded"
        assert "without cited evidence" in gate["note"]

    @pytest.mark.asyncio
    async def test_gate_flags_fabricated_numbers(self, storage):
        from kawkab.services.groundedness_service import GroundednessService

        _add_match(storage, 1, "Kawkab", "Rivals")
        _add_event(storage, 1, "goal", 10.0, team="home", track_id=7)
        svc = GroundednessService(storage)
        ev = await svc.register_match_evidence(1)

        grounded = await svc.gate_claims(
            "Kawkab scored 1 goal against Rivals.", [ev["evidence_id"]]
        )
        assert grounded["status"] == "grounded"

        fabricated = await svc.gate_claims(
            "Kawkab scored 4 goals against Rivals.", [ev["evidence_id"]]
        )
        assert fabricated["status"] == "ungrounded"
        s = fabricated["sentences"][0]
        assert "4" in s["unverified_numbers"]

    @pytest.mark.asyncio
    async def test_generate_and_gate_report_attaches_grounding(self, storage):
        from kawkab.services.groundedness_service import GroundednessService

        _add_match(storage, 1, "Kawkab", "Rivals")
        _add_event(storage, 1, "goal", 10.0, team="home", track_id=7)

        class _StubLLM:
            async def generate_coach_report(self, summary: str, language: str) -> str:
                return "Kawkab scored 1 goal. The moon is made of 9 cheeses."

        out = await GroundednessService(storage).generate_and_gate_report(
            _StubLLM(), 1, "en", "summarize"
        )
        assert out["report_text"] == "Kawkab scored 1 goal. The moon is made of 9 cheeses."
        assert out["grounding"]["status"] == "ungrounded"
        assert out["evidence_id"] > 0
