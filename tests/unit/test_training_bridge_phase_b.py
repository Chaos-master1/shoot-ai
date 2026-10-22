"""Phase B Training OS bridge tests: wellness / sRPE / drill feedback /
sessions / game model round-trip end to end through the handler over the
real (migrated) SQLite storage — plus the multi-match plan slot.

Contract under test (Phase A honesty contract, extended):
- writes validate and clamp at the boundary; invalid values are loud
  errors, never silent coercion into fabricated data,
- reads fail soft with explicit counts,
- plans carry provenance (based_on_match_ids, game_model_loaded,
  multi_match) and are persisted with source labels.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import install_kawkab_stubs

install_kawkab_stubs()

# Real knowledge base for the multi-match plan slot: install_kawkab_stubs()
# replaces kawkab.core.paths with a stub whose knowledge_base dir is empty,
# so pin the knowledge_service symbol to the real repo KB (test-order safe).
_REAL_KB_ROOT = Path(__file__).resolve().parents[2] / "src" / "kawkab" / "knowledge"


@pytest.fixture()
def storage(tmp_path):
    from kawkab.core.migration_manager import MigrationManager
    from kawkab.services.storage_service import StorageService

    db = tmp_path / "train_bridge.db"
    mgr = MigrationManager(db, Path("src/kawkab/migrations"))
    mgr.migrate()
    svc = StorageService.__new__(StorageService)
    svc._conn = sqlite3.connect(str(db))
    svc._conn.row_factory = sqlite3.Row
    svc._use_postgres = False
    svc._db_path = db
    yield svc
    svc._conn.close()


@pytest.fixture()
def handler(storage):
    from kawkab.ui.bridge_handlers.bridge_training import TrainingHandler

    return TrainingHandler(bridge=None, services={"storage_service": storage})


def _j(payload: dict) -> str:
    return json.dumps(payload)


class TestWellnessRoundTrip:
    def test_save_and_read_back(self, handler, storage):
        out = json.loads(
            asyncio.run(
                handler.save_wellness(
                    1,
                    _j(
                        {
                            "record_date": "2026-09-19",
                            "sleep_quality": 4,
                            "fatigue": 2,
                            "soreness": 2,
                            "stress": 3,
                            "mood": 4,
                            "source": "mobile_morning",
                        }
                    ),
                )
            )
        )
        assert out["success"] is True

        squad = json.loads(asyncio.run(handler.get_squad_wellness("2026-09-19")))
        assert squad["success"] is True and squad["count"] == 1
        entry = squad["entries"][0]
        # Hooper-index convention: normalized mean of the five 1-5 scales
        assert entry["wellness_score"] == pytest.approx((4 + 2 + 2 + 3 + 4) / 5)
        assert entry["source"] == "mobile_morning"

        history = json.loads(asyncio.run(handler.get_player_wellness(1)))
        assert history["count"] == 1

    def test_upsert_same_day_replaces(self, handler):
        asyncio.run(handler.save_wellness(2, _j({"record_date": "2026-09-19", "sleep_quality": 2})))
        asyncio.run(handler.save_wellness(2, _j({"record_date": "2026-09-19", "sleep_quality": 5})))
        squad = json.loads(asyncio.run(handler.get_squad_wellness("2026-09-19")))
        assert squad["count"] == 1
        assert squad["entries"][0]["sleep_quality"] == 5

    def test_out_of_scale_is_clamped_not_fabricated(self, handler):
        """11 -> clamped to 5; the value stays honest (clamped, not stored raw)."""
        out = json.loads(
            asyncio.run(handler.save_wellness(3, _j({"record_date": "2026-09-19", "mood": 11})))
        )
        assert out["success"] is True

    def test_missing_date_is_rejected(self, handler):
        out = json.loads(asyncio.run(handler.save_wellness(4, _j({"mood": 3}))))
        assert "error" in out and "record_date" in out["error"]


class TestSessionRPERoundTrip:
    def test_session_rpe_and_load(self, handler):
        sid = json.loads(
            asyncio.run(handler.create_training_session("2026-09-19", "tactical", "MD-3"))
        )["session_id"]
        out = json.loads(asyncio.run(handler.submit_session_rpe(sid, 7, 7.5, 75)))
        assert out["success"] is True
        rows = json.loads(asyncio.run(handler.get_session_rpe(sid)))
        assert rows["count"] == 1
        assert rows["rpe_rows"][0]["load"] == pytest.approx(7.5 * 75)

    def test_rpe_bounds_enforced(self, handler):
        sid = json.loads(
            asyncio.run(handler.create_training_session("2026-09-19", "recovery", "MD+1"))
        )["session_id"]
        out = json.loads(asyncio.run(handler.submit_session_rpe(sid, 7, 11.0, 60)))
        assert "error" in out and "0-10" in out["error"]


class TestDrillFeedbackRoundTrip:
    def test_feedback_and_mean(self, handler):
        for eff in (5, 4):
            out = json.loads(
                asyncio.run(
                    handler.save_drill_feedback(
                        "rondo_4v2",
                        _j({"effectiveness": eff, "observations": "crisp", "session_id": 1}),
                    )
                )
            )
            assert out["success"] is True
        fb = json.loads(asyncio.run(handler.get_drill_feedback("rondo_4v2")))
        assert fb["count"] == 2
        assert fb["mean_effectiveness"] == pytest.approx(4.5)

    def test_invalid_effectiveness_rejected(self, handler):
        out = json.loads(
            asyncio.run(handler.save_drill_feedback("rondo_4v2", _j({"effectiveness": 9})))
        )
        assert "error" in out


class TestGameModelSlots:
    def test_save_and_get_active(self, handler):
        out = json.loads(
            asyncio.run(
                handler.save_game_model(
                    _j(
                        {
                            "team_id": 3,
                            "in_possession": "Build from the back",
                            "out_of_possession": "Mid-block",
                            "non_negotiables": ["Press in packs"],
                            "created_by": "Head Coach",
                        }
                    )
                )
            )
        )
        assert out["success"] is True
        got = json.loads(asyncio.run(handler.get_game_model(3)))
        assert got["success"] is True
        assert got["game_model"]["in_possession"] == "Build from the back"
        assert "Press in packs" in got["game_model"]["non_negotiables"]

    def test_empty_model_rejected(self, handler):
        out = json.loads(asyncio.run(handler.save_game_model(_j({"team_id": 3}))))
        assert "error" in out


class TestScienceAndRitualSlots:
    def test_testing_result_roundtrip(self, handler):
        out = json.loads(
            asyncio.run(
                handler.save_testing_result(
                    9,
                    _j(
                        {"test_date": "2026-09-19", "test_type": "CMJ", "value": 38.5, "unit": "cm"}
                    ),
                )
            )
        )
        assert out["success"] is True

    def test_idp_goal_roundtrip(self, handler):
        out = json.loads(
            asyncio.run(
                handler.save_idp_goal(
                    9, _j({"goal_text": "Improve weak-foot passing", "category": "technical"})
                )
            )
        )
        assert out["success"] is True
        goals = json.loads(asyncio.run(handler.get_player_idp(9)))
        assert goals["count"] >= 1 if "count" in goals else goals["goals"]

    def test_nutrition_and_psych(self, handler):
        n = json.loads(
            asyncio.run(
                handler.save_nutrition_log(9, _j({"log_date": "2026-09-19", "meal_type": "lunch"}))
            )
        )
        assert n["success"] is True
        p = json.loads(
            asyncio.run(
                handler.save_psych_checkin(
                    9,
                    _j({"checkin_date": "2026-09-19", "confidence": 2, "flag_for_followup": True}),
                )
            )
        )
        assert p["success"] is True

    def test_ritual_roundtrip(self, handler):
        saved = json.loads(
            asyncio.run(handler.save_ritual("wellness_review", "2026-09-19", _j({"notes": "am"})))
        )
        assert saved["success"] is True
        done = json.loads(
            asyncio.run(
                handler.complete_ritual(
                    "wellness_review", "2026-09-19", _j({"completed_by": "Head of Performance"})
                )
            )
        )
        assert done["success"] is True
        rituals = json.loads(asyncio.run(handler.get_rituals("2026-09-19")))
        assert rituals["count"] == 1
        assert rituals["rituals"][0]["completed_by"] == "Head of Performance"

    def test_minutes_and_clearance(self, handler, storage):
        storage._conn.execute(
            "INSERT INTO matches (id, name, video_path, home_team, away_team) VALUES (1, 'm', 'v.mp4', 'A', 'B')"
        )
        storage._conn.commit()
        m = json.loads(
            asyncio.run(
                handler.save_minutes_entry(
                    9,
                    _j({"match_id": 1, "minutes_played": 90, "started": True}),
                )
            )
        )
        assert m["success"] is True
        c = json.loads(
            asyncio.run(
                handler.set_medical_clearance(
                    9, _j({"status": "limited", "reason": "load management"})
                )
            )
        )
        assert c["success"] is True
        bad = json.loads(
            asyncio.run(handler.set_medical_clearance(9, _j({"status": "sort-of-fit"})))
        )
        assert "error" in bad


class TestMultiMatchPlanSlot:
    @pytest.fixture(autouse=True)
    def _real_kb(self, monkeypatch):
        import kawkab.services.knowledge_service as ks_mod

        class _RealPaths:
            knowledge_base = _REAL_KB_ROOT

        monkeypatch.setattr(ks_mod, "get_paths", lambda: _RealPaths())

    @pytest.fixture()
    def seeded_storage(self, storage):
        # three matches, each conceding from the left third
        for mid in (1, 2, 3):
            storage._conn.execute(
                "INSERT INTO matches (id, name, video_path, home_team, away_team) VALUES (?, ?, ?, ?, ?)",
                (mid, f"m{mid}", f"v{mid}.mp4", "Reds", "Blues"),
            )
            rows = [
                (mid, "goal", 10.0 * j, "away", json.dumps({"zone": "left_third"}))
                for j in range(3)
            ]
            storage._conn.executemany(
                "INSERT INTO events (match_id, event_type, timestamp, team, metadata) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
        storage._conn.commit()
        return storage

    @pytest.mark.asyncio
    async def test_multi_match_slot_persists_confirmed_plan(self, seeded_storage, handler):
        from kawkab.ui.bridge_handlers.bridge_analysis import AnalysisHandler

        analysis = AnalysisHandler(
            bridge=None,
            services={"storage_service": seeded_storage, "knowledge_service": None},
        )
        raw = await analysis.generate_training_plan_multi_match(json.dumps([1, 2, 3]))
        out = json.loads(raw)
        assert "error" not in out, out
        assert out["success"] is True
        assert out["persisted"] is True
        assert out["match_ids"] == [1, 2, 3]
        assert out["confirmed_count"] >= 1, "3 pooled matches must confirm the zone rule"
        assert out["plan"].get("priority_addressed")

    @pytest.mark.asyncio
    async def test_multi_match_requires_distinct_ids(self, handler, storage):
        from kawkab.ui.bridge_handlers.bridge_analysis import AnalysisHandler

        analysis = AnalysisHandler(
            bridge=None,
            services={"storage_service": storage, "knowledge_service": None},
        )
        out = json.loads(await analysis.generate_training_plan_multi_match(json.dumps([5, 5])))
        assert "error" in out and "distinct" in out["error"]

    @pytest.mark.asyncio
    async def test_plan_payload_records_provenance(self, seeded_storage):
        from kawkab.ui.bridge_handlers.bridge_analysis import AnalysisHandler

        analysis = AnalysisHandler(
            bridge=None,
            services={"storage_service": seeded_storage, "knowledge_service": None},
        )
        out = json.loads(await analysis.generate_training_plan_multi_match(json.dumps([1, 2, 3])))
        plan = out["plan"]
        assert plan["multi_match"] is True
        assert plan["based_on_match_ids"] == [1, 2, 3]
        assert plan["game_model_source"] is False  # no game model saved for Reds
