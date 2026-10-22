"""Phase C sports-science service tests over real (migrated) SQLite.

Pins the honesty contracts of the three new services:

- LoadMonitoringService: sRPE ACWR from real stored session RPE rows,
  GPS/sRPE sources never blended, flags descriptive not predictive,
  provenance always present.
- TestingBatteryService: percentiles only from the club's own
  distribution, insufficient-distribution honesty, personal trend.
- MaturationService: verified Mirwald male equation values, honest
  not_implemented for females, validity provenance on every result,
  peak-growth-window load flags.
- PlayerProtocolService: conversation flags, never diagnoses; explicit
  follow-up requests are honored and escalate.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import install_kawkab_stubs

install_kawkab_stubs()


@pytest.fixture()
def storage(tmp_path):
    from kawkab.core.migration_manager import MigrationManager
    from kawkab.services.storage_service import StorageService

    db = tmp_path / "phase_c.db"
    mgr = MigrationManager(db, Path("src/kawkab/migrations"))
    mgr.migrate()
    svc = StorageService.__new__(StorageService)
    svc._conn = sqlite3.connect(str(db))
    svc._conn.row_factory = sqlite3.Row
    svc._use_postgres = False
    svc._db_path = db
    yield svc
    svc._conn.close()


class TestLoadMonitoring:
    @pytest.mark.asyncio
    async def test_acwr_from_real_stored_rpe(self, storage):
        from datetime import date, timedelta

        from kawkab.services.load_monitoring_service import LoadMonitoringService

        # 4 sessions: 3 in the last week (heavy acute), 1 four weeks ago
        # -> acute > chronic, but chronic mean across 4 windows nonzero.
        today = date.today()
        loads = [(today - timedelta(days=d), 900.0) for d in (0, 2, 4)] + [
            (today - timedelta(days=21), 800.0)
        ]
        for d, _load in loads:
            sid = await storage.save_training_session(d.isoformat(), "physical")
            await storage.save_session_rpe(sid, 10, 9.0, 100)  # load = 900, player 10
        svc = LoadMonitoringService(storage)
        state = await svc.player_load_state(10)
        assert state["srpe"]["acwr"] is not None
        assert state["srpe"]["band"] in {"normal", "high", "very_high"}
        assert state["srpe"]["source"] == "player_reported_sRPE"
        assert state["provenance"]["method"]

    @pytest.mark.asyncio
    async def test_thin_history_is_no_data_not_zero(self, storage):
        from kawkab.services.load_monitoring_service import LoadMonitoringService

        sid = await storage.save_training_session("2026-09-19", "tactical")
        await storage.save_session_rpe(sid, 11, 7.0, 60)
        state = await LoadMonitoringService(storage).player_load_state(11)
        assert state["srpe"]["acwr"] is None
        assert state["srpe"]["band"] == "no_data"
        assert state["flags"] == []

    @pytest.mark.asyncio
    async def test_high_band_with_declining_wellness_is_corroborated(self, storage):
        from datetime import date, timedelta

        from kawkab.services.load_monitoring_service import LoadMonitoringService

        today = date.today()
        # spike: 3 heavy days this week, then a light 4-week tail
        for d in (0, 2, 4):
            sid = await storage.save_training_session(
                (today - timedelta(days=d)).isoformat(), "physical"
            )
            await storage.save_session_rpe(sid, 12, 10.0, 120)
        for d in (7, 14, 21, 28):
            sid = await storage.save_training_session(
                (today - timedelta(days=d)).isoformat(), "recovery"
            )
            await storage.save_session_rpe(sid, 12, 3.0, 60)
        # declining wellness: old good scores, recent poor ones
        for i, (sq, mo) in enumerate([(5, 5), (4, 5), (2, 1), (1, 2)]):
            await storage.save_wellness(
                12, (today - timedelta(days=13 - i * 4)).isoformat(), sq, 2, 2, 2, mo
            )
        state = await LoadMonitoringService(storage).player_load_state(12)
        assert state["srpe"]["band"] in {"high", "very_high"}
        assert state["wellness_trend"] == "declining"
        flags = state["flags"]
        assert flags and flags[0]["type"] == "load_spike"
        assert flags[0]["corroborated_by_wellness"] is True

    @pytest.mark.asyncio
    async def test_gps_and_srpe_never_blended(self, storage):
        from kawkab.services.load_monitoring_service import LoadMonitoringService

        await storage.save_acwr(13, "2026-09-19", acute=600.0, chronic=400.0, acwr=1.5)
        state = await LoadMonitoringService(storage).player_load_state(13)
        assert state["gps"]["acwr"] == 1.5
        assert state["srpe"]["acwr"] is None  # no sRPE rows exist
        assert state["srpe"]["source"] != state["gps"]["source"]

    def test_acwr_math_needs_four_active_days(self):
        from kawkab.services.load_monitoring_service import LoadMonitoringService

        assert LoadMonitoringService._acwr_from_sessions([]) is None
        three = LoadMonitoringService._acwr_from_sessions(
            [("2026-09-19", 500.0), ("2026-09-18", 500.0), ("2026-09-17", 500.0)]
        )
        assert three is None
        # balanced loads -> ratio ~1
        four = LoadMonitoringService._acwr_from_sessions(
            [
                ("2026-09-19", 400.0),
                ("2026-09-12", 400.0),
                ("2026-09-05", 400.0),
                ("2026-08-29", 400.0),
            ]
        )
        assert four == pytest.approx(1.0, abs=0.01)


class TestTestingBattery:
    @pytest.mark.asyncio
    async def test_insufficient_distribution_is_honest(self, storage):
        from kawkab.services.testing_battery_service import TestingBatteryService

        # 3 results: far below the n>=8 bar
        for i, pid in enumerate((20, 21, 22)):
            await storage.save_testing_result(pid, "2026-09-01", "CMJ", 30.0 + i)
        svc = TestingBatteryService(storage)
        out = await svc.interpret_result(20, "CMJ", 31.0)
        assert out["interpretation"] == "insufficient_distribution"
        assert out["squad_percentile"] is None
        assert "not evidence" in out["note"]

    @pytest.mark.asyncio
    async def test_percentile_from_club_distribution(self, storage):
        from kawkab.services.testing_battery_service import TestingBatteryService

        # 10 recorded CMJ results 30..39; player scores 35 -> beats 5 of 9 others
        for i in range(10):
            pid = 30 + i
            await storage.save_testing_result(pid, "2026-09-01", "CMJ", 30.0 + i)
        svc = TestingBatteryService(storage)
        out = await svc.interpret_result(30, "CMJ", 35.0)
        assert out["interpretation"] == "interpreted"
        assert out["distribution_size"] == 10
        assert out["squad_percentile"] == pytest.approx(50.0, abs=12.0)
        assert out["provenance"]["normative"] is False

    @pytest.mark.asyncio
    async def test_personal_trend_direction(self, storage):
        from kawkab.services.testing_battery_service import TestingBatteryService

        await storage.save_testing_result(40, "2026-08-01", "CMJ", 30.0)
        svc = TestingBatteryService(storage)
        out = await svc.interpret_result(40, "CMJ", 33.0, test_date="2026-09-01")
        assert out["personal_trend"]["direction"] == "improved"
        assert out["personal_trend"]["delta_pct"] == pytest.approx(10.0, abs=0.1)


class TestMaturation:
    def test_male_offset_matches_published_example_range(self):
        from kawkab.services.maturation_service import MaturationService

        svc = MaturationService()
        # A 14.0y boy, 165cm standing, 82cm sitting (LL=83) computes to
        # about -0.7 via the verified Mirwald male coefficients:
        # offset = -9.236 + 0.0002708*83*82 - 0.001663*14*83
        #          + 0.007216*14*82 + 0.02292*14
        out = svc.estimate_offset(14.0, 165.0, 82.0, sex="male")
        assert out["maturity_offset"] == pytest.approx(-0.72, abs=0.05)
        assert out["band"] == "circum_phv"
        assert out["provenance"]["method"].startswith("Mirwald")

    def test_validity_notes_always_ship(self):
        from kawkab.services.maturation_service import MaturationService

        out = MaturationService().estimate_offset(13.0, 160.0, 80.0)
        assert any("±0.59" in n for n in out["provenance"]["validity_notes"])
        assert any("European-ancestry" in n for n in out["provenance"]["validity_notes"])

    def test_female_is_honestly_not_implemented(self):
        from kawkab.services.maturation_service import MaturationService

        out = MaturationService().estimate_offset(13.0, 160.0, 80.0, sex="female")
        assert out["maturity_offset"] is None
        assert out["classification"] == "not_implemented"
        assert "not verifiable" in out["provenance"]["error"]

    def test_implausible_input_fails_loudly(self):
        from kawkab.services.maturation_service import MaturationService

        out = MaturationService().estimate_offset(14.0, 165.0, 200.0)  # sitting > standing
        assert out["maturity_offset"] is None
        assert "error" in out["provenance"]

    def test_peak_growth_window_flags_high_minutes(self):
        from kawkab.services.maturation_service import MaturationService

        svc = MaturationService()
        flags = svc.growth_load_flags(maturity_offset=0.1, minutes_last_28d=450, age_phase="u15")
        types = {f["type"] for f in flags["flags"]}
        assert "peak_growth_window" in types
        assert "high_minutes_in_growth_window" in types
        assert flags["band"] == "circum_phv"


class TestPlayerProtocols:
    @pytest.mark.asyncio
    async def test_low_wellness_flagged_for_conversation(self, storage):
        from kawkab.services.player_protocol_service import PlayerProtocolService

        await storage.save_wellness(50, "2026-09-19", 1, 2, 2, 3, 2)
        await storage.save_wellness(51, "2026-09-19", 5, 4, 4, 4, 5)
        out = await PlayerProtocolService(storage).squad_readiness_summary("2026-09-19")
        assert out["submissions"] == 2
        assert out["flagged_for_conversation"][0]["player_id"] == 50
        assert "not a diagnosis" in out["provenance"]["note"]

    @pytest.mark.asyncio
    async def test_explicit_psych_followup_is_honored_and_escalates(self, storage):
        from kawkab.services.player_protocol_service import PlayerProtocolService

        for i in range(7):
            await storage.save_psych_checkin(
                60,
                f"2026-09-{19 - i:02d}",
                confidence=1,
                focus=2,
                motivation=2,
                anxiety=5,
                flag_for_followup=(i < 2),
            )
        out = await PlayerProtocolService(storage).psych_flags(60)
        types = {f["type"] for f in out["flags"]}
        assert "explicit_followup_request" in types
        assert out["escalate"], "explicit request + converging flags must escalate"

    @pytest.mark.asyncio
    async def test_nutrition_flags_are_conversations(self, storage):
        from kawkab.services.player_protocol_service import PlayerProtocolService

        for i in range(8):
            await storage.save_nutrition_log(61, f"2026-09-{19 - i:02d}", "lunch", 2, 2)
        out = await PlayerProtocolService(storage).nutrition_flags(61)
        types = {f["type"] for f in out["flags"]}
        assert "sustained_low_fueling" in types
        assert "not dietary prescriptions" in out["provenance"]["note"]
