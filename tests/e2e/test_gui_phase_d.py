"""Phase D GUI e2e: academy, operating program, opposition, trust layer.

Everything runs through the real MainWindow -> Bridge -> handlers ->
services -> migrated SQLite stack, one feature per test:

1. Academy loop: profile -> EPPP phase view -> minutes management ->
   safeguarding clearance blocks selection.
2. Operating program loop: publish weekly rhythm -> instantiate rituals
   -> KPI snapshot (honest no_data on an empty club).
3. Opposition + trust loop: stored opponent match -> real dossier;
   kloppy no-provider status; evidence registration; a fabricated
   number in a review is refused by the claim gate; intervention
   efficacy over a seeded plan reports honest windows.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json

import pytest

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    from kawkab.core.config import get_settings

    get_settings.cache_clear()  # settings bake tmp paths; stale cache leaks DBs across modules
    app = QApplication.instance() or QApplication(["kawkab-gui-e2e"])
    yield app
    get_settings.cache_clear()


def _reset_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DOCUMENTS_DIR", str(tmp_path / "xdg-docs"))
    monkeypatch.setenv("HOME", str(tmp_path))
    import kawkab.core.paths as paths_mod

    if hasattr(paths_mod, "_paths"):
        monkeypatch.setattr(paths_mod, "_paths", None)


def test_academy_loop_through_bridge(qapp, tmp_path, monkeypatch):
    """Phases, minutes, and safeguarding-blocked selection end to end."""
    _reset_paths(monkeypatch, tmp_path)
    from kawkab.app import MainWindow

    window = MainWindow()
    try:

        async def main():
            storage = window.storage
            try:
                await storage.initialize()
                pid = await storage.save_player_profile(
                    {
                        "global_id": "e2e_academy_1",
                        "display_name": "Academy Winger",
                        "date_of_birth": "2011-05-01",
                    }
                )
                mid = await storage.save_match("E2E academy", "e2e_ac.mp4", "Kawkab", "Rivals")
                await storage.save_minutes_entry(
                    player_id=pid,
                    match_id=mid,
                    minutes_played=45,
                    started=True,
                    age_phase="youth_development",
                )

                phases = json.loads(await window.bridge.get_academy_squad_phases("2026-09-20"))
                assert "error" not in phases, str(phases)
                assert phases["counts"]["youth_development"] == 1, str(phases["counts"])
                assert phases["provenance"]["profiles_without_dob"] == 0

                minutes = json.loads(await window.bridge.get_minutes_management(str(pid)))
                assert "error" not in minutes, str(minutes)
                assert minutes["sample_size"] == "small_sample"
                assert minutes["phase"] == "youth_development"

                # Safeguarding hold -> hard block in the selection view.
                await storage.set_medical_clearance(
                    player_id=pid,
                    status="unavailable",
                    reason="welfare case",
                    source="safeguarding",
                )
                view = json.loads(await window.bridge.get_academy_selection_view(json.dumps([pid])))
                assert "error" not in view, str(view)
                player = view["players"][0]
                assert player["safeguarding_hold"] is True
                assert player["selection"] == "blocked"
                assert view["provenance"]["hard_blocks"] == 1
            finally:
                await storage.close()

        asyncio.run(main())
    finally:
        window.close()
        qapp.processEvents()


def test_operating_program_loop_through_bridge(qapp, tmp_path, monkeypatch):
    """Publish rhythm -> instantiate rituals -> honest KPI snapshot."""
    _reset_paths(monkeypatch, tmp_path)
    from kawkab.app import MainWindow

    window = MainWindow()
    try:

        async def main():
            storage = window.storage
            try:
                await storage.initialize()
                week = [
                    {"day_label": "MD+1", "rituals": [{"ritual_type": "post_match_review"}]},
                    {"day_label": "MD-3", "rituals": [{"ritual_type": "wellness_huddle"}]},
                ]
                pub = json.loads(
                    await window.bridge.publish_program_document("weekly_rhythm", json.dumps(week))
                )
                assert pub.get("success") is True, str(pub)

                rit = json.loads(await window.bridge.instantiate_week_rituals("2026-09-21"))
                assert rit.get("success") is True, str(rit)
                assert rit["rituals_scheduled"] == 2, str(rit)

                kpi = json.loads(await window.bridge.get_kpi_snapshot())
                # Empty club: states must be no_data, never fabricated zeros.
                assert "error" not in kpi, str(kpi)
                for entry in kpi.get("kpis", []):
                    assert entry["state"] == "no_data", str(entry)
                    assert entry["basis"], str(entry)
            finally:
                await storage.close()

        asyncio.run(main())
    finally:
        window.close()
        qapp.processEvents()


def test_opposition_dossier_and_vendor_honesty_through_bridge(qapp, tmp_path, monkeypatch):
    """Real dossier from stored matches; kloppy no-provider stays honest."""
    _reset_paths(monkeypatch, tmp_path)
    from kawkab.app import MainWindow

    window = MainWindow()
    try:

        async def main():
            storage = window.storage
            try:
                await storage.initialize()
                mid = await storage.save_match("E2E opp 1", "e2e_opp.mp4", "Kawkab", "Rivals")
                await storage.save_players_bulk(
                    mid,
                    [{"track_id": 9, "name": "Opp Striker", "team": "away", "jersey_number": 9}],
                )
                await storage.save_events_bulk(
                    mid,
                    [
                        {"type": "goal", "timestamp": 60.0, "team": "away"},
                        {"type": "pass", "timestamp": 90.0, "team": "home", "completed": True},
                    ],
                )

                dossier = json.loads(await window.bridge.get_opponent_dossier("Rivals"))
                assert "error" not in dossier, str(dossier)
                assert dossier["state"] == "ok", str(dossier.get("state"))
                assert dossier["provenance"]["matches_analyzed"] == 1
                assert "completed-pass share" in dossier["provenance"]["honesty_note"]

                # Unknown opponent must be no_data, never a template.
                empty = json.loads(await window.bridge.get_opponent_dossier("Ghost FC"))
                assert empty["state"] == "no_data", str(empty)

                vendor = json.loads(await window.bridge.get_vendor_import_status())
                assert "error" not in vendor, str(vendor)
                # The honesty contract is environment-independent: the
                # probe reports reality (available + version, or absent +
                # reason), never a fabricated middle state.
                assert vendor["provider"] == "kloppy", str(vendor)
                assert vendor["supported_formats"], str(vendor)
                if vendor["provider_available"]:
                    assert "no vendor account" in vendor["honesty_note"], str(vendor)
                else:
                    assert vendor["reason"], str(vendor)

                lib = json.loads(await window.bridge.get_set_play_library())
                assert "error" not in lib, str(lib)
                assert "drills" in lib and "rules" in lib
            finally:
                await storage.close()

        asyncio.run(main())
    finally:
        window.close()
        qapp.processEvents()


def test_trust_layer_evidence_and_gate_through_bridge(qapp, tmp_path, monkeypatch):
    """Evidence registry, claim gate, and efficacy windows via the bridge."""
    _reset_paths(monkeypatch, tmp_path)
    from kawkab.app import MainWindow

    window = MainWindow()
    try:

        async def main():
            storage = window.storage
            try:
                await storage.initialize()

                # --- evidence + claim gate ------------------------------
                mid = await storage.save_match("E2E trust", "e2e_trust.mp4", "Kawkab", "Rivals")
                await storage.save_events_bulk(
                    mid, [{"type": "goal", "timestamp": 60.0, "team": "home"}]
                )
                ev = json.loads(await window.bridge.register_match_evidence(str(mid)))
                assert "error" not in ev, str(ev)
                assert ev["payload"]["goal_count"] == 1, str(ev["payload"])

                grounded = json.loads(
                    await window.bridge.gate_report_claims(
                        "Kawkab scored 1 goal against Rivals.", json.dumps([ev["evidence_id"]])
                    )
                )
                assert grounded["status"] == "grounded", str(grounded)

                fabricated = json.loads(
                    await window.bridge.gate_report_claims(
                        "Kawkab scored 4 goals against Rivals.", json.dumps([ev["evidence_id"]])
                    )
                )
                assert fabricated["status"] == "ungrounded", str(fabricated)
                assert "4" in fabricated["sentences"][0]["unverified_numbers"], str(fabricated)

                uncited = json.loads(
                    await window.bridge.gate_report_claims("Kawkab scored 1 goal.", "[]")
                )
                assert uncited["status"] == "ungrounded", str(uncited)

                # --- intervention efficacy ------------------------------
                for i, mid_i in enumerate((1, 2, 3, 4, 5), start=1):
                    if mid_i == mid:
                        continue
                    await storage.save_match(f"E2E eff {i}", f"e2e_eff{i}.mp4", "Kawkab", f"Opp{i}")
                # 5/4/5 turnovers before, 1/2 after -> measured improvement.
                counts = {1: 5, 2: 4, 3: 5, 4: 1, 5: 2}
                for m_i, n_to in counts.items():
                    await storage.save_events_bulk(
                        m_i,
                        [
                            {"type": "turnover", "timestamp": 60.0 * j, "team": "home"}
                            for j in range(n_to)
                        ]
                        + [
                            {
                                "type": "pass",
                                "timestamp": 600.0 + j,
                                "team": "home",
                                "completed": True,
                            }
                            for j in range(2)
                        ],
                    )
                plan_payload = {
                    "based_on_match_ids": [1, 2, 3],
                    "priority_diagnoses": [
                        {
                            "rule_id": "defensive_third_errors_v1",
                            "rule_name": "defensive third errors",
                            "confidence": 0.8,
                        }
                    ],
                }
                storage._conn.execute(
                    """INSERT INTO training_plans (match_id, title, status, duration_weeks,
                       priority_diagnoses, payload, source, created_by)
                       VALUES (3, 'E2E plan', 'active', 1, ?, ?, 'reasoning_engine', 'e2e')""",
                    (
                        json.dumps(plan_payload["priority_diagnoses"]),
                        json.dumps(plan_payload),
                    ),
                )
                storage._conn.commit()
                plan_id = storage._conn.execute("SELECT MAX(id) FROM training_plans").fetchone()[0]

                eff = json.loads(await window.bridge.measure_intervention(str(plan_id)))
                assert "error" not in eff, str(eff)
                r = eff["rules"][0]
                assert r["state"] == "measured", str(r)
                assert r["direction"] == "lower_after_is_better"
                assert r["improved"] is True, str(r)
                assert eff["after_window_complete"] is True, str(eff)

                # A fresh-storage read proves the evidence registry persists
                # (the gate cites ids that must resolve across restarts).
                def _fresh():
                    from kawkab.services.storage_service import StorageService

                    return StorageService()

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    fresh = await asyncio.get_event_loop().run_in_executor(pool, _fresh)
                await fresh.initialize()
                try:
                    rows = await fresh.get_evidence_records(record_ids=[ev["evidence_id"]])
                    assert rows and rows[0]["id"] == ev["evidence_id"], (
                        "evidence missing after restart"
                    )
                finally:
                    await fresh.close()
            finally:
                await storage.close()

        asyncio.run(main())
    finally:
        window.close()
        qapp.processEvents()
