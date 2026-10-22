"""GUI e2e: the Training OS daily rituals and the multi-match plan loop.

Phase B pins, through the real application stack (real MainWindow boot,
real bridge, real sqlite):

1. Daily wellness ritual: save_wellness -> get_squad_wellness over the
   real bridge, with the normalized Hooper score computed by storage.
2. Session RPE: create_training_session -> submit_session_rpe ->
   get_session_rpe, load = RPE x minutes persisted verbatim.
3. Drill feedback: save -> read with mean_effectiveness.
4. Multi-match plan loop: three real matches seeded via the production
   storage API, each conceding from the left third ->
   generate_training_plan_multi_match -> the zone rule confirms
   (min_matches=3) -> plan persisted with provenance and readable
   through a fresh storage instance (restart simulation).

Anti-fabrication assertions apply throughout: values read back must be
the values written, and diagnoses come only from the real engine over
real stored events.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json

import pytest

pyside6 = pytest.importorskip("PySide6")


@pytest.fixture()
def qapp(monkeypatch, tmp_path):
    """Offscreen QApplication with an isolated environment.

    Same isolation contract as test_gui_handler_dispatch.py.
    """
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.setenv("XDG_DOCUMENTS_DIR", str(tmp_path / "xdg-docs"))
    monkeypatch.setenv("KAWKAB_DEBUG", "false")
    monkeypatch.delenv("KAWKAB_DB_URL", raising=False)

    import kawkab.core.paths as paths_mod

    if hasattr(paths_mod, "_paths"):
        monkeypatch.setattr(paths_mod, "_paths", None)
    from kawkab.core.config import get_settings

    get_settings.cache_clear()

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(["kawkab-gui-e2e"])
    yield app
    get_settings.cache_clear()


def _reset_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DOCUMENTS_DIR", str(tmp_path / "xdg-docs"))
    monkeypatch.setenv("HOME", str(tmp_path))
    import kawkab.core.paths as paths_mod

    if hasattr(paths_mod, "_paths"):
        monkeypatch.setattr(paths_mod, "_paths", None)


def _boot_window(qapp):
    from kawkab.app import MainWindow

    return MainWindow()


def test_daily_wellness_and_rpe_rituals_through_bridge(qapp, tmp_path, monkeypatch):
    """Wellness and sRPE round-trip through the real bridge and DB."""
    _reset_paths(monkeypatch, tmp_path)
    window = _boot_window(qapp)
    try:

        async def main():
            storage = window.storage
            try:
                await storage.initialize()

                # --- morning wellness ritual ---------------------------
                saved = json.loads(
                    await window.bridge.save_wellness(
                        "1",
                        json.dumps(
                            {
                                "record_date": "2026-09-19",
                                "sleep_quality": 4,
                                "fatigue": 2,
                                "soreness": 2,
                                "stress": 3,
                                "mood": 4,
                                "source": "e2e",
                            }
                        ),
                    )
                )
                assert saved.get("success") is True, str(saved)

                squad = json.loads(await window.bridge.get_squad_wellness("2026-09-19"))
                assert squad.get("success") is True and squad["count"] == 1, str(squad)
                entry = squad["entries"][0]
                assert entry["wellness_score"] == pytest.approx(3.0), str(entry)

                # --- session RPE collection ----------------------------
                session = json.loads(
                    await window.bridge.create_training_session("2026-09-19", "tactical", "MD-3")
                )
                assert session.get("success") is True, str(session)
                sid = str(session["session_id"])

                rpe = json.loads(await window.bridge.submit_session_rpe(sid, "1", "7.5", "75"))
                assert rpe.get("success") is True, str(rpe)

                rows = json.loads(await window.bridge.get_session_rpe(sid))
                assert rows["count"] == 1, str(rows)
                assert rows["rpe_rows"][0]["load"] == pytest.approx(7.5 * 75), str(rows)

                # --- drill feedback loop -------------------------------
                fb = json.loads(
                    await window.bridge.save_drill_feedback(
                        "rondo_4v2",
                        json.dumps(
                            {"effectiveness": 5, "observations": "crisp", "session_id": int(sid)}
                        ),
                    )
                )
                assert fb.get("success") is True, str(fb)
                read = json.loads(await window.bridge.get_drill_feedback("rondo_4v2"))
                assert read["count"] == 1 and read["mean_effectiveness"] == 5.0, str(read)
            finally:
                await storage.close()

        asyncio.run(main())
    finally:
        window.close()
        qapp.processEvents()


def test_multi_match_plan_loop_confirms_and_persists(qapp, tmp_path, monkeypatch):
    """3 pooled matches -> confirmed zone diagnosis -> persisted plan."""
    _reset_paths(monkeypatch, tmp_path)
    window = _boot_window(qapp)
    try:

        async def main():
            storage = window.storage
            try:
                await storage.initialize()
                match_ids = []
                for i in (1, 2, 3):
                    mid = await storage.save_match(f"E2E multi {i}", f"e2e{i}.mp4", "Reds", "Blues")
                    match_ids.append(mid)
                    # zone rides in metadata: save_events_bulk preserves
                    # the metadata dict verbatim, and the plan slot's
                    # normalizer merges it back for the rule checkers.
                    await storage.save_events_bulk(
                        mid,
                        [
                            {
                                "type": "goal",
                                "timestamp": 60.0 * j,
                                "team": "away",
                                "metadata": {"zone": "left_third"},
                            }
                            for j in range(3)
                        ],
                    )

                raw = await window.bridge.generate_training_plan_multi_match(json.dumps(match_ids))
                out = json.loads(raw)
                assert "error" not in out, f"multi-match loop failed: {out}"
                assert out.get("success") is True, str(out)
                assert out.get("persisted") is True, str(out)
                assert out.get("confirmed_count", 0) >= 1, (
                    f"3 pooled matches must confirm the zone rule: {out}"
                )
                plan = out["plan"]
                assert plan["multi_match"] is True
                assert plan["based_on_match_ids"] == match_ids

                # the plan survives a "restart": read back fresh
                plan_id = out["plan_id"]
                assert plan_id, str(out)

                def _fresh():
                    from kawkab.services.storage_service import StorageService

                    return StorageService()

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    fresh = await asyncio.get_event_loop().run_in_executor(pool, _fresh)
                await fresh.initialize()
                try:
                    row = await fresh.get_training_plan(plan_id)
                    assert row is not None, "multi-match plan missing after restart"
                    assert row.get("source") == "reasoning_engine_multi_match"
                    priorities = row.get("priority_diagnoses", [])
                    assert priorities and any(
                        p.get("confirmation") == "confirmed" for p in priorities
                    ), f"confirmed provenance not persisted: {priorities}"
                finally:
                    await fresh.close()
            finally:
                await storage.close()

        asyncio.run(main())
    finally:
        window.close()
        qapp.processEvents()
