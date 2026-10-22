"""GUI e2e: the diagnosis -> training plan -> persistence loop (Phase A WS2.3).

Closes the historical façade regression window. The original
``generate_training_plan`` slot fabricated five hardcoded diagnoses
(R001-R005) from raw event-type counts with constant confidences,
referenced drill IDs (D001-D009) that never existed in the knowledge
base, and never persisted anything — and every unit test of the slot
used stubs, so CI stayed green while users saw invented football.

This module runs the same anti-façade assertion the unit regression
file makes, but through the *real* application stack:

  real MainWindow boot -> real bridge -> real sqlite (WAL)
  -> real events persisted via the production storage API
  -> real ReasoningService over a real KnowledgeService (the actual
     YAML rule files and drill files on disk)
  -> real TrainingPlanGenerator
  -> real save_training_plan persistence
  -> read back through a fresh storage instance (restart simulation)

Seeding uses the production StorageService API, mirroring
test_gui_handler_dispatch.py's contract.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pyside6 = pytest.importorskip("PySide6")


@pytest.fixture()
def qapp(monkeypatch, tmp_path):
    """Offscreen QApplication with an isolated environment.

    Same isolation contract as test_gui_handler_dispatch.py: outward
    isolation from the developer's real HOME/Documents/cache, inward
    reset of the cached paths/settings singletons.
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

    window = MainWindow()
    return window


# ---------------------------------------------------------------- events
# Real-looking build-up failure: many incomplete home passes out of the
# defensive third, two turnovers in the build-up zone, one scrappy shot.
# Chosen so the real rule files have something concrete to evaluate.
_EVENTS = [
    {"type": "pass", "timestamp": 10.0 + i, "team": "home", "completed": False} for i in range(12)
] + [
    {"type": "pass", "timestamp": 30.0, "team": "home", "completed": True},
    {"type": "pass", "timestamp": 31.0, "team": "home", "completed": True},
    {"type": "turnover", "timestamp": 33.0, "team": "home", "metadata": {"zone": "buildup"}},
    {"type": "turnover", "timestamp": 35.0, "team": "home", "metadata": {"zone": "buildup"}},
    {"type": "pass", "timestamp": 40.0, "team": "away", "completed": True},
    {"type": "shot", "timestamp": 42.0, "team": "away"},
    {"type": "shot", "timestamp": 60.0, "team": "home"},
]


def test_training_plan_loop_real_pipeline_no_fabrication(qapp, tmp_path, monkeypatch):
    """Full loop: persist events -> generate -> persist plan -> read back.

    Anti-fabrication assertions on the *real* KnowledgeService output:
    every drill referenced in the returned plan must resolve to a drill
    in the on-disk knowledge base, and the plan must be persisted to the
    real database (the old code returned an unpersisted plan).
    """
    _reset_paths(monkeypatch, tmp_path)
    window = _boot_window(qapp)
    try:

        async def main():
            storage = window.storage
            try:
                await storage.initialize()
                match_id = await storage.save_match("E2E training loop", "e2e.mp4", "Reds", "Blues")
                saved = await storage.save_events_bulk(match_id, _EVENTS)
                assert saved == len(_EVENTS), "event seeding failed"

                raw = await window.bridge.generate_training_plan(str(match_id))
                out = json.loads(raw)
                assert "error" not in out, f"loop failed on real data: {out}"
                assert out.get("success") is True, str(out)
                assert out.get("persisted") is True, f"plan not persisted: {out}"

                plan = out["plan"]
                assert plan.get("duration_weeks", 0) >= 1, str(plan)[:200]
                assert plan.get("weeks"), "plan must contain weeks"

                # Anti-fabrication: every drill in the plan resolves in
                # the real knowledge base.
                from kawkab.services.knowledge_service import KnowledgeService

                kb = KnowledgeService()
                await kb.initialize()
                unresolved = set(plan.get("unresolved_drills", []))
                referenced = set()
                for wk in plan.get("weeks", []):
                    for day in wk.get("daily_schedule", []) or wk.get("days", []) or []:
                        sessions = day.get("sessions", []) if isinstance(day, dict) else []
                        for sess in sessions or []:
                            for d in sess.get("drills", []) or []:
                                did = d.get("drill_id") if isinstance(d, dict) else d
                                if did:
                                    referenced.add(str(did))
                for did in referenced:
                    drill = await kb.get_drill(did)
                    assert drill is not None, (
                        f"phantom drill {did!r} rendered in plan — fabrication regression"
                    )
                assert not unresolved, f"unresolved drills leaked silently: {unresolved}"

                # Persistence: the plan survives a "restart" — read it
                # back through a brand-new storage instance.
                plan_id = out.get("plan_id")
                assert plan_id, f"no plan id: {out}"

                def _fresh():
                    from kawkab.services.storage_service import StorageService

                    return StorageService()

                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    fresh = await asyncio.get_event_loop().run_in_executor(pool, _fresh)
                await fresh.initialize()
                try:
                    row = await fresh.get_training_plan(plan_id)
                    assert row is not None, "plan row missing after restart"
                    assert row.get("payload", {}).get("duration_weeks") == plan["duration_weeks"], (
                        "persisted payload diverges from returned plan"
                    )
                    assert row.get("source") == "reasoning_engine"
                finally:
                    await fresh.close()
            finally:
                await storage.close()

        asyncio.run(main())
    finally:
        window.close()
        qapp.processEvents()


def test_training_plan_loop_on_empty_match_is_honest(qapp, tmp_path, monkeypatch):
    """A match with no events produces a plan with zero fabricated diagnoses.

    The old fabricated path always fired R001-R005 regardless of data.
    The real engine on an empty match must return no diagnoses, and the
    response must still be honest about that.
    """
    _reset_paths(monkeypatch, tmp_path)
    window = _boot_window(qapp)
    try:

        async def main():
            storage = window.storage
            try:
                await storage.initialize()
                match_id = await storage.save_match("E2E empty match", "e2e.mp4", "Reds", "Blues")

                raw = await window.bridge.generate_training_plan(str(match_id))
                out = json.loads(raw)
                assert "error" not in out, f"loop failed on empty match: {out}"
                assert out.get("success") is True, str(out)
                assert out.get("diagnosis_count") == 0, (
                    f"fabricated diagnoses on an empty match: {out}"
                )
                assert out.get("unresolved_drills") == []
            finally:
                await storage.close()

        asyncio.run(main())
    finally:
        window.close()
        qapp.processEvents()
