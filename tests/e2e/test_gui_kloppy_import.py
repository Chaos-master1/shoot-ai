"""kloppy import GUI e2e — through the real MainWindow → Bridge →
ImportHandler → services → migrated SQLite stack.

Proves the phase-D descope is closed for real: the same corpus file that
Phase D could only *probe* now round-trips through the bridge into
matches / events / players, and the imported match feeds the exact
downstream consumers the audit demanded — the opposition dossier and the
reasoning engine's training plan — with honesty states intact.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS = PROJECT_ROOT / "data" / "statsbomb_corpus"


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


def _corpus_file() -> Path:
    files = sorted(CORPUS.glob("*.json"))
    assert files, "data/statsbomb_corpus is required for this e2e"
    return files[0]


def test_kloppy_import_feeds_dossier_and_plan(qapp, tmp_path, monkeypatch):
    """Corpus import via bridge → match feeds dossier + training plan."""
    _reset_paths(monkeypatch, tmp_path)
    from kawkab.app import MainWindow

    window = MainWindow()
    try:

        async def main():
            storage = window.storage
            try:
                await storage.initialize()

                # The handler's security validator confines imports to the
                # app's documents area — stage the corpus file there, the
                # same way a user's own vendor export would live.
                from kawkab.core.paths import get_paths

                allowed = get_paths().documents / "vendor_test"
                allowed.mkdir(parents=True, exist_ok=True)
                staged = allowed / _corpus_file().name
                staged.write_bytes(_corpus_file().read_bytes())

                # Import through the real bridge slot.
                raw = await window.bridge.import_kloppy_statsbomb(str(staged), "", "", "", "")
                out = json.loads(raw)
                assert out["success"] is True, str(out)[:400]
                assert out["source"] == "kloppy", str(out)[:200]
                assert out["events_imported"] > 0
                assert out["shots"] > 0
                match_id = out["match_id"]

                # Honest states from the real window: storage has the match.
                matches = await storage.get_all_matches()
                assert any(m["id"] == match_id for m in matches)

                # 1. The imported match feeds the opposition dossier —
                #    the scouting consumer that Phase D could only
                #    report no_data for, is now fed by imported data.
                name = next(m["name"] for m in matches if m["id"] == match_id)
                home_or_away = out["home_team"] if out["home_team"] in name else out["away_team"]
                dossier = json.loads(await window.bridge.get_opponent_dossier(home_or_away))
                assert "error" not in dossier, str(dossier)
                assert dossier["state"] == "ok", str(dossier)[:300]
                assert dossier["provenance"]["matches_analyzed"] >= 1

                # 2. …and the reasoning engine consumes it end to end.
                plan = json.loads(await window.bridge.generate_training_plan(str(match_id)))
                assert "error" not in plan, f"plan failed: {plan}"
                assert plan["success"] is True, str(plan)[:300]
                assert plan["persisted"] is True, str(plan)[:300]
                assert plan["plan_id"] and plan["plan_id"] > 0, str(plan)[:300]

                # 3. Honest metadata: the probe still tells the truth.
                status = json.loads(await window.bridge.get_vendor_import_status())
                assert status["provider_available"] is True
                assert any(c["available"] for c in status["providers"])
            finally:
                await storage.close()

        asyncio.run(main())
    finally:
        window.close()
        qapp.processEvents()
