"""GUI e2e: the sports-science daily loop (Phase C).

Through the real application stack (MainWindow -> bridge -> sqlite):

  morning wellness submissions -> squad readiness summary (conversation
  flags only) -> per-player triangulated load state (sRPE ACWR from real
  stored RPE rows; GPS never blended in) -> squad overview merging hard
  availability gates with advisory readiness flags.

Honesty pins: a player with no medical blockers is *available* regardless
of load flags (flags are advisory), readiness flags never block, and
provenance travels with every summary.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pyside6 = pytest.importorskip("PySide6")


@pytest.fixture()
def qapp(monkeypatch, tmp_path):
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


def test_sports_science_daily_loop_through_bridge(qapp, tmp_path, monkeypatch):
    _reset_paths(monkeypatch, tmp_path)
    from kawkab.app import MainWindow

    window = MainWindow()
    try:

        async def main():
            storage = window.storage
            try:
                await storage.initialize()

                # --- morning wellness ritual: three submissions ---------
                for pid, mood in (("1", 4), ("2", 1), ("3", 3)):
                    saved = json.loads(
                        await window.bridge.save_wellness(
                            pid,
                            json.dumps(
                                {
                                    "record_date": "2026-09-19",
                                    "sleep_quality": int(mood),
                                    "fatigue": 3,
                                    "soreness": 3,
                                    "stress": 3,
                                    "mood": int(mood),
                                    "source": "e2e",
                                }
                            ),
                        )
                    )
                    assert saved.get("success") is True, str(saved)

                # --- readiness summary: one player flagged to talk to ---
                readiness = json.loads(await window.bridge.get_squad_readiness("2026-09-19"))
                assert readiness.get("success") is True, str(readiness)
                assert readiness["submissions"] == 3
                flagged_ids = [f["player_id"] for f in readiness["flagged_for_conversation"]]
                assert flagged_ids == [2], f"low-wellness player must be flagged: {readiness}"
                assert "not a diagnosis" in readiness["provenance"]["note"]

                # --- per-player load state (no sRPE history: no_data) ---
                load = json.loads(await window.bridge.get_player_load_state("2"))
                assert load.get("success") is True, str(load)
                assert load["srpe"]["band"] == "no_data"  # honest, not zero
                assert load["srpe"]["source"] == "player_reported_sRPE"
                assert load["gps"]["source"] == "device_measured_GPS"
                assert load["flags"] == []

                # --- maturation estimate with validity provenance -------
                mat = json.loads(
                    await window.bridge.estimate_maturity_offset(
                        json.dumps(
                            {
                                "age_years": 14.0,
                                "standing_height_cm": 165.0,
                                "sitting_height_cm": 82.0,
                                "sex": "male",
                            }
                        )
                    )
                )
                assert mat.get("success") is True, str(mat)
                assert abs(mat["maturity_offset"] - (-0.72)) < 0.05
                assert any("±0.59" in n for n in mat["provenance"]["validity_notes"])

                # female estimate must be an honest not_implemented
                mat_f = json.loads(
                    await window.bridge.estimate_maturity_offset(
                        json.dumps(
                            {
                                "age_years": 13.0,
                                "standing_height_cm": 158.0,
                                "sitting_height_cm": 80.0,
                                "sex": "female",
                            }
                        )
                    )
                )
                assert mat_f["classification"] == "not_implemented"
                assert "not verifiable" in mat_f["provenance"]["error"]

                # --- squad overview: availability % + readiness ---------
                overview = json.loads(
                    await window.bridge.get_squad_overview("2026-09-19", json.dumps([1, 2, 3]))
                )
                assert overview.get("success") is True, str(overview)
                assert overview["availability_pct"] == 100.0, (
                    f"no medical blockers -> everyone available (flags are advisory): {overview}"
                )
                assert overview["readiness"]["flagged_for_conversation"][0]["player_id"] == 2
                assert overview["provenance"]["availability"].startswith("hard gates")
                assert overview["provenance"]["readiness"].startswith("advisory")
            finally:
                await storage.close()

        asyncio.run(main())
    finally:
        window.close()
        qapp.processEvents()
