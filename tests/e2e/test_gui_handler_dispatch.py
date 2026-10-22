"""GUI e2e: one real dispatch slot per post-split handler (Phase C3).

The C1 refactor moved 46 bridge methods into MatchIntelHandler,
PhysicalHandler and DomainHandler. Unit tests verify each handler class
against stub services; this module verifies the *app-level* contract the
user actually exercises:

  real MainWindow boot -> real bridge -> real sqlite (WAL) -> dispatch
  one representative slot per new handler -> parse JSON -> assert on the
  values the UI would render.

Seeding is done through the production StorageService API (save_match /
save_events_bulk / save_acwr), so the analytics below prove the slots
read and aggregate real persisted data -- including the v0.13.2 fixes:
GPS slots must await the async storage (previously a coroutine
serialization error), and xa/pressing reports must normalize
event_type -> type (previously silent zeros on real rows).
"""

from __future__ import annotations

import asyncio
import json

import pytest

pyside6 = pytest.importorskip("PySide6")


@pytest.fixture()
def qapp(monkeypatch, tmp_path):
    """Offscreen QApplication with an isolated environment.

    Same isolation contract as test_gui_boot.qapp (duplicated here rather
    than imported: tests/ has no __init__.py): outward isolation from the
    developer's real HOME/Documents/cache, inward reset of the cached
    paths/settings singletons so the env vars actually take effect.
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
    """Point the paths singleton at this test's tmp docs dir.

    Per-test (not fixture-level) because seeding writes real files under
    get_paths().documents and each test boots a fresh MainWindow.
    """
    monkeypatch.setenv("XDG_DOCUMENTS_DIR", str(tmp_path / "xdg-docs"))
    monkeypatch.setenv("HOME", str(tmp_path))
    import kawkab.core.paths as paths_mod

    if hasattr(paths_mod, "_paths"):
        monkeypatch.setattr(paths_mod, "_paths", None)


def _boot_window(qapp):
    from kawkab.app import MainWindow

    window = MainWindow()
    assert window.bridge._match_intel is not None, "MatchIntelHandler must be wired"
    assert window.bridge._physical is not None, "PhysicalHandler must be wired"
    assert window.bridge._domain is not None, "DomainHandler must be wired"
    return window


def _catapult_csv() -> str:
    return (
        "Time,Speed (m/s),Accel X,Accel Y,Accel Z,Heart Rate,"
        "Distance (m),Player Load,Metabolic Power (W/kg),Lat,Lon\n"
        "00:00:00.000,0.0,0.01,-0.02,0.99,72,0.0,0.0,0.0,51.5,-0.12\n"
        "00:00:00.100,2.5,0.12,-0.15,1.02,75,0.25,0.5,3.2,51.5,-0.12\n"
        "00:00:00.200,5.0,0.25,-0.30,1.10,78,0.50,1.0,6.5,51.5,-0.12\n"
        "00:00:00.300,7.5,0.40,-0.45,1.15,82,0.75,1.5,10.0,51.5,-0.12\n"
    )


def test_match_intel_xa_report_on_real_persisted_events(qapp, tmp_path, monkeypatch):
    """Dispatch get_xa_report through the real bridge over real sqlite rows.

    Pins two things at once: the slot awaits the async storage and the
    handler maps the rows' event_type to the model's "type" convention --
    without that mapping this report is all zeros on any real match.

    Boot order mirrors the real app: MainWindow() constructs (and
    asyncio.run-initializes) storage synchronously, then all async work
    runs in one explicit event loop below.
    """
    _reset_paths(monkeypatch, tmp_path)
    window = _boot_window(qapp)
    try:

        async def main():
            storage = window.storage
            try:
                await storage.initialize()
                match_id = await storage.save_match("C3 e2e fixture", "e2e.mp4", "Reds", "Blues")
                assert match_id > 0
                saved = await storage.save_events_bulk(
                    match_id,
                    [
                        {"type": "pass", "timestamp": 1.0, "team": "home", "completed": True},
                        {"type": "pass", "timestamp": 2.0, "team": "home", "completed": True},
                        {"type": "pass", "timestamp": 3.0, "team": "away", "completed": True},
                    ],
                )
                assert saved == 3

                raw = await window.bridge.get_xa_report(str(match_id))
                report = json.loads(raw)
                assert "error" not in report, f"xa report failed on real data: {report}"
                assert report["home"] > 0, f"home xA must be non-zero: {report}"
                assert report["away"] > 0, f"away xA must be non-zero: {report}"
                assert report["total"] == pytest.approx(report["home"] + report["away"], abs=1e-6)
            finally:
                await storage.close()

        asyncio.run(main())
    finally:
        window.close()
        qapp.processEvents()


def test_pressing_report_uses_real_event_types(qapp, tmp_path, monkeypatch):
    """get_pressing_report must see real persisted traps (tackle/interception).

    Same normalization pin as the xA report, on the pressing analyzer:
    a persisted tackle followed by a shot must register as a trap-to-shot
    conversion, not the all-zero dict the un-normalized rows produced.
    """
    _reset_paths(monkeypatch, tmp_path)
    window = _boot_window(qapp)
    try:

        async def main():
            storage = window.storage
            try:
                await storage.initialize()
                match_id = await storage.save_match(
                    "C3 pressing fixture", "e2e.mp4", "Reds", "Blues"
                )
                await storage.save_events_bulk(
                    match_id,
                    [
                        {"type": "tackle", "timestamp": 10.0, "team": "home"},
                        {"type": "shot", "timestamp": 12.0, "team": "home"},
                        {"type": "pass", "timestamp": 13.0, "team": "away"},
                    ],
                )

                raw = await window.bridge.get_pressing_report(str(match_id))
                report = json.loads(raw)
                assert "error" not in report, f"pressing report failed: {report}"
                assert report["home"]["traps"] == 1.0, f"trap not counted: {report}"
                assert report["home"]["shots_from_traps"] == 1.0, f"conversion missed: {report}"
                assert report["away"]["traps"] == 0.0
            finally:
                await storage.close()

        asyncio.run(main())
    finally:
        window.close()
        qapp.processEvents()


def test_physical_gps_and_acwr_slots_persist_and_read(qapp, tmp_path, monkeypatch):
    """PhysicalHandler end-to-end through the real bridge.

    Round-trips a genuine Catapult CSV through import_gps_file into the
    real sqlite database, then reads it back through get_gps_sessions /
    get_gps_samples; plus save_acwr -> get_player_acwr. Pre-v0.13.2 every
    one of these slots returned a coroutine serialization error against
    the real (async) storage.
    """
    _reset_paths(monkeypatch, tmp_path)
    window = _boot_window(qapp)
    try:

        async def main():
            storage = window.storage
            try:
                await storage.initialize()
                match_id = await storage.save_match(
                    "C3 physical fixture", "e2e.mp4", "Reds", "Blues"
                )
                # gps_sessions.player_id has a real FK to players(id), so
                # create the player through the production API first --
                # the same prerequisite the UI enforces by only offering
                # GPS import for tracked players.
                player_id = await storage.save_player(
                    match_id, {"track_id": 7, "name": "E2E Seven", "team": "Reds"}
                )
                assert player_id > 0

                # A real Catapult export in the (stubbed) documents
                # directory -- exactly where the app's file picker would
                # leave a user's file.
                from kawkab.core.paths import get_paths

                gps_file = get_paths().documents / "catapult_export.csv"
                gps_file.write_text(_catapult_csv(), encoding="utf-8")

                pid = str(player_id)
                import_raw = await window.bridge.import_gps_file(
                    str(match_id), pid, str(gps_file), "match", "catapult"
                )
                import_out = json.loads(import_raw)
                assert import_out.get("success") is True, f"GPS import failed: {import_out}"
                assert import_out["sample_count"] == 4, f"samples lost: {import_out}"

                sessions = json.loads(await window.bridge.get_gps_sessions(str(match_id)))
                assert sessions["success"] is True, str(sessions)
                assert len(sessions["sessions"]) == 1, str(sessions)
                session_id = sessions["sessions"][0]["id"]

                samples = json.loads(await window.bridge.get_gps_samples(str(session_id)))
                assert samples["success"] is True, str(samples)
                assert len(samples["samples"]) == 4, str(samples)
                speeds = {s.get("speed_ms") for s in samples["samples"]}
                assert 7.5 in speeds, f"imported speeds not persisted verbatim: {speeds}"

                summary = json.loads(await window.bridge.get_player_gps_summary(pid))
                assert summary["success"] is True and summary["sessions"], str(summary)

                await storage.save_acwr(
                    player_id, "2026-09-01", acute=550.0, chronic=500.0, acwr=1.1
                )
                acwr = json.loads(await window.bridge.get_player_acwr(pid))
                assert acwr["success"] is True, str(acwr)
                assert acwr["acwr"] and acwr["acwr"][0]["acwr"] == pytest.approx(1.1), str(acwr)
            finally:
                await storage.close()

        asyncio.run(main())
    finally:
        window.close()
        qapp.processEvents()


def test_domain_setpiece_slot(qapp, tmp_path, monkeypatch):
    """Dispatch analyze_setpieces through the real bridge.

    Two home corners must come back as total_corners == 2 for home and 0
    for away, proving the domain handler -> SetPieceService path end to
    end.
    """
    _reset_paths(monkeypatch, tmp_path)
    window = _boot_window(qapp)
    try:
        events = [
            {
                "set_piece_type": "corner",
                "minute": 12,
                "second": 30,
                "team": "Reds",
                "delivery_x": 90.0,
                "delivery_y": 2.0,
                "delivery_style": "inswinging",
                "first_contact_x": 94.0,
                "first_contact_y": 8.0,
                "outcome": "shot",
            },
            {
                "set_piece_type": "corner",
                "minute": 40,
                "second": 5,
                "team": "Reds",
                "delivery_x": 90.0,
                "delivery_y": 66.0,
                "delivery_style": "outswinging",
                "first_contact_x": 93.0,
                "first_contact_y": 60.0,
                "outcome": "clearance",
            },
        ]
        raw = asyncio.run(window.bridge.analyze_setpieces(json.dumps(events), "Reds"))
        report = json.loads(raw)
        assert "error" not in report, f"setpiece analysis failed: {report}"
        assert report["home"]["total_corners"] == 2, str(report)
        assert report["away"]["total_corners"] == 0, str(report)
        assert report["home"]["shots_per_corner"] == pytest.approx(0.5), str(report)
    finally:
        window.close()
        qapp.processEvents()
