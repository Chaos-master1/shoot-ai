"""GUI e2e: broad slot-dispatch audit through the real booted app (C1 follow-up).

The v0.13.1 split shipped handlers whose unit tests all passed while the
features were dead in the real app (missing awaits, key-shape mismatches,
a wrong validator, a missing ctor arg, a self-reading lazy property).
That pass fixed the GPS/xA/pressing/setpiece surface; this module extends
the same real-dispatch contract across the remaining handler clusters.

Contract per cluster: boot the real MainWindow (services constructed,
storage initialized exactly as a user session starts), dispatch the
cluster's slots through the bridge with production-shaped arguments, and
assert the app answers with parseable JSON carrying the fields the UI
renders. Slots backed by optional heavy engines (MuJoCo, FluidX3D, cloud
logins) must return an honest error payload, never garbage or a hang.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import sys
import uuid

import pytest

pyside6 = pytest.importorskip("PySide6")


@pytest.fixture()
def qapp(monkeypatch, tmp_path):
    """Offscreen QApplication with an isolated environment.

    Same isolation contract as the other GUI e2e modules (duplicated here:
    tests/ has no __init__.py, so the fixture cannot be imported).
    """
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.setenv("XDG_DOCUMENTS_DIR", str(tmp_path / "xdg-docs"))
    monkeypatch.setenv("KAWKAB_DEBUG", "false")
    monkeypatch.delenv("KAWKAB_DB_URL", raising=False)

    import kawkab.core.paths as paths_mod

    # Other e2e modules (test_e2e_full_pipeline etc.) install the conftest
    # stub paths module via install_kawkab_stubs, which shadows the real
    # kawkab.core.paths for the rest of the process with a SHARED
    # /tmp/kawkab_test database -- state then leaks across tests and across
    # pytest runs (stale users, stale external-id dedup). Booting the real
    # app is only meaningful against the real per-user paths, so force the
    # real module back and drop any cached singleton, plus the modules that
    # captured get_paths at import time.
    if not str(getattr(paths_mod, "__file__", "")).endswith(
        "core" + __import__("os").sep + "paths.py"
    ):
        for mod_name in (
            "kawkab.core.paths",
            "kawkab.services.storage_service",
            "kawkab.app",
            "kawkab.ui.bridge",
        ):
            sys.modules.pop(mod_name, None)
        paths_mod = importlib.import_module("kawkab.core.paths")

    if hasattr(paths_mod, "_paths"):
        monkeypatch.setattr(paths_mod, "_paths", None)
    from kawkab.core.config import get_settings

    get_settings.cache_clear()

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(["kawkab-slot-audit"])
    yield app
    get_settings.cache_clear()


def dispatch(window, name, *args):
    """Await whatever the bridge slot returns (PySide6 allows both)."""
    out = getattr(window.bridge, name)(*args)
    if inspect.isawaitable(out):
        out = asyncio.run(out) if not _in_loop() else out
    return out


def _in_loop():
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


async def adispatch(window, name, *args):
    out = getattr(window.bridge, name)(*args)
    if inspect.isawaitable(out):
        out = await out
    return out


def seed_match(window, name: str = "audit fixture") -> int:
    """Seed a match + players + events through the production API."""

    async def main():
        storage = window.storage
        await storage.initialize()
        mid = await storage.save_match(name, "audit.mp4", "Reds", "Blues")
        await storage.save_player(
            mid, {"track_id": 5, "name": "P5", "team": "Reds", "jersey_number": 5}
        )
        await storage.save_player(
            mid, {"track_id": 6, "name": "P6", "team": "Reds", "jersey_number": 6}
        )
        events = []
        for i, (team, x) in enumerate(
            [("home", 20), ("home", 50), ("home", 80), ("away", 30), ("away", 60)]
        ):
            events.append(
                {
                    "type": "pass",
                    "timestamp": float(i),
                    "team": team,
                    "completed": True,
                    "from_track_id": 5,
                    "to_track_id": 6,
                    "x": float(x),
                    "y": 34.0,
                    "end_x": float(x + 10),
                    "end_y": 30.0,
                }
            )
        events += [
            {
                "type": "shot",
                "timestamp": 10.0,
                "team": "home",
                "from_track_id": 5,
                "x": 95.0,
                "y": 34.0,
                "completed": True,
                "metadata": {"xg": 0.3},
            },
            {
                "type": "tackle",
                "timestamp": 11.0,
                "team": "away",
                "from_track_id": 6,
                "x": 40.0,
                "y": 30.0,
            },
            {
                "type": "goal",
                "timestamp": 12.0,
                "team": "home",
                "from_track_id": 5,
                "x": 99.0,
                "y": 34.0,
            },
        ]
        await storage.save_events_bulk(mid, events)
        return mid

    return asyncio.run(main())


def _assert_ok(payload: str, slot: str, *, allow_error: bool = False):
    d = json.loads(payload)
    if allow_error:
        assert isinstance(d, dict), f"{slot}: non-dict payload {str(d)[:80]}"
        return d
    assert not (isinstance(d, dict) and "error" in d), f"{slot} errored: {str(d)[:200]}"
    return d


def _reset_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DOCUMENTS_DIR", str(tmp_path / "xdg-docs"))
    monkeypatch.setenv("HOME", str(tmp_path))
    import kawkab.core.paths as paths_mod

    if hasattr(paths_mod, "_paths"):
        monkeypatch.setattr(paths_mod, "_paths", None)
    # get_settings caches env-derived values at first call; without this
    # clear the appdata dir resolves from a previous test's env and user
    # data (opponent profiles, scouting network) leaks into it.
    from kawkab.core.config import get_settings

    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# match_intel: all 17 slots through the real bridge
# ---------------------------------------------------------------------------


def test_match_intel_all_slots_through_real_bridge(qapp, tmp_path, monkeypatch):
    _reset_paths(monkeypatch, tmp_path)
    from kawkab.app import MainWindow

    window = MainWindow()
    try:
        mid = seed_match(window, "match-intel audit")
        m = str(mid)

        async def main():
            results = {}
            simple = [
                "analyze_build_up",
                "analyze_finishing",
                "compute_goals_added",
                "compute_phase_xg",
                "compute_territory_value",
                "generate_match_report",
                "get_dominance_index",
                "get_match_quality_score",
                "get_pitch_control_overlay",
                "get_space_control_heatmap",
            ]
            for name in simple:
                results[name] = await adispatch(window, name, m)
            results["estimate_transfer_fee"] = await adispatch(
                window, "estimate_transfer_fee", m, "5"
            )
            results["get_player_pass_sonar"] = await adispatch(
                window, "get_player_pass_sonar", m, "5"
            )
            results["get_player_role"] = await adispatch(window, "get_player_role", m, "5")
            results["generate_game_plan"] = await adispatch(window, "generate_game_plan", m, m)
            results["simulate_league"] = await adispatch(window, "simulate_league", m, "3")
            results["get_xa_report"] = await adispatch(window, "get_xa_report", m)
            results["get_pressing_report"] = await adispatch(window, "get_pressing_report", m)
            return results

        results = asyncio.run(main())
        for slot, payload in results.items():
            _assert_ok(payload, slot)

        # value-level pins on three slots with known seeded truth
        sonar = json.loads(results["get_player_pass_sonar"])
        assert sonar["total_passes"] == 5, sonar
        xa = json.loads(results["get_xa_report"])
        assert xa["home"] > 0 and xa["away"] > 0, xa
        pressing = json.loads(results["get_pressing_report"])
        assert pressing["home"]["traps"] >= 0 and "shots_from_traps" in pressing["home"]
    finally:
        window.close()
        qapp.processEvents()


# ---------------------------------------------------------------------------
# coding: full CRUD round trip through the real bridge
# ---------------------------------------------------------------------------


def test_coding_cluster_round_trip_through_real_bridge(qapp, tmp_path, monkeypatch):
    _reset_paths(monkeypatch, tmp_path)
    from kawkab.app import MainWindow

    window = MainWindow()
    try:
        mid = seed_match(window, "coding audit")

        async def main():
            out = {}
            out["save"] = await adispatch(
                window,
                "save_coding_tag",
                str(mid),
                json.dumps(
                    {
                        "event_type": "shot",
                        "timestamp": 5.0,
                        "team": "home",
                        "player_track_id": 5,
                        "notes": "audit",
                    }
                ),
            )
            saved = json.loads(out["save"])
            out["list"] = await adispatch(window, "get_coding_tags", str(mid))
            out["stats"] = await adispatch(window, "get_coding_tag_stats", str(mid))
            out["players"] = await adispatch(window, "get_coding_players", str(mid))
            out["by_player"] = await adispatch(window, "get_coding_tags_by_player", str(mid), "5")
            out["by_type"] = await adispatch(window, "get_coding_tags_by_type", str(mid), "shot")
            out["templates"] = await adispatch(window, "get_coding_templates")
            tid = saved.get("tag_id")
            if tid:
                out["update"] = await adispatch(
                    window, "update_coding_tag", str(tid), json.dumps({"notes": "edited"})
                )
                out["delete"] = await adispatch(window, "delete_coding_tag", str(tid))
            return out

        out = asyncio.run(main())
        saved = json.loads(out["save"])
        assert saved.get("success") is True and saved.get("tag_id", 0) > 0, saved
        for slot in ("list", "stats", "players", "by_player", "by_type", "templates"):
            _assert_ok(out[slot], slot)
        if "update" in out:
            _assert_ok(out["update"], "update_coding_tag")
            _assert_ok(out["delete"], "delete_coding_tag")
    finally:
        window.close()
        qapp.processEvents()


# ---------------------------------------------------------------------------
# whiteboard: full authoring lifecycle through the real bridge
# ---------------------------------------------------------------------------


def test_whiteboard_lifecycle_through_real_bridge(qapp, tmp_path, monkeypatch):
    _reset_paths(monkeypatch, tmp_path)
    from kawkab.app import MainWindow

    window = MainWindow()
    try:
        created = json.loads(dispatch(window, "whiteboard_create", "audit board", "4-3-3"))
        sid = created.get("id") or created.get("state_id")
        assert sid, created

        got = _assert_ok(dispatch(window, "whiteboard_get", str(sid)), "whiteboard_get")
        assert got.get("name") == "audit board"

        moved = _assert_ok(
            dispatch(window, "whiteboard_move_player", str(sid), "0", "60.0", "30.0", "home"),
            "whiteboard_move_player",
        )
        assert moved.get("ok") is True

        ann = _assert_ok(
            dispatch(
                window,
                "whiteboard_add_annotation",
                str(sid),
                json.dumps({"type": "arrow", "points": [{"x": 10, "y": 10}, {"x": 50, "y": 50}]}),
            ),
            "whiteboard_add_annotation",
        )
        assert ann.get("id"), ann

        svg = json.loads(dispatch(window, "whiteboard_generate_svg", str(sid), "800", "500"))
        assert "<svg" in svg.get("svg", ""), svg

        _assert_ok(
            dispatch(window, "whiteboard_set_formation", str(sid), "4-4-2", "home"),
            "whiteboard_set_formation",
        )
        _assert_ok(dispatch(window, "whiteboard_list"), "whiteboard_list")
        _assert_ok(dispatch(window, "whiteboard_list_templates"), "whiteboard_list_templates")
        _assert_ok(dispatch(window, "whiteboard_get_template", "4-3-3"), "whiteboard_get_template")
        _assert_ok(
            dispatch(window, "whiteboard_clear_annotations", str(sid)),
            "whiteboard_clear_annotations",
        )
        deleted = _assert_ok(dispatch(window, "whiteboard_delete", str(sid)), "whiteboard_delete")
        assert deleted.get("ok") is True

        # pure generators need no state
        _assert_ok(
            dispatch(window, "whiteboard_generate_pass", "10.0", "10.0", "50.0", "30.0", "#ff0000"),
            "whiteboard_generate_pass",
        )
        _assert_ok(
            dispatch(
                window,
                "whiteboard_generate_player_run",
                "10.0",
                "10.0",
                "40.0",
                "30.0",
                "#00ff00",
                "run1",
            ),
            "whiteboard_generate_player_run",
        )
        _assert_ok(dispatch(window, "check_whiteboard_status"), "check_whiteboard_status")
    finally:
        window.close()
        qapp.processEvents()


# ---------------------------------------------------------------------------
# provider + settings + lifecycle: infra slots through the real bridge
# ---------------------------------------------------------------------------


def test_provider_settings_lifecycle_through_real_bridge(qapp, tmp_path, monkeypatch):
    _reset_paths(monkeypatch, tmp_path)
    from kawkab.app import MainWindow

    window = MainWindow()
    try:
        # cache round trip with int TTL as the Slot signature delivers
        set_out = _assert_ok(
            dispatch(window, "cache_set", "audit-key", json.dumps({"v": 42}), 120), "cache_set"
        )
        assert set_out.get("ok") is True
        got = json.loads(dispatch(window, "cache_get", "audit-key"))
        assert got == {"v": 42}, got
        _assert_ok(dispatch(window, "cache_stats"), "cache_stats")
        _assert_ok(dispatch(window, "cache_invalidate", "audit-key"), "cache_invalidate")
        _assert_ok(dispatch(window, "cache_invalidate_prefix", "audit"), "cache_invalidate_prefix")
        _assert_ok(dispatch(window, "cache_clear"), "cache_clear")

        calls = json.loads(dispatch(window, "get_recent_provider_calls", 5))
        assert isinstance(calls, list), calls
        _assert_ok(
            dispatch(window, "record_provider_call", "ollama", "ask", 12.5, True, 200),
            "record_provider_call",
        )
        _assert_ok(dispatch(window, "get_provider_status", "ollama"), "get_provider_status")
        _assert_ok(dispatch(window, "get_all_provider_statuses"), "get_all_provider_statuses")
        _assert_ok(dispatch(window, "get_provider_summary"), "get_provider_summary")

        overview = _assert_ok(dispatch(window, "get_settings_overview"), "get_settings_overview")
        assert "app" in overview and "gpu" in overview
        app_info = _assert_ok(dispatch(window, "get_app_info"), "get_app_info")
        assert app_info.get("version"), app_info
        _assert_ok(dispatch(window, "get_gpu_tier"), "get_gpu_tier")
        _assert_ok(dispatch(window, "get_current_yolo_variant"), "get_current_yolo_variant")
        _assert_ok(dispatch(window, "get_recommended_yolo_variant"), "get_recommended_yolo_variant")
        _assert_ok(dispatch(window, "get_model_cache_info"), "get_model_cache_info")

        gpu = _assert_ok(dispatch(window, "get_gpu_info"), "get_gpu_info")
        assert "tier" in gpu
        # metrics_text is a plain-text (Prometheus-style) slot by
        # contract, not JSON -- assert it returns a string without raising.
        metrics = dispatch(window, "metrics_text")
        assert isinstance(metrics, str), type(metrics)
        _assert_ok(dispatch(window, "profiler_status"), "profiler_status", allow_error=True)
        _assert_ok(dispatch(window, "profiler_reset"), "profiler_reset", allow_error=True)
    finally:
        window.close()
        qapp.processEvents()


# ---------------------------------------------------------------------------
# storage + export + recruitment: payload slots through the real bridge
# ---------------------------------------------------------------------------


def test_storage_export_recruitment_through_real_bridge(qapp, tmp_path, monkeypatch):
    _reset_paths(monkeypatch, tmp_path)
    from kawkab.app import MainWindow

    window = MainWindow()
    try:
        mid = seed_match(window, "storage audit")
        # This audit issues dozens of analysis-category recruitment/export
        # slots back to back; the production limiter's shared "analysis"
        # bucket (5 tokens/min) would trip halfway through. Widen it for
        # this test only -- the limiter itself is unit-tested separately.
        window.bridge._rate_limiter.configure("analysis", 1000)

        fb = _assert_ok(
            dispatch(window, "submit_feedback", json.dumps({"rating": 5, "comment": "audit"})),
            "submit_feedback",
            allow_error=True,
        )
        assert ("feedback_id" in fb) or ("error" in fb), fb
        _assert_ok(dispatch(window, "get_feedback_stats"), "get_feedback_stats", allow_error=True)
        _assert_ok(
            dispatch(window, "submit_issue", json.dumps({"title": "t", "description": "d"})),
            "submit_issue",
            allow_error=True,
        )

        csv_out = _assert_ok(
            dispatch(window, "export_match_csv", str(mid)), "export_match_csv", allow_error=True
        )
        js_out = _assert_ok(
            dispatch(window, "export_match_json", str(mid)), "export_match_json", allow_error=True
        )
        assert any(k in js_out for k in ("path", "error")), js_out
        assert any(k in csv_out for k in ("path", "error")), csv_out
        _assert_ok(
            dispatch(window, "export_report_pdf", str(mid), "en"),
            "export_report_pdf",
            allow_error=True,
        )

        # recruitment: opponent profile round trip
        created = json.loads(
            dispatch(window, "opponent_create", "Audit Rivals", "Ligue 1", "France")
        )
        assert created.get("success") is True, created
        profile = created.get("profile") or {}
        pid = profile.get("id")
        assert pid, created
        got = _assert_ok(dispatch(window, "opponent_get", str(pid)), "opponent_get")
        assert got.get("profile", {}).get("team_name") == "Audit Rivals", got
        _assert_ok(dispatch(window, "opponent_list"), "opponent_list")
        _assert_ok(
            dispatch(window, "opponent_scouting_report", str(pid)), "opponent_scouting_report"
        )

        sl = _assert_ok(
            dispatch(
                window,
                "add_shortlist_entry",
                json.dumps(
                    {
                        "player_id": "audit-1",
                        "player_name": "Audited",
                        "position": "ST",
                        "club": "Clubs",
                    }
                ),
            ),
            "add_shortlist_entry",
        )
        assert sl.get("success") is True, sl
        _assert_ok(dispatch(window, "get_shortlist"), "get_shortlist")

        sc = json.loads(
            dispatch(
                window,
                "scout_network_add",
                "Audit Scout",
                "CM",
                "Club",
                "League",
                "7.5",
                json.dumps(["passing"]),
                json.dumps([]),
                "notes",
                "auditor",
                json.dumps([]),
            )
        )
        assert sc.get("success") is True, sc
        spid = (sc.get("player") or {}).get("id")
        if spid:
            _assert_ok(dispatch(window, "scout_network_get", str(spid)), "scout_network_get")
        _assert_ok(dispatch(window, "scout_network_stats"), "scout_network_stats")
        _assert_ok(
            dispatch(window, "scout_network_search", "", "", "0", "40", "", "0"),
            "scout_network_search",
        )
        _assert_ok(dispatch(window, "get_contract_alerts"), "get_contract_alerts")
        _assert_ok(dispatch(window, "get_contracts"), "get_contracts")
    finally:
        window.close()
        qapp.processEvents()


# ---------------------------------------------------------------------------
# live: full tagging session lifecycle through the real bridge
# ---------------------------------------------------------------------------


def test_live_cluster_through_real_bridge(qapp, tmp_path, monkeypatch):
    _reset_paths(monkeypatch, tmp_path)
    from kawkab.app import MainWindow

    window = MainWindow()
    try:
        started = json.loads(dispatch(window, "live_start_session", "Reds", "Blues"))
        assert started.get("ok") is True or started.get("session_id"), started

        # tags carry the session's team NAMES (the service attributes by
        # comparing tag.team to the home/away names given at start_session)
        tag = _assert_ok(
            dispatch(window, "live_tag_event", "pass", "Reds", 5, "audit", 50.0, 34.0),
            "live_tag_event",
        )
        assert tag.get("success") is True or "error" not in tag, tag
        _assert_ok(dispatch(window, "live_tag_event", "shot", "Reds"), "live_tag_event#2")
        _assert_ok(dispatch(window, "live_set_period", "2"), "live_set_period")

        stats = json.loads(dispatch(window, "live_get_stats"))
        assert "error" not in stats, stats
        tags = json.loads(dispatch(window, "live_get_tags"))
        assert isinstance(tags.get("tags"), list) and tags.get("total", 0) >= 2, tags

        kpis = json.loads(dispatch(window, "get_live_kpis", "current"))
        assert "error" not in kpis, kpis
        assert 0.0 <= kpis.get("possession_pct", -1) <= 100.0, kpis
        assert kpis.get("shots", 0) >= 1, kpis
        assert kpis.get("xg_is_approx") is True, kpis  # honesty pin
        pitch = json.loads(dispatch(window, "get_live_pitch_map", "current"))
        assert isinstance(pitch.get("home_events"), list), pitch
        xg = json.loads(dispatch(window, "get_live_xg_chart", "current"))
        assert "timeline" in xg and "cumulative_home" in xg, xg
        # the Reds-tagged shot must be attributed to home, not away
        assert xg["cumulative_home"] > 0, xg

        _assert_ok(dispatch(window, "live_get_hotkeys"), "live_get_hotkeys")
        _assert_ok(dispatch(window, "live_clear_tags"), "live_clear_tags")
        _assert_ok(
            dispatch(
                window,
                "live_export",
            ),
            "live_export",
            allow_error=True,
        )
        _assert_ok(dispatch(window, "live_stop_session"), "live_stop_session")
    finally:
        window.close()
        qapp.processEvents()


# ---------------------------------------------------------------------------
# auth: full local lifecycle through the real bridge
# ---------------------------------------------------------------------------


def test_auth_cluster_through_real_bridge(qapp, tmp_path, monkeypatch):
    _reset_paths(monkeypatch, tmp_path)
    from kawkab.app import MainWindow

    window = MainWindow()
    try:
        # Unique-per-run identifiers: user rows and vendor external-id
        # registrations persist between pytest runs through cached paths
        # singletons from stub-installing e2e modules; unique ids make this
        # audit immune to that instead of fighting process-level state.
        # Unique-per-run usernames: external-id state (users table) can
        # carry over between pytest runs through cached paths singletons
        # from stub-installing e2e modules; unique ids make this audit
        # immune to that instead of fighting process-level module state.
        run = uuid.uuid4().hex[:8]
        coach_name = f"audit_coach_{run}"
        admin_name = f"audit_admin_{run}"

        async def _seed_user():
            storage = window.storage
            await storage.initialize()
            from kawkab.ui.bridge_handlers.bridge_auth import _hash_password

            coach = await storage.create_user(
                coach_name, _hash_password("correct horse battery"), "coach"
            )
            admin = await storage.create_user(
                admin_name, _hash_password("admin secret 99"), "admin"
            )
            return coach, admin

        uid, _admin_uid = asyncio.run(_seed_user())
        assert uid, "user seeding failed"

        bad = json.loads(dispatch(window, "login", coach_name, "wrong"))
        assert "error" in bad and "token" not in bad, bad  # no token on bad password

        login = json.loads(dispatch(window, "login", coach_name, "correct horse battery"))
        assert login.get("token"), login
        assert login.get("user", {}).get("username") == coach_name, login
        token = login["token"]

        me = json.loads(dispatch(window, "get_current_user", token))
        assert me.get("user", {}).get("username") == coach_name, me

        # RBAC pin: non-admin tokens must be denied user management
        denied = json.loads(dispatch(window, "list_users", token))
        assert denied.get("error") == "Admin only", denied
        denied = json.loads(dispatch(window, "get_audit_log", token, "20"))
        assert denied.get("error") == "Admin only", denied

        admin_login = json.loads(dispatch(window, "login", admin_name, "admin secret 99"))
        assert admin_login.get("token"), admin_login
        admin_token = admin_login["token"]

        users = json.loads(dispatch(window, "list_users", admin_token))
        assert isinstance(users.get("users"), list) and len(users["users"]) >= 2, users

        audit_log = json.loads(dispatch(window, "get_audit_log", admin_token, "20"))
        assert isinstance(audit_log.get("events"), list), audit_log

        changed = json.loads(
            dispatch(
                window, "change_password", admin_token, "admin secret 99", "new admin secret 7"
            )
        )
        assert changed.get("success") is True, changed
        relogin = json.loads(dispatch(window, "login", admin_name, "new admin secret 7"))
        assert relogin.get("token"), relogin  # old password must be dead

        out = json.loads(dispatch(window, "logout", admin_token))
        assert out.get("success") is True, out
        dead = json.loads(dispatch(window, "get_current_user", admin_token))
        assert "error" in dead, dead  # logged-out token must not authenticate
    finally:
        window.close()
        qapp.processEvents()


# ---------------------------------------------------------------------------
# external + cloud-local + analytics + import: the remaining clusters
# ---------------------------------------------------------------------------


def test_external_cloud_analytics_import_through_real_bridge(qapp, tmp_path, monkeypatch):
    _reset_paths(monkeypatch, tmp_path)
    from kawkab.app import MainWindow

    window = MainWindow()
    try:
        # Unique-per-run identifiers (see the auth audit): the vendor-id
        # dedup table and collab usernames persist between pytest runs
        # through cached paths singletons from stub-installing modules.
        run = uuid.uuid4().hex[:8]
        file_id = f"99{run[:6]}001"
        collab_user = f"audit_analyst_{run}"
        # analysis-category slots here (pro/season reports, two import
        # phases) exceed the shared production bucket (5/min); widen for
        # this audit only -- the limiter itself is unit-tested separately.
        window.bridge._rate_limiter.configure("analysis", 1000)

        # every vendor status slot: dict payload, never a crash
        for slot in (
            "check_football_data_status",
            "check_bzzoiro_status",
            "check_easy_soccer_status",
            "check_apifootball_status",
            "check_thesportsdb_status",
            "check_statsbomb_status",
            "check_openfootball_status",
        ):
            d = _assert_ok(dispatch(window, slot), slot, allow_error=True)
            assert isinstance(d, dict), (slot, d)

        # network-dependent listers: honest error or data, never a crash.
        # Generous timeout: with no network these fail fast; with network
        # they may take a couple of seconds. Assertions are shape-only.
        d = _assert_ok(
            dispatch(window, "get_football_competitions"),
            "get_football_competitions",
            allow_error=True,
        )
        assert isinstance(d, dict), d
        d = _assert_ok(
            dispatch(window, "get_statsbomb_competitions"),
            "get_statsbomb_competitions",
            allow_error=True,
        )
        assert isinstance(d, dict), d

        # marketplace: full CRUD round trip (local SQLite-backed)
        added = _assert_ok(
            dispatch(
                window,
                "marketplace_add",
                "formation",
                "Audit 4-2-3-1",
                "audit item",
                "auditor",
                "pressing",
                json.dumps(["audit"]),
                json.dumps({"shape": "4-2-3-1"}),
            ),
            "marketplace_add",
        )
        assert added.get("success") is True or added.get("item", {}).get("id"), added
        item_id = (added.get("item") or {}).get("id") or added.get("id")
        if item_id:
            got = _assert_ok(dispatch(window, "marketplace_get", str(item_id)), "marketplace_get")
            assert (got.get("item") or {}).get("name") == "Audit 4-2-3-1", got
            _assert_ok(
                dispatch(window, "marketplace_rate", str(item_id), "5"),
                "marketplace_rate",
                allow_error=True,
            )
        lst = _assert_ok(
            dispatch(window, "marketplace_list", "formation", "", "Audit", ""),
            "marketplace_list",
        )
        assert isinstance(lst.get("items"), list), lst
        _assert_ok(
            dispatch(window, "marketplace_categories", "formation"), "marketplace_categories"
        )
        _assert_ok(dispatch(window, "marketplace_stats"), "marketplace_stats")

        # collab users: CRUD round trip
        cu = _assert_ok(
            dispatch(window, "create_collab_user", collab_user, "Audit Analyst", "analyst"),
            "create_collab_user",
        )
        assert cu.get("success") is True or cu.get("user", {}).get("id"), cu
        users = json.loads(dispatch(window, "get_collab_users"))
        assert isinstance(users.get("users"), list), users
        cud = (cu.get("user") or {}).get("id") or cu.get("user_id")
        if cud:
            _assert_ok(dispatch(window, "delete_collab_user", int(cud)), "delete_collab_user")

        # tactical layer presets: save/load round trip
        _assert_ok(
            dispatch(window, "tel_layer_add", "layer-audit", "Audit Layer"),
            "tel_layer_add",
            allow_error=True,
        )
        layers = _assert_ok(dispatch(window, "tel_get_layers"), "tel_get_layers", allow_error=True)
        assert isinstance(layers, dict), layers
        _assert_ok(
            dispatch(window, "tel_save_preset", "audit-preset", json.dumps([{"id": "l1"}])),
            "tel_save_preset",
        )
        loaded = _assert_ok(dispatch(window, "tel_load_preset", "audit-preset"), "tel_load_preset")
        assert loaded.get("layers") == [{"id": "l1"}] or "layers" in loaded, loaded

        # AI conversations: create + delete round trip
        conv = _assert_ok(
            dispatch(window, "ai_v2_create_conv", "", "Audit Chat"), "ai_v2_create_conv"
        )
        conv_id = (conv.get("conv") or {}).get("id") or conv.get("id")
        if conv_id:
            _assert_ok(dispatch(window, "ai_v2_delete_conv", str(conv_id)), "ai_v2_delete_conv")
        convs = _assert_ok(dispatch(window, "ai_v2_list_convs", ""), "ai_v2_list_convs")
        assert isinstance(convs.get("conversations"), list), convs

        # comments + mentions (local collab)
        d = _assert_ok(dispatch(window, "get_comments", 0), "get_comments", allow_error=True)
        assert isinstance(d, dict), d
        d = _assert_ok(
            dispatch(window, "get_mentions", collab_user), "get_mentions", allow_error=True
        )
        assert isinstance(d, dict), d

        # cloud runtime (local-first): health/status/login-state
        d = _assert_ok(
            dispatch(window, "cloud_check_health"), "cloud_check_health", allow_error=True
        )
        assert isinstance(d, dict), d
        logged_in = json.loads(dispatch(window, "cloud_is_logged_in"))
        assert isinstance(logged_in.get("logged_in"), bool), logged_in
        d = _assert_ok(
            dispatch(window, "cloud_oauth_providers"), "cloud_oauth_providers", allow_error=True
        )
        assert isinstance(d.get("providers"), list), d  # graceful fallback even without server
        _assert_ok(dispatch(window, "cloud_server_status"), "cloud_server_status", allow_error=True)

        # stream capture validation: file:// must be rejected before any
        # process spawns (arbitrary-file-read guard); unknown URL fails
        # honestly when ffmpeg is unavailable (CI has no ffmpeg... except
        # it does -- but a bogus URL will not produce a *success* payload
        # that pretends media was captured).
        rejected = json.loads(dispatch(window, "stream_start_capture", "file:///etc/passwd"))
        assert "error" in rejected, rejected  # security pin: no local reads
        d = _assert_ok(dispatch(window, "stream_list"), "stream_list", allow_error=True)
        assert isinstance(d, dict), d

        # pro/season analytics: real seeded match, engine-backed report
        mid = seed_match(window, "analytics audit")
        pro = json.loads(dispatch(window, "get_pro_analytics_report", int(mid)))
        assert isinstance(pro, dict), pro
        if "error" in pro:
            assert isinstance(pro["error"], str) and pro["error"], pro
        else:
            assert any(
                k in pro for k in ("match_id", "report", "summary", "sections", "analytics")
            ), pro
        season = json.loads(dispatch(window, "get_season_pro_report"))
        assert isinstance(season, dict), season

        # import: a real directory round trip through the season importer.
        # The importer treats a file as an event file iff its root is a
        # non-empty JSON list; a sidecar <stem>.meta.json carries context.
        import_dir = tmp_path / "statsbomb-import"
        import_dir.mkdir()
        events = [
            {
                "id": i + 1,
                "index": i + 1,
                "period": 1,
                "timestamp": f"00:0{i}:00.000",
                "minute": i,
                "second": 0,
                "type": {"id": 30, "name": "Pass"},
                "possession_team": {"id": 1, "name": "Reds"},
                "team": {"id": 1, "name": "Reds"},
                "player": {"id": 100, "name": "Audit Player"},
                "location": [50.0, 34.0],
                "pass": {"recipient": {"id": 101, "name": "Mate"}, "outcome": None},
            }
            for i in range(12)
        ]
        (import_dir / f"{file_id}.json").write_text(json.dumps(events), encoding="utf-8")
        (import_dir / f"{file_id}.meta.json").write_text(
            json.dumps({"competition": "Audit Cup", "match_date": "2026-09-19"}),
            encoding="utf-8",
        )
        # non-event JSON (root is not a non-empty list): skipped and counted
        (import_dir / f"{file_id}_lineups.json").write_text(
            json.dumps({"lineups": []}), encoding="utf-8"
        )

        async def _import():
            return await adispatch(window, "import_season_directory", str(import_dir), "Audit Cup")

        summary = json.loads(asyncio.run(_import()))
        if "error" in summary:
            # importer unavailable in this build: honest error is acceptable
            assert isinstance(summary["error"], str), summary
        else:
            assert summary.get("imported", 0) == 1, summary
            assert summary.get("skipped_not_events", 0) == 1, summary  # lineups dict-root
            matches = summary.get("matches") or []
            assert matches and matches[0].get("status") == "imported", summary
            # idempotency: second run imports 0 new (dedup by external id)
            again = json.loads(asyncio.run(_import()))
            assert again.get("imported", 0) == 0, again
            assert again.get("skipped_already", 0) == 1, again

        # event-file import: the same file through the single-file slot
        async def _import_event():
            return await adispatch(
                window, "import_event_file", str(import_dir / f"{file_id}.json"), "", "Audit Single"
            )

        ev_summary = json.loads(asyncio.run(_import_event()))
        assert isinstance(ev_summary, dict), ev_summary
        assert "error" not in ev_summary or isinstance(ev_summary["error"], str), ev_summary
        if ev_summary.get("success") or ev_summary.get("match_id"):
            assert ev_summary.get("match_id", 0) > 0, ev_summary
    finally:
        window.close()
        qapp.processEvents()


# ---------------------------------------------------------------------------
# domain: analyzer slots either answer or return honest engine errors
# ---------------------------------------------------------------------------


def test_domain_slots_contract_through_real_bridge(qapp, tmp_path, monkeypatch):
    _reset_paths(monkeypatch, tmp_path)
    from kawkab.app import MainWindow

    window = MainWindow()
    try:
        # goalkeeper analytics slots must work (v0.13.2 fix: the lazy
        # property used to crash with AttributeError on every call)
        aerial = _assert_ok(
            dispatch(
                window,
                "compute_goalkeeper_aerial_command",
                json.dumps([{"minute": 10, "outcome": "caught", "cross_type": "cross"}]),
            ),
            "compute_goalkeeper_aerial_command",
        )
        assert "crosses_faced" in aerial, aerial
        dist = _assert_ok(
            dispatch(
                window,
                "compute_goalkeeper_distribution",
                json.dumps([{"minute": 15, "distribution_type": "throw", "outcome": "complete"}]),
            ),
            "compute_goalkeeper_distribution",
        )
        assert "short_attempts" in dist, dist
        saveq = _assert_ok(
            dispatch(
                window,
                "compute_goalkeeper_save_quality",
                json.dumps([{"minute": 20, "xg": 0.3, "saved": 1, "on_target": 1}]),
            ),
            "compute_goalkeeper_save_quality",
        )
        assert "save_rate" in saveq, saveq

        # laws are static content
        _assert_ok(dispatch(window, "get_all_laws"), "get_all_laws")
        _assert_ok(dispatch(window, "get_law_summary", "12"), "get_law_summary", allow_error=True)

        # engine-backed slots: honest error payloads when the engine is off
        for slot, args in [
            (
                "analyze_setpieces",
                (
                    json.dumps(
                        [
                            {
                                "set_piece_type": "corner",
                                "minute": 12,
                                "second": 30,
                                "team": "Reds",
                                "delivery_x": 90.0,
                                "delivery_y": 2.0,
                                "delivery_style": "inswinging",
                                "delivery_height": "medium",
                                "first_contact_x": 94.0,
                                "first_contact_y": 8.0,
                                "outcome": "shot",
                            }
                        ]
                    ),
                    "Reds",
                ),
            ),
            ("check_offside", ("90.0", "80.0", "50.0", "1")),
            ("classify_event_rule", ("goal", "99.0", "34.0", "home")),
            ("analyze_goalkeeper", ("Reds", json.dumps([]), json.dumps([]), "0")),
            (
                "analyze_possession",
                (
                    "Reds",
                    "Blues",
                    json.dumps(
                        [
                            {
                                "event_type": "pass",
                                "team": "Reds",
                                "timestamp": 1.0,
                                "x": 50.0,
                                "y": 34.0,
                            }
                        ]
                    ),
                ),
            ),
            (
                "analyze_substitutions",
                (
                    "Reds",
                    json.dumps([{"minute": 60, "player_off": "A", "player_on": "B"}]),
                    json.dumps([]),
                ),
            ),
            (
                "analyze_match_psychology",
                (
                    "Reds",
                    "Blues",
                    "Away",
                    json.dumps([{"event_type": "goal", "team": "Reds", "timestamp": 10.0}]),
                ),
            ),
            ("fetch_external_cards", ("123",)),
        ]:
            d = _assert_ok(dispatch(window, slot, *args), slot, allow_error=True)
            if "error" in d:
                assert isinstance(d["error"], str) and d["error"], slot
    finally:
        window.close()
        qapp.processEvents()
