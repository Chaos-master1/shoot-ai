"""Tests for the Kawkab AI plugin system."""

from __future__ import annotations

import pytest

from kawkab.plugins import KawkabPlugin
from kawkab.plugins.manager import PluginManager


class DummyPlugin(KawkabPlugin):
    """Minimal plugin for testing."""

    def __init__(self) -> None:
        self._loaded = False
        self._unloaded = False
        self._analysis_start_calls: list[int] = []
        self._analysis_end_calls: list[int] = []
        self._event_calls: list[dict] = []

    @property
    def name(self) -> str:
        return "DummyPlugin"

    async def on_plugin_load(self, app) -> None:
        self._loaded = True

    async def on_plugin_unload(self, app) -> None:
        self._unloaded = True

    async def on_analysis_start(self, match_id, video_path, config) -> None:
        self._analysis_start_calls.append(match_id)

    async def on_analysis_end(self, match_id, result) -> None:
        self._analysis_end_calls.append(match_id)

    async def on_event_detected(self, match_id, event) -> None:
        self._event_calls.append(event)


class FailingPlugin(KawkabPlugin):
    """Plugin that raises in all hooks (for error-tolerance testing)."""

    @property
    def name(self) -> str:
        return "FailingPlugin"

    async def on_plugin_load(self, app) -> None:
        raise RuntimeError("load failed")

    async def on_analysis_start(self, match_id, video_path, config) -> None:
        raise RuntimeError("analysis start failed")

    async def on_analysis_end(self, match_id, result) -> None:
        raise RuntimeError("analysis end failed")

    async def on_event_detected(self, match_id, event) -> None:
        raise RuntimeError("event failed")


class TestPluginBase:
    def test_abstract_prevents_instantiation(self):
        with pytest.raises(TypeError):
            KawkabPlugin()  # type: ignore[abstract]

    def test_dummy_plugin_name(self):
        p = DummyPlugin()
        assert p.name == "DummyPlugin"
        assert p.version == "0.1.0"
        assert "DummyPlugin" in repr(p)

    def test_dummy_plugin_lifecycle(self):
        import anyio

        p = DummyPlugin()
        anyio.run(p.on_plugin_load, None)
        assert p._loaded
        anyio.run(p.on_plugin_unload, None)
        assert p._unloaded

    def test_dummy_plugin_hooks(self):
        import anyio

        p = DummyPlugin()
        anyio.run(p.on_analysis_start, 1, "/path", {})
        assert p._analysis_start_calls == [1]
        anyio.run(p.on_analysis_end, 1, {})
        assert p._analysis_end_calls == [1]
        anyio.run(p.on_event_detected, 1, {"type": "shot"})
        assert p._event_calls == [{"type": "shot"}]


class TestPluginManager:
    def test_manager_init(self):
        mgr = PluginManager()
        assert mgr.plugins == {}

    def test_register_and_unregister(self):
        import anyio

        mgr = PluginManager()
        p = DummyPlugin()
        mgr._plugins["dummy"] = p
        assert "dummy" in mgr.plugins
        anyio.run(mgr.unload, "dummy")
        assert "dummy" not in mgr.plugins
        assert p._unloaded

    def test_hooks_no_plugins_dont_raise(self):
        import anyio

        mgr = PluginManager()
        anyio.run(mgr.on_analysis_start, 1, "", {})
        anyio.run(mgr.on_analysis_end, 1, {})
        anyio.run(mgr.on_frame, 1, 0, 0.0, [])
        anyio.run(mgr.on_event_detected, 1, {})

    def test_hooks_call_all_plugins(self):
        import anyio

        mgr = PluginManager()
        p1 = DummyPlugin()
        p2 = DummyPlugin()
        mgr._plugins["p1"] = p1
        mgr._plugins["p2"] = p2

        anyio.run(mgr.on_analysis_start, 42, "/v", {"k": "v"})
        assert p1._analysis_start_calls == [42]
        assert p2._analysis_start_calls == [42]

        anyio.run(mgr.on_analysis_end, 99, {"ok": True})
        assert p1._analysis_end_calls == [99]
        assert p2._analysis_end_calls == [99]

        anyio.run(mgr.on_event_detected, 1, {"type": "goal"})
        assert len(p1._event_calls) == 1
        assert len(p2._event_calls) == 1

    def test_failing_plugin_does_not_block_others(self):
        import anyio

        mgr = PluginManager()
        good = DummyPlugin()
        bad = FailingPlugin()
        mgr._plugins["good"] = good
        mgr._plugins["bad"] = bad

        anyio.run(mgr.on_analysis_start, 1, "", {})
        assert good._analysis_start_calls == [1]

        anyio.run(mgr.on_analysis_end, 1, {})
        assert good._analysis_end_calls == [1]

        anyio.run(mgr.on_event_detected, 1, {"type": "foul"})
        assert len(good._event_calls) == 1

    def test_discover_returns_empty_when_no_entry_points(self):
        mgr = PluginManager()
        found = mgr.discover()
        assert isinstance(found, list)
