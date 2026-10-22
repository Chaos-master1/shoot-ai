"""Plugin discovery and lifecycle management.

Uses ``importlib.metadata.entry_points`` (Python 3.12+) to find plugins
registered under the ``kawkab.plugins`` entry-point group.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import Any

from kawkab.plugins import KawkabPlugin

logger = logging.getLogger(__name__)


class PluginManager:
    """Discovers, loads, and manages Kawkab AI plugins.

    Usage::

        mgr = PluginManager()
        mgr.discover()
        await mgr.load_all(app)
        ...
        await mgr.on_analysis_start(match_id, video_path, config)
        ...
        await mgr.unload_all(app)
    """

    def __init__(self) -> None:
        self._plugins: dict[str, KawkabPlugin] = {}

    @property
    def plugins(self) -> dict[str, KawkabPlugin]:
        return dict(self._plugins)

    # ── Discovery ──────────────────────────────────────────────────

    def discover(self) -> list[str]:
        """Find all entry points in the ``kawkab.plugins`` group.

        Returns:
            List of entry-point names that were found.
        """
        eps = entry_points(group="kawkab.plugins")
        found: list[str] = []
        for ep in eps:
            if ep.name not in self._plugins:
                logger.info("Discovered plugin entry-point: %s = %s", ep.name, ep.value)
                found.append(ep.name)
        return found

    # ── Load / Unload ──────────────────────────────────────────────

    async def load(self, name: str, app: Any = None) -> KawkabPlugin:
        """Load a single plugin by entry-point name."""
        eps = entry_points(group="kawkab.plugins")
        for ep in eps:
            if ep.name == name:
                plugin_cls: type[KawkabPlugin] = ep.load()
                if not issubclass(plugin_cls, KawkabPlugin):
                    raise TypeError(f"Plugin {name!r} does not subclass KawkabPlugin")
                instance: KawkabPlugin = plugin_cls()
                await instance.on_plugin_load(app)
                self._plugins[name] = instance
                logger.info("Loaded plugin: %s v%s", instance.name, instance.version)
                return instance
        raise KeyError(f"Plugin {name!r} not found in entry points")

    async def load_all(self, app: Any = None) -> list[KawkabPlugin]:
        """Discover and load all available plugins."""
        names = self.discover()
        loaded: list[KawkabPlugin] = []
        for name in names:
            try:
                loaded.append(await self.load(name, app))
            except Exception as exc:
                logger.error("Failed to load plugin %r: %s", name, exc)
        return loaded

    async def unload(self, name: str, app: Any = None) -> None:
        plugin = self._plugins.pop(name, None)
        if plugin is not None:
            await plugin.on_plugin_unload(app)
            logger.info("Unloaded plugin: %s", plugin.name)

    async def unload_all(self, app: Any = None) -> None:
        for name in list(self._plugins):
            await self.unload(name, app)

    # ── Hooks ──────────────────────────────────────────────────────

    async def on_analysis_start(
        self, match_id: int, video_path: str, config: dict[str, Any]
    ) -> None:
        for plugin in self._plugins.values():
            try:
                await plugin.on_analysis_start(match_id, video_path, config)
            except Exception as exc:
                logger.error("Plugin %r on_analysis_start error: %s", plugin.name, exc)

    async def on_analysis_end(self, match_id: int, result: dict[str, Any]) -> None:
        for plugin in self._plugins.values():
            try:
                await plugin.on_analysis_end(match_id, result)
            except Exception as exc:
                logger.error("Plugin %r on_analysis_end error: %s", plugin.name, exc)

    async def on_frame(
        self,
        match_id: int,
        frame_number: int,
        timestamp: float,
        detections: list[dict[str, Any]],
    ) -> None:
        for plugin in self._plugins.values():
            try:
                await plugin.on_frame(match_id, frame_number, timestamp, detections)
            except Exception as exc:
                logger.error("Plugin %r on_frame error: %s", plugin.name, exc)

    async def on_event_detected(self, match_id: int, event: dict[str, Any]) -> None:
        for plugin in self._plugins.values():
            try:
                await plugin.on_event_detected(match_id, event)
            except Exception as exc:
                logger.error("Plugin %r on_event_detected error: %s", plugin.name, exc)

    def __repr__(self) -> str:
        return f"<PluginManager plugins={list(self._plugins)}>"
