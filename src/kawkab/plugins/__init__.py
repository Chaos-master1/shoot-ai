"""Plugin system for Kawkab AI.

Plugins are discovered via the ``kawkab.plugins`` entry-point group
at ``importlib.metadata.entry_points``.  Each plugin must subclass
:class:`KawkabPlugin` and implement at least one of the lifecycle hooks.

Example plugin definition in ``pyproject.toml``::

    [project.entry-points."kawkab.plugins"]
    my_plugin = "my_package.my_plugin:MyPlugin"
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class KawkabPlugin(ABC):
    """Abstract base class for all Kawkab AI plugins.

    Subclasses should override the hooks they need; unused hooks
    default to no-ops.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable plugin name (e.g. ``"Heatmap Export"``)."""

    @property
    def version(self) -> str:
        return "0.1.0"

    # ── Lifecycle hooks ────────────────────────────────────────────

    async def on_plugin_load(self, app: Any) -> None:  # noqa: B027  (optional hook)
        """Called once when the plugin is first loaded.

        *app* is the main :class:`~kawkab.app.KawkabApp` instance.
        """

    async def on_plugin_unload(self, app: Any) -> None:  # noqa: B027  (optional hook)
        """Called when the application is shutting down."""

    async def on_analysis_start(  # noqa: B027  (optional hook)
        self, match_id: int, video_path: str, config: dict[str, Any]
    ) -> None:
        """Called before a match analysis begins."""

    async def on_analysis_end(self, match_id: int, result: dict[str, Any]) -> None:  # noqa: B027  (optional hook)
        """Called after a match analysis completes."""

    async def on_frame(  # noqa: B027  (optional hook)
        self,
        match_id: int,
        frame_number: int,
        timestamp: float,
        detections: list[dict[str, Any]],
    ) -> None:
        """Called for each processed frame during CV detection."""

    async def on_event_detected(  # noqa: B027  (optional hook)
        self,
        match_id: int,
        event: dict[str, Any],
    ) -> None:
        """Called for each detected event during analysis."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} v{self.version}>"
