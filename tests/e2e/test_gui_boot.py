"""Real GUI boot test: MainWindow must come up offscreen, end to end.

This exercises the actual startup path a user hits: settings -> services
(storage initialize + WAL) -> UI -> bridge (with the split handlers) ->
system-tray skip -> shutdown. It caught the v0.12.0 window-title drift on
its first run.

Skipped automatically when PySide6 is not installed (CI core job);
the dedicated gui-e2e CI job installs it and runs this module.
"""

from __future__ import annotations

import os

import pytest

pyside6 = pytest.importorskip("PySide6")


@pytest.fixture()
def qapp(monkeypatch, tmp_path):
    """Offscreen QApplication with an isolated environment.

    Isolation matters in two directions:
    - outward: the app must never read/write the developer's real Documents,
      cache, or database -- HOME and the XDG vars are pointed into tmp_path.
    - inward: get_paths() caches a module-level singleton, so env vars set
      here arrive too late unless the singleton is reset; same for the
      lru_cache'd get_settings() and any KAWKAB_DB_URL a dev shell exported
      (which would silently flip the boot into Postgres mode).
    """
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.setenv("XDG_DOCUMENTS_DIR", str(tmp_path / "xdg-docs"))
    monkeypatch.setenv("KAWKAB_DEBUG", "false")
    monkeypatch.delenv("KAWKAB_DB_URL", raising=False)

    import kawkab.core.paths as paths_mod

    # The real module caches a singleton in _paths; the unit-test stub
    # (tests/conftest.py) instead exposes _default_paths. Reset whichever
    # exists so env vars set above take effect. When running under the
    # stubbed package, Paths() is already tmp-isolated and needs no reset.
    if hasattr(paths_mod, "_paths"):
        monkeypatch.setattr(paths_mod, "_paths", None)
    from kawkab.core.config import get_settings

    get_settings.cache_clear()

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(["kawkab-gui-test"])
    yield app
    get_settings.cache_clear()


def test_mainwindow_boots_offscreen(qapp):
    """The real startup chain: MainWindow() constructs every service."""
    from kawkab.app import MainWindow

    window = MainWindow()
    try:
        assert window.windowTitle(), "window has no title"
        assert "Kawkab" in window.windowTitle()
        assert window.storage._conn is not None, (
            "MainWindow boot must leave storage connected (WAL sqlite)"
        )
        # The split handlers must be wired on the real bridge instance.
        assert window.bridge._recruitment is not None
        assert window.bridge._settings is not None
    finally:
        window.close()
        qapp.processEvents()


def test_mainwindow_title_matches_package_version(qapp):
    """The window title must show the same version as kawkab.__version__.

    Regression pin: the boot probe that inspired this suite caught the
    title advertising v0.12.0 while pyproject/CHANGELOG said 0.13.0.
    """
    from kawkab import __version__
    from kawkab.app import MainWindow

    window = MainWindow()
    try:
        assert f"v{__version__}" in window.windowTitle(), (
            f"window title {window.windowTitle()!r} does not advertise "
            f"package version {__version__} -- version sources drifted"
        )
    finally:
        window.close()
        qapp.processEvents()


def test_offscreen_platform_is_usable(qapp):
    """Sanity: the offscreen plugin must be the active platform in tests."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    assert app is not None, "QApplication fixture must have run"
    assert os.environ.get("QT_QPA_PLATFORM") == "offscreen"
