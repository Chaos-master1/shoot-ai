"""Tests for the kawkab CLI entry point (src/kawkab/__main__.py).

Regression coverage for the bug where `python -m kawkab` with no subcommand
printed --help and exited 1 instead of launching the desktop app -- which
meant the packaged .exe, `python -m kawkab` (the README's documented launch
command), and the `kawkab` console script (pyproject.toml) all failed to
open the GUI. See CLAUDE.md for the full writeup.

kawkab.__main__ has no module-level kawkab.* imports (all service/app
imports are lazy, inside function bodies), so this file does not need
install_kawkab_stubs() -- it can import the module directly.
"""

from __future__ import annotations

import sys
import types

import pytest

import kawkab.__main__ as kawkab_main


class TestGuiIsDefaultCommand:
    """`python -m kawkab` with no subcommand must launch the desktop app."""

    def test_no_subcommand_dispatches_to_gui(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = []
        monkeypatch.setattr(kawkab_main, "_run_gui", lambda: calls.append(1) or 0)
        monkeypatch.setattr(sys, "argv", ["kawkab"])

        with pytest.raises(SystemExit) as exc:
            kawkab_main.main()

        assert calls == [1]
        assert exc.value.code == 0

    def test_explicit_gui_subcommand_dispatches_to_gui(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = []
        monkeypatch.setattr(kawkab_main, "_run_gui", lambda: calls.append(1) or 0)
        monkeypatch.setattr(sys, "argv", ["kawkab", "gui"])

        with pytest.raises(SystemExit) as exc:
            kawkab_main.main()

        assert calls == [1]
        assert exc.value.code == 0

    def test_gui_exit_code_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A non-zero Qt exec() result must surface as the process exit code."""
        monkeypatch.setattr(kawkab_main, "_run_gui", lambda: 3)
        monkeypatch.setattr(sys, "argv", ["kawkab"])

        with pytest.raises(SystemExit) as exc:
            kawkab_main.main()

        assert exc.value.code == 3

    def test_track_subcommand_does_not_launch_gui(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Other subcommands must keep working and must not redirect to the GUI."""
        gui_calls = []
        monkeypatch.setattr(kawkab_main, "_run_gui", lambda: gui_calls.append(1) or 0)
        # _run_tracking is a coroutine function; asyncio.run() would otherwise
        # drive it for real (importing CVService, touching the filesystem).
        # Creating the coroutine object executes none of its body, so closing
        # it here is a safe no-op stand-in for actually running it.
        monkeypatch.setattr(kawkab_main.asyncio, "run", lambda coro: coro.close())
        monkeypatch.setattr(sys, "argv", ["kawkab", "track", "--video", "x.mp4"])

        kawkab_main.main()  # must return normally, not sys.exit()

        assert gui_calls == []


class TestRunGui:
    """`_run_gui()` is a thin, lazily-imported delegate to kawkab.app.run_app."""

    def test_delegates_to_app_run_app(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_app_module = types.ModuleType("kawkab.app")
        fake_app_module.run_app = lambda: 42
        monkeypatch.setitem(sys.modules, "kawkab.app", fake_app_module)

        assert kawkab_main._run_gui() == 42


class TestAppModuleImportable:
    """kawkab.app must actually be importable, or _run_gui's lazy import fails at runtime."""

    def test_run_app_is_importable(self) -> None:
        pytest.importorskip("PySide6", reason="PySide6 not installed in this environment")
        from kawkab.app import run_app

        assert callable(run_app)
