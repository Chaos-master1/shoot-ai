"""Tests for calibration_v2.js (browser-side module).

The module is browser-side, so we test for syntax errors and required
API surface.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

JS_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "kawkab"
    / "web"
    / "js"
    / "calibration_v2.js"
)


def test_file_exists() -> None:
    assert JS_PATH.exists()


def test_file_nonempty() -> None:
    assert JS_PATH.stat().st_size > 100


def test_js_syntax_with_node() -> None:
    try:
        result = subprocess.run(
            ["node", "--check", str(JS_PATH)],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        import pytest

        pytest.skip("node not installed")
    assert result.returncode == 0, f"JS syntax error: {result.stderr}"


def test_uses_iife() -> None:
    content = JS_PATH.read_text(encoding="utf-8")
    assert re.search(r"^\s*\(function\s*\(\)\s*\{", content, re.MULTILINE)


def test_exposes_global_api() -> None:
    content = JS_PATH.read_text(encoding="utf-8")
    for sym in ["create", "HANDLES", "CORNERS"]:
        assert f"{sym}" in content, f"Missing public symbol: {sym}"


def test_has_all_8_handles() -> None:
    content = JS_PATH.read_text(encoding="utf-8")
    for h in ["tl", "tm", "tr", "lm", "rm", "bl", "bm", "br"]:
        assert f'"{h}"' in content or f"'{h}'" in content, f"Missing handle: {h}"


def test_supports_keyboard_navigation() -> None:
    content = JS_PATH.read_text(encoding="utf-8")
    for key in ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Enter"]:
        assert key in content, f"Missing keyboard support: {key}"


def test_supports_touch_events() -> None:
    content = JS_PATH.read_text(encoding="utf-8")
    assert "touchstart" in content
    assert "touchmove" in content
    assert "touchend" in content


def test_supports_shift_snap() -> None:
    content = JS_PATH.read_text(encoding="utf-8")
    assert "shiftKey" in content


def test_has_aria_attributes() -> None:
    content = JS_PATH.read_text(encoding="utf-8")
    assert "role" in content
    assert "aria-label" in content
    assert "aria-valuetext" in content
    assert "tabIndex" in content


def test_has_polygon_svg() -> None:
    content = JS_PATH.read_text(encoding="utf-8")
    assert "polygon" in content
    assert "createElementNS" in content


def test_has_pitch_preview() -> None:
    content = JS_PATH.read_text(encoding="utf-8")
    assert "kawkab-cal-preview" in content
    assert "viewBox" in content


def test_has_validation_badge() -> None:
    content = JS_PATH.read_text(encoding="utf-8")
    assert "kawkab-cal-validation" in content
    assert "aria-live" in content


def test_has_snap_aspect_ratio() -> None:
    content = JS_PATH.read_text(encoding="utf-8")
    assert "snapAR" in content
    assert "105 / 68" in content or "105/68" in content


def test_has_destroy_method() -> None:
    content = JS_PATH.read_text(encoding="utf-8")
    assert "destroy" in content
