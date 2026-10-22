"""Tests for KawkabPolish JS module.

The polish module is browser-side, so we test the static JavaScript
file for syntax errors and required API surface. Full DOM tests would
require jsdom — out of scope for the unit tier.
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
    / "kawkab_polish.js"
)


def test_polish_file_exists() -> None:
    assert JS_PATH.exists(), f"Polish module not found: {JS_PATH}"


def test_polish_file_nonempty() -> None:
    assert JS_PATH.stat().st_size > 100


def test_polish_syntax_with_node() -> None:
    """Verify JS syntax. Skip if node is unavailable."""
    try:
        result = subprocess.run(
            ["node", "--check", str(JS_PATH)],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        pytest_skip("node not installed")
    except subprocess.TimeoutExpired:
        pytest_fail("node --check timed out")
    assert result.returncode == 0, f"JS syntax error: {result.stderr}"


def test_polish_exposes_global_api() -> None:
    """Verify the public API surface is declared."""
    content = JS_PATH.read_text(encoding="utf-8")
    for symbol in ["announce", "setLang", "prefersReducedMotion", "showShortcutHelp"]:
        assert f"{symbol}" in content, f"Missing public symbol: {symbol}"


def test_polish_uses_iife() -> None:
    """Verify IIFE wrapper to avoid global namespace pollution."""
    content = JS_PATH.read_text(encoding="utf-8")
    assert re.search(r"^\s*\(function\s*\(\)\s*\{", content, re.MULTILINE), "Missing IIFE wrapper"


def test_polish_creates_live_region() -> None:
    content = JS_PATH.read_text(encoding="utf-8")
    assert "aria-live" in content
    assert "sr-only" in content


def test_polish_handles_rtl() -> None:
    content = JS_PATH.read_text(encoding="utf-8")
    assert '"rtl"' in content or "'rtl'" in content
    assert '"ar"' in content or "'ar'" in content


def test_polish_respects_reduced_motion() -> None:
    content = JS_PATH.read_text(encoding="utf-8")
    assert "prefers-reduced-motion" in content


def test_polish_has_keyboard_shortcuts() -> None:
    content = JS_PATH.read_text(encoding="utf-8")
    for key in ["Escape", "?", "/", "g"]:
        assert key in content, f"Missing keyboard shortcut: {key}"


def test_polish_localizes_app_subtitle() -> None:
    content = JS_PATH.read_text(encoding="utf-8")
    assert "app_subtitle" in content


def test_polish_arabic_dict_has_subtitle() -> None:
    content = JS_PATH.read_text(encoding="utf-8")
    en_idx = content.find('"app_subtitle": "Football Coach"')
    ar_idx = content.find('"app_subtitle": "مدرب كرة القدم"')
    assert en_idx > 0, "Missing English subtitle translation"
    assert ar_idx > 0, "Missing Arabic subtitle translation"


def test_polish_dicts_have_same_keys() -> None:
    """EN and AR dictionaries must have identical key sets for parity."""
    content = JS_PATH.read_text(encoding="utf-8")
    en_block = content.split('"tab_analyze": "Analyze"')[0]
    if en_block.rfind("return {") > en_block.rfind("};"):
        en_block = en_block[en_block.rfind("return {") :]
    ar_block = content.split('"tab_analyze": "تحليل"')[0]
    if ar_block.rfind("return {") > ar_block.rfind("};"):
        ar_block = ar_block[ar_block.rfind("return {") :]
    en_keys = set(re.findall(r'"([a-z_]+)":', en_block))
    ar_keys = set(re.findall(r'"([a-z_]+)":', ar_block))
    assert en_keys, "No English keys found"
    assert ar_keys, "No Arabic keys found"
    missing_in_ar = en_keys - ar_keys
    missing_in_en = ar_keys - en_keys
    assert not missing_in_ar, f"Missing AR keys: {missing_in_ar}"
    assert not missing_in_en, f"Missing EN keys: {missing_in_en}"


def test_polish_has_at_least_30_keys() -> None:
    """Dictionaries should have at least 30 keys for meaningful coverage."""
    content = JS_PATH.read_text(encoding="utf-8")
    en_count = len(re.findall(r'"[a-z_]+":\s*"[^"]+"', content))
    assert en_count >= 60, f"Expected >=60 i18n entries (en+ar), got {en_count}"


def test_polish_has_alert_translations() -> None:
    content = JS_PATH.read_text(encoding="utf-8")
    for key in ["alert_shot", "alert_goal", "alert_offside", "alert_card", "alert_tackle"]:
        assert key in content, f"Missing alert translation: {key}"


def test_polish_has_metric_translations() -> None:
    content = JS_PATH.read_text(encoding="utf-8")
    for key in [
        "metric_distance",
        "metric_sprints",
        "metric_passes",
        "metric_shots",
        "metric_xg",
        "metric_goals",
    ]:
        assert key in content, f"Missing metric translation: {key}"


def test_polish_has_section_translations() -> None:
    content = JS_PATH.read_text(encoding="utf-8")
    for key in [
        "section_pro",
        "section_realtime",
        "section_psychology",
        "section_weather",
        "section_rules",
    ]:
        assert key in content, f"Missing section translation: {key}"


def test_polish_init_syncs_selector() -> None:
    """The init function should sync the language-selector element."""
    content = JS_PATH.read_text(encoding="utf-8")
    assert "language-selector" in content, "Init must reference #language-selector"
    assert "selector.value" in content, "Init must update selector value"


def test_polish_uses_aria_live_region() -> None:
    content = JS_PATH.read_text(encoding="utf-8")
    assert "aria-live" in content
    assert "aria-atomic" in content


def test_polish_falls_back_to_english() -> None:
    content = JS_PATH.read_text(encoding="utf-8")
    assert 'startsWith("ar")' in content
    assert (
        'return "en"' in content
        or '?: "en"' in content
        or 'return "ar" : "en"' in content
        or '"ar" : "en"' in content
    )


def test_polish_persists_lang() -> None:
    content = JS_PATH.read_text(encoding="utf-8")
    assert "localStorage" in content
    assert "STORAGE_KEY" in content


def test_accessibility_css_exists() -> None:
    css = JS_PATH.parent.parent / "css" / "accessibility.css"
    assert css.exists()
    content = css.read_text(encoding="utf-8")
    assert "skip-link" in content
    assert "focus-visible" in content
    assert "prefers-reduced-motion" in content


def test_accessibility_css_rtl_support() -> None:
    css = JS_PATH.parent.parent / "css" / "accessibility.css"
    content = css.read_text(encoding="utf-8")
    assert "rtl" in content
    assert "Cairo" in content or "Tahoma" in content


def test_accessibility_css_high_contrast() -> None:
    css = JS_PATH.parent.parent / "css" / "accessibility.css"
    content = css.read_text(encoding="utf-8")
    assert "prefers-contrast" in content


def pytest_skip(msg: str) -> None:
    import pytest

    pytest.skip(msg)


def pytest_fail(msg: str) -> None:
    import pytest

    pytest.fail(msg)


if __name__ == "__main__":
    test_polish_file_exists()
    test_polish_file_nonempty()
    test_polish_exposes_global_api()
    test_polish_uses_iife()
    test_polish_creates_live_region()
    test_polish_handles_rtl()
    test_polish_respects_reduced_motion()
    test_polish_has_keyboard_shortcuts()
    test_polish_localizes_app_subtitle()
    test_polish_arabic_dict_has_subtitle()
    test_accessibility_css_exists()
    test_accessibility_css_rtl_support()
    test_accessibility_css_high_contrast()
    print("All tests passed")
