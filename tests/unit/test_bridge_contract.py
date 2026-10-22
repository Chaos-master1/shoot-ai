"""Regression guard for the frontend <-> Bridge contract.

Found during the professional-readiness audit: 6 frontend calls
(bridge.get_dashboard_stats, bridge.comparePlayers, bridge.get_player_stats,
bridge.get_match_players, plus two name typos) had no backing @Slot in
kawkab.ui.bridge -- meaning the Dashboard, Player Compare, and 3D pitch
features were silently broken (the JS caught the resulting exception and
degraded, so nothing crashed loudly). This test makes that class of bug
fail CI instead of shipping silently: every `bridge.X(`/`kawkab.X(` call
found anywhere in src/kawkab/web/js/*.js must have a matching @Slot method
on Bridge.

Deliberately strict: even a call inside an `if (bridge.foo)` feature-detect
guard must have a real backing slot. A guarded call to a nonexistent method
never crashes, but it also never does anything -- which is exactly how the
6 bugs above went unnoticed. Half-wired speculative JS should fail fast,
not degrade silently.

Pure text parsing (regex), no kawkab imports needed -- this test only reads
source files, so it doesn't need install_kawkab_stubs().
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BRIDGE_PY = PROJECT_ROOT / "src" / "kawkab" / "ui" / "bridge.py"
WEB_JS_DIR = PROJECT_ROOT / "src" / "kawkab" / "web" / "js"

# Matches `bridge.foo(`, `kawkab.foo(`, `window.bridge.foo(`, `window.kawkab.foo(`.
_JS_CALL_RE = re.compile(r"\b(?:window\.)?(?:kawkab|bridge)\s*\.\s*(\w+)\s*\(")
# Matches an `@Slot(...)` decorator immediately followed by a (possibly async) def.
_SLOT_DEF_RE = re.compile(r"@Slot\([^)]*\)\s*\r?\n\s*(?:async\s+)?def\s+(\w+)")


def _bridge_slots() -> set[str]:
    text = BRIDGE_PY.read_text(encoding="utf-8")
    names = _SLOT_DEF_RE.findall(text)
    assert names, f"No @Slot methods found in {BRIDGE_PY} -- parser regression?"
    return set(names)


def _js_bridge_calls() -> dict[str, set[str]]:
    """Map of {method_name: {js_filenames_that_call_it}}."""
    calls: dict[str, set[str]] = {}
    js_files = sorted(WEB_JS_DIR.glob("*.js"))
    assert js_files, f"No .js files found in {WEB_JS_DIR} -- path regression?"
    for path in js_files:
        text = path.read_text(encoding="utf-8")
        for m in _JS_CALL_RE.finditer(text):
            calls.setdefault(m.group(1), set()).add(path.name)
    return calls


def test_every_js_bridge_call_has_a_backing_slot() -> None:
    slots = _bridge_slots()
    calls = _js_bridge_calls()

    missing = {name: files for name, files in calls.items() if name not in slots}
    if missing:
        lines = [
            f"  bridge.{name}(...)  <- called from: {', '.join(sorted(files))}"
            for name, files in sorted(missing.items())
        ]
        raise AssertionError(
            "The following frontend bridge.*() / kawkab.*() calls have NO matching "
            "@Slot method in src/kawkab/ui/bridge.py -- they will throw (or, if "
            "feature-detect guarded, silently no-op) at runtime:\n"
            + "\n".join(lines)
            + "\n\nEither add the missing @Slot (delegating to a bridge_handlers/ "
            "method, per the pattern every other slot follows), or fix the JS "
            "call if it's a typo of an existing slot name."
        )


def test_unreachable_slots_inventory() -> None:
    """Non-failing: lists @Slot methods no JS file currently calls.

    Not necessarily a bug -- Phase 3 of the professional-readiness audit
    intentionally leaves some analytical slots dormant-but-documented
    rather than wiring all of them into the UI at once (see CLAUDE.md,
    "built but not exposed"). Run with `-s` to see the current count.
    """
    slots = _bridge_slots()
    calls = _js_bridge_calls()
    unreachable = sorted(slots - calls.keys())
    print(
        f"\n{len(unreachable)} of {len(slots)} @Slot methods have no JS caller "
        f"(see CLAUDE.md for the intentionally-dormant list):"
    )
    for name in unreachable:
        print(f"  {name}")


# ── Second contract layer: slot delegations must resolve to real methods ────

# bridge.py delegates like `return self._analysis.get_homography(...)`
# or `return await self._storage.list_matches(...)`. The attribute names
# map to handler classes (self._analysis -> AnalysisHandler etc.).
_HANDLER_MODULES = {
    "_analysis": "bridge_analysis.py",
    "_match_intel": "bridge_match_intel.py",
    "_physical": "bridge_physical.py",
    "_domain": "bridge_domain.py",
    "_cloud": "bridge_cloud.py",
    "_live": "bridge_live.py",
    "_whiteboard": "bridge_whiteboard.py",
    "_auth": "bridge_auth.py",
    "_coding": "bridge_coding.py",
    "_export": "bridge_export.py",
    "_external": "bridge_external.py",
    "_lifecycle": "bridge_lifecycle.py",
    "_provider": "bridge_provider.py",
    "_pro_analytics": "bridge_pro_analytics.py",
    "_season_analytics": "bridge_season_analytics.py",
    "_storage": "bridge_storage.py",
    "_video": "bridge_video.py",
}
BRIDGE_HANDLERS_DIR = PROJECT_ROOT / "src" / "kawkab" / "ui" / "bridge_handlers"

# `self._analysis.method_name(` — capture attr + method
_DELEGATION_RE = re.compile(r"self\.(_[a-z]+)\.(\w+)\s*\(")


def _handler_method_names(module_file: str) -> set[str]:
    """All def/async def names in a handler module (class-agnostic)."""
    text = (BRIDGE_HANDLERS_DIR / module_file).read_text(encoding="utf-8")
    return set(re.findall(r"(?:async\s+)?def\s+(\w+)", text))


def _slot_method_bodies() -> dict[str, str]:
    """Map slot name -> its (possibly multi-line) body from bridge.py."""
    text = BRIDGE_PY.read_text(encoding="utf-8")
    bodies: dict[str, str] = {}
    # Match @Slot(...) decorator + def line + body up to the next decorator/def
    pattern = re.compile(
        r"@Slot\([^)]*\)\s*\r?\n\s*(?:async\s+)?def\s+(\w+)\s*\((?:[^)]*)\)\s*(?:->\s*[^:]+)?:\s*\r?\n(.*?)(?=\r?\n\s*(?:@Slot|def |async def |class |\Z))",
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        bodies[m.group(1)] = m.group(2)
    return bodies


def test_every_slot_delegation_targets_a_real_handler_method() -> None:
    """Every `self._X.method(...)` in bridge.py must exist in the handler.

    This closes the gap that let `ai_v2_list_convs` ship broken: the
    @Slot existed, its body called `self._analysis.ai_v2_list_convs`,
    but AnalysisHandler had no such method -- AttributeError on every
    call, invisible to the JS-vs-slot-name check above.
    """
    known_attrs = set(_HANDLER_MODULES)
    method_pools = {attr: _handler_method_names(f) for attr, f in _HANDLER_MODULES.items()}
    text = BRIDGE_PY.read_text(encoding="utf-8")

    broken: list[str] = []
    for m in _DELEGATION_RE.finditer(text):
        attr, method = m.group(1), m.group(2)
        if attr not in known_attrs:
            continue  # e.g. self._rate_limiter — not a handler delegation
        if method not in method_pools[attr]:
            broken.append(f"self.{attr}.{method}(...) — no such method in {_HANDLER_MODULES[attr]}")

    if broken:
        raise AssertionError(
            "bridge.py @Slot bodies delegate to handler methods that do not exist "
            "(AttributeError at runtime, caught by broad except, silently broken "
            "feature — the ai_v2_list_convs bug class):\n  " + "\n  ".join(sorted(set(broken)))
        )
