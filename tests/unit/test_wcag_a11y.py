"""WCAG 2.1 AA regression checks for the web UI.

These are the "a11y badges" -- CI-enforced assertions that the app keeps
its key WCAG 2.1 AA guarantees. They are static/structural checks (no
browser), complementing the behavioral focus-trap tests in
src/kawkab/web/tests/ui.test.js.

Covered success criteria:
  - 2.1.2 No Keyboard Trap  (modals: trap + Escape, via ui.js helpers)
  - 2.4.3 Focus Order       (modals: focus moves in and is restored out)
  - 1.3.1 Info & Relationships  (dialog markup: role/aria-modal/labelling)
  - 1.4.3 Contrast Minimum  (accessibility.css provides focus ring,
    skip link, high-contrast support)
  - 3.3.2 Labels or Instructions  (inputs are tied to labels/placeholders)
"""

import re
from pathlib import Path

WEB = Path("src/kawkab/web")
UI_JS = WEB / "js" / "ui.js"
INDEX = WEB / "index.html"
ACC_CSS = WEB / "css" / "accessibility.css"
UI_TEST = WEB / "tests" / "ui.test.js"


def _inside_label(html: str, pos: int) -> bool:
    """True when the tag at *pos* is nested inside a <label>...</label>."""
    open_pos = html.rfind("<label", 0, pos)
    if open_pos == -1:
        return False
    close_pos = html.find("</label>", pos)
    open_close = html.find(">", open_pos)
    return close_pos != -1 and open_close < pos


class TestNoKeyboardTrap:
    """WCAG 2.1 -- 2.1.2 No Keyboard Trap."""

    def test_modal_helpers_wire_escape_and_tab_cycle(self):
        code = UI_JS.read_text(encoding="utf-8")
        assert "Escape" in code, "ui.js modal helper must close on Escape (2.1.2)"
        assert 'e.key !== "Tab"' in code.replace("'", '"'), (
            "ui.js modal helper must intercept Tab for focus cycling (2.1.2)"
        )
        assert "shiftKey" in code, "Shift+Tab must cycle backwards (2.1.2)"

    def test_focus_trap_behavior_is_tested(self):
        code = UI_TEST.read_text(encoding="utf-8")
        assert "modal focus trap" in code, "ui.test.js must contain the behavioral focus-trap suite"
        for needed in ("Escape closes the modal", "Shift+Tab on the first"):
            assert needed in code, f"focus-trap test missing: {needed}"


class TestFocusOrder:
    """WCAG 2.1 -- 2.4.3 Focus Order."""

    def test_open_modal_moves_focus_in_and_close_restores_it(self):
        code = UI_JS.read_text(encoding="utf-8")
        assert "_modalReturnFocus" in code, (
            "ui.js must remember the opener element to restore focus on close (2.4.3)"
        )
        assert ".focus()" in code, "ui.js must move focus into the modal on open (2.4.3)"


class TestDialogMarkup:
    """WCAG 2.1 -- 1.3.1 Info & Relationships (dialog semantics)."""

    def test_first_run_modal_has_dialog_role_and_label(self):
        html = INDEX.read_text(encoding="utf-8")
        m = re.search(r'<div id="first-run-modal"[^>]*>', html)
        assert m, "first-run-modal missing"
        tag = m.group(0)
        assert 'role="dialog"' in tag, "first-run-modal must have role=dialog"
        assert 'aria-modal="true"' in tag, "first-run-modal must set aria-modal"
        assert "aria-labelledby=" in tag, "first-run-modal must be labelled"

    def test_shortcuts_modal_has_dialog_role(self):
        html = INDEX.read_text(encoding="utf-8")
        m = re.search(r'<div id="shortcuts-modal"[^>]*>', html)
        assert m, "shortcuts-modal missing"
        assert 'role="dialog"' in m.group(0)


class TestContrastAndVisibility:
    """WCAG 2.1 -- 1.4.3 Contrast Minimum & 2.4.7 Focus Visible."""

    def test_focus_ring_defined(self):
        css = ACC_CSS.read_text(encoding="utf-8")
        assert ":focus-visible" in css, "visible focus indicator required (2.4.7)"
        assert "--focus-ring" in css or "outline" in css

    def test_skip_link_present_and_styled(self):
        css = ACC_CSS.read_text(encoding="utf-8")
        assert "skip-link" in css, "keyboard skip link required"
        html = INDEX.read_text(encoding="utf-8")
        assert "skip-link" in html, "index.html must include the skip link element"

    def test_high_contrast_media_support(self):
        css = ACC_CSS.read_text(encoding="utf-8")
        assert "prefers-contrast" in css or "forced-colors" in css, (
            "accessibility.css must support high-contrast modes"
        )


class TestLabelsOrInstructions:
    """WCAG 2.1 -- 3.3.2 Labels or Instructions."""

    def test_inputs_have_label_or_placeholder_or_aria(self):
        html = INDEX.read_text(encoding="utf-8")
        inputs = [(m.group(0), m.start()) for m in re.finditer(r"<input\b[^>]*>", html)]
        assert inputs, "sanity: index.html should contain inputs"
        bad = []
        for tag, pos in inputs:
            in_label = _inside_label(html, pos)
            has_id_for_label = "id=" in tag
            has_aria = "aria-label=" in tag or "aria-labelledby=" in tag
            has_placeholder = "placeholder=" in tag
            if not (in_label or has_id_for_label or has_aria or has_placeholder):
                bad.append(tag[:80])
        assert not bad, f"inputs with no label association: {bad[:3]}"
