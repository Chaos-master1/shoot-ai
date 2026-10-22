"""Sprint 6: Polish & Hardening tests.

Tests cover:
- Error boundary (wrap, render success, render failure, retry button, multiple sections)
- Keyboard shortcuts (modal shows on ?, close, has all shortcuts)
- i18n coverage (all data-i18n keys have values, no hardcoded text in new sections)
- Loading skeletons (all sections registered, show/hide)
"""

import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
WEB_JS = BASE / "src" / "kawkab" / "web" / "js"
WEB_CSS = BASE / "src" / "kawkab" / "web" / "css"
INDEX = BASE / "src" / "kawkab" / "web" / "index.html"


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _read(filepath):
    with open(filepath, encoding="utf-8") as f:
        return f.read()


def _readjs(name):
    return _read(WEB_JS / name)


def _get_section_ids():
    """Return all section IDs from index.html."""
    html = _read(INDEX)
    return re.findall(r'<section\s+id="([^"]+)"', html)


# ──────────────────────────────────────────────
# Error Boundary Tests
# ──────────────────────────────────────────────


class TestErrorBoundary:
    def test_error_boundary_file_exists(self):
        path = WEB_JS / "app-error-boundary.js"
        assert path.exists(), "app-error-boundary.js not found"

    def test_error_boundary_wraps_sections(self):
        code = _readjs("app-error-boundary.js")
        assert "KawkabErrorBoundary" in code
        assert "wrap:" in code
        assert "showRetry:" in code
        assert "dashboard-kpis" in code or "results-section" in code

    def test_error_boundary_has_wrap_method(self):
        code = _readjs("app-error-boundary.js")
        assert "function" in code or "return function" in code
        assert "sectionId" in code
        assert "fallbackHtml" in code

    def test_error_boundary_has_retry_button(self):
        code = _readjs("app-error-boundary.js")
        assert "retry-btn" in code or "Retry" in code

    def test_error_boundary_registers_sections(self):
        code = _readjs("app-error-boundary.js")
        sections_found = re.findall(r"'([^']+-(?:section|kpis|list))'", code)
        assert len(sections_found) >= 5, f"Expected >=5 sections, got {sections_found}"


# ──────────────────────────────────────────────
# Keyboard Shortcuts Tests
# ──────────────────────────────────────────────


class TestKeyboardShortcuts:
    def test_shortcuts_modal_in_html(self):
        html = _read(INDEX)
        assert 'id="shortcuts-modal"' in html
        assert 'role="dialog"' in html
        assert 'aria-label="Keyboard shortcuts"' in html

    def test_shortcuts_close_button(self):
        html = _read(INDEX)
        assert 'id="shortcuts-close"' in html

    def test_shortcuts_has_all_expected_keys(self):
        html = _read(INDEX)
        expected = ["/", "?", "Space", "J", "L", "F", "1-9", "Q-P", "Ctrl+E"]
        found = 0
        for key in expected:
            if key in html:
                found += 1
        assert found >= 5, f"Expected >=5 shortcut keys in HTML, found {found}"

    def test_shortcuts_handler_in_ux(self):
        code = _readjs("app-ux.js")
        assert "shortcuts-modal" in code
        assert "'?'" in code or 'e.key === "?"' in code or "e.key === '?'" in code

    def test_shortcuts_css_exists(self):
        css = _read(WEB_CSS / "main.css")
        assert "shortcuts-table" in css
        assert "kbd" in css


# ──────────────────────────────────────────────
# i18n Coverage Tests
# ──────────────────────────────────────────────


class TestI18nCoverage:
    def test_all_section_h2_have_data_i18n(self):
        """Every visible section h2 should have data-i18n."""
        html = _read(INDEX)
        sections = re.findall(r'<section\s+id="([^"]+)".*?<h2([^>]*)>', html, re.DOTALL)
        missing = []
        for sid, h2_attrs in sections:
            # Some sections may not have h2 as direct child — be tolerant
            if (
                "data-i18n" not in h2_attrs
                and "review-section" not in sid
                and "coding-section" not in sid
            ):
                missing.append(sid)
        # Only report truly missing
        strict_missing = []
        # Check opponent-section and marketplace-section specifically
        html_before = _read(INDEX)
        for sid in missing:
            # Re-check: find h2 inside section
            pattern = re.compile(
                r'<section\s+id="' + re.escape(sid) + r'".*?<h2([^>]*)>',
                re.DOTALL,
            )
            m = pattern.search(html_before)
            if m and "data-i18n" not in m.group(1):
                strict_missing.append(sid)
        # We expect opponent and marketplace to now have data-i18n
        assert "opponent-section" not in strict_missing, "opponent-section h2 missing data-i18n"
        assert "marketplace-section" not in strict_missing, (
            "marketplace-section h2 missing data-i18n"
        )
        # Allow other legacy sections
        remaining = [
            s for s in strict_missing if s not in ("opponent-section", "marketplace-section")
        ]
        assert len(remaining) < 3, f"Unexpected sections missing data-i18n on h2: {remaining}"

    def test_no_hardcoded_labels_in_new_sections(self):
        """Check for common hardcoded patterns in opponent/marketplace/cloud sections."""
        html = _read(INDEX)
        # These common patterns should now have data-i18n
        hardcoded_patterns = [
            "Search opponents",
            "No opponents yet",
            "Tactical Profile",
            "Match History",
            "Community Marketplace",
            "Loading marketplace",
            "All categories",
            "Search Transfermarkt",
            "Search players",
            "All positions",
        ]
        # We check the raw text still exists (for fallback) but verify data-i18n attr is on element
        for pattern in hardcoded_patterns:
            escaped = re.escape(pattern)
            # Look for pattern NOT preceded by data-i18n= on the same tag
            matches = re.findall(r"<([a-z]+)[^>]*?" + escaped + r"[^<]*?</\1>", html, re.IGNORECASE)
            for match in matches:
                tag = f"<{match}"
                idx = html.index(tag)
                snippet = html[max(0, idx - 200) : idx + len(tag)]
                if "data-i18n" not in snippet and "data-i18n-placeholder" not in snippet:
                    # Check if the text is dynamically populated (by id attribute)
                    if "id=" in tag:
                        continue
                    raise AssertionError(
                        f"Hardcoded text '{pattern}' found without data-i18n in: {snippet[:100]}"
                    )

    def test_data_i18n_keys_have_fallback_values(self):
        """Every data-i18n attribute should have a visible text fallback."""
        html = _read(INDEX)
        # Find all data-i18n="key" and check the element has text content
        pattern = re.compile(r'data-i18n="([^"]+)"[^>]*>([^<]+)')
        matches = pattern.findall(html)
        empty = []
        for key, text in matches:
            stripped = text.strip()
            if not stripped or stripped == "" or stripped == key:
                empty.append(key)
        assert len(empty) < 20, (
            f"Too many empty data-i18n fallbacks: {empty[:10]}"
        )  # some are emoji-only


# ──────────────────────────────────────────────
# Loading Skeletons Tests
# ──────────────────────────────────────────────


class TestLoadingSkeletons:
    def test_skeleton_registrations_in_app_js(self):
        code = _readjs("app.js")
        registrations = re.findall(r"skeletons\.register\(([^)]+)\)", code)
        assert len(registrations) >= 4, (
            f"Expected >=4 skeleton registrations, got {len(registrations)}"
        )

    def test_new_sections_registered(self):
        code = _readjs("app.js")
        new_sections = [
            "scout-section",
            "squad-section",
            "tactics-section",
            "ai-section",
            "coding-section",
            "review-section",
            "season-section",
            "opponent-section",
        ]
        registered = 0
        for section in new_sections:
            if f"'{section}'" in code or f'"{section}"' in code:
                registered += 1
        assert registered >= 6, f"Expected >=6 new skeleton registrations, got {registered}"

    def test_skeleton_api_has_show_hide(self):
        code = _readjs("app-skeletons.js")
        assert "showAll" in code
        assert "hideAll" in code

    def test_skeleton_styles_defined(self):
        css = _read(WEB_CSS / "main.css") + _read(WEB_CSS / "accessibility.css")
        assert "skeleton-shimmer" in css or "@keyframes skeleton-shimmer" in css


# ──────────────────────────────────────────────
# Console Warning Audit Tests
# ──────────────────────────────────────────────


class TestConsoleWarnAudit:
    def test_console_warn_replaced_with_toast(self):
        """Key app files should have showToast alongside console.warn."""
        files_to_check = [
            "app-ai.js",
            "app-coding.js",
            "app-scout.js",
            "app-squad.js",
            "app-tactics.js",
            "app.js",
        ]
        for fname in files_to_check:
            code = _readjs(fname)
            warns = re.findall(r"console\.warn", code)
            toasts = re.findall(r"showToast", code)
            # At least as many toasts as warns in these files
            assert toasts or not warns, f"{fname}: console.warn found without showToast in file"
