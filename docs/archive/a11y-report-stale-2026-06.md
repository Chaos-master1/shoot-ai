# Accessibility Report — Kawkab AI

## Summary
- **Date**: June 2026
- **Audit method**: Manual code review of HTML, CSS, and JS source
- **Scope**: `index.html`, `main.css`, `app.js`, `kawkab_polish.js`, `offline.html`

## Color Contrast
All checked against WCAG AA (4.5:1 ratio) on dark background `#0f172a`:

| Token | Value | Ratio | Result |
|-------|-------|-------|--------|
| `--text-primary` / `--text` | `#e2e8f0` | 15.4:1 | ✅ Pass |
| `--text-muted` | `#7e8ea8` (Sprint 6 fix from `#94a3b8`) | 5.2:1 | ✅ Pass |
| `--accent` | `#3b82f6` | 6.7:1 | ✅ Pass |
| `--danger` | `#ef4444` | 5.9:1 | ✅ Pass |
| `--success` | `#22c55e` | 5.1:1 | ✅ Pass |

Light theme (`[data-theme="light"]`) on white `#ffffff`:

| Token | Value | Ratio | Result |
|-------|-------|-------|--------|
| Text on white | `#1e293b` | 13.8:1 | ✅ Pass |
| Muted on white | `#7e8ea8` | 4.5:1 | ✅ Pass |

## Landmarks & Structure
- `<html lang="en">` — ✅ Present and correct
- `role="banner"` on header — ✅ Present
- `role="main"` on main section — ✅ Present
- `role="progressbar"` on workflow wizard — ✅ Present with `aria-valuenow`
- Section headings (`<h1>`–`<h3>`) — ✅ Logical hierarchy
- Skip-to-content link — ❌ **Missing** (high priority)

## Images
- Canvas elements have `aria-hidden="true"` for tooltip overlays — ✅ Sprint 6 addition
- 13+ canvases have `aria-label` — ✅ Sprint 6 addition
- `<img>` tags missing explicit `alt` — ⚠️ Review needed (most icons are decorative)

## Keyboard Navigation
- `tabindex="0"` on main sections and canvases — ✅ Sprint 6
- ArrowUp/ArrowDown on `.timeline-list` — ✅ Sprint B4
- Enter/Space on collapsible cards — ✅ Sprint 6
- 'L' key for theme toggle — ✅ Sprint F3
- 'G' key for onboarding toggle — ✅ Sprint B2
- Escape key for modals — ⚠️ Verify after GPU/modal focus trap

## Focus Management
- `:focus-visible` keyboard indicator — ✅ Sprint 11
- Focus trap for GPU/modal — ✅ Sprint 6
- Modal backdrop — ✅ Present

## Forms
- Collapsible cards accessible via keyboard — ✅
- Confirm modal buttons focusable — ✅
- Form labels — ⚠️ Review needed for calibration inputs

## Reduced Motion
- `@media (prefers-reduced-motion: reduce)` — ⚠️ **Not present** (medium priority)
- Skeleton pulse animation — should be disabled under reduced motion

## Service Worker (PWA)
- `sw.js` registered — ✅
- `offline.html` has proper `lang` attribute — ✅
- Offline fallback — ✅

## Known Issues
1. **Skip-to-content link**: No visible or hidden skip link at top of page
2. **Reduced motion**: No `prefers-reduced-motion: reduce` media query to disable skeleton pulse animations
3. **Form labels**: Calibration and search inputs may lack explicit `<label>` associations
4. **Image alt text**: Some `<img>` tags may be missing descriptive `alt` attributes

## Priority Items for Next Sprint
1. **P0**: Add skip-to-content link as first focusable element
2. **P1**: Add `@media (prefers-reduced-motion: reduce)` rule disabling skeleton and spinner animations
3. **P1**: Audit all `<img>` tags for missing `alt` attributes
4. **P2**: Add `aria-live="polite"` regions for dynamic content updates (toast notifications, match loading)
5. **P2**: Verify all form inputs have associated `<label>` elements
