## What was fixed this session (2026-09-16 — elite-readiness implementation pass)

The approved "make it elite" plan, implemented in full. Every item below is
pinned by tests and committed separately. Final state: **5,164 unit tests,
0 failures, no ignores** (the "6 documented env-only failures" from the
previous session turned out to be two real bugs — see items 10/11 — plus a
workflow-structure test updated for the new release pipeline).

1. **xG honesty (core/xg_model.py)** — XP-GStatsBomb's meta no longer claims
   0.45 calibration when it is an uncalibrated logistic baseline; live
   pipeline (services/analysis/core.py) routes shots through the ACTIVE
   trained model, not the heuristic (tests/unit/test_live_xg_trained_path.py
   pins both, including the 2026-09-16 wrong-angle-convention failure mode).
2. **SoccerNet loader hardened (core/validation/soccernet_loader.py)** —
   verified against real extracted SNMOT-060 GT (750 frames); benchmark
   script consumes the extracted image-sequence layout; real e2e tracking
   benchmark ran on the RTX 4070.
3. **4-point homography confidence fusion (services/homography_service.py)**
   — with exactly 4 correspondences `findHomography` fits exactly, so
   error_px≈0 and confidence≈1.0 even for mirrored/garbage corners. The rich
   `validate_4corner_calibration` geometry score is now fused into the
   stored confidence (tests/unit/test_homography_validation.py).
4. **Tactical whiteboard UI (web/js/app-whiteboard.js + index.html)** — the
   19 orphaned `whiteboard_*` bridge slots now all have real callers: board
   CRUD, formation templates, run/pass generators, undo, ball placement,
   SVG export, DB-persisted annotations. Jest suite (20 tests) loads the
   real IIFE against a mock QWebChannel bridge.
5. **CRITICAL global bridge fix (web/js/app.js)** — the split modules
   (tactics, briefing, squad, coding, opponent, proanalytics, 3d) referenced
   a free variable `bridge` that never existed in the shipped bundle
   (app.js's bridge is IIFE-local + minified). Those features silently
   no-op'd at runtime. Now resolved via `window.__kawkab.bridge`, injected
   by the Python side.
6. **Settings + model-cache manager UI (web/js/app-settings.js)** — wires
   `get_settings_overview`, `get_model_cache_info`, `download_model`,
   `delete_cached_model`, `get_contract_alerts`. Plus a real Settings nav
   section replacing a banned placeholder id.
7. **DB-backed Shortlist/Contracts (services/storage_service.py)** — SQLite
   `StorageService` gained the 7 shortlist/contract methods that previously
   existed only on the Postgres adapter (migrations 016/017 tables were
   dead weight on SQLite). The Scout Portal's shortlist tab, which was
   frontend-local-only (persistence silently discarded), now persists
   through the real slots.
8. **Game Plan card (web/js/app-proanalytics.js)** — the orphaned
   `generate_game_plan` slot + `core/game_plan.py` are wired into Pro
   Analytics with one-click match-level analysis chaining.
9. **ModelManager sha256 pinning (core/model_manager.py)** — 5 YOLO
   weights sha256-verified against live downloads; two dead ReID URLs
   (404: boxmot v3 osnet_sportsmot, SoccerNet reid) removed — boxmot v19
   manages its own weights from TRAINED_URLS; manifest tamper-protection
   added (tests/unit/test_model_manager.py).
10. **cv2.groupRectangles removed in OpenCV 5 (services/
    raindrop_detection_service.py)** — hard crash on modern OpenCV, not an
    "env issue"; replaced with cv2.NMSBoxes + legacy fallback.
11. **Dead esd import in get_event (services/easy_soccer_service.py)** —
    re-imported esd after the client was resolved; any injected/unavailable
    client returned None. Import removed.
12. **Real onboarding wizard (web/js/app-onboarding.js)** — replaces the
    toast-only first-run stub: 4-step flow with live system check (GPU
    tier, model cache), one-click model download, import-vs-sample paths;
    modal focus-trap wired through ui.js helpers.
13. **Release automation (.github/workflows/release.yml)** — gated
    Windows/Linux PyInstaller desktop builds + wheel attached to the GitHub
    Release; publish job requires test + build-wheel + desktop matrix.
14. **myPy bug-mining** — fixed the misleading `-> BallSegment` contract lie
    in scripts/detect_events.py and confusing shadowings in
    periodization_service.py / opponent_dossier.py (style-level; the three
    suspected runtime bugs were verified benign).
15. **Modal focus trap + WCAG gates (web/js/ui.js,
    tests/unit/test_wcag_a11y.py)** — Tab cycling, Escape, focus save/restore
    in ui.js; new CI-enforced WCAG 2.1 AA structural assertions.
16. **OpenCV 5 / raindrop + easy_soccer fixes** — see items 10/11; full unit
    suite now green on this environment with zero documented env-failures.

