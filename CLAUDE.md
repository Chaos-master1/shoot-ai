# CLAUDE.md — Kawkab AI

Source of truth for working on this codebase. Supersedes AGENTS.md and every
other root-level markdown file (all archived under `docs/archive/` — kept for
history, not maintained). If something here conflicts with an archived file,
this file is right; the archived one was a snapshot in time.

## What this is

Kawkab AI is a football (soccer) analytics platform for amateur-through-elite
coaches: a PySide6 desktop app that runs YOLO11 player/ball tracking on match
video, detects events, computes 117 analytical models (xG, xT, VAEP, pitch
control, pressing, set pieces, ...), and surfaces them through a vanilla-JS
SPA served inside a `QWebEngineView` over `QWebChannel`. SQLite by default;
optional PostgreSQL backend and a separate multi-user FastAPI cloud server for
team collaboration. Bilingual (English/Arabic, full RTL).

The analytical depth is real and the test suite is real. The layers *between*
the models and the coach had multiple breaks — this file exists so they don't
come back.

## Correct commands (read this before running anything)

```powershell
# Desktop app
$env:PYTHONPATH="src"; python -m kawkab
# or explicitly:
$env:PYTHONPATH="src"; python -m kawkab gui

# Full unit suite (5,164 tests as of 2026-09-16, 0 failures)
$env:PYTHONPATH="src"; python -m pytest tests/unit/ -q

# Lint / types (installed as of 2026-07-30 -- see "Correct commands" note below)
python -m ruff check src/ tests/
python -m mypy src/kawkab --ignore-missing-imports

# Root-level suite (includes tests/test_load.py, tests/e2e/, tests/integration/)
$env:PYTHONPATH="src"; python -m pytest tests/ -q

# Frontend build (run after touching any src/kawkab/web/js/*.js file)
node scripts/bundle-js.mjs

# Frontend tests
cd src/kawkab/web && npm test

# Coverage
$env:PYTHONPATH="src"; python -m pytest tests/unit/ --cov=src/kawkab --cov-report=term
```

**Why `PYTHONPATH=src` in every command above:** `kawkab` is not installed
(editable or otherwise) into the system Python on this machine — every
dependency was installed directly, but the project itself never went
through `pip install -e .` / `uv sync`. A fresh clone following the README's
own `uv sync` install flow gets a real editable install and doesn't need
`PYTHONPATH` at all (`uv run python -m kawkab` just works). Both are fine;
this file documents what actually works on the machine this audit ran on.

**Use the system `python`, not `.venv\Scripts\python.exe`.** This repo has two
Python installs: a `.venv` at 3.12 (has almost nothing installed in it) and
the system `python` at 3.11.9 (has every real dependency — PySide6, torch,
opencv, ultralytics, pytest, everything). The system one is what actually
works on this machine; `.venv` looks "correct" by version number but is
functionally empty. `requires-python` was `">=3.12"` — which made the
*correct* interpreter look unsupported by the project's own metadata, despite
demonstrably running the full suite clean. Nothing in `src/` uses 3.12-only
syntax (checked: no PEP 695 `type X = ...` / `class Foo[T]`), so relaxed to
`">=3.11"` (also updated `[tool.ruff] target-version` and `[tool.mypy]
python_version` to match). If you're setting up a fresh clone, `pip install
-e ".[test,cloud]"` into whatever interpreter you use (see the packaging note
below — that command was broken until this session's fixes).

**`ruff`/`mypy`/`pytest-xdist` were added to `pyproject.toml`'s `dev`/`test`
extras this session but never actually installed** — they were configured,
not verified, until the final verification pass caught it (`python -m ruff`
→ `No module named ruff`). `pip install -e ".[dev,test]"` also currently
fails on this machine's system Python (3.11.9) for an unrelated reason: pip
re-resolves the full dependency graph and *that* re-triggers `requires-python`
enforcement against whatever it was at the time — which is exactly the
metadata mismatch fixed above, but if you hit an install error that looks
like it again, run `pip install ruff mypy pytest-xdist` directly (bypasses
the project's own metadata resolution entirely) rather than fighting it.

**`KAWKAB_TEST_MODE` does nothing.** It's referenced only in a doc file
(`CLEANUP_PROMPT.md`), never read by any source file. Harmless to set, but
don't treat it as meaningful — the AGENTS.md-documented test commands that
included it were following a convention that was never implemented.

## Architecture map

```
src/kawkab/
  core/            117 analytical modules (pure functions/classes: xg_model,
                   xt_model, vaep, pitch_control, pass_network, ...). No I/O,
                   no bridge/service knowledge — importable and testable in
                   isolation. 59 of 117 are wired into services/bridge; 58 are
                   built + tested but not reachable from the UI (see below).
  services/        Business logic + I/O: cv_service (YOLO+tracking),
                   analysis/ (AnalysisService, split into mixins: core.py,
                   passing.py, ...), storage_service.py (SQLite),
                   postgres_storage.py (Postgres — NOT a drop-in twin of
                   storage_service, see "Storage backend divergence" below),
                   116 total service classes.
  ui/bridge.py     QWebChannel bridge exposed to JS as `kawkab.<method>()`.
                   Thin dispatcher only — every @Slot is a 2-line delegator
                   into ui/bridge_handlers/*.py. 325+ slots.
  ui/bridge_handlers/  AnalysisHandler (4,100+ lines, the biggest one),
                   AuthHandler, CodingHandler, ExportHandler, ExternalHandler,
                   LifecycleHandler, ProviderHandler, StorageHandler,
                   VideoHandler. Real implementations live here, not in
                   bridge.py.
  web/             Vanilla JS SPA. index.html (3000+ lines) + css/main.css
                   (5000+ lines) + js/*.js (28 IIFE files, concatenated by
                   scripts/bundle-js.mjs into dist/app.bundle.min.js, which
                   is what index.html actually loads).
  cloud/           Separate FastAPI multi-user server (auth, teams, sync) —
                   optional, desktop app never needs it.
  migrations/      27 numbered SQLite migration files, applied in order by
                   core/migration_manager.py. pg_schema.sql (no number
                   prefix, correctly skipped by the numeric-migration
                   scanner) is the separate Postgres schema.
```

## Invariants (why each of these exists — don't undo them without reading the reason)

- **Every `bridge.X(...)` / `kawkab.X(...)` call in `web/js/*.js` needs a
  matching `@Slot` in `ui/bridge.py`.** Enforced by
  `tests/unit/test_bridge_contract.py` — it fails the build if this drifts.
  6 calls were silently broken this way (Dashboard, Player Compare, 3D pitch,
  Scout search) before this session; the JS caught the resulting exception
  and degraded quietly, so nothing crashed loudly and nobody noticed.
- **Frontend load order lives in `scripts/frontend-manifest.json`, never in
  index.html's `<script>` tags.** `scripts/bundle-js.mjs` used to parse the
  file list out of index.html's script tags, bundle them, then *delete those
  tags from index.html*. The next run had nothing left to parse and would
  have silently emitted a near-empty bundle. The manifest is now the single,
  version-controlled source of truth; index.html's bundle `<script>` tag is
  permanent, hand-maintained markup. If you add a new IIFE file to
  `web/js/`, add it to the manifest, not to index.html.
- **`services/` delegates metrics to `core/`, never re-implements them.**
  `services/analysis/passing.py::_compute_pass_network` used to be its own,
  cruder pass-network builder (no betweenness, no eigenvector centrality)
  that happened to share a name with the real, richer `core/pass_network.py`.
  Same pattern in `advanced_event_detection_service.py::_detect_progressive_actions`
  vs `core/progressive_actions.py` (flat 10m-distance threshold vs the real
  ratio-of-remaining-distance + attacking-third + zone-aware logic driven by
  `game_constants.py`). Both now delegate. If you're about to write threshold
  logic in a `services/` file for something that sounds like an existing
  `core/` concept, check `core/` first.
- **norfair `Detection.points` must be `(N, 2)` for the `"iou"` distance
  function** (`services/norfair_tracker.py`). An older norfair accepted
  4-corner points; norfair>=2.3 validates strictly and raises
  `AssertionError: Bounding boxes must be defined as np.array with (N, 4)
  shape` (the "4" there means 4 *numbers* — 2 points × 2 coords — not 4
  points). `tests/unit/test_norfair_tracker.py` stubs the real package
  unconditionally now (it used to skip stubbing if norfair was *already*
  imported by something upstream, which meant the test's own isolation
  silently depended on import order and which norfair version happened to be
  installed).
- **`playwright` is a required dependency despite no kawkab code importing
  it.** `EasySoccerData` (import name `esd`)'s Sofascore client does `import
  playwright` internally without declaring it as *its own* dependency. A
  bare `grep -r playwright` across `src/` finds nothing and looks safe to
  remove — it isn't. Confirmed by installing into a clean venv and running
  `test_easy_soccer_service.py`.
- **CSS custom properties: map new names onto existing tokens, don't invent
  literals.** `--accent`, `--accent-color`, and `--primary-color` were three
  independently-invented names (in three different features, at different
  times) for the same "this is the active/selected one" semantic, each with
  a different undefined-elsewhere fallback color (teal, dark green, medium
  green) that didn't match the app's real primary blue. All three now alias
  `var(--primary)`. See `:root` in `main.css` for the full list and why each
  exists.
- **`KAWKAB_JWT_SECRET` must be ≥32 characters.** `cloud/auth.py` enforces
  this now (`MIN_JWT_SECRET_LENGTH`). HS256 with a short key is a real
  forgery risk, not just a lint warning — PyJWT already warns about it
  (`InsecureKeyLengthWarning`), this makes it a hard failure instead.
- **`tool.hatch.metadata.allow-direct-references = true` is required and
  load-bearing.** The `sports` extra pins `sports @ git+https://...`
  (a VCS dependency). Hatchling validates *every* `optional-dependencies`
  group at metadata-generation time regardless of which extra you're
  actually installing — without this flag, `pip install -e ".[anything]"`
  fails immediately, before pytest ever runs a single test. This is not
  about the `sports` extra specifically; removing this flag breaks
  installing the project at all.
- **`core/security.py`'s `SecurityValidator` is the one, real, actually-used
  validator. If a `services/storage/*.py` file's `try: from
  kawkab.core.security import SecurityValidator / except ImportError: class
  _SecurityValidator: ...` fallback defines a method the real class
  doesn't have, that is not a harmless backup — it is proof the real class
  is missing something callers depend on.** The `except ImportError` branch
  never runs (the import always succeeds), so whatever the fallback defines
  and the real class lacks was *always* calling a nonexistent method,
  caught by the caller's own broad `except Exception`, silently returning a
  failure value forever. Found via mypy (`attr-defined` on
  `"type[SecurityValidator]"`), confirmed via `test_storage_submodules.py`,
  which had monkey-patched 5 of these methods onto the real class at
  *module import time* to make its own tests pass — meaning the test
  suite's green status was actively hiding this, not catching it. All 6
  missing methods (`validate_track_id`, `validate_positive_float`,
  `validate_float_range`, `validate_event_type`, `validate_event_dict`,
  `validate_int`) are now on the real class; the monkey-patch is deleted.
  See "What was fixed this session" for the full blast radius. If you add a
  new fallback stub method to any `services/storage/*.py` file, add it to
  the real `SecurityValidator` in the same change, or don't add the
  fallback at all — an unreachable branch that *looks* like a safety net is
  worse than no safety net.
- **mypy `strict = true` was configured in `pyproject.toml` months before
  anyone actually ran `mypy` against it.** It produces 2,885 errors across
  203 of 299 files — 82% of them pure annotation-completeness noise
  (`type-arg`, `no-untyped-def`, `no-untyped-call`, `no-any-return`) on a
  125k-LOC codebase that was never written with complete type coverage.
  Config alone proves nothing; a gate has to actually run before you know
  if it's honest or aspirational. Dialed back to non-strict (820 errors,
  mostly the same annotation-completeness category) — the real bugs mypy
  is good at (`name-defined`, `attr-defined` on nonexistent methods) surface
  either way. See Known Gaps for the current count and how to use it.

## Session history (where the "what was fixed" logs went)

This file used to grow a "What was fixed this session" section every pass —
five of them by 2026-09-16, ~1,000 lines of narrative. They are preserved
**verbatim** under [`docs/archive/session-logs/`](docs/archive/session-logs/)
(read the README there for the index). New passes: append a log file there;
do not grow this file again.

Everything from those logs that is still load-bearing lives in the two
sections that matter: **Invariants** above (with the reason each exists) and
**Known gaps** below. One-paragraph version of the history:

- **2026-07-30 audit** — the product was unreachable (`python -m kawkab`
  printed help and exited; `app.py` couldn't import), 6 frontend→bridge
  calls were silently broken, the CI gates were hollow, and the entire
  `services/storage/` save path raised on every call. All fixed; gates made
  honest. → `2026-07-30-audit.md`
- **2026-09-07 elite-readiness** — xG/PSxG/xT trained on real StatsBomb
  data (Brier 0.0707 vs StatsBomb's own 0.0667), reproducible validation
  CLI, StatsBomb import path, Pro Analytics tranches 1+2 (25 report
  blocks). → `2026-09-07-elite-readiness.md`
- **2026-09-16, three passes** — broken-HEAD repair + the xG
  angle-convention bug (model had learned wide shots score more) + first
  real production-tracker MOT benchmarks (MOTA 0.78 e2e on SoccerNet GT);
  the 16-item "make it elite" implementation pass (tactical whiteboard,
  onboarding wizard, Settings UI, the critical `window.__kawkab.bridge`
  fix, sha256-pinned model manifest, gated release pipeline); and a
  full-codebase pre-release audit (4 real defects). → the three
  `2026-09-16-*.md` logs

## Known gaps (found, understood, deliberately not fixed this session — read before assuming any of these "just work")

- **Storage backend divergence.** `StorageService` (SQLite) and
  `PostgresStorageAdapter` (Postgres) are not the same interface with two
  implementations — they've drifted into genuinely different feature
  sets. The rough "20/57/10" counts from the original audit turned out to
  be very close, but were estimates; here's the exact, code-introspected
  breakdown (2026-07-30, via `inspect.signature` over both classes, not
  manual reading — see method for reproducing this if the classes change):
  **88 SQLite public methods, 124 Postgres, 68 shared by name, 20
  SQLite-only, 56 Postgres-only, 10 shared-name-but-different-signature.**
  - **SQLite-only (20)**: `audit_log`, `change_password`, `create_user`,
    `delete_session`, `get_all_users`, `get_audit_log`, `get_gps_samples`,
    `get_gps_sessions`, `get_player_acwr`, `get_player_gps_summary`,
    `get_user_by_id`, `get_user_by_username`, `record_failed_login`,
    `save_acwr`, `save_gps_samples_bulk`, `save_gps_session`,
    `save_session`, `update_gps_session_stats`, `update_user_login`,
    `validate_session` — local user auth/sessions/audit log, plus all
    GPS/ACWR tracking. A Postgres/cloud deployment has no GPS import or
    local-auth-session feature at all.
  - **Postgres-only (56)**, minus 4 that are generic query primitives
    (`execute`, `executemany`, `fetch`, `fetchrow` — a different API
    style, not really "missing SQLite features") = **52 real gaps**.
    **Confirmed NOT safe to delete as "duplicates"**: 10 of them
    (`save_shortlist_entry`/`get_shortlist`/`update_shortlist_entry`/
    `delete_shortlist_entry`, `save_injury`/`get_injuries`/
    `update_injury`, `save_contract`/`get_contracts`/
    `get_contracts_expiring_soon`) look like duplicates of
    `shortlist_service.py`/`injury_tracker.py`/`contract_tracker.py`, but
    those three dedicated services are hardcoded to `sqlite3.Connection`
    (checked directly) — meaning for Postgres deployments, the
    "duplicate" methods on `PostgresStorageAdapter` are the *only*
    implementation of shortlist/injury/contract tracking that exists.
    Deleting them would remove a real feature, not dead code. **Update
    2026-09-16:** the shortlist/contract half of this gap is now closed
    from the SQLite side — `StorageService` gained the 7 missing methods
    (save/update/get/delete_shortlist_entry, save_contract,
    get_contracts, get_contracts_expiring_soon, backed by the existing
    migrations 016/017 tables) and the bridge + Scout Portal UI now call
    the storage-backed API for both DB modes, replacing the previously
    frontend-local shortlist that silently discarded persistence. The
    injury-tracker trio (`save_injury`/`get_injuries`/`update_injury`)
    remains Postgres-only. The
    remaining ~42 Postgres-only methods (seasons, match comparisons,
    analysis quality, batch jobs, cache, card/psychology events, collab
    comments/mentions/users, wearable sessions, medical history, audit
    events, settings, schema version) genuinely have no SQLite
    equivalent, or SQLite gets the same feature through a different,
    dedicated service instead (same shortlist/injury/contract pattern).
  - **10 signature mismatches (exact params)**: `get_coding_tags_by_type`
    (`event_type` vs `tag_type`), `save_advanced_metrics` (Postgres adds
    `metrics`/`category` params SQLite doesn't have),
    `save_advanced_metrics_bulk` (`metrics` vs `metrics_list`),
    `save_correction` (SQLite: `event_id, correction_type,
    original_value, corrected_value`; Postgres: a much wider, looser
    `match_id_or_event_id, correction, correction_type, original_value,
    corrected_value, event_id, match_id, field, old_value, new_value,
    reason` — accepts several overlapping calling conventions at once),
    `save_report` (`report_text_or_language` overload param on Postgres
    only), `update_match_analysis` (`analysis_data` param Postgres-only),
    `update_match_apifootball`/`update_match_bzzoiro`/
    `update_match_football_data` (all three: `data` param Postgres-only),
    `update_player_profile_face` (`face_embedding_json` vs
    `face_embedding_or_path`). Any code path calling one of these 10 by
    positional args instead of keywords would silently call the wrong
    parameter across backends.
  - Fixing this properly still needs either the ~52 real-gap methods
    written and verified against a live Postgres instance, or a real
    architectural unification (e.g. making `shortlist_service.py` etc.
    backend-agnostic so `PostgresStorageAdapter` can delegate to them
    too) — both bigger than what this pass attempted. What changed this
    session is that the exact list now exists, so that work no longer
    has to start with re-deriving it.
  - If you switch a deployment to `KAWKAB_DB_URL` (Postgres), do not
    assume feature parity with SQLite — check the specific methods above.
- ~~**22 dead nav links** in the left sidebar~~ — **fixed 2026-09-16, and
  the original diagnosis understated the severity**: the links did NOT
  "do nothing". Sidebar navigation is handled by an inline
  `syncSidebarClicks` handler in index.html that toggles `hidden` on
  every `.section` whose id doesn't match the clicked link's
  `data-section` — so clicking any of the 23 links whose target id had
  never existed (Timeline, xG, xT, VAEP, Charts, Heatmap, Pass Network,
  Momentum, Transitions, Finishing, Set Pieces, Phases, Tactics Report,
  Game Plan, Compare, Shortlist, Contracts, Recruitment, Settings,
  Search, Import, Data Export, plus a 24th, Sandbox, never counted)
  **hid every section and blanked the whole app**. The mobile bottom
  nav's Settings button had the same bug. Resolution per link: the 14
  Analysis/Tactics links whose views are panels INSIDE the Results page
  (xG/xT cards, charts, timeline sidebar, heatmap, pass network,
  momentum, set pieces, phases, report, transitions/finishing panels)
  now route to `results-section`; Compare → `professional-section`
  (where the pc-* compare panel actually lives); Sandbox →
  `results-section` (its `tactical-sandbox-section` div is embedded
  there — it was never a `.section`); Import → `results-section`
  (vendor-import panel); Data Export → `results-section` (export
  buttons); Search link **removed** (global search is the always-visible
  header box); Shortlist/Contracts/Recruitment and Settings and Game
  Plan **removed** (no UI exists anywhere — Settings had no section, no
  modal, no JS at all; Game Plan is backend-only
  `generate_game_plan`); bottom-nav Settings → Calibration (a real
  section that was previously unreachable from mobile nav).
  `calibration-section` was wrongly on the original dead list — it is a
  real `<section>` and worked. Regression gate:
  `src/kawkab/web/tests/nav-integrity.test.js` fails the frontend build
  if any `data-section` target (sidebar or bottom nav) stops resolving
  to a real `.section`. Remaining true gaps (no UI yet, link removed
  rather than faked): Shortlist, Contracts, Recruitment, Settings,
  Game Plan.
- ~~**34 frontend Jest failures**~~ — fixed (2026-07-30, same session as the
  harness fix below). All three were exactly "stale assertions against
  DOM/JS that moved on," confirmed via root-cause investigation rather than
  patched blind: `ui.test.js` (21) loaded `ui.js` with a regex expecting
  ES-module `export function` syntax that hasn't existed since `ui.js` was
  refactored to the IIFE-attaches-to-`window` convention every other
  `web/js/*.js` file uses — fixed by loading it the same way
  `js/__tests__/router.test.js` already does (`require()` + read the
  resulting `window.*` properties) rather than reinventing a loader.
  `app-charts.test.js` (12) mocked `Chart` without a `.register` static
  method; real Chart.js v3+ needs it and `app-charts.js` calls it at load
  time — added `MockChart.register = jest.fn()`. `analytics.test.js` (1)
  expected error output wrapped in a `.error`-class element; the actual
  (dead, see next bullet) code just does `container.textContent =
  data.error` — fixed the assertion to match.
- ~~**`web/js/analytics.js` was dead code with its own test suite**~~ —
  investigated and deleted (2026-07-30, follow-up pass), along with
  `tests/analytics.test.js`. Confirmed safe to delete, not "wire it up":
  it `import`s from `./core.js`, which doesn't exist anywhere in
  `web/js/`, and `import { showToast, ... } from './ui.js'` against a
  `ui.js` that is a plain IIFE with no `export` at all (ES-module syntax
  against a non-module file — this could never have worked, bundled or
  not). Its 9 backend calls (`get_xa_report`, `get_vaep_report`,
  `get_xt_report`, `get_momentum_index`, `get_set_piece_report`,
  `get_pass_flow`, `get_match_narrative`, `get_progressive_report`,
  `get_defensive_report`) don't exist anywhere in `bridge.py` either —
  unlike the 113 unreachable-but-real slots below, these were never
  implemented on the backend at all. Most of its own render functions
  were empty stubs (`function renderX(data) {}`). Net effect of deleting:
  zero working functionality lost; Jest suite unaffected (114/114 still
  pass). The underlying feature gap this abandoned prototype was reaching
  for (xA/VAEP/xT/momentum/set-piece/progressive/defensive report views)
  is still real and now folds into the existing "58 orphaned `core/`
  modules" gap below rather than being tracked twice.
- **Jest's parallel worker mode crashes** (`StackOverflowException`) on this
  Node/`jest-environment-jsdom` combination — `--runInBand` (sequential)
  works fine. Fixed via `maxWorkers: 1` in `jest.config.cjs`, so `npm test`
  is usable again, but the underlying version-compat crash itself wasn't
  root-caused.
- ~~**~8 remaining `SELECT *` sites in `postgres_storage.py`**~~ — fixed
  (2026-07-30, follow-up pass). Actually 21 sites across 15 tables, not ~8.
  `src/kawkab/migrations/pg_schema.sql` turned out to be a complete,
  authoritative column source (43 tables, matches every `INSERT` in the
  same file) — no live Postgres connection was needed after all, just
  cross-referencing against it. Verified via `ast.parse` (syntax) and the
  14 non-integration tests in `test_postgres_storage.py`; the 20
  integration tests still require a real `KAWKAB_DB_URL` connection to
  exercise, same as before. Note: `pg_schema.sql` itself has a latent bug
  found in the process — `seasons` is `CREATE TABLE IF NOT EXISTS`'d
  *twice* with two different column sets (line ~30 and line ~186); since
  `IF NOT EXISTS` makes the second a no-op against a fresh DB, only the
  first (narrower) definition ever actually applies. Not fixed — flagged
  here since it means the second definition's extra columns
  (`team_name`, `competition`) don't exist in any real deployment, despite
  looking like part of the schema.
- ~~**91 `except ... : pass` blocks**~~ — the 4 flagged-as-highest-risk
  files audited per-site (2026-07-30, follow-up pass); the rest (spread
  thin across the remaining files) intentionally not swept, see below.
  - **`cv_service.py` and `advanced_event_detection_service.py` had 2 real
    bugs, not just silence** — both follow the identical shape: on
    `homography_matrix.pixel_to_pitch()` failure, the code fell back to
    the *raw pixel* coordinates and kept going as if they were pitch-space
    meters. `_detect_dribbles`: the resulting "distance_m" (compared
    against a 1.0m threshold) would be the raw pixel distance --
    trivially over threshold, so any homography hiccup mid-match produced
    a false-positive dribble event with a nonsense distance. `_detect_clearances`:
    same mechanism, but for `"speed_mps"` against an 8 m/s threshold --
    guaranteed false-positive clearance the instant homography failed once.
    Both fixed: reset the tracking window instead of mixing pixel- and
    pitch-space points in it. Regression tests in
    `test_advanced_event_detection_service.py` use a `_FailingHomography`
    that always raises, confirming no false positives now.
  - **`_detect_corners`/`_detect_free_kicks`/`_detect_throw_ins` in the
    same file already used a `-1` sentinel + explicit `>= 0` guards before
    using pitch coordinates** — i.e. the same class of bug the two above
    had, but already correctly handled. Only missing diagnostic logging
    on the swallowed exception, added for consistency.
  - **`audit_service.py`'s except-blocks are correctly designed as-is** --
    every one already returns a safe, honest sentinel (`""`, `0`, `[]`,
    `{}`) rather than fabricating data; this file is audit-log
    bookkeeping, not match analytics, so a swallowed exception here means
    "lost a log entry," never "silently wrong stat." Also found and fixed
    an additional `SELECT *` in `get_events()` missed by the earlier
    SQLite `SELECT *` cleanup (columns: `id, timestamp, action,
    entity_type, entity_id, details_json, user, prev_hash` -- matches the
    explicit-column style `_get_last_hash()` in the same file already used).
  - **`gpu_acceleration.py`'s except-blocks are also correctly designed**
    -- this file is hardware-capability probing (which GPU backend,
    which VRAM tier), not analytics; most blocks already narrow to
    specific exception types (`FileNotFoundError`, `subprocess
    .TimeoutExpired`, `ImportError`, `ValueError`/`IndexError`) rather
    than bare `Exception`, and every fallback is an honest degraded mode
    ("no GPU backend detected -- using CPU"), not a fabricated result.
    Not logging every miss is correct here, not a gap -- most machines
    legitimately lack `nvidia-smi`/CUDA, and logging that as a warning on
    every single run would be noise, not signal.
  - **Remaining `except: pass` sites outside these 4 files** were not
    swept this session -- the 91 count was concentrated enough in these 4
    that auditing them covered the highest-risk surface; a full repo-wide
    sweep is still open if you want it.
- **~195 hardcoded hex colors + 204 inline `style=` attributes** remain in
  the frontend (down from the original ~295/310 estimate — see the CSS
  design-tokens entry above for the white/black primitive migration, the
  `.nav-item.active` contrast fix, and the two real `--text`/
  `--bg-elevated` light-theme bugs already fixed). What's left is mostly
  semantic status colors (`#dc2626`/`#16a34a`/`#d97706`/`#2563eb`, etc.)
  needing per-site judgment, not a blanket regex: some are already
  `var(--token, #stale-fallback)` (safe, mechanical cleanup — the fallback
  is dead since the token is unconditionally defined), but others, like
  `.pitch3d-legend-item .home-dot/.away-dot` and
  `.btn-pitch3d-home/away`, are "team side" colors that coincidentally
  match `--primary`/`--danger`'s current values and should map onto the
  already-existing but under-used `--team-home`/`--team-away` tokens
  instead — not the brand/status ones, or a future retune of "danger red"
  would unintentionally repaint the away-team markers too. A few (the
  `.fs-btn-*` fullscreen-tagging gradient family — goal/shot/pass/tackle/
  foul/corner/save/card, each its own two-stop gradient) are deliberately
  decorative, self-consistent one-offs not worth forcing onto unrelated
  tokens. A real token layer (spacing scale, type scale) still hasn't been
  built — these two passes fixed what was *broken* and what was safely
  mechanical, not a full design-system migration.
- **57 of 117 `core/` analytical modules are built and tested but not
  wired into the UI** (down ~12 as of 2026-09-07 — the Pro Analytics
  handler surfaces EPV, OBV, pass_flow, pressing_clusters, duels,
  ball_recovery, box_entries, switch_of_play, crossing, set_pieces,
  through_balls, offball_metrics; Season Form/xA/pressing-efficiency
  wired earlier). Remaining by area (full list shrunk accordingly):
  - Attacking: `xa_split` (xA itself now wired, see below), `expected_pass`, `through_ball`,
    `crossing_analysis`, `crossing_xg`, `corner_xg`, `box_entries`,
    `packing`, `switch_of_play`
  - Build-up/possession: `pass_network` (delegated-to but not surfaced as
    its own view), `passing_lanes`, `passing_triangles`, `pass_flow`,
    `carry_xt`
  - Defending: `defensive_actions`, `defensive_xt`, `pressing_efficiency`
    (trap-to-shot conversion now wired, see below; trap-to-goal rate and
    press-recovery-attack from the same module still aren't),
    `pressing_clusters` (spatial zone data — needs a pitch-map
    visualization, not a stat card; not attempted), `ball_recovery`, `duel_analysis`
  - Set pieces: `set_piece_analysis`, `set_piece_xt`
  - Player/squad: `obv`, `epv` (both need per-frame tracking data that
    isn't reliably persisted for most matches — see below), `offball_metrics`,
    `velocity_analysis`, `lineup_optimizer`
  - Season/admin: `fixture_difficulty`, `referee_analysis`,
    `suspension_tracker`, `formation_effectiveness`, `confidence_intervals`,
    `uncertainty`
  - Also fully untouched: `ball_physics_pitch_control`, `match_scripting`,
    `match_timeline`, `model_comparison_service`, `model_manager`,
    `pattern_detection`, `scoreline_distribution`, `scout_report_upgrade`, `xg_chain`,
    `xt_confidence`, `game_state`, `event_schema`, `export_converters`,
    `config` (the *tracking-pipeline* config, unrelated to the
    `Settings` class reconstructed this session), `fatigue_model`,
    `flank_analysis`, `form_analysis` (partially wired — see Season Form
    above), `influence_map`
  - **`OBV`/`EPV` specifically need full per-frame tracking data**
    (`frames: [{timestamp, possession, ball_pos, home_positions,
    away_positions}, ...]`), which the `tracking_frames` table only
    persists when `CVService.process_video()` is called with a
    `storage_service` param — not the default path. Wiring these without
    checking for that data first would silently show an empty/misleading
    report for most matches.
  - **PSxG unification completed 2026-09-16 (earlier pass):** the two
    hand-tuned alternate implementations (`core/psxg_improved.py`,
    `core/psxg_model_trained.py`) were DELETED; their behavioral contracts
    were ported into `tests/unit/test_psxg_contracts.py`. `core/psxg_model.py`
    (fitted coefficients, kept legacy API for `goalkeeper_analytics`) is
    now the single PSxG implementation. Still open: none of the PSxG path
    is used by the live CV pipeline's event path yet
    (goalkeeper_analytics is the only consumer of psxg_model).
- **113 of 325+ `@Slot` methods are never called from any JS file**
  (down from the original audit's count as this session wired a few more
  in) — includes the entire tactical whiteboard feature (19 slots, 45
  passing tests, zero UI). `tests/unit/test_bridge_contract.py`'s
  `test_unreachable_slots_inventory` prints the current live list — run it
  with `-s` rather than trusting this document, since the exact list will
  drift as more gets wired in.
- **Accessibility: real gaps found, some fixed, some flagged** — see the
  dated entry above for the full detail. Fixed: coding-matrix button
  text contrast, light-theme `--text-muted`, light-theme focus ring,
  dashboard `aria-live` coverage. Still open: dark-theme `--text-muted`
  vs `--bg-elevated` (4.04:1, marginal), white-on-`--success`/`--warning`
  badge text at small sizes (a brand-color judgment call, not made
  unilaterally), a full keyboard/screen-reader walkthrough, a WCAG 2.2
  target-size (24×24px) audit, and confirming whether the
  `[data-focus-trap]` CSS marker is backed by any real JS focus trap
  (found zero call sites setting the attribute).
- **XSS sweep: real bugs fixed, most sites spot-checked not hand-read.**
  See the dated entry above. Fixed: 3 files silently missing `escapeHtml`
  entirely (broken reference, not just unescaped output), a shadowed
  duplicate `escapeHtml` in `app.js` missing quote-escaping, a
  cross-user-reachable gap in the cloud-team-list renderer, and
  `calibration.js` having no escaping at all. ~280 of ~334 total
  `innerHTML` sites (concentrated in `app-tactics.js`/`app-scout.js`/
  `app-squad.js`/`app-highlight.js`/`app-briefing.js`/`events.js`) were
  confirmed via heuristic search to already use their own correctly-scoped
  `escapeHtml`, not read site-by-site by hand.
- **820 mypy errors remain** (non-strict; see the mypy Invariant above for
  why strict mode isn't the baseline) across ~130 files. Triaged by
  category, not by file: `attr-defined` (optional-attribute-without-a-
  None-check in `mujoco_ball_service.py`, cv2 stub incompleteness,
  mixin-composition false positives in `services/analysis/*.py` where
  `self.pitch_length` etc. really is set — just on the composed class, not
  the mixin mypy is checking in isolation) and `object has no attribute
  append` in `reasoning_service.py` (a heterogeneous `dict` literal mixing
  `int` and `list` values, inferred too broadly — runs fine) were sampled
  and are noise, not bugs. The two categories that *were* bugs
  (`name-defined`, `attr-defined` on `"type[X]"` i.e. calling a
  classmethod/staticmethod that doesn't exist) were mined exhaustively
  this session — see the SecurityValidator/ai_v2_list_convs/
  quality_scoring/card_detection/arabic_glossary fixes above. Re-run
  `mypy src/kawkab --ignore-missing-imports` and grep for those two error
  codes before assuming any *new* mypy error is noise.
  - **Follow-up sweep (2026-07-30) of `union-attr` and `return-value`,
    the two next-most-plausible-for-real-bugs categories**: mostly the
    same "mypy can't trace a guard through a separate method/function
    call" false positive already documented above (`self._conn`
    Optional-narrowed by `_ensure_initialized()`, `AsyncClient | None`
    narrowed by a lazy `_ensure_client()`-style setup, `_fernet` narrowed
    by `init_fernet()` in `core/encryption.py`) — confirmed, not fixed,
    since "fixing" would mean threading `assert`/inline-`if` everywhere
    mypy needs to see the narrowing, for zero behavior change. Found 2
    genuine issues in the same sweep: `bzzoiro_service.py::get_match_detail`
    called `.get()` on a value that could be a bare `list` (not just a
    `dict`) if the API ever responds unusually for a single-match-detail
    endpoint — added an `isinstance` guard. `player_profile_service
    .py::create_profile` is typed to return a non-Optional `PlayerProfile`
    but its last line delegated to `get_profile()`, which *is* Optional;
    callers (`bridge_analysis.py:950` among others) index straight into
    `profile.id` with no None-check. Fixed by raising a clear
    `RuntimeError` if the post-insert re-fetch is ever empty, instead of
    letting `None` silently satisfy a non-Optional return type. Also
    cleaned up `multi_camera_service.py::_merge_tracks`, which had a
    pointless `return track_groups` inside a `-> None` function (the one
    caller never used the return value) — cosmetic, zero behavior change.
- **115 ruff errors remain on the ~93 files touched this session** (of an
  original 590 before autofix + per-file-ignores; 2,073 across the whole,
  never-linted-before codebase). Dominated by `F841` unused-variable (38,
  spread across a dozen files, all pre-existing, none in code added this
  session — checked `test_health_endpoints.py`'s specifically since it's
  the one this session had already touched for an unrelated reason; it
  was genuine dead code with a misleading comment, fixed). Not chased
  further — mass-editing pre-existing unused-variable sites in files
  touched only incidentally is exactly the kind of scope creep this audit
  was trying to avoid elsewhere.
- **`pytest tests/ -n auto` (full root tree, all of it) is not clean —
  found 3 more issues while running it for the first time as part of the
  final verification pass, past this two fixed above:
  1. ~~**`test_bridge_advanced_metrics_wiring` and
     `test_bridge_frame_skip_parameter`**~~ — **fixed 2026-09-07, and
     the original diagnosis was wrong**: the failures were NOT the
     tempdir-path policy — they were a stale `FakeCVService
     .process_video` stub missing the `match_id`/`storage_service`
     parameters the real method takes (every analyze_match call
     TypeError'd into error-payload asserts). The stub now mirrors the
     real signature; 4/4 pass.
  2. **Tests that pass individually can fail when the *entire* `tests/`
     tree runs together** (confirmed: `test_cv_service_model_manager.py`
     and all of `tests/e2e/test_e2e_video_pipeline.py` pass 100% clean in
     isolation, but showed `TypeError: CVService() takes no arguments`
     when run as part of the full tree under `-n auto`). This is
     `pytest-xdist` worker-process co-scheduling exposing some
     `sys.modules` stub leaking from one test file into another's process
     — the same *species* of bug as the `test_norfair_tracker.py` leakage
     already fixed this session (see the norfair Invariant above), just a
     different pair of files, and distribution/order-dependent so it may
     not reproduce the same way every run. Not root-caused to the specific
     leaking file — would need the same kind of `sys.modules` save/restore
     audit across the stub-installing test files that
     `test_norfair_tracker.py` got.
  3. **`test_e2e_pipeline.py::TestCoverageConfig::test_coverage_fail_under_is_50`**
     ~~fixed before 2026-09-07~~ — now asserts pyproject's
     `fail_under = 65` (passes; see `tests/e2e/test_e2e_pipeline.py`
     line ~227 for the current assertion).
- **`test_bridge_contract.py` only checks that `bridge.py` has a `@Slot`
  for every JS call — not that the handler method the slot delegates to
  (`self._analysis.X(...)`, `self._auth.Y(...)`, etc.) actually exists.**
  ~~Closed 2026-09-07~~: `test_every_slot_delegation_targets_a_real_handler_method`
  parses bridge.py's `self._X.method(...)` delegations and verifies each
  against the real handler modules' method sets. Verified fail-positive
  by planting a fake broken delegation.
- **2 more shadowed-duplicate methods in `bridge_analysis.py`** — ~~fixed
  2026-09-07~~: both merged (see "What was fixed" above); `ruff
  --select F811` against `ui/bridge_handlers/` is clean as of this
  session. The live-video xG event path still calls the legacy heuristic
  `compute_xg_from_dict` in several places other than the fixed
  `compute_goals_added` — migrating every call site to
  `compute_xg_trained_from_dict` is open (grep
  `compute_xg_from_dict` in services/ + bridge_handlers/).

## Graphify

This project has a knowledge graph at `graphify-out/` with god nodes,
community structure, and cross-file relationships.

When the user types `/graphify`, invoke the `Skill` tool with
`skill: "graphify"` before doing anything else.

- For codebase questions, first run `graphify query "<question>"` when
  `graphify-out/graph.json` exists. Use `graphify path "<A>" "<B>"` for
  relationships and `graphify explain "<concept>"` for focused concepts.
  These return a scoped subgraph, usually much smaller than
  `GRAPH_REPORT.md` or raw grep output.
- Dirty `graphify-out/` files are expected after hooks or incremental
  updates; dirty graph files are not a reason to skip graphify. Only skip
  graphify if the task is about stale/incorrect graph output, or the user
  explicitly says not to use it.
- If `graphify-out/wiki/index.md` exists, use it for broad navigation
  instead of raw source browsing.
- Read `graphify-out/GRAPH_REPORT.md` only for broad architecture review or
  when query/path/explain don't surface enough context.
- After modifying code, run `graphify update .` to keep the graph current
  (AST-only, no API cost).
