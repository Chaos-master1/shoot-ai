# Changelog

All notable changes to Kawkab AI are documented here.

## v0.13.2 (2026-09-18) — Bridge handlers actually work against the real app, GUI e2e, security hardening

### Fixed
- **GPS/ACWR surface was dead in the real app**: PhysicalHandler called the
  async StorageService without awaiting — all five read slots
  (get_gps_sessions/samples, get_player_gps_summary, get_player_acwr) and all
  three import_gps_file writes returned
  `{"error": "Object of type coroutine is not JSON serializable"}` on every
  use. Handler methods are now async, every call awaited, bridge delegation
  sites updated. Unit tests that hid this behind sync stubs were rewritten to
  exercise the real async contract, plus a regression pin with AsyncMock
  storage that fails on the pre-fix shape.
- **xA and pressing reports were silent zeros on real matches**: both handlers
  fed raw storage rows (`event_type` key) into models that filter on `"type"`.
  MatchIntelHandler now normalizes event_type -> type at the storage boundary
  (its single consumer of both models).
- **Every legitimate GPS import was rejected**: import_gps_file validated
  vendor CSV/JSON through validate_video_path (video-only extension
  allowlist). Now uses validate_wearable_path (documents-dir confinement,
  .gpx/.fit/.tcx/.csv/.json allowlist).
- **analyze_setpieces could never parse an event**: the handler omitted the
  required `delivery_height` field when building SetPieceEvent from UI JSON —
  every dispatch failed with a TypeError. Default "medium" now supplied.
- **pressing high-press index crashed on real rows**: same NULL-spatial-value
  trap compute_trap_to_shot_rate had already fixed (storage json_extract emits
  x=None); explicit None-check added.

### Added
- **GUI e2e over the split handlers** (`tests/e2e/test_gui_handler_dispatch.py`):
  real MainWindow boot -> real bridge -> real WAL sqlite -> one dispatch per
  new handler (MatchIntel xA, pressing, Physical GPS import + ACWR round trip
  with FK-correct player creation, Domain setpieces). Seeded and asserted
  through the production StorageService API; all five v0.13.2 fixes above fail
  loudly against these tests.

### Security
- **reel_compose output filename validated** (path traversal, separators,
  extension allowlist) before joining onto the service output directory —
  same injection class previously hardened in stream_start_capture.
- **Vendor-import bridges validate user paths before parsing**: documents-
  directory confinement plus data-extension allowlist on every path handed
  to the Opta/Metrica/SkillCorner parsers. 9 regression tests in
  tests/unit/test_c2_security_fixes.py; 7 of them fail against pre-fix
  behavior (verified by stash).


## v0.13.1 (2026-09-17) — Handler split completion, cloud-server audit, hot-path benchmarks

### Changed
- **bridge_analysis.py split finished** (4981 → ~3910 lines): whiteboard (14
  methods), live-tagging (11), collaboration (12), telestration/stream capture
  (19), and cloud/OAuth/AI-v2/marketplace (29) surfaces extracted into
  `WhiteboardHandler`, `LiveHandler`, and `CloudCollabHandler`; 88 bridge.py
  delegation sites rewired with public slots unchanged. The orphaned
  `_compute_hot_zones` helper moved with its only consumers into
  `bridge_live.py` (caught by ruff F821 before it could NameError at runtime).

### Security
- **Removed `/auth/link-oauth`**: the endpoint trusted a client-asserted
  `provider_user_id`, letting any authenticated user inject an external
  identity into any account (an account-takeover primitive); it had no
  consumer anywhere. Regression test now asserts the route stays gone
  (404, not 401 — a reintroduction fails loudly).
- Cloud-server audit findings: OAuth `state` already one-time-pop and
  provider-matched; every mutating route auth-gated; WebSocket handshake
  checks token + origin + project membership before accept; SQL fully
  parameterized (the single dynamic-identifier UPDATE uses hardcoded column
  names with bound values).

### Fixed
- **Cross-file test pollution** (two independent leaks): e2e pipeline tests
  reinstated each other's CV-service stub at module teardown (later files
  imported a spec-less stub — `AttributeError: no _interpolate_skip_frames`),
  and the visualization-service tests permanently shadowed the real
  `networkx`, which made torch's dynamo importer crash with
  `ValueError: networkx.__spec__ is None` in any later real-inference test.
  Stub installs are now module-scoped fixtures that always pop.

### Added
- `scripts/bench_hot_path.py`: deterministic hot-path micro-benchmarks on
  synthetic 25fps tracking data — overlay computation 21 µs/frame, bridge
  JSON serialization 14 µs/frame, OBV schema conversion 9 µs/frame, live
  hot zones 5.5 µs/frame — plus cProfile top-N for regression tracking.

## v0.13.0 (2026-09-17) — Type-safety bug-mining + silent-failure fixes

### Fixed
- **Postgres mode initialize() regression**: `StorageService.initialize()` in
  PostgreSQL mode claimed to delegate to the adapter but never called
  `_pg.initialize()` — the pool was never created and every storage call
  silently returned empty. Now delegates and is idempotent.
- **Settings contracts panel never rendered**: the bridge slot, handler, and
  JS renderer all shipped, but `index.html` had no `#settings-contract-alerts`
  mount point. Panel now mounts and the whole chain is pinned by smoke tests.
- **DL xG model-comparison branch silently never trained**:
  `model_comparison_service` called the enhanced model's single-event
  `extract_features` where the DL model's list→matrix extractor was required.
- **SDK client paginated-endpoint bugs**: `/matches`, `/events`, `/players`
  return `{"items": [...]}` envelopes server-side; the client read bare-list
  shapes (`matches` key that never exists / returned the envelope as a list).
- **thesportsdb `_get` annotation lie** (`dict | list | None` — the API only
  ever returns JSON objects), clearing ~26 downstream type errors honestly.
- **ShotEvent distance/progression honesty**: fields are `float | None` when
  absent (no fabricated 0.0); `to_dict`/`from_dict` handle None round-trips.
- **Dead SecurityValidator fallback deleted** in six storage modules (the
  ImportError branch was unreachable and one variant never even assigned the
  fallback — a latent NameError).
- ~30 further genuine bugs mined out of the mypy backlog: None-flow guards
  (cloud server rows, storage connections), wrong element types, operator
  misuse, missing awaits/imports, multi-camera frame init regression caught
  by tests before it shipped.

### Changed
- **mypy exits 0 across 315 files** (from 878 errors at baseline): real bugs
  fixed at source, the guarded `self._conn` storage pattern converted to a
  `_require_conn()` helper, and the documented quarantine approach extended
  only for annotation-debt files (mirroring the existing precedent).
- **`AnalysisHandler` split** (5.7k lines): recruitment hub (shortlist,
  contracts, scout search, opponent DB, scouting network, Transfermarkt) and
  the Settings surface moved to `RecruitmentHandler` / `SettingsHandler`;
  bridge slots unchanged.
- **Single JS bridge resolver**: `KawkabUtils.getBridge()` in utils.js replaces
  three byte-identical `resolveBridge()` copies (onboarding, settings,
  whiteboard) and the app-3d inline probes; legacy `kawkabBridge` alias kept.
- **Contract SELECT de-duplicated**: the 13-column player-contracts query now
  lives in one constant consumed by both `StorageService` and
  `ContractTracker`.
- JS bundle rebuilt (34 sources) and frontend manifest unchanged.

### Tests
- Contract storage first-ever coverage (CRUD + expiry windows + alert levels).
- `test_storage_parity` extended: PG-mode initialize delegation + idempotency.
- New `test_desktop_smoke.py` (12 tests): Settings mount points, bundle
  contents, bridge chain per slot, shared getBridge helper, startup storage
  initialize on a real migrated DB (WAL mode), PG-mode adapter delegation.

## v0.12.0 (2026-06-18) — All 25 audit gaps closed + production hardening

### New
- **25 audit gaps closed**: SetPieceService, GoalkeeperService, SubstitutionService, PositioningService, PlayerDevelopmentService, WorkloadService, ScoutingService, VideoReviewService, PeriodizationService, PitchDetector, calibration_v2.js, Arabic glossary, accessibility.css, profiler, strict mypy CI, API docs
- **Profiler wired** into analysis pipeline with frontend UI (stage breakdown, bottlenecks, p50/p95/p99)
- **core/observability.py**: Prometheus-style Counter, Gauge, Histogram with exposition format render
- **metrics_text bridge slot** for on-demand metrics retrieval
- **Plugin system**: KawkabPlugin ABC, PluginManager with entry-point discovery (group: `kawkab.plugins`)
- **TacticalReviewService**: LLM-powered per-section tactical analysis (formation, attack, defense, transitions, set pieces, key players, momentum)
- **.po-based i18n**: locales/en.po + locales/ar.po (77 keys), compile_i18n.py → JSON, JS loads via fetch()
- **PWA support**: manifest.json, service-worker.js (network-first cache)
- **E2E test scaffold**: 18 tests across observability, profiler, bridge slots, .po compilation, CI config
- **Coverage threshold 50%** in CI, badge row in README

### Changes
- PyInstaller spec updated: locales data, new hidden imports
- STATUS.md updated to v0.12.0
- CHANGELOG.md created

## v0.11.0 (2026-06-17) — 6 new services + 355 tests

### New
- SetPieceService (corners, free-kicks, throw-ins, threat, routines)
- GoalkeeperService (saves, xGOT, distribution, sweeps, crosses)
- SubstitutionService (xG-delta impact, rating, verdicts)
- PositioningService (off-ball runs, RunType, xT creation)
- PlayerDevelopmentService (per-player trends, slope, rolling stats)
- WorkloadService (ACWR, monotony, strain, injury risk)
- ScoutingService (opponent profiles, formation prefs, vulnerabilities)
- VideoReviewService (clips, annotations, tags, export/import)
- PeriodizationService (multi-week load, taper, congestion, macrocycle)
- PitchDetector (CV-based line detection via Hough)
- calibration_v2.js (8 drag handles, snap-to-grid, validation badge)
- Arabic glossary: docs/translations/ar.yml (70+ terms)
- accessibility.css + kawkab_polish.js (skip-link, focus-visible, high-contrast, RTL, ARIA)
- Profiler in utils/profiler.py (p50/p95/p99, bottleneck detection)
- API.md + per-service docs
- xG simple model (distance + angle), xT zone threat model
- analysis_service extensions: formations timeline, line-breaking passes, robust possession

## v0.10.2 (2026-06-16) — integration smoke tests

### New
- Integration pro-pipeline smoke tests
- 429 total tests (27× growth from initial 16)

## v0.10.1 (2026-06-15) — i18n + service tests

### New
- Arabic glossary tests
- 294 unit tests

## v0.10.0 (2026-06-14) — real-time streaming + UX polish

### New
- RealtimeService with ShotAlertRule, LowFpsAlertRule, LowConfidenceAlertRule
- ConsoleSubscriber, CallbackSubscriber
- accessibility.css (skip-link, focus-visible, high-contrast, reduced-motion, RTL)
- kawkab_polish.js (ARIA live, keyboard shortcuts, i18n dict, reduced-motion detection)
- Arabic glossary loader

## v0.9.0 (2026-06-10) — 9 native services, 8 external sources, 222 tests

### New
- **8 external data sources**: football-data.org, Bzzoiro, EasySoccerData, API-Football, TheSportsDB, StatsBomb, OpenFootball, RoboFlow Sports
- PsychologyService, FootballRulesService, CardDetectionService, WeatherService
- PoseAnalysisService (YOLO26-pose), MuJoCoBallService, FluidX3DService
- Advanced event detection, physical load, pressure metrics
- Multi-match analysis, data export, visualization, anomaly detection
- Quality scoring, LightGlue homography
- Batch processing, validation framework
- Professional analytics UI (player profiles, match comparison, export)
- PDF report, clip extraction, team swap, visualizations

## v0.5.5 (2026-05-28) — frame skip 3, shot detection, bugfixes

### Fixed
- Enhancement cache crash (wrong path reference)
- frame_skip=3 default (50 → 16.7 fps effective, ~0.75x realtime)
- Shot detection: ball velocity + homography-based, 22 events per 5min

## v0.5.4 (2026-05-26) — pitch-side home/away heuristic

### New
- Median pitch-x per cluster determines home (left) vs away (right)
- Falls back to larger-cluster heuristic

## v0.5.3 (2026-05-24) — hard 36 km/h cap, Kalman smoother

### New
- Hard 36 km/h speed cap (belt-and-suspenders)
- Kalman smoother infrastructure (not wired for fragmented highlights)

## v0.5.2 (2026-05-22) — cluster color logging

### New
- Log home/away BGR cluster colors for manual verification

## v0.5.1 (2026-05-20) — speed sanity caps

### Fixed
- dt correction (0.02s → 0.04s with frame_skip=2)
- 0.4m per-frame delta cap, max speed 36 km/h

## v0.5.0 (2026-05-18) — frame skipping + team colors

### New
- Frame skip parameter (1=full, 2=default, 3+)
- Real team color assignment via k-means clustering
- Formation detection with relaxed lifetime filter

## v0.4.3 (2026-05-15) — LLM guardrails

### New
- build_match_context(), is_clip flag
- System prompt forbids claiming results on short clips

## v0.4.2 (2026-05-12) — knowledge base

### New
- 30 tactical rules, 19 drills
- KnowledgeService with EN/AR text

## v0.4.1 (2026-05-10) — top-N filter

### Fixed
- Top-28 track filter achieves 'excellent' tracking quality

## v0.4.0 (2026-05-08) — homography UI

### New
- Camera calibration UI (4-click corner selection)
- Team color clustering

## v0.3.1 (2026-05-05) — foundation fixes

### Fixed
- Review issues from gap analysis
- STATUS.md, README updates

## v0.3.0 (2026-05-01) — initial desktop app

### New
- PySide6 + QWebEngineView desktop app
- YOLOv11 + BoT-SORT tracking
- Basic event detection (passes, shots)
- LLM coach report via Ollama
- PyInstaller build
