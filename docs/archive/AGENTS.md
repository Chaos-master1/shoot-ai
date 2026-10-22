## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, invoke the `skill` tool with `skill: "graphify"` before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## Build & Test
- Python tests: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_core_events.py tests/unit/test_xg_model.py tests/unit/test_pitch_control.py tests/unit/test_player_rating.py tests/unit/test_formation_analysis.py tests/unit/test_offball_metrics.py tests/unit/test_substitution_analysis.py tests/unit/test_export_converters.py tests/unit/test_season_aggregator.py tests/unit/test_analysis_service_xg_xt.py tests/unit/test_analysis_service_extensions.py tests/unit/test_xt_model.py tests/unit/test_pass_network.py tests/unit/test_heatmap.py tests/unit/test_tactical_periods.py tests/unit/test_game_state.py tests/unit/test_security.py tests/unit/test_xa_model.py tests/unit/test_benchmarks.py tests/unit/test_match_timeline.py tests/unit/test_migration_manager.py tests/unit/test_logging.py tests/unit/test_win_probability.py tests/unit/test_psxg_model.py tests/unit/test_momentum.py tests/unit/test_defensive_actions.py tests/unit/test_transitions.py tests/unit/test_vaep.py tests/unit/test_set_piece_analysis.py tests/unit/test_storage_service.py tests/unit/test_coding_handler.py tests/unit/test_cv_service.py tests/unit/test_api_external_services.py tests/e2e/test_e2e_pipeline.py tests/e2e/test_e2e_full_pipeline.py -v`
- Frontend tests: `cd src/kawkab/web && npm test`
- Full suite (w/ known pre-existing failures): `$env:PYTHONPATH="src"; python -m pytest tests/unit/ --ignore=tests/unit/test_audio_service.py -q`

## Anchored Summary
- **Professional Readiness Sprint (July 2026)**: Full 9-dimension audit published (rating 3.5/10 as-is)
  - **P0 Schema drift fixed**: `get_match()`/`get_match_events()` use real column names + `json_extract()`
  - **P0 SQL injection closed**: `_sanitize_column_name()` in postgres/event/match storage
  - **P0 Migration 012 fixed**: removed FK index on non-existent `player_name` column
  - **P0 test_encryption.py**: 15 new security tests (all pass)
  - **P1 Migration 025**: 8 composite indexes for common query patterns
  - **P1 SELECT * → explicit columns**: 17 methods across storage_service and subpackage
  - **Ball tracker rewritten**: Dual-mode YOLO11+HSV with 8-state Kalman, bounce detection, FPS-aware DT
  - **Scripts created**: `train_ball_detector.py`, `evaluate_ball_tracker.py`, `download_soccernet.py`
  - **Full PostgreSQL schema** (831 lines, 43 tables, RLS, triggers) + `postgres_storage.py` (122 methods)
  - **Multi-camera fusion**: 6 bugs fixed (self-matching, single-camera merge, ball filtering, occlusion decay, identity homography, camera filter) — 58/58 tests pass
  - **PostgreSQL migration complete**: 43 tables, 46 FK constraints, 124 indexes, 4 triggers on localhost:5432
  - **Ball detector trained**: synthetic YOLO11n fine-tune (30 epochs, mAP50=0.995) — needs SoccerNet real data for broadcast generalization
  - **Bug fixes**: contract_tracker.py + shortlist_service.py syntax errors (missing `def` keywords)
  - **Commit `0b8944e` pushed**: 13 files, +258/-33 lines
- **Goal**: Professional football analytics platform — correct algorithms, polished UX, professional visualizations, comprehensive test coverage
- **904+ analytical tests passing (Phase A/B improvements in progress)** across Sprint 1–15 ✅
- **Completed sprints**: Sprint 1A–1F (critical bugs, PPDA, pass netwok betweenness, xT 16×12, video overlay perf, heatmap scaling), Sprint 2.1–2.3 (xg_model rewrite, win_prob MC, formation k-means)
- **Sprint 3** (i18n consolidation, app.js modularization, StorageService exception handling), **Sprint 4** (VAEP module, set piece analysis, Chart.js integration)
- **Sprint 5** (test coverage — CVService 35 tests, External API 49 tests, StorageService expanded 46 tests), **Sprint 6** (RTL coverage, memory leaks, accessibility, narrative generation, positional benchmarks)
- **Sprint 7** — xT: changed default grid from 16×12→20×32 with calibrated zone values; VAEP 2.0: spatiotemporal features (player-relative distance/velocity to event, teammate density, opponent density); Ball-physics pitch control: 3D trajectory model via RK4 integration in new `ball_physics_pitch_control.py`; Carry xT: new `carry_xt.py` valuing carry progressions through xT grid zones; Off-Ball Value (OBV): new `obv.py` computing off-ball movement value via pass-probability weighting + xT deltas; Lineup optimizer: new `lineup_optimizer.py` with 4-4-2, 4-3-3, 3-5-2 templates using mixed-integer linear programming
- **Sprint 7 (Frontend)** — Split monolithic app.js (3205 lines) into `app-router.js` (hash-based SPA routing), `app-skeletons.js` (loading skeleton states), `app-perf.js` (passive event listeners, `$K` memoized DOM queries, throttle+debounce helpers), `app-ux.js` (keyboard nav, empty-state messages, confirmation dialogs, multi-language fallback i18n); fixed tail-call recursion in `tactical_sandbox.js` via `stopRenderLoop` + `document.contains` guard
- **Sprint 8 (Security)** — SQL column-name validation in `storage_service.py` (`_sanitize_column_name` regex); Bridge arg sanitization in app.js (`sanitizeString`, `validateInt`, `sanitizeBridgeArg`); untracked sqlite connections (null-connection safe-return on all 40+ methods)
- **Sprint 8 (Performance)** — Collapsed 16-pass IO pattern in `reasoning_service.py` (`_precompute_event_stats` single-pass dict); numpy vectorization in `xg_model.py` (np.fromiter) and `formation_analysis.py` (vectorized compactness); `lru_cache` on `compute_xg`, `_zone_value`, `_zone_from_position`
- **Sprint 8 (Reasoning fix)** — Fixed `DiagnoseReport` unpacking bug in `_assemble_report`: changed `key: diag + tuple` unpacking to explicit `key: DiagnosisReport(...)` construction; eliminated all `TypeError: cannot unpack non-iterable` exceptions — 29/29 reasoning tests pass
- **Sprint 8 (Storage hardening)** — Added 4 missing bulk methods (`save_events_bulk`, `get_match_players`, `save_players_bulk`, `save_advanced_metrics_bulk`); replaced 20+ `raise RuntimeError` with safe-return for uninitialized connection; added explicit required-field validation (`if track_id is None: return 0`) preserving original test expectations; all 46 storage tests pass
- **Sprint 9 (Features)** — Player similarity engine ✅ (existing), Pressure/PPDA ✅ (existing), `physical_metrics.py` (sprint/threshold/zone detection), `dl_xg_model.py` (PyTorch deep-learning xG with attention pooling), Set piece analysis ✅ (existing)
- **Sprint 9 (UX)** — Multi-language i18n fallback (`loadMissingKeys` in ui.js), keyboard navigation (`initKeyboardNav` in app-ux.js), empty-state messages, confirmation dialogs
- **Sprint 10 (Analytical v2)** — EPV model `test_epv.py` (19 tests), Transitions `test_transitions.py` (9 tests), Security `test_security.py` (14 tests), xT model `test_xt_model.py` (extended 157 analytical tests)
- **Key patterns**: Bridge slot pattern for frontend↔backend; showToast replaces alert(); showSkeleton/hideSkeleton for loading states; collapsible via `.pro-card.collapsible.collapsed` toggle; i18n via `data-i18n` + `data-i18n-placeholder` attributes; Chart.js wrapper in `window.KawkabCharts` with original renderer fallback; conftest stubs for loguru/httpx/paths/migration_manager + `load_service_module` with `_ensure_package_loaded`
- **Sprint 11 (Comprehensive Audit & Plan Execution)** — 24/24 items completed:
  - **P0 Wrong-Results Bugs**: Fixed VAEP zero-value bug (`post_score_prob` recalculated from next event); Fixed DL xG random weights (module-level singleton + heuristic fallback); Dead bridge_handlers/ deleted (2446 lines); AnalysisService deduplicated
  - **P1 UX/Performance/Security**: showToast replaces all 35 `alert()` + 3 `confirm()` in app.js; Skeleton system activated on 7 async ops; aria-live regions on 4 dynamic sections; Focus trap fixed; 4 missing `.catch()` handlers added; `cursor.executemany()` in 3 bulk DB methods; `@lru_cache` on `_simulate_remaining` (win_probability) and k-means results (formation_analysis, 4→2 calls); Pitch control vectorized ~291× faster (numpy broadcasting); XSS fixed in analytics.js + app-data-providers.js; Path traversal fixed in data_export_service.py
  - **P2 Test/Model Improvements**: 107 new tests for 5 previously untested modules (ball_physics_pitch_control, carry_xt, dl_xg_model, lineup_optimizer, obv); 51 new VAEP correctness + property-based tests (VAEP/EPV/pitch_control bounds); Eigenvector centrality via power iteration in pass_network.py; Cross-subtype granularity (early/cutback/driven/lofted) in xa_model.py; Pitch constants unified from game_constants.py across 14 files; Magic numbers → 6 named constants in xg_model.py
  - **P3 Advanced Features**: Scout report generation from player similarity (17 tests); Trap→transition linkage with temporal/spatial thresholds (14 tests); Credible intervals for xG (Beta conjugate) / xT (bootstrap) / VAEP (block bootstrap) (11 tests)
  - **Result**: 322 analytical tests pass in 4.9s (up from 250), all 24 plan items delivered
- **Sprint 12 (Professional Level-Up — 19 items across 4 phases)**:
  - **A — UX (5/5)**: Dashboard section built with KPI cards, recent matches, quick actions, season overview; Light theme CSS implemented (`[data-theme="light"]` + `prefers-color-scheme`); Player-vs-player comparison with dual radar charts + stat table + delta/insight cards; Global search over matches/players/events with keyboard shortcut (`/`), debounced filtering, and result dropdown; i18n expanded from ~30 to 200+ elements with `data-i18n` on all visible text, locale-aware number formatting, and Arabic completion
  - **B — Analytics (5/5)**: Finishing analysis (`finishing_analysis.py`) with shot quality tiers (big/half/low chance), hot/cold streak detection, placement skill metric — 27 tests; Goals Added (g+) framework (`goals_added.py`) combining xG/xA/xT/defensive/OBV into single per-90 value with position percentiles — 18 tests; Fixture difficulty (`fixture_difficulty.py`) with opponent strength, home/away weighting, schedule density — 19 tests; League simulation (`league_simulation.py`) via 10,000-run Monte Carlo with Poisson(xG) goal generation, title/top4/relegation probabilities — 20 tests; Squad valuation (`squad_valuation.py`) with age curve, position baselines, contract/league multipliers, squad-level aggregation — 38 tests
  - **C — Architecture (4/4)**: Bridge broken up from 3352→755 lines (77% reduction) into 6 handler modules (analysis, export, video, storage, external, lifecycle) with constructor injection; Token bucket rate limiter in security.py (analysis=5/min, export=10/min, search=30/min); Audit service with `audit_events` table, structured JSON details, filtered queries, 14 tests; Migrations 012 (6 missing indexes) and 013 (audit_events table + 3 indexes); 42 tests for all 7 previously-untested external API services
  - **D — Polish (4/4)**: Chart annotations (goal markers, xG=±0.5 reference lines) via inline Chart.js plugin; PNG/CSV export per chart card; Notification center with bell icon, dropdown history, localStorage persistence, toast integration; Tooltip system (`data-tooltip` attribute, rich HTML, 300ms delay, viewport-aware positioning) on all KPI cards, stats, and buttons
  - **Result**: 206 new tests across all phases (total analytical: 528), bridge.py 77% smaller, 7 new analytical modules, all external API services tested, dashboard + player compare shipped
- **Sprint 13 (Professional Data Trust & Presentation — 15/15 items)**:
  - **P0 Data Trust (5/5)**: `CoordinateValidator` validates/clamps event x/y on all spatial fields with warnings; StatsBomb export fixed (period detection from timestamps, real shot outcomes, correct xG, pass length/angle from coords); LightGlue `error_px` propagated from computed reprojection error instead of hardcoded 0.0; UNIQUE constraint migration 015 on `events(match_id, timestamp, type, track_id)` prevents duplicate event storage
  - **P1 Professional Workflows (6/6)**: Sortable/filterable data tables (timeline events + player roster with click-to-sort, column filters, pagination 25/50/100); Chart click cross-filter (xG timeline/momentum/winprob click → filter event timeline to time range); Multi-selection + batch operations (checkbox, Shift+click range, batch delete/export); Timeline scrub/zoom (canvas density histogram with draggable time range handles); Sparklines (zero-dep SVG utility on KPI cards + comparison bars); Local file import (CSV, generic JSON, StatsBomb JSON with auto-detect, coordinate validation, 14 tests); Data density toggle (compact/normal/spacious CSS variable mode)
  - **P2 Polish & Power (4/4)**: Color customization (team home/away color pickers, CSS variable-driven chart colors, localStorage persistence); Video keyboard shortcuts (Space=play/pause, J/L=±10s, ←/→=±5s, F=fullscreen); Persistent filter state (sessionStorage for timeline/table/search/navigation filters); Game plan scouting report (opponent profile, formation tendencies, key players to neutralize, set piece plan, scoreline prediction, 16 tests)
  - **Result**: 93 new tests (total analytical: 621), all 15 sprint items delivered
- **Sprint 14 (Tactical Phase Analytics + Recruitment + Discipline — 13/13 items)**:
  - **P0 Tactical Phases (3/3)**: Phase xG breakdown (`phase_xg.py`) classifying shots as settled/transition/counter/set-piece/direct — 25 tests; Build-up analysis (`build_up.py`) with goal kick patterns, line-breaking passes, build-out under pressure — 32 tests; Territory compounding (`territory_value.py`) accumulating xT per possession chain with zone-level net advantage — 16 tests
  - **P1 Recruitment Pipeline (4/4)**: Player shortlist service with migration 016 (status/priority pipeline, scout rating, 12 tests); Player search (`player_search.py`) with age/position/league/stat-threshold multi-criteria filtering, match scoring — 19 tests; Transfer fee estimation (`estimate_player_transfer_fee()`) in squad_valuation.py with age curve, contract/market/international/injury factors, fee range — 21 tests; Contract tracker service with migration 017 (contracts table, expiry alerts, squad summary, 9 tests)
  - **P1 Post-Match (2/2)**: Day After Match report (`match_report.py`) with auto-generated executive summary, key moments, tactical observations, areas for improvement — 17 tests; Report template system (`ReportTemplate`) with configurable sections and detail levels
  - **P2 Discipline & Form (3/3)**: Referee analysis (`referee_analysis.py`) with card rate profiling, home/away bias, inconsistency scoring — 15 tests; Suspension tracker (`suspension_tracker.py`) with configurable yellow thresholds, upcoming risk detection, fair play score — 12 tests; Form by competition type + opponent strength tier extended into `form_analysis.py` — 14 tests
  - **P2 Model Quality (1/1)**: xG model comparison (`model_comparison.py`) evaluating heuristic/logistic/DL models with log-loss, Brier, AUC-ROC, calibration error, distance/angle bucketing, feature importance — 15 tests
  - **Result**: 249 new tests (total analytical: 870), all 13 sprint items delivered
- **Sprint 15 (Video Tagging Engine — Phase 1.1–1.6, 7/7 items)**:
  - **Backend**: Migration 018 (`coding_tags` table with indexes on match_id/type/player/time); StorageService CRUD (save/get/update/delete/by_type/by_player/stats — 21 new storage tests); CodingHandler bridge module (11 methods: save_tag, get_tags, update_tag, delete_tag, get_tag_stats, get_tags_by_type, get_tags_by_player, get_match_players_simple, extract_tag_clip, extract_tag_clips_batch, get_default_tag_templates — 13 handler tests); Registered in bridge.py with 11 new `@Slot` methods
  - **Frontend**: New "🎬 Coding" nav tab + `#coding-section` with 3-panel layout (left: matrix buttons + player/team/period/lead-lag/notes controls; center: video player + interactive coding timeline canvas + quick stats bar; right: tag list with seek/clip/delete actions + CSV/JSON export); Matrix tagging buttons (4 categories × 22 buttons: Attack 8, Defense 6, Mistake 6, Set Piece 5) with color-coded visual feedback + keyboard shortcuts (1-9, q,w,e,r,t,z,x,c,v,b,n,m, comma, period, slash, p); Coding timeline canvas renders tag markers as colored dots with current-time cursor; Tag filtering by type/player/notes; Auto-clip extraction via existing `ClipExtractionService`; Auto-seek to tag on click
  - **Quality**: Dark theme + CSS variables + RTL support; Safe error handling throughout; HTML/CSS/JS all follow existing codebase patterns; 34 new tests (13 handler + 21 storage); All 34 pass
  - **Result**: 34 new tests (total analytical: 904), all 7 sub-items delivered
- **Phase 2 (Auto Event Detection + Correction — 4/4 items)**:
  - **2.1 Smart auto-detection**: Already existed in `AdvancedEventDetectionService` (16 methods)
  - **2.2 Event correction UI**: Review section with event queue, confidence badges, confirm/reject/edit actions, video seek-to-event, detection summary dashboard, auto-advance mode; Bridge methods (`get_unreviewed_events`, `get_detection_summary`, `submit_event_correction`); 16 new tests; Fixed `user_corrected` in `update_event` allowed fields
  - **2.3 Auto-phase detection**: Bridge method `get_tactical_periods(match_id)` derives phases (settled/transition/set_piece/direct) from events with percentage breakdown; Frontend "🧠 Tactics" tab with phase visualization bars + formation cards
  - **2.4 Auto-formation detection**: Bridge method `analyze_formation(match_id)` uses `FormationAnalyzer` on event data to detect home/away formations + width/depth/compactness; Frontend renders formation cards per side
- **Phase 3 (AI Coach Intelligence — 1/1 item)**:
  - NL query bridge method `ask_llm(match_id, question)` that builds match context from events + calls `LLMService.generate()` with tactical analyst system prompt; Frontend "🤖 AI" tab with chat-style interface (message history, typing indicator, LLM status indicator)
- **Phase 4 (Professional Workflow — 3/3 items)**:
  - Player Rating (0-100): Bridge method `get_player_rating(match_id, track_id)` computes composite from pass accuracy, shot impact, tackles, carries, dribbles, goals, event volume
  - Squad roster: Bridge method `get_squad_summary(match_id)` returns per-team player list with pass/shot/tackle counts; Frontend "👥 Squad" tab with roster table + per-player rating badges (color-coded high/mid/low)
  - **Result**: 16 new tests (total analytical: 920+), all Phase 2-4 items delivered across bridge_analysis.py (+7 methods), bridge.py (+6 @Slots), index.html (3 new sections), main.css (~100 lines), app.js (~350 lines)

## Broadcast Tracking Pipeline (Experimental — Phases A-I)

| Phase | Status | Items | Key Deliverables |
|-------|--------|-------|-----------------|
| **A** — Critical Bugs | ✅ Complete | 10/10 | Fixed confidence=0 bug, inverted runtime formula, wrong FPS calc, Google Drive URL, checkpoint version lock, detect_frame try/except, boxmot reset guard, auto-calibration KeyError, goal direction heuristic, frame-rate-dependency |
| **B** — Detection | ✅ Code | 6/8 | BallTracker (HSV+Kalman), adaptive HSV pitch mask, confidence calibration by bbox size, YOLO fine-tune script, `fine_tune_yolo.py`, SoccerNet annotation converter |
| **C** — Tracking | ✅ Code | 3/10 | `TrackSmoother` (RTS Kalman), cross-segment propagation guard, config-driven stitch thresholds |
| **D** — Events | ✅ Code | 2/8 | `ball_tracker.py` service, possession assignment placeholder in `detect_events.py`, segment-based ball detector |
| **E** — Validation | ✅ Code | 4/8 | `mot_metrics.py` (MOTA/MOTP/IDF1), `regression_test.py`, `render_tracking_overlay.py`, ground-truth comparison in `evaluate_tracking.py` |
| **F** — Architecture | ✅ Code | 4/8 | `config.py` (YAML/JSON tracking config), `tracker_base.py` (ABC + registry), ball tracker module, streaming mode (keep last 500 frames) |
| **G** — Performance | ✅ Code | 3/8 | Batch ReID (every 30 frames), lazy FaceRecognitionService, streaming frame limit |
| **H** — Advanced | ✅ New | 3/10 | `physical_metrics.py`, `heatmap_generator.py`, render overlay script |
| **I** — Deploy | ✅ New | 3/8 | `setup.py` (one-command install), `__main__.py` (CLI entry: `python -m kawkab track`), Docker-compatible project structure |

### New files added (21 total)
- `src/kawkab/__main__.py` — CLI entry point (track, evaluate, render, events commands)
- `src/kawkab/core/config.py` — Centralized YAML/JSON configuration (70+ parameters)
- `src/kawkab/core/mot_metrics.py` — CLEAR MOT metrics (MOTA, MOTP, IDF1)
- `src/kawkab/services/ball_tracker.py` — HSV+Kalman dedicated ball tracker
- `src/kawkab/services/tracker_base.py` — Abstract tracker interface + registry
- `src/kawkab/services/track_smoother.py` — RTS Kalman smoother for track positions
- `src/kawkab/services/physical_metrics.py` — Player distance, speed, sprint metrics
- `src/kawkab/services/heatmap_generator.py` — 2D Gaussian KDE heat maps
- `scripts/setup.py` — One-command environment setup
- `scripts/regression_test.py` — Automated regression test suite
- `scripts/render_tracking_overlay.py` — Visual debugging overlay video
- `scripts/fine_tune_yolo.py` — YOLO training pipeline + SoccerNet annotation converter

### Modified files (4)
- `cv_service.py` — 8 fixes: confidence tracking, adaptive pitch mask, adaptive confidence threshold, ball_tracker integration, batch ReID, lazy FaceRecognition, streaming mode, camera cut error handling, auto-calibration KeyError guard
- `model_manager.py` — Fixed osnet_x1_0 URL (Google Drive → boxmot release)
- `run_full_match.py` — Fixed inverted runtime formula
- `evaluate_tracking.py` — Fixed FPS calculation
- `detect_events.py` — Fixed goal direction heuristic, time-based segment gaps, lower ball confidence threshold

### New files added (3)
- `scripts/synthetic_benchmark.py` — ByteTrack association robustness test: degrades Metrica GT with configurable noise/drop/FP, runs ByteTrack, reports MOTA/IDF1
- `scripts/real_video_eval.py` — End-to-end evaluation on real video: runs YOLO at pipeline conf vs GT conf, tracks via ByteTrack, reports MOTA/Precision/Recall vs pseudo-GT
- `scripts/manual_annotate.py` — Lightweight OpenCV manual annotation tool for independent ground truth

### Modified files (2)
- `ball_tracker.py` — Fixed Kalman `.ravel()` bug: `self.kalman.predict()` returns 2D column vector, `float(pred[0])` crashed with `only 0-dimensional arrays can be converted to Python scalars`; added `.ravel()` before indexing
- `__main__.py` — Fixed `Path(video_path)` wrapper: `process_video()` expects `Path`, was receiving string

## Tracking Accuracy Results

### A. ByteTrack association robustness (synthetic — Metrica GT with injected noise)
Tests the ByteTrack association algorithm in isolation (no YOLO, no images, no homography). Underlying positions are real match data from Metrica; noise parameters (position std, dropout rate, FP rate) are assumed, not measured from the actual YOLO pipeline. 22 players, 2000 frames.

| Label | Noise | Drop | FP | MOTA | IDF1 | IDSW |
|-------|-------|------|----|------|------|------|
| near-perfect | 0.001 | 0% | 0% | 0.9989 | 1.0000 | 50 |
| **mild (assumed noise, not measured)** | **0.005** | **10%** | **2%** | **0.8683** | **0.9406** | **864** |
| moderate-noise | 0.010 | 10% | 2% | 0.4792 | 0.8312 | 10200 |
| high-noise | 0.020 | 10% | 2% | 0.1751 | 0.5904 | 10597 |
| high-dropout | 0.005 | 20% | 2% | 0.7679 | 0.8796 | 758 |
| severe-dropout | 0.005 | 30% | 2% | 0.6691 | 0.8133 | 716 |
| combined-moderate | 0.010 | 20% | 5% | 0.4262 | 0.7536 | 7844 |
| combined-severe | 0.020 | 30% | 5% | 0.1273 | 0.4315 | 6423 |

**Key findings:**
- ByteTrack robust to 0.5% position noise + 10% dropout: MOTA=0.87, IDF1=0.94
- Breaking point at 1% position noise — IoU association fails catastrophically
- Dropout handled well (max_age=30 keeps tracks alive for occlusions)
- FP rejection excellent — ByteTrack filters most false positives via confidence/matching thresholds
- ID switches are main weakness: 864 switches at mild settings (reacquisition after dropout)

### B. End-to-end pipeline on broadcast video (pseudo-GT)
Tests the full pipeline (YOLO11m + ByteTrack + filtering + team assignment) on 15 min of broadcast footage (france_sweden_15min.mp4, 3600 detection frames). Pseudo-ground-truth = YOLO detections at conf>0.5; pipeline threshold = conf>0.4. This is NOT independent ground truth (GT and predictions come from the same model at different thresholds), so numbers should be read as "how well does tracking preserve the highest-confidence YOLO detections" rather than absolute accuracy.

| Metric | Raw detection (conf=0.4 vs conf=0.5) | Tracked (ByteTrack on conf=0.4 vs conf=0.5) |
|--------|-------|---------|
| MOTA | 0.844 | **0.538** |
| Precision | 0.865 | **0.925** |
| Recall | 1.000 | **0.586** |
| F1 | 0.928 | **0.717** |

**Key findings:**
- **Precision = 92.5%** — When the pipeline outputs a detection, it almost certainly matches a real high-confidence player detection
- **Recall = 58.6%** — But the pipeline misses ~41% of high-confidence detections, mainly from camera-cut fragmentation (frequent in broadcast footage) and ByteTrack's min_hits filter
- **MOTA = 0.538** — Positive but limited by the recall gap
- Primary recall bottleneck is broadcast camerawork (frequent cuts, close-ups, panning) not detector quality
- An independent ground truth (manual annotation or SoccerNet) is needed to validate whether the recall gap is from missed players or from the pseudo-GT including non-player detections

### C. Self-consistency (Metrica vs Metrica)
- MOTA=1.0, IDF1=1.0 — validates MOT metric computation code

### To run next
```powershell
# Full match with all fixes
$env:PYTHONPATH="src"; python -m kawkab track --video "France vs Sweden_match.mp4" --output tracking_output_full --skip 6

# ByteTrack association robustness test (synthetic)
$env:PYTHONPATH="src"; python scripts/synthetic_benchmark.py --max-frames 5000 --noise 0.005 --drop 0.10 --fp-rate 0.02

# End-to-end pipeline on real video (pseudo-GT)
$env:PYTHONPATH="src"; python scripts/real_video_eval.py --video france_sweden_15min.mp4 --gt-conf 0.5 --pipeline-conf 0.4

# Self-evaluation
$env:PYTHONPATH="src"; python scripts/evaluate_tracking.py --self tracking_output/track_summary.json

# CI benchmark (Metrica self-consistency)
$env:PYTHONPATH="src"; python scripts/benchmark_tracking.py

# Manual annotation (independent GT)
$env:PYTHONPATH="src"; python scripts/manual_annotate.py --video France_Sweden_clip_2min.mp4

# Ball-only tracking (independent module)
$env:PYTHONPATH="src"; python -c "from kawkab.services.ball_tracker import BallTracker; print('OK')"

# Physical metrics from tracking output
$env:PYTHONPATH="src"; python -c "from kawkab.services.physical_metrics import compute_physical_metrics; print('OK')"

# Regression test
$env:PYTHONPATH="src"; python scripts/regression_test.py --video france_sweden_15min.mp4
```
## Professional Audit Sprint (June 2026 — All P0-P2 Actionable Items Delivered)

Following an independent professional readiness audit (rated 2.5/10), all actionable items across P0-P2 were completed in a single session:

### P0 — Structural (5/5)
- **Persist per-frame tracking data**: `tracking_frames` table (migration 022), `save_tracking_frame/bulk/get/delete` in `storage_service.py`, wired into `CVService.process_video()` as optional `storage_service` param
- **StatsBomb regression runnable**: auto-fetch conftest downloads 5 match files on `pytest_configure`, `skipif` guard for network-failure graceful degradation
- **Team normalization**: `ensure_team()` get-or-create, `save_match()` auto-links via `home_team_id`/`away_team_id` FK, migration 022 FK columns + indexes
- **Medical encryption**: Fernet/PBKDF2 module (`encryption.py`), key storage in `encryption_keys` table, encrypt/decrypt wired into `concussion_protocol.py` (notes), `injury_tracker.py` (notes), `rehab_service.py` (milestones/notes)
- **LightGlue default**: `save_homography()` bridge handler tries `lightglue` auto-calibration before falling back to manual 4-click corners

### P1 — Semi-pro (3/3 actionable)
- **Tracking recall**: `max_age=30→90`, `min_hits=3→1`, removed full tracker reset on camera cuts, lowered broadcast filter to `min_segments=1`/`min_pct=0.08`
- **xG training pipeline**: `scripts/train_xg_from_statsbomb.py` trains logistic regression on real StatsBomb data → `trained_xg_coefficients.json` auto-loaded by `xg_model.py` with `ENHANCED_COEFFICIENTS` fallback
- **SQLite WAL mode**: `PRAGMA journal_mode=WAL` + `PRAGMA synchronous=NORMAL` in `StorageService.initialize()`

### P2 — Professional (1/3 actionable)
- **Injury-risk model**: Replaced 8-entry heuristic with 26 evidence-based diagnosis-specific recovery ranges (Ekstrand 2011, NCAA ISP, BMJ Open SEM). Added `recovery_range_days()`, `days_until_expected_recovery()`, `injury_risk_score()` exponential curve, recurrence detection in `get_injury_stats()`, high-risk flagging in `get_squad_injury_report()`

### Other fixes from audit
- Hungarian matcher non-square crash fix, checkpoint HMAC env var, CORS env var, orphaned pipeline deletion (3 files), README numbers/version unification, tracking accuracy publication

### Still blocked (need human/external input)
- P1.6: Amateur footage evaluation (no video available)
- P2.12: Multi-camera calibration UI (architecture ready — LightGlue + segment_homography exist)
- P1.9: Event QA forcing function (exists at DB level)
- P2.11/13/15, all P3: field testing, ToS review, deployment strategy

## Professional Readiness Sprint (July 2026 — 26 items across 5 tiers)

Full 9-dimension audit completed — rating **3.5/10** → **~5/10** after Tiers 1-5 shipped.

### Tier 1 — Critical Correctness (5/5)
- **League sim median bug**: `_median_pos` was using `max()` across all sims → fixed to per-simulation tracking with correct median
- **FK enforcement**: `PRAGMA foreign_keys=ON` added to `StorageService.initialize()`
- **Kalman DT scaling**: Removed fixed `KALMAN_DT=1/24`, process noise now scales with actual FPS
- **Dead VAEP v2**: Removed `compute_vaep_v2()` (was identical to v1, never imported)
- **Dead carry_frames list**: Removed from `carry_xt_from_tracking` (appended but never read)

### Tier 2 — Data Trust (7/7)
- **attacking_direction**: Added param to all 6 spatial models (xT, xA, carry_xT, VAEP, defensive_actions, lineup_optimizer)
- **StatsBomb expanded**: 5 new match IDs (3753, 69301, 7189, 20388, 20464), exponential backoff on 429, coefficient validation
- **Bulk insert perf**: `save_tracking_frames_bulk` switched from per-row `execute` to `executemany`
- **Migration 023**: Missing indexes on `player_profiles(team)`, `player_profiles(global_id)`
- **SELECT * → explicit columns**: 10 read methods across storage service
- **LightGlue in CV pipeline**: Tried first per segment before persisted calibrations → PitchDetector → fallback
- **xG regression tests enabled**: `@_need_shots` skipif removed, tests always run

### Tier 3 — Pro Features (5/5)
- **MILP lineup optimizer**: ortools CP-SAT binary variable model with greedy fallback, 7 formations
- **Frame-based VAEP v2**: Uses `WeightedPitchControl` on tracking frames for event valuation
- **RK4 ball trajectory**: Real drag (Cd=0.25) + Magnus force, replacing simplified kinematic model
- **Pagination**: `limit`/`offset` on `get_match_events`, `get_match_players`, `get_player_profiles`, `get_reports`, `get_recent_benchmarks`, `get_validation_results`
- **Confidence intervals**: xG via Beta conjugate (`compute_xg_with_ci`), xT via bootstrap (`compute_action_xt_with_ci`), VAEP via block bootstrap (`compute_vaep_with_ci`)

### Tier 4 — UX Polish (5/5)
- **JS bundling**: esbuild concatenates + minifies 27 IIFE files → `dist/app.bundle.min.js` (397 KB, 46% reduction); updated `index.html` to 4 special + 1 bundle script tag
- **Undo/redo**: `app-undo.js` — `UndoManager` with 50-deep stacks, Ctrl+Z / Ctrl+Shift+Z / Ctrl+Y bindings
- **Keyboard HUD**: `app-hud.js` — `?` key toggles overlay listing video shortcuts (Space/J/L/arrows/F), undo/redo, search; `window.KawkabShortcuts`
- **Nav consolidation**: 25 flat tabs → 6 nav groups (Dashboard/Analysis/Tactics/Coding/Squad/Admin) with group titles
- **PDF export**: `window.print()` button + `@media print` styles stripping nav/chrome, force sections visible

### Tier 5 — Infrastructure (4/4)
- **OS keychain**: `encryption.py` uses `keyring` for medical key storage (service: `kawkab-medical`), falls back to `~/.kawkab/.medical_key` file; dependency `keyring>=24.0` added
- **Backup/restore**: `StorageService.backup()` (timestamped via `PRAGMA wal_checkpoint` + `conn.backup()`), `restore()` (validate + `shutil.copy2` + re-init + migrations), `auto_backup()`
- **Migration 024**: Soft-delete columns (`is_deleted`, `deleted_at`, `deleted_by`) + indexes on `matches`, `events`, `players`, `coding_tags`; SELECT filters `WHERE (is_deleted IS NULL OR is_deleted=0)`; DELETE → `UPDATE SET is_deleted=1`; new `hard_delete_*`/`restore_*` methods
- **PostgreSQL schema alignment** (partial): pagination added to match storage_service; 15 missing tables still pending

### Result
- **38 files modified**, 2,438 insertions, 1,141 deletions
- **229+ unit tests passing** (all pre-existing failures unchanged)
- **Rating: ~5/10** after all 26 items (up from 3.5/10)
- **Next targets**: 7/10 with independent tracking GT validation + amateur footage eval
