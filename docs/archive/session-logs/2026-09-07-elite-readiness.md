## What was fixed this session (2026-09-07 — elite-readiness pass)

**Environment note:** this session ran on a Linux machine (the repo's
history is Windows-first); a `.venv-linux` (Python 3.12, uv-managed) was
created with the analysis-surface deps (numpy/scipy/loguru/pydantic/pytest/
fastapi/opencv-headless/ultralytics/PySide6). The documented Windows
`.venv` remains untouched. Tests that import the full CV stack need those
packages; pure-core tests (validation package, xG/xT/PSxG models) don't.

**P1 — Models trained on real data (the elite-readiness unlock):**
- `src/kawkab/core/validation/` — new package: StatsBomb loader (corpus →
  Kawkab schema, freeze-frame GK distance, 120×80→105×68m conversion),
  Metrica tracking loader (→ OBV/EPV frame schema), strict-validated metric
  suite (Brier, BSS, reliability, tie-aware AUC, log loss, ECE, bootstrap
  CI — NaN/out-of-range inputs raise, never silently propagate into a
  report), and three trainers (`train_xg`, `train_psxg`, `train_xt`).
- **xG trained**: 5,905 shots from 297 StatsBomb matches → fitted logistic
  weights (`trained_xg_coefficients.json`, auto-loaded, heuristic kept as
  fallback). Held-out 1,449 shots: **Brier 0.0707 vs StatsBomb's own
  0.0667**, AUC 0.772 vs 0.780, ECE 0.0143 (heuristic was ECE 0.068 and
  underestimated mean xG 5×). Provenance API:
  `EnhancedXgModel().coeffs_source`, `trained_model_available()`;
  `compute_xg_trained_from_dict()` is the function new call sites must
  use (legacy `compute_xg_from_dict` kept for compat).
- **PSxG trained + 3-implementation unification started**: fitted from
  2,768 on-target shots (Brier 0.167, AUC 0.753). The empirical zone
  table (top corners 0.34, top-center 0.156) caught the first feature
  set conflating lateral/vertical into one center-distance radius —
  redesigned as lateral×height interaction. `psxg_model.py` keeps its
  legacy signature (goalkeeper_analytics depends on it) but loads
  fitted coefficients; `psxg_improved.py`/`psxg_model_trained.py` remain
  as compat heuristics (full deletion still open — see Known Gaps).
- **xT reference grid**: league-wide 20×32 grid from 534,927 actions
  (`trained_xt_grid.json`), monotonic (own-half 0.025 < final-third
  0.897). `ExpectedThreatModel.build_transition_matrix` now falls back
  to a reference-grid blend when a match has <200 actions (cold-start
  matches previously got an all-zero grid).
- `scripts/validate_models.py` — reproducible CLI emitting
  `docs/validation/validation_report.json` + `REPORT.md` covering all
  three models with bootstrap CIs. `MODEL_CARD.md` updated with the
  measured tables and honest caveats (GK-distance features come from
  SB freeze frames — not yet extracted from live video).

**P2 — Trust debt cleared:**
- `test_bridge_contract.py` grew a second contract layer: every
  `self._X.method(...)` delegation in bridge.py must exist in the
  matching `bridge_handlers/*.py` (the `ai_v2_list_convs` bug class).
  Verified by planting a fake broken delegation — the test caught it.
- Shadowed-duplicate methods merged: `get_injury_risk` (both old versions
  were wrong — one read never-populated workload_d1..28 fields, the other
  FABRICATED ACWR from a `100 + (i%20-10)` sawtooth; canonical version
  uses real GPS `get_player_acwr` data with honest `insufficient_data`
  when absent, response carries BOTH field-name conventions) and
  `generate_training_plan` (kept the event-driven version; deleted the
  mock-diagnosis dead copy). `ruff --select F811` on bridge_handlers/ is
  now clean. `tests/unit/test_phase6_sprint1.py`'s unknown-player test
  now pins the honest "Player not found" contract (it previously
  asserted fabricated scores for nonexistent players).
- `tests/integration/test_bridge_advanced_metrics.py`'s FakeCVService
  signature was stale (missing `match_id`/`storage_service` params the
  real `CVService.process_video` takes) — every analyze_match call
  TypeError'd into error-payload asserts. Stub mirrors the real
  signature now; 4/4 pass. (The CLAUDE.md note blaming the tempdir
  path policy for these failures was wrong — the real cause was the
  stale stub, found by actually reading the error.)
- `pg_schema.sql` duplicate `seasons` table: second (silently no-op)
  definition commented out with an explanatory note — canonical is the
  first, narrower one.
- `pressing_efficiency.py::compute_trap_to_shot_rate` had the same
  None-vs-default `dict.get` bug class as tactical_shape_analyzer (fixed
  there 2026-08-19 but not here): storage's json_extract emits `x: None`
  for events without spatial data → `float(None)` TypeError. Fixed with
  the same explicit None-check. Found via the interop tests below
  running real imported data through it.

**P4b — StatsBomb interop (the elite-club wedge):**
- `services/statsbomb_import_service.py`: imports a StatsBomb events
  file into the Kawkab DB (match + events + players + cached headline
  metrics, xG computed by the trained model at import time). Clubs run
  the full 117-model stack on their existing event data with zero video
  capture. Handles the migration-015 dedup index (same-type/same-second
  events by one player = true duplicates, skipped + counted).
- `POST /api/v1/matches/import/statsbomb` — RBAC-gated (analysis:run),
  accepts file_path (SecurityValidator-validated) or inline events_json.
- `tests/unit/test_statsbomb_import_service.py` (7 tests, all against a
  REAL migration-chained SQLite DB + real corpus files) — includes a
  downstream-models smoke test proving pass-network, pressing, and
  tactical-shape analyzers run on imported data unchanged.

**New-test totals this session:** 41 (validation package) + 7 (import
service) + 3 test files repaired (phase6: 18, bridge_advanced_metrics: 4,
phase5 collection now works with cv2 installed). All green.

**P4a — Pro Analytics suite wired (the 57-orphaned-modules gap, first tranche):**
- `ui/bridge_handlers/bridge_pro_analytics.py` (new handler): one
  `get_pro_analytics_report(match_id)` aggregating 12 elite modules —
  OBV, EPV, pass flow, pressing clusters, duels, ball recovery, box
  entries, switches of play, crossing, set pieces, through balls,
  off-ball — each block with an honest `data_available` flag + reason.
  Delegates to core/ exclusively; storage rows normalized ONCE centrally
  (`_normalize_event`: event_type→type, metadata JSON → top-level
  x/y/is_goal) instead of per-module adaptation.
- Bridge slot `get_pro_analytics_report` + API endpoint
  `GET /api/v1/matches/{id}/analysis/pro` (RBAC analysis:read) +
  frontend section (`app-proanalytics.js`, nav "Pro Analytics", route
  `proanalytics`, manifest entry, bundle rebuilt). Frontend cards render
  per-block availability — never fabricated zeros.
- **3 real core-module crashes fixed via real-imported-data testing**
  (all the same None-vs-default bug class): `epv._to_zone` and
  `epv.compute_possession_epv` (None/ZONE_WIDTH TypeError; end_x-start_x
  on None), `ball_recovery._to_zone` + `classify_recovery`
  (`math.isfinite(None)` TypeError). Each guarded with pitch-center
  fallback.
- `statsbomb_import_service._SB_TYPE_MAP` completed with the full
  open-data type census (Ball Recovery→interception so core/ recovery
  modules count it, Pressure, Block, Dribbled Past, Shield, 50/50→duel,
  Error→miscontrol, Bad Behaviour, Player On/Off, Own Goal variants).
- **`analyze_match` now passes `storage_service` to `process_video`** —
  tracking-frame persistence (the documented non-default path that made
  OBV/EPV/off-ball silently empty) is now the default for every video
  analysis. Event-data imports still have no frames; those blocks report
  data_available:False with the reason, per the honest-empty principle.
- **Frontend test gate repaired** (was: jest-circus missing /
  "Module not found" — the CLAUDE.md-documented flaky Node/jest combo
  finally broken by a stale node_modules): `npm install` in
  `src/kawkab/web` restored 118/118 tests passing. The bundler needed
  `esbuild` resolvable from repo root (`npm install --no-save esbuild`
  at root — web/node_modules has it but scripts/bundle-js.mjs resolves
  from its own location).
- `tests/unit/test_pro_analytics_handler.py` (6 tests): report shape,
  event-blocks-compute on a real import, tracking-blocks honest-when-
  absent, OBV/off-ball compute with synthetic frames, empty-store honest
  failure, and the None-coordinate regression tests for EPV/ball_recovery.

**P4a tranche 2 + season analytics (2026-09-07, second build pass):**
- **Trained-model migration complete**: the last two live call sites on
  the legacy heuristic migrated — `api_v1.get_shots_analysis` (the
  `/analysis/shots` endpoint) and `vaep.py`'s frame-based scoring-prob
  term (legacy heuristic underestimated xG ~5x, deflating every VAEP
  value with it). `model_comparison.py`'s heuristic-vs-enhanced calls are
  deliberate (that module's purpose is comparing them) — untouched.
- **Pro Analytics tranche 2 (13 new blocks)**: carry_xt, xg_chain,
  game_state, flank_analysis, defensive_xt, corner_xg, crossing_xg,
  expected_pass, passing_triangles, scoreline + honest-empty velocity /
  influence_map / lineup_optimizer (need calibrated tracking or season
  history — reasons say so). 17 of 25 blocks compute live on a plain
  StatsBomb import. Frontend ORDER/BLOCK_META extended; bundle rebuilt.
- **3 more real core-module crashes fixed** (the None-coordinate bug
  class, found by feeding real imported data through): `defensive_xt`
  (zone math on None — unlocated defensive events are now SKIPPED, not
  center-fallback: an xT-prevented value needs a real location);
  `passing_triangles` (sorted() over a set containing None track_ids);
  `scoreline_distribution` (sum over xg=None feeding Poisson rate).
  Plus handler-side: scoreline's nested return shape was flattened
  wrongly (reads result["scorelines"] now).
- **PSxG trio unified**: `psxg_improved.py` and `psxg_model_trained.py`
  DELETED (hand-tuned zone heuristics, zero product consumers — only
  their own tests imported them). Their behavioral contracts (header <
  foot, distance decay, bounds) were ported to `test_psxg_model.py`'s
  new `TestTrainedModelBehavior`. Canonical: `psxg_model.py` with fitted
  coefficients. The docstring history note in train_psxg.py updated.
- **Season analytics handler** (`bridge_season_analytics.py` +
  `get_season_pro_report` slot + `GET /api/v1/season/pro` + frontend
  "Season Report" button): formation trends across matches (honestly
  labeled approximation from event x-distributions — the true formation
  detector needs tracking data and there is no formation column in the
  schema), discipline/suspension risk from accumulated card events,
  fixture difficulty with opponent strengths inferred from stored
  results. 4 tests; single-match stores get an honest "needs ≥ 2
  matches" response.
- **Known-gaps count**: orphaned core/ modules down to ~30 (25 wired
  across the two Pro Analytics tranches + season handler; the rest need
  tracking-frame team assignment, season history, or are deliberately
  dormant).

