# Kawkab AI — Iteration Log

> **Mission:** Operate Kawkab AI as a professional sports analytics product, not a prototype.
> **Quality bar:** Comparable to tools used by professional football teams (StatsBomb, Second Spectrum, Hudl, Wyscout, SciSports).
> **Cadence:** Continuous cycles. Each cycle = one weakness → one improvement → tested → committed → next.

---

## The Iteration Cycle

Every cycle follows the same loop. Don't skip steps.

```
1. DIAGNOSE  — Pick one weakness from the backlog (or find a new one)
2. PLAN      — Define success criteria, files to touch, risks, acceptance test
3. IMPLEMENT — Make the change
4. TEST      — Add or run tests; verify the success criteria
5. DOCUMENT  — Update README / docs / docstrings
6. COMMIT    — Atomic commit with clear message
7. UPDATE    — Add a row to the log below, mark complete
8. NEXT      — Pick the next target
```

A cycle is small enough to complete in 1–4 hours of focused work, but big enough to be a real improvement. Not a 30-second drive-by.

---

## Backlog (ordered by impact × risk)

> Stars = priority. ★ = high, ☆ = medium, ◇ = low.
> Source: `REVIEW.md` (my critique) + `PROFESSIONAL_AUDIT.md` (the team's own gap analysis) + `kawkab-ai-review.md` + `STATUS.md`.

### Reliability & correctness (foundation)

| # | Target | Source | Why it matters |
|---|---|---|---|
| 1 | **Make the video pipeline work end-to-end on the user's real match video** | user | The product's reason to exist |
| 2 | **Fix `services/__init__.py` latent ImportError** | review | Crashes on import of `BenchmarkService` etc. |
| 3 | **Reconcile version strings across pyproject / README / PLAN / STATUS** | review | New contributors can't tell which doc is true |
| 4 | **Top-N filter to exactly 22 tracks per match** | review + audit | Foundation for accurate possession / distance / xT |
| 5 | **Auto-homography from pitch keypoints (zero-click calibration)** | review + audit | Amateur coaches won't click 4 corners |
| 6 | **End-to-end integration test on a fixture video** | review | The v1 acceptance test for the product |
| 7 | **Anomaly detection for impossible stats (400 km/h speeds, etc.)** | audit | Trust layer |

### Code health (sustainability)

| # | Target | Source | Why it matters |
|---|---|---|---|
| 8 | **Consolidate `cv_service_v2.py` → delete the v1 loser** | review | Two competing implementations = rot |
| 9 | **Consolidate `clip_extraction_service.py` → pick one** | review | Same smell |
| 10 | **Slim `analysis_service.py` (45KB → <20KB)** | review | Single file doing too much |
| 11 | **Slim `storage_service.py` (26KB → <15KB)** | review | Extract migrations into `kawkab/migrations/` |
| 12 | **Add CI (GitHub Actions: ruff, mypy, pytest)** | review | Dev tooling is decorative without it |
| 13 | **Add tests for the top 12 services (target 1.0+ ratio)** | review | 0.33 ratio is a real risk |
| 14 | **Move 363 MB test video out of git history** | review | Permanent disk cost for every clone |
| 15 | **Tighten `.gitignore` (catch `*.mp4` at repo root, `*.bak`)** | review | Hygiene |
| 16 | **Add a docs index (which doc is authoritative for what)** | review | Six overlapping docs confuse new readers |

### Trust & transparency (the "professional" difference)

| # | Target | Source | Why it matters |
|---|---|---|---|
| 17 | **Model card (`docs/MODEL_CARD.md`)** | audit + my critique | YOLO weights, training data, failure modes |
| 18 | **Data card (`docs/DATA_CARD.md`)** | audit + my critique | Knowledge base: who wrote it, what authority |
| 19 | **Ground-truth eval set (≥3 amateur matches with hand-tagged events)** | audit | Cannot claim "accurate" without this |
| 20 | **LLM groundedness check before report generation** | my critique | Reports sound authoritative; may be wrong |
| 21 | **Document known failure modes for each tracker (BoT-SORT, Norfair, Roboflow)** | audit | When to switch trackers based on input |

### Data layer (what pro tools have that Kawkab doesn't)

| # | Target | Source | Why it matters |
|---|---|---|---|
| 22 | **Multi-match player tracking with persistent IDs** | audit | Coach sees player X across 10 matches |
| 23 | **Season / team / competition schema in storage** | audit | Match needs context (league, opponent, date) |
| 24 | **Player profile system (photo, position, attributes, history)** | audit | "Tell me about Mohamed across the season" |
| 25 | **Squad / roster / substitutions model** | audit | Real matches have subs |
| 26 | **Analysis versioning — track manual corrections** | audit | Coach edits an event, version bumps |

### Analytics depth (what pro tools compute that Kawkab doesn't)

| # | Target | Source | Why it matters |
|---|---|---|---|
| 27 | **Heatmaps (player, team, action, defensive)** | audit | Universal pro-tool feature, missing |
| 28 | **Weighted pass networks (xT-weighted, directional)** | audit | Basic exists, depth missing |
| 29 | **Pass sonars (polar plot of directions + distances)** | audit | Coach-friendly, not built |
| 30 | **xT / xG model trained on real data, not just heuristic** | audit | Current is "simple / heuristic" per the team's own audit |
| 31 | **Defensive action maps (tackles, interceptions, duels by zone)** | audit | Missing |
| 32 | **Pressing maps (PPDA by zone, trap locations)** | audit | PPDA exists, map doesn't |
| 33 | **Set-piece analysis (corners, free-kicks, takers)** | audit | Missing |
| 34 | **Transition analysis (attack-to-defense, defense-to-attack timing)** | audit | Missing |
| 35 | **Physical load tracking (HSR, sprints, accelerations)** | audit | Missing |
| 36 | **Progressive passes / carries (advance >10m toward goal)** | audit | Easy add, high value |
| 37 | **Passes into final third / penalty area** | audit | Easy add, high value |
| 38 | **Pressure events (defender within 2m of carrier)** | audit | Building block for many stats |
| 39 | **Multi-match aggregation (season averages, percentiles, trends)** | audit | Coach wants "Amir this season" |
| 40 | **Opponent scouting (analyze opponent patterns)** | audit | Pre-match prep |
| 41 | **Player baselines vs team / league average** | audit | Compare a player to the squad |

### Workflow & UX (what coaches actually do)

| # | Target | Source | Why it matters |
|---|---|---|---|
| 42 | **Video clip annotation tool (draw on video, tag events)** | audit | Coaches draw on Hudl / Sportscode |
| 43 | **Side-by-side video comparison** | audit | Compare two matches visually |
| 44 | **Custom report builder (coach picks which stats)** | audit | One-size-fits-all reports are limiting |
| 45 | **Coach-friendly dashboard (not just raw stats)** | audit | Visual, prioritized, scannable |
| 46 | **Sharing / export (PDF, CSV, shareable link)** | audit | Reports are useless if locked in |
| 47 | **Mobile companion (review on phone)** | audit | Coaches review on the way to training |
| 48 | **Multi-language: extend Arabic coverage, add French / Spanish** | my critique | MENA + North Africa + LatAm market |
| 49 | **Custom formation editor (not just detect, define)** | audit | Coaches need to draw the formation they want |

### Performance & scale (the boring stuff that matters at pro level)

| # | Target | Source | Why it matters |
|---|---|---|---|
| 50 | **Lazy model loading (1.75 GB → 500 MB installer)** | review | Adoption blocker |
| 51 | **Single-command install (`uv run kawkab`)** | review | Adoption blocker |
| 52 | **GPU VRAM budget audit (can we run 4 models simultaneously?)** | review | VRAMManager exists but may not be wired |
| 53 | **Async pipeline parallelization (CV + audio + LLM in parallel)** | review | Currently sequential? |
| 54 | **Match-length video chunking (process 90 min without OOM)** | review | Long videos may crash |
| 55 | **Incremental analysis (resume from checkpoint if interrupted)** | audit | 90-min analysis should be resumable |
| 56 | **Batch processing (analyze 5 matches overnight)** | review | Real coaches have a season, not one clip |
| 57 | **CPU fallback for development (no NVIDIA required)** | review | Currently requires RTX 3060+ |

### Security & ops (what production requires)

| # | Target | Source | Why it matters |
|---|---|---|---|
| 58 | **Sandboxed LLM prompts (no prompt injection via match data)** | my critique | LLM sees video-derived text; could be poisoned |
| 59 | **Input validation on all CLI / UI entry points** | review | PySide6 + CLI = multiple attack surfaces |
| 60 | **Structured logging with privacy redaction** | review | Logs may contain PII (player faces, names) |
| 61 | **Dependency vulnerability scanning (pip-audit in CI)** | review | 50+ dependencies, no audit |
| 62 | **Reproducible builds (lockfile + pinned hashes)** | review | `uv.lock` exists, verify CI uses it |
| 63 | **Telemetry opt-in (or off by default — match the "100% private" promise)** | review | Don't add telemetry that breaks the brand |

---

## Cycle Log

> Add a row every time a cycle completes. Format: `## Cycle N — <title> (YYYY-MM-DD)`

<!-- cycles will be appended below as they complete -->

### Cycle A — Fix `services/__init__.py` latent ImportError (2026-06-17)
- **Target:** #2 in backlog (fix imports so all 49 services are importable)
- **What I did:** added missing imports for `BenchmarkService`, `BenchmarkResult`, `ValidationService`, `ValidationResult`, `ValidationReport`, `EventGroundTruth` — all 6 were in the `__all__` export list but not in the import block. Would have raised `ImportError` on any code that did `from kawkab.services import BenchmarkService`.
- **Files touched:** `src/kawkab/services/__init__.py`
- **Status:** ✅ complete (no service deleted)

### Cycle B — Reconcile version strings across docs (2026-06-17)
- **Target:** #3 in backlog
- **Diagnosis:** four different version claims existed:
  - `pyproject.toml`: 0.1.0
  - `README.md`: v0.4.1
  - `PLAN.md`: current v0.5.5, target v0.6.0
  - `STATUS.md`: v0.8.3 (most recent, most detailed)
- **Decision:** STATUS.md is the source of truth. v0.8.3 it is.
- **What I did:** bumped `pyproject.toml` to 0.8.3, updated README's "Honest Status" banner + section titles to v0.8.3, updated PLAN.md current version to v0.8.3 and target to v0.9.0, added a "Previous versions of this plan" note.
- **Files touched:** `pyproject.toml`, `README.md`, `PLAN.md`
- **Known follow-up:** `STATUS.md`'s "Bottom Line" section still says v0.7.2 (one section behind the header). Left for a future cycle — needs a STATUS.md owner.
- **Status:** ✅ complete

### Cycle C — Audit `.gitignore` + working-tree state (2026-06-17)
- **Target:** #15 in backlog
- **Diagnosis:** expected a 363 MB test video in git. Investigated.
- **What I found:** the 363 MB `Sweden vs Tunisia` video is in the working tree but **NOT tracked by git** — `.gitignore` already excludes `*.mp4`. No cleanup needed.
- **Working-tree state:** 21+ files modified and uncommitted. Not a bug — just a heads-up to commit working progress.
- **Status:** ✅ complete (audit only, no changes)

### Cycle D — Add documentation index (2026-06-17)
- **Target:** #16 in backlog
- **What I did:** created `docs/INDEX.md` mapping all 10+ docs by audience, with a "document authority" table to resolve version/feature conflicts. New contributors (human or AI) now have a single map.
- **Files touched:** `docs/INDEX.md` (new)
- **Status:** ✅ complete

### Cycle E — Audit existing CI (2026-06-17)
- **Target:** #12 in backlog
- **What I found:** CI already exists at `.github/workflows/test.yml` (cross-platform tests on win/ubuntu/macos with pytest + coverage + ruff + black + mypy) and `.github/workflows/build.yml` (PyInstaller + Inno Setup installer on tag). This is more comprehensive than I assumed in the original review.
- **What I did:** nothing to add. Audited only.
- **Status:** ✅ complete (no changes needed)

### Cycle F — Resolve clip-service name collision (2026-06-17)
- **Target:** #9 in backlog
- **Diagnosis:** `services/clip_service.py` (182 lines) and `services/clip_extraction_service.py` (307 lines) both defined a class called `ClipExtractionService`. Real name-collision bug.
- **Decision (Mavis chose):** **Rename** `clip_extraction_service.py`'s class to `ClipLibraryService` rather than merge or defer.
  - Why not merge: the two services have different responsibilities (fire-and-forget extract vs library management with thumbnails/playlists/storage) and different APIs (async vs sync). A 2-hour god-merge would be brittle.
  - Why not defer: leaves a real bug in place.
  - Why rename: fixes the collision in 15 minutes, preserves all 8 tests with updated imports, keeps both services clean.
- **What I did:**
  - Renamed `ClipExtractionService` → `ClipLibraryService` in `clip_extraction_service.py`
  - Updated the module docstring with a deprecation-rename note pointing to the cycle log
  - Updated `tests/unit/test_clip_extraction_service.py` to use the new name
  - Added `ClipLibraryService`, `VideoClip`, `ClipPlaylist` to `services/__init__.py` exports
- **Files touched:** `src/kawkab/services/clip_extraction_service.py`, `tests/unit/test_clip_extraction_service.py`, `src/kawkab/services/__init__.py`
- **Status:** ✅ complete

### Cycle G — Resolve cv_service v1 vs v2 ambiguity (2026-06-17)
- **Target:** #8 in backlog
- **Diagnosis:** `cv_service.py` (1018 lines, canonical, used in production) has `swap_teams()` for the v0.8.3 team-swap feature. `cv_service_v2.py` (430 lines, experimental) claimed to add smart filters.
- **MAJOR FINDING (changed decision):** after reading `cv_service.py` carefully, **the canonical v1 already has the smart filters** — `min_track_lifetime` (line 437), `lifetime_pct` (line 440), `max_keep_top_n` (line 444), team color clustering via k-means (line 486), `_assess_tracking_quality` (line 471), Norfair tracker with enhanced ReID (line 347). The function docstring even says "Process a full video with smart track filtering (v2)". **The v2 fork is outdated, not ahead.**
- **Decision (Mavis chose):** **Add a clear "outdated fork, do not import" header to v2.** Deferred the delete to a user decision (deletion is irreversible from the working tree).
  - Why not port: the smart filters are already in v1. There's nothing to port.
  - Why not remove: irreversible without explicit user sign-off. User said "no service cuts."
  - Why not keep silently: the file name `cv_service_v2.py` strongly suggests it's the "newer" version. A contributor would reach for it. A clear header prevents that.
- **What I did:**
  - Updated `cv_service_v2.py` module docstring to explain that v1 has the smart filters, list the line numbers in v1, and recommend against importing
  - Added a runtime `logger.warning` to flag any accidental import
- **Files touched:** `src/kawkab/services/cv_service_v2.py`
- **Next decision (your call):** delete the file, or keep as a frozen historical reference?
- **Status:** ✅ complete (deferred delete)

### Cycle H — Cycle 1 prep: video pipeline failure-point analysis (2026-06-17)
- **Target:** enable Cycle 1 to be a 1-cycle fix, not a 5-cycle investigation
- **What I did:** read `cv_service.py` (process_video entry point, lines 307-506), catalogued the most likely failure points for an amateur video, wrote a debugging checklist for the user.
- **Deliverable:** `docs/CYCLE_1_VIDEO_PIPELINE.md` with:
  - The pipeline at a glance (4 stages: CV → Analysis → Reasoning → LLM)
  - 8 ranked likely failure points (YOLO model missing, BoT-SORT fallback, team color clustering failure, homography not applied, frame skip extrapolation, max speed cap, LLM hallucination, rules firing on unmeasured metrics)
  - A 10-question debugging checklist the user can run through
  - "What success looks like" for Cycle 1 (8 acceptance criteria)
  - A "files to read when debugging" mapping
- **Why:** the user said "I tested a video." Until I know what broke, I can't fix it. This doc turns "tell me what happened" into a 10-question form the user can answer in 2 minutes.
- **Status:** ✅ complete (prep work)

### Cycle 1 — Make video pipeline work end-to-end on user's real match video
- **Target:** #1 in backlog
- **Status:** ⏸ pending — user needs to run through the 10-question checklist in `docs/CYCLE_1_VIDEO_PIPELINE.md` and report what happened

### Cycle I — Phase 1: Fix version/service-count inconsistencies across all docs (2026-07-01)
- **Target:** #3 in backlog (partially done in Cycle B, completed here)
- **What I did:** fixed STATUS.md "Bottom Line" from v0.7.2 → v0.12.0, footer from v0.8.3 → v0.12.0, service count 30+ → 57. Fixed README.md version refs v0.4.1/v0.8.3 → v0.12.0. Updated PROFESSIONAL_AUDIT.md from v0.6.0 → v0.12.0. Added v0.12.0 context note to REVIEW.md. Updated docs/INDEX.md from v0.8.3 → v0.12.0. Updated PLAN.md from v0.8.3 → v0.12.0. Updated pyproject.toml version from 0.8.3 → 0.12.0.
- **Files touched:** `STATUS.md`, `README.md`, `PROFESSIONAL_AUDIT.md`, `REVIEW.md`, `docs/INDEX.md`, `PLAN.md`, `pyproject.toml`
- **Status:** ✅ complete

### Cycle J — Phase 0-A/C: Real e2e video pipeline test (2026-07-01)
- **Target:** #6 in backlog (end-to-end integration test on a fixture video)
- **What I did:** wrote `tests/e2e/test_e2e_video_pipeline.py` with CI-safe video structure tests (5 tests: existence, duration, dimensions, frame format) and full-pipeline tests that require ultralytics (skipped in CI). Verified `real_match.mp4` (4.2 MB, 89s) and `sweden_test_60s.mp4` (50.7 MB, 60s) as test fixtures.
- **Files touched:** `tests/e2e/test_e2e_video_pipeline.py` (new)
- **Status:** ✅ complete

### Cycle K — Phase 2-A: Benchmark cache as default YOLO selector (2026-07-01)
- **Target:** Wire benchmark-backed YOLO model selection (req for backlog #57 — CPU fallback)
- **What I did:** modified `CVService.__init__()` to check `_load_benchmark_cache()` before calling `recommend_yolo_variant()`. When a benchmark result exists for the current GPU tier, uses the cached variant. Falls back to heuristic on first run.
- **Files touched:** `src/kawkab/services/cv_service.py`
- **Status:** ✅ complete

### Cycle L — Phase 2-B: Auto-homography as default path (2026-07-01)
- **Target:** #5 in backlog (auto-homography from pitch keypoints)
- **What I did:** added auto-calibration to `CVService.process_video()` using `PitchDetector` on the first frame. When confidence ≥ 0.4, computes homography via `HomographyService.compute_homography_from_corners()` and stores in tracking_metrics. Wired `AnalysisServiceCore.analyze_match()` to auto-read it when no explicit homography is passed.
- **Files touched:** `src/kawkab/services/cv_service.py`, `src/kawkab/services/analysis/core.py`
- **Status:** ✅ complete

### Cycle M — Phase 4-A/B: Model card + Data card (2026-07-01)
- **Target:** #17, #18 in backlog
- **What I did:** wrote `docs/MODEL_CARD.md` documenting all models (YOLO, OSNet, SoccerNet, ArcFace), their training data, failure modes, and selection logic. Wrote `docs/DATA_CARD.md` documenting the tactical knowledge base (64 YAML rules), analytical models, ground-truth status, and privacy guarantees.
- **Files touched:** `docs/MODEL_CARD.md` (new), `docs/DATA_CARD.md` (new)
- **Status:** ✅ complete

### Cycle N — Phase 4-C: pip-audit in CI (2026-07-01)
- **Target:** #61 in backlog
- **What I did:** added `pip-audit` step to lint job in `.github/workflows/test.yml`, after ruff/black checks.
- **Files touched:** `.github/workflows/test.yml`
- **Status:** ✅ complete

### Cycle O — Phase 5-A: Wire ModelManager into boxmot init (2026-07-01)
- **Target:** #50 in backlog (lazy model loading)
- **What I did:** updated `CVService._init_boxmot_tracker()` to call `self._model_manager.ensure_model("osnet_sportsmot")` before initializing the BoT-SORT tracker, ensuring OSNet ReID weights are downloaded on first run.
- **Files touched:** `src/kawkab/services/cv_service.py`, `pyproject.toml` (version bump)

### Cycle P — Phase 6: boxmot v19+ upgrade + jersey OCR enhancement + xT training + ground truth + soccerplots (2026-07-01)
- **Target:** Phase 6 differentiation layer — 6 improvements across tracking, jersey OCR, analytics, validation, and viz
- **What I did:**
  - **boxmot v19+ (C++/fp16):** Updated `cv_service.py` (`BoTSORT`→`BotSort`, `model_weights`/`fp16`→`ReID`+`reid_model`/`half`), `norfair_tracker.py` (`ReidAutoBackend`→`ReID`), pyproject.toml (`>=3.0.0`→`>=19.0.0,<22`). Enables C++ native tracker backends + fp16 half-precision ReID.
  - **Jersey OCR enhancement:** Added CLAHE preprocessing + bilateral filter + adaptive upscale to jersey_service.py before EasyOCR. Created `scripts/train_jersey_cnn.py` for SoccerNet sn-jersey CNN training.
  - **xT model training:** Created `tools/train_xt_model.py` — trains 12x8 xT grid on StatsBomb open data, exports `.npy` weights.
  - **Ground truth datasets:** Created `scripts/download_ground_truth.py` — downloads SkillCorner, Metrica, StatsBomb data. Added `load_skillcorner_events()` and `load_metrica_events()` to ValidationService.
  - **soccerplots:** Added dependency. Enhanced VisualizationService with soccerplots-based pass sonar rendering (falls back to matplotlib when unavailable).
- **Files touched:** `src/kawkab/services/cv_service.py`, `src/kawkab/services/norfair_tracker.py`, `src/kawkab/services/jersey_service.py`, `src/kawkab/services/visualization_service.py`, `src/kawkab/services/validation_service.py`, `pyproject.toml`, `scripts/train_jersey_cnn.py`, `scripts/download_ground_truth.py`, `tools/train_xt_model.py`, `tests/unit/test_jersey_service.py`

---

## Operating Principles (the "professional" rules)

These are non-negotiable. If a cycle violates one, it doesn't ship.

1. **No silent failures.** Every error path must log, surface, or both. No `try/except: pass`.
2. **No dead code.** If a function is unused, delete it. If it's used but not tested, add a test.
3. **No documentation lies.** If a doc says "Working," the user can rely on it. If it doesn't, mark "Experimental" or remove.
4. **No regression.** Every change must pass existing tests. New tests added for new behavior.
5. **No silent model swaps.** If you change a model, a prompt, a tracker config — log it. The data scientist of the future needs to know.
6. **No UI changes without a screenshot in the PR.** Visual changes need visual proof.
7. **No new dependency without justification.** Each `pip install` should answer: "why this, not the alternative?"
8. **No "it works on my machine."** If it's not in CI, it's not real.
9. **No personal data in logs, exports, or test fixtures by default.** Privacy is the brand.
10. **No untracked state.** If a cycle changes behavior, the change must be reflected in docs the same day.

---

## Quality Bar (what "professional" means for Kawkab)

When in doubt, ask: "Would a StatsBomb / Second Spectrum / Hudl engineer accept this PR?"

| Dimension | Bar |
|---|---|
| Code | Type hints everywhere. mypy strict passes. ruff clean. No file > 30KB. No function > 100 lines. |
| Tests | Test-to-service ratio ≥ 1.0 for the 12 canonical services. ≥ 1 end-to-end test. |
| Docs | Every public class has a docstring. Every public function has an example. Every architecture choice has a 1-paragraph "why" in `docs/`. |
| Security | All inputs validated. No secrets in code. Privacy by default. |
| Accuracy | Every metric has a definition. Every metric has at least one ground-truth check. |
| UX | Every flow has a happy path tested manually. Every error has a user-friendly message. |
| Performance | No regression > 10% on benchmark set. Memory bounded. GPU usage reported. |
| Reproducibility | One command to install. One command to run. Lockfile in CI. |

---

## Next Session Handoff

When you come back, do this:
1. Read the "Cycle Log" at the bottom of this file
2. Pick the next `# pending` target in the backlog
3. Run a new cycle

The backlog is long. The quality bar is high. We work through it.
