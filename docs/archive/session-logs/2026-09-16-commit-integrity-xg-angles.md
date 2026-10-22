## What was fixed this session (2026-09-16 — commit-integrity + xG angle-convention pass)

**P0 — HEAD (907798a) was a broken commit.** The 2026-09-15 commit applied
COMMIT-PLAN.md's "exclude the concurrent xG session's files" rule at
whole-file granularity instead of hunk granularity, so the committed tree
referenced files that were never committed:
- `ui/bridge_handlers/__init__.py` imported `bridge_pro_analytics.py` /
  `bridge_season_analytics.py` → `import kawkab.ui.bridge` crashed on a
  fresh checkout of HEAD.
- `core/xg_model.py` loaded `trained_xg_coefficients.json` (masked at
  runtime by the heuristic fallback) and `__main__.py` wired a `validate`
  CLI whose `scripts/validate_models.py` + `core/validation/train_*.py`
  were also untracked.
Fixed by committing the working tree in thematic groups (analytics
handlers first — that alone repairs the import crash; then models; then
the 2026-09-02 accuracy-audit fixes; then docs), with a scripted
tracked→untracked reference audit run before staging.

**P1 — the trained xG fit was built on the wrong angle convention (real
bug, caught by the committed sanity guard shipping in the same commit):**
- `validation/statsbomb_loader.shot_distance_angle` returns the
  goal-OPENING angle (posts subtended from the shot; central ≈ 36°,
  goal-line-wide ≈ 90°+). `EnhancedXgModel` (and the CV pipeline via
  `atan2(|dy|,|dx|)`) consumes the DEVIATION-from-central convention
  (0° = central; `1 − cos(angle)` with a negative coefficient). Training
  on opening angle taught the model that WIDE shots score MORE —
  `TestTrainedXgCoefficientsSanity` failed with 11m-central = 0.057 and
  11m-wide = 0.49 — and distance decay collapsed onto the angle term
  (fitted `angle_sin` = +5.89 on `1 − cos`). Same disease the 56-shot
  fit had; scripts/train_xg_from_statsbomb.py's `_get_angle` had already
  been fixed for exactly this and its docstring documents the history.
- Fix: added `shot_deviation_angle` to the loader (opening angle KEPT —
  PSxG's serving model is calibrated for it and its tests pin it) +
  `StatsBombShot.angle_deviation_deg` (appended LAST so positional
  constructions keep their meaning). `train_xg.py` feeds the deviation
  on both the fit and predict boundaries; `statsbomb_import_service`
  stores `angle_deviation_deg` alongside `angle_deg` (metadata key
  `angle_deg` stays OPENING — the PSxG serving path reads it with a
  positive-coefficient convention; live-CV events already emit
  deviation-style `angle_to_goal_deg`, so imports are now consistent
  with them).
- Retrained from the committed 297-match corpus (offline, no network):
  Brier 0.0707 / AUC **0.7724** (was 0.7667) / ECE 0.0143 / mean xG
  0.0999 vs true 0.10 — wide < central now holds, distance decays.
  `docs/validation/REPORT.md` + `MODEL_CARD.md` regenerated with the
  corrected numbers.
- Convention table (put this near any angle-eating model): xG →
  deviation (0 = central); PSxG → opening (posts subtended). The
  DB/event metadata `angle_deg` key holds OPENING for imports and the
  CV pipeline stores `angle_to_goal_deg` (deviation) — a known wart,
  documented rather than silently renamed.

**P0.1 — first measured tracking benchmark (mot_metrics finally exercised
in anger).** `mot_metrics` was complete and tested but nothing ever ran the
*production* tracker through it. New
`scripts/benchmark_production_tracking.py`: drives the real
`NorfairTracker` (norfair installed into `.venv-linux` via uv — declared
dependency) over GT player positions from the committed Metrica fixture,
converted to a 20 px/m pixel grid, scored with the project's own
`compute_mot_metrics`. Two rows: *ceiling* (GT as perfect detections —
MOTA 0.9903, IDF1 0.9954: association is near-perfect, the detector is the
ceiling) and *degraded* (3 px jitter σ, 10% drop, 2% FPs — MOTA 0.9155,
IDF1 0.9620, 268 swaps / 43,978 positions ≈ 0.6%). Deterministic (same
seed → byte-identical metrics). Numbers + honest scope caveats published in
MODEL_CARD.md ("Player Tracking — association benchmark"). Regression test
`tests/unit/test_tracking_benchmark.py` (9 tests): pins ceiling > degraded
on MOTA + IDF1, pins degraded against collapse (the first degraded config
used 10 px jitter on ~20 px boxes — consecutive-frame IoU fell below the
tracker's 0.6 match threshold so no track ever initialized: physically
unrealistic, caught by the test, calibrated to 3 px), and pops/restores
norfair stubs around the real-norfair e2e test so it neither depends on
import order nor leaks real norfair into stubbed tests (the leak direction
that bit test_cv_service). E2E window sized at 120 frames deliberately:
the tracker's 3-frame init costs a fixed ~66 FN, which is 3.75% of an
80-frame window (ceiling MOTA 0.899 there) but amortized at 120. Still
open, honestly scoped: end-to-end camera→detection→tracking MOTA on
labeled video, and ball tracking (fixture is players only).

**Environment note:** this session ran in `.venv-linux` (Python 3.12).
5 `test_raindrop_detection_service.py` failures are env-only (this venv's
opencv build lacks `groupRectangles`; Windows CI has it) and 1
`test_easy_soccer_service.py::test_get_event` failure likewise (the
service does `import esd` inside the method; esd isn't installed here —
CLAUDE.md documents playwright as esd's undeclared dependency). Not code
bugs; not chased further.

