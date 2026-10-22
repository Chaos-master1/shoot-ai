# Model Card — Kawkab AI

## Models in Use

### YOLOv11 (n/s/m/l/x)

| Property | Value |
|---|---|
| **Task** | Object detection (person, sports ball) |
| **Architecture** | Ultralytics YOLOv11 (CSPDarknet + PAN neck) |
| **Weights** | `yolo11{n/s/m/l/x}.pt` from ultralytics assets |
| **Training data** | COCO 2017 (118k images, 80 classes, includes "person") |
| **Fine-tuning** | None — used as-is from ultralytics |
| **Input** | BGR image (variable resolution, auto-scaled by YOLO) |
| **Output** | Bounding boxes + class labels + confidence scores |
| **Classes used** | `0` (person), `32` (sports ball) |
| **Confidence threshold** | Person: 0.4, Ball: 0.15 |
| **Failure modes** | Occluded players, distant players (<40px tall), overlapping kits, unusual camera angles, extreme lighting |

### OSNet SportsMOT (boxmot)

| Property | Value |
|---|---|
| **Task** | Person ReID embedding |
| **Architecture** | OSNet (Omni-Scale Network) |
| **Weights** | `osnet_x1_0_msmt17.pt` auto-downloaded by boxmot at first use (verified against boxmot's TRAINED_URLS registry; sha256-pinned entries for all other weights live in `core/model_manager.py`) |
| **Training data** | SportsMOT (multi-sport tracking dataset) |
| **Output** | 512-d L2-normalized embedding vector |
| **Used in** | boxmot BoT-SORT tracker (GPU tier medium+); boxmot downloads and manages the weights itself |
| **Failure modes** | Same-kit players on same team, extreme motion blur, very low resolution crops |

### SoccerNet ReID (optional)

| Property | Value |
|---|---|
| **Task** | Football-specific person ReID embedding |
| **Architecture** | ResNet-50 + CircleLoss |
| **Weights** | `soccernet_reid.pt` (200 MB) — **local file only** (`core/model_manager.py` deliberately ships no download URL; no trustworthy public host exists) |
| **Training data** | SoccerNet tracking dataset (football matches) |
| **Output** | 512-d L2-normalized embedding vector |
| **Used in** | Norfair tracker (fallback after OSNet) |
| **Status** | Optional dependency (`pip install kawkab[reid]`) |

### ArcFace (face recognition)

| Property | Value |
|---|---|
| **Task** | Face embedding for cross-cut identity verification |
| **Architecture** | InsightFace ArcFace (ResNet-100 backbone) |
| **Training data** | MS1M-V2 (5.8M faces, 85k identities) |
| **Output** | 512-d normalized embedding |
| **Used in** | Post-hoc track stitching (A1), sampled every 2s |
| **Failure modes** | Players facing away, distant faces (<30px), low light, occlusion by other players |

## Model Selection Logic

```
GPU tier "ultra" → YOLO11x + boxmot BoT-SORT (OSNet)
GPU tier "high"  → YOLO11l + boxmot BoT-SORT (OSNet)
GPU tier "medium" → YOLO11m + boxmot BoT-SORT (OSNet)
GPU tier "low"   → YOLO11s + Norfair (HSV ReID)
Fallback (no GPU) → YOLO11n + Ultralytics built-in tracker
```

Benchmark cache overrides static tier mapping. After one successful benchmark run per GPU tier, the best measured variant is used instead of the heuristic.

## Known Limitations

1. **YOLO "ball" class** is COCO's sports ball — on amateur footage it frequently misses the ball or misclassifies small objects as ball. Ball tracking is less reliable than person tracking.
2. **No team-specific training** — all weights are generic (COCO, SportsMOT). Football-specific fine-tuning would improve detection of players in match kits vs. warm-up kits.
3. **No occlusion model** — when two players overlap, one track may drop or IDs may swap. The track stitching module recovers some of these post-hoc.
4. **ReID fails on identical kits** — teammates wearing the same kit can only be distinguished by face (sparse) or spatial position (breaks down at half-time formation reset).
5. **Homography depends on pitch lines** — if the pitch has no visible lines (muddy field, snow, unusual markings), auto-calibration will fail and the analysis will run in pixel space.

---

## xG Model (Expected Goals) — trained & validated

| Property | Value |
|---|---|
| **Architecture** | Logistic regression, 16 features (distance, distance², opening angle terms, body part, assist/technique flags, GK distance + squared, pressure, one-on-one, rebound, big-chance) |
| **Training data** | 297 StatsBomb open-data matches (7,354 non-penalty shots; train split 5,905) |
| **Validation** | Held-out 1,449 shots, match-level split (seed 13) — no leakage |
| **Weights** | `src/kawkab/core/trained_xg_coefficients.json` (auto-loaded; heuristic fallback if missing) |
| **Provenance API** | `EnhancedXgModel().coeffs_source`, `kawkab.core.xg_model.trained_model_available()` |
| **Retrain** | `PYTHONPATH=src python -m kawkab.core.validation.train_xg --corpus data/statsbomb_corpus --force` |
| **Full report** | `docs/validation/xg_training_report.json`, `docs/validation/REPORT.md` |

### Measured performance (held-out validation, 1,449 shots)

| Model | Brier ↓ | AUC ↑ | LogLoss ↓ | ECE ↓ | mean xG |
|---|---|---|---|---|---|
| **kawkab trained** | **0.0707** | **0.7724** | 0.2547 | 0.0143 | 0.0999 |
| kawkab heuristic (pre-training) | 0.0804 | 0.7715 | 0.3289 | 0.0682 | 0.0195 |
| StatsBomb xG (reference bar) | 0.0667 | 0.7797 | 0.2454 | 0.0089 | 0.0941 |

The trained model is within 6% of StatsBomb's proprietary xG on Brier score
(0.0707 vs 0.0667) and close on discrimination (AUC 0.772 vs 0.780). The
pre-training heuristic underestimated xG ~5× (mean 0.020 vs the true 0.094
goal rate) and was 5× worse calibrated (ECE 0.068 vs 0.014).

**Feature notes (honest):** GK-distance and one-on-one features come from
StatsBomb freeze frames — available for this validation corpus but *not* yet
extracted by Kawkab's own CV pipeline, so live-video xG currently runs with
gk_distance=0 (the trained model treats 0 as "feature absent"). Rebound and
big-chance flags are False in the current loader (no open-data source).
Distance and angle — the dominant features — are always real.

**Data license:** StatsBomb open data is CC BY-NC-SA 4.0 (non-commercial).
Commercial deployments must retrain on licensed data or keep the heuristic
fallback; see `docs/DATA_CARD.md`.

---

## PSxG Model (Post-Shot xG) — trained & validated

| Property | Value |
|---|---|
| **Architecture** | Logistic regression, 9 features (distance, distance², opening angle, placement height, lateral offset, lateral×height interaction, header, free kick) |
| **Training data** | 2,768 StatsBomb on-target shots with 3D end_location (2,219 train) |
| **Validation** | 549 held-out shots, match-level split (seed 13) |
| **Weights** | `src/kawkab/core/trained_psxg_coefficients.json` (auto-loaded; hand-tuned fallback if missing) |
| **Retrain** | `PYTHONPATH=src python -m kawkab.core.validation.train_psxg --force` |
| **Full report** | `docs/validation/psxg_training_report.json` |

### Measured performance (549 held-out on-target shots)

| Metric | Value |
|---|---|
| Brier | **0.1673** |
| AUC | **0.7533** |
| ECE (10-bin) | 0.0537 |
| Mean PSxG vs true rate | 0.3057 vs 0.2750 (11% high — see note) |

Empirical goal rates by goal-mouth zone (from the same corpus) confirm the
learned geometry: top corners score most (~0.34), top-center least (~0.16),
which the earlier blended center-distance feature got backwards until the
empirical zone table forced the lateral×height interaction redesign.

**Supersedes** the three hand-tuned implementations (psxg_model.py's
coefficients — API kept for goalkeeper_analytics; psxg_improved.py and
psxg_model_trained.py — zone-grid heuristics, kept for backward compat).

---

## xT Reference Grid (Expected Threat) — league-wide prior

| Property | Value |
|---|---|
| **Architecture** | 20×32 zone transition matrix + power iteration (the shipped `ExpectedThreatModel` algorithm, unchanged) |
| **Training data** | 297 matches / 534,927 pass+carry+shot actions, direction-normalized |
| **Grid asset** | `src/kawkab/core/trained_xt_grid.json` (auto-loaded as cold-start default when a match has <200 actions) |
| **Retrain** | `PYTHONPATH=src python -m kawkab.core.validation.train_xt --force` |
| **Sanity check** | max own-half zone 0.0245 < max final-third zone 0.8966 (monotonic ✓) |

Sparse matches now get a league-wide threat prior instead of an all-zero
grid; data-rich matches still learn from their own events (reference only
blends in below the 200-action threshold).

---

## Player Tracking (NorfairTracker) — association benchmark

First measured tracking benchmark (2026-09-16). Run the production
`NorfairTracker` — the exact code path used on user video, including
MotionEstimator camera compensation — over ground-truth player positions
from the committed Metrica sample fixture, and score with the project's own
`compute_mot_metrics` (MOTA/MOTP/IDF1).

| Property | Value |
|---|---|
| **Script** | `scripts/benchmark_production_tracking.py` (deterministic, seeded) |
| **Corpus** | Metrica sample 2 home+away CSV (committed fixture), 22 players, 43,978 GT positions, 200 frames @ 25 fps, 20 px/m |
| **Rows** | *Ceiling* = GT as perfect detections; *Degraded* = 3 px jitter σ, 10% drop, 2% false positives (realistic detector noise at this scale) |

| Row | MOTA ↑ | MOTP ↓ | IDF1 ↑ | ID swaps | Fragments | FP / FN |
|---|---|---|---|---|---|---|
| **Ceiling** | **0.9903** | 0.08 px | **0.9954** | 24 | 12 | 280 / 122 |
| **Degraded** | **0.9155** | 4.62 px | **0.9620** | 268 | 52 | 3,101 / 347 |

**Reading:** with perfect detections the tracker's association is
near-perfect (MOTA 0.99) — the ceiling on pipeline quality is the detector,
not the tracker. Under realistic noise, identity survives well (IDF1 0.962;
swap rate ≈ 0.6% of positions) and MOTA degrades gracefully (−8.5 pp, most
of it the injected false positives). The regression test
(`tests/unit/test_tracking_benchmark.py`) pins ceiling > degraded on both
MOTA and IDF1, and pins the degraded row against collapse (the unphysical
10 px-jitter config that zeroed it during development).

**Honest scope:** this measures *tracker association quality on synthetic
detections derived from GT* — not end-to-end camera→detection→tracking
accuracy. A true end-to-end MOTA on labeled match video (e.g. SoccerNet
tracking) remains open work, as does ball tracking (the fixture is players
only). Numbers are reproducible offline in ~25 s.
