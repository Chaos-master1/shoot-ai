#!/usr/bin/env python3
"""Production tracking benchmark — Kawkab's real tracker vs degraded ground truth.

Why this exists (2026-09-16): Kawkab publishes validated xG/PSxG/xT numbers
but nothing measurable about its TRACKING quality. Elite-club buyers ask for
MOTA/IDF1 the way they ask for Brier scores. This benchmark runs the actual
production tracker (``services/norfair_tracker.NorfairTracker`` — the same
code that processes real video) over ground-truth Metrica positions degraded
with seeded, reproducible detector noise, and scores the output with the
in-repo CLEAR MOT implementation (``core/mot_metrics.py``).

Rows:
  ceiling  — zero degradation (GT fed straight through as detections):
             pure association / ID quality with no detector error.
  degraded — seeded detector noise (jitter px std, drop rate, false-positive
             rate): the honest end-to-end number under realistic detector
             imperfection.

Ground truth: tests/fixtures/tracking/metrica_sample2_{home,away}.csv
(Metrica open sample game 2, first 2,000 frames — real vendor data, MIT),
parsed with evaluate_tracking.parse_metrica_csv (explicit paths, so the
committed truncated fixtures work).

Honesty rules (matching the model-validation discipline):
  - Seeded RNG (default seed 13) → reproducible reruns.
  - Deterministic GT ids (sorted jersey labels → home 1000+i, away 2000+i);
    never hash(), which differs across processes.
  - The tracker sees ONLY the degraded detections, never ground truth.
  - Persons only (ball excluded) — the convention of the existing
    evaluate_tracking.py tooling.
  - One coordinate mapping for detections AND GT (no Y-flip mismatch);
    both live in the same pixel space before scoring.
  - Tracker updated on EVERY frame (empty detection lists included), so
    occlusion frames age track state exactly like real video.
  - ReID embeddings on black frames collapse to zero vectors → the cosine
    distance is inf → ReID never fires → pure IoU+motion association.
    This measures the tracker core, not the appearance model.

Usage:
    PYTHONPATH=src python scripts/benchmark_production_tracking.py
    PYTHONPATH=src python scripts/benchmark_production_tracking.py \
        --frames 2000 --noise-px 3 --drop 0.10 --fp-rate 0.03 --seed 13
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "tracking"
PITCH_LEN_M = 105.0
PITCH_WID_M = 68.0
PLAYER_W_PX = 20.0
PLAYER_H_PX = 40.0
DET_CONFIDENCE = 0.85


def load_gt_tracks(
    fixture_dir: Path,
    frame_limit: int,
    px_per_m: float,
) -> dict[int, list[tuple[int, float, float]]]:
    """Parse both committed Metrica CSVs into {gt_id: [(frame, x_px, y_px)]}.

    Deterministic ids: home players (sorted jersey labels) get 1000+i, away
    players 2000+i. GT is mapped into the SAME pixel space the detections
    use (x*105*px_per_m, y*68*px_per_m) so metric distances are consistent.
    """
    from evaluate_tracking import parse_metrica_csv

    gt: dict[int, list[tuple[int, float, float]]] = {}
    for team, base in (("home", 1000), ("away", 2000)):
        csv_path = fixture_dir / f"metrica_sample2_{team}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"missing fixture: {csv_path}")
        tracks = parse_metrica_csv(csv_path, team)
        for i, label in enumerate(sorted(tracks.keys(), key=lambda s: (len(s), s))):
            tid = base + i
            for f in tracks[label].frames:
                if f.frame > frame_limit:
                    continue
                if f.x != f.x or f.y != f.y:
                    continue  # NaN
                gt.setdefault(tid, []).append(
                    (f.frame, f.x * PITCH_LEN_M * px_per_m, f.y * PITCH_WID_M * px_per_m)
                )
    return gt


def build_detections(
    gt_tracks: dict[int, list[tuple[int, float, float]]],
    *,
    seed: int,
    noise_px_std: float,
    drop_rate: float,
    fp_rate: float,
    max_frame: int,
) -> dict[int, list[tuple[float, float, float, float, float]]]:
    """Degrade GT into detector-shaped per-frame boxes.

    Returns {frame: [(x1, y1, x2, y2, conf), ...]} in the same pixel space
    as load_gt_tracks (GT x/y are already in pixels when passed in here).
    """
    rng = np.random.default_rng(seed)
    n_players = len(gt_tracks)
    pitch_w_px = PITCH_LEN_M * (PITCH_WID_M / PITCH_LEN_M)  # == PITCH_WID_M in px units
    # px_per_m is baked into GT already; FP boxes span the full GT extent.
    xs = [x for positions in gt_tracks.values() for _, x, _ in positions]
    ys = [y for positions in gt_tracks.values() for _, _, y in positions]
    extent_x, extent_y = max(xs), max(ys)
    hw, hh = PLAYER_W_PX / 2.0, PLAYER_H_PX / 2.0

    by_frame: dict[int, list[tuple[float, float, float, float, float]]] = {}
    for _tid, positions in gt_tracks.items():
        for frame, x, y in positions:
            if rng.random() < drop_rate:
                continue
            cx = x + rng.normal(0.0, noise_px_std)
            cy = y + rng.normal(0.0, noise_px_std)
            by_frame.setdefault(frame, []).append(
                (max(0.0, cx - hw), max(0.0, cy - hh), cx + hw, cy + hh, DET_CONFIDENCE)
            )
    # False positives: uniform over the observed pitch extent, Poisson per frame.
    for frame in range(1, max_frame + 1):
        n_fp = int(rng.poisson(fp_rate * n_players))
        for _ in range(n_fp):
            cx = float(rng.uniform(0.0, extent_x))
            cy = float(rng.uniform(0.0, min(extent_y, pitch_w_px)))
            by_frame.setdefault(frame, []).append(
                (cx - hw, cy - hh, cx + hw, cy + hh, float(rng.uniform(0.3, 0.7)))
            )
    return by_frame


def run_tracker_row(
    name: str,
    detections: dict[int, list[tuple[float, float, float, float, float]]],
    gt_tracks: dict[int, list[tuple[int, float, float]]],
    *,
    max_frame: int,
    match_threshold: float,
) -> dict[str, Any]:
    """Drive the PRODUCTION NorfairTracker over the detections and score it."""
    from kawkab.core.mot_metrics import compute_mot_metrics
    from kawkab.services.norfair_tracker import NorfairTracker

    tracker = NorfairTracker()
    # Frame content is irrelevant on purpose: on black frames the
    # MotionEstimator finds no features (→ identity transform) and ReID
    # embeddings are zero (→ ReID off). Size must be SMALL — the estimator's
    # Lucas-Kanade pass runs on the full frame every update, and a broadcast-
    # sized zero frame made the whole benchmark compute-bound (85 s / 400
    # frames). Tracking geometry is unaffected: detections carry absolute
    # pixel coordinates regardless of frame dimensions.
    dummy_frame = np.zeros((96, 128, 3), dtype=np.uint8)

    pred: dict[int, list[tuple[int, float, float]]] = {}
    t0 = time.perf_counter()
    for frame in range(1, max_frame + 1):
        dets = [
            {"bbox": (x1, y1, x2, y2), "confidence": conf, "label": "person"}
            for (x1, y1, x2, y2, conf) in detections.get(frame, [])
        ]
        for out in tracker.update(dummy_frame, dets, period=1):
            if out["label"] != "person":
                continue
            x1, y1, x2, y2 = out["bbox"]
            pred.setdefault(int(out["track_id"]), []).append(
                (frame, (x1 + x2) / 2.0, (y1 + y2) / 2.0)
            )
    elapsed = time.perf_counter() - t0

    mot = compute_mot_metrics(pred, gt_tracks, fp_threshold=match_threshold, is_normalized=False)
    return {
        "row": name,
        "elapsed_s": round(elapsed, 2),
        "n_pred_tracks": len(pred),
        "n_gt_tracks": len(gt_tracks),
        **mot,
    }


def main(argv: list[str] | None = None) -> int:
    # Norfair's MotionEstimator logs a WARNING on every frame where no
    # homography can be computed — which is EVERY frame here (black frames).
    # Silence root-logger warnings so the report stays readable.
    import logging

    logging.getLogger().setLevel(logging.ERROR)

    ap = argparse.ArgumentParser(
        description="Benchmark Kawkab's production tracker against degraded Metrica GT",
    )
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument(
        "--noise-px",
        type=float,
        default=3.0,
        help="detector jitter std in pixels (degraded row). "
        "Calibration: a 20x40 px box survives IoU>=0.6 "
        "matching up to ~5 px lateral shift, so per-frame "
        "jitter must stay well under that — 10 px std "
        "(tried first) is unphysical: NO track ever "
        "initializes and the row degenerates to all-FN.",
    )
    ap.add_argument("--drop", type=float, default=0.10, help="detection drop rate (degraded row)")
    ap.add_argument(
        "--fp-rate",
        type=float,
        default=0.03,
        help="false positives per player per frame (degraded row)",
    )
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument(
        "--px-per-m",
        type=float,
        default=20.0,
        help="pixel scale used consistently for detections and GT",
    )
    ap.add_argument(
        "--match-threshold",
        type=float,
        default=40.0,
        help="CLEAR MOT match distance threshold in pixels",
    )
    ap.add_argument("--out", default="docs/validation/tracking_benchmark.json")
    args = ap.parse_args(argv)

    gt = load_gt_tracks(FIXTURE_DIR, args.frames, args.px_per_m)
    if not gt:
        print("no ground truth loaded — fixtures missing?", file=sys.stderr)
        return 1
    n_positions = sum(len(p) for p in gt.values())
    print(
        f"[gt] {len(gt)} players, {n_positions} positions "
        f"(frames 1..{args.frames}, Metrica sample game 2, MIT)"
    )

    rows: list[dict[str, Any]] = []

    ceiling_dets = build_detections(
        gt,
        seed=args.seed,
        noise_px_std=0.0,
        drop_rate=0.0,
        fp_rate=0.0,
        max_frame=args.frames,
    )
    rows.append(
        run_tracker_row(
            "ceiling",
            ceiling_dets,
            gt,
            max_frame=args.frames,
            match_threshold=args.match_threshold,
        )
    )

    degraded_dets = build_detections(
        gt,
        seed=args.seed,
        noise_px_std=args.noise_px,
        drop_rate=args.drop,
        fp_rate=args.fp_rate,
        max_frame=args.frames,
    )
    rows.append(
        run_tracker_row(
            "degraded",
            degraded_dets,
            gt,
            max_frame=args.frames,
            match_threshold=args.match_threshold,
        )
    )

    report = {
        "status": "ok",
        "generated_at": datetime.now(UTC).isoformat(),
        "provenance": {
            "ground_truth": (
                f"Metrica open sample game 2 (MIT), first {args.frames} frames, committed fixtures"
            ),
            "tracker": "kawkab.services.norfair_tracker.NorfairTracker (production)",
            "metrics": "kawkab.core.mot_metrics.compute_mot_metrics (CLEAR MOT)",
            "seed": args.seed,
            "degradation": {
                "noise_px_std": args.noise_px,
                "drop_rate": args.drop,
                "fp_rate_per_player_per_frame": args.fp_rate,
            },
            "pixel_scale_px_per_m": args.px_per_m,
            "match_threshold_px": args.match_threshold,
            "ball_tracking": "excluded (persons only; evaluate_tracking convention)",
            "reid": "inactive on black frames (zero embeddings → inf distance) — "
            "measures the tracker core, not the appearance model",
        },
        "rows": rows,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 64}")
    print("  PRODUCTION TRACKING BENCHMARK (NorfairTracker)")
    print("=" * 64)
    for r in rows:
        print(f"\n  {r['row']}  ({r['elapsed_s']}s, {r['n_pred_tracks']} pred tracks)")
        print(f"    MOTA        {r['mota']:.4f}")
        print(f"    MOTP        {r['motp']:.2f} px")
        print(f"    IDF1        {r['idf1']:.4f}")
        print(f"    ID swaps    {r['id_switches']}")
        print(f"    Fragments   {r['fragments']}")
        print(f"    FP / FN     {r['false_positives']} / {r['false_negatives']}")
    print(f"\n  report -> {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
