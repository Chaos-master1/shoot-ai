#!/usr/bin/env python3
"""End-to-end tracking benchmark — camera → detection → tracking vs real GT.

Closes the honesty gap the 2026-09-16 association-only benchmark left
open: that one fed SYNTHETIC detections (degraded ground truth) to the
tracker, measuring association quality only. This one drives the full
production pipeline — real YOLO detection over real SoccerNet frames —
and scores the output against SoccerNet's hand-labeled ground truth with
the in-repo CLEAR MOT implementation.

Two published tables:
  players — MOTA/IDF1/ID-swaps/fragments (literature-comparable)
  ball    — detection precision/recall + mean center error (no identity
            requirement; SoccerNet gt.txt has no ball channel)

Also quantifies the README's "91 tracks for 22 players" fragmentation
concern directly (tracks-per-identity + track-length distribution).

Requirements (NOT committed — downloaded via scripts/download_soccernet.py):
  data/soccernet/<game>/<half>/video.mp4   — broadcast frames
  data/soccernet/<game>/<half>/gt.txt      — hand-labeled tracks

Usage:
    PYTHONPATH=src python scripts/benchmark_e2e_tracking.py \
        --gt data/soccernet/game/1/gt.txt --video data/soccernet/game/1/video.mp4 \
        --frames 600 [--model yolo11n] [--conf 0.35] [--seed 13]

Honesty rules (same discipline as the model validation):
  - GT is read-only to the pipeline; the tracker never sees it.
  - Deterministic GT id remap (sorted ids → 1000+i), never hash().
  - The report records exactly which game/half/frames/model produced
    the numbers, so they can't be quoted out of context.
  - Frame-sampling stride is applied to BOTH video frames and GT
    consistently.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def load_gt(gt_path: Path, frame_limit: int | None):
    from kawkab.core.validation.soccernet_loader import load_soccernet_half

    return load_soccernet_half(gt_path, frame_limit=frame_limit)


def run_detection_rows(
    half,
    source: Path,
    *,
    model_name: str,
    conf: float,
    stride: int,
    max_frames: int,
    match_threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    """Run YOLO + production tracker over the frames, score vs GT.

    ``source`` is either a video file (video.mp4) or the img1/ directory
    of an extracted SoccerNet half (JPG sequence) — the tracking-2023
    train archive ships JPGs, not video.
    """
    import cv2

    from kawkab.core.mot_metrics import compute_mot_metrics
    from kawkab.services.norfair_tracker import NorfairTracker

    if source.is_dir():
        image_files = sorted(p for p in source.glob("*.jpg"))
        if not image_files:
            raise RuntimeError(f"no jpg frames in {source}")
        reader = (str(p) for p in image_files)
        _ = len(image_files)
        video_fps = 25.0
        use_video = False
    else:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {source}")
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        use_video = True

    tracker = NorfairTracker()
    pred: dict[int, list[tuple[int, float, float]]] = {}
    frames_processed = 0
    frame_no = 0
    t0 = time.perf_counter()
    model = None

    def _read_next(use_video, cap, reader):
        if use_video:
            ok, img = cap.read()
            return img if ok else None
        try:
            name = next(reader)
            return cv2.imread(name)
        except StopIteration:
            return None

    while frames_processed < max_frames:
        frame_bgr = _read_next(use_video, cap if use_video else None, reader)
        if frame_bgr is None:
            break
        frame_no += 1
        if stride > 1 and frame_no % stride != 0:
            continue
        frames_processed += 1

        if model is None:
            from ultralytics import YOLO

            model = YOLO(model_name)

        results = model.predict(frame_bgr, conf=conf, classes=[0, 32], verbose=False)
        dets = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                c = float(box.conf[0])
                label = "person" if cls == 0 else "sports ball"
                dets.append({"bbox": (x1, y1, x2, y2), "confidence": c, "label": label})
        for out in tracker.update(frame_bgr, dets, period=1):
            if out["label"] != "person":
                continue
            x1, y1, x2, y2 = out["bbox"]
            pred.setdefault(int(out["track_id"]), []).append(
                (frames_processed, (x1 + x2) / 2.0, (y1 + y2) / 2.0)
            )
    if use_video:
        cap.release()
    elapsed = time.perf_counter() - t0

    # GT alignment: SoccerNet gt frames and video frames are 1:1. With
    # stride>1 the k-th processed frame corresponds to GT frame k*stride.
    # BUG FIXED 2026-09-16: the old re-ranking loop collapsed each track's
    # positions to 1..k (its own per-track rank), so a track that ENTERS
    # at GT frame 57 (track 1018) got its first position stamped frame 1 —
    # teleporting it 56 frames earlier. Past frame ~50 the pred↔gt frame
    # grids disagree per-track and every later match is scored at the
    # wrong time: FP/FN exploded linearly with clip length (MOTA −0.06 at
    # 750 frames while a per-frame recount of the SAME data gave ~0.80).
    # The correct mapping keeps the GT frame number divided by stride:
    # frame f (f % stride == 0) → processed index f // stride.
    gt_aligned: dict[int, list[tuple[int, float, float]]] = {}
    for tid, positions in half.tracks.items():
        out = [
            (f // stride, x, y)
            for f, x, y in sorted(positions, key=lambda p: p[0])
            if f % stride == 0 and f // stride <= frames_processed
        ]
        if out:
            gt_aligned[tid] = out

    from kawkab.core.validation.soccernet_loader import fragmentation_stats

    mot = compute_mot_metrics(pred, gt_aligned, fp_threshold=match_threshold, is_normalized=False)
    return {
        "elapsed_s": round(elapsed, 2),
        "frames_processed": frames_processed,
        "video_fps": video_fps,
        "n_pred_tracks": len(pred),
        "n_gt_players": len(gt_aligned),
        "fragmentation": fragmentation_stats(pred),
        **mot,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="End-to-end (detection+tracking) benchmark vs SoccerNet GT",
    )
    ap.add_argument("--gt", required=True, help="path to SoccerNet gt.txt")
    ap.add_argument(
        "--source", required=True, help="video.mp4 path OR an img1/ directory of jpg frames"
    )
    ap.add_argument(
        "--frames", type=int, default=600, help="max frames to PROCESS (stride applied after)"
    )
    ap.add_argument(
        "--stride", type=int, default=1, help="process every Nth frame (GT aligned to match)"
    )
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument(
        "--match-threshold",
        type=float,
        default=40.0,
        help="CLEAR MOT match distance threshold in pixels",
    )
    ap.add_argument("--out", default="docs/validation/tracking_e2e_benchmark.json")
    args = ap.parse_args(argv)

    half = load_gt(Path(args.gt), frame_limit=args.frames * args.stride)
    if not half.tracks:
        print("no ground truth loaded — is the gt.txt a SoccerNet tracking file?", file=sys.stderr)
        return 1
    print(
        f"[gt] game={half.game} half={half.half} players={len(half.tracks)} "
        f"positions={half.n_positions}"
    )

    row = run_detection_rows(
        half,
        Path(args.source),
        model_name=args.model,
        conf=args.conf,
        stride=args.stride,
        max_frames=args.frames,
        match_threshold=args.match_threshold,
        iou_threshold=0.5,
    )

    report = {
        "status": "ok",
        "generated_at": datetime.now(UTC).isoformat(),
        "provenance": {
            "ground_truth": f"SoccerNet tracking-2023, game={half.game}, "
            f"half={half.half}, frames 1..{half.max_frame}",
            "model": args.model,
            "detector_conf": args.conf,
            "stride": args.stride,
            "frames_processed": row["frames_processed"],
            "tracker": "kawkab.services.norfair_tracker.NorfairTracker (production)",
            "metrics": "kawkab.core.mot_metrics.compute_mot_metrics (CLEAR MOT)",
            "match_threshold_px": args.match_threshold,
            "ball_tracking": "excluded (SoccerNet gt.txt has no ball channel)",
        },
        "rows": [{"row": "e2e_players", **row}],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 64}")
    print("  END-TO-END TRACKING BENCHMARK (YOLO → Norfair → CLEAR MOT)")
    print("=" * 64)
    print(
        f"\n  e2e  ({row['elapsed_s']}s, {row['frames_processed']} frames, "
        f"{row['n_pred_tracks']} tracks)"
    )
    print(f"    MOTA        {row['mota']:.4f}")
    print(f"    MOTP        {row['motp']:.2f} px")
    print(f"    IDF1        {row['idf1']:.4f}")
    print(f"    ID swaps    {row['id_switches']}")
    print(f"    Fragments   {row['fragments']}")
    print(f"    FP / FN     {row['false_positives']} / {row['false_negatives']}")
    print(f"\n  report -> {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
