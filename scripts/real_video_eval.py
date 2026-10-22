#!/usr/bin/env python3
"""End-to-end tracking evaluation on real video using pseudo-ground-truth.

Pipeline:
  1. Run YOLO at conf=0.9 on video → pseudo-ground-truth (high-confidence detections)
  2. Run YOLO at conf=0.4 (pipeline threshold) → raw detections
  3. Feed raw detections through ByteTrack → tracked predictions
  4. Compare predictions vs pseudo-GT using MOT metrics

Usage:
    python scripts/real_video_eval.py --video France_Sweden_clip_2min.mp4 --output benchmark_results/real_video_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger("real_video_eval")

# ── GT threshold: only YOLO detections above this conf are "ground truth" ──
GT_CONF = 0.80
PIPELINE_CONF = 0.40
IOU_THRESH = 0.5


def _load_model():
    import torch
    from ultralytics import YOLO

    model = YOLO("yolo11m.pt")
    if torch.cuda.is_available():
        model.to("cuda")
        logger.info("YOLO loaded on GPU")
    else:
        logger.info("YOLO loaded on CPU")
    return model


def _run_detections(
    model: Any,
    video_path: str,
    conf: float,
    frame_skip: int = 6,
    max_frames: int = 0,
) -> dict[int, np.ndarray]:
    """Run YOLO on video and return {frame_num: np.ndarray[N, 6]} with [x1,y1,x2,y2,conf,cls]."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if max_frames > 0:
        total_frames = min(total_frames, max_frames * int(fps))

    detections: dict[int, list[np.ndarray]] = {}
    frame_number = 0
    det_frame = 0  # detection frame counter (after skip)

    logger.info(f"Processing {total_frames} frames, skip={frame_skip}, conf={conf}")

    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_number >= total_frames:
            break

        if frame_number % frame_skip == 0:
            results = model(frame, conf=conf, iou=0.5, classes=[0, 32], verbose=False)

            frame_dets = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    import torch

                    dets_np = (
                        torch.cat(
                            [
                                boxes.xyxy,
                                boxes.conf.unsqueeze(1),
                                boxes.cls.unsqueeze(1),
                            ],
                            dim=1,
                        )
                        .cpu()
                        .numpy()
                    )
                    for d in dets_np:
                        # Only keep person class (0) for GT
                        if int(d[5]) == 0:  # cls == person
                            frame_dets.append(d[:6])

            if frame_dets:
                detections[det_frame] = np.array(frame_dets)
            else:
                detections[det_frame] = np.empty((0, 6))

            det_frame += 1

        frame_number += 1

        if frame_number % 500 == 0:
            logger.info(f"  Frame {frame_number}/{total_frames} ({det_frame} det frames)")

    cap.release()
    elapsed = time.time() - t0
    logger.info(
        f"Done: {det_frame} detection frames in {elapsed:.1f}s ({det_frame / elapsed:.1f} fps)"
    )

    return detections


def _run_tracker(
    detections: dict[int, np.ndarray],
    dummy_frame_shape: tuple = (1080, 1920, 3),
    tracker_type: str = "bytetrack",
) -> dict[int, list[tuple[int, np.ndarray]]]:
    """Run ByteTrack on the detection stream. Returns {track_id: [(frame, bbox)]}."""
    from boxmot.trackers.bbox.bytetrack.bytetrack import ByteTrack

    tracker = ByteTrack()
    dummy_frame = np.zeros(dummy_frame_shape, dtype=np.uint8)
    pred_tracks: dict[int, list[tuple[int, np.ndarray]]] = {}

    frame_nums = sorted(detections.keys())
    logger.info(f"Running ByteTrack on {len(frame_nums)} frames...")

    for fn in frame_nums:
        dets = detections.get(fn, np.empty((0, 6)))
        tracked = tracker.update(dets, dummy_frame)

        if tracked is not None and len(tracked) > 0:
            for t in tracked:
                x1, y1, x2, y2, tid, conf, cls_id, *_ = t
                tid = int(tid)
                bbox = np.array([float(x1), float(y1), float(x2), float(y2)])
                pred_tracks.setdefault(tid, []).append((fn, bbox))

    return pred_tracks


def _build_gt_tracks(
    detections: dict[int, np.ndarray],
) -> dict[int, list[tuple[int, float, float]]]:
    """Build pseudo GT: each high-conf detection gets a unique per-frame ID (frame_number * 1000 + index).

    This is a limitation: each detection is unique across frames (no identity linking).
    We compare using detection-level metrics (no ID-based metrics).
    """
    gt_tracks: dict[int, list[tuple[int, float, float]]] = {}

    for fn in sorted(detections.keys()):
        dets = detections.get(fn, np.empty((0, 6)))
        for i, d in enumerate(dets):
            if int(d[5]) != 0:  # not a person
                continue
            gt_id = fn * 1000 + i  # unique per-frame ID
            cx = (float(d[0]) + float(d[2])) / 2
            cy = (float(d[1]) + float(d[3])) / 2
            gt_tracks.setdefault(gt_id, []).append((fn, cx, cy))

    return gt_tracks


def compute_detection_metrics(
    pred_tracks: dict[int, list[tuple[int, float, float]]],
    gt_tracks: dict[int, list[tuple[int, float, float]]],
    threshold: float = 50.0,
) -> dict[str, Any]:
    """Compute MOT metrics between predictions and pseudo-GT.

    Since pseudo-GT has unique IDs per frame, ID-switch metrics are not meaningful.
    """
    from kawkab.core.mot_metrics import compute_mot_metrics

    return compute_mot_metrics(pred_tracks, gt_tracks, fp_threshold=threshold, is_normalized=False)


def _match_by_iou_and_id(
    pred_dets: dict[int, np.ndarray],
    gt_dets: dict[int, np.ndarray],
    iou_thresh: float = 0.5,
) -> dict[str, Any]:
    """Compute frame-by-frame detection metrics using IoU matching.

    GT has no IDs, so each frame is an independent detection problem.
    """
    total_gt = 0
    total_pred = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    def _bbox_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    all_frames = sorted(set(pred_dets.keys()) | set(gt_dets.keys()))

    for fn in all_frames:
        gt = gt_dets.get(fn, np.empty((0, 6)))
        pr = pred_dets.get(fn, np.empty((0, 6)))
        total_gt += len(gt)
        total_pred += len(pr)

        matched_gt = set()
        matched_pr = set()

        for gi, g in enumerate(gt):
            if int(g[5]) != 0:
                matched_gt.add(gi)
                continue
            best_iou = iou_thresh
            best_pi = -1
            for pi, p in enumerate(pr):
                if pi in matched_pr:
                    continue
                iou = _bbox_iou(g[:4], p[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_pi = pi
            if best_pi >= 0:
                matched_gt.add(gi)
                matched_pr.add(best_pi)

        total_tp += len(matched_gt)
        total_fn += len(gt) - len(matched_gt)
        total_fp += len(pr) - len(matched_pr)

    precision = total_tp / max(total_pred, 1)
    recall = total_tp / max(total_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "total_gt": total_gt,
        "total_pred": total_pred,
    }


def _compute_iou_mota(
    pred_dets: dict[int, np.ndarray],
    gt_dets: dict[int, np.ndarray],
    iou_thresh: float = 0.5,
) -> float:
    """Compute a MOTA-like metric from frame-level detection matching.

    Since pseudo-GT has no ID consistency, ID switches cannot be measured.
    This is a detection-level MOTA: MOTA = 1 - (FN + FP) / total_GT
    """
    metrics = _match_by_iou_and_id(pred_dets, gt_dets, iou_thresh)
    total_errors = metrics["false_positives"] + metrics["false_negatives"]
    mota = 1.0 - total_errors / max(metrics["total_gt"], 1)
    return round(mota, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="France_Sweden_clip_2min.mp4")
    parser.add_argument("--output", default="benchmark_results/real_video_eval.json")
    parser.add_argument("--frame-skip", type=int, default=6)
    parser.add_argument("--max-frames", type=int, default=0, help="Max seconds of video to process")
    parser.add_argument("--gt-conf", type=float, default=GT_CONF)
    parser.add_argument("--pipeline-conf", type=float, default=PIPELINE_CONF)
    args = parser.parse_args()

    # ── 1. Load model ──
    model = _load_model()

    # ── 2. Run detections at pipeline threshold ──
    logger.info(f"Running YOLO at conf={args.pipeline_conf} (pipeline threshold)...")
    raw_dets = _run_detections(
        model, args.video, args.pipeline_conf, args.frame_skip, args.max_frames
    )
    logger.info(f"  Total raw detections: {sum(len(d) for d in raw_dets.values())}")

    # ── 3. Run detections at high threshold (pseudo-GT) ──
    logger.info(f"Running YOLO at conf={args.gt_conf} (pseudo ground truth)...")
    gt_dets = _run_detections(model, args.video, args.gt_conf, args.frame_skip, args.max_frames)
    gt_count = sum(len(d) for d in gt_dets.values())
    logger.info(f"  Total pseudo-GT detections: {gt_count}")

    if gt_count == 0:
        logger.error("No pseudo-GT detections found! GT conf threshold too high.")
        sys.exit(1)

    # ── 4. Run ByteTrack on raw detections ──
    pred_tracks = _run_tracker(raw_dets)
    logger.info(f"  Total predicted tracks: {len(pred_tracks)}")

    # ── 5. Build pseudo-GT tracks (unique ID per detection) ──
    _ = _build_gt_tracks(gt_dets)

    # ── 6. Compute metrics ──
    det_metrics = _match_by_iou_and_id(raw_dets, gt_dets)
    mota = _compute_iou_mota(raw_dets, gt_dets)

    tracked_dets: dict[int, np.ndarray] = {}
    for _tid, positions in pred_tracks.items():
        for fn, bbox in positions:
            if fn not in tracked_dets:
                tracked_dets[fn] = []
            tracked_dets[int(fn)].append(np.array([bbox[0], bbox[1], bbox[2], bbox[3], 1.0, 0]))

    for fn in tracked_dets:
        tracked_dets[fn] = np.array(tracked_dets[fn])

    tracked_metrics = _match_by_iou_and_id(tracked_dets, gt_dets)
    tracked_mota = _compute_iou_mota(tracked_dets, gt_dets)

    report = {
        "status": "ok",
        "timestamp": datetime.now(UTC).isoformat(),
        "config": {
            "video": args.video,
            "frame_skip": args.frame_skip,
            "gt_conf": args.gt_conf,
            "pipeline_conf": args.pipeline_conf,
            "iou_threshold": IOU_THRESH,
            "tracker": "bytetrack",
        },
        "data": {
            "total_frames_det": len(raw_dets),
            "raw_detections": sum(len(d) for d in raw_dets.values()),
            "pseudo_gt_detections": gt_count,
            "predicted_tracks": len(pred_tracks),
        },
        "raw_detection_metrics": {
            "mota_detection_only": mota,
            "precision": det_metrics["precision"],
            "recall": det_metrics["recall"],
            "f1": det_metrics["f1"],
            "tp": det_metrics["true_positives"],
            "fp": det_metrics["false_positives"],
            "fn": det_metrics["false_negatives"],
        },
        "tracked_detection_metrics": {
            "mota_tracked": tracked_mota,
            "precision": tracked_metrics["precision"],
            "recall": tracked_metrics["recall"],
            "f1": tracked_metrics["f1"],
            "tp": tracked_metrics["true_positives"],
            "fp": tracked_metrics["false_positives"],
            "fn": tracked_metrics["false_negatives"],
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    sep = "=" * 60
    print(f"\n{sep}")
    print("  REAL VIDEO BENCHMARK")
    print(f"{sep}")
    print(f"  Video:          {args.video}")
    print(f"  GT confidence:  {args.gt_conf}")
    print(f"  Pipeline conf:  {args.pipeline_conf}")
    print(f"  Detection frames: {len(raw_dets)}")
    print(f"  Raw detections: {report['data']['raw_detections']}")
    print(f"  Pseudo-GT:      {gt_count}")
    print(f"  Tracks:         {len(pred_tracks)}")
    print()
    print(f"  -- Raw Detection Metrics (conf={args.pipeline_conf} vs GT={args.gt_conf}) --")
    dm = report["raw_detection_metrics"]
    print(f"    MOTA (det only):  {dm['mota_detection_only']:.4f}")
    print(f"    Precision:        {dm['precision']:.4f}")
    print(f"    Recall:           {dm['recall']:.4f}")
    print(f"    F1:               {dm['f1']:.4f}")
    print(f"    TP/FP/FN:        {dm['tp']}/{dm['fp']}/{dm['fn']}")
    print()
    print("  -- Tracked Metrics (ByteTrack on pipeline detections vs GT) --")
    tm = report["tracked_detection_metrics"]
    print(f"    MOTA (tracked):   {tm['mota_tracked']:.4f}")
    print(f"    Precision:        {tm['precision']:.4f}")
    print(f"    Recall:           {tm['recall']:.4f}")
    print(f"    F1:               {tm['f1']:.4f}")
    print(f"    TP/FP/FN:        {tm['tp']}/{tm['fp']}/{tm['fn']}")
    print(f"{sep}\n")

    logger.info(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
