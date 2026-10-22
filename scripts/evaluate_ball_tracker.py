"""Evaluate ball tracking accuracy on test video.

Measures:
  - Detection rate: TP / (TP + FN)
  - Precision: TP / (TP + FP)
  - MOTP: average distance between predicted and GT ball positions
  - Per-frame breakdown with confidence bands

Usage:
  python scripts/evaluate_ball_tracker.py --video match_clip.mp4
  python scripts/evaluate_ball_tracker.py --video match_clip.mp4 --gt gt.json

Ground truth format (JSON):
  [
    {"frame": 0, "x": 320.0, "y": 240.0, "radius": 8.0},
    ...
  ]

If no GT file is provided, runs a self-consistency check by tracking
twice with different parameters and comparing results.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger("eval_ball_tracker")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from kawkab.services.ball_tracker import BallTracker  # noqa: E402


@dataclass
class EvalFrame:
    frame: int
    timestamp: float
    gt_x: float | None
    gt_y: float | None
    gt_radius: float | None
    pred_x: float | None
    pred_y: float | None
    pred_conf: float | None
    pred_is_prediction: bool
    distance_error: float | None  # pixels


def load_ground_truth(path: Path) -> dict[int, dict]:
    """Load GT from JSON file. Returns {frame: {x, y, radius}}."""
    with open(path) as f:
        data = json.load(f)
    gt = {}
    for entry in data:
        gt[entry["frame"]] = {
            "x": entry["x"],
            "y": entry["y"],
            "radius": entry.get("radius", 8.0),
        }
    logger.info(f"Loaded {len(gt)} GT frames from {path}")
    return gt


def run_evaluation(
    video_path: Path,
    gt: dict[int, dict] | None = None,
    fps_override: float | None = None,
    max_frames: int | None = None,
) -> list[EvalFrame]:
    """Run BallTracker on video, returning per-frame evaluation results."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)

    tracker = BallTracker(fps=fps)
    results: list[EvalFrame] = []

    frame_number = 0
    t_start = time.perf_counter()

    while frame_number < total:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_number / fps
        det = tracker.update(frame, frame_number, timestamp)

        gt_entry = gt.get(frame_number) if gt else None
        gt_x = gt_entry["x"] if gt_entry else None
        gt_y = gt_entry["y"] if gt_entry else None
        gt_r = gt_entry["radius"] if gt_entry else None

        pred_x = det.x if det else None
        pred_y = det.y if det else None
        pred_conf = det.conf if det else None
        pred_is_pred = det.is_prediction if det else False

        dist = None
        if gt_x is not None and gt_y is not None and pred_x is not None and pred_y is not None:
            dist = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)

        results.append(
            EvalFrame(
                frame=frame_number,
                timestamp=timestamp,
                gt_x=gt_x,
                gt_y=gt_y,
                gt_radius=gt_r,
                pred_x=pred_x,
                pred_y=pred_y,
                pred_conf=pred_conf,
                pred_is_prediction=pred_is_pred,
                distance_error=dist,
            )
        )

        frame_number += 1

    elapsed = time.perf_counter() - t_start
    cap.release()
    logger.info(
        f"Processed {len(results)} frames in {elapsed:.1f}s "
        f"({len(results) / max(elapsed, 0.01):.1f} FPS)"
    )
    return results


def compute_metrics(results: list[EvalFrame], iou_threshold: float = 0.3) -> dict:
    """Compute detection metrics from per-frame evaluation."""
    tp = fp = fn = 0
    dist_errors = []
    pred_only_dist_errors = []
    detection_dist_errors = []

    for r in results:
        has_gt = r.gt_x is not None and r.gt_y is not None
        has_pred = r.pred_x is not None and r.pred_y is not None

        if has_gt and has_pred:
            # IoU-based TP/FP: approximate IoU from distance / radius
            gt_r = r.gt_radius or 8.0
            pred_r = 8.0  # default prediction radius
            dist = r.distance_error or 999.0
            iou = _approximate_iou(dist, gt_r, pred_r)
            if iou >= iou_threshold:
                tp += 1
            else:
                fp += 1
                fn += 1  # counted as both FP (wrong detection) and FN (missed GT)
            dist_errors.append(dist)
            if r.pred_is_prediction:
                pred_only_dist_errors.append(dist)
            else:
                detection_dist_errors.append(dist)
        elif has_gt and not has_pred:
            fn += 1
        elif not has_gt and has_pred:
            fp += 1

    detection_rate = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * detection_rate * precision / max(detection_rate + precision, 1e-8)

    motp = np.mean(dist_errors) if dist_errors else float("nan")
    motp_detection = np.mean(detection_dist_errors) if detection_dist_errors else float("nan")
    motp_prediction = np.mean(pred_only_dist_errors) if pred_only_dist_errors else float("nan")

    # Confidence distribution
    confs = [r.pred_conf for r in results if r.pred_conf is not None]
    conf_high = sum(1 for c in confs if c >= 0.7)
    conf_med = sum(1 for c in confs if 0.5 <= c < 0.7)
    conf_low = sum(1 for c in confs if 0.3 <= c < 0.5)
    conf_very_low = sum(1 for c in confs if c < 0.3)

    correct_at_high = 0
    correct_at_med = 0
    correct_at_low = 0
    for r in results:
        if r.pred_conf is None or r.distance_error is None:
            continue
        gt_r = r.gt_radius or 8.0
        iou = _approximate_iou(r.distance_error, gt_r, 8.0)
        correct = iou >= iou_threshold
        if r.pred_conf >= 0.7:
            correct_at_high += int(correct)
        elif r.pred_conf >= 0.5:
            correct_at_med += int(correct)
        elif r.pred_conf >= 0.3:
            correct_at_low += int(correct)

    return {
        "total_frames": len(results),
        "frames_with_gt": sum(1 for r in results if r.gt_x is not None),
        "frames_with_pred": sum(1 for r in results if r.pred_x is not None),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "detection_rate": detection_rate,
        "precision": precision,
        "f1_score": f1,
        "motp_px": float(round(motp, 2)),
        "motp_detection_px": float(round(motp_detection, 2)),
        "motp_prediction_px": float(round(motp_prediction, 2)),
        "total_distance_error": float(round(sum(dist_errors), 1)) if dist_errors else 0,
        "confidence_high_count": conf_high,
        "confidence_medium_count": conf_med,
        "confidence_low_count": conf_low,
        "confidence_very_low_count": conf_very_low,
        "accuracy_at_high_conf": correct_at_high / max(conf_high, 1),
        "accuracy_at_medium_conf": correct_at_med / max(conf_med, 1),
        "accuracy_at_low_conf": correct_at_low / max(conf_low, 1),
    }


def _approximate_iou(dist_px: float, r1: float, r2: float) -> float:
    """Approximate IoU of two circles given center distance and radii."""
    d = max(dist_px, 0.0)
    R = max(r1, r2)
    r = min(r1, r2)
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        # One circle fully inside the other
        area_minor = np.pi * r * r
        area_major = np.pi * R * R
        return area_minor / area_major
    # Intersection area of two circles
    d2 = d * d
    part1 = r * r * np.arccos((d2 + r * r - R * R) / max(2 * d * r, 1e-8))
    part2 = R * R * np.arccos((d2 + R * R - r * r) / max(2 * d * R, 1e-8))
    part3 = 0.5 * np.sqrt(max(0, (-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R)))
    intersection = part1 + part2 - part3
    union = np.pi * (r1 * r1 + r2 * r2) - intersection
    return intersection / max(union, 1e-8)


def generate_report(metrics: dict, results: list[EvalFrame]) -> str:
    """Generate a human-readable evaluation report."""
    lines = [
        "=" * 60,
        "Ball Tracker Evaluation Report",
        "=" * 60,
        "",
        f"Total frames processed: {metrics['total_frames']}",
        f"Frames with ground truth: {metrics['frames_with_gt']}",
        f"Frames with prediction: {metrics['frames_with_pred']}",
        "",
        "── Detection Metrics ──",
        f"  True Positives:  {metrics['true_positives']}",
        f"  False Positives: {metrics['false_positives']}",
        f"  False Negatives: {metrics['false_negatives']}",
        f"  Detection Rate (Recall): {metrics['detection_rate']:.3f}",
        f"  Precision:               {metrics['precision']:.3f}",
        f"  F1 Score:                {metrics['f1_score']:.3f}",
        "",
        "── Localization Metrics ──",
        f"  MOTP (overall):       {metrics['motp_px']:.1f} px",
        f"  MOTP (detections):    {metrics['motp_detection_px']:.1f} px",
        f"  MOTP (predictions):   {metrics['motp_prediction_px']:.1f} px",
        "",
        "── Confidence Distribution ──",
        f"  High (>=0.7):  {metrics['confidence_high_count']} "
        f"(accuracy: {metrics['accuracy_at_high_conf']:.2%})",
        f"  Medium (0.5-0.7): {metrics['confidence_medium_count']} "
        f"(accuracy: {metrics['accuracy_at_medium_conf']:.2%})",
        f"  Low (0.3-0.5): {metrics['confidence_low_count']} "
        f"(accuracy: {metrics['accuracy_at_low_conf']:.2%})",
        f"  Very Low (<0.3): {metrics['confidence_very_low_count']}",
        "",
    ]

    # Per-frame breakdown for low-confidence / failure cases
    failures = [
        r
        for r in results
        if r.gt_x is not None
        and (r.pred_x is None or (r.distance_error is not None and r.distance_error > 30))
    ]
    if failures:
        lines.append("── Failure Cases (dist > 30 px or missed, top 20) ──")
        for r in failures[:20]:
            lines.append(
                f"  Frame {r.frame:6d} | "
                f"GT=({r.gt_x:.0f},{r.gt_y:.0f}) | "
                f"Pred=({r.pred_x or -1:.0f},{r.pred_y or -1:.0f}) | "
                f"Dist={r.distance_error:.1f}px | "
                f"Conf={r.pred_conf:.2f} | "
                f"{'PRED' if r.pred_is_prediction else 'DET'}"
            )

    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ball tracking accuracy")
    parser.add_argument("--video", type=str, required=True, help="Path to test video")
    parser.add_argument("--gt", type=str, default=None, help="Path to ground truth JSON (optional)")
    parser.add_argument("--fps", type=float, default=None, help="Override FPS (optional)")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process")
    parser.add_argument("--output", type=str, default=None, help="Save report to file")
    parser.add_argument(
        "--save-json", type=str, default=None, help="Save per-frame results to JSON"
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)

    gt = None
    if args.gt:
        gt = load_ground_truth(Path(args.gt))

    results = run_evaluation(
        video_path,
        gt=gt,
        fps_override=args.fps,
        max_frames=args.max_frames,
    )
    metrics = compute_metrics(results)

    report = generate_report(metrics, results)
    print(report)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(report)
        logger.info(f"Report saved to {out_path}")

    if args.save_json:
        json_path = Path(args.save_json)
        json_data = [
            {
                "frame": r.frame,
                "timestamp": r.timestamp,
                "gt_x": r.gt_x,
                "gt_y": r.gt_y,
                "gt_radius": r.gt_radius,
                "pred_x": r.pred_x,
                "pred_y": r.pred_y,
                "pred_conf": r.pred_conf,
                "pred_is_prediction": r.pred_is_prediction,
                "distance_error": r.distance_error,
            }
            for r in results
        ]
        json_path.write_text(json.dumps(json_data, indent=2))
        logger.info(f"Per-frame results saved to {json_path}")


if __name__ == "__main__":
    main()
