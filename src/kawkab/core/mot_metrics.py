"""CLEAR MOT metrics for tracking evaluation.

Implements:
  - MOTA (Multiple Object Tracking Accuracy)
  - MOTP (Multiple Object Tracking Precision)
  - IDF1 (Identification F1 Score)
  - Fragmentation count
  - ID switch count

Usage:
    metrics = compute_mot_metrics(
        pred_tracks={track_id: [(frame, x, y), ...]},
        gt_tracks={gt_id: [(frame, x, y), ...]},
        fp_threshold=20.0,  # pixel distance
    )
"""

from __future__ import annotations

import logging
from math import dist
from typing import Any

logger = logging.getLogger("mot_metrics")


def compute_mot_metrics(
    pred_tracks: dict[int, list[tuple[int, float, float]]],
    gt_tracks: dict[int, list[tuple[int, float, float]]],
    fp_threshold: float = 20.0,
    fps: float = 25.0,
    is_normalized: bool = False,
) -> dict[str, Any]:
    # Normalized [0,1] coords need a threshold in the same space
    if is_normalized and fp_threshold > 1.0:
        fp_threshold = 0.05  # 5% of pitch
    fn = 0
    fp = 0
    tp = 0
    id_switches = 0
    fragments = 0
    total_distance = 0.0
    total_matches = 0

    gt_frame_map: dict[int, list[tuple[int, float, float]]] = {}
    for gt_id, positions in gt_tracks.items():
        for frame, x, y in positions:
            if frame not in gt_frame_map:
                gt_frame_map[frame] = []
            gt_frame_map[frame].append((gt_id, x, y))

    pred_frame_map: dict[int, list[tuple[int, float, float]]] = {}
    for pred_id, positions in pred_tracks.items():
        for frame, x, y in positions:
            if frame not in pred_frame_map:
                pred_frame_map[frame] = []
            pred_frame_map[frame].append((pred_id, x, y))

    all_frames = sorted(set(gt_frame_map.keys()) | set(pred_frame_map.keys()))
    _: dict[int, int | None] = {}
    prev_assignment: dict[int, int | None] = {}
    prev_gt_matched: set[int] = set()
    gt_ever_matched_before: set[int] = set()

    for frame in all_frames:
        gt_in_frame = gt_frame_map.get(frame, [])
        pred_in_frame = pred_frame_map.get(frame, [])
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()

        # Match predictions to ground truth by nearest distance
        for pred_id, px, py in pred_in_frame:
            best_gt = None
            best_dist = fp_threshold
            for gt_id, gx, gy in gt_in_frame:
                if gt_id in matched_gt:
                    continue
                d = dist((px, py), (gx, gy))
                if d < best_dist:
                    best_dist = d
                    best_gt = gt_id
            if best_gt is not None:
                tp += 1
                matched_gt.add(best_gt)
                matched_pred.add(pred_id)
                total_distance += best_dist
                total_matches += 1
                # Check ID switch
                if best_gt in prev_assignment and prev_assignment[best_gt] != pred_id:
                    id_switches += 1
                prev_assignment[best_gt] = pred_id

        fn += len(gt_in_frame) - len(matched_gt)
        fp += len(pred_in_frame) - len(matched_pred)

        # Fragmentation: GT matched before, NOT matched previous frame, matched now
        for gt_id in matched_gt:
            if gt_id in gt_ever_matched_before and gt_id not in prev_gt_matched:
                fragments += 1
        gt_ever_matched_before.update(prev_gt_matched)
        prev_gt_matched = matched_gt.copy()

    total_gt = sum(len(v) for v in gt_tracks.values())
    total_pred = sum(len(v) for v in pred_tracks.values())

    mota = 1.0 - (fn + fp + id_switches) / max(total_gt, 1)
    motp = (total_distance / max(total_matches, 1)) if total_matches > 0 else 0.0

    # IDF1
    id_true_positives = tp
    id_gt_positives = total_gt
    id_pred_positives = total_pred
    id_precision = id_true_positives / max(id_pred_positives, 1)
    id_recall = id_true_positives / max(id_gt_positives, 1)
    idf1 = 2 * id_precision * id_recall / max(id_precision + id_recall, 1e-6)

    return {
        "mota": round(mota, 4),
        "motp": round(motp, 2),
        "idf1": round(idf1, 4),
        "id_precision": round(id_precision, 4),
        "id_recall": round(id_recall, 4),
        "false_positives": fp,
        "false_negatives": fn,
        "id_switches": id_switches,
        "fragments": fragments,
        "total_gt_detections": total_gt,
        "total_pred_detections": total_pred,
        "total_matches": total_matches,
    }
