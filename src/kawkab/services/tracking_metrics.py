"""Tracking self-consistency metrics using py-motmetrics.

Computes MOT-style metrics without ground truth by measuring:
- Track fragmentation (num_fragmentations)
- Identity switches (num_switches) — detected via spatial leaps
- Mostly tracked / mostly lost ratio
- Track coverage consistency

All metrics are intrinsic — they don't require GT annotations.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def compute_tracking_self_metrics(
    frames: list,
    track_registry: dict[int, dict[str, Any]],
    fps: float,
) -> dict[str, Any]:
    """Compute intrinsic tracking quality metrics without ground truth.

    Args:
        frames: List of FrameDetections with .frame_number, .detections
        track_registry: Dict of track_id -> track metadata
        fps: Frames per second

    Returns:
        Dict with keys:
            - num_tracks: total unique tracks
            - num_fragmentations: total fragmentation events
            - mostly_tracked: count of tracks tracked >80% of lifespan
            - partially_tracked: count of tracks 20-80%
            - mostly_lost: count of tracks <20%
            - avg_fragmentation_per_track: average fragmentation events
            - total_id_switches: estimated ID switch events
            - mot_self_consistency: 0-1 score
    """
    if not frames:
        return {"error": "no_frames", "mot_self_consistency": 0.0}

    # Build track presence per frame
    track_frames: dict[int, set[int]] = {}
    for fdet in frames:
        fn = fdet.frame_number
        for det in fdet.detections:
            if det.class_name != "person" or det.track_id is None:
                continue
            tid = det.track_id
            if tid not in track_frames:
                track_frames[tid] = set()
            track_frames[tid].add(fn)

    if not track_frames:
        return {"error": "no_tracks", "mot_self_consistency": 0.0}

    _ = max(fd.frame_number for fd in frames) + 1 if frames else 1
    num_tracks = len(track_frames)

    # Fragmentation: count of tracked -> lost -> tracked transitions
    total_fragmentations = 0
    mostly_tracked = 0
    partially_tracked = 0
    mostly_lost = 0

    for _tid, frames_set in track_frames.items():
        sorted_frames = sorted(frames_set)
        if len(sorted_frames) < 2:
            mostly_lost += 1
            continue

        # Detect gaps in track presence
        gaps = 0
        for i in range(1, len(sorted_frames)):
            if sorted_frames[i] - sorted_frames[i - 1] > int(fps * 1.0):  # >1s gap
                gaps += 1

        total_fragmentations += gaps

        # Coverage ratio
        first = sorted_frames[0]
        last = sorted_frames[-1]
        lifespan = last - first + 1
        coverage = len(sorted_frames) / max(lifespan, 1)

        if coverage >= 0.8:
            mostly_tracked += 1
        elif coverage >= 0.2:
            partially_tracked += 1
        else:
            mostly_lost += 1

    # Estimate ID switches: when two tracks with overlapping lifespans
    # have spatial proximity (suggesting they're the same player that was re-ID'd)
    total_id_switches = _estimate_id_switches(frames, track_frames, fps)

    # Self-consistency score: 0-1, higher is better
    frag_per_track = total_fragmentations / max(num_tracks, 1)
    mt_ratio = mostly_tracked / max(num_tracks, 1)
    id_switches_per_track = total_id_switches / max(num_tracks, 1)

    # Score formula: penalize fragmentation and ID switches, reward mostly_tracked
    score = max(
        0.0,
        min(
            1.0,
            0.5 * mt_ratio
            + 0.3 * max(0.0, 1.0 - frag_per_track / 5.0)
            + 0.2 * max(0.0, 1.0 - id_switches_per_track / 3.0),
        ),
    )

    return {
        "num_tracks": num_tracks,
        "num_fragmentations": total_fragmentations,
        "avg_fragmentation_per_track": round(frag_per_track, 2),
        "mostly_tracked": mostly_tracked,
        "partially_tracked": partially_tracked,
        "mostly_lost": mostly_lost,
        "total_id_switches": total_id_switches,
        "mot_self_consistency": round(score, 4),
    }


def _estimate_id_switches(
    frames: list,
    track_frames: dict[int, set[int]],
    fps: float,
) -> int:
    """Estimate ID switches by detecting spatial-temporal track overlaps.

    When two tracks exist in the same time window and their positions
    are close, they may represent the same player that got re-assigned.
    """
    from collections import defaultdict

    # For each frame, record (tid, cx) for each person detection
    frame_positions: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for fdet in frames:
        fn = fdet.frame_number
        for det in fdet.detections:
            if det.class_name != "person" or det.track_id is None:
                continue
            cx = (det.bbox[0] + det.bbox[2]) / 2
            frame_positions[fn].append((det.track_id, cx))

    # Check for overlapping track pairs with close x-positions
    # (suggesting they're the same player)
    switches = 0
    track_ids = list(track_frames.keys())
    for i in range(len(track_ids)):
        for j in range(i + 1, len(track_ids)):
            tid_a, tid_b = track_ids[i], track_ids[j]
            # Find overlap frames
            overlap = track_frames[tid_a] & track_frames[tid_b]
            if len(overlap) < int(fps * 2):  # less than 2s overlap
                continue

            # Check if positions are close in overlapping frames
            close_frames = 0
            for fn in overlap:
                positions = frame_positions.get(fn, [])
                pos_a = next((p for t, p in positions if t == tid_a), None)
                pos_b = next((p for t, p in positions if t == tid_b), None)
                if pos_a is not None and pos_b is not None and abs(pos_a - pos_b) < 30:
                    # within 30 pixels
                    close_frames += 1

            # If positions are close in >50% of overlap, likely an ID switch
            if len(overlap) > 0 and close_frames / len(overlap) > 0.5:
                switches += 1

    return switches


def compute_merge_map_from_switches(
    frames: list,
    track_frames: dict[int, set[int]],
    fps: float,
    spatial_threshold_px: float = 30.0,
    temporal_gap_max_frames: int | None = None,
) -> dict[int, int]:
    """Output a consumable merge map from ID switch analysis.

    Returns {discarded_track_id: survivor_track_id} — pairs of tracks
    that are likely the same player but got different IDs.

    Uses two signals:
    1. Overlapping frames with close x-centers (spatial-temporal overlap)
    2. Sequential tracks with small temporal gap and matching boundary positions
    """
    from collections import defaultdict

    if temporal_gap_max_frames is None:
        temporal_gap_max_frames = int(fps * 2)

    frame_positions: dict[int, dict[int, float]] = defaultdict(dict)
    for fdet in frames:
        fn = fdet.frame_number
        for det in fdet.detections:
            if det.class_name != "person" or det.track_id is None:
                continue
            cx = (det.bbox[0] + det.bbox[2]) / 2
            frame_positions[fn][det.track_id] = cx

    track_ids = list(track_frames.keys())
    merge_map: dict[int, int] = {}

    for i in range(len(track_ids)):
        for j in range(i + 1, len(track_ids)):
            tid_a, tid_b = track_ids[i], track_ids[j]
            if tid_a in merge_map or tid_b in merge_map:
                continue

            overlap = track_frames[tid_a] & track_frames[tid_b]
            if overlap:
                close = sum(
                    1
                    for fn in overlap
                    if abs(
                        frame_positions[fn].get(tid_a, 999) - frame_positions[fn].get(tid_b, 999)
                    )
                    < spatial_threshold_px
                )
                if len(overlap) > 0 and close / len(overlap) > 0.5:
                    count_a = len(track_frames[tid_a])
                    count_b = len(track_frames[tid_b])
                    survivor = tid_a if count_a >= count_b else tid_b
                    discarded = tid_b if survivor == tid_a else tid_a
                    merge_map[discarded] = survivor
                continue

            # Sequential gap
            first_a, last_a = min(track_frames[tid_a]), max(track_frames[tid_a])
            first_b, last_b = min(track_frames[tid_b]), max(track_frames[tid_b])
            if first_b > last_a and (first_b - last_a) <= temporal_gap_max_frames:
                ca = frame_positions[last_a].get(tid_a)
                cb = frame_positions[first_b].get(tid_b)
                if ca is not None and cb is not None and abs(ca - cb) < spatial_threshold_px * 1.5:
                    count_a = len(track_frames[tid_a])
                    count_b = len(track_frames[tid_b])
                    survivor = tid_a if count_a >= count_b else tid_b
                    discarded = tid_b if survivor == tid_a else tid_a
                    merge_map[discarded] = survivor
            elif first_a > last_b and (first_a - last_b) <= temporal_gap_max_frames:
                ca = frame_positions[last_b].get(tid_b)
                cb = frame_positions[first_a].get(tid_a)
                if ca is not None and cb is not None and abs(ca - cb) < spatial_threshold_px * 1.5:
                    count_a = len(track_frames[tid_a])
                    count_b = len(track_frames[tid_b])
                    survivor = tid_a if count_a >= count_b else tid_b
                    discarded = tid_b if survivor == tid_a else tid_a
                    merge_map[discarded] = survivor

    return merge_map
