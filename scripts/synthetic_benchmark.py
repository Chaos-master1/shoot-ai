#!/usr/bin/env python3
"""Synthetic tracking benchmark: degrades Metrica ground truth, runs production tracker, reports MOTA/IDF1.

Usage:
    python scripts/synthetic_benchmark.py
    python scripts/synthetic_benchmark.py --noise 0.03 --drop 0.15 --fp-rate 0.05 --max-frames 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger("synthetic_benchmark")

# ── Config ──────────────────────────────────────────────────────────
PITCH_WIDTH_M = 105.0
PITCH_HEIGHT_M = 68.0
PX_PER_M = 10  # pixel scale: 1050 x 680
PLAYER_WIDTH_PX = 30
PLAYER_HEIGHT_PX = 30
CONFIDENCE = 0.85  # fixed confidence for synthetic detections
CLASS_ID = 0  # person class

RNG = np.random.default_rng(42)


def _load_metrica(data_dir: Path) -> dict[str, Any]:
    """Load Metrica CSV and return frame-by-frame player positions."""
    from evaluate_tracking import load_metrica_ground_truth

    gt = load_metrica_ground_truth(data_dir)
    return {
        "home": {
            label: [(f.frame, f.x, f.y) for f in track.frames]
            for label, track in gt.home_tracks.items()
        },
        "away": {
            label: [(f.frame, f.x, f.y) for f in track.frames]
            for label, track in gt.away_tracks.items()
        },
        "fps": gt.fps,
        "total_frames": gt.total_frames,
    }


def _build_detections(
    gt_data: dict[str, Any],
    max_frames: int,
    noise_std: float,
    drop_rate: float,
    fp_rate: float,
) -> dict[int, np.ndarray]:
    """Build frame-by-frame detections from Metrica GT with injected degradation.

    Returns {frame_num: np.ndarray[N, 6]} where each row is [x1, y1, x2, y2, conf, cls_id].
    Player positions are scaled to (1050, 680) pixel space.
    Each player has a fixed track_id derived from their label for ground truth comparison.
    """
    all_players = {}
    for team in ("home", "away"):
        for label, positions in gt_data[team].items():
            # Get first frame for this player
            if not positions:
                continue
            base_id = hash(f"{team}:{label}") % 10_000_000
            for frame, x, y in positions:
                if x != x or y != y:
                    continue  # skip NaN
                all_players.setdefault(int(frame), []).append(
                    {
                        "gt_id": base_id,
                        "x": x,
                        "y": y,
                        "team": team,
                        "label": label,
                    }
                )

    if not all_players:
        return {}

    frame_nums = sorted(all_players.keys())
    if max_frames:
        frame_nums = [f for f in frame_nums if f <= max_frames]

    if not frame_nums:
        return {}

    detections_by_frame: dict[int, list[np.ndarray]] = {}

    for fn in frame_nums:
        players = all_players.get(fn, [])
        dets = []

        for p in players:
            # Decide to drop this detection
            if RNG.random() < drop_rate:
                continue

            # Scale to pixel coords with noise
            sx = p["x"] * PITCH_WIDTH_M * PX_PER_M
            sy = (1.0 - p["y"]) * PITCH_HEIGHT_M * PX_PER_M  # flip Y

            sx += RNG.normal(0, noise_std * PITCH_WIDTH_M * PX_PER_M)
            sy += RNG.normal(0, noise_std * PITCH_HEIGHT_M * PX_PER_M)

            hw = PLAYER_WIDTH_PX / 2
            hh = PLAYER_HEIGHT_PX / 2

            dets.append(
                np.array(
                    [
                        max(0, sx - hw),
                        max(0, sy - hh),
                        sx + hw,
                        sy + hh,
                        CONFIDENCE,
                        CLASS_ID,
                    ]
                )
            )

            # Ground truth ID stored separately for comparison

        # Add false positive detections
        n_fp = RNG.poisson(fp_rate * max(1, len(players)))
        fp_scale_x = PITCH_WIDTH_M * PX_PER_M
        fp_scale_y = PITCH_HEIGHT_M * PX_PER_M
        for _ in range(n_fp):
            fx = RNG.uniform(0, fp_scale_x)
            fy = RNG.uniform(0, fp_scale_y)
            hw = PLAYER_WIDTH_PX / 2
            hh = PLAYER_HEIGHT_PX / 2
            dets.append(
                np.array(
                    [
                        max(0, fx - hw),
                        max(0, fy - hh),
                        fx + hw,
                        fy + hh,
                        RNG.uniform(0.3, 0.7),
                        CLASS_ID,
                    ]
                )
            )

        if dets:
            detections_by_frame[fn] = np.array(dets)

    return detections_by_frame


def _init_tracker(tracker_type: str = "bytetrack"):
    """Initialize a boxmot tracker instance."""
    if tracker_type == "bytetrack":
        from boxmot.trackers.bbox.bytetrack.bytetrack import ByteTrack

        return ByteTrack()
    elif tracker_type == "botsort":
        from boxmot.trackers.bbox.botsort.botsort import BotSort

        return BotSort()
    else:
        raise ValueError(f"Unknown tracker: {tracker_type}")


def run_synthetic_benchmark(
    gt_data: dict[str, Any],
    max_frames: int = 5000,
    noise_std: float = 0.02,
    drop_rate: float = 0.10,
    fp_rate: float = 0.05,
    tracker_type: str = "bytetrack",
) -> dict[str, Any]:
    """Run the synthetic benchmark and return results."""
    detections = _build_detections(gt_data, max_frames, noise_std, drop_rate, fp_rate)

    if not detections:
        return {"status": "error", "message": "No detections generated"}

    frame_nums = sorted(detections.keys())
    logger.info(
        f"Processing {len(frame_nums)} frames with {sum(len(d) for d in detections.values())} total detections"
    )

    # Build ground truth tracks dict for MOT computation
    gt_tracks: dict[int, list[tuple[int, float, float]]] = {}
    _ = {}
    for team in ("home", "away"):
        for label, positions in gt_data[team].items():
            if not positions:
                continue
            base_id = hash(f"{team}:{label}") % 10_000_000
            for frame, x, y in positions:
                if x != x or y != y:
                    continue
                gt_tracks.setdefault(base_id, []).append(
                    (
                        int(frame),
                        x * PITCH_WIDTH_M * PX_PER_M,
                        (1.0 - y) * PITCH_HEIGHT_M * PX_PER_M,
                    )
                )

    # Run tracker
    tracker = _init_tracker(tracker_type)
    # Create a dummy frame for tracker (needed for some trackers)
    dummy_frame = np.zeros(
        (int(PITCH_HEIGHT_M * PX_PER_M), int(PITCH_WIDTH_M * PX_PER_M), 3), dtype=np.uint8
    )

    # Build prediction tracks from tracker output
    pred_tracks: dict[int, list[tuple[int, float, float]]] = {}  # track_id -> [(frame, x, y)]

    for fn in frame_nums:
        dets = detections.get(fn)
        if dets is None or len(dets) == 0:
            # No detections this frame — still need to call update
            tracker.update(np.empty((0, 6)), dummy_frame)
            continue

        tracked = tracker.update(dets, dummy_frame)

        if tracked is not None and len(tracked) > 0:
            for t in tracked:
                x1, y1, x2, y2, tid, conf, cls_id, *_ = t
                tid = int(tid)
                cx = (float(x1) + float(x2)) / 2
                cy = (float(y1) + float(y2)) / 2
                pred_tracks.setdefault(tid, []).append((fn, cx, cy))

    # Compute MOT metrics
    from kawkab.core.mot_metrics import compute_mot_metrics

    # Filter GT tracks to only frames that exist in our detection set
    valid_frames = set(frame_nums)
    filtered_gt_tracks = {}
    for tid, positions in gt_tracks.items():
        filtered = [(f, x, y) for f, x, y in positions if f in valid_frames]
        if filtered:
            filtered_gt_tracks[tid] = filtered

    mot = compute_mot_metrics(
        pred_tracks, filtered_gt_tracks, fp_threshold=50.0, is_normalized=False
    )

    # Compute fragmentation from tracking output
    total_frames = max(1, len(frame_nums))
    track_summary = {
        "tracks": {
            str(tid): {"frames": len(positions), "pct": len(positions) / total_frames * 100}
            for tid, positions in pred_tracks.items()
        }
    }
    from evaluate_tracking import compute_fragmentation

    frag = compute_fragmentation(track_summary)

    n_gt_players = len(filtered_gt_tracks)

    return {
        "status": "ok",
        "timestamp": datetime.now(UTC).isoformat(),
        "config": {
            "max_frames": max_frames,
            "noise_std": noise_std,
            "drop_rate": drop_rate,
            "fp_rate": fp_rate,
            "tracker": tracker_type,
        },
        "ground_truth": {
            "total_players": n_gt_players,
            "total_frames": len(frame_nums),
            "total_detections": sum(len(d) for d in detections.values()),
            "tracked_frames": sum(len(p) for p in pred_tracks.values()),
        },
        "mot_metrics": mot,
        "fragmentation": frag,
    }


def print_report(report: dict[str, Any]):
    sep = "=" * 60
    print(f"\n{sep}")
    print("  SYNTHETIC TRACKING BENCHMARK")
    print(f"{sep}")

    if report.get("status") != "ok":
        print(f"  ERROR: {report.get('message', 'unknown')}")
        print(f"{sep}\n")
        return

    cfg = report.get("config", {})
    print("\n  Config:")
    print(f"    Tracker:       {cfg.get('tracker', 'N/A')}")
    print(f"    Noise std:     {cfg.get('noise_std', 'N/A')}")
    print(f"    Drop rate:     {cfg.get('drop_rate', 'N/A')}")
    print(f"    FP rate:       {cfg.get('fp_rate', 'N/A')}")
    print(f"    Max frames:    {cfg.get('max_frames', 'N/A')}")

    gt = report.get("ground_truth", {})
    print("\n  Ground Truth:")
    print(f"    Players:       {gt.get('total_players', 'N/A')}")
    print(f"    Frames:        {gt.get('total_frames', 'N/A')}")
    print(f"    Detections:    {gt.get('total_detections', 'N/A')}")
    print(f"    Tracked:       {gt.get('tracked_frames', 'N/A')}")

    mot = report.get("mot_metrics", {})
    print("\n  -- MOT Metrics --")
    mota = mot.get("mota", "N/A")
    print(
        f"    MOTA:          {mota:.4f}"
        if isinstance(mota, float)
        else f"    MOTA:          {mota}"
    )
    motp = mot.get("motp", "N/A")
    print(
        f"    MOTP:          {motp:.4f}"
        if isinstance(motp, float)
        else f"    MOTP:          {motp}"
    )
    idf1 = mot.get("idf1", "N/A")
    print(
        f"    IDF1:          {idf1:.4f}"
        if isinstance(idf1, float)
        else f"    IDF1:          {idf1}"
    )
    idp = mot.get("id_precision", "N/A")
    print(
        f"    ID Precision:  {idp:.4f}" if isinstance(idp, float) else f"    ID Precision:  {idp}"
    )
    idr = mot.get("id_recall", "N/A")
    print(
        f"    ID Recall:     {idr:.4f}" if isinstance(idr, float) else f"    ID Recall:     {idr}"
    )
    print(f"    ID Switches:   {mot.get('id_switches', 'N/A')}")
    print(f"    False Pos:     {mot.get('false_positives', 'N/A')}")
    print(f"    False Neg:     {mot.get('false_negatives', 'N/A')}")

    frag = report.get("fragmentation", {})
    if "error" not in frag:
        print("\n  -- Fragmentation --")
        print(f"    Tracks:        {frag.get('n_tracks', 'N/A')}")
        print(f"    Avg Lifetime:  {frag.get('avg_lifetime_frames', 'N/A')} frames")

    print(f"{sep}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/ground_truth")
    parser.add_argument("--output", default="benchmark_results/synthetic_benchmark.json")
    parser.add_argument(
        "--noise", type=float, default=0.02, help="Position noise std dev (fraction of pitch)"
    )
    parser.add_argument("--drop", type=float, default=0.10, help="Detection drop rate")
    parser.add_argument(
        "--fp-rate", type=float, default=0.05, help="False positives per player per frame"
    )
    parser.add_argument("--max-frames", type=int, default=5000, help="Max frames to process")
    parser.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "botsort"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    logger.info(f"Loading Metrica ground truth from {data_dir}")
    gt_data = _load_metrica(data_dir)
    logger.info(f"Loaded {len(gt_data['home'])} home + {len(gt_data['away'])} away players")

    report = run_synthetic_benchmark(
        gt_data,
        max_frames=args.max_frames,
        noise_std=args.noise,
        drop_rate=args.drop,
        fp_rate=args.fp_rate,
        tracker_type=args.tracker,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print_report(report)
    logger.info(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
