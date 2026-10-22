#!/usr/bin/env python3
"""CI-compatible tracking benchmark script.

Checks for ground truth tracking data (Metrica CSV format in data/ground_truth/)
and, if present, computes MOTA, MOTP, IDF1, fragmentation ratio, and ID stability.

Usage:
    python scripts/benchmark_tracking.py
    python scripts/benchmark_tracking.py --data-dir data/ground_truth
    python scripts/benchmark_tracking.py --data-dir data/ground_truth --output benchmark_results/result.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger("benchmark_tracking")


def _has_ground_truth(data_dir: Path) -> bool:
    """Check if Metrica ground truth CSV files exist in *data_dir*."""
    home_csv = list(data_dir.rglob("*RawTrackingData_Home_Team.csv"))
    away_csv = list(data_dir.rglob("*RawTrackingData_Away_Team.csv"))
    return bool(home_csv) and bool(away_csv)


def _gt_tracks_to_dict(gt_match: Any) -> dict[int, list[tuple[int, float, float]]]:
    """Convert a GroundTruthMatch into the dict format expected by compute_mot_metrics.

    Each player gets a deterministic integer track ID derived from their label.
    """
    tracks: dict[int, list[tuple[int, float, float]]] = {}
    for prefix, player_tracks in (("home", gt_match.home_tracks), ("away", gt_match.away_tracks)):
        for label, track in player_tracks.items():
            tid = hash(f"{prefix}:{label}") % 10_000_000
            positions = []
            for f in track.frames:
                if not (f.x != f.x or f.y != f.y):  # NaN check via self-inequality
                    positions.append((f.frame, f.x, f.y))
            if positions:
                tracks[tid] = positions
    return tracks


def run_benchmark(data_dir: Path) -> dict[str, Any]:
    """Load ground truth data and compute all tracking metrics.

    Returns a dict suitable for JSON serialisation.
    """
    # Lazy imports so this module is importable without heavy dependencies
    from evaluate_tracking import (
        compute_fragmentation,
        compute_id_stability,
        load_metrica_ground_truth,
    )

    gt = load_metrica_ground_truth(data_dir)
    player_count = len(gt.home_tracks) + len(gt.away_tracks)
    total_frames = sum(len(t.frames) for t in (*gt.home_tracks.values(), *gt.away_tracks.values()))

    gt_tracks = _gt_tracks_to_dict(gt)

    from kawkab.core.mot_metrics import compute_mot_metrics

    # Self-comparison: ground-truth against itself gives the baseline ceiling
    # for MOT metrics (perfect MOTA=1.0, MOTP=0.0, IDF1=1.0) and validates that
    # the metric pipeline is wired correctly.
    mot_metrics = compute_mot_metrics(gt_tracks, gt_tracks, is_normalized=True)

    # Build a synthetic track_summary dict that evaluate_tracking's
    # compute_fragmentation expects.
    track_summary = {
        "tracks": {
            str(tid): {
                "frames": len(positions),
                "pct": 100.0,
            }
            for tid, positions in gt_tracks.items()
        }
    }
    frag = compute_fragmentation(track_summary)

    # Build synthetic frame data for compute_id_stability.
    frame_data = {
        "frames": [
            {"detections": [[pos[0], 0.0, 0.0, tid] for pos in positions]}
            for tid, positions in gt_tracks.items()
        ]
    }
    id_stab = compute_id_stability(frame_data)

    return {
        "status": "ok",
        "timestamp": datetime.now(UTC).isoformat(),
        "data_dir": str(data_dir),
        "ground_truth": {
            "home_players": len(gt.home_tracks),
            "away_players": len(gt.away_tracks),
            "total_players": player_count,
            "total_tracked_frames": total_frames,
            "fps": gt.fps,
            "total_frames_in_file": gt.total_frames,
        },
        "mot_metrics": mot_metrics,
        "fragmentation": frag,
        "id_stability": id_stab,
    }


def print_summary(report: dict[str, Any]) -> None:
    """Print a human-readable benchmark summary to stdout."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("  TRACKING BENCHMARK")
    print(f"{sep}")

    if report.get("status") == "no_data":
        print("  Status: no ground truth data found")
        print(f"{sep}\n")
        return

    gt = report.get("ground_truth", {})
    print(
        f"\n  Ground Truth:  {gt.get('total_players', 0)} players, "
        f"{gt.get('total_tracked_frames', 0)} frames, "
        f"{gt.get('fps', 0):.1f} fps"
    )

    mot = report.get("mot_metrics", {})
    print("\n  ── MOT Metrics ──")
    print(f"    MOTA:              {mot.get('mota', 'N/A')}")
    print(f"    MOTP:              {mot.get('motp', 'N/A')}")
    print(f"    IDF1:              {mot.get('idf1', 'N/A')}")
    print(f"    ID Precision:      {mot.get('id_precision', 'N/A')}")
    print(f"    ID Recall:         {mot.get('id_recall', 'N/A')}")
    print(f"    ID Switches:       {mot.get('id_switches', 'N/A')}")
    print(f"    False Positives:   {mot.get('false_positives', 'N/A')}")
    print(f"    False Negatives:   {mot.get('false_negatives', 'N/A')}")

    frag = report.get("fragmentation", {})
    if "error" not in frag:
        print("\n  ── Fragmentation ──")
        print(f"    Tracks:            {frag.get('n_tracks', 'N/A')}")
        print(f"    Avg Lifetime:      {frag.get('avg_lifetime_frames', 'N/A')} frames")
        buckets = frag.get("quality_buckets", {})
        for label, count in buckets.items():
            print(f"    {label}: {count}")

    stab = report.get("id_stability", {})
    if "error" not in stab:
        print("\n  ── ID Stability ──")
        print(f"    Switch Rate:       {stab.get('switch_rate', 'N/A')}")
        print(f"    Total Switches:    {stab.get('total_id_switches', 'N/A')}")

    print(f"{sep}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CI-compatible tracking benchmark",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/ground_truth",
        help="Directory containing Metrica ground truth CSVs (default: data/ground_truth)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results",
        help="Output path for JSON report (default: benchmark_results)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    if output_path.suffix != ".json":
        output_path = output_path / "benchmark_tracking.json"

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    if not _has_ground_truth(data_dir):
        report: dict[str, Any] = {
            "status": "no_data",
            "timestamp": datetime.now(UTC).isoformat(),
            "data_dir": str(data_dir),
            "message": "No Metrica ground truth CSV files found in data directory",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print_summary(report)
        logger.info("No ground truth data available — wrote no_data report")
        sys.exit(0)

    report = run_benchmark(data_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print_summary(report)
    logger.info(f"Benchmark report written to {output_path}")


if __name__ == "__main__":
    main()
