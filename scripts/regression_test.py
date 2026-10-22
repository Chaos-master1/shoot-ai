"""Regression test suite for tracking pipeline.

Compares current pipeline output against stored baseline metrics.
Fails if any metric regresses beyond tolerance.

Usage:
    python scripts/regression_test.py --video france_sweden_15min.mp4 [--baseline baseline.json] [--update-baseline]

Stored baseline format (JSON):
    {
        "tracks": 10,
        "fragmentation": 120.5,
        "ball_detections": 552,
        "id_stability": 0.344,
        "homography_success_pct": 85.0,
        "processing_time_s": 780.0
    }
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kawkab.services.cv_service import CVService

# Default tolerances (relative, e.g., 0.15 = 15% tolerance)
DEFAULT_TOLERANCES = {
    "tracks": 0.25,  # 25% — most variable
    "fragmentation": 0.30,  # 30%
    "ball_detections": 0.20,  # 20%
    "id_stability": 0.15,  # 15%
    "homography_success_pct": 0.10,
    "processing_time_s": 0.30,
}

BASELINE_FILE = "tracking_output/baseline.json"

# Known stable baseline for France vs Sweden 15-min clip
DEFAULT_BASELINE = {
    "tracks": 10,
    "fragmentation": 120.5,
    "ball_detections": 552,
    "id_stability": 0.344,
    "homography_success_pct": 85.0,
    "processing_time_s": 780.0,
}


async def run_regression(video_path: str, baseline_path: str | None = None, update: bool = False):
    video = Path(video_path)
    if not video.exists():
        print(f"ERROR: Video not found: {video}")
        sys.exit(1)

    print(f"Running regression on: {video}")
    print(f"  Size: {video.stat().st_size / 1024 / 1024:.1f} MB")

    svc = CVService(model_size="m", gpu_enabled=True)
    await svc.initialize()

    t0 = time.time()
    result = await svc.process_video(str(video), frame_skip=6)
    elapsed = time.time() - t0

    metrics = result.tracking_metrics
    track_registry = result.track_registry
    n_tracks = len(track_registry)

    # Count ball detections
    ball_count = sum(
        1 for fd in result.frames for d in (fd.detections or []) if d.class_name == "sports ball"
    )

    # Compute ID stability
    id_switches = 0
    total_det_frames = 0
    track_ids_by_frame: dict[int, set[int]] = {}
    for fd in result.frames:
        if fd.frame_number not in track_ids_by_frame:
            track_ids_by_frame[fd.frame_number] = set()
        for d in fd.detections or []:
            if d.class_name == "person" and d.track_id is not None:
                track_ids_by_frame[fd.frame_number].add(d.track_id)
                total_det_frames += 1

    prev_ids: set[int] = set()
    for fn in sorted(track_ids_by_frame.keys()):
        curr_ids = track_ids_by_frame[fn]
        new_ids = curr_ids - prev_ids
        lost_ids = prev_ids - curr_ids
        id_switches += min(len(new_ids), len(lost_ids))
        prev_ids = curr_ids

    id_stability = 1.0 - (id_switches / max(total_det_frames, 1))

    # Homography success
    auto_h = metrics.get("auto_homography")
    homography_success = 100.0 if auto_h is not None else 0.0

    results = {
        "tracks": n_tracks,
        "fragmentation": metrics.get("fragmentation_rate", 0),
        "ball_detections": ball_count,
        "id_stability": round(id_stability, 4),
        "homography_success_pct": homography_success,
        "processing_time_s": round(elapsed, 1),
    }

    print("\n=== RESULTS ===")
    for k, v in results.items():
        print(f"  {k}: {v}")

    # Load or create baseline
    baseline = DEFAULT_BASELINE.copy()
    if baseline_path and Path(baseline_path).exists():
        with open(baseline_path) as f:
            baseline.update(json.load(f))
        print(f"\nLoaded baseline from {baseline_path}")

    if update:
        save_path = baseline_path or BASELINE_FILE
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nBaseline updated: {save_path}")
        await svc.shutdown()
        return

    # Compare
    failures = 0
    print("\n=== COMPARISON vs BASELINE ===")
    for k, v in results.items():
        base_val = baseline.get(k)
        tol = DEFAULT_TOLERANCES.get(k, 0.25)
        if base_val is None:
            print(f"  {k}: {v} (no baseline)")
            continue
        rel_diff = abs(v - base_val) if base_val == 0 else abs(v - base_val) / base_val
        status = "PASS" if rel_diff <= tol else "FAIL"
        if status == "FAIL":
            failures += 1
        print(f"  {k}: {v} vs {base_val} (diff={rel_diff:.1%}, tol={tol:.0%}) [{status}]")

    print(f"\n{failures} failures out of {len(results)} metrics")
    await svc.shutdown()
    sys.exit(1 if failures > 0 else 0)


def main():
    parser = argparse.ArgumentParser(description="Regression test for tracking pipeline")
    parser.add_argument("--video", type=str, default=None, help="Input video (single)")
    parser.add_argument(
        "--pattern", type=str, default=None, help="Glob pattern (e.g. '*.mp4') for multi-video"
    )
    parser.add_argument(
        "--input-dir", type=str, default=".", help="Input directory for --pattern glob"
    )
    parser.add_argument("--baseline", type=str, default=None, help="Baseline JSON path")
    parser.add_argument("--update-baseline", action="store_true", help="Update stored baseline")
    args = parser.parse_args()

    if args.video:
        asyncio.run(
            run_regression(
                args.video,
                baseline_path=args.baseline,
                update=args.update_baseline,
            )
        )
    elif args.pattern:
        input_dir = Path(args.input_dir)
        videos = sorted(input_dir.glob(args.pattern))
        if not videos:
            print(f"No videos matching '{args.pattern}' in {input_dir}")
            sys.exit(1)
        print(
            f"Multi-match regression: {len(videos)} videos from {input_dir} (pattern: {args.pattern})\n"
        )
        all_results = []
        for i, video in enumerate(videos):
            print(f"[{i + 1}/{len(videos)}] {video.name}...")
            try:
                asyncio.run(
                    run_regression(
                        str(video), baseline_path=args.baseline, update=args.update_baseline
                    )
                )
                all_results.append((video.name, "PASS"))
            except SystemExit as e:
                all_results.append((video.name, "FAIL" if e.code != 0 else "PASS"))
            except Exception as e:
                all_results.append((video.name, f"ERROR: {e}"))
            print()

        print("=" * 60)
        print("AGGREGATE SUMMARY")
        print("=" * 60)
        print(f"{'Video':<40} {'Result':<10}")
        print("-" * 50)
        passes = 0
        failures = 0
        for name, result in all_results:
            print(f"{name:<40} {result:<10}")
            if result == "PASS":
                passes += 1
            elif result == "FAIL":
                failures += 1
        print("-" * 50)
        print(f"{'TOTAL':<40} {passes + failures:<10}")
        print(f"{'PASS':<40} {passes:<10}")
        print(f"{'FAIL':<40} {failures:<10}")
        sys.exit(1 if failures > 0 else 0)
    else:
        print(
            "Provide --video for single video or --pattern/--input-dir for multi-video regression"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
