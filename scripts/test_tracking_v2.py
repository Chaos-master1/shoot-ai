"""Test improved CV pipeline with smart track filtering."""

import asyncio
import os

os.environ["PYTHONIOENCODING"] = "utf-8"
from pathlib import Path


async def main() -> int:
    from kawkab.services import CVService, VRAMManager

    print("Testing improved CV pipeline with smart filtering...")
    vram = VRAMManager()
    vram.allocate_for_yolo()
    cv = CVService(model_size="l", gpu_enabled=True)
    await cv.initialize()
    track_data = await cv.process_video(Path("data/real_match.mp4"))
    print()
    print("=" * 60)
    print("TRACKING QUALITY RESULTS")
    print("=" * 60)
    metrics = track_data.tracking_metrics
    raw = metrics["raw_tracks_detected"]
    valid = metrics["validated_player_tracks"]
    expected = metrics["expected_player_count"]
    count_ratio = metrics.get("count_ratio_vs_expected", 1.0)
    quality = metrics["tracking_quality"]
    print(f"  Raw tracks (before filtering): {raw}")
    print(f"  Validated player tracks:      {valid}")
    print(f"  Count ratio vs expected:      {count_ratio}x")
    print(f"  Tracking quality:              {quality}")
    print(f"  Expected (real match):         {expected} players")
    print()
    if valid > 0:
        print(f"  Detection vs Expected: {valid} / {expected} = {count_ratio:.2f}x")
    print()
    print("Top 10 longest-lived tracks:")
    sorted_tracks = sorted(
        track_data.track_registry.items(),
        key=lambda x: x[1].get("lifetime_pct", 0),
        reverse=True,
    )[:10]
    for tid, info in sorted_tracks:
        print(
            f"  Track {tid:3d}: {info['lifetime_pct']:5.1f}% lifetime, "
            f"{info['frames_tracked']:4d} frames, "
            f"conf={info['confidence_avg']:.2f}"
        )
    await cv.shutdown()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
