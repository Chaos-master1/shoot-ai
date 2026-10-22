"""Test tracking with max_keep_top_n filter."""

import asyncio
import os

os.environ["PYTHONIOENCODING"] = "utf-8"
from pathlib import Path


async def main() -> int:
    from kawkab.services.cv_service import CVService

    cv = CVService(model_size="l", gpu_enabled=True, max_keep_top_n=28)
    await cv.initialize()
    track_data = await cv.process_video(Path("data/real_match.mp4"))
    metrics = track_data.tracking_metrics
    print("With max_keep_top_n=28:")
    print(f"  Raw: {metrics['raw_tracks_detected']}")
    print(f"  Validated: {metrics['validated_player_tracks']}")
    print(f"  Fragmentation: {metrics['fragmentation_rate']}x")
    print(f"  Quality: {metrics['tracking_quality']}")
    await cv.shutdown()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
