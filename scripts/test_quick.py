"""Quick iteration test - 1 min clip. ~3 min total run time."""

import asyncio
import os
import sys
import time

os.environ["PYTHONIOENCODING"] = "utf-8"
from pathlib import Path


async def main() -> int:
    from kawkab.services import AnalysisService, CVService, HomographyService

    print("=" * 60)
    print("QUICK TEST — 1-min Sweden vs Tunisia")
    print("=" * 60)

    video_path = Path("data/sweden_1min.mp4").resolve()
    print(f"Video: {video_path.name} ({video_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Path: {video_path}")

    cv = CVService(model_size="l", gpu_enabled=True)
    await cv.initialize()
    analysis = AnalysisService()
    homography = HomographyService()
    matrix = homography.compute_homography_from_corners(
        pixel_corners=[(150, 100), (1770, 100), (1770, 980), (150, 980)],
        pitch_length_m=105.0,
        pitch_width_m=68.0,
    )

    print("\n[1] CV...")
    t0 = time.time()
    track_data = await cv.process_video(video_path, frame_skip=2)
    cv_time = time.time() - t0
    m = track_data.tracking_metrics
    print(f"  {cv_time:.1f}s ({60 / cv_time:.1f}x realtime)")
    print(f"  Tracks: {m['validated_player_tracks']} ({m['tracking_quality']})")
    td = m.get("team_detection", {})
    print(f"  Teams: {td.get('home_size', 0)} home / {td.get('away_size', 0)} away")

    print("\n[2] Analysis...")
    t0 = time.time()
    ma = await analysis.analyze_match(track_data, homography_matrix=matrix)
    a_time = time.time() - t0
    print(f"  {a_time:.1f}s, conf={ma.confidence_overall:.1%}")
    print(
        f"  Possession: H{ma.home_team.possession_pct:.1f}% / A{ma.away_team.possession_pct:.1f}%"
    )
    for t in ["home", "away"]:
        f = ma.formations.get(t, {})
        print(f"  {t}: {f.get('formation')} (line={f.get('line_height_m')}m)")

    print("\n[3] Top speeds (should be ~30-36 km/h, not 400+):")
    top = sorted(ma.players.values(), key=lambda p: p.distance_covered_m, reverse=True)[:5]
    for p in top:
        print(
            f"  Track {p.track_id:4d}: {p.distance_covered_m:5.0f}m, max={p.max_speed_kmh:5.1f} km/h"
        )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
