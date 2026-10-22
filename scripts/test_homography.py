"""Test homography + team colors + improved tracking integrated."""

import asyncio
import os

os.environ["PYTHONIOENCODING"] = "utf-8"
from pathlib import Path


async def main() -> int:
    from kawkab.services import AnalysisService, CVService, HomographyService, VRAMManager

    print("=" * 60)
    print("KAWKAB AI - HOMOGRAPHY INTEGRATION TEST")
    print("=" * 60)

    video_path = Path("data/real_match.mp4")
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        return 1

    print(f"\nVideo: {video_path}")

    print("\n[1/4] Initializing services...")
    vram = VRAMManager()
    vram.allocate_for_yolo()
    cv = CVService(model_size="l", gpu_enabled=True)
    await cv.initialize()
    analysis = AnalysisService()
    homography = HomographyService()

    print("\n[2/4] CV pipeline with smart filters...")
    track_data = await cv.process_video(video_path)
    metrics = track_data.tracking_metrics
    print(f"  Validated tracks: {metrics['validated_player_tracks']}")
    print(f"  Fragmentation:    {metrics['fragmentation_rate']}x")
    print(f"  Quality:          {metrics['tracking_quality']}")

    print("\n[3/4] Team color detection...")
    team_colors = await cv.detect_team_colors(video_path, sample_frames=10)
    team_counts = {}
    for _tid, info in team_colors.items():
        cluster = info.get("cluster_id", -1)
        team_counts[cluster] = team_counts.get(cluster, 0) + 1
    print(f"  Players colored:  {len(team_colors)}")
    print(f"  Teams detected:    {len(team_counts)}")
    for cluster_id, count in team_counts.items():
        print(f"    Team {cluster_id}: {count} players")
    for tid in list(team_colors.keys())[:3]:
        info = team_colors[tid]
        print(f"    Track {tid}: color={info['color_hex']}, cluster={info['cluster_id']}")

    print("\n[4/4] Homography calibration + analysis comparison...")
    matrix_manual = homography.compute_homography_from_corners(
        pixel_corners=[(100, 80), (1180, 80), (1180, 640), (100, 640)],
        pitch_length_m=105.0,
        pitch_width_m=68.0,
    )
    print(
        f"  Manual homography: error={matrix_manual.error_px:.1f}px, "
        f"confidence={matrix_manual.confidence:.0%}"
    )

    matrix_est = homography.compute_homography_from_visible_markings(
        frame_width=1280, frame_height=720
    )
    print(
        f"  Estimated homography: error={matrix_est.error_px:.1f}px, "
        f"confidence={matrix_est.confidence:.0%}"
    )

    print("\n  Analysis WITHOUT homography (pixel-based):")
    analysis_pixel = await analysis.analyze_match(track_data, match_id=0)
    home_players_pixel = sorted(
        analysis_pixel.players.values(), key=lambda p: p.distance_covered_m, reverse=True
    )[:3]
    for p in home_players_pixel:
        print(f"    Track {p.track_id}: {p.distance_covered_m:.0f}m (pixels-based)")

    print("\n  Analysis WITH homography (meters-based):")
    analysis_meter = await analysis.analyze_match(
        track_data, match_id=0, homography_matrix=matrix_manual
    )
    home_players_meter = sorted(
        analysis_meter.players.values(), key=lambda p: p.distance_covered_m, reverse=True
    )[:3]
    for p in home_players_meter:
        print(f"    Track {p.track_id}: {p.distance_covered_m:.0f}m (real meters)")

    print("\n  Formation detection (with homography):")
    home_f = analysis_meter.formations.get("home", {})
    print(
        f"    Home: {home_f.get('formation', '?')} "
        f"(line_height_m={home_f.get('line_height_m', 'N/A')}m, "
        f"coords={home_f.get('coordinates', '?')})"
    )
    away_f = analysis_meter.formations.get("away", {})
    print(
        f"    Away: {away_f.get('formation', '?')} "
        f"(line_height_m={away_f.get('line_height_m', 'N/A')}m, "
        f"coords={away_f.get('coordinates', '?')})"
    )

    print("\n  xG / xT sample calculation (with homography):")
    for shot in analysis_meter.events[:3]:
        if shot.get("type") == "shot":
            xg = shot.get("metadata", {}).get("xg", "N/A")
            ts = shot.get("timestamp", 0)
            print(f"    Shot at t={ts:.1f}s: xG={xg}")

    await cv.shutdown()
    print("\n" + "=" * 60)
    print("HOMOGRAPHY INTEGRATION TEST COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
