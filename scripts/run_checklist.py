"""Run the 10-question debugging checklist from docs/archive/CYCLE_1_VIDEO_PIPELINE.md."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO = PROJECT_ROOT / "data" / "real_match.mp4"
SWEDEN = PROJECT_ROOT / "data" / "sweden_test_60s.mp4"
FRANCE_SWEDEN = PROJECT_ROOT / "france_sweden_15min.mp4"


async def run_checklist():
    video = FRANCE_SWEDEN if FRANCE_SWEDEN.exists() else (VIDEO if VIDEO.exists() else SWEDEN)
    print("=== Cycle 1 — 10-Question Checklist ===")
    print("")
    print("Q1: Did the desktop app launch?")
    print("A1: Headless pipeline test running")
    print("")

    from kawkab.services.cv_service import CVService

    svc = CVService(model_size="m", gpu_enabled=True)

    print("Q2: Did you select the video and click Analyze?")
    print(f"A2: Video = {video.name} ({video.stat().st_size / 1024 / 1024:.1f} MB)")
    print("")

    print("Q3: Did YOLO load?")
    await svc.initialize()
    print(f"A3: YOLO loaded successfully: {svc._initialized}")
    print("")

    print("Q4: How many raw tracks were detected?")
    print("A4: Processing...")
    frame_skip = 3  # lower = better quality but slower
    match_data = await svc.process_video(
        video,
        frame_skip=frame_skip,
        enable_team_detection=True,
    )
    print(f"    frame_skip={frame_skip}, effective={match_data.fps / (frame_skip + 1):.1f} FPS")

    raw = match_data.tracking_metrics.get("raw_tracks_detected", "?")
    valid = match_data.tracking_metrics.get("validated_player_tracks", "?")
    quality = match_data.tracking_metrics.get("tracking_quality", "?")
    frag = match_data.tracking_metrics.get("fragmentation_rate", "?")
    team_count = len(match_data.player_teams)

    print(f"Q4: Raw tracking: {raw} unique tracks before filtering")
    print("")
    print("Q5: How many valid player tracks after filtering?")
    print(f"A5: {valid} validated player tracks (fragmentation={frag}x, quality={quality})")
    print("")

    print("Q6: Did team detection work?")
    print(f"A6: Teams assigned to {team_count} tracks (home/away labels)")
    td = match_data.tracking_metrics.get("team_detection", {})
    if td:
        print(
            f"    home_size={td.get('home_size')}, away_size={td.get('away_size')}, "
            f"ref_size={td.get('ref_size')}"
        )
    print("")

    print("Q7: Auto-calibration?")
    auto_h = match_data.tracking_metrics.get("auto_homography")
    print(
        f"A7: {'Auto-calibrated (homography matrix)' if auto_h else 'Not calibrated (pixel space)'}"
    )
    print("")

    print("Q8: Did the pipeline process without crash?")
    print(f"A8: {'Yes — no exception' if match_data else 'No'}")
    print("")

    print("Q9: What does the report look like?")
    print(
        f"A9: Match type={match_data.match_type}, FPS={match_data.fps:.1f}, "
        f"frames={match_data.total_frames}, duration={match_data.duration_seconds:.1f}s, "
        f"sampled_frames={len(match_data.frames)}"
    )
    print("")

    print("Q10: What was the overall tracking quality assessment?")
    print(f"A10: {quality}")
    print("")

    # Check for any issues
    print("=== Additional diagnostics ===")
    print(f"stitched_tracks: {match_data.tracking_metrics.get('stitched_tracks', 0)}")
    merge_map = match_data.tracking_metrics.get("stitch_merge_map", {})
    print(f"stitch_merge_map entries: {len(merge_map)}")
    if float(frag) > 3.0:
        print("WARNING: Fragmentation > 3.0x — high ID-switch rate")
    if valid == 0:
        print("WARNING: Zero valid tracks — pipeline may have failed silently")
    if "error" in match_data.tracking_metrics:
        print(f"ERROR: {match_data.tracking_metrics['error']}")

    await svc.shutdown()
    print("=== Checklist complete ===")


if __name__ == "__main__":
    asyncio.run(run_checklist())
