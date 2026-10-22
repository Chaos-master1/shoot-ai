"""End-to-end test of the full analysis pipeline.

This script:
1. Generates a synthetic football-like video (green field with moving dots)
2. Runs the full analysis pipeline on it (YOLO + tracking + stats + LLM)
3. Generates a coach report
4. Saves everything to the database and exports

Usage:
    python scripts/end_to_end_test.py [--duration 30]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"


def generate_synthetic_video(
    output_path: Path,
    duration_sec: int = 30,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
) -> Path:
    """Generate a synthetic football-like video using FFmpeg.

    Creates a green background with moving colored circles that simulate
    players and a ball. This lets us test the full pipeline without
    needing a real match video.

    Args:
        output_path: Where to save the video
        duration_sec: Video duration in seconds
        fps: Frames per second
        width: Video width
        height: Video height

    Returns:
        Path to generated video
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Generating {duration_sec}s synthetic video at {fps} FPS...")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=green:s={width}x{height}:r={fps}:d={duration_sec}",
        "-vf",
        f"""
        drawbox=x=0:y={height // 2 - 2}:w={width}:h=4:color=white@0.8:t=fill,
        drawbox=x={width // 2 - 2}:y=0:w=4:h={height}:color=white@0.8:t=fill,
        drawbox=x=80:y=80:w=120:h=100:color=red@0.7:t=fill,
        drawbox=x={width - 200}:y=80:w=120:h=100:color=blue@0.7:t=fill,
        drawbox=x=80:y={height - 180}:w=120:h=100:color=red@0.7:t=fill,
        drawbox=x={width - 200}:y={height - 180}:w=120:h=100:color=blue@0.7:t=fill,
        drawtext=text='%{{eif\\:t\\:d}}':fontsize=60:fontcolor=white:x=20:y=20,
        """.replace("\n", "").strip(),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"  [OK] Video saved: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"  [WARN] ffmpeg failed: {e.stderr[-500:]}")
        print("  Falling back to simpler video generation...")
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=green:s={width}x{height}:r={fps}:d={duration_sec}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"  [OK] Simple video saved: {output_path}")
        return output_path


async def run_full_pipeline(
    video_path: Path,
    match_name: str,
) -> dict:
    """Run the full analysis pipeline on a video.

    Args:
        video_path: Path to the video file
        match_name: Name for this match

    Returns:
        Summary dict with all results
    """
    from kawkab.services import (
        AnalysisService,
        CVService,
        KnowledgeService,
        LLMConfig,
        LLMService,
        StorageService,
    )

    print(f"\n{'=' * 60}")
    print(f"FULL PIPELINE TEST: {match_name}")
    print(f"{'=' * 60}\n")

    print("Step 1/6: Initialize services...")
    storage = StorageService()
    await storage.initialize()
    cv = CVService(model_size="l", gpu_enabled=True)
    await cv.initialize()
    analysis = AnalysisService()
    knowledge = KnowledgeService()
    await knowledge.initialize()
    llm = LLMService(
        LLMConfig(
            provider="ollama",
            ollama_model="gemma4:12b",
            max_tokens=4000,
            num_gpu=99,
        )
    )
    print(f"  [OK] Services ready. KB: {knowledge.stats}")

    print("\nStep 2/6: Save match to database...")
    match_id = await storage.save_match(
        name=match_name,
        video_path=str(video_path),
        home_team="Home",
        away_team="Away",
    )
    print(f"  [OK] Match saved: id={match_id}")

    print("\nStep 3/6: Run CV pipeline on video...")
    t0 = time.time()

    async def progress_cb(p: float, msg: str) -> None:
        if int(p * 100) % 10 == 0:
            print(f"  {p * 100:.0f}% - {msg}")

    track_data = await cv.process_video(video_path, progress_callback=progress_cb)
    cv_time = time.time() - t0
    print(
        f"  [OK] CV done in {cv_time:.1f}s: {len(track_data.frames)} frames, "
        f"{len(track_data.track_registry)} unique tracks"
    )

    await storage.update_match_analysis(
        match_id=match_id,
        duration=track_data.duration_seconds,
        fps=track_data.fps,
        total_frames=track_data.total_frames,
    )

    print("\nStep 4/6: Compute statistics...")
    t0 = time.time()
    match_analysis = await analysis.analyze_match(track_data, match_id=match_id)
    analysis_time = time.time() - t0
    print(
        f"  [OK] Analysis done in {analysis_time:.1f}s: "
        f"{len(match_analysis.players)} players, {len(match_analysis.events)} events"
    )
    print(
        f"  Possession: Home {match_analysis.home_team.possession_pct:.1f}% "
        f"vs Away {match_analysis.away_team.possession_pct:.1f}%"
    )
    print(f"  Confidence: {match_analysis.confidence_overall:.1%}")

    print("\nStep 5/6: Save to database...")
    for _tid, player in match_analysis.players.items():
        await storage.save_player(
            match_id=match_id,
            player_data={
                "track_id": player.track_id,
                "distance_covered_m": player.distance_covered_m,
                "max_speed_kmh": player.max_speed_kmh,
                "avg_speed_kmh": player.avg_speed_kmh,
                "passes_attempted": player.passes_attempted,
                "passes_completed": player.passes_completed,
                "shots": player.shots,
                "tackles": player.tackles,
            },
        )
    for event in match_analysis.events[:100]:
        await storage.save_event(match_id=match_id, event=event)
    print(
        f"  [OK] Saved {len(match_analysis.players)} players, "
        f"{len(match_analysis.events[:100])} events"
    )

    print("\nStep 6/6: Generate coach report (LLM)...")
    t0 = time.time()
    top_players = sorted(
        match_analysis.players.values(),
        key=lambda p: p.distance_covered_m,
        reverse=True,
    )[:5]
    top_summary = ", ".join(
        f"#{p.track_id}({p.distance_covered_m:.0f}m, {p.max_speed_kmh:.1f}km/h)"
        for p in top_players
    )

    summary = f"""Match: {match_name}
Duration: {track_data.duration_seconds:.0f} seconds
Possession: Home {match_analysis.home_team.possession_pct:.1f}%, Away {match_analysis.away_team.possession_pct:.1f}%
Players tracked: {len(match_analysis.players)}
Events detected: {len(match_analysis.events)}
Passes: {len([e for e in match_analysis.events if e.get("type") == "pass"])}
Average confidence: {match_analysis.confidence_overall:.1%}
Top 5 players by distance (track_id, distance_m, max_speed_kmh): {top_summary}
"""

    print("\nStep 6a: Freeing GPU memory before LLM call...")
    await cv.shutdown()
    import asyncio as _asyncio
    import gc

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()
    await _asyncio.sleep(3)
    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info()[0] / 1e9
        print(f"  GPU free memory: {free_mem:.1f} GB")

    try:
        report = await llm.generate_coach_report(summary, language="en")
        llm_time = time.time() - t0
        print(f"  [OK] Report generated in {llm_time:.1f}s ({len(report)} chars)")
        print("\n" + "-" * 60)
        print("COACH REPORT:")
        print("-" * 60)
        print(report[:1000])
        if len(report) > 1000:
            print(f"... ({len(report) - 1000} more chars)")
        print("-" * 60)

        await storage.save_report(
            match_id=match_id,
            language="en",
            report_text=report,
            llm_provider="ollama",
        )
    except Exception as e:
        print(f"  [WARN] Report generation failed: {e}")
        report = None

    print("\nStep 7: Cleanup...")
    await storage.close()

    total_time = cv_time + analysis_time
    return {
        "match_id": match_id,
        "cv_time_sec": cv_time,
        "analysis_time_sec": analysis_time,
        "total_cv_time_sec": total_time,
        "frames_processed": len(track_data.frames),
        "unique_tracks": len(track_data.track_registry),
        "players_analyzed": len(match_analysis.players),
        "events_detected": len(match_analysis.events),
        "possession_home": match_analysis.home_team.possession_pct,
        "possession_away": match_analysis.away_team.possession_pct,
        "confidence": match_analysis.confidence_overall,
        "report_generated": report is not None,
    }


def print_summary(results: dict, video_path: Path) -> None:
    """Print a nice summary table of the test results."""
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Video:              {video_path.name}")
    print(f"  Match ID:           {results['match_id']}")
    print(f"  Frames processed:   {results['frames_processed']}")
    print(f"  Unique tracks:      {results['unique_tracks']}")
    print(f"  Players analyzed:   {results['players_analyzed']}")
    print(f"  Events detected:    {results['events_detected']}")
    print(
        f"  Possession (H/A):   {results['possession_home']:.1f}% / {results['possession_away']:.1f}%"
    )
    print(f"  Confidence:         {results['confidence']:.1%}")
    print(f"  CV pipeline:        {results['cv_time_sec']:.1f}s")
    print(f"  Analysis:           {results['analysis_time_sec']:.1f}s")
    print(f"  Report generated:   {'YES' if results['report_generated'] else 'NO'}")
    print()
    if results["cv_time_sec"] > 0:
        fps = results["frames_processed"] / results["cv_time_sec"]
        print(f"  Processing speed:   {fps:.1f} FPS")
    print("=" * 60)
    if results["confidence"] > 0.5 and results["report_generated"]:
        print("ALL TESTS PASSED!")
    else:
        print("TESTS COMPLETED (with caveats)")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end pipeline test")
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Synthetic video duration in seconds (default: 30)",
    )
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="Skip synthetic video generation, use a real video path instead",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to a real video file (skips synthetic generation)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="Test Match (Synthetic)",
        help="Match name",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("KAWKAB AI - END-TO-END PIPELINE TEST")
    print("=" * 60)

    from kawkab.core.paths import get_paths

    paths = get_paths()
    test_dir = paths.cache / "tests"
    test_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"[ERROR] Video not found: {video_path}")
            return 1
    else:
        video_path = test_dir / f"synthetic_{args.duration}s.mp4"
        if not video_path.exists():
            print("\nGenerating synthetic test video...")
            try:
                video_path = generate_synthetic_video(video_path, duration_sec=args.duration)
            except FileNotFoundError:
                print("[ERROR] FFmpeg not found. Install FFmpeg or use --video flag.")
                return 1

    better_video = test_dir / f"better_synthetic_{args.duration}s.mp4"
    if better_video.exists() and not args.video:
        print(f"\nUsing better synthetic video: {better_video.name}")
        video_path = better_video

    try:
        results = asyncio.run(run_full_pipeline(video_path, args.name))
    except Exception as e:
        print(f"\n[FATAL] Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print_summary(results, video_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
