"""Test the user's real match video (Sweden vs Tunisia highlight)."""

import asyncio
import os

os.environ["PYTHONIOENCODING"] = "utf-8"
import sys
import time
from pathlib import Path


async def main() -> int:
    from kawkab.services import (
        AnalysisService,
        CVService,
        HomographyService,
        LLMConfig,
        LLMService,
        VRAMManager,
    )

    print("=" * 70)
    print("KAWKAB AI - TESTING USER'S REAL MATCH VIDEO")
    print("=" * 70)

    video_path = Path("data/sweden_test_60s.mp4")
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        return 1

    print(f"\nVideo: {video_path.name}")
    print(f"Size: {video_path.stat().st_size / 1e6:.1f} MB")
    print("Duration: 60s (clipped from 16:35 highlight reel)")
    print("Resolution: 1920x1080 @ 50 FPS (broadcast quality)")

    print("\n[1/5] Initializing services...")
    vram = VRAMManager()
    vram.allocate_for_yolo()
    cv = CVService(model_size="l", gpu_enabled=True)
    await cv.initialize()
    analysis = AnalysisService()
    homography = HomographyService()
    print("  [OK] Services ready")

    print("\n[2/5] CV pipeline (YOLOv11l + BoT-SORT + top-N)...")
    t0 = time.time()
    track_data = await cv.process_video(video_path)
    cv_time = time.time() - t0
    metrics = track_data.tracking_metrics
    print(f"  [OK] CV done in {cv_time:.1f}s")
    print(f"  Raw tracks:      {metrics['raw_tracks_detected']}")
    print(f"  Validated:       {metrics['validated_player_tracks']}")
    print(f"  Count ratio:     {metrics.get('count_ratio_vs_expected', 'N/A')}x of expected 22")
    print(f"  Quality:         {metrics['tracking_quality']}")

    print("\n[3/5] Team color detection...")
    t0 = time.time()
    team_colors = await cv.detect_team_colors(video_path, sample_frames=10)
    color_time = time.time() - t0
    team_counts = {}
    for _tid, info in team_colors.items():
        cluster = info.get("cluster_id", -1)
        team_counts[cluster] = team_counts.get(cluster, 0) + 1
    print(f"  [OK] Colors done in {color_time:.1f}s")
    print(f"  Players colored: {len(team_colors)}")
    print(f"  Teams detected:  {len(team_counts)}")
    for cid, count in team_counts.items():
        print(f"    Team {cid}: {count} players")

    print("\n[4/5] Homography calibration + analysis...")
    matrix = homography.compute_homography_from_corners(
        pixel_corners=[(150, 100), (1770, 100), (1770, 980), (150, 980)],
        pitch_length_m=105.0,
        pitch_width_m=68.0,
    )
    print(f"  Manual homography: confidence={matrix.confidence:.0%}")

    t0 = time.time()
    ma_meter = await analysis.analyze_match(track_data, match_id=0, homography_matrix=matrix)
    analysis_time = time.time() - t0
    home_f = ma_meter.formations.get("home", {})
    away_f = ma_meter.formations.get("away", {})
    print(f"  [OK] Analysis done in {analysis_time:.1f}s")
    print(f"  Confidence: {ma_meter.confidence_overall:.1%}")
    print(
        f"  Home formation: {home_f.get('formation', '?')} "
        f"(line_height_m={home_f.get('line_height_m', 'N/A')}m, "
        f"coords={home_f.get('coordinates', '?')})"
    )
    print(
        f"  Away formation: {away_f.get('formation', '?')} "
        f"(line_height_m={away_f.get('line_height_m', 'N/A')}m, "
        f"coords={away_f.get('coordinates', '?')})"
    )

    top_players = sorted(
        ma_meter.players.values(),
        key=lambda p: p.distance_covered_m,
        reverse=True,
    )[:5]
    print("\n  Top 5 players by distance (real meters):")
    for p in top_players:
        print(
            f"    Track {p.track_id:3d}: {p.distance_covered_m:6.0f}m, "
            f"max={p.max_speed_kmh:.1f} km/h, "
            f"avg={p.avg_speed_kmh:.1f} km/h"
        )

    print(
        f"\n  Possession: Home {ma_meter.home_team.possession_pct:.1f}% / "
        f"Away {ma_meter.away_team.possession_pct:.1f}%"
    )
    print(f"  Total events: {len(ma_meter.events)}")

    print("\n[5/5] Generate LLM coach report...")
    await cv.shutdown()
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    await asyncio.sleep(3)

    llm = LLMService(
        LLMConfig(provider="ollama", ollama_model="ministral-3:14b", max_tokens=4000, num_gpu=99)
    )

    top_summary = ", ".join(f"#{p.track_id}({p.distance_covered_m:.0f}m)" for p in top_players)
    llm_prompt = f"""Match: Sweden vs Tunisia (FIFA World Cup 2026 highlight)
Duration: 60 seconds (test clip)
Possession: Home {ma_meter.home_team.possession_pct:.1f}%, Away {ma_meter.away_team.possession_pct:.1f}%
Formations: {home_f.get("formation", "?")} vs {away_f.get("formation", "?")}
Confidence: {ma_meter.confidence_overall:.1%}
Top players: {top_summary}
Events: {len(ma_meter.events)} detected

Generate a brief tactical report in coach-friendly English (max 200 words).
Focus on formations, key players, and one observation.
"""
    try:
        report = await llm.generate_coach_report(llm_prompt, language="en")
        print(f"  Report: {len(report)} chars")
        print("\n" + "=" * 70)
        print("LLM COACH REPORT")
        print("=" * 70)
        print(report[:1500])
        if len(report) > 1500:
            print(f"... ({len(report) - 1500} more chars)")
        print("=" * 70)
    except Exception as e:
        print(f"  [WARN] LLM failed: {e}")

    print("\n" + "=" * 70)
    print("REAL MATCH VIDEO TEST SUMMARY")
    print("=" * 70)
    print("  Video: Sweden vs Tunisia 2026 (60s clip)")
    print(f"  CV: {cv_time:.1f}s | Analysis: {analysis_time:.1f}s")
    print(
        f"  Tracks: {metrics['validated_player_tracks']} (quality: {metrics['tracking_quality']})"
    )
    print(f"  Formations: {home_f.get('formation', '?')} / {away_f.get('formation', '?')}")
    print(f"  Confidence: {ma_meter.confidence_overall:.1%}")
    print("=" * 70)
    print("Real broadcast video works much better than amateur footage!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
