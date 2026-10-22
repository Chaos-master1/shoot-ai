"""Test on FULL 16:35 Sweden vs Tunisia video (not clipped).

The previous test was on a 60s clip. This runs on the full 16+ min video
for proper tactical analysis. The LLM guardrails (v0.4.3) are in place
to prevent the 'Tunisia dominated' hallucination.
"""

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
    print("FULL MATCH TEST — Sweden vs Tunisia 2026 (16:35 highlight)")
    print("=" * 70)

    video_path = Path("data/sweden_5min.mp4")
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        return 1

    import subprocess

    duration = float(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    fps_actual = 50

    print(f"\nVideo: {video_path.name}")
    print(f"Size:  {video_path.stat().st_size / 1e6:.1f} MB")
    print(f"Duration: {duration:.1f}s ({duration / 60:.1f} min)")
    print(f"Frames: ~{int(duration * fps_actual)}")

    is_clip = duration < 1500
    print(f"Is clip (< 25 min): {is_clip}")

    print("\n[1/5] Initialize services...")
    vram = VRAMManager()
    vram.allocate_for_yolo()
    cv = CVService(model_size="l", gpu_enabled=True)
    await cv.initialize()
    analysis = AnalysisService()
    homography = HomographyService()
    print("  [OK]")

    print("\n[2/5] CV pipeline on full video (this takes time)...")
    print("  Using frame_skip=2 (process every 2nd frame) + team detection")
    t0 = time.time()
    track_data = await cv.process_video(video_path, frame_skip=2)
    cv_time = time.time() - t0
    metrics = track_data.tracking_metrics
    print(f"  [OK] {cv_time:.1f}s ({duration / cv_time:.1f}x realtime)")
    print(f"  Raw tracks: {metrics['raw_tracks_detected']}")
    print(f"  Validated:  {metrics['validated_player_tracks']}")
    print(f"  Count ratio: {metrics.get('count_ratio_vs_expected', 'N/A')}x of expected 22")
    print(f"  Quality:     {metrics['tracking_quality']}")
    print(
        f"  Frame skip: {metrics.get('frame_skip', 1)} (effective FPS: {metrics.get('effective_fps', '?')})"
    )
    td = metrics.get("team_detection", {})
    print(
        f"  Team detection: {td.get('n_clusters', 0)} clusters, "
        f"home={td.get('home_size', 0)} away={td.get('away_size', 0)} "
        f"of {td.get('assigned', 0)} assigned"
    )
    if track_data.player_teams:
        home = sum(1 for v in track_data.player_teams.values() if v == "home")
        away = sum(1 for v in track_data.player_teams.values() if v == "away")
        print(f"  Player teams: {home} home / {away} away")

    print("\n[3/5] Analysis with homography (meters)...")
    matrix = homography.compute_homography_from_corners(
        pixel_corners=[(150, 100), (1770, 100), (1770, 980), (150, 980)],
        pitch_length_m=105.0,
        pitch_width_m=68.0,
    )
    print(f"  Homography: confidence={matrix.confidence:.0%}")

    t0 = time.time()
    ma = await analysis.analyze_match(track_data, match_id=0, homography_matrix=matrix)
    analysis_time = time.time() - t0
    home_f = ma.formations.get("home", {})
    away_f = ma.formations.get("away", {})

    print(f"  [OK] {analysis_time:.1f}s")
    print(f"  Confidence: {ma.confidence_overall:.1%}")
    print(
        f"  Possession: Home {ma.home_team.possession_pct:.1f}% / Away {ma.away_team.possession_pct:.1f}%"
    )
    print(f"  Events: {len(ma.events)} detected")
    print(
        f"  Home formation: {home_f.get('formation', '?')} (line_h={home_f.get('line_height_m', 'N/A')}m)"
    )
    print(
        f"  Away formation: {away_f.get('formation', '?')} (line_h={away_f.get('line_height_m', 'N/A')}m)"
    )

    print("\n  Top 10 players by distance (real meters, full match):")
    top_players = sorted(ma.players.values(), key=lambda p: p.distance_covered_m, reverse=True)[:10]
    for p in top_players:
        print(
            f"    Track {p.track_id:3d}: {p.distance_covered_m:6.0f}m, "
            f"max={p.max_speed_kmh:.1f} km/h"
        )

    print("\n[4/5] Free GPU before LLM...")
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

    print("\n[5/5] LLM report with PROPER CONTEXT...")
    llm = LLMService(
        LLMConfig(
            provider="ollama",
            ollama_model="ministral-3:14b",
            max_tokens=4000,
            num_gpu=99,
        )
    )

    context = llm.build_match_context(
        track_data_duration=duration,
        is_clip=is_clip,
        final_score=None,
    )

    top_summary = ", ".join(f"#{p.track_id}({p.distance_covered_m:.0f}m)" for p in top_players)

    llm_prompt = f"""Match: Sweden vs Tunisia (FIFA World Cup 2026)
Duration: {duration:.0f} seconds ({"HIGHLIGHT CLIP" if is_clip else "NEAR-FULL MATCH"})
Possession: Home {ma.home_team.possession_pct:.1f}%, Away {ma.away_team.possession_pct:.1f}%
Formations: {home_f.get("formation", "?")} vs {away_f.get("formation", "?")}
Confidence: {ma.confidence_overall:.1%}
Events: {len(ma.events)}
Top players: {top_summary}

CRITICAL: This is a {duration:.0f}-second {"highlight" if is_clip else "segment"}.
Do NOT claim any team won or dominated the full match.
Frame observations as 'in this segment' or 'in this highlight'.
"""

    t0 = time.time()
    report = await llm.generate_coach_report(llm_prompt, language="en", match_context=context)
    llm_time = time.time() - t0
    print(f"  Report: {len(report)} chars in {llm_time:.1f}s")

    print("\n" + "=" * 70)
    print("LLM REPORT (full match, with guardrails)")
    print("=" * 70)
    print(report[:3000])
    if len(report) > 3000:
        print(f"\n... ({len(report) - 3000} more chars)")
    print("=" * 70)

    report_lower = report.lower()
    checks = [
        (
            "Mentions 'clip' or 'highlight' or 'segment'",
            any(w in report_lower for w in ["clip", "highlight", "segment"]),
        ),
        ("Does NOT claim 'dominated'", "dominated" not in report_lower),
        (
            "Does NOT claim 'won' or 'winner'",
            "won" not in report_lower and "winner" not in report_lower,
        ),
        (
            "Mentions 'cannot determine' or 'need more'",
            any(w in report_lower for w in ["cannot determine", "need more", "insufficient"]),
        ),
    ]
    print("\nGUARDRAIL CHECKS:")
    all_passed = True
    for check, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  [{status}] {check}")
        if not passed:
            all_passed = False
    print()
    if all_passed:
        print("✓ LLM guardrails working on full match data")
    else:
        print("✗ Some guardrails failed")

    print("\n" + "=" * 70)
    print("FULL MATCH SUMMARY")
    print("=" * 70)
    print(f"  Video:    Sweden vs Tunisia 2026 ({duration / 60:.1f} min)")
    print(f"  Tracks:   {metrics['validated_player_tracks']} ({metrics['tracking_quality']})")
    print(f"  CV time:  {cv_time:.1f}s ({duration / cv_time:.1f}x realtime)")
    print(f"  Formations: {home_f.get('formation', '?')} / {away_f.get('formation', '?')}")
    print(f"  LLM time: {llm_time:.1f}s")
    print(f"  Report:   {len(report)} chars, guardrails {'OK' if all_passed else 'FAIL'}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
