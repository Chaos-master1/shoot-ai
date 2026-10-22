"""Re-test on user's video with PROPER LLM guardrails.

This test demonstrates the fix for the LLM hallucination issue.
The previous test said "Tunisia dominated" which was WRONG (Tunisia lost badly).

The fix: pass is_clip=True so the LLM knows this is a short clip, not a full match.
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
    print("RE-TEST WITH LLM GUARDRAILS")
    print("Sweden vs Tunisia 2026 — 60s clip")
    print("=" * 70)

    video_path = Path("data/sweden_test_60s.mp4")
    if not video_path.exists():
        print("[ERROR] Video not found")
        return 1

    print("\n[1/4] Initializing services...")
    vram = VRAMManager()
    vram.allocate_for_yolo()
    cv = CVService(model_size="l", gpu_enabled=True)
    await cv.initialize()
    analysis = AnalysisService()
    homography = HomographyService()

    print("\n[2/4] CV pipeline...")
    t0 = time.time()
    track_data = await cv.process_video(video_path)
    cv_time = time.time() - t0
    metrics = track_data.tracking_metrics
    print(
        f"  Done in {cv_time:.1f}s: {metrics['validated_player_tracks']} tracks, "
        f"quality={metrics['tracking_quality']}"
    )

    print("\n[3/4] Analysis with homography...")
    matrix = homography.compute_homography_from_corners(
        pixel_corners=[(150, 100), (1770, 100), (1770, 980), (150, 980)],
        pitch_length_m=105.0,
        pitch_width_m=68.0,
    )
    ma = await analysis.analyze_match(track_data, match_id=0, homography_matrix=matrix)

    top_players = sorted(ma.players.values(), key=lambda p: p.distance_covered_m, reverse=True)[:5]
    top_summary = ", ".join(f"#{p.track_id}({p.distance_covered_m:.0f}m)" for p in top_players)

    print("\n[4/4] Generate report with PROPER CONTEXT...")
    await cv.shutdown()
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    await asyncio.sleep(3)

    llm = LLMService(
        LLMConfig(
            provider="ollama",
            ollama_model="ministral-3:14b",
            max_tokens=4000,
            num_gpu=99,
        )
    )

    actual_duration = track_data.duration_seconds
    is_clip = actual_duration < 1500
    print(f"  Video duration: {actual_duration:.0f}s")
    print(f"  Is clip (vs full match): {is_clip}")

    context = llm.build_match_context(
        track_data_duration=actual_duration,
        is_clip=is_clip,
        final_score=None,
        goals=[],
        cards=[],
    )
    print(f"  Context: {context}")

    llm_prompt = f"""Match: Sweden vs Tunisia (FIFA World Cup 2026 highlight)
Duration: {actual_duration:.0f} seconds (CLIP, not full match)
Possession: Home {ma.home_team.possession_pct:.1f}%, Away {ma.away_team.possession_pct:.1f}%
Formations: {ma.formations.get("home", {}).get("formation", "?")} vs {ma.formations.get("away", {}).get("formation", "?")}
Confidence: {ma.confidence_overall:.1%}
Top players (60s clip): {top_summary}
Events: {len(ma.events)} detected

CRITICAL: This is a 60-second highlight CLIP, not a full match.
Match outcomes can completely change over 90 minutes.
Do NOT claim any team won or dominated based on this data.
"""

    report = await llm.generate_coach_report(llm_prompt, language="en", match_context=context)

    print("\n" + "=" * 70)
    print("LLM REPORT (with guardrails)")
    print("=" * 70)
    print(report[:2000])
    if len(report) > 2000:
        print(f"... ({len(report) - 2000} more chars)")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("VERIFICATION CHECKS")
    print("=" * 70)
    report_lower = report.lower()
    checks = [
        (
            "Mentions 'clip' or '60-second'",
            any(w in report_lower for w in ["clip", "60-second", "60 second", "60s"]),
        ),
        (
            "Does NOT claim 'dominated' or 'won'",
            "dominated" not in report_lower and "won" not in report_lower,
        ),
        ("Does NOT claim 'controlled'", "controlled" not in report_lower),
        (
            "Mentions 'need more data' or 'cannot determine'",
            any(
                w in report_lower
                for w in ["need more", "cannot determine", "incomplete", "limited"]
            ),
        ),
    ]
    for check, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  [{status}] {check}")

    all_passed = all(p for _, p in checks)
    print()
    if all_passed:
        print("ALL GUARDRAILS WORKING — LLM no longer hallucinates outcomes!")
    else:
        print("Some guardrails failed — LLM still making unsupported claims")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
