"""Phase 2 pipeline test - tests CV + Analysis (formations, PPDA, xG/xT) + Reasoning + LLM."""

from __future__ import annotations

import asyncio
import os

os.environ["PYTHONIOENCODING"] = "utf-8"

import sys
import time
from pathlib import Path


async def main() -> int:
    print("=" * 60)
    print("KAWKAB AI - PHASE 2 FULL PIPELINE TEST")
    print("=" * 60)

    from kawkab.services import (
        AnalysisService,
        CVService,
        KnowledgeService,
        LLMConfig,
        LLMService,
        ReasoningService,
        StorageService,
    )

    video_path = Path("data/real_match.mp4")
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        print("  Run scripts/generate_synthetic_video.py or download a match")
        return 1

    print(f"\nVideo: {video_path} ({video_path.stat().st_size / 1024:.0f} KB)")

    print("\n[1/7] Initializing services...")
    storage = StorageService()
    await storage.initialize()
    cv = CVService(model_size="l", gpu_enabled=True)
    await cv.initialize()
    analysis = AnalysisService()
    knowledge = KnowledgeService()
    await knowledge.initialize()
    print(f"  KB: {knowledge.stats}")
    reasoning = ReasoningService(knowledge)
    await reasoning.initialize()
    llm = LLMService(
        LLMConfig(
            provider="ollama",
            ollama_model="ministral-3:14b",
            max_tokens=4000,
            num_gpu=99,
        )
    )

    print("\n[2/7] Saving match to database...")
    match_id = await storage.save_match(
        name="Phase 2 Test",
        video_path=str(video_path),
        home_team="Home",
        away_team="Away",
    )
    print(f"  Match ID: {match_id}")

    print("\n[3/7] CV pipeline (YOLOv11l + BoT-SORT)...")
    t0 = time.time()

    async def progress_cb(p: float, msg: str) -> None:
        if int(p * 100) % 10 == 0:
            print(f"  {p * 100:.0f}% - {msg}")

    track_data = await cv.process_video(video_path, progress_callback=progress_cb)
    cv_time = time.time() - t0
    print(
        f"  CV done: {len(track_data.frames)} frames, "
        f"{len(track_data.track_registry)} tracks in {cv_time:.1f}s"
    )

    await storage.update_match_analysis(
        match_id=match_id,
        duration=track_data.duration_seconds,
        fps=track_data.fps,
        total_frames=track_data.total_frames,
    )

    print("\n[4/7] Statistics + Formations + PPDA + xG/xT...")
    t0 = time.time()
    match_analysis = await analysis.analyze_match(track_data, match_id=match_id)
    analysis_time = time.time() - t0

    print(
        f"  Possession: Home {match_analysis.home_team.possession_pct:.1f}% / "
        f"Away {match_analysis.away_team.possession_pct:.1f}%"
    )
    print(
        f"  Formations: Home {match_analysis.formations.get('home', {}).get('formation', '?')} "
        f"({match_analysis.formations.get('home', {}).get('confidence', 0):.0%}) / "
        f"Away {match_analysis.formations.get('away', {}).get('formation', '?')} "
        f"({match_analysis.formations.get('away', {}).get('confidence', 0):.0%})"
    )
    print(f"  PPDA: {match_analysis.pressing_intensity:.1f}")
    print(f"  Players: {len(match_analysis.players)}, Events: {len(match_analysis.events)}")
    print(f"  Confidence: {match_analysis.confidence_overall:.1%}")

    print("\n[5/7] Tactical reasoning (Detective layer)...")
    t0 = time.time()
    diagnosis_report = await reasoning.diagnose_match(match_analysis, language="en")
    reasoning_time = time.time() - t0

    print(f"  Diagnoses: {len(diagnosis_report.diagnoses)}")
    for i, d in enumerate(diagnosis_report.diagnoses[:5], 1):
        print(f"    {i}. {d.rule_name} (confidence: {d.confidence:.0%})")
        if d.recommended_drills:
            drills = [knowledge.get_drill(dr) for dr in d.recommended_drills[:2]]
            drills = [dr.name for dr in drills if dr]
            if drills:
                print(f"       Recommended: {', '.join(drills)}")

    print("\n  Priority actions:")
    for action in diagnosis_report.priority_actions[:3]:
        print(f"    - {action}")

    print("\n[6/7] Freeing GPU before LLM...")
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

    print("\n[7/7] Generating AI coach report...")
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

    llm_summary = f"""Match: Phase 2 Test
Duration: {track_data.duration_seconds:.0f}s
Possession: Home {match_analysis.home_team.possession_pct:.1f}%, Away {match_analysis.away_team.possession_pct:.1f}%
Formation Home: {match_analysis.formations.get("home", {}).get("formation", "unknown")}
Formation Away: {match_analysis.formations.get("away", {}).get("formation", "unknown")}
PPDA: {match_analysis.pressing_intensity:.1f}
Events: {len(match_analysis.events)}, Players tracked: {len(match_analysis.players)}
Confidence: {match_analysis.confidence_overall:.1%}

TACTICAL DIAGNOSES (top 3):
"""
    for i, d in enumerate(diagnosis_report.diagnoses[:3], 1):
        llm_summary += f"\n{i}. {d.rule_name} (confidence {d.confidence:.0%})"
        if d.explanation:
            llm_summary += f"\n   {d.explanation[:200]}"

    llm_summary += f"\n\nTop players by distance: {top_summary}"

    try:
        report = await llm.generate_coach_report(llm_summary, language="en")
        llm_time = time.time() - t0
        print(f"  Report: {len(report)} chars in {llm_time:.1f}s")
        print("\n" + "=" * 60)
        print("COACH REPORT (with tactical diagnosis):")
        print("=" * 60)
        print(report[:2000])
        if len(report) > 2000:
            print(f"... ({len(report) - 2000} more chars)")
        print("=" * 60)
    except Exception as e:
        print(f"  [WARN] LLM failed: {e}")
        report = None

    print("\n" + "=" * 60)
    print("PHASE 2 RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Frames:           {len(track_data.frames)}")
    print(f"  Unique tracks:    {len(track_data.track_registry)}")
    print(f"  Players:          {len(match_analysis.players)}")
    print(f"  Events:           {len(match_analysis.events)}")
    print(
        f"  Formations:       {match_analysis.formations.get('home', {}).get('formation', '?')} "
        f"vs {match_analysis.formations.get('away', {}).get('formation', '?')}"
    )
    print(f"  PPDA:             {match_analysis.pressing_intensity:.1f}")
    print(f"  Diagnoses:        {len(diagnosis_report.diagnoses)}")
    print(f"  Confidence:       {match_analysis.confidence_overall:.1%}")
    print(f"  CV:               {cv_time:.1f}s")
    print(f"  Analysis:         {analysis_time:.1f}s")
    print(f"  Reasoning:        {reasoning_time:.1f}s")
    print(f"  LLM:              {llm_time:.1f}s")
    print(f"  Total:            {cv_time + analysis_time + reasoning_time + llm_time:.1f}s")
    print(f"  Report:           {len(report) if report else 0} chars")
    print("=" * 60)

    if diagnosis_report.diagnoses and report:
        print("PHASE 2 COMPLETE!")
    else:
        print("PHASE 2 PARTIAL (some features unavailable)")
    print("=" * 60)

    await storage.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
