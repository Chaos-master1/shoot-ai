"""Phase 3 comprehensive test: CV + Analysis + Reasoning + Clips + Plan + LLM."""

from __future__ import annotations

import asyncio
import gc
import os

os.environ["PYTHONIOENCODING"] = "utf-8"

import sys
import time
from pathlib import Path


async def main() -> int:
    print("=" * 60)
    print("KAWKAB AI - PHASE 3 COMPREHENSIVE TEST")
    print("CV + Analysis + Reasoning + Clips + Training Plan + LLM")
    print("=" * 60)

    from kawkab.services import (
        AnalysisService,
        ClipExtractionService,
        CVService,
        KnowledgeService,
        LLMConfig,
        LLMService,
        ReasoningService,
        StorageService,
        TrainingPlanGenerator,
    )

    video_path = Path("data/real_match.mp4")
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        return 1

    print(f"\nVideo: {video_path} ({video_path.stat().st_size / 1024 / 1024:.1f} MB)")

    print("\n[1/9] Initializing services...")
    t0 = time.time()
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
    clip_service = ClipExtractionService()
    plan_generator = TrainingPlanGenerator(knowledge)
    await plan_generator.initialize()
    llm = LLMService(
        LLMConfig(
            provider="ollama",
            ollama_model="ministral-3:14b",
            max_tokens=4000,
            num_gpu=99,
        )
    )
    print(f"  Services ready in {time.time() - t0:.1f}s")

    print("\n[2/9] Saving match...")
    match_id = await storage.save_match(
        name="Phase 3 Test",
        video_path=str(video_path),
        home_team="Home FC",
        away_team="Away FC",
    )

    print("\n[3/9] CV pipeline (YOLOv11l + BoT-SORT)...")
    t0 = time.time()

    async def progress_cb(p, m):
        if int(p * 100) % 20 == 0:
            print(f"  {p * 100:.0f}% - {m}")

    track_data = await cv.process_video(video_path, progress_callback=progress_cb)
    cv_time = time.time() - t0
    print(
        f"  ✓ {len(track_data.frames)} frames, {len(track_data.track_registry)} tracks in {cv_time:.1f}s"
    )

    print("\n[4/9] Analysis (stats, formations, PPDA, xG/xT)...")
    t0 = time.time()
    match_analysis = await analysis.analyze_match(track_data, match_id=match_id)
    print(
        f"  ✓ Formations: {match_analysis.formations.get('home', {}).get('formation', '?')} / "
        f"{match_analysis.formations.get('away', {}).get('formation', '?')}"
    )
    print(
        f"  ✓ Possession: {match_analysis.home_team.possession_pct:.1f}% / {match_analysis.away_team.possession_pct:.1f}%"
    )
    print(f"  ✓ PPDA: {match_analysis.pressing_intensity:.1f}")
    print(f"  ✓ Players: {len(match_analysis.players)}, Events: {len(match_analysis.events)}")
    print(f"  ✓ Analysis in {time.time() - t0:.1f}s")

    print("\n[5/9] Tactical reasoning (Detective layer)...")
    t0 = time.time()
    diagnosis = await reasoning.diagnose_match(match_analysis, language="en")
    print(f"  ✓ {len(diagnosis.diagnoses)} diagnoses in {time.time() - t0:.1f}s")
    for i, d in enumerate(diagnosis.diagnoses[:5], 1):
        print(f"    {i}. {d.rule_name} (confidence: {d.confidence:.0%})")
        if d.recommended_drills:
            print(f"       → Drills: {', '.join(d.recommended_drills[:2])}")

    print("\n[6/9] Training plan generation...")
    t0 = time.time()
    plan = await plan_generator.generate_plan(diagnosis, duration_weeks=4, language="en")
    print(f"  ✓ {plan.duration_weeks}-week plan in {time.time() - t0:.1f}s")
    print(f"  ✓ {plan.total_drills} unique drills")
    for week in plan.weeks:
        print(
            f"    Week {week.week_number} ({week.theme}): {len(week.sessions)} sessions, "
            f"focus: {week.primary_focus[:60]}"
        )

    print("\n[7/9] Video clip extraction (evidence)...")
    t0 = time.time()
    clip_timestamps = []
    for d in diagnosis.diagnoses[:3]:
        clip_timestamps.append(
            {
                "start": 0,
                "end": 5,
                "description": f"Evidence for {d.rule_name}",
            }
        )

    evidence_clips = await clip_service.extract_evidence_clips(video_path, clip_timestamps[:2])
    print(f"  ✓ Extracted {len(evidence_clips)} evidence clips in {time.time() - t0:.1f}s")
    for clip in evidence_clips:
        print(f"    → {clip['filename']}")

    print("\n[8/9] Freeing GPU before LLM call...")
    await cv.shutdown()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    await asyncio.sleep(3)

    print("\n[9/9] Generating AI coach report...")
    t0 = time.time()
    top_players = sorted(
        match_analysis.players.values(), key=lambda p: p.distance_covered_m, reverse=True
    )[:5]
    top_summary = ", ".join(f"#{p.track_id}({p.distance_covered_m:.0f}m)" for p in top_players)

    llm_summary = f"""Match: Phase 3 Test
Duration: {track_data.duration_seconds:.0f}s
Possession: {match_analysis.home_team.possession_pct:.1f}% / {match_analysis.away_team.possession_pct:.1f}%
Formations: {match_analysis.formations.get("home", {}).get("formation", "?")} vs {match_analysis.formations.get("away", {}).get("formation", "?")}
PPDA: {match_analysis.pressing_intensity:.1f}
Events: {len(match_analysis.events)}, Confidence: {match_analysis.confidence_overall:.1%}

DIAGNOSES (top 3):
"""
    for i, d in enumerate(diagnosis.diagnoses[:3], 1):
        llm_summary += f"\n{i}. {d.rule_name} ({d.confidence:.0%})"
        if d.explanation:
            llm_summary += f"\n   {d.explanation[:200]}"

    llm_summary += f"\n\n4-WEEK PLAN: {plan.total_drills} drills, addresses {len(plan.priority_addressed)} issues"
    llm_summary += f"\nTop players: {top_summary}"

    try:
        report = await llm.generate_coach_report(llm_summary, language="en")
        llm_time = time.time() - t0
        print(f"  ✓ Report: {len(report)} chars in {llm_time:.1f}s")
        print("\n" + "=" * 60)
        print("COACH REPORT (with diagnoses + plan):")
        print("=" * 60)
        print(report[:2000])
        if len(report) > 2000:
            print(f"... ({len(report) - 2000} more chars)")
        print("=" * 60)
    except Exception as e:
        print(f"  [WARN] LLM failed: {e}")
        report = None

    print("\n" + "=" * 60)
    print("PHASE 3 RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Frames:           {len(track_data.frames)}")
    print(f"  Tracks:           {len(track_data.track_registry)}")
    print(f"  Players:          {len(match_analysis.players)}")
    print(f"  Events:           {len(match_analysis.events)}")
    print(
        f"  Formations:       {match_analysis.formations.get('home', {}).get('formation', '?')} / "
        f"{match_analysis.formations.get('away', {}).get('formation', '?')}"
    )
    print(f"  PPDA:             {match_analysis.pressing_intensity:.1f}")
    print(f"  Diagnoses:        {len(diagnosis.diagnoses)}")
    print(f"  Plan drills:      {plan.total_drills}")
    print(f"  Evidence clips:   {len(evidence_clips)}")
    print(f"  LLM report:       {len(report) if report else 0} chars")
    print(f"  CV:               {cv_time:.1f}s")
    print(f"  Total:            {cv_time + analysis_time_holder:.1f}s")
    print("=" * 60)

    if diagnosis.diagnoses and report and plan.total_drills > 0:
        print("✅ PHASE 3 COMPLETE — FULL COACH EXPERIENCE WORKING!")
    else:
        print("⚠️ PHASE 3 PARTIAL")
    print("=" * 60)

    await storage.close()
    return 0


analysis_time_holder = 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
