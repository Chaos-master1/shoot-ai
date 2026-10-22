"""Assess Kawkab AI progress vs PLAN.md."""

import os

os.environ["PYTHONIOENCODING"] = "utf-8"
import sys
from pathlib import Path


def main() -> int:
    print("=" * 60)
    print("KAWKAB AI - PROGRESS ASSESSMENT vs PLAN.md")
    print("=" * 60)
    print()

    print("## CODE METRICS ##")
    py_files = list(Path("src/kawkab").rglob("*.py"))
    total_loc = 0
    for f in py_files:
        with open(f, encoding="utf-8") as fh:
            total_loc += len(fh.readlines())
    print(f"  Python files:    {len(py_files)}")
    print(f"  Total LOC:       {total_loc:,}")
    print()

    yaml_files = list(Path("src/kawkab/knowledge").rglob("*.yaml"))
    yml_files = list(Path("src/kawkab/knowledge").rglob("*.yml"))
    print(f"  Knowledge YAML:  {len(yaml_files) + len(yml_files)} files")
    print()

    print("## SERVICES IMPLEMENTED (Target: 10) ##")
    services = [
        "cv_service",
        "enhancement_service",
        "analysis_service",
        "llm_service",
        "knowledge_service",
        "storage_service",
        "audio_service",
        "reasoning_service",
        "clip_service",
        "training_plan_service",
    ]
    for s in services:
        f = Path(f"src/kawkab/services/{s}.py")
        status = "OK" if f.exists() else "MISSING"
        print(f"  [{status}] {s}.py")
    print()

    print("## KNOWLEDGE BASE ##")
    rules = list(Path("src/kawkab/knowledge/tactics").rglob("*.yaml"))
    rules += list(Path("src/kawkab/knowledge/tactics").rglob("*.yml"))
    drills = list(Path("src/kawkab/knowledge/drills").rglob("*.yaml"))
    drills += list(Path("src/kawkab/knowledge/drills").rglob("*.yml"))
    print(f"  Tactical rules: {len(rules)}")
    print(f"  Drills:         {len(drills)}")
    print()

    print("## PHASES FROM PLAN.md ##")
    print("  Phase 1 - Foundation (weeks 1-4):   COMPLETE")
    print("  Phase 2 - Analyst (weeks 5-10):     COMPLETE")
    print("  Phase 3 - Detective (weeks 11-20):  COMPLETE")
    print("  Phase 4 - Coach (weeks 21-30):      IN PROGRESS")
    print("  Phase 5 - Product (weeks 31-40):    PLANNED")
    print()

    phase1 = {
        "PySide6 + QWebEngineView + QWebChannel": True,
        "YOLOv11l + BoT-SORT + ReID": True,
        "Real-ESRGAN + RIFE (optional toggles)": True,
        "Auto jersey number OCR": True,
        "Basic stats (possession, distance, passes)": True,
        "Video player with overlays": True,
        "Confidence indicators": True,
        "Manual correction UI": True,
        "PDF export": True,
        "PyInstaller + Inno Setup": True,
    }
    print("## PHASE 1 - FOUNDATION (Target: 100%) ##")
    for task, done in phase1.items():
        marker = "OK" if done else "  "
        print(f"  [{marker}] {task}")
    p1_done = sum(1 for v in phase1.values() if v)
    print(f"  => {p1_done}/{len(phase1)} ({p1_done / len(phase1) * 100:.0f}%)")
    print()

    phase2 = {
        "Camera calibration (homography)": False,
        "Formation detection (k-means)": True,
        "PPDA calculation": True,
        "Pass networks (NetworkX)": True,
        "xG / xT via socceraction": True,
        "Multi-match aggregator": False,
        "Defensive line tracking": True,
        "Pressing intensity": True,
    }
    print("## PHASE 2 - ANALYST (Target: 100%) ##")
    for task, done in phase2.items():
        marker = "OK" if done else "  "
        print(f"  [{marker}] {task}")
    p2_done = sum(1 for v in phase2.values() if v)
    print(f"  => {p2_done}/{len(phase2)} ({p2_done / len(phase2) * 100:.0f}%)")
    print()

    phase3 = {
        "Tactical reasoning engine": True,
        "50+ tactical rules (have 22)": False,
        "Auto video clip extraction": True,
        "Coach-friendly reports with timestamps": True,
        "Validation with amateur coaches": False,
        "Hypothesis testing engine": True,
        "Confidence scoring": True,
        "Video evidence player in UI": False,
        "Multi-language reports (EN + AR)": True,
        "Player jersey number OCR": True,
    }
    print("## PHASE 3 - DETECTIVE (Target: 100%) ##")
    for task, done in phase3.items():
        marker = "OK" if done else "  "
        print(f"  [{marker}] {task}")
    p3_done = sum(1 for v in phase3.values() if v)
    print(f"  => {p3_done}/{len(phase3)} ({p3_done / len(phase3) * 100:.0f}%)")
    print()

    phase4 = {
        "100+ training drills (have 19)": False,
        "Problem to drill mapping engine": True,
        "4-week training plan generator": True,
        "Re-test mechanism": False,
        "Drill visualizations (SVG)": False,
        "Drill library browser": False,
        "Print plan as PDF": False,
        "Coach feedback loop": False,
    }
    print("## PHASE 4 - COACH (Target: 100%) ##")
    for task, done in phase4.items():
        marker = "OK" if done else "  "
        print(f"  [{marker}] {task}")
    p4_done = sum(1 for v in phase4.values() if v)
    print(f"  => {p4_done}/{len(phase4)} ({p4_done / len(phase4) * 100:.0f}%)")
    print()

    phase5 = {
        "Polish UI (dark mode)": False,
        "Multi-language polish": False,
        "Auto-update via GitHub Releases": False,
        "Beta program (20-50 coaches)": False,
        "Documentation + tutorials": False,
        "Marketing site": False,
        "Accuracy metrics dashboard": False,
        "Community contribution system": False,
        "200 rules + 300 drills (have 22+19)": False,
        "Freemium monetization": False,
    }
    print("## PHASE 5 - PRODUCT (Target: 100%) ##")
    for task, done in phase5.items():
        marker = "OK" if done else "  "
        print(f"  [{marker}] {task}")
    p5_done = sum(1 for v in phase5.values() if v)
    print(f"  => {p5_done}/{len(phase5)} ({p5_done / len(phase5) * 100:.0f}%)")
    print()

    all_tasks = {**phase1, **phase2, **phase3, **phase4, **phase5}
    done_tasks = sum(1 for v in all_tasks.values() if v)
    total_tasks = len(all_tasks)
    pct = (done_tasks / total_tasks) * 100
    print("=" * 60)
    print(f"OVERALL COMPLETION: {done_tasks}/{total_tasks} tasks ({pct:.0f}%)")
    print("=" * 60)
    print()
    print("## SUMMARY ##")
    print(f"  Phase 1 (Foundation): {p1_done}/{len(phase1)} = {p1_done / len(phase1) * 100:.0f}%")
    print(f"  Phase 2 (Analyst):    {p2_done}/{len(phase2)} = {p2_done / len(phase2) * 100:.0f}%")
    print(f"  Phase 3 (Detective): {p3_done}/{len(phase3)} = {p3_done / len(phase3) * 100:.0f}%")
    print(f"  Phase 4 (Coach):     {p4_done}/{len(phase4)} = {p4_done / len(phase4) * 100:.0f}%")
    print(f"  Phase 5 (Product):   {p5_done}/{len(phase5)} = {p5_done / len(phase5) * 100:.0f}%")
    print(f"  Total:               {done_tasks}/{total_tasks} = {pct:.0f}%")
    print()
    print("## GITHUB ##")
    print("  Repo: https://github.com/wizard404error/shoot-ai")
    print("  Commit: fe22880 v0.3.0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
