# Cycle 1 Prep — Video Pipeline Failure Points

> **Goal:** when you report what broke on your match video, we can target the fix in 1 cycle, not 5.
> **Compiled:** 2026-06-17, after reading `cv_service.py`, `analysis_service.py` (entry points).

---

## Pipeline at a glance

The video pipeline runs in this order:

```
1. CVService.process_video(video)         → MatchTrackData (raw tracks)
   ├─ YOLOv11 + BoT-SORT (or Norfair if available and frame_skip ≤ 3)
   ├─ Frame skip (default 1, recommend 3 for speed)
   ├─ Smart filters: min_track_lifetime, lifetime_pct, max_keep_top_n
   ├─ Team color clustering (k-means on torso pixels)
   └─ Quality assessment (raw/filtered ratio → "excellent" / "fair" / "poor")

2. AnalysisService.analyze(match_data, homography)  → stats
   ├─ Possession %, distance, speed, formations
   ├─ xT, xG, PPDA, progressive passes (in meters if homography applied)
   └─ Requires homography for any spatial stat in meters

3. ReasoningService.diagnose(stats, match_data)  → DiagnosisReport
   ├─ Loads rules from knowledge/tactics/*.yaml
   ├─ Fires rules against measured stats
   └─ May fire on stats that are wrong (no groundedness check)

4. LLMService.generate_report(diagnosis, language)  → narrative
   ├─ Ollama (local) or stub
   ├─ Prompt: stats + diagnosis + rule citations
   └─ No ground-truth check; may hallucinate
```

---

## Likely failure points on an amateur video

Ranked by probability × impact. **If your video failed, check these in order.**

### 1. YOLO model weights missing or wrong path
- **Where:** `cv_service.py:initialize()` loads YOLO weights
- **Symptom:** `FileNotFoundError` or "model not found" on startup
- **Likely on:** fresh install, no model downloaded
- **Fix path:** add a `scripts/download_models.py` or first-run downloader

### 2. BoT-SORT not installed (Norfair fallback)
- **Where:** `cv_service.py:346` — `if _NORFAIR_AVAILABLE and frame_skip <= 3:`
- **Symptom:** tracking fragmentation worse than expected
- **Likely on:** amateur footage from phone, broadcast works
- **Fix path:** install boxmot with `uv add boxmot`; verify `pip show boxmot`

### 3. Team color clustering fails (no valid color samples)
- **Where:** `cv_service.py:493` — "No color samples for valid tracks"
- **Symptom:** all players assigned to one team, or `player_teams` empty
- **Likely on:** dark kits, low light, similar colors
- **Fix path:** fall back to pitch-side heuristic (already in v0.5.4 per STATUS.md)
- **UX:** coach can use `swap_teams()` button to flip assignments

### 4. Homography not applied (all stats in pixels)
- **Where:** `AnalysisService` — only converts to meters if homography passed
- **Symptom:** distance, xT, formation positions in "weird units" (pixels, not meters)
- **Likely on:** coach didn't click 4 pitch corners (per README, manual only)
- **Fix path:** auto-homography from pitch keypoints (planned cycle #5 in backlog)
- **Workaround:** in app, ask coach to click 4 corners before analysis

### 5. Frame skip extrapolation drifts
- **Where:** `cv_service.py:393-399` — frames between detections copy `last_detections`
- **Symptom:** player positions "teleport" or appear stationary during long gaps
- **Likely on:** fast camera pans, occlusions
- **Fix path:** use Kalman smoother for full matches (already exists per STATUS.md v0.6.0)
- **Verify:** the Kalman wiring is in `AnalysisService` per PLAN.md

### 6. Max speed cap (36 km/h) clips amateur players
- **Where:** `analysis_service.py` somewhere — STATUS.md says "hard 36 km/h cap"
- **Symptom:** reported max speeds look low, distance is underestimated
- **Likely on:** athletic amateur players, fast wingers
- **Fix path:** configurable cap, or use percentiles not absolute

### 7. LLM hallucinates on bad input
- **Where:** `LLMService.generate_report()` — no groundedness check
- **Symptom:** report sounds authoritative but contradicts the stats
- **Likely on:** always, if stats are noisy
- **Fix path:** add a groundedness check (cycle #20 in backlog)
- **Immediate fix:** show the stats alongside the report so coach can sanity-check

### 8. Knowledge rules fire on metrics we don't measure
- **Where:** `ReasoningService` + YAML rules
- **Symptom:** rule fires (e.g. "low pressing shape") but the underlying metric was never computed
- **Likely on:** any rule whose `requires` field references an unmeasured metric
- **Fix path:** audit each rule's `requires` against what the analysis actually computes

---

## Debugging checklist (when you report what broke)

Run through these and tell me the results:

1. **Did the desktop app launch?** (UI showed up?)
2. **Did you select the video and click "Analyze"?**
3. **Did YOLO load?** (Look in `logs/kawkab.log` for "YOLO model loaded" or similar)
4. **How many raw tracks were detected?** (Log line: "Raw tracking: N unique tracks before filtering")
5. **How many valid player tracks after filtering?** ("After filtering: M validated player tracks")
6. **Did team detection work?** (Log line: "Team color clustering on N valid tracks" or "No color samples")
7. **Did you calibrate homography?** (Clicked 4 corners before re-analysis?)
8. **Did the LLM generate a report?** (Even if stats were wrong, was the text readable?)
9. **What did the report say vs. what the stats said?** (Copy-paste both)
10. **What was the overall tracking quality assessment?** ("excellent" / "fair" / "poor")

With those 10 answers I can target the fix in one cycle.

---

## What "success" looks like for Cycle 1

A pass = the following all work end-to-end on your amateur video:

- [ ] App launches without crash
- [ ] YOLO loads
- [ ] Tracker produces N raw tracks
- [ ] After filtering, the player count is reasonable (18-30, with known noise)
- [ ] Teams are assigned (or coach can swap_teams to correct)
- [ ] Stats are computed (even if numbers are wrong)
- [ ] LLM report is readable in EN or AR
- [ ] App doesn't OOM or hang for 90+ minutes on a 90-min match

A fail = any of the above is broken, and the error is in a log.

---

## Files to read when debugging

| Symptom | Read first |
|---|---|
| App won't start | `src/kawkab/app.py` |
| YOLO fails | `src/kawkab/services/cv_service.py` `initialize()` |
| Tracking wrong | `src/kawkab/services/cv_service.py` `process_video()` (line 307+) |
| Stats wrong | `src/kawkab/services/analysis_service.py` |
| LLM wrong | `src/kawkab/services/llm_service.py` |
| Rules fire wrongly | `src/kawkab/services/reasoning_service.py` + `src/kawkab/knowledge/tactics/` |
| Storage fails | `src/kawkab/services/storage_service.py` |
| UI hangs | `src/kawkab/ui/bridge.py` |

---

## Cycle 1 next steps

Once you tell me what broke, the cycle will be:

1. Read the relevant code (table above)
2. Identify the specific failure point
3. Write a 1-line regression test
4. Fix the bug
5. Run the test
6. Log the cycle

Estimated: 1-2 hours per failure point.
