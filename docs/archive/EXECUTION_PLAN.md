# Kawkab AI — Execution Plan (v2, long-term professional)

> **Supersedes:** the earlier v1 of this doc, which proposed cutting services and a 30-day decision. **That was wrong for this project.** The user is a long-term builder, not a startup founder, and wants professional standards, not MVP scoping.
> **Companion:** `REVIEW.md` (diagnosis), `ITERATION_LOG.md` (the actual operating plan)
> **Mission:** treat Kawkab AI as a professional sports analytics product, iterated continuously.

---

## The framing, corrected

The previous v1 of this plan assumed "ship fast, validate with 5 coaches, decide in 30 days." That was my bias. The correct framing for this project is:

- **No artificial deadline.** Years are fine.
- **No service cuts.** The 49 services are the project's breadth. Keep them.
- **No "MVP."** Aim for pro-tool parity (StatsBomb, Hudl, Wyscout, Second Spectrum, SciSports).
- **Continuous iteration.** Each cycle: pick one weakness → fix it → test it → ship it → next.

The `ITERATION_LOG.md` is the real operating plan. This doc is the long-arc context: what's coming over months, and why.

---

## What's coming, in arc order

### Arc 1 — Foundation (next ~4 weeks)
Get the existing system working reliably enough to be the basis for everything else.

1. Make the video pipeline robust on a real match video (the user's test video)
2. Fix the latent `__init__.py` ImportError
3. Reconcile version strings + doc consistency
4. Add CI (GitHub Actions: ruff, mypy, pytest)
5. Consolidate `cv_service_v2.py` and `clip_extraction_service.py` (no service deleted — pick the winner in each pair, delete the loser)
6. Tighten `.gitignore`; remove the 363 MB test video from git index
7. Add the first end-to-end integration test on a fixture video
8. Slim `analysis_service.py` (45 KB → <20 KB) and `storage_service.py` (26 KB → <15 KB) by extracting sub-modules

### Arc 2 — Trust layer (~2 weeks)
The "professional" difference. Make claims falsifiable.

9. Model card (`docs/MODEL_CARD.md`)
10. Data card (`docs/DATA_CARD.md`)
11. Ground-truth eval set (≥3 amateur matches with hand-tagged events)
12. LLM groundedness check (block report generation if input stats are below a confidence threshold)
13. Document known failure modes for each tracker
14. Anomaly detection for impossible stats

### Arc 3 — Data layer (~3 weeks)
What pro tools have that Kawkab doesn't, but every coach needs.

15. Season / team / competition schema
16. Player profile system (photo, position, attributes, history)
17. Multi-match player tracking with persistent IDs (ReID)
18. Squad / roster / substitutions model
19. Analysis versioning (track manual corrections)

### Arc 4 — Analytics depth (~6 weeks, largest arc)
The big "missing analytics" list from the team's own `PROFESSIONAL_AUDIT.md`.

20. Heatmaps (player, team, action, defensive)
21. Weighted pass networks (xT-weighted, directional)
22. Pass sonars
23. xT / xG model trained on real data
24. Defensive action maps
25. Pressing maps
26. Set-piece analysis
27. Transition analysis
28. Physical load tracking
29. Progressive passes / carries
30. Passes into final third / penalty area
31. Pressure events
32. Multi-match aggregation (season averages, percentiles, trends)
33. Opponent scouting
34. Player baselines vs team / league average

### Arc 5 — Workflow & UX (~3 weeks)
What coaches actually do with the data.

35. Video clip annotation tool (draw on video, tag events)
36. Side-by-side video comparison
37. Custom report builder
38. Coach-friendly dashboard
39. Sharing / export (PDF, CSV, shareable link)
40. Mobile companion (review on phone)
41. Extended language support (French, Spanish beyond the current EN/AR)
42. Custom formation editor

### Arc 6 — Performance & scale (~3 weeks)
The boring stuff that matters at pro level.

43. Lazy model loading (1.75 GB → 500 MB installer)
44. Single-command install
45. GPU VRAM budget audit + fix
46. Async pipeline parallelization
47. Match-length video chunking
48. Incremental analysis (resume from checkpoint)
49. Batch processing
50. CPU fallback for development

### Arc 7 — Security & ops (~1 week)

51. Sandboxed LLM prompts (no prompt injection)
52. Input validation on all CLI / UI entry points
53. Structured logging with privacy redaction
54. Dependency vulnerability scanning (pip-audit in CI)
55. Reproducible builds verification
56. Telemetry opt-in (or off by default)

---

## What this looks like in practice

Each arc takes 1–6 weeks. Each cycle inside an arc takes 1–4 hours. Each cycle is one commit. Over a year, that's ~250–500 cycles, which is a real professional product pace.

The `ITERATION_LOG.md` is where the actual work is tracked. This doc is the map.

---

## What I (Mavis) will do in this session

Given the per-session budget, here's what I can deliver in THIS conversation:

1. ✅ `REVIEW.md` — the diagnosis
2. ✅ `ITERATION_LOG.md` — the backlog of 63 targets, the cycle log, the quality bar
3. ✅ This file (`EXECUTION_PLAN.md` v2) — the long-arc context
4. ⏳ Cycle 1 setup: read the actual CV pipeline code to find the real blockers
5. ⏳ Get the user's video test results so cycle 1 has input
6. ⏳ Pick the next 3–5 cycle targets based on what the user wants to prioritize

## What happens in future sessions

Each session, we do one of these:
- **Run cycle N** — pick a target, execute it, commit it, log it
- **Plan arc X** — break an arc into specific cycles with file lists and acceptance tests
- **Audit** — re-read the code, find new weaknesses, add to the backlog
- **Test the user** — ship a build to a coach, get feedback, update priorities

---

## The discipline

The user said: "Propose improvements, implement them, test them, and repeat."

The loop is in `ITERATION_LOG.md`. The quality bar is in `ITERATION_LOG.md`. The backlog is in `ITERATION_LOG.md`. This doc is the **why** — the long-term plan that explains why each cycle matters.

**Don't break the loop. Don't ship without tests. Don't lie in docs. Don't add code without justification. Don't cut features because someone on the internet said to.**

This is a real product. Treat it that way.
