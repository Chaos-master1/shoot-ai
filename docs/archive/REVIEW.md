# Kawkab AI — Full Project Review

> **Note (v0.12.0):** This review was written at v0.4.1/v0.6.0. Since then, version strings have been reconciled to v0.12.0 across all docs, and most of the issues below have been resolved. Cross-reference with STATUS.md before acting on any finding.
>
> **Reviewer:** Mavis (AI assistant, fresh eyes)
> **Project version stated in README:** v0.4.1
> **Project version stated in PLAN.md:** v0.6.0
> **Date:** 2026-06-17
> **Scope:** README, PLAN.md, AGENTS.md, pyproject.toml, PROFESSIONAL_AUDIT.md, kawkab-ai-review.md, STATUS.md, source tree, knowledge base, test suite, scripts

## TL;DR (the verdict in one paragraph)

This is a **genuinely interesting product idea with real domain knowledge** that's been **sabotaged by uncontrolled scope creep** and **papered over with aspirational documentation**. The README admits "not production-ready," but the docs (PLAN.md, kawkab-ai-review.md, PROFESSIONAL_AUDIT.md) and the codebase disagree about what version we're at, what's done, and what works. There are **49 service files** for a project that documents 12–13 — including a `MuJoCoBallService` (CFD ball physics), a `FluidX3DService` (CFD air simulation), a `PsychologyService` (momentum tracking), a `WeatherService`, a `FaceRecognitionService`, and **six** external football data API integrations (StatsBomb, API-Football, TheSportsDB, OpenFootball, EasySoccerData, Bzzoiro). For a tool that cannot yet track 22 players without producing 28+ fragments, that is the wrong fight. **Cut 70% of the code, fix tracking + homography, ship to 5 coaches, then add features.**

---

## Part 1 — What's Genuinely Good

### 1.1 The product thesis is sharp
The amateur football coaching market is underserved. Professional tools (Hudl, Wyscout, Veo) are cloud, expensive, and English-only. A **100% offline, $0, Arabic-aware** alternative is a real wedge — especially in MENA where uploading amateur match footage to a foreign cloud is a non-starter for cultural and privacy reasons. Whoever wrote the original pitch understood the user.

### 1.2 Domain knowledge is real, not cargo-culted
The 42 tactical rule YAMLs (defensive/offensive/transitions/individual/meta) and 25 drill YAMLs aren't generic AI slop. The taxonomy matches how coaches actually think. PPDA, xT, VAEP, pressing shape, counter-pressing, fullback isolation — these are the right primitives. The `knowledge/tactics/` tree is a real asset and a defensible moat against Western competitors who'd need to learn the domain to replicate it.

### 1.3 Tech choices at each layer are correct
- **YOLOv11l + BoT-SORT** for tracking — right combo for 2026
- **Ollama + Ministral/Qwen** for local LLM — correct default for offline-first
- **PySide6 + QWebChannel** instead of a FastAPI/React split — correct for a desktop app; saves latency and complexity
- **`uv` for packaging** — modern, fast, correct
- **`pydantic` v2, `loguru`, `ruff`, `mypy strict`** — proper hygiene
- **Kowl/Kloppy + socceraction** for SPADL action valuation — right libraries
- **`pytest-asyncio`, `hypothesis`, `pytest-cov`** — the dev extras are real, not sticker decorations

### 1.4 Honesty is unusually high
`kawkab-ai-review.md` and `PROFESSIONAL_AUDIT.md` openly admit: tracking fragmentation, no homography by default, OCR unreliable, no coach validation, 1.75 GB bundle. Most projects in this stage hide that. The honesty is a strength — but the docs also conflate aspiration with reality, which I'll get to.

### 1.5 Knowledge base is production-quality YAML
The 42 rules + 25 drills are consistent schema, readable, and loadable. Whoever wrote these understood both football and good data modeling. This is the part of the codebase most likely to be valuable three years from now.

### 1.6 Graphify integration
Having `graphify-out/` with `graph.json`, a wiki, and a `GRAPH_REPORT.md` is unusual and useful for an AI-collaborative project. The fact that the project ships with a knowledge graph of its own code is forward-thinking.

---

## Part 2 — What's Bad (severity-ordered)

### 🔴 CRITICAL — Scope has exploded past the product's actual capability

**The single biggest problem.** I counted the actual source:

| Module | Count | README claim |
|---|---|---|
| Service files in `src/kawkab/services/` | **49 .py files** | "13 Services" |
| External football data integrations | 6 (StatsBomb, API-Football, TheSportsDB, OpenFootball, EasySoccerData, Bzzoiro) | not mentioned |
| Tracker implementations | 3+ (BoT-SORT via YOLO, Norfair, Roboflow Sports) | "BoT-SORT" |
| Ball physics simulators | 2 (MuJoCo, FluidX3D CFD) | 0 |
| "Soft" analytics services | PsychologyService, WeatherService, FaceRecognitionService, CardDetectionService, FootballRulesService | 0 |
| Test files | 16 | (unstated) |
| Test-to-service ratio | **0.33** | should be 1.0+ |

**Translation:** the project has built a 50-service Swiss-army-knife to analyze a video it can't yet track cleanly. The README says the system produces 28 tracks for a 22-player match — the team should be focused exclusively on that number, not on CFD ball physics and momentum psychology.

This isn't "feature creep" any more, it's **distraction engineering**. Every hour spent on `MuJoCoBallService` is an hour not spent on the thing the product's value depends on.

### 🔴 CRITICAL — Documentation contradicts itself about version and feature status

- `README.md` says **v0.4.1**
- `PLAN.md` says **v0.6.0** (current target)
- The architectures in `kawkab-ai-review.md` describe 8 services, README 12, actual code 49
- The README claims xT/xG are "✅ Working" with homography, but `PROFESSIONAL_AUDIT.md` says xT is "⚠️ Simple / heuristic" and homography is "manual only"
- The PLAN says "Phase 2 done, Phase 3 next," but the audit document says multi-match aggregation, player profiles, scouting are "Not started — Critical"
- `kawkab-ai-review.md` and `PROFESSIONAL_AUDIT.md` are more honest than `README.md`, but **the README is what a coach will see first**

This is dangerous: new contributors (or the next AI session) will pick the wrong source of truth and waste time.

### 🔴 CRITICAL — The "core loop" still doesn't work

Both existing reviews call this out, and they're right. For a tool that claims to give "real meters" for distance, xT, formations:

1. **Tracking is fragmented** (28 tracks for 22 players is "excellent" only by comparison to the 91-tracks before)
2. **Homography is manual** (coach must click 4 corners per match — an amateur coach will not do this)
3. **No auto keypoint detection** for the pitch lines
4. **OCR for jersey numbers is unreliable** (the README admits this)
5. **No team formation detection validates against the truth** (claim "4-4-3 detected" with no ground truth)

Until the coach opens the app, drops a phone-recorded amateur match, and gets back possession %, distance, and a formation **without touching any controls**, the product is a research demo.

### 🟠 HIGH — Test coverage is structurally insufficient

- 16 test files for 49 services → 0.33 ratio
- A test exists for `cv_service_model_manager` and `kalman_and_match_type` but not for `cv_service.py` itself
- A test exists for `analysis_service` and `validation_service` but not for `reasoning_service`, `homography_service`, `llm_service`, `storage_service`, `training_plan_service`, or any of the 30+ "secondary" services
- `tests/integration/` only has 3 files
- There is **no end-to-end test that takes a real amateur video and asserts usable output** — `scripts/end_to_end_test.py` exists but it's a script, not a CI test

The dev extras (`pytest`, `hypothesis`, `mypy strict`) are configured but not enforced by CI. There's no GitHub Actions config visible at the top level — the `.github/` directory exists but I didn't audit it; if there's no CI, all that tooling is decorative.

### 🟠 HIGH — Bundle size is an unsolved product problem, not an engineering footnote

The README admits: **1.75 GB installer, 66 MB exe**. For an amateur coach with a 2019 laptop and 60 GB free, that's a 3% disk hit before they've even opened the app. Combined with the requirement for an NVIDIA GPU with 8GB+ VRAM, the addressable market is "amateur coaches who happen to own a gaming PC and have 1.75 GB to spare." That's maybe 5% of the target market.

The fix is `VRAMManager` (which exists but `ModelPriority` is suspicious — see code smells below) and **lazy model loading** (which is on the roadmap but not built). Until this is solved, "private + free" is undercut by "but the download will take 40 minutes and won't run on your hardware."

### 🟠 HIGH — Two competing CV pipelines is technical debt

`services/cv_service.py` (41 KB) and `services/cv_service_v2.py` (14 KB) coexist. The v1 is exported in `__init__.py`; v2 is not. This pattern — the "v2 in a side file" — is how a codebase rots. Either v2 is better and v1 should be deleted, or v2 is dead code. Same smell with `clip_service.py` (6 KB) vs `clip_extraction_service.py` (9.5 KB).

The `__init__.py` exports `ClipExtractionService` from `clip_service` (line 15) but the file `clip_extraction_service.py` also exists. Which is canonical? **A new contributor will not know.**

### 🟠 HIGH — Version drift in `pyproject.toml`

- `[project]` says `version = "0.1.0"`
- `[project.scripts]` says `kawkab = "kawkab.__main__:main"`
- README says v0.4.1, PLAN says v0.6.0, kawkab-ai-review says "0.5.5-ish"
- `norfair`, `easyocr`, `playwright`, `sports @ git+...` are in main dependencies, but `playwright` and `easyocr` aren't used in any service I've seen — they're speculative bloat

The mix of `onnx<1.19` (a hard pin) with `torch>=2.2.0` (loose) suggests the lockfile was hand-tuned. That's normal during dev, but it should be cleaned before any release.

### 🟡 MEDIUM — "Async" claim is half-true

The README and PLAN emphasize async services. But the knowledge base loads, the LLM call, the homography math, and most analysis are CPU-bound — `asyncio` doesn't help with the GIL or with numpy code. The async wrapping is a layer of complexity for clarity, not throughput. If that's the choice, document it; if not, the "async services" framing is misleading.

### 🟡 MEDIUM — The LLM is being asked to do too much

`LLMService` generates reports in EN/AR. The system is local, free, private — that's the moat. But:
- Models like Ministral 14B **will hallucinate** tactical conclusions
- There's no ground-truth validation against actual match events
- The reasoning layer (`ReasoningService`) fires rules that depend on metrics that may be wrong
- The LLM is then asked to *narrate* those potentially-wrong rules

This is the classic "garbage in, eloquent out" failure mode. The reports will sound authoritative and may be wrong. A coach who follows the report and is undermined in front of their players will never come back. **The LLM is the most dangerous component in the stack and is the least validated.**

### 🟡 MEDIUM — Five-track solution sprawl on the left side of the stack

```
CVService           (YOLO+BoT-SORT)
NorfairTracker      (own implementation)
RoboflowSportsService (third party)
LightGlueHomographyService
HomographyService
```
Three different trackers. Two different homography implementations. This is "did not commit to an architecture" expressed as code. Pick one. Make it the default. Add a flag for the others.

### 🟡 MEDIUM — Graphify output is stale or not committed regularly

`graph.json` is 1.9 MB. `graph.html` is 1.78 MB. The `manifest.json` is 28 KB. Either the graph is being kept up to date (great) or these are large stale artifacts committed once and never refreshed. Without CI integration, I can't tell. The AGENTS.md says "after modifying code, run `graphify update .`" but nothing enforces it.

### 🟢 LOW — Repo hygiene

- Two model weights (`yolo11l.pt` 51 MB, `yolo11n.pt` 5.6 MB) checked in. Should be in `.gitignore` and pulled on first run, or referenced via a download script. 56 MB of binary in git history is a real cost.
- A 363 MB test video (`Sweden vs Tunisia Highlight | FIFA World Cup 2026T [D-yjyIPVCfE].mp4`) is sitting in the repo root. **363 MB.** This should not be in git.
- `uv.lock` is 801 KB — fine, lockfiles belong in version control.
- `build/`, `dist/`, `.pytest_cache/`, `.venv/` are present at the root — should be `.gitignore`d.
- `__pycache__` directories are inside `src/`, meaning at least one path-based import (`from kawkab.services...`) was tested after a stale `.pyc`. Should be ignored.

### 🟢 LOW — README and PLAN don't reflect the actual workflow

The README quick-start is 7 steps deep with `winget` commands, `uv sync --extra gpu --extra audio --extra tactical --extra dev`, a manual CUDA PyTorch reinstall, an Ollama model pull of 8-9 GB, then a verification script. That's a 30-minute install on a good day, with two points of likely failure (CUDA wheel index, Ollama model download). For a non-technical amateur coach, this is a hard **no**.

The "v0.4.1 Workflow" at the bottom of the README still has 5 manual steps including "click 4 pitch corners." For an amateur audience, this entire workflow is a UX problem.

---

## Part 3 — The Extras (things people miss)

### 3.1 The thing nobody's talking about: storage layer
`storage_service.py` is 26 KB. `analysis_service.py` is 45 KB. `reasoning_service.py` is 19 KB. These three files alone are ~90 KB — roughly the size of three normal applications. They're each doing too much. **MultiMatchAnalysisService, PlayerProfileService, DataExportService, FeedbackService, ValidationService, AnomalyDetectionService, QualityScoringService** are all probably either reusing or duplicating logic in those three. Refactor or consolidate before adding the next thing.

### 3.2 The CV service is the only one that matters right now
Look at what the coach actually needs from a v1 product:
1. Drop a video
2. Get back: possession %, distance, formation, 2-3 tactical observations, a 2-minute audio summary
3. Done

That's it. None of: weather, momentum psychology, CFD ball physics, facial recognition, jersey OCR, card detection, multiple data API integrations, MuJoCo ball simulation, multi-match aggregation.

The "vertical slice" of v1 is: 1 video → 1 report. **The project has built a platform, not a vertical slice.**

### 3.3 The Arabic angle is genuine IP
The bilingual EN/AR support in `web/locales/` and the LLM's EN/AR switching is a real differentiator. Most "AI sports analysis" tools are English-only. For amateur coaches in Egypt, Saudi Arabia, Morocco, Tunisia, Algeria, Iraq, this is a 5x+ larger addressable market than for the equivalent English-only tool. The README mentions this in one line; the rest of the docs treat it as a checkbox. It should be in the hero, the install flow, and the marketing. The Arabic reports need a native-Arabic football coach to validate, just like the English ones do.

### 3.4 The knowledge base has unused structure
There's a `src/kawkab/knowledge/mappings/` directory I didn't fully explore, plus a `tactics/` taxonomy with five sub-categories, plus `drills/` (25), plus `rules/laws_of_the_game.yaml` (which is a static dump of FIFA Laws, not a tactical rule). The 11K `laws_of_the_game.yaml` file is **dead weight** unless the project plans to actually do referee/rule inference from video — which I see no plan for. If the project isn't going to do that, delete the file. If it is, put it on the roadmap explicitly.

### 3.5 There's no model card, no data card, no eval set
For a CV/LLM system making claims about football matches, there's no:
- Documented evaluation set (what counts as a good track? a good xG? a good report?)
- Comparison to a baseline (Hudl, Veo, free tools)
- Held-out test video(s) with ground truth
- A "model card" explaining YOLO weights used, training data origin, known failure modes
- A "data card" for the rules — who wrote them, what sources, what authority

The "trust" layer of an AI product is invisible work, and it's missing here. Coaches can't evaluate this on their own — they need a benchmark.

### 3.6 The graphify graph is 1.9 MB and could be the real product
If you took just the **knowledge base** (42 rules + 25 drills) and exposed it as a queryable graph (a "football tactical knowledge graph" with a web UI), that alone is publishable IP. The graph already exists in the project. A separate, much smaller product — a tactical knowledge graph for coaches — could be a free funnel into the full app. **You're sitting on a graph you haven't monetized yet.**

### 3.7 The "professional" service file size tells the story
- `analysis_service.py` — 45 KB
- `cv_service.py` — 41 KB
- `storage_service.py` — 26 KB
- `multi_match_analysis_service.py` — 19 KB
- `reasoning_service.py` — 18 KB
- `card_detection_service.py` — 15 KB
- `pressure_metrics_service.py` — 16 KB
- `app.py` — 13 KB

These are large files. A well-factored service is usually under 10 KB. **Either these services are doing too much, or they have low test coverage and are accumulating dead branches.** Both are likely. Split them or test them.

### 3.8 The `__init__.py` is also a public API contract
Look at the export list in `services/__init__.py`: 49 services and ~50 classes. Any one of those 50 breaking changes is a public API break. A new contributor cannot reasonably know what the "surface" is. The fact that `BenchmarkService` and `ValidationService` are imported in `__init__.py` but **not in the imports at the top** (line 7 onwards) means there's a runtime ImportError waiting to happen if those classes aren't actually importable — but the export list claims they are. (Lines 93-99 export names that aren't imported on lines 7-60.) **This is a latent bug.**

### 3.9 The BzzoiroService is named after something I cannot identify
There's a `bzzoiro_service.py` in the services directory. There's no Bzzoiro in any of the public docs, in PLAN.md, in the audit, or in the README. This is either a personal/internal name that leaked, a placeholder, or a feature I can't see. Worth a rename to whatever it actually does, or a deletion.

### 3.10 The "psychology service" is the canary
`psychology_service.py` produces `ScoreState`, `MomentumPoint`, `PsychologyEvent`. None of these are measurable from a single amateur video. **This service should not exist yet.** It exists because someone had an idea at 2am. It is the canary in the coal mine for scope creep.

---

## Part 4 — The Honest Path Forward

If I were the maintainer, here's what I'd do in the next 30 days. In priority order.

### Week 1 — Cut and document
1. **Delete 30+ services** that don't serve the v1 vertical slice. Keep:
   - `CVService` (or merge v2 in, delete v2 file)
   - `HomographyService` (or `LightGlueHomographyService`, pick one)
   - `VRAMManager` (or delete it)
   - `AnalysisService` (refactor down to <20 KB)
   - `ReasoningService` (keep, but the rule firing logic must validate against measured metrics)
   - `LLMService` (keep, but add a "groundedness" check before it speaks)
   - `KnowledgeService` (keep, this is the IP)
   - `StorageService` (refactor; the 26 KB is too much)
   - `TrainingPlanGenerator` (keep, it's a separate feature)
   - `ClipService` (pick one of the two)
   - `BatchService` (keep if used)
   - `App` entry point
2. **Delete the stale `cv_service_v2.py` and `clip_extraction_service.py`** or merge them in. Don't leave v2 files.
3. **Delete `bzzoiro_service.py`** unless it does something named-and-documented. Don't ship mystery code.
4. **Delete the 363 MB test video from the repo.** Move to a release artifact or external storage.
5. **Add `.gitignore` for `*.pt`, `*.mp4`, `build/`, `dist/`, `__pycache__/`, `.venv/`, `.pytest_cache/`.**
6. **Update `pyproject.toml` version** to match README. Pick one source of truth.
7. **Fix `services/__init__.py`** — either import everything in the export list, or remove the exports. Currently broken.

### Week 2 — Make the v1 loop work
1. **Auto-homography from pitch-line keypoints** (use a small CNN, not LightGlue). Coach should never click corners.
2. **Top-N filter to 22 tracks.** Don't return 28, return 22 by team assignment. The extra 6 are likely referees or duplicate tracks; rank by track length, take the 22 longest.
3. **One end-to-end integration test** that runs `analyze_video()` on a fixture video and asserts the output contains: possession percentages that sum to 100, exactly 22 player stats, a formation string, and a non-empty LLM report. This is the v1 acceptance test.
4. **Move the install path to a single `uv run kawkab` command.** No CUDA manual install, no Ollama model pull as a separate step — bundle a small model in the installer.

### Week 3 — Get 5 coaches
1. Ship a Windows installer to 5 amateur coaches in different countries.
2. Give them 3 matches to analyze. Don't ask "do you like it" — ask "what stat did you show your assistant? What did you change in training based on it?"
3. If the answer to the second question is "nothing," the product is a fancy video player, not a coaching tool.

### Week 4 — Decide
1. If 3/5 coaches say "I used the report to change a drill this week" — keep going. Fundraise or charge.
2. If 0/5 do — go back to the technical foundation. Don't add features.
3. The next 6 months should be one of two things: (a) ship a working v1 to 100 coaches, or (b) become the amateur-coach tactical knowledge graph product, which is a smaller, faster, more fundable wedge.

---

## Final Word

This is the kind of project I want to be optimistic about, and I'm trying not to be unkind, but the gap between **what the docs say** and **what the code shows** is large enough that a friendly review is more harmful than a critical one. The vision is sound. The execution is over-extended. The right move is not "add more" — the right move is "cut to 12 services, fix the core loop, ship to 5 coaches, decide."

If the maintainer can do that, this becomes a real product in 6 months. If they can't — if the temptation to add `WeatherService` integrations wins again — it'll be a beautiful, sprawling, unmaintained research repo in 12.

**The next 30 days are decisive. Pick the fight.**

---

*Review by Mavis. Bring questions or disagreement — this is a critique, not a verdict.*
