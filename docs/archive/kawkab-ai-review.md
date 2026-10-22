# Kawkab AI — Full Code Review & Opinion
> Reviewed from: `README.md`, `PLAN.md`, `pyproject.toml`, `BUILD.md`, `AGENTS.md` + repo structure
> Verdict: **Promising vision. Serious execution gaps. Keep reading.**

---

## Executive Summary

Kawkab AI is an AI football coach for amateur teams — 100% offline, bilingual (EN/AR),
local LLM via Ollama, computer vision via YOLOv11 + BoT-SORT. The idea is genuinely
differentiated in an underserved market. The domain knowledge embedded in the plan is
real. But after reading all five documents carefully, there's a widening gap between
what is *documented as complete* and what is *likely to actually work* on real amateur
footage. That gap needs to be addressed before anything else.

---

## Part 1 — What's Genuinely Good

### 1.1 The Vision is Correct
The amateur football coaching market is genuinely underserved. Professional teams have
Wyscout, ChyronHego, Veo, Hudl — all cloud-based, expensive, English-only. A private,
offline, Arabic-language tool at $0 is a real gap. This is the right problem to solve.

### 1.2 Domain Knowledge is Real
Whoever designed this stack knows football analytics. Picking PPDA (Passes Per Defensive
Action) as a pressing metric, using socceraction's SPADL format for action valuation,
implementing xT (Expected Threat) and VAEP — these aren't cargo-culted terms. The
tactical taxonomy in `knowledge/tactics/` (defensive, offensive, transitions, individual)
matches how coaches actually think. This is rare in AI sports projects.

### 1.3 Service Architecture is Clean
Eight async Python services behind a QWebChannel bridge is extensible and testable.
The separation between CVService, ReasoningService, KnowledgeService, and LLMService
means you can improve or swap any layer independently. That's good design.

### 1.4 Right Technology Choices at Each Layer
- **YOLOv11l + BoT-SORT**: state-of-the-art and the right combo for this use case
- **Ollama + Qwen 2.5 14B**: correct default LLM for an offline-first app in 2025-26
- **QWebChannel (no FastAPI)**: correct for a desktop app; FastAPI would add latency and
  complexity for no benefit
- **WeasyPrint for PDF**: smart — the UI is already HTML, so PDF export is nearly free
- **BGE-M3 for embeddings**: exactly right for Arabic + English multilingual RAG
- **kloppy + socceraction**: the right libraries for standardized football data pipelines
- **uv for package management**: fast, modern, correct choice

### 1.5 Privacy-First + Arabic = Real Differentiator
For the MENA market specifically, coaches won't upload match footage to a cloud server.
Privacy concerns are cultural and practical. Offline-by-default + Arabic UI is a genuine
moat that Western competitors won't prioritize. This is the strongest strategic insight
in the entire project.

---

## Part 2 — Critical Issues (Severity Ordered)

### 🔴 CRITICAL #1 — 160 Tracks for 22 Players Breaks Everything

The README reports 160 unique tracks from an 88-second clip.
A football match has exactly 22 outfield players.

**160 ÷ 22 ≈ 7.3x fragmentation rate.**

This means BoT-SORT is losing and re-assigning player identities every few seconds.
The downstream consequences are catastrophic and cascade through the entire analytics
stack:

| Stat | Effect of Fragmentation |
|------|------------------------|
| Possession % | Wrong — same player counted as multiple team IDs |
| Distance Covered | Wrong — each fragment has partial path |
| Player Speed | Wrong — short fragments spike velocity artificially |
| Pass Count | Wrong — pass sender/receiver IDs are inconsistent |
| xT / VAEP | **Completely invalid** — requires stable player IDs across the full sequence |
| Formation Detection | Distorted — 160 points ≠ 22 players in a k-means cluster |

Every number shown in the dashboard is statistically unreliable until this is fixed.
There is no Phase 2 or Phase 3 until Phase 1 tracking actually works.

**Root causes to investigate:**
- BoT-SORT's ReID model was likely trained on broadcast footage (stable high angle).
  Amateur footage from a touchline phone is a completely different distribution.
- The adaptive field masking may not be filtering referees and linesman correctly,
  adding extra "phantom" players to the track pool.
- Camera motion (phone shake) confuses the Global Motion Compensation (GMC) step.

**Fix:** Replace `boxmot`'s default ReID backbone with one fine-tuned on amateur footage,
or use `SoccerNet/tracklab` which has football-specific ReID models built in.

---

### 🔴 CRITICAL #2 — No Homography = All Spatial Stats Are Pixel Coordinates

xT, xG, formation positions, distance covered in meters, defensive line height in meters
— all of these require converting from **pixel space → metric pitch space**.

Without homography:
- "Player covers 1.2km" is actually pixel distance, not meters
- "xT = 0.07" is calculated from pixel (x, y), not from a 105×68m pitch position
- "Formation 4-3-3" is detected by clustering pixel coordinates, which are
  aspect-ratio-distorted by the camera angle and lens
- "Defensive line height = 35m" is meaningless without metric calibration

Homography is listed in **Phase 2** (Weeks 5-10 in PLAN.md). But the README claims
xT computation and formation detection are Phase 2 features marked as **complete**.
That is not possible without homography.

This is not an optional enhancement — it is a prerequisite for every spatial metric.
It should be in Phase 1, Week 1.

---

### 🔴 CRITICAL #3 — VRAM Budget Is Never Discussed, Will OOM on RTX 4070

The tech stack lists these GPU-resident models:

| Model | Estimated VRAM |
|-------|---------------|
| YOLOv11l (inference) | ~3–4 GB |
| BoT-SORT ReID (OSNet/ResNet) | ~0.5–1 GB |
| Qwen 2.5 14B Q4_K_M (via Ollama) | ~8–9 GB |
| Real-ESRGAN (Phase 1 toggle) | ~1–2 GB |
| Sports2D Pose (optional) | ~1–2 GB |

An RTX 4070 has 12 GB VRAM.

Running CV pipeline + LLM generation simultaneously = **~12–14 GB demand on a 12 GB
card**. This is an OOM crash waiting to happen, not a theoretical edge case.

The current solution implied by the code structure is to run services sequentially
(CV first, then LLM), which works, but is never explicitly documented or enforced in
the architecture. There is no VRAM manager, no model unloading/reloading strategy,
no priority queue for GPU resources.

Real-ESRGAN on a 90-minute match at 30fps = 162,000 frames of upscaling. At ~0.1s
per frame on an RTX 4070, that's **4.5 hours** before YOLO even starts. This makes
Real-ESRGAN impractical as a default on full matches. It should be sample-mode only
(enhance keyframes, not every frame).

---

### 🔴 CRITICAL #4 — README Has Internal Self-Contradictions

The README's **Features section** says Phases 1, 2, and 3 are all "Complete" (✅).
The README's **Roadmap table** at the bottom says:

- Phase 1: ✅ Done
- Phase 2: 🚧 In Progress
- Phase 3: 📋 Planned

The same document contradicts itself within the same file. PLAN.md is labeled v4.0.
The `pyproject.toml` still contains `github.com/yourusername/kawkab-ai` placeholder URLs.

This signals the content was AI-generated and not proofread before publishing. For a
public repo trying to build community trust, this is a credibility hit. Fix the
README first — pick one story and make it accurate.

---

### 🔴 CRITICAL #5 — One Commit Is Not a Repository

The entire project was bulk-uploaded in a single commit. This means:
- No build history. Can't see what was built incrementally.
- Can't bisect a bug. Can't see what changed between working and broken.
- Signals to contributors: this codebase was created in one session, not developed.
- Makes it impossible to verify "Phase 3 Complete" — there's no history of it being built.

A single commit is the signature of an AI-assisted code dump. That's not inherently bad
— the code may be good — but it actively undermines trust from external contributors
and validators. Start committing meaningfully from today forward.

---

### 🟡 IMPORTANT #6 — Missing `qasync` — The Qt + asyncio Bridge

PySide6 has its own event loop (Qt's). Python has asyncio. The 8 async services in the
architecture presumably use `async/await`. Running asyncio coroutines inside Qt's event
loop without a bridge **will block the UI thread** or raise `RuntimeError: no running
event loop`.

The standard solution is `qasync` (a library that patches Qt to run asyncio):

```python
import qasync
import asyncio
from PySide6.QtWidgets import QApplication

app = QApplication([])
loop = qasync.QEventLoop(app)
asyncio.set_event_loop(loop)
```

`qasync` is not in `pyproject.toml`. This is either a real gap, or it's handled via
`QThread` + thread pools (which is fine but different from async Python). Either way,
the architecture document should make this explicit.

---

### 🟡 IMPORTANT #7 — Jersey Number OCR Is Significantly Harder Than Stated

PLAN.md Locked Decisions: "Jersey Numbers: Auto-OCR from day 1."

Reality: Jersey number OCR on amateur footage is one of the hardest problems in sports
computer vision. Reasons:

- At typical touchline distances, jersey numbers are **8–20 pixels tall**
- EasyOCR requires ~30px minimum for reliable digit recognition
- Numbers are often obscured by arms, opponent players, motion blur
- Amateur jerseys use varying fonts, colors, shading — no standardization
- Even the cited `SoccerNet/tracklab` achieves ~65–70% accuracy on broadcast footage
  (which is far better quality than amateur)

The plan mentions "YOLOv11 fine-tuned on jersey numbers" but this model doesn't exist
in `pyproject.toml` and would require labeled training data of amateur jersey numbers —
a months-long effort.

Rating this risk as 🟢 Low in the risk matrix ("Manual confirmation UI, improves with
corrections") is optimistic to the point of being misleading. It should be 🟡 Medium
at minimum. The manual correction UI is the right fallback, but "Auto-OCR from day 1"
sets a wrong expectation.

---

### 🟡 IMPORTANT #8 — Monetization Plan Has a Structural Blind Spot

PLAN.md Phase 6: "Payment: Stripe"

Stripe does not support Tunisia as a home country for Stripe accounts. A Tunisian
developer cannot create a Stripe account that accepts payments into a Tunisian bank.
The Risk Matrix rates Monetization as 🟢 Low risk — this underestimates a real
structural constraint.

Viable alternatives that work in or around Tunisia: Lemon Squeezy (supports global
payouts), Gumroad (limited), Paddle (merchant of record model), or routing through
a legal entity in a Stripe-supported country. This needs a real plan, not a Stripe
assumption.

---

### 🟡 IMPORTANT #9 — FAISS Is Wrong for This Scale

FAISS is a billion-scale approximate nearest neighbor search library from Meta AI.
It is being used here as a key-value lookup for 22 tactical rules.

This is like using a freight train to commute to work. FAISS requires:
- Manual index file management (`faiss.write_index` / `faiss.read_index`)
- Manual synchronization between SQLite (structured data) and FAISS (vectors)
- No metadata filtering without building your own wrapper

For 22 rules → 5,000 rules, a purpose-built vector DB handles all of this natively.
`ChromaDB` in embedded mode is a full drop-in replacement that adds metadata filtering,
persistence, and deletion — all things FAISS doesn't do by default.

---

### 🟡 IMPORTANT #10 — The Bundle Size Will Hurt Adoption

BUILD.md confirms: bundle size is 1.5–2.5 GB, compressed to ~800 MB for the installer.

For the target audience (amateur coaches in MENA):
- Average Tunisian/Moroccan home internet: 10–30 Mbps
- Download time for 800 MB installer: 4–10 minutes
- Uncompressed on-disk: 1.5–2.5 GB takes up significant space on a laptop
- Windows SmartScreen will flag an unsigned binary (confirmed in BUILD.md)
- Code signing costs $200–500/year for an EV certificate

The "compressed to ~800 MB" estimate is optimistic — PyTorch alone unpacked is ~700 MB.
Consider model-lazy-loading: ship a small 50 MB launcher, download YOLO and Qwen models
on first run. Many major AI desktop apps (LM Studio, Jan.ai) use this pattern.

---

## Part 3 — Architecture Deep Dive

### 3.1 The QWebChannel Bridge Is a Good Choice But Has Latency

The `JS ↔ Python` bridge via QWebChannel serializes all calls through JSON messages.
For UI interactions (button click → start analysis), this is fine. For real-time video
playback with overlays (overlay positions updating at 30fps), JSON serialization overhead
will be visible. Consider using shared memory or a direct pixel buffer for frame overlays
rather than routing them through QWebChannel.

### 3.2 The LLM Context Window Will Limit Report Quality at Scale

Ollama + Qwen 2.5 14B has a 32K context window (or 128K with extended config). A 90-
minute match generates:
- 162,000 frames of raw CV data
- Thousands of detected events
- 22 player track histories

Feeding all of this raw into an LLM prompt will exceed the context window. The plan
mentions "scoped context via FAISS" in Phase 3 — this is the right idea (RAG over
match events), but it needs to be implemented carefully to avoid losing context about
critical moments (the goals conceded, the pressing failures).

### 3.3 The SQLite Schema Will Determine Everything

None of the service files were readable in this review (single commit, no GitHub
rendering of the `src/kawkab/` tree). But the choice of SQLite as the primary store
is correct for an offline app. The critical design question: does the schema normalize
`player_id → match_id → events → positions` cleanly, or does it store raw YOLO outputs
as JSON blobs? If the latter, querying becomes painful as the knowledge base grows.

### 3.4 The Python Async + Qt Async Problem

The 8 async services need careful event loop management:

```
Qt Event Loop (PySide6)
    └── QWebChannel (async message passing)
          └── Python asyncio (service calls)
                └── YOLO inference (blocking, GPU thread)
                └── Ollama HTTP API (async HTTP)
```

The blocking YOLO inference call **must not** run on the Qt main thread or asyncio
event loop directly — it will freeze the UI. It needs to run in a `concurrent.futures`
thread pool or a dedicated process. This is solvable but requires explicit design.

---

## Part 4 — What AGENTS.md Actually Tells You

The `AGENTS.md` file isn't an AI agent configuration. It's instructions for `graphify`
— a code knowledge graph tool that builds a graph of the codebase and lets you query
it with `graphify query "what does CVService do?"`. This is a developer productivity
tool, not part of the product.

The presence of `.opencode` directory and `.graphifyignore` file tells us this project
was developed using **opencode** (an AI-assisted terminal coding tool) as the primary
IDE. This explains the single commit: opencode generates code in sessions and the
developer bulk-committed at the end.

This is not a criticism — opencode + graphify is a legitimate and increasingly common
development workflow. But it does explain the gap between plan richness and commit history.

---

## Part 5 — Repository Recommendations

### 5.1 Fix the 160-Tracks Problem First

**[mikel-brostrom/boxmot](https://github.com/mikel-brostrom/boxmot)**

The authoritative multi-tracker Python library. Wraps ByteTrack, StrongSORT, BoT-SORT,
DeepOCSORT, and others under one API. More importantly, it ships with multiple ReID
backbones (OSNet, ResNet50, ClipReID) that you can benchmark against each other on
your own footage to find which one fragments least. Ultralytics-compatible. This is
what you should be using as the foundation, not vanilla BoT-SORT.

```python
from boxmot import StrongSort
tracker = StrongSort(reid_weights=Path("osnet_x0_25.pt"), device="cuda:0")
```

**[SoccerNet/tracklab](https://github.com/SoccerNet/tracklab)**

SoccerNet's own tracking pipeline, purpose-built for football. Includes:
- Football-specific ReID (trained on SoccerNet-ReID dataset — real football footage)
- Homography estimation (pitch keypoint detection → projection matrix)
- Jersey number detection and OCR (their own fine-tuned model)
- Plug-in architecture so you can swap tracker backends

This alone would solve problems #1, #2, and partially #7 from the Critical section.

---

### 5.2 Homography (Required for All Spatial Stats)

**[SoccerNet/sn-calibration-geometry](https://github.com/SoccerNet/sn-calibration-geometry)**

SoccerNet's camera calibration toolkit. Detects pitch keypoints (penalty spots, center
circle, corner arcs) and computes the homography matrix from camera view → top-down
pitch view. This is the missing link that makes every spatial metric meaningful.

Without this, you can't compute xT, real distances, or accurate formations. With it,
every metric jumps from "pixel estimate" to "ground truth in meters."

```python
# After calibration, convert any pixel (px, py) to pitch coords (mx, my) in meters
pitch_point = H_inv @ np.array([px, py, 1])
mx, my = pitch_point[0]/pitch_point[2], pitch_point[1]/pitch_point[2]
```

---

### 5.3 Replace FAISS with Something Purpose-Built

**[chroma-core/chroma](https://github.com/chroma-core/chroma)**

Runs fully offline in embedded mode (no server needed). Drop-in replacement for FAISS
with added metadata filtering, persistent storage, and built-in deletion — all the
things FAISS doesn't do natively.

```python
import chromadb
client = chromadb.PersistentClient(path="./knowledge_db")
collection = client.get_or_create_collection("tactical_rules")
collection.add(documents=rules, embeddings=embeddings, ids=rule_ids)
results = collection.query(query_texts=["poor pressing in midfield"], n_results=5)
```

This also solves the SQLite ↔ FAISS sync problem. Chroma handles its own persistence.

---

### 5.4 Ground Truth Data for Benchmarking

**[statsbomb/statsbombpy](https://github.com/statsbomb/statsbombpy)**

Free event data from 900+ matches with full location data. Use this to:
- Benchmark your xG model against StatsBomb's ground truth
- Validate pass detection rates
- Build and test tactical rules against real-world data patterns
- Measure how far your amateur model drifts from professional ground truth

This doesn't require a video. It's structured JSON event data with pitch coordinates
that lets you validate the entire analytics pipeline independently of the CV layer.

**[eddwebster/football_analytics](https://github.com/eddwebster/football_analytics)**

The most comprehensive collection of football analytics resources, datasets, papers,
and tutorials. Already listed in your PLAN.md resource section — but lean into it
harder. Many of the 500-rule knowledge base items can be seeded from the academic
literature here.

---

### 5.5 For the Knowledge Base at Scale

**[run-llama/llama_index](https://github.com/run-llama/llama_index)**

When you have 500 rules and 500 drills, simple embedding search won't be enough.
LlamaIndex provides structured retrieval over heterogeneous knowledge sources (YAML
files, PDFs, structured data) with Ollama as the backend — fully offline.

This would replace the manual "FAISS + scoped context" approach with a proper RAG
pipeline:
- Index all YAML rules and drills at startup
- When reasoning about a match problem, query the index: "Show me all defensive
  pressing rules that apply when PPDA > 12"
- Pass the retrieved context to Qwen 2.5 14B for narrative generation

**[PySport/kloppy](https://github.com/PySport/kloppy)**

Already in your deps — but under-leveraged. Kloppy can standardize your internally
computed events into SPADL (Soccer Action Data Language) format, which socceraction
then uses for VAEP/xT. The bridge is:

```
Raw YOLO events → kloppy SPADL → socceraction VAEP → LLM narrative
```

Without kloppy normalizing the intermediate format, the socceraction integration
is fragile.

---

### 5.6 Video and Reporting

**[Zulko/moviepy](https://github.com/Zulko/moviepy)**

For Phase 4 clip extraction with overlays — much higher-level than raw ffmpeg-python.
Compositing text, arrows, and player highlights onto video clips for the "evidence
player" feature is 10x easier with moviepy than writing FFmpeg filter graphs manually.

**[davidpagnon/Sports2D](https://github.com/davidpagnon/Sports2D)** *(use cautiously)*

Already listed as optional — keep it that way. Pose estimation on every player in every
frame of a 90-minute match is computationally prohibitive. Consider using Sports2D only
on identified keyframe clips (e.g., a 5-second window around a goal), not as a default
pass.

---

### 5.7 Multilingual Arabic Support

**[google-research/arabic-bert](https://github.com/google-research/bert)** → prefer:
**[CAMeL-Lab/CAMeLBERT](https://github.com/CAMeL-Lab/camel_tools)**

For Arabic-specific NLP beyond what BGE-M3 gives you. If the LLM report generation
in Arabic needs to understand dialectal Arabic (Darija, Tunisian, Gulf) vs MSA
(Modern Standard Arabic), BGE-M3 alone won't distinguish them. CAMeL Tools handles
Arabic-specific tokenization, morphology, and dialect identification.

For coach feedback in Arabic, the coaching terminology varies heavily by region.
A Tunisian coach's vocabulary differs significantly from a Saudi or Egyptian coach's.
This is a product depth opportunity — not a Day 1 requirement.

---

### 5.8 Awesome Lists Worth Bookmarking

- **[diegopastor/awesome-football-analytics](https://github.com/diegopastor/awesome-football-analytics)** — most comprehensive curated list
- **[JanVanHaaren/soccer-analytics-resources](https://github.com/JanVanHaaren/soccer-analytics-resources)** — academic + practitioner mix
- **[moose-lab/awesome-sports-ai](https://github.com/moose-lab/awesome-sports-ai)** — broader sports CV context
- **[SoccerNet/SoccerNet-Benchmarks](https://github.com/SoccerNet)** — all SoccerNet challenge papers + baselines

---

## Part 6 — Prioritized Action Plan

This is not the PLAN.md's 6-phase roadmap. This is what should actually happen first
given the current state of the code.

### Week 1–2: Make the Foundation Honest
1. Fix the README — remove the Phase 3 "Complete" checkboxes or reconcile with the roadmap table
2. Fix pyproject.toml placeholder URLs
3. Start committing incrementally from now on (even small commits)
4. Add `qasync` to dependencies or document the threading model explicitly

### Week 3–6: Fix Tracking (Nothing Works Without This)
1. Swap to `boxmot` and benchmark StrongSORT vs BoT-SORT vs ByteTrack on YOUR footage
2. Integrate `SoccerNet/tracklab` for football-specific ReID
3. Target: < 30 unique tracks for a 22-player match clip
4. Add homography from `sn-calibration-geometry` — convert all positions to meters

### Week 7–10: Validate the Stats
1. Run socceraction xT/VAEP on calibrated, correctly tracked data
2. Cross-validate possession % against human manual count on a test clip
3. Cross-validate distance covered against a known GPS value if available
4. Document real-world accuracy metrics, not assumed ones

### Week 11–14: Replace FAISS, Seed the Knowledge Base
1. Migrate from FAISS to ChromaDB embedded mode
2. Write 50 real tactical rules (not placeholders) in Arabic + English
3. Write 30 real training drills with progressions
4. Integrate LlamaIndex for structured RAG over the knowledge base

### Week 15–20: Get 5 Real Coaches Using It
1. Ship v0.1.0 only after tracking is working
2. Recruit 5 amateur coaches in Tunisia, Morocco, or Algeria
3. Get them to analyze a real match with the tool
4. The feedback from this session is worth more than 6 months of solo development

### Month 6+: Revisit Monetization
1. Research Lemon Squeezy, Paddle, or legal entity setup for Tunisia
2. Don't build payment infrastructure until you have 50+ active users
3. The freemium model is right — validate the free tier first

---

## Part 7 — The Honest Verdict

### What This Project Is

A well-researched, intelligently designed plan for an underserved problem, built with
the right technology choices at every layer — **that has not yet been validated on real
amateur footage in the real world.**

The PLAN.md is v4.0. The README claims three phases complete. The pyproject.toml
references the right libraries. The architecture diagram is clean. The football
analytics knowledge is genuine.

But the repo has one commit, 160 tracks where there should be 22, no homography, FAISS
for 22 rules, and a Stripe plan that won't work in Tunisia.

### What This Project Could Be

The moat described in PLAN.md — "WHAT happened → WHY it happened → HOW TO FIX it
→ with video proof → in Arabic → offline" — is real and defensible. No other tool
combines all of these for amateur teams. But the moat only exists if:

1. Tracking is reliable enough that coaches trust the numbers (fix #1 first)
2. Arabic reports sound like a real coach, not a translated transcript
3. The knowledge base has enough rules/drills that recommendations feel specific

### The One Thing That Matters Most Right Now

Get one real amateur coach to upload one real match video and produce one real report
that the coach looks at and says *"yes, that's accurate, I'll use this."*

Everything else — the 6 phases, the 500 rules, the B2B licensing, the basketball
support, the multi-camera setup — is irrelevant until that single moment happens.

The vision is good. The technical ambition is appropriate. The plan is too optimistic
about what's already working. Close that gap and the project has a real shot.

---

*Review based on public repository state as of June 2026. Single commit makes code-level
review of service implementations, tests, and UI components impossible — only
architecture, plans, dependencies, and build config were available.*
