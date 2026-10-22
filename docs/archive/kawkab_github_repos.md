# Kawkab AI — Full GitHub Repository Reference

Every repo here maps to a specific problem in your project. Organized by category, with priority ratings and exact notes on what it fixes in your codebase.

Priority key:
- P0 = fixes your two critical gaps (ReID fragmentation + speed). Do these first.
- P1 = directly improves existing services
- P2 = adds missing professional features
- P3 = reference / research only

---

## CATEGORY 1: TRACKING & RE-IDENTIFICATION
> The core of your fragmentation problem (91 IDs for 22 players)

---

### P0 — SoccerNet/tracklab
**URL:** https://github.com/SoccerNet/tracklab
**Stars:** Active, CVPR-affiliated
**What it is:** The official plug-and-play tracking framework from SoccerNet. Modular tracker pipeline designed specifically for football. Used by the top teams in the Game State Reconstruction challenge at CVPR'24 and CVPR'25.
**What it fixes in Kawkab:** Drop-in replacement integration path for your BoT-SORT. Contains football-tuned ReID models trained on 340k+ player thumbnails. This is the single highest-impact integration you can make.
**How to integrate:** Replace your cv_service BoT-SORT call with tracklab's pipeline. Feed your YOLO detections into it. Their Tracker State system saves embeddings to disk so you don't recompute on reruns.
**Critical note:** Already named in your own README as "next integration." This is backlog item priority — not optional.

---

### P0 — SoccerNet/sn-reid
**URL:** https://github.com/SoccerNet/sn-reid
**Stars:** Official SoccerNet org
**What it is:** The official SoccerNet Re-Identification dataset and challenge kit. 340,993 player thumbnails extracted from broadcast video. Challenge-winning ReID models trained specifically on football players.
**What it fixes in Kawkab:** Your current BoT-SORT uses generic pedestrian ReID weights (trained on MOT17 / CrowdHuman — people on sidewalks). Football players share kit colors, move similarly, and occlude constantly. A model trained on 340k football player crops will cut your 91-track fragmentation problem drastically.
**How to integrate:** Download their pretrained weights. Load them into your existing BoT-SORT via the appearance_model parameter, or switch to boxmot (see below) which handles this natively.

---

### P0 — SoccerNet/sn-gamestate
**URL:** https://github.com/SoccerNet/sn-gamestate
**Stars:** CVPR'24 workshop paper
**What it is:** End-to-end system for reconstructing game state on a 2D minimap from broadcast video. Combines player detection, team classification, jersey number recognition, ReID, and pitch calibration into one unified pipeline.
**What it fixes in Kawkab:** This is the reference architecture for everything your app is trying to do. Study it as the gold standard. Their minimap output is exactly what your homography + tracking pipeline should produce. Uses uv for install (same as you).
**How to integrate:** Do not replace your pipeline with it. Use it as a benchmark — run it on the same Sweden-Tunisia clip you use for your own tests and compare output quality directly.

---

### P0 — mikel-brostrom/boxmot
**URL:** https://github.com/mikel-brostrom/boxmot
**Stars:** 5,700+
**What it is:** The definitive pluggable multi-object tracking library for Python. Supports BoTSORT, ByteTrack, DeepOCSORT, StrongSORT, OcSort, ImprAssoc, BoostTrack, and more — all with a single API. Includes automatic download of ReID weights (OSNet, LightMBN, CLIPReID).
**What it fixes in Kawkab:** You are currently locked to one tracker (BoT-SORT). boxmot gives you swap-in access to every SOTA tracker in one line of code. Critical for your backlog item #21 (document known failure modes per tracker) — with boxmot you can benchmark all trackers on the same video and pick the best one for football.
**How to integrate:** Replace your BoT-SORT init with `from boxmot import BoTSORT` — same algorithm, but now you can also try `DeepOCSORT` (best ReID quality) or `ByteTrack` (fastest) as drop-ins. Supports CUDA fp16 for speed.
**Speed benefit:** boxmot includes native C++17 implementations of BoTSORT and ByteTrack, which are 2-3x faster than the Python equivalent — directly attacks your processing speed problem.

---

### P1 — NikolasEnt/soccernet-calibration-sportlight
**URL:** https://github.com/NikolasEnt/soccernet-calibration-sportlight
**Stars:** 1st place CVPR'23 Camera Calibration Challenge
**What it is:** 1st-place solution for the SoccerNet Camera Calibration Challenge 2023. Uses HRNetV2 to detect 57 pitch keypoints (lines + ellipses + tangent points) and compute homography automatically with zero manual input.
**What it fixes in Kawkab:** Your PitchDetector (Hough transform) is built but unvalidated. This is a working, benchmarked, CVPR-winning auto-calibration implementation. Directly implements your backlog item #5 (auto-homography from pitch keypoints).
**How to integrate:** Use their keypoint detection model as the backbone for your PitchDetector. Their model outputs the pitch keypoints → your existing homography_service computes the matrix from them. Zero manual clicking.

---

### P1 — roboflow/sports
**URL:** https://github.com/roboflow/sports
**Stars:** Official Roboflow repo, actively maintained 2025
**What it is:** Roboflow's official computer vision + sports toolkit. Includes pitch keypoint detection (32-point soccer field model), player detection dataset, ball detection dataset, homography via keypoints, minimap projection, speed estimation.
**What it fixes in Kawkab:** Three things: (1) their pitch keypoint model (`football-field-detection-f07vi`) is pre-trained and free to use — use it for your auto-homography path; (2) their ball detection model is a direct upgrade over your current YOLO ball class; (3) their speed estimation tutorial (perspective transform + meters/second) maps exactly to your distance underestimation problem.
**How to integrate:** `pip install git+https://github.com/roboflow/sports.git`. Their `football.py` module has pitch keypoint detection + homography computation in ~50 lines.

---

### P1 — roboflow/supervision
**URL:** https://github.com/roboflow/supervision
**Stars:** 28,000+
**What it is:** The standard Python library for computer vision annotation, visualization, and tracking utilities. ByteTrack integration, annotators, DetectionDataset, zone analysis, speed estimation, heatmaps.
**What it fixes in Kawkab:** Your VideoReviewService and VisualizationService are reinventing wheels that supervision already built and battle-tested. Their `sv.ByteTracker`, `sv.Heatmap`, `sv.TraceAnnotator`, `sv.PolygonZone` (for pressing zones) are all production-quality drop-ins.
**How to integrate:** `pip install supervision`. Use `sv.Detections.from_ultralytics(results)` to bridge your YOLO output to supervision's ecosystem.

---

## CATEGORY 2: FOOTBALL ANALYTICS & METRICS
> Improving your xG, xT, PPDA, and adding validated metric calculations

---

### P1 — ML-KULeuven/socceraction
**URL:** https://github.com/ML-KULeuven/socceraction
**Stars:** 500+ — academic gold standard
**What it is:** The reference Python library for football action valuation. Implements xT (Expected Threat), VAEP (Valuing Actions by Estimating Probabilities), and Atomic-VAEP — all trained on real StatsBomb/Wyscout data. From the KU Leuven research group that invented xT.
**What it fixes in Kawkab:** Your `compute_xt_simple()` uses a 4x4 heuristic grid. Socceraction's xT model uses a 12x8 grid trained on 100,000+ real actions. This is the difference between a guess and a validated model. Replace your xT computation with socceraction's trained weights.
**How to integrate:** `pip install socceraction`. Load StatsBomb open data → train their xT model → export weights → import in your AnalysisService. One-time setup, permanent improvement.

---

### P1 — statsbomb/statsbombpy
**URL:** https://github.com/statsbomb/statsbombpy
**Stars:** Official StatsBomb Python library
**What it is:** Official Python library for loading StatsBomb open data into pandas DataFrames. Covers 50+ competitions, 3,400+ events per match.
**What it fixes in Kawkab:** Your ValidationService needs ground truth to compare against. StatsBomb open data IS the ground truth. Load their event data for the same matches your CV pipeline processes and compare your detected events against theirs. This closes backlog item #19 (ground-truth eval set).
**How to integrate:** `pip install statsbombpy`. `from statsbombpy import sb; events = sb.events(match_id=3788741)`. Free, no API key needed.

---

### P1 — statsbomb/open-data
**URL:** https://github.com/statsbomb/open-data
**Stars:** 1,700+
**What it is:** Raw JSON files of StatsBomb event data. FIFA World Cup 2022 (64 matches with events AND 360 freeze frame data), UEFA EURO 2024, Champions League finals, and more. Free, no license required for research.
**What it fixes in Kawkab:** Training data for your xG model. Use their shot events (distance, angle, assist type, body part, freeze frame GK position) to train a proper logistic regression xG model instead of your current distance+angle heuristic. Backlog item #30.
**Data available:** FIFA World Cup 2022 — 64 matches with events + 360. UEFA EURO 2024 — 51 matches. Champions League finals — 15 matches.

---

### P1 — PySport/kloppy
**URL:** https://github.com/PySport/kloppy
**Stars:** 400+
**What it is:** The football data standardization library. Loads event data AND tracking data from multiple providers (StatsBomb, Wyscout, Opta, Metrica, SkillCorner, TRACAB) into a unified format.
**What it fixes in Kawkab:** Your DataExportService exports to CSV/JSON with your own schema. If you add kloppy compatibility, your data becomes importable by every tool in the football analytics ecosystem — analysts who already use kloppy can plug your data straight into socceraction, mplsoccer, or their own notebooks.
**How to integrate:** `pip install kloppy`. Use kloppy's serializers as the output format for your DataExportService. This is a 1-2 day integration with massive ecosystem compatibility payoff.

---

### P2 — andrewjohnsonsports/mplsoccer
**URL:** https://github.com/andrewjohnsonsports/mplsoccer
**Stars:** 700+ (already in your pyproject.toml)
**What it is:** Python library for creating football pitch visualizations — heatmaps, shot maps, pass maps, radar charts, pitch control maps.
**What it fixes in Kawkab:** You already have this. The question is whether your VisualizationService uses it fully. Their `Pitch.kdeplot()` produces proper kernel density heatmaps (not your current basic version). Their `VerticalPitch` is better for goal mouth shots. Their `Standardizer` handles coordinate system conversions between providers.
**Priority action:** Check if your heatmaps use `kdeplot` (smooth kernel density) or just scatter plots. If scatter, upgrade to kdeplot. 20-line change, massive visual quality improvement.

---

### P2 — soccer_xg / soccer-xg
**URL:** https://github.com/ML-KULeuven/soccer_xg
**Stars:** Academic
**What it is:** A Python package specifically for training and analyzing expected goals models. Provides multiple xG model architectures and their evaluation on StatsBomb data.
**What it fixes in Kawkab:** Directly replaces your `compute_xg_simple()` with a properly trained model. Includes logistic regression, gradient boosting, and neural network xG models. All training code included so you can retrain on your own data.

---

### P2 — Friends of Tracking (Python notebooks)
**URL:** https://github.com/Friends-of-Tracking-Data-FoTD
**Stars:** Multiple repos, 500-2000 each
**What it is:** YouTube + GitHub series by working data scientists from top European clubs teaching football analytics with code. The best free football analytics education that exists. Covers pitch control, pressure, xT, EPV, off-ball scoring opportunities.
**What it fixes in Kawkab:** PositioningService, PressureMetricsService, ReasoningService — your implementations of these were likely inspired by this community. Go back to the source and compare your implementations against theirs. The "Pitch Control" notebooks will directly improve your PositioningService off-ball run detection.
**Key repos:**
- `Friends-of-Tracking-Data-FoTD/LaurieOnTracking` — Laurie Shaw's (Harvard + Man City) tracking data tutorials
- `Friends-of-Tracking-Data-FoTD/SoccermaticsForPython` — David Sumpter's Soccermatics Python implementations

---

### P2 — JanVanHaaren/soccer-analytics-resources
**URL:** https://github.com/JanVanHaaren/soccer-analytics-resources
**Stars:** Curated master list
**What it is:** The definitive curated list of football analytics datasets, papers, and libraries maintained by Jan Van Haaren (UEFA data scientist). Every serious tool in the analytics ecosystem is listed here with descriptions.
**What it fixes in Kawkab:** Use it as your reference bible when choosing between competing implementations. When you need "should I use library X or Y for this metric", check here first.

---

### P3 — eddwebster/football_analytics
**URL:** https://github.com/eddwebster/football_analytics
**Stars:** 2,500+
**What it is:** Massive curated collection of football analytics projects, datasets, and analysis notebooks. The most comprehensive single resource in the community.
**What it fixes in Kawkab:** Research reference. When you want to understand how a specific metric is computed professionally, this is the first place to look.

---

## CATEGORY 3: PITCH CALIBRATION / HOMOGRAPHY
> Your auto-homography path — zero-click calibration (backlog #5)

---

### P0 — SoccerNet/sn-gamestate (tvcalib path)
**URL:** https://github.com/SoccerNet/sn-gamestate
**Notes:** Already listed above in tracking. The calibration component uses either "tvcalib", "pnlcalib", or "nbjw_calib" — all three are available as drop-in calibration backends via their `soccernet.yaml` config.

---

### P1 — niklasent/soccernet-calibration-sportlight
Already listed above. The keypoint-based calibration pipeline that won CVPR'23.

---

### P1 — roboflow/sports pitch keypoints
Already listed above. Their `football-field-detection-f07vi` model gives you 32 pitch keypoints from a single image with confidence scores. Filter by confidence > 0.5 → use for homography. No training required.

---

## CATEGORY 4: DATA SOURCES (FREE)

---

### P1 — SkillCorner/open-data
**URL:** https://github.com/SkillCorner/opendata
**What it is:** SkillCorner's free broadcast tracking dataset — 10 matches from the 2024/25 A-League with full positional tracking data derived from TV broadcast. This is the exact same technology SkillCorner sells to elite clubs, given away free for research.
**What it fixes in Kawkab:** Your ValidationService needs ground truth tracking data (not just event data) to validate your own tracking output. SkillCorner's data gives you real tracking ground truth — player positions 25 times per second from a real match — to compare against your YOLO+BoT-SORT output.

---

### P1 — metrica-sports/metrica-sports sample data
**URL:** https://github.com/metrica-sports/sample-data
**What it is:** Metrica Sports' free sample tracking data — two anonymized matches with full player tracking + synchronized event data. The industry reference dataset for tracking analytics research.
**What it fixes in Kawkab:** Same as SkillCorner — ground truth tracking data for your ValidationService. The Metrica data also comes with the pitch coordinate system pre-defined, so you can validate your homography output directly.

---

### P1 — sportsdataverse/footballR / worldfootballR
**URL:** https://github.com/JaseZiv/worldfootballR
**What it is:** Scraper for FBref, Understat, Transfermarkt, Sofascore. Gets clean per-player stats for any match.
**What it fixes in Kawkab:** Your ScoutingService needs historical data. Use worldfootballR-style scrapers to pull opponent stats before a match and feed them into your pre-match scouting report.

---

### P2 — soccerdata
**URL:** https://github.com/probberechts/soccerdata
**Stars:** 450+
**What it is:** Python package for scraping Club Elo, ESPN, FBref, Football-Data.co.uk, Sofascore, SoFIFA, Understat, WhoScored — all in a unified API.
**What it fixes in Kawkab:** Your 8 external data sources (football-data.org, TheSportsDB, etc.) are wired individually. soccerdata gives you all of them through one interface with caching, rate limiting, and standardized output schemas.

---

## CATEGORY 5: VISUALIZATION

---

### P1 — mplsoccer (already in your stack)
Already listed above. Make sure you're using `kdeplot` for heatmaps and the Pitch object properly.

---

### P1 — soccerplots
**URL:** https://github.com/Slothfulwave612/soccerplots
**What it is:** Pass sonars, radar charts, pizza charts, bumpy charts — all football-specific visualizations beyond what mplsoccer covers.
**What it fixes in Kawkab:** Your backlog item #29 (pass sonars). Pass sonars (polar plots showing pass directions and distances per player) are a standard professional tool that your VisualizationService doesn't have yet. soccerplots implements them in ~10 lines.

---

### P2 — unravelsports
**URL:** https://github.com/unravelsports/unravelsports
**What it is:** Graph Neural Network framework for football analytics. Converts tracking data into graph representations for ML models. From a research group doing GNN-based player valuation.
**What it fixes in Kawkab:** Future direction for your ReasoningService. Instead of rule-based tactical diagnosis, a GNN trained on tracking data can learn tactical patterns automatically. Not for right now — but know it exists for Phase 6.

---

## CATEGORY 6: SPECIFIC TOOLS FOR YOUR EXISTING SERVICES

---

### P1 — ultralytics/ultralytics
**URL:** https://github.com/ultralytics/ultralytics
**Stars:** 45,000+ — already in your stack
**Priority action:** Check your current YOLO11 version. In v0.12.0 status you mention YOLO26. Ultralytics regularly releases new versions. Make sure you're on the latest stable YOLO11 release, not an arbitrary version. The latest models have 15-25% better mAP on football footage than versions from 6 months ago.

---

### P1 — SoccerNet/sn-jersey
**URL:** https://github.com/SoccerNet/sn-jersey
**Stars:** Official SoccerNet org
**What it is:** The official jersey number recognition challenge. Fine-tuned OCR models specifically trained on jersey numbers in broadcast football footage. Your Jersey OCR is currently "unreliable" (your own STATUS.md words) because EasyOCR requires 30px minimum for 8-20px numbers.
**What it fixes in Kawkab:** Their models are trained on exactly the 8-20px jersey number range from broadcast video. Directly replaces your EasyOCR-based approach with a purpose-built model. Backlog item — currently marked "unreliable."

---

### P2 — SoccerNet/sn-spotting (action spotting)
**URL:** https://github.com/SoccerNet/sn-spotting
**What it is:** Event detection in football broadcast video — goals, cards, substitutions, fouls. The SoccerNet action spotting challenge and baseline models.
**What it fixes in Kawkab:** Your CardDetectionService exists but your shot/event detection is basic (velocity-based). Their models detect 17 event types from video directly, without needing homography or tracking — they work on the raw broadcast stream. Add as an optional high-quality event detection path.

---

### P2 — labeling tool: CVAT or Label Studio
**URL:** https://github.com/opencv/cvat
**What it is:** Professional video/image annotation tool. Used by professional sports analytics teams for ground truth labeling.
**What it fixes in Kawkab:** Your backlog item #19 (ground-truth eval set — 3+ matches with hand-tagged events). You need a way to annotate your own match videos with ground truth events so your ValidationService has something to compare against. CVAT is the standard tool for this.

---

## CATEGORY 7: PERFORMANCE & BUNDLE SIZE

---

### P0 — onnxruntime
**URL:** https://github.com/microsoft/onnxruntime
**Stars:** 15,000+
**What it is:** ONNX Runtime — cross-platform inference engine for ML models. Already likely in your stack transitively. The key for your bundle size: export YOLO models to ONNX and load with OnnxRuntime instead of PyTorch. ONNX models are typically 2-3x smaller than PyTorch .pt files and load faster.
**What it fixes in Kawkab:** Your 1.75 GB bundle is dominated by PyTorch + torchvision. If you export your YOLO models to ONNX and load them with OnnxRuntime (which is ~50 MB vs PyTorch's ~500 MB), your bundle drops significantly. Backlog item #50.

---

### P1 — nuitka
**URL:** https://github.com/Nuitka/Nuitka
**What it is:** Python-to-C compiler. Compiles your Python source into a standalone binary. Results in smaller executables than PyInstaller and faster startup.
**What it fixes in Kawkab:** Alternative to your current PyInstaller workflow if bundle size is still too large after model optimization. Not a priority until you've exhausted the ONNX + ModelManager lazy loading approach first.

---

## CATEGORY 8: REFERENCE ARCHITECTURES (STUDY THESE)

---

### abdullahtarek/football_analysis
**URL:** https://github.com/abdullahtarek/football_analysis
**Stars:** 2,000+ — the most-referenced football CV tutorial
**What it is:** Complete football analysis with YOLO + K-means + optical flow + perspective transform. Well-documented, heavily forked. Shows the exact pipeline your app implements.
**What it fixes in Kawkab:** Reference for your own architecture review. Compare their perspective transform implementation against your homography_service. Their optical flow camera motion compensation may be relevant to your tracking fragmentation problem.

---

### SkalskiP/sports
**URL:** https://github.com/SkalskiP/sports
**Stars:** Roboflow's Piotr Skalski — highly cited in sports CV
**What it is:** Experiments combining CV and sports. Includes GPT-4V for team color classification (exact problem you solved with K-means — their GPT approach is more robust), offside detection with pose estimation, and player tracking.
**What it fixes in Kawkab:** Their GPT-4V team assignment approach is more robust than K-means for edge cases (unusual kit colors, shadow, mud). Consider this for a future iteration of your team_assignment service.

---

### PySport community
**URL:** https://github.com/PySport
**What it is:** Open-source football analytics community. Maintains kloppy, unravelsports, and a comprehensive list of all open-source football projects.
**Why it matters for Kawkab:** This is the community you should publish to. When your ValidationService produces good results, share them here. When you need a collaborator who understands football data, this is where they are.

---

## PRIORITY ORDER FOR INTEGRATION (MATCHES YOUR PHASE ROADMAP)

| Phase | Repo | What it closes |
|---|---|---|
| Phase 2 — ReID | SoccerNet/sn-reid + SoccerNet/tracklab | Backlog #1, #4 + fragmentation gap |
| Phase 2 — ReID | mikel-brostrom/boxmot | Tracker flexibility + speed gain |
| Phase 2 — Speed | boxmot native C++ | Processing time on mid-tier GPUs |
| Phase 2 — Calibration | NikolasEnt/soccernet-calibration-sportlight | Backlog #5, auto-homography |
| Phase 2 — Calibration | roboflow/sports (keypoints model) | Backlog #5, zero-click path |
| Phase 3 — Validation | statsbomb/open-data + statsbombpy | Backlog #19, ground truth |
| Phase 3 — Validation | SkillCorner/open-data | Tracking ground truth |
| Phase 3 — Validation | metrica-sports/sample-data | Tracking ground truth |
| Phase 3 — Jersey | SoccerNet/sn-jersey | Jersey OCR reliability |
| Phase 4 — Metrics | ML-KULeuven/socceraction | Backlog #30, trained xT model |
| Phase 4 — Metrics | soccer_xg | Backlog #30, trained xG model |
| Phase 4 — Viz | soccerplots | Backlog #29, pass sonars |
| Phase 5 — Bundle | onnxruntime export | Backlog #50, bundle size |
| Phase 6 — Data | PySport/kloppy | Data export compatibility |
| Phase 6 — Data | soccerdata | Multi-source data scraper |
