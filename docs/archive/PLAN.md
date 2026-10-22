# Kawkab AI — Product Plan

## Product Vision

A **private, offline-first professional football analytics platform** that runs entirely on local hardware (GPU RTX 4070). Combines computer vision-based tracking, advanced event analytics (xG/xT/VAEP/PSxG), 3D match visualization, player scouting, and wearable device integration — all without sending data to the cloud.

---

## Competitive Analysis

| Feature Area | Kawkab AI | Hudl | Wyscout | StatsBomb | ScoutAI |
|---|---|---|---|---|---|
| **100% Offline** | ✅ Full local processing | ❌ Cloud-only | ❌ Cloud-only | ❌ Cloud API | ❌ Cloud |
| **Auto Tracking** | ✅ YOLO + DeepOCSORT + BoxMOT | ✅ (subscription) | ❌ | ❌ | ✅ |
| **Event Detection** | ✅ 16+ event types | Manual tagging | Manual tagging | ❌ | Manual tagging |
| **xG / xT / VAEP** | ✅ All 3 + PSxG + EPV | ❌ | ✅ (basic) | ✅ (advanced) | ❌ |
| **3D Pitch Viz** | ✅ Three.js + hybrid dot/card | ❌ | ❌ | ❌ | ❌ |
| **Player Scouting** | ✅ External APIs + FIFA-style cards | ✅ (limited) | ✅ | ✅ | ✅ |
| **Wearable Integration** | ✅ Catapult/STATSports/Polar/FIT/TCX | ❌ | ❌ | ❌ | ❌ |
| **Video Tagging** | ✅ Coding workspace + timeline | ✅ | ❌ | ❌ | ✅ |
| **FIFA-style Cards** | ✅ FUT card rendering on pitch + in scouting | ❌ | ❌ | ❌ | ❌ |
| **Full Event Analytics** | ✅ 40+ modules, all tested | ❌ | ✅ | ✅ | ❌ |
| **Price** | **Free** (self-hosted) | $$$ | $$$$ | $$$ | $$ |

**Key Differentiators:**
- Only platform combining **video tracking + advanced analytics + 3D viz + wearable integration** in one offline app
- **FIFA-style card rendering** for intuitive player representation — makes analytics feel like a game
- 904+ analytical tests pass (100% deterministic)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Frontend (PySide6 QWebEngineView)            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │ app.js   │ │ app-3d.js│ │app-scout │ │app-coding│ │app-charts│ │
│  │ (core)   │ │(Three.js)│ │ (scout)  │ │ (tagging)│ │(Chart.js)│ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ │
│       │            │            │            │            │        │
│  ┌────▼────────────▼────────────▼────────────▼────────────▼─────┐  │
│  │                  QWebChannel Bridge (bridge.py)              │  │
│  │  7 handlers: analysis, coding, export, video, storage,      │  │
│  │  external, lifecycle — all rate-limited                      │  │
│  └────────────────────────────┬─────────────────────────────────┘  │
└───────────────────────────────┼─────────────────────────────────────┘
                                │
┌───────────────────────────────┼─────────────────────────────────────┐
│                     Backend (Python)                               │
│  ┌──────────────┐  ┌──────────▼──────┐  ┌──────────────────────┐   │
│  │ Core Modules │  │ Services Layer  │  │ Storage (SQLite)     │   │
│  │  xg_model    │  │  cv_service     │  │  events              │   │
│  │  xt_model    │  │  ball_tracker   │  │  matches/players     │   │
│  │  vaep        │  │  track_smoother │  │  coding_tags         │   │
│  │  pitch_ctrl  │  │  clip_extraction│  │  wearable_sessions   │   │
│  │  obv/carry_xt│  │  external_apis  │  │  collab_comments     │   │
│  │  win_prob    │  │  (7 providers)  │  │  +19 migrations      │   │
│  │  +35 more    │  │  wearables/     │  └──────────────────────┘   │
│  └──────────────┘  │  (5 parsers)    │                             │
│                    └─────────────────┘                             │
└────────────────────────────────────────────────────────────────────┘
```

---

## Roadmap to v1.0

### v0.14.0 (Current — 3D Pitch + Scouting V2)
- ✅ 3D pitch visualization with Three.js
- ✅ Hybrid dot/card player rendering  
- ✅ Enhanced scouting with FUT-style player cards
- ✅ Wearable migration 020 + FIT/TCX tests
- ✅ PLAN.md

### v0.15.0 — Production Tracking
- [ ] Full-match production tracking (fix YOLO ceiling ~17 tracks via tiling/inference tuning)
- [ ] Per-frame position persistence in DB (migration 021)
- [ ] Camera cut detection across whole match
- [ ] Goal direction heuristic validation

### v0.16.0 — Cloud Sync (Optional)
- [ ] End-to-end encrypted sync adapter
- [ ] Match data export/import via encrypted bundles
- [ ] Multi-device support

### v1.0.0 — Release
- [ ] Complete test coverage (100% pass rate on `pytest tests/unit/ -q`)
- [ ] Performance profiling report
- [ ] User documentation & onboarding wizard
- [ ] Windows installer build

---

## Module Inventory

### Core Analytics (40+ modules)
xg_model, xt_model, vaep, epv, pitch_control, ball_physics_pitch_control, win_probability, psxg_model, xa_model, carry_xt, obv, goals_added, finishing_analysis, through_ball, defensive_xt, dominance_index, space_control, xg_chain, set_piece_analysis, set_piece_xt, crossing_xg, xg_calibration, xa_split, pressing_clusters, role_classifier, model_comparison, game_state, transitions, momentum, pass_network, formation_analysis, heatmap, physical_metrics, injury_risk, player_rating, player_similarity, scout_reports, scout_report_upgrade, player_search, form_analysis, fixture_difficulty, league_simulation, squad_valuation, referee_analysis, suspension_tracker, phase_xg, build_up, territory_value, match_report, match_anomaly_detection, event_schema, report_templates

### Frontend Modules
app.js (core), app-3d.js (3D pitch), app-scout.js (scouting), app-coding.js (tagging), app-charts.js (Chart.js viz), app-squad.js, app-tactics.js, app-ai.js, app-highlight.js, app-offline.js, app-error-boundary.js, app-router.js, app-skeletons.js, app-tooltips.js, app-ux.js, app-perf.js, app-sparklines.js, app-data-providers.js, app-marketplace.js, app-opponent.js, tactical_sandbox.js, kawkab_animations.js, calibration.js, calibration_v2.js

### Services
cv_service, ball_tracker, track_smoother, clip_extraction, model_manager, detection_events, wearable_import_service, wearables/ (5 parsers), workload_service, injury_risk, physical_load_service, physiological_merge_service, external_api (7 providers), storage_service, shortlist_service, collaboration_service, llm_service, audio_service, data_export_service

### Migrations
001–020 covering: core tables, professional, face_embedding, video_clips, feedback, batch, benchmarks, validation, football_data, external_sources, api-football, indexes, audit, weather/cards, dedup, shortlist, contracts, coding_tags, collab, wearable_sessions

### Tests
904+ analytical tests + 101 frontend Jest tests = **1005+ total**

---

## Key Metrics

- **Analytical tests**: 904+ passing
- **Frontend tests**: 101 passing (135 total, 34 pre-existing Chart.js/ESM mock failures)
- **Core modules**: 40+ analytical modules
- **CLI commands**: 10 (track, batch, evaluate, render, events, possession, link-players, train-yolo, prepare-data)
- **External APIs**: 7 providers integrated
- **Wearable parsers**: 5 (Catapult, STATSports GPX/CSV, Polar HR, FIT, TCX)
- **Migrations**: 20
- **Total test count**: 1005+
