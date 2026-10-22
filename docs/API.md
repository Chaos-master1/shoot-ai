# Kawkab AI API Reference

> **Updated:** Phase 4 (Sprint 5) — 96 service modules, 86 core modules, 8 bridge handlers documented.
> **Previous coverage:** ~25 services (26%) → **Now: 96 services + 86 core modules + 8 handlers (100%)**

---

## Module layout

```
kawkab/
├── services/           96 domain services (async/dataclass-based)
│   ├── analysis/        analysis subpackage (re-exported by analysis_service.py)
│   ├── cv_service.py   YOLO + tracker pipeline (core CV)
│   ├── storage/         SQLite persistence layer
│   └── ...              94 more service modules
├── core/               86 analytical engine modules (pure functions + dataclasses)
├── ui/
│   └── bridge_handlers/  8 handler classes for WebBridge slot methods
├── migrations/         SQL schema migrations (001-018)
├── i18n/               Translation support
└── web/                Browser-side assets (JS, CSS)
```

---

## Services (96 modules)

### 🎯 Analysis & Tactical Services

| Module | Class(es) | Key Methods | Description | Quality |
|--------|-----------|-------------|-------------|---------|
| `analysis_service.py` | AnalysisService | analyze_match, get_summary, get_player_stats, get_team_stats, get_formation, get_possession, get_xt, get_xg, get_pitch_control | Core match analysis — xG, xT, formations, possession, pitch control, line breaks | STABLE |
| `advanced_event_detection_service.py` | AdvancedEventDetectionService | detect_events, detect_dribbles, detect_tackles, detect_interceptions, detect_clearances, detect_crosses, detect_blocks, detect_duels, detect_carries, detect_progressive_actions, detect_high_turnovers, detect_set_pieces | Derives 15+ event types from tracking data without CV models | STABLE |
| `setpiece_service.py` | SetPieceService | analyze_set_pieces, get_corner_stats, get_free_kick_stats, get_throw_in_stats, analyze_delivery_type, find_first_contact, score_effectiveness | Corner/free kick/throw-in analytics with delivery classification | STABLE |
| `goalkeeper_service.py` | GoalkeeperService | analyze_goalkeeper, compute_save_rate, compute_xgot, compute_distribution_stats, find_sweeper_actions, analyze_claiming | Goalkeeper-specific metrics (xGOT, save rate, distribution) | STABLE |
| `substitution_service.py` | SubstitutionService | analyze_substitution_impact, compute_xg_delta, compute_possession_delta, compute_win_prob_impact | Substitution impact via xG/possession/win-probability deltas | STABLE |
| `possession_service.py` | PossessionService, PlayerPossessionStats, TeamPossessionStats | analyze_possession, get_possession_chains, get_player_touches, detect_counter_press, detect_tackles | Frame-by-frame possession with tackle/loss attribution and hysteresis | STABLE |
| `positioning_service.py` | PositioningService, RunClassification | analyze_runs, find_behind_runs, find_width_stretching, find_decoy_runs, compute_space_created | Off-ball run analysis (behind/width/decoy runs with xT impact) | STABLE |
| `psychology_service.py` | PsychologyService, ScoreState, MomentumPhase | analyze_psychology, detect_momentum_shifts, detect_post_goal_regression, analyze_late_game, detect_capitulation, compute_comeback_momentum | Score-state transitions, momentum windows, post-goal psychology | STABLE |
| `pressure_metrics_service.py` | PressureMetricsService, PPDAReport | compute_ppda, compute_passes_under_pressure, compute_counter_press_success, compute_defensive_line_height, compute_compactness, analyze_pressure_by_period | PPDA, pressure events, counter-press, defensive line metrics | STABLE |
| `football_rules_service.py` | FootballRulesService, Law | classify_event, detect_offside, get_restart_type, get_law_summary, get_event_penalty | Encodes IFAB Laws of the Game (17 laws) for event classification | STABLE |
| `card_detection_service.py` | CardDetectionService, CardEvent, CardType | detect_cards_visual, detect_cards_audio, detect_cards_tactical, detect_cards_external, fuse_detections, get_card_stats | Multi-source card detection: visual + audio + tactical + external | ALPHA |
| `reasoning_service.py` | ReasoningService, Diagnosis, DiagnosisReport | diagnose_match, generate_report, get_recommended_drills, rank_diagnoses | "Detective" layer — diagnosis engine using knowledge base rules | STABLE |
| `knowledge_service.py` | KnowledgeService, TacticalRule, Drill | get_rules_by_category, get_drills_by_focus, search_knowledge_base, get_rule_pattern_signature | YAML-based tactical rule/drill knowledge base (40 rule files, 24 drills — counted by scripts/check_docs_truth.py) | STABLE |
| `tactical_review_service.py` | TacticalReviewService | generate_tactical_review, generate_arabic_review | LLM-generated tactical analysis reports (EN/AR) via Ollama | BETA |
| `scouting_service.py` | ScoutingService, OpponentProfile | generate_scout_report, get_formation_tendencies, get_set_piece_tendencies, get_key_players, get_vulnerability_flags | Pre-match scouting reports from historical data | STABLE |
| `scouting_network_service.py` | ScoutingNetworkService, NetworkPlayer | search_network, add_player, rate_player, get_player_details | Community scouting network with anonymized player ratings | BETA |
| `shortlist_service.py` | ShortlistService | add_player, update_status, get_shortlist, search_shortlist, remove_player, get_stats | Recruitment shortlist with status/priority pipeline | STABLE |
| `contract_tracker.py` | ContractTracker | add_contract, get_contract, get_expiring_contracts, get_squad_summary, update_contract | Player contract management with expiry alerts | STABLE |
| `multi_match_analysis_service.py` | MultiMatchAnalysisService | analyze_season, compare_matches, get_performance_trends, get_team_evolution, get_opposition_analysis | Season-level aggregation and cross-match trending | STABLE |
| `player_development_service.py` | PlayerDevelopmentService, TrendDirection | analyze_trends, detect_improvement, detect_regression, get_trend_summary, compare_peers | Per-player trend tracking across matches with slope detection | STABLE |
| `game_plan_service.py` (core) | GamePlanAnalyzer | generate_game_plan, analyze_opponent, propose_tactics, set_piece_plan | Game plan scouting report generation | STABLE |

### 🤖 AI & Language Services

| Module | Class(es) | Key Methods | Description | Quality |
|--------|-----------|-------------|-------------|---------|
| `llm_service.py` | LLMService, LLMConfig, LLMProvider (ABC) | generate, generate_streaming, embed, count_tokens, list_models | Multi-provider LLM wrapper (Ollama, Groq, Google AI, OpenRouter) | STABLE |
| `reasoning_service.py` | ReasoningService | *(see Analysis section)* | Tactical diagnosis engine | STABLE |
| `tactical_review_service.py` | TacticalReviewService | *(see Analysis section)* | LLM match analysis | BETA |
| `ai_assistant_v2_service.py` | AIAssistantV2, Conversation, ConversationMessage | start_conversation, send_message, get_conversation_history, generate_tactical_suggestion, generate_report | Voice-capable AI coach with conversation history | BETA |
| `live_tagging_service.py` | LiveTaggingService, LiveTag | start_tagging, record_tag, get_session_tags, export_tags, get_stats | Real-time manual event tagging with keyboard shortcuts | BETA |

### 📡 External Data Services

| Module | Class(es) | Key Methods | Description | Quality |
|--------|-----------|-------------|-------------|---------|
| `api_football_service.py` | ApiFootballService | get_leagues, get_teams, get_fixtures, get_squad, get_standings, get_predictions, get_head_to_head | API-Football v3 (api-sports.io) — 100 req/day, free tier | STABLE |
| `statsbomb_service.py` | StatsBombService, SbCompetition, SbMatch | get_competitions, get_matches, get_events, get_lineups, get_360_frames | StatsBomb Open Data (free, attribution required) | STABLE |
| `football_data_service.py` | FootballDataService, TeamSearchResult | get_competitions, get_standings, get_fixtures, get_squad, search_team, get_head_to_head | football-data.org v4 — free tier (10 req/min) | STABLE |
| `bzzoiro_service.py` | BzzoiroService | get_live_scores, get_fixtures, get_predictions, get_shotmaps, get_standings, get_odds | sports.bzzoiro.com v2 — free, 65 leagues, Botola Pro coverage | STABLE |
| `thesportsdb_service.py` | TheSportsDBService, TeamResult | search_team, get_team_details, get_standings, get_venues, get_events | TheSportsDB v1 — 30 req/min, free API key | STABLE |
| `openfootball_service.py` | OpenFootballService | get_matches, get_standings, get_league_info, get_worldcup_matches | openfootball JSON repos — free, public domain, CC0 | STABLE |
| `easy_soccer_service.py` | EasySoccerService | get_matches, get_match_details, get_player_stats, get_incidents | EasySoccerData Sofascore wrapper — no API key needed | BETA |
| `transfermarkt_integration_service.py` | TransfermarktIntegrationService | get_market_value, get_squad_values, search_players, get_player_details | Transfermarkt market value import + squad data | BETA |

### 📹 Computer Vision & Tracking

| Module | Class(es) | Key Methods | Description | Quality |
|--------|-----------|-------------|-------------|---------|
| `cv_service.py` | CVService, MatchTrackData, FrameDetections | process_video, process_frame, initialize, get_track_data, get_ball_positions, get_team_assignment | YOLOv11 + BoT-SORT/Norfair + ReID pipeline (2133 lines) | STABLE |
| `pitch_detector.py` | PitchDetector, CalibrationGuess | detect_pitch, find_lines, get_corners, compute_confidence | CV-based pitch line detection (Hough transform, no GPU) | STABLE |
| `homography_service.py` | HomographyService, HomographyMatrix | calibrate_from_points, pixel_to_pitch, pitch_to_pixel, validate_homography, project_tracks | Pixel↔pitch coordinate conversion (3 modes: manual/auto/default) | STABLE |
| `lightglue_homography_service.py` | LightGlueHomographyService | auto_calibrate, propagate_homography, compute_error | SuperPoint + LightGlue ONNX auto-calibration | BETA |
| `camera_cut_detector.py` | CameraCutDetector | detect_cuts, get_cut_timestamps, get_segments | HSV histogram camera cut detection for broadcast footage | STABLE |
| `ball_tracker.py` | BallTracker | track_ball, predict_position, reset, get_trajectory | HSV + Kalman filter dedicated ball tracker (independent thread) | STABLE |
| `norfair_tracker.py` | NorfairTracker | update, reset, get_tracks, get_detections | Norfair-based tracking wrapper for football player + ball tracking | BETA |
| `tracker_base.py` | BaseTracker (ABC), TrackedObject, TrackerRegistry | update, reset, get_tracks | Abstract tracker interface for interchangeable backends | STABLE |
| `track_smoother.py` | TrackSmoother | smooth, smooth_batch | Rauch-Tung-Striebel (RTS) Kalman smoother for track jitter reduction | STABLE |
| `kalman_smoother.py` | PlayerPositionSmoother | smooth_position, get_velocity, reset | Single-player Kalman smoother for position trajectories | STABLE |
| `reid_feature_extractor.py` | ReidFeatureExtractor | extract_embedding, compare_embeddings, build_gallery | ResNet-50 CircleLoss for player re-identification | BETA |
| `face_recognition_service.py` | FaceRecognitionService | build_gallery, identify_player, link_across_matches, add_to_gallery | InsightFace ArcFace for player identification from crops | BETA |
| `jersey_service.py` | JerseyNumberService | detect_number, detect_numbers_batch, set_roster, match_to_roster | Jersey number detection (EasyOCR + pixel fallback + CNN future) | STABLE |
| `jersey_ocr.py` | JerseyOCR | read_number, read_numbers_batch, set_roster_mapping | EasyOCR-based jersey number OCR | STABLE |
| `pose_analysis_service.py` | PoseAnalysisService | analyze_pose, classify_activity, detect_fall, estimate_fatigue, compute_orientation | YOLO-pose keypoint analysis (17 COCO keypoints) | ALPHA |
| `roboflow_sports_service.py` | RoboflowSportsService | draw_pitch, draw_voronoi, annotate_ball, transform_view | roboflow/sports Python package wrapper | BETA |
| `tracking_metrics.py` | *module-level functions* | compute_tracking_self_metrics, compute_fragmentation, compute_id_switches | Intrinsic tracking quality metrics (no GT needed) | STABLE |

### 🏃 Physical & Physiological Services

| Module | Class(es) | Key Methods | Description | Quality |
|--------|-----------|-------------|-------------|---------|
| `physical_metrics.py` | *module-level functions* | compute_player_physical_profile, compute_team_physical_summary | Per-player distance, speed, sprint count, HIR from tracking data | STABLE |
| `physical_load_service.py` | PhysicalLoadService, PhysicalLoadMetrics | compute_load, get_sprint_profile, get_acceleration_profile, compute_work_rest_ratio, estimate_metabolic_power | Professional-grade load metrics (accelerations, metabolic power) | STABLE |
| `workload_service.py` | WorkloadService, RiskLevel | compute_acwr, compute_trimp, compute_session_rpe, get_injury_risk, get_workload_trend | ACWR, TRIMP, sRPE — injury-risk monitoring | STABLE |
| `periodization_service.py` | PeriodizationService, CyclePhase | analyze_weekly_load, detect_taper, detect_fixture_congestion, compute_recovery_index, detect_peaking | Multi-week training macrocycle planning | STABLE |
| `physio_tactical_correlation.py` | PhysioTacticalCorrelation | correlate_physio_tactical, compute_pre_post_event_stats, detect_fatigue_periods | Correlates physiological data with tactical events | BETA |
| `physiological_merge_service.py` | PhysiologicalMergeService | merge_video_wearable, synchronize_streams, compute_merged_metrics | Merges video-derived metrics with wearable sensor data | BETA |
| `wearable_import_service.py` | WearableImportService | import_csv, import_json, import_fit, import_tcx, get_timeseries | Import GPS/HR data from wearable devices (CSV, JSON, FIT, TCX) | BETA |
| `fatigue_model.py` (core) | FatigueModel | estimate_fatigue, predict_performance_decline | Fatigue estimation from physical metrics | BETA |

### 🎬 Video & Media Services

| Module | Class(es) | Key Methods | Description | Quality |
|--------|-----------|-------------|-------------|---------|
| `video_review_service.py` | VideoReviewService, Annotation, Tag, Clip | add_annotation, remove_annotation, add_tag, create_clip, get_frame_annotations | Frame-accurate clip annotation with drawing tools + tags | STABLE |
| `clip_service.py` | ClipExtractionService | extract_clip, extract_clips_batch, get_clip_path, clear_cache | FFmpeg-based video clip extraction with pre/post padding | STABLE |
| `clip_extraction_service.py` | ClipLibraryService | create_playlist, add_to_playlist, generate_thumbnail, export_clip, get_playlists | Higher-level clip library with thumbnails, playlists, DB storage | STABLE |
| `highlight_reel_service.py` | HighlightReelService, ReelClip, ReelResult | create_reel, add_clips, render, get_reel_info | Automatic highlight reel compilation from event clips | BETA |
| `video_sync_service.py` | MultiAngleSyncService | add_source, sync, get_sync_state, align_timelines | Multi-angle video synchronization | BETA |
| `telestration_service.py` | TelestrationService, TelestrationLayer | create_telestration, add_element, render_overlay, save_preset, load_preset | Drawing/annotation overlay for coaching telestration | BETA |
| `live_stream_service.py` | LiveStreamCaptureService | start_capture, stop_capture, detect_stream_type, list_active_streams | Live stream capture from m3u8/RTMP/YouTube/Twitch | BETA |
| `enhancement_service.py` | EnhancementService | enhance_video, stabilize, denoise, upscale, interpolate | FFmpeg filters + Real-ESRGAN + RIFE video enhancement | BETA |
| `audio_service.py` | AudioService | transcribe, detect_whistle, analyze_crowd, get_audio_timeline | faster-whisper transcription + whistle/crowd detection | BETA |

### 💾 Storage & Persistence

| Module | Class(es) | Key Methods | Description | Quality |
|--------|-----------|-------------|-------------|---------|
| `storage_service.py` | StorageService | save_match, get_match, save_events, get_events, save_players, update_event, delete_event, save_feedback, get_feedback, save_advanced_metrics, get_advanced_metrics, save_events_bulk, get_match_players, save_players_bulk, save_advanced_metrics_bulk, save_audit_event, get_audit_events, save_coding_tag, get_coding_tags, update_coding_tag, delete_coding_tag, get_coding_tag_stats | SQLite persistence — 40+ CRUD methods across 10+ tables | STABLE |
| `audit_service.py` | AuditService | log_event, query_events, get_recent_events, get_events_by_action | Structured audit trail with timestamped action logging | STABLE |
| `data_export_service.py` | DataExportService | export_csv, export_json, export_statsbomb, export_spadl, export_pdf, export_match_report | Multi-format export (CSV, JSON, StatsBomb, SPADL, PDF) | STABLE |
| `data_import_service.py` | DataImportService | import_file, import_csv, import_json, import_statsbomb, detect_format | CSV/JSON/StatsBomb file import with auto-detect | STABLE |
| `sample_data_generator.py` | *module-level functions* | generate_sample_match, generate_sample_events | Demo data generator for match simulation | BETA |

### 🔬 Quality & Validation Services

| Module | Class(es) | Key Methods | Description | Quality |
|--------|-----------|-------------|-------------|---------|
| `quality_scoring_service.py` | QualityScoringService, QualityScores | compute_quality_scores, get_tracking_quality, get_event_quality, get_homography_quality, get_team_assignment_quality | Per-match data quality scoring (0-1 across 5 dimensions) | STABLE |
| `anomaly_detection_service.py` | AnomalyDetectionService, Anomaly | scan_match, check_physical_stats, check_tracking_quality, check_event_stats, check_team_stats | Detects data quality issues (impossible stats, outliers) | STABLE |
| `validation_service.py` | ValidationService, EventGroundTruth | validate_events, validate_tracking, validate_stats, compute_accuracy_metrics | Compares computed metrics against ground truth benchmarks | STABLE |
| `benchmark_service.py` | BenchmarkService, BenchmarkResult | run_benchmark, get_gpu_benchmark, compare_performance, cache_result | Performance benchmarking per GPU tier (speed, memory, FPS) | STABLE |
| `coordinate_validator.py` (core) | CoordinateValidator | validate_position, validate_event_coords, clamp_coordinates | Validates/clamps event coordinates with warnings | STABLE |
| `weather_image_classifier.py` | WeatherImageClassifier | classify_weather, classify_frame, get_weather_confidence | CNN weather classifier (MobileNetV3 — rainy/cloudy/sunny/snowy/foggy) | ALPHA |

### 🌤️ Environment Services

| Module | Class(es) | Key Methods | Description | Quality |
|--------|-----------|-------------|-------------|---------|
| `weather_service.py` | WeatherService, WeatherConditions | get_match_weather, analyze_weather_impact, get_historical_weather, detect_weather_from_video | Multi-source weather (Open-Meteo + manual + video inference) | STABLE |
| `raindrop_detection_service.py` | RaindropDetectionService | detect_raindrops, get_raindrop_density, classify_precipitation | AlexNet-style raindrop detection in video frames | ALPHA |

### ⚙️ Utility & Infrastructure Services

| Module | Class(es) | Key Methods | Description | Quality |
|--------|-----------|-------------|-------------|---------|
| `vram_manager.py` | VRAMManager, ModelPriority | load_model, unload_model, get_usage, optimize, fallback_to_cpu | GPU VRAM orchestration across YOLO/LLM/Whisper models | STABLE |
| `batch_service.py` | BatchService, BatchJob, BatchStatus | create_batch, run_batch, cancel_batch, get_status, get_results | Overnight multi-match analysis queue | STABLE |
| `auto_updater_service.py` | AutoUpdaterService, ReleaseInfo | check_for_update, download_update, install_update, get_release_notes | GitHub release auto-updater | BETA |
| `cloud_sync_service.py` | CloudSyncService | login, logout, sync_up, sync_down, get_status | Optional cloud synchronization service | BETA |
| `collaboration_service.py` | CollaborationService, CollabUser, Comment | invite_user, add_comment, share_match, get_shared_matches | Multi-user collaboration for shared match analysis | BETA |
| `marketplace_service.py` | MarketplaceService, MarketplaceItem | list_items, search, install_item, rate_item, publish_item | Community drill/template/plugin marketplace | BETA |
| `feedback_service.py` | FeedbackService, CoachFeedback | submit_feedback, get_feedback, get_analytics, export_feedback | Structured coach feedback collection (local SQLite) | STABLE |
| `observation_service.py` (core) | ObservabilityService, metrics | track_metric, get_metrics, export_metrics | Metrics tracking for observability | STABLE |
| `player_profile_service.py` | PlayerProfileService | create_profile, link_player, get_career_stats, update_profile, search_profiles | Persistent player identity across matches | STABLE |
| `opponent_database_service.py` | OpponentDatabaseService, OpponentProfile | create_profile, update_profile, get_profile, search_profiles, add_match_note | Structured opponent profile storage with tactical tendencies | STABLE |
| `cross_match_linking_service.py` | CrossMatchLinkingService | link_all_matches, auto_link, flag_for_review, create_new_profiles | Automatic player linking across matches via ReID/face/jersey | BETA |
| `mujoco_ball_service.py` | MuJoCoBallService, TrajectoryPoint | simulate_trajectory, simulate_with_spin, get_trajectory_points, get_landing_position | MuJoCo ball trajectory simulation with drag + Magnus effect | BETA |
| `fluidx3d_service.py` | FluidX3DService | simulate_aerodynamics, get_flow_field, is_available | FluidX3D CFD wrapper for ball aerodynamics (stub, optional dep) | ALPHA |
| `visualization_service.py` | VisualizationService | generate_heatmap, generate_pass_network, generate_pass_sonar, generate_formation_diagram, save_chart | matplotlib + mplsoccer professional chart generation | STABLE |
| `heatmap_generator.py` | *module-level functions* | generate_heatmap, generate_per_player_heatmaps | 2D Gaussian KDE heat map generation (standalone) | STABLE |

---

## Core Modules (86 analytical engine modules)

### ⚽ Expected Goals (xG) & Finishing

| Module | Classes/Functions | Key API | Description | Quality |
|--------|------------------|---------|-------------|---------|
| `xg_model.py` | compute_xg, XgModel | compute_xg(distance, angle, header, ...) | Heuristic xG model with named constants | STABLE |
| `dl_xg_model.py` | DLXgModel | predict_xg(features), load_model, save_model | PyTorch deep-learning xG (attention pooling) | BETA |
| `xg_chain.py` | compute_xg_chain | compute_xg_chain(events) | xG chain computation from possession sequences | STABLE |
| `xg_trainer.py` | XgTrainer | train, evaluate, save_model | xG model training utilities | BETA |
| `xga_model.py` | compute_xga | compute_xga(shot_features) | Expected Goals Assisted model | BETA |
| `crossing_xg.py` | compute_crossing_xg | compute_crossing_xg(cross_features) | xG model for crosses | BETA |
| `corner_xg.py` | compute_corner_xg | compute_corner_xg(corner_features) | xG model for corner kicks | BETA |
| `finishing_analysis.py` | FinishingAnalyzer | analyze_finishing, compute_streaks, placement_skill | Shot quality tiers, hot/cold streaks, placement skill | STABLE |
| `psxg_model.py` | compute_psxg | compute_psxg(shot_features) | Post-shot expected goals model | STABLE |
| `psxg_improved.py` | compute_psxg_improved | compute_psxg_improved(features) | Improved PSxG with additional features | BETA |

### 🎯 Expected Threat (xT) & Pitch Control

| Module | Classes/Functions | Key API | Description | Quality |
|--------|------------------|---------|-------------|---------|
| `xt_model.py` | XtModel, compute_xt | compute_xt(start_zone, end_zone), zone_value | Expected Threat model (20×32 grid) | STABLE |
| `carry_xt.py` | compute_carry_xt | compute_carry_xt(trajectory, xt_model) | Carry progression xT through grid zones | STABLE |
| `defensive_xt.py` | compute_defensive_xt | compute_defensive_xt(zones) | Defensive expected threat | BETA |
| `set_piece_xt.py` | compute_set_piece_xt | compute_set_piece_xt(event_features) | Set-piece expected threat | BETA |
| `pitch_control.py` | compute_pitch_control | compute_pitch_control(player_positions, ball_position) | Pitch control surface via numpy broadcasting | STABLE |
| `ball_physics_pitch_control.py` | simulate_ball_trajectory, compute_pitch_control_physics | compute_pitch_control(tracks, ball_pos) | 3D trajectory RK4 integration + physics-based pitch control | STABLE |
| `territory_value.py` | compute_territory_value, TeamTerritory | compute_territory_value(possession_chains, xt_model) | xT accumulation per possession chain with zone-level advantage | STABLE |
| `phase_xg.py` | classify_shot_phase, compute_phase_xg | classify_phase(event), compute_phase_summary | Shot phase classification (settled/transition/counter/set-piece/direct) | STABLE |
| `build_up.py` | analyze_build_up, BuildUpPattern | analyze_goal_kicks, detect_line_breaking_passes, detect_build_under_pressure | Build-up pattern analysis with line-breaking pass detection | STABLE |

### 📊 Pass Analysis & Networks

| Module | Classes/Functions | Key API | Description | Quality |
|--------|------------------|---------|-------------|---------|
| `pass_network.py` | PassNetwork, compute_pass_network | compute_pass_network(events), betweenness_centrality | Weighted pass network graph with eigenvector centrality | STABLE |
| `pass_flow.py` | PassFlowAnalyzer | analyze_pass_flow, get_flow_directions | Pass flow direction analysis | STABLE |
| `pass_sonars.py` | PassSonar, compute_pass_sonar | compute_pass_sonar(events, player_id) | Polar pass direction/distance visualization | STABLE |
| `pass_patterns.py` | PassPatternDetector | detect_patterns, find_rotations | Passing pattern detection (triangles, rotations) | BETA |
| `passing_triangles.py` | PassingTriangleAnalyzer | analyze_triangles, find_common_triangles | Passing triangle identification and analysis | BETA |
| `passing_lanes.py` | PassingLaneAnalyzer | compute_lanes, find_available_lanes | Passing lane computation and availability | BETA |
| `expected_pass.py` | compute_xp | compute_xp(pass_features) | Expected pass completion model | BETA |
| `xa_model.py` | compute_xa | compute_xa(pass_features) | Expected assist with cross-subtype granularity | STABLE |
| `xa_split.py` | compute_xa_split | compute_xa_split(pass_features, pass_type) | xA split by pass subtype (early/cutback/driven/lofted) | BETA |
| `through_ball.py` | ThroughBallAnalyzer | analyze_through_balls, classify_through_ball | Through-ball detection and classification | BETA |
| `switch_of_play.py` | SwitchOfPlayAnalyzer | detect_switches, analyze_switch_effectiveness | Switch-of-play detection and analysis | BETA |
| `crossing_analysis.py` | CrossingAnalyzer | analyze_crosses, classify_cross_type | Crossing analysis with type classification | STABLE |
| `flank_analysis.py` | FlankAnalyzer | analyze_flank_usage, compute_flank_balance | Left/right flank balance analysis | BETA |

### 🛡️ Defensive & Pressing

| Module | Classes/Functions | Key API | Description | Quality |
|--------|------------------|---------|-------------|---------|
| `defensive_actions.py` | DefensiveActionAnalyzer | analyze_defensive_actions, classify_tackle_type | Defensive action analysis (tackles, interceptions, clearances, blocks) | STABLE |
| `pressing_efficiency.py` | compute_pressing_efficiency | compute_pressing_efficiency(tracks) | Pressing success rate and efficiency metrics | STABLE |
| `pressing_traps.py` | detect_pressing_traps | detect_pressing_traps(events, positions) | Pressing trap identification (forcing play into zones) | STABLE |
| `pressing_clusters.py` | PressingClusterAnalyzer | analyze_clusters, find_pressing_coordination | Pressing cluster/coordination analysis | BETA |
| `duel_analysis.py` | DuelAnalyzer | analyze_duels, classify_duel_type, compute_duel_success | Duel detection and analysis (aerial, ground, 50-50) | BETA |
| `ball_recovery.py` | BallRecoveryAnalyzer | analyze_recoveries, classify_recovery_type | Ball recovery analysis with type classification | BETA |
| `packing.py` | compute_packing | compute_packing(events, positions) | Packing rate (opponents bypassed per pass) | BETA |
| `trap_transition_linkage.py` | TrapTransitionLinkage | link_traps_to_transitions, compute_linkage_metrics | Temporal/spatial linkage between pressing traps and transitions | STABLE |

### 🔄 Transitions & Momentum

| Module | Classes/Functions | Key API | Description | Quality |
|--------|------------------|---------|-------------|---------|
| `transitions.py` | TransitionAnalyzer | analyze_transitions, classify_transition_type, compute_transition_efficiency | Attack↔defense transition analysis | STABLE |
| `momentum.py` | compute_momentum, MomentumCalculator | compute_momentum(events, windows) | Rolling momentum computation from xG/possession | STABLE |
| `game_state.py` | GameStateTracker, ScoreState | track_game_state, get_state_transitions | Score-line state tracking across match | STABLE |
| `dominance_index.py` | compute_dominance_index | compute_dominance_index(stats) | Composite dominance index from multiple metrics | BETA |

### 👤 Player Analysis

| Module | Classes/Functions | Key API | Description | Quality |
|--------|------------------|---------|-------------|---------|
| `player_rating.py` | compute_player_rating, RatingService | compute_rating(stats), get_overall | Composite player rating (0-100) from match stats | STABLE |
| `player_similarity.py` | PlayerSimilarityEngine | compute_similarity, find_similar_players, get_similarity_matrix | Multi-dimensional player similarity (scout reports) | STABLE |
| `player_search.py` | PlayerSearchService | search_by_criteria, score_match, get_results | Multi-criteria player search (age, position, league, stats) | STABLE |
| `role_classifier.py` | RoleClassifier | classify_role, get_role_attributes | Position/role classification from tracking data | BETA |
| `offball_metrics.py` | OffballMetrics | compute_offball_metrics, space_creation_score | Off-ball movement and space creation metrics | STABLE |
| `obv.py` | compute_obv | compute_obv(player_positions, ball_pos) | Off-Ball Value via pass-probability + xT deltas | STABLE |

### 🏆 Formations & Tactics

| Module | Classes/Functions | Key API | Description | Quality |
|--------|------------------|---------|-------------|---------|
| `formation_analysis.py` | FormationAnalyzer, Formation | detect_formation, compute_compactness, compute_width_depth | Formation detection via k-means clustering | STABLE |
| `formation_effectiveness.py` | FormationEffectivenessAnalyzer | analyze_effectiveness, compare_formations | Formation effectiveness against specific opponents | BETA |
| `tactical_periods.py` | TacticalPeriodAnalyzer | detect_periods, classify_phase, get_period_breakdown | Match phase classification (settled/transition/set_piece) | STABLE |
| `game_plan.py` | GamePlanAnalyzer | generate_game_plan, analyze_opponent | Opponent game plan generation with scoreline prediction | STABLE |
| `lineup_optimizer.py` | LineupOptimizer | optimize_lineup(template, player_ratings) | MILP-based lineup optimization (4-4-2, 4-3-3, 3-5-2) | STABLE |
| `match_scripting.py` | MatchScriptingService | simulate_script, get_script_outcomes | Match scenario scripting simulation | BETA |
| `scoreline_distribution.py` | compute_scoreline_distribution | compute_scoreline_distribution(xg_matrix) | Scoreline probability distribution from xG | BETA |

### 💰 Recruitment & Valuation

| Module | Classes/Functions | Key API | Description | Quality |
|--------|------------------|---------|-------------|---------|
| `squad_valuation.py` | SquadValuationService, PlayerValuation | compute_player_value, get_squad_value, estimate_transfer_fee | Player/squad valuation with age curve, position baselines, multipliers | STABLE |
| `scout_reports.py` | ScoutReportGenerator | generate_report, compare_players | Automated scout report generation | STABLE |
| `scout_report_upgrade.py` | ScoutReportUpgrade | upgrade_report, add_video_evidence | Video-enhanced scout reports | BETA |
| `fixture_difficulty.py` | FixtureDifficultyAnalyzer | analyze_fixtures, compute_difficulty, detect_congestion | Fixture difficulty with opponent strength, H/A weight, schedule density | STABLE |
| `league_simulation.py` | LeagueSimulator | simulate_season(n_runs=10000), get_probabilities | Monte Carlo league simulation (Poisson xG → title/Top4/relegation) | STABLE |
| `goals_added.py` | GoalsAddedCalculator | compute_goals_added(game_states, player_id) | Goals Added (g+) framework — xG/xA/xT/defensive/OBV composite | STABLE |

### 📋 Discipline & Form

| Module | Classes/Functions | Key API | Description | Quality |
|--------|------------------|---------|-------------|---------|
| `referee_analysis.py` | RefereeAnalyzer | analyze_referee, compute_card_rate, detect_home_bias, compute_inconsistency | Referee profiling with card rate, bias, inconsistency scoring | STABLE |
| `suspension_tracker.py` | SuspensionTracker | check_suspension_risk, get_fair_play_score, get_upcoming_risks | Yellow-card accumulation and suspension risk detection | STABLE |
| `form_analysis.py` | FormAnalyzer | analyze_form_by_competition, analyze_form_by_opponent_strength | Form analysis by competition type + opponent strength tier | STABLE |

### 📈 Season & Match Reporting

| Module | Classes/Functions | Key API | Description | Quality |
|--------|------------------|---------|-------------|---------|
| `season_aggregator.py` | SeasonAggregator | aggregate_season, get_player_season_stats, get_team_season_stats | Season-level statistics aggregation | STABLE |
| `match_report.py` | MatchReportGenerator | generate_report, get_summary, get_key_moments, get_tactical_observations | Day After Match report with executive summary + key moments | STABLE |
| `match_timeline.py` | MatchTimeline | build_timeline, get_events_by_phase, get_events_by_type | Chronological match event timeline | STABLE |
| `substitution_analysis.py` | SubstitutionAnalyzer | analyze_substitution, compute_impact | Substitution event impact analysis | STABLE |
| `model_comparison.py` | ModelComparisonService | compare_models, evaluate(log_loss, Brier, AUC-ROC), compute_feature_importance | xG model comparison (heuristic vs logistic vs DL) | STABLE |

### 🔧 Infrastructure Core

| Module | Classes/Functions | Key API | Description |
|--------|------------------|---------|-------------|
| `events.py` | event_from_dict, MatchEvent, ShotEvent, PassEvent, ... | Event data model with type hierarchy | Core event dataclasses |
| `coords.py` | PitchCoordinate, convert_coordinates | Coordinate system helpers | Spatial coordinate utilities |
| `coordinate_validator.py` | CoordinateValidator | validate_position, clamp_coordinates, validate_event_coords | Spatial coordinate validation + clamping |
| `game_constants.py` | GAME (namespace) | Pitch dimensions, goal size, ball specs, thresholds | Unified pitch constants used by 14+ modules |
| `config.py` | load_config, get_config | load_config(path) | Centralized YAML/JSON config (70+ params) |
| `security.py` | SecurityValidator, ErrorSanitizer, RateLimiter | validate_input, sanitize, acquire(category) | Input validation, sanitization, token-bucket rate limiter |
| `secrets.py` | get_api_key, set_api_key | API key management | Secure API key storage |
| `logging.py` | get_logger, LogConfig | get_logger(name) | Structured logging setup |
| `paths.py` | get_paths, Paths | get_paths() → Paths | Centralized path configuration |
| `migration_manager.py` | MigrationManager | run_migrations, get_version | SQL schema migration management (001-018) |
| `database_sharding.py` | DatabaseShardingService | shard_data, get_shard | Database sharding utilities |
| `model_manager.py` | ModelManager | load_model, download_model, cache_model | ML model download/cache (YOLO, OSNet, etc.) |
| `gpu_acceleration.py` | detect_gpu_tier, recommend_yolo_variant | GPU capability detection | GPU tier detection and optimization |
| `vram_manager.py` | VRAMManager | *(see Services section)* | GPU memory orchestration |
| `perf_timing.py` | PerfTimer, timed | time_function, get_stats | Execution timing utilities |
| `benchmarks.py` | run_inference_benchmark, run_tracking_benchmark | Performance baselines | Inference/tracking benchmark constants |
| `observability.py` | ObservabilityService | track_metric, get_metrics | Metrics collection and export |
| `progressive_actions.py` | compute_progressive_actions | compute_progressive_actions(events) | Progressive pass/carry/dribble detection |
| `space_control.py` | compute_space_control | compute_space_control(positions) | Space control analysis |
| `influence_map.py` | compute_influence_map | compute_influence_map(positions) | Player influence map computation |
| `velocity_analysis.py` | compute_velocity_analysis | compute_velocity_analysis(tracks) | Player velocity and acceleration analysis |
| `box_entries.py` | compute_box_entries | compute_box_entries(events) | Penalty area entry detection |
| `confidence_intervals.py` | compute_xg_ci, compute_xt_ci, compute_vaep_ci | Bootstrap/Beta conjugate CIs for xG/xT/VAEP | Credible intervals for analytical models |
| `uncertainty.py` | UncertaintyEstimator | estimate_uncertainty(predictions) | Prediction uncertainty estimation |
| `mot_metrics.py` | compute_mota, compute_motp, compute_idf1 | CLEAR MOT metrics | MOTA/MOTP/IDF1 computation |
| `physical_metrics.py` | PhysicalMetricsAnalyzer | compute_sprint_profile, compute_distance_zones | Core physical metric algorithms |
| `heatmap.py` | HeatmapGenerator | generate_heatmap, generate_team_heatmap | Heat map generation algorithms |
| `pattern_detection.py` | PatternDetector | detect_patterns(events) | Tactical pattern detection |
| `vaep.py` | compute_vaep | compute_vaep(events, xt_model) | Video Action Effectiveness Prediction | STABLE |
| `epv.py` | compute_epv | compute_epv(state) | Expected Possession Value | STABLE |
| `win_probability.py` | compute_win_probability, WinProbSimulator | compute_win_prob(match_state) | Monte Carlo win probability from xG, time, score | STABLE |
| `confidence_intervals.py` | *(see above)* | xG/VAEP/EPV credible intervals | Bootstrap CIs | STABLE |
| `injury_risk.py` | InjuryRiskAnalyzer | compute_risk_from_load(past_load) | Injury risk estimation from load data | BETA |
| `vaep.py` | *(see above)* | VAEP 2.0 with spatiotemporal features | Player-relative distance/velocity + density features | STABLE |
| `export_converters.py` | to_statsbomb, to_spadl | StatsBomb/SPADL format converters | Data format conversion utilities | STABLE |

---

## Bridge Handlers (8 modules)

| Handler | File | Exposed Slots | Description |
|---------|------|---------------|-------------|
| `AnalysisHandler` | `bridge_analysis.py` | getAnalysis, getPlayerProfile, getSetPieceAnalysis, getGoalkeeperAnalysis, getSubstitutionImpact, getPossession, getPsychologyReport, getTacticalReview, getCards, getPoseAnalysis, getWeatherImpact, getMuJoCoSimulation, getFluidX3DSimulation, getRoboflowDraw, analyzeFormation, getPressureMetrics, getPlayerRating, getSquadSummary, ask_llm, getTacticalPeriods, analyzeFormation (3129 lines) | All match analysis + specialized service operations |
| `ExternalHandler` | `bridge_external.py` | getApiFootballData, getStatsBombData, getFootballData, getBzzoiroData, getTheSportsDB, getOpenFootball, getEasySoccer, searchAllSources, getTransfermarktData (1011 lines) | All external football data API integrations |
| `CodingHandler` | `bridge_coding.py` | saveTag, getTags, updateTag, deleteTag, getTagStats, getTagsByType, getTagsByPlayer, getMatchPlayersSimple, extractTagClip, extractTagClipsBatch, getDefaultTagTemplates (252 lines) | Manual video tagging workspace (Sportscode/Nacsport-style) |
| `VideoHandler` | `bridge_video.py` | getRealtimeAnalysis, startLiveStream, stopLiveStream, multiAngleSync, trimVideo, createHighlightReel (163 lines) | Video/realtime/streaming operations |
| `ExportHandler` | `bridge_export.py` | exportCSV, exportJSON, exportPDF, exportStatsBomb, exportSPADL, extractVideoClips, batchExport (193 lines) | Data export operations |
| `StorageHandler` | `bridge_storage.py` | saveEvent, getEvents, deleteEvent, updateEvent, getMatches, getFeedback, saveFeedback, getAdvancedMetrics, batchOps (113 lines) | CRUD storage operations |
| `LifecycleHandler` | `bridge_lifecycle.py` | getAppState, getGPUStatus, getProfilerData, getMetrics, getBenchmarkInfo (99 lines) | App lifecycle + GPU/profiler/benchmark queries |

---

## CLI Commands

| Command | Subcommands | Module | Description |
|---------|-------------|--------|-------------|
| `python -m kawkab` | `track`, `evaluate`, `render`, `events` | `__main__.py` | CLI entry points for tracking pipeline |
| `python -m kawkab track` | `--video`, `--output`, `--skip` | `__main__.py` | Run full tracking pipeline on a video |
| `python -m kawkab evaluate` | `--video`, `--gt` | `__main__.py` | Evaluate tracking against ground truth |
| `python -m kawkab render` | `--video`, `--tracks`, `--output` | `__main__.py` | Render tracking overlay video |
| `python -m kawkab events` | `--video`, `--output` | `__main__.py` | Detect events from tracking output |

---

## Quality Legend

| Badge | Meaning |
|-------|---------|
| **STABLE** | Tested in CI, production-ready, 20+ tests |
| **BETA** | Functional, tested, may have edge cases |
| **ALPHA** | Early-stage, partial tests, API may change |

---

## Quick start

```python
import asyncio
from kawkab.services.cv_service import CVService
from kawkab.services.analysis_service import AnalysisService

async def analyze_video(path):
    cv = CVService()
    await cv.initialize()
    analysis = AnalysisService()
    track_data = await cv.process_video(path)
    report = await analysis.analyze_match(track_data)
    return report

asyncio.run(analyze_video("match.mp4"))
```
