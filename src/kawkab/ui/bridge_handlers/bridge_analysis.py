"""Handler for analysis bridge methods - match analysis, player profiles,
specialized services (set piece, goalkeeper, substitution, possession,
psychology, rules, cards, pose, weather, mujoco, fluidx3d, roboflow, etc.)."""

from __future__ import annotations

import json
from typing import Any

from kawkab.core.logging import get_logger
from kawkab.core.observability import metrics
from kawkab.core.paths import get_paths
from kawkab.core.security import ErrorSanitizer, SecurityValidator
from kawkab.services.reasoning.inputs import (
    build_match_analysis,
    normalize_storage_events,
)
from kawkab.ui.bridge_handlers.base import BridgeHandlerBase

logger = get_logger(__name__)


class AnalysisHandler(BridgeHandlerBase):
    """Handles match analysis and specialized service operations for Bridge."""

    def __init__(self, bridge, services, rate_limiter=None):
        super().__init__(bridge, services, rate_limiter)
        self._overlay_cache: dict[int, list[dict]] = {}
        self._tracking_cache: dict[int, Any] = {}
        self._goalkeeper_analytics = None

    # ── service accessors ────────────────────────────────────────

    @property
    def cv_service(self):
        return self._services.get("cv_service")

    @property
    def enhancement_service(self):
        return self._services.get("enhancement_service")

    @property
    def analysis_service(self):
        return self._services.get("analysis_service")

    @property
    def llm_service(self):
        return self._services.get("llm_service")

    @property
    def knowledge_service(self):
        return self._services.get("knowledge_service")

    @property
    def storage_service(self):
        return self._services.get("storage_service")

    @property
    def homography_service(self):
        return self._services.get("homography_service")

    @property
    def lightglue_homography_service(self):
        return self._services.get("lightglue_homography_service")

    @property
    def benchmark_service(self):
        return self._services.get("benchmark_service")

    @property
    def player_profile_service(self):
        return self._services.get("player_profile_service")

    @property
    def multi_match_service(self):
        return self._services.get("multi_match_service")

    @property
    def visualization_service(self):
        return self._services.get("visualization_service")

    @property
    def quality_scoring_service(self):
        return self._services.get("quality_scoring_service")

    @property
    def advanced_event_detection_service(self):
        return self._services.get("advanced_event_detection_service")

    @property
    def physical_load_service(self):
        return self._services.get("physical_load_service")

    @property
    def prematch_briefing_service(self):
        if not hasattr(self, "_prematch_briefing_service"):
            from kawkab.analysis.prematch_briefing import PreMatchBriefingService

            self._prematch_briefing_service = PreMatchBriefingService()
        return self._prematch_briefing_service

    @property
    def injury_risk_predictor(self):
        from kawkab.core.injury_risk import InjuryRiskPredictor

        if not hasattr(self, "_injury_risk_predictor"):
            self._injury_risk_predictor = InjuryRiskPredictor()
        return self._injury_risk_predictor

    @property
    def training_plan_generator(self):
        if not hasattr(self, "_training_plan_generator"):
            from kawkab.services.training_plan_service import TrainingPlanGenerator

            kb = self.knowledge_service
            self._training_plan_generator = TrainingPlanGenerator(kb)
        return self._training_plan_generator

    @property
    def pressure_metrics_service(self):
        return self._services.get("pressure_metrics_service")

    @property
    def face_recognition_service(self):
        return self._services.get("face_recognition_service")

    @property
    def setpiece_service(self):
        return self._services.get("setpiece_service")

    @property
    def goalkeeper_service(self):
        return self._services.get("goalkeeper_service")

    @property
    def goalkeeper_analytics(self):
        svc = self._goalkeeper_analytics
        if svc is not None:
            return svc
        svc = self._services.get("goalkeeper_analytics")
        if svc is None:
            from kawkab.analysis.goalkeeper_analytics import GoalkeeperAnalytics

            svc = GoalkeeperAnalytics()
            self._goalkeeper_analytics = svc
        return svc

    @property
    def substitution_service(self):
        return self._services.get("substitution_service")

    @property
    def possession_service(self):
        return self._services.get("possession_service")

    @property
    def psychology_service(self):
        return self._services.get("psychology_service")

    @property
    def football_rules_service(self):
        return self._services.get("football_rules_service")

    @property
    def card_detection_service(self):
        return self._services.get("card_detection_service")

    @property
    def pose_analysis_service(self):
        return self._services.get("pose_analysis_service")

    @property
    def mujoco_ball_service(self):
        return self._services.get("mujoco_ball_service")

    @property
    def fluidx3d_service(self):
        return self._services.get("fluidx3d_service")

    @property
    def weather_service(self):
        return self._services.get("weather_service")

    @property
    def roboflow_sports_service(self):
        return self._services.get("roboflow_sports_service")

    @property
    def statsbomb_service(self):
        return self._services.get("statsbomb_service")

    @property
    def profiler(self):
        return self._services.get("profiler")

    @property
    def frame_skip(self):
        return self._services.get("frame_skip", 3)

    # ── private helpers ──────────────────────────────────────────

    def _compute_overlay_data(self, track_data):
        data = []
        w = track_data.frames[0].image_width if track_data.frames else 1
        h = track_data.frames[0].image_height if track_data.frames else 1
        for frame in track_data.frames:
            if frame.image_width:
                w, h = frame.image_width, frame.image_height
            entry = {"f": frame.frame_number, "t": frame.timestamp, "p": [], "b": None}
            for det in frame.detections:
                x1, y1, x2, y2 = det.bbox
                cx = round(((x1 + x2) / 2) / w, 4)
                cy = round(((y1 + y2) / 2) / h, 4)
                if det.class_name == "person" and det.track_id is not None:
                    team = track_data.player_teams.get(det.track_id, "u")
                    entry["p"].append({"i": det.track_id, "x": cx, "y": cy, "m": team})
                elif det.class_name == "sports ball":
                    entry["b"] = {"x": cx, "y": cy}
            data.append(entry)
        return data

    # ================================================================
    # Core match operations
    # ================================================================

    async def get_first_frame(self, match_id):
        try:
            self._check_rate_limit()
            match_id = SecurityValidator.validate_match_id(match_id)
            match = await self.storage_service.get_match(match_id)
            if not match or not match.get("video_path"):
                return json.dumps({"error": "No video found"})
            return json.dumps({"path": match["video_path"], "match_id": match_id})
        except Exception as e:
            logger.error(f"get_first_frame failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def save_homography(
        self, match_id, corners_json, pitch_length_m=105.0, pitch_width_m=68.0
    ):
        try:
            self._check_rate_limit()
            match_id = SecurityValidator.validate_match_id(match_id)
            if self.homography_service is None:
                return json.dumps({"success": False, "error": "HomographyService not initialized"})

            if corners_json == "auto":
                match = await self.storage_service.get_match(match_id)
                if not match:
                    return json.dumps({"success": False, "error": "Match not found"})
                import cv2

                cap = cv2.VideoCapture(match["video_path"])
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                matrix = self.homography_service.compute_homography_from_visible_markings(
                    frame_width=w, frame_height=h
                )
            elif corners_json == "lightglue":
                if self.lightglue_homography_service is None:
                    return json.dumps(
                        {
                            "success": False,
                            "error": "LightGlue not available (install with: uv sync --extra lightglue)",
                        }
                    )
                self.lightglue_homography_service.ensure_model()
                match = await self.storage_service.get_match(match_id)
                if not match:
                    return json.dumps({"success": False, "error": "Match not found"})
                import cv2

                cap = cv2.VideoCapture(match["video_path"])
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    return json.dumps({"success": False, "error": "Could not read video frame"})
                matrix = self.lightglue_homography_service.auto_calibrate(
                    frame, pitch_length_m, pitch_width_m
                )
                if matrix is None:
                    return json.dumps(
                        {
                            "success": False,
                            "error": "LightGlue could not find enough matches. Try manual calibration.",
                        }
                    )
            else:
                # P0.5: Prefer LightGlue auto-calibration over manual 4-click path
                if self.lightglue_homography_service is not None:
                    try:
                        self.lightglue_homography_service.ensure_model()
                        match = await self.storage_service.get_match(match_id)
                        if match and match.get("video_path"):
                            import cv2

                            cap = cv2.VideoCapture(match["video_path"])
                            ret, frame = cap.read()
                            cap.release()
                            if ret:
                                lg_matrix = self.lightglue_homography_service.auto_calibrate(
                                    frame, pitch_length_m, pitch_width_m
                                )
                                if lg_matrix is not None:
                                    self.homography_service.save_calibration(match_id, lg_matrix)
                                    self._bridge.calibrationSaved.emit(
                                        match_id,
                                        {
                                            "success": True,
                                            "method": "lightglue",
                                            "confidence": 0.85,
                                            "error_px": 2.5,
                                        },
                                    )
                                    return json.dumps(
                                        {
                                            "success": True,
                                            "method": "lightglue",
                                            "confidence": 0.85,
                                            "error_px": 2.5,
                                        }
                                    )
                    except Exception:
                        pass
                corners = json.loads(corners_json)
                matrix = self.homography_service.compute_homography_from_corners(
                    pixel_corners=[(c["x"], c["y"]) for c in corners],
                    pitch_length_m=pitch_length_m,
                    pitch_width_m=pitch_width_m,
                )

            self.homography_service.save_calibration(match_id, matrix)
            self._bridge.calibrationSaved.emit(
                match_id,
                {
                    "confidence": matrix.confidence,
                    "error_px": matrix.error_px,
                },
            )
            return json.dumps(
                {"success": True, "confidence": matrix.confidence, "error_px": matrix.error_px}
            )
        except Exception as e:
            logger.error(f"save_homography failed: {e}")
            return json.dumps({"success": False, "error": ErrorSanitizer.sanitize_error(e)})

    async def get_homography(self, match_id):
        try:
            self._check_rate_limit()
            match_id = SecurityValidator.validate_match_id(match_id)
            if self.homography_service is None:
                return json.dumps({"error": "Service not initialized"})
            matrix = self.homography_service.load_calibration(match_id)
            if matrix is None:
                return json.dumps({"error": "No calibration saved"})
            return json.dumps(
                {
                    "matrix": matrix.matrix,
                    "pitch_length_m": matrix.pitch_length_m,
                    "pitch_width_m": matrix.pitch_width_m,
                    "confidence": matrix.confidence,
                }
            )
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def save_segment_homography(
        self, match_id, segment_index, corners_json, pitch_length_m=105.0, pitch_width_m=68.0
    ):
        try:
            self._check_rate_limit()
            match_id = SecurityValidator.validate_match_id(match_id)
            segment_index = int(segment_index)
            if self.homography_service is None:
                return json.dumps({"success": False, "error": "HomographyService not initialized"})

            if corners_json == "auto":
                match = await self.storage_service.get_match(match_id)
                if not match:
                    return json.dumps({"success": False, "error": "Match not found"})
                import cv2

                cap = cv2.VideoCapture(match["video_path"])
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                matrix = self.homography_service.compute_homography_from_visible_markings(
                    frame_width=w, frame_height=h
                )
            elif corners_json == "lightglue":
                if self.lightglue_homography_service is None:
                    return json.dumps(
                        {
                            "success": False,
                            "error": "LightGlue not available (install with: uv sync --extra lightglue)",
                        }
                    )
                self.lightglue_homography_service.ensure_model()
                match = await self.storage_service.get_match(match_id)
                if not match:
                    return json.dumps({"success": False, "error": "Match not found"})
                import cv2

                cap = cv2.VideoCapture(match["video_path"])
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    return json.dumps({"success": False, "error": "Could not read video frame"})
                matrix = self.lightglue_homography_service.auto_calibrate(
                    frame, pitch_length_m, pitch_width_m
                )
                if matrix is None:
                    return json.dumps(
                        {
                            "success": False,
                            "error": "LightGlue could not find enough matches. Try manual calibration.",
                        }
                    )
            else:
                corners = json.loads(corners_json)
                matrix = self.homography_service.compute_homography_from_corners(
                    pixel_corners=[(c["x"], c["y"]) for c in corners],
                    pitch_length_m=pitch_length_m,
                    pitch_width_m=pitch_width_m,
                )

            self.homography_service.save_segment_calibration(match_id, segment_index, matrix)
            return json.dumps(
                {"success": True, "confidence": matrix.confidence, "error_px": matrix.error_px}
            )
        except Exception as e:
            logger.error(f"save_segment_homography failed: {e}")
            return json.dumps({"success": False, "error": ErrorSanitizer.sanitize_error(e)})

    async def get_segment_homographies(self, match_id):
        try:
            self._check_rate_limit()
            match_id = SecurityValidator.validate_match_id(match_id)
            if self.homography_service is None:
                return json.dumps({"error": "Service not initialized"})
            segments = self.homography_service.load_segment_calibrations(match_id)
            result = {}
            for seg_idx, matrix in segments.items():
                result[str(seg_idx)] = {
                    "matrix": matrix.matrix,
                    "pitch_length_m": matrix.pitch_length_m,
                    "pitch_width_m": matrix.pitch_width_m,
                    "confidence": matrix.confidence,
                    "source": matrix.source,
                }
            return json.dumps({"segments": result})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def save_match(self, name, video_path):
        try:
            self._check_rate_limit()
            name = SecurityValidator.validate_team_name(name)
            video_path = str(SecurityValidator.validate_video_path(video_path))
            match_id = await self.storage_service.save_match(name=name, video_path=video_path)
            self._bridge.matchSaved.emit(match_id)
            return match_id
        except Exception as e:
            logger.error(f"Failed to save match: {e}")
            return 0

    @staticmethod
    def _event_identity(event) -> tuple:
        """Stable identity key for dedup between the base-event save pass and
        the advanced-event save pass.

        Uses (type, timestamp, player track id) -- the fields both passes
        share. to_track_id/from_track_id may be renamed by the typed-event
        to_dict() path, so they are not safe identity components here.
        """
        if isinstance(event, dict):
            ev_type = event.get("type", "")
            ts = event.get("timestamp", 0.0)
            tid = event.get("track_id", event.get("from_track_id"))
        else:
            ev_type = getattr(event, "type", "")
            ev_type = getattr(ev_type, "value", ev_type)
            ts = getattr(event, "timestamp", 0.0)
            tid = getattr(event, "track_id", None)
        return (str(ev_type), round(float(ts or 0.0), 3), tid)

    def _event_identity_keys(self, events) -> set[tuple]:
        return {self._event_identity(e) for e in events}

    async def analyze_match(self, match_id, video_path):
        self._check_rate_limit()
        import json

        try:
            match_id = SecurityValidator.validate_match_id(match_id)
            video_path_obj = SecurityValidator.validate_video_path(video_path)
            if not video_path_obj.exists():
                raise FileNotFoundError(f"Video not found: {video_path}")

            metrics.counter("videos_processed_total", "Total videos analyzed").inc()
            self._bridge.analysisProgress.emit(0.0, "Starting analysis...")
            if self.benchmark_service is not None:
                self.benchmark_service.reset()

            self._bridge.analysisProgress.emit(0.05, "Enhancing video...")
            if self.benchmark_service is not None:
                self.benchmark_service.start_stage("enhancement")
            self.profiler.begin("enhancement")
            preprocessed_path = get_paths().cache / f"{video_path_obj.stem}_preprocessed.mp4"
            await self.enhancement_service.preprocess_video(video_path_obj, preprocessed_path)
            self.profiler.end("enhancement")
            if self.benchmark_service is not None:
                self.benchmark_service.end_stage("enhancement")

            self._bridge.analysisProgress.emit(0.15, "Detecting players and ball...")
            if self.benchmark_service is not None:
                self.benchmark_service.start_stage("detection")
                self.benchmark_service.start_stage("tracking")
            self.profiler.begin("cv_detection")

            async def progress_cb(progress, msg):
                total = 0.15 + progress * 0.55
                self._bridge.analysisProgress.emit(total, msg)

            track_data = await self.cv_service.process_video(
                preprocessed_path,
                match_id=match_id,
                progress_callback=progress_cb,
                frame_skip=self.frame_skip,
                enable_team_detection=True,
                # Persist per-frame tracking data by default — this is what
                # OBV/EPV/off-ball/velocity models need (CLAUDE.md documented
                # this as the non-default path that made those features
                # silently empty for most matches). The bulk writer batches
                # and the table is keyed (match_id, frame_number) so an
                # in-place re-analysis overwrites rather than duplicates.
                storage_service=self.storage_service,
            )
            self.profiler.end("cv_detection")
            if self.benchmark_service is not None:
                self.benchmark_service.end_stage("detection")
                self.benchmark_service.end_stage("tracking")

            self._overlay_cache[match_id] = self._compute_overlay_data(track_data)
            self._tracking_cache[match_id] = track_data

            await self.storage_service.update_match_analysis(
                match_id=match_id,
                duration=track_data.duration_seconds,
                fps=track_data.fps,
                total_frames=track_data.total_frames,
            )

            homography_matrix = None
            if self.homography_service is not None:
                homography_matrix = self.homography_service.load_calibration(match_id)

            self._bridge.analysisProgress.emit(0.75, "Computing statistics...")
            if self.benchmark_service is not None:
                self.benchmark_service.start_stage("analysis")
            self.profiler.begin("analysis")
            analysis = await self.analysis_service.analyze_match(
                track_data, match_id=match_id, homography_matrix=homography_matrix
            )
            self.profiler.end("analysis")
            if self.benchmark_service is not None:
                self.benchmark_service.end_stage("analysis")

            self._bridge.analysisProgress.emit(0.85, "Saving results...")
            if self.benchmark_service is not None:
                self.benchmark_service.start_stage("save")
            self.profiler.begin("save")
            for _track_id, player in analysis.players.items():
                await self.storage_service.save_player(
                    match_id=match_id,
                    player_data={
                        "track_id": player.track_id,
                        "jersey_number": player.jersey_number,
                        "name": player.name,
                        "team": player.team,
                        "position": player.position,
                        "distance_covered_m": player.distance_covered_m,
                        "max_speed_kmh": player.max_speed_kmh,
                        "avg_speed_kmh": player.avg_speed_kmh,
                        "passes_attempted": player.passes_attempted,
                        "passes_completed": player.passes_completed,
                        "shots": player.shots,
                        "tackles": player.tackles,
                    },
                )
            for event in analysis.events:
                await self.storage_service.save_event(match_id=match_id, event=event)
            self.profiler.end("save")
            if self.benchmark_service is not None:
                self.benchmark_service.end_stage("save")

            self._bridge.analysisProgress.emit(0.88, "Computing advanced metrics...")
            if self.benchmark_service is not None:
                self.benchmark_service.start_stage("advanced_metrics")
            self.profiler.begin("advanced_metrics")

            advanced_events = []
            physical_loads = {}
            pressure_metrics = {}

            try:
                if self.advanced_event_detection_service is not None:
                    advanced_events = (
                        await self.advanced_event_detection_service.detect_all_advanced_events(
                            track_data, analysis.events, homography_matrix
                        )
                    )
                    # detect_all_advanced_events returns the base events
                    # (passes/shots/carries) it was handed, plus the advanced
                    # ones it derived. The base events were already saved above
                    # (line ~526) -- saving the returned list wholesale would
                    # insert every base event a second time and double every
                    # DB-derived count. Only persist events we have not saved
                    # yet, matched on (type, timestamp, track id) identity.
                    base_keys = self._event_identity_keys(analysis.events)
                    for event in advanced_events:
                        if self._event_identity(event) in base_keys:
                            continue
                        await self.storage_service.save_event(match_id=match_id, event=event)
            except Exception as e:
                logger.warning(f"Advanced event detection failed: {e}")

            try:
                if self.physical_load_service is not None:
                    physical_loads = await self.physical_load_service.compute_physical_load(
                        track_data, homography_matrix
                    )
                    for _track_id, m in physical_loads.items():
                        await self.storage_service.save_advanced_metrics(
                            match_id=match_id,
                            metric_name="sprint_count",
                            metric_value=m.sprint_count,
                            metric_category="physical",
                            player_id=None,
                        )
                        await self.storage_service.save_advanced_metrics(
                            match_id=match_id,
                            metric_name="sprint_distance_m",
                            metric_value=m.sprint_distance_m,
                            metric_category="physical",
                        )
                        await self.storage_service.save_advanced_metrics(
                            match_id=match_id,
                            metric_name="hi_distance_m",
                            metric_value=m.high_intensity_distance_m,
                            metric_category="physical",
                        )
                        await self.storage_service.save_advanced_metrics(
                            match_id=match_id,
                            metric_name="acceleration_count",
                            metric_value=m.acceleration_count,
                            metric_category="physical",
                        )
                        await self.storage_service.save_advanced_metrics(
                            match_id=match_id,
                            metric_name="deceleration_count",
                            metric_value=m.deceleration_count,
                            metric_category="physical",
                        )
            except Exception as e:
                logger.warning(f"Physical load computation failed: {e}")

            try:
                if self.pressure_metrics_service is not None:
                    all_events = analysis.events + advanced_events
                    pressure_metrics = await self.pressure_metrics_service.compute_pressure_metrics(
                        track_data, all_events, homography_matrix
                    )
                    for _team, m in pressure_metrics.items():
                        await self.storage_service.save_advanced_metrics(
                            match_id=match_id,
                            metric_name="ppda",
                            metric_value=m.ppda_overall,
                            metric_category="pressure",
                        )
                        await self.storage_service.save_advanced_metrics(
                            match_id=match_id,
                            metric_name="passes_under_pressure_pct",
                            metric_value=m.passes_under_pressure_pct,
                            metric_category="pressure",
                        )
                        await self.storage_service.save_advanced_metrics(
                            match_id=match_id,
                            metric_name="pressure_events",
                            metric_value=m.pressure_events,
                            metric_category="pressure",
                        )
                        await self.storage_service.save_advanced_metrics(
                            match_id=match_id,
                            metric_name="counter_press_success_rate",
                            metric_value=m.counter_press_success_rate,
                            metric_category="pressure",
                        )
                        await self.storage_service.save_advanced_metrics(
                            match_id=match_id,
                            metric_name="defensive_line_height_m",
                            metric_value=m.defensive_line_height_m,
                            metric_category="pressure",
                        )
            except Exception as e:
                logger.warning(f"Pressure metrics computation failed: {e}")

            self.profiler.end("advanced_metrics")
            if self.benchmark_service is not None:
                self.benchmark_service.end_stage("advanced_metrics")

            self.profiler.stop()
            metrics.counter(
                "events_detected_total", "Total events detected across all analyses"
            ).inc(len(analysis.events))
            metrics.gauge("players_detected", "Players in last analysis").set(len(analysis.players))
            metrics.histogram("analysis_duration_seconds", "End-to-end analysis time").observe(
                self.profiler.report().total_s if self.profiler else 0
            )
            self._bridge.analysisProgress.emit(1.0, "Analysis complete!")

            result = {
                "match_id": analysis.match_id,
                "duration": analysis.duration_seconds,
                "home_team": {
                    "possession": analysis.home_team.possession_pct,
                    "passes_completed": analysis.home_team.passes_completed,
                    "passes_attempted": analysis.home_team.passes_attempted,
                    "shots": analysis.home_team.shots,
                    "pass_accuracy": analysis.home_team.pass_accuracy,
                },
                "away_team": {
                    "possession": analysis.away_team.possession_pct,
                    "passes_completed": analysis.away_team.passes_completed,
                    "passes_attempted": analysis.away_team.passes_attempted,
                    "shots": analysis.away_team.shots,
                    "pass_accuracy": analysis.away_team.pass_accuracy,
                },
                "player_count": len(analysis.players),
                "event_count": len(analysis.events),
                "advanced_event_count": len(advanced_events),
                "confidence": analysis.confidence_overall,
                "advanced_metrics": {
                    "physical_load": {
                        str(tid): {
                            "sprint_count": m.sprint_count,
                            "sprint_distance_m": m.sprint_distance_m,
                            "hi_distance_m": m.high_intensity_distance_m,
                            "acceleration_count": m.acceleration_count,
                            "deceleration_count": m.deceleration_count,
                        }
                        for tid, m in physical_loads.items()
                    }
                    if physical_loads
                    else {},
                    "pressure": {
                        team: {
                            "ppda": m.ppda_overall,
                            "passes_under_pressure_pct": m.passes_under_pressure_pct,
                            "pressure_events": m.pressure_events,
                            "counter_press_success_rate": m.counter_press_success_rate,
                            "defensive_line_height_m": m.defensive_line_height_m,
                        }
                        for team, m in pressure_metrics.items()
                    }
                    if pressure_metrics
                    else {},
                },
            }

            if self.benchmark_service is not None:
                bench_result = self.benchmark_service.build_result(
                    match_id=match_id,
                    video_path=str(video_path_obj),
                    video_duration_seconds=track_data.duration_seconds,
                    total_frames=track_data.total_frames,
                    model_size=self.cv_service.model_size if self.cv_service else "l",
                    frame_skip=self.frame_skip,
                )
                try:
                    await self.storage_service.save_benchmark(bench_result)
                except Exception as e:
                    logger.warning(f"Failed to save benchmark: {e}")
                result["benchmark"] = {
                    "total_time_seconds": bench_result.total_time_seconds,
                    "realtime_ratio": bench_result.realtime_ratio,
                    "fps_effective": bench_result.fps_effective,
                    "peak_memory_mb": bench_result.peak_memory_mb,
                    "peak_gpu_memory_mb": bench_result.peak_gpu_memory_mb,
                    "stages": {
                        "enhancement": bench_result.stage_enhancement_seconds,
                        "detection": bench_result.stage_detection_seconds,
                        "tracking": bench_result.stage_tracking_seconds,
                        "analysis": bench_result.stage_analysis_seconds,
                        "advanced_metrics": bench_result.stage_advanced_metrics_seconds,
                        "save": bench_result.stage_save_seconds,
                    },
                }

            self._bridge.analysisComplete.emit(result)
            return json.dumps(result)
        except Exception as e:
            logger.exception(f"Analysis failed: {e}")
            self._bridge.analysisError.emit(ErrorSanitizer.sanitize_error(e))
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    def get_overlay_data(self, match_id, timestamp):
        data = self._overlay_cache.get(match_id)
        if not data:
            return "{}"
        lo, hi = 0, len(data) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if data[mid]["t"] <= timestamp:
                lo = mid
            else:
                hi = mid - 1
        return json.dumps(data[lo])

    async def get_all_matches(self):
        try:
            self._check_rate_limit()
            matches = await self.storage_service.get_all_matches()
            return json.dumps(matches)
        except Exception as e:
            logger.error(f"Failed to get matches: {e}")
            return json.dumps([])

    async def get_match_events(self, match_id):
        try:
            self._check_rate_limit()
            match_id = SecurityValidator.validate_match_id(match_id)
            events = await self.storage_service.get_match_events(match_id)
            return json.dumps(events)
        except Exception as e:
            logger.error(f"Failed to get events: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_match_players(self, match_id):
        """Player roster for one match (track_id/name/team/jersey) -- used
        by the 3D pitch view to label players (see app-3d.js loadPlayerNameMap)."""
        try:
            self._check_rate_limit()
            match_id = SecurityValidator.validate_match_id(match_id)
            players = await self.storage_service.get_match_players(match_id)
            return json.dumps({"players": players})
        except Exception as e:
            logger.error(f"get_match_players failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ── Event Review / Correction (Phase 2) ──────────────────────

    async def get_unreviewed_events(self, match_id, min_confidence=0.0, max_confidence=0.7):
        self._check_rate_limit()
        """Get auto-detected events with low confidence that need review.

        Returns events sorted by confidence ascending (lowest first),
        with auto_detected flag and metadata parsed from JSON.
        """
        try:
            match_id = SecurityValidator.validate_match_id(match_id)
            events = await self.storage_service.get_match_events(match_id)
            unreviewed = []
            for ev in events:
                if ev.get("user_corrected", 0):
                    continue
                conf = ev.get("confidence", 0.0)
                if conf is None:
                    conf = 0.0
                if conf < min_confidence or conf > max_confidence:
                    continue
                meta = ev.get("metadata", "{}")
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except (json.JSONDecodeError, TypeError):
                        meta = {}
                ev["_meta"] = meta
                ev["_needs_review"] = conf < 0.5
                unreviewed.append(ev)

            unreviewed.sort(key=lambda e: e.get("confidence", 0) or 0)
            return json.dumps({"success": True, "events": unreviewed, "total": len(unreviewed)})
        except Exception as e:
            logger.error(f"get_unreviewed_events failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_detection_summary(self, match_id):
        self._check_rate_limit()
        """Get auto-detection stats by event type with average confidence."""
        try:
            match_id = SecurityValidator.validate_match_id(match_id)
            events = await self.storage_service.get_match_events(match_id)

            by_type = {}
            for ev in events:
                etype = ev.get("event_type", "unknown")
                conf = ev.get("confidence", 0) or 0
                corrected = ev.get("user_corrected", 0)
                if etype not in by_type:
                    by_type[etype] = {
                        "count": 0,
                        "corrected": 0,
                        "total_confidence": 0.0,
                        "avg_confidence": 0.0,
                    }
                by_type[etype]["count"] += 1
                by_type[etype]["total_confidence"] += conf
                if corrected:
                    by_type[etype]["corrected"] += 1

            for _etype, stats in by_type.items():
                stats["avg_confidence"] = (
                    round(stats["total_confidence"] / stats["count"], 3) if stats["count"] else 0
                )
                del stats["total_confidence"]

            total = len(events)
            corrected = sum(1 for e in events if e.get("user_corrected", 0))
            return json.dumps(
                {
                    "success": True,
                    "by_type": by_type,
                    "total": total,
                    "corrected": corrected,
                    "unreviewed": total - corrected,
                }
            )
        except Exception as e:
            logger.error(f"get_detection_summary failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def submit_event_correction(self, match_id, event_id, action, corrections_json):
        self._check_rate_limit()
        """Submit a correction for an auto-detected event.

        Actions: 'confirm' (mark reviewed), 'reject' (delete event),
                 'edit' (update event fields + save correction record)
        """
        try:
            match_id = SecurityValidator.validate_match_id(match_id)
            event_id = (
                SecurityValidator.validate_match_id(event_id)
                if hasattr(SecurityValidator, "validate_match_id")
                else int(event_id)
            )

            if action == "confirm":
                ok = await self.storage_service.update_event(event_id, {"user_corrected": 1})
                return json.dumps({"success": bool(ok), "action": "confirmed"})

            elif action == "reject":
                ok = await self.storage_service.delete_event(event_id)
                return json.dumps({"success": bool(ok), "action": "deleted"})

            elif action == "edit":
                corrections = json.loads(corrections_json) if corrections_json else {}
                allowed_updates = {}
                original = None
                events = await self.storage_service.get_match_events(match_id)
                for ev in events:
                    if ev.get("id") == event_id:
                        original = dict(ev)
                        break

                if "event_type" in corrections:
                    allowed_updates["event_type"] = corrections["event_type"]
                if "team" in corrections:
                    allowed_updates["team"] = corrections["team"]
                if "completed" in corrections:
                    allowed_updates["completed"] = bool(corrections["completed"])
                if "confidence" in corrections:
                    allowed_updates["confidence"] = float(corrections["confidence"])
                if "metadata" in corrections:
                    allowed_updates["metadata"] = corrections["metadata"]

                allowed_updates["user_corrected"] = 1
                ok = await self.storage_service.update_event(event_id, allowed_updates)

                if ok and original:
                    await self.storage_service.save_correction(
                        event_id=event_id,
                        correction_type="edit",
                        original_value={
                            "event_type": original.get("event_type"),
                            "team": original.get("team"),
                        },
                        corrected_value=corrections,
                    )

                return json.dumps({"success": bool(ok), "action": "edited"})

            else:
                return json.dumps({"error": f"Unknown action: {action}"})
        except Exception as e:
            logger.error(f"submit_event_correction failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_video_path(self, match_id):
        try:
            self._check_rate_limit()
            match_id = SecurityValidator.validate_match_id(match_id)
            match = await self.storage_service.get_match(match_id)
            if not match or not match.get("video_path"):
                return json.dumps({"error": "No video found"})
            return json.dumps({"path": match["video_path"]})
        except Exception as e:
            logger.error(f"get_video_path failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def generate_report(self, match_id, language, summary):
        try:
            self._check_rate_limit()
            match_id = SecurityValidator.validate_match_id(match_id)
            language = SecurityValidator.sanitize_string(language, max_length=10)
            if language not in ("en", "ar"):
                language = "en"
            report = await self.llm_service.generate_coach_report(
                match_analysis_summary=summary, language=language
            )
            await self.storage_service.save_report(
                match_id=match_id,
                language=language,
                report_text=report,
                llm_provider=self.llm_service.config.provider,
            )
            return report
        except Exception as e:
            logger.exception(f"Report generation failed: {e}")
            return f"Error generating report: {ErrorSanitizer.sanitize_error(e)}"

    async def get_knowledge_base_stats(self):
        """Knowledge-base stats; the only slot in this handler that let an
        exception escape un-JSON'd (the Qt slot would return undefined to
        the UI instead of an error payload)."""
        try:
            if self.knowledge_service is None:
                return json.dumps({"error": "Knowledge service not available"})
            await self.knowledge_service.initialize()
            return json.dumps(self.knowledge_service.stats)
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def check_llm_availability(self):
        try:
            self._check_rate_limit()
            ollama_available = False
            for provider in self.llm_service.providers:
                if hasattr(provider, "is_available") and await provider.is_available():
                    ollama_available = True
                    break
            return json.dumps(
                {
                    "ollama": ollama_available,
                    "provider": self.llm_service.config.provider,
                    "model": (
                        self.llm_service.config.ollama_model
                        if self.llm_service.config.provider == "ollama"
                        else "external"
                    ),
                }
            )
        except Exception as e:
            logger.error(f"LLM check failed: {e}")
            return json.dumps({"ollama": False, "error": str(e)})

    # ================================================================
    # Player Profile operations
    # ================================================================

    async def create_player_profile(self, name, jersey, number, position):
        try:
            self._check_rate_limit()
            name = SecurityValidator.sanitize_string(name, max_length=100)
            number = SecurityValidator.validate_jersey_number(number)
            position = SecurityValidator.sanitize_string(position, max_length=50)
            if self.player_profile_service is None:
                return json.dumps({"error": "PlayerProfileService not available"})
            profile = await self.player_profile_service.create_profile(
                display_name=name,
                jersey_number=number,
                preferred_position=position,
            )
            return json.dumps(
                {"success": True, "profile_id": profile.id, "global_id": profile.global_id}
            )
        except Exception as e:
            logger.error(f"Create profile failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_all_player_profiles(self):
        try:
            self._check_rate_limit()
            if self.player_profile_service is None:
                return json.dumps({"profiles": []})
            profiles = await self.player_profile_service.get_all_profiles()
            return json.dumps(
                {
                    "profiles": [
                        {
                            "id": p.id,
                            "name": p.display_name,
                            "jersey": p.jersey_number,
                            "position": p.preferred_position,
                        }
                        for p in profiles
                    ]
                }
            )
        except Exception as e:
            logger.error(f"Get profiles failed: {e}")
            return json.dumps({"profiles": []})

    async def get_player_stats(self, player_id):
        """Career stats for one player profile, aggregated across every
        match appearance -- used by the player-vs-player comparison radar
        (see app.js renderPlayerComparison / statKeys).

        Note: 'sprints' is always 0 -- sprint counts require re-deriving
        from raw per-match tracking_frames (physical_metrics.py), which
        isn't aggregated at the profile level anywhere yet. Both players
        show 0 on that radar axis rather than a fabricated number.
        """
        try:
            self._check_rate_limit()
            if self.player_profile_service is None:
                return json.dumps({"error": "PlayerProfileService not available"})
            player_id = SecurityValidator.validate_match_id(player_id)
            profile = await self.player_profile_service.get_profile(player_id)
            if profile is None:
                return json.dumps({"error": "Player not found"})
            appearances = await self.player_profile_service.get_profile_appearances(player_id)

            return json.dumps(
                {
                    "id": profile.id,
                    "name": profile.display_name or f"Player {profile.id}",
                    "jersey": profile.jersey_number,
                    "position": profile.preferred_position,
                    "matches_played": len(appearances),
                    "passes": sum(a.passes_completed for a in appearances),
                    "shots": sum(a.shots for a in appearances),
                    "tackles": sum(a.tackles for a in appearances),
                    "sprints": 0,
                    "distance": round(sum(a.distance_covered_m for a in appearances), 1),
                    "xg": round(sum(a.xg for a in appearances), 3),
                }
            )
        except Exception as e:
            logger.error(f"get_player_stats failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def compare_players(self, player_a_id, player_b_id):
        """Single-round-trip variant of get_player_stats for both players at
        once (the fast path app.js's handlePlayerCompare prefers when
        available, falling back to two sequential get_player_stats calls
        otherwise)."""
        try:
            self._check_rate_limit()
            a_json = await self.get_player_stats(player_a_id)
            b_json = await self.get_player_stats(player_b_id)
            return json.dumps(
                {
                    "player_a": json.loads(a_json),
                    "player_b": json.loads(b_json),
                }
            )
        except Exception as e:
            logger.error(f"compare_players failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_face_gallery(self):
        try:
            profiles = await self.storage_service.get_all_player_profiles()
            result = []
            for p in profiles:
                result.append(
                    {
                        "id": p["id"],
                        "display_name": p.get("display_name", ""),
                        "jersey_number": p.get("jersey_number"),
                        "team": p.get("team", "home"),
                        "has_face": bool(p.get("face_embedding")),
                        "face_confidence": p.get("face_confidence", 0.0),
                        "photo_path": p.get("photo_path", ""),
                    }
                )
            return json.dumps({"success": True, "profiles": result})
        except Exception as e:
            logger.error(f"get_face_gallery failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def upload_face_photo(self, photo_path, display_name, jersey_number):
        try:
            # Image-specific allowlist: the generic data one (.json/.xml/.csv)
            # would reject every legitimate photo.
            SecurityValidator.validate_image_path(photo_path)
            if self.face_recognition_service is None:
                return json.dumps(
                    {
                        "success": False,
                        "error": "FaceRecognitionService not available (insightface not installed)",
                    }
                )
            import cv2

            img = cv2.imread(photo_path)
            if img is None:
                return json.dumps({"success": False, "error": "Could not read image"})
            faces = self.face_recognition_service.detect_faces(img)
            if not faces:
                return json.dumps({"success": False, "error": "No face detected in photo"})
            best = max(faces, key=lambda f: f["confidence"])
            global_id = f"upload_{jersey_number}_{display_name.replace(' ', '_')}"
            profile = await self.storage_service.save_player_profile(
                {
                    "global_id": global_id,
                    "display_name": display_name,
                    "jersey_number": jersey_number,
                    "team": "home",
                    "face_embedding": json.dumps(best["embedding"]),
                    "face_confidence": best["confidence"],
                }
            )
            return json.dumps(
                {
                    "success": True,
                    "profile_id": profile,
                    "display_name": display_name,
                    "confidence": best["confidence"],
                }
            )
        except Exception as e:
            logger.error(f"upload_face_photo failed: {e}")
            return json.dumps({"success": False, "error": ErrorSanitizer.sanitize_error(e)})

    async def match_faces_in_match(self, match_id):
        try:
            if self.face_recognition_service is None:
                return json.dumps(
                    {
                        "success": False,
                        "error": "FaceRecognitionService not available (install insightface)",
                    }
                )
            track_data = self._tracking_cache.get(match_id)
            if not track_data:
                return json.dumps(
                    {"success": False, "error": "No tracking data cached. Run analysis first."}
                )
            profiles = await self.storage_service.get_all_player_profiles()
            identified = self.face_recognition_service.identify_players_in_match(
                profiles, track_data
            )
            return json.dumps({"success": True, "identified_count": len(identified)})
        except Exception as e:
            logger.error(f"match_faces_in_match failed: {e}")
            return json.dumps({"success": False, "error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Multi-match & quality
    # ================================================================

    async def compare_matches(self, match_id_1, match_id_2, focus):
        try:
            self._check_rate_limit()
            m1 = SecurityValidator.validate_match_id(match_id_1)
            m2 = SecurityValidator.validate_match_id(match_id_2)
            focus = SecurityValidator.sanitize_string(focus, max_length=50)
            if self.multi_match_service is None:
                return json.dumps({"error": "MultiMatchAnalysisService not available"})
            comparison = await self.multi_match_service.compare_matches(m1, m2)
            return json.dumps(
                {
                    "match_1": comparison.match_1_name,
                    "match_2": comparison.match_2_name,
                    "possession_diff": comparison.possession_diff,
                    "shots_diff": comparison.shots_diff,
                    "formation_diff": comparison.formation_diff,
                    "key_differences": comparison.key_differences,
                    "tactical_evolution": comparison.tactical_evolution,
                }
            )
        except Exception as e:
            logger.error(f"Compare matches failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_match_quality_report(self, match_id_str):
        try:
            self._check_rate_limit()
            match_id = SecurityValidator.validate_match_id(match_id_str)
            if self.quality_scoring_service is None:
                return json.dumps({"error": "QualityScoringService not available"})
            scores = await self.quality_scoring_service.get_scores(match_id)
            if scores is None:
                return json.dumps({"error": "No quality scores found"})
            return json.dumps(
                {
                    "overall": scores.overall,
                    "tracking": scores.tracking,
                    "events": scores.events,
                    "homography": scores.homography,
                    "team_assignment": scores.team_assignment,
                }
            )
        except Exception as e:
            logger.error(f"Quality report failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Team & visualization operations
    # ================================================================

    async def swap_teams(self, match_id):
        try:
            self._check_rate_limit()
            match_id = SecurityValidator.validate_match_id(match_id)
            match = await self.storage_service.get_match(match_id)
            if not match:
                return json.dumps({"error": "Match not found"})
            home = match.get("home_team", "Home")
            away = match.get("away_team", "Away")
            await self.storage_service.update_match_teams(
                match_id=match_id, home_team=away, away_team=home
            )
            logger.info(f"Swapped teams for match {match_id}: {home} ↔ {away}")
            return json.dumps({"success": True, "home": away, "away": home})
        except Exception as e:
            logger.error(f"swap_teams failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def generate_visualizations(self, match_id):
        try:
            self._check_rate_limit()
            match_id = SecurityValidator.validate_match_id(match_id)
            if self.visualization_service is None:
                return json.dumps({"error": "VisualizationService not available"})
            events = await self.storage_service.get_match_events(match_id)
            pass_events = [e for e in events if e.get("type") == "pass" and e.get("completed")]
            if not pass_events:
                return json.dumps({"error": "No pass events to visualize"})
            pass_network_path = None
            heatmap_path = None
            if pass_events:
                player_positions = {}
                for pe in pass_events:
                    meta = pe.get("metadata", {})
                    src = pe.get("from_track_id")
                    dst = pe.get("to_track_id")
                    if src is not None:
                        sx = meta.get("start_x_pct", 0.5) * 105.0
                        sy = meta.get("start_y_pct", 0.5) * 68.0
                        if src not in player_positions:
                            player_positions[src] = (sx, sy)
                    if dst is not None:
                        ex = meta.get("end_x_pct", 0.5) * 105.0
                        ey = meta.get("end_y_pct", 0.5) * 68.0
                        if dst not in player_positions:
                            player_positions[dst] = (ex, ey)
                pass_network_path = await self.visualization_service.generate_pass_network(
                    pass_events=pass_events,
                    player_positions=player_positions,
                    output_name=f"pass_network_{match_id}.png",
                )
                positions_list = list(player_positions.values())
                if positions_list:
                    heatmap_path = await self.visualization_service.generate_heatmap(
                        positions=positions_list,
                        output_name=f"heatmap_{match_id}.png",
                    )
            return json.dumps(
                {
                    "success": True,
                    "pass_network": str(pass_network_path) if pass_network_path else None,
                    "heatmap": str(heatmap_path) if heatmap_path else None,
                }
            )
        except Exception as e:
            logger.error(f"generate_visualizations failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Set-piece analysis
    # ================================================================

    async def check_pose_status(self):
        if self.pose_analysis_service is None:
            return json.dumps({"available": False})
        try:
            return json.dumps({"available": self.pose_analysis_service.available})
        except Exception:
            return json.dumps({"available": False})

    async def get_activity_summary(self, track_id):
        if self.pose_analysis_service is None:
            return json.dumps({"summary": {}})
        try:
            summary = self.pose_analysis_service.summarize_activity(track_id)
            return json.dumps({"summary": summary})
        except Exception as e:
            logger.error(f"get_activity_summary failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_activity_segments(self, track_id):
        if self.pose_analysis_service is None:
            return json.dumps({"segments": []})
        try:
            segs = self.pose_analysis_service.get_activity_segments(track_id)
            return json.dumps(
                {
                    "segments": [
                        {
                            "activity": s.activity,
                            "start_time": s.start_time,
                            "end_time": s.end_time,
                            "duration_s": s.duration_s,
                            "avg_speed_kmh": s.avg_speed_kmh,
                        }
                        for s in segs
                    ]
                }
            )
        except Exception as e:
            logger.error(f"get_activity_segments failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # MuJoCo ball trajectory
    # ================================================================

    async def check_mujoco_status(self):
        if self.mujoco_ball_service is None:
            return json.dumps({"available": False, "uses_mujoco": False})
        return json.dumps(
            {
                "available": self.mujoco_ball_service.available,
                "uses_mujoco": self.mujoco_ball_service.uses_mujoco,
            }
        )

    async def get_setpiece_presets(self):
        if self.mujoco_ball_service is None:
            return json.dumps({"presets": []})
        return json.dumps({"presets": self.mujoco_ball_service.get_preset_setpieces()})

    async def simulate_trajectory(
        self, initial_speed, launch_angle_deg, spin_rps, direction_deg, duration_s
    ):
        if self.mujoco_ball_service is None:
            return json.dumps({"error": "MuJoCo service not initialized"})
        try:
            result = await self.mujoco_ball_service.simulate(
                initial_speed=float(initial_speed),
                launch_angle_deg=float(launch_angle_deg),
                spin_rps=float(spin_rps),
                direction_deg=float(direction_deg),
                duration_s=float(duration_s) if duration_s > 0 else 2.5,
            )
            return json.dumps(
                {
                    "method": result.method,
                    "landing_x": result.landing_x,
                    "landing_y": result.landing_y,
                    "max_height": result.max_height,
                    "duration_s": result.duration_s,
                    "final_speed_mps": result.final_speed_mps,
                    "points": [
                        {"t": p.t, "x": p.x, "y": p.y, "z": p.z} for p in result.points[::5]
                    ],
                }
            )
        except Exception as e:
            logger.error(f"simulate_trajectory failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # FluidX3D
    # ================================================================

    async def check_fluidx3d_status(self):
        if self.fluidx3d_service is None:
            return json.dumps(
                {
                    "available": False,
                    "license_notice": "FluidX3D is free for non-commercial use only.",
                }
            )
        return json.dumps(
            {
                "available": self.fluidx3d_service.available,
                "license_notice": self.fluidx3d_service.license_notice,
            }
        )

    async def simulate_ball_cfd(self, wind_speed, spin_rps, ball_radius):
        if self.fluidx3d_service is None:
            return json.dumps({"error": "FluidX3D service not initialized"})
        try:
            result = await self.fluidx3d_service.simulate_ball_aerodynamics(
                ball_radius=float(ball_radius) if ball_radius > 0 else 0.11,
                wind_speed=float(wind_speed),
                spin_rps=float(spin_rps),
            )
            return json.dumps(
                {
                    "success": result.success,
                    "method": result.method,
                    "notes": result.notes,
                    "error": result.error,
                }
            )
        except Exception as e:
            logger.error(f"simulate_ball_cfd failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Weather
    # ================================================================

    async def check_weather_status(self):
        if self.weather_service is None:
            return json.dumps({"available": False})
        return json.dumps(
            {
                "available": self.weather_service.available,
                "has_video_classifier": self.weather_service.has_video_classifier,
            }
        )

    async def fetch_match_weather(self, latitude, longitude, date, is_forecast):
        if self.weather_service is None:
            return json.dumps({"error": "Weather service not initialized"})
        try:
            result = await self.weather_service.fetch_conditions(
                latitude, longitude, date, is_forecast=is_forecast
            )
            if result is None:
                return json.dumps({"error": "Failed to fetch weather"})
            return json.dumps(
                {
                    "temperature_c": result.temperature_c,
                    "precipitation_mm": result.precipitation_mm,
                    "wind_speed_kmh": result.wind_speed_kmh,
                    "humidity_pct": result.humidity_pct,
                    "conditions": result.conditions,
                    "pitch_state": result.pitch_state.value,
                    "source": result.source.value,
                    "is_daylight": result.is_daylight,
                }
            )
        except Exception as e:
            logger.error(f"fetch_match_weather failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def set_manual_weather(self, temperature, precipitation, wind, humidity, conditions):
        if self.weather_service is None:
            return json.dumps({"error": "Weather service not initialized"})
        from kawkab.services.weather_service import WeatherService

        result = WeatherService.from_manual(
            temperature_c=temperature,
            precipitation_mm=precipitation,
            wind_speed_kmh=wind,
            humidity_pct=humidity,
            conditions=conditions,
        )
        impact = WeatherService.analyze_impact(result)
        return json.dumps(
            {
                "conditions": {
                    "temperature_c": result.temperature_c,
                    "precipitation_mm": result.precipitation_mm,
                    "wind_speed_kmh": result.wind_speed_kmh,
                    "humidity_pct": result.humidity_pct,
                    "conditions": result.conditions,
                    "pitch_state": result.pitch_state.value,
                },
                "impact": {
                    "goals_delta": impact.expected_goals_delta,
                    "passing_delta_pct": impact.passing_accuracy_delta_pct,
                    "sprint_delta_pct": impact.sprint_distance_delta_pct,
                    "set_piece_advantage": impact.set_piece_advantage,
                    "notes": impact.notes,
                },
            }
        )

    async def analyze_weather_impact(self, temperature, precipitation, wind, humidity, conditions):
        if self.weather_service is None:
            return json.dumps({"error": "Weather service not initialized"})
        try:
            from kawkab.services.weather_service import WeatherService

            wc = WeatherService.from_manual(
                temperature_c=temperature,
                precipitation_mm=precipitation,
                wind_speed_kmh=wind,
                humidity_pct=humidity,
                conditions=conditions,
            )
            impact = WeatherService.analyze_impact(wc)
            return json.dumps(
                {
                    "goals_delta": impact.expected_goals_delta,
                    "passing_delta_pct": impact.passing_accuracy_delta_pct,
                    "sprint_delta_pct": impact.sprint_distance_delta_pct,
                    "set_piece_advantage": impact.set_piece_advantage,
                    "notes": impact.notes,
                }
            )
        except Exception as e:
            logger.error(f"analyze_weather_impact failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def check_raindrop_status(self):
        if self.weather_service is None:
            return json.dumps({"available": False})
        return json.dumps(
            {
                "available": self.weather_service.has_raindrop_detector,
                "has_cnn": (
                    self.weather_service._raindrop_service.has_cnn
                    if self.weather_service._raindrop_service
                    else False
                ),
            }
        )

    async def check_weather_classifier_status(self):
        if self.weather_service is None:
            return json.dumps({"available": False})
        return json.dumps(
            {
                "available": self.weather_service.has_multi_class_classifier,
                "has_cnn": (
                    self.weather_service._weather_classifier.has_cnn
                    if self.weather_service._weather_classifier
                    else False
                ),
            }
        )

    async def detect_raindrops_in_video(self, video_path, sample_every_n, max_frames):
        if self.weather_service is None or not self.weather_service.has_raindrop_detector:
            return json.dumps({"error": "Raindrop service not available"})
        try:
            validated_path = SecurityValidator.validate_video_path(str(video_path))
            result = self.weather_service._raindrop_service.detect_from_video_file(
                str(validated_path), sample_every_n, max_frames
            )
            return json.dumps(
                {
                    "frame_count": result.frame_count,
                    "raindrop_count": result.raindrop_count,
                    "raindrop_density": result.raindrop_density,
                    "avg_confidence": result.avg_confidence,
                    "is_rainy": result.is_rainy,
                    "method": result.method,
                }
            )
        except Exception as e:
            logger.error(f"detect_raindrops_in_video failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def classify_video_weather(self, video_path):
        if self.weather_service is None:
            return json.dumps({"error": "Weather service not available"})
        try:
            validated_path = SecurityValidator.validate_video_path(str(video_path))
            import cv2

            cap = cv2.VideoCapture(str(validated_path))
            if not cap.isOpened():
                return json.dumps({"error": f"Could not open video: {video_path}"})
            frames = []
            frame_idx = 0
            sampled = 0
            while sampled < 15:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % 30 == 0:
                    frames.append(frame)
                    sampled += 1
                frame_idx += 1
            cap.release()
            if not frames:
                return json.dumps({"error": "No frames read from video"})
            result = self.weather_service.classify_from_video_advanced(frames)
            return json.dumps(result)
        except Exception as e:
            logger.error(f"classify_video_weather failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Phase 2.3 — Tactical Periods (from events)
    # ================================================================

    async def get_tactical_periods(self, match_id):
        try:
            match_id = SecurityValidator.validate_match_id(match_id)
            events = await self.storage_service.get_match_events(match_id)
            if not events:
                return json.dumps({"phases": []})

            total_time = max(1.0, events[-1].get("timestamp", 0) - events[0].get("timestamp", 0))
            _ = {"settled_possession", "transition", "counter", "set_piece", "direct"}
            phases = []
            window = 5.0
            _ = events[0].get("timestamp", 0)

            i = 0
            while i < len(events):
                t = events[i].get("timestamp", 0)
                window_end = t + window
                window_events = [e for e in events if t <= e.get("timestamp", 0) < window_end]
                types_in_window = [e.get("event_type", "") for e in window_events]
                type_counts = {}
                for et in types_in_window:
                    type_counts[et] = type_counts.get(et, 0) + 1

                if any(
                    "corner" in e.get("event_type", "")
                    or "free_kick" in e.get("event_type", "")
                    or e.get("event_type", "") == "throw_in"
                    for e in window_events
                ):
                    label = "set_piece"
                elif type_counts.get("counter", 0) >= 2:
                    label = "transition"
                elif type_counts.get("pass", 0) >= 3:
                    label = "settled_possession"
                elif type_counts.get("tackle", 0) >= 2 or type_counts.get("interception", 0) >= 2:
                    label = "transition"
                elif type_counts.get("carry", 0) >= 2 and type_counts.get("shot", 0) > 0:
                    label = "direct"
                else:
                    label = "settled_possession"

                phase_end = window_end
                j = i + 1
                while j < len(events):
                    if events[j].get("timestamp", 0) > phase_end:
                        break
                    j += 1

                if phases and phases[-1]["label"] == label:
                    phases[-1]["end"] = max(phases[-1]["end"], min(t + window, total_time))
                    phases[-1]["duration_s"] = round(phases[-1]["end"] - phases[-1]["start"], 1)
                else:
                    phases.append(
                        {
                            "start": round(t, 1),
                            "end": round(min(t + window, total_time), 1),
                            "label": label,
                            "duration_s": round(min(window, total_time - t), 1),
                        }
                    )
                i = j

            counts = {}
            for p in phases:
                counts[p["label"]] = counts.get(p["label"], 0) + p["duration_s"]
            return json.dumps(
                {
                    "phases": phases,
                    "press_pct": round(counts.get("transition", 0) / total_time * 100, 1)
                    if total_time
                    else 0,
                    "settled_possession_pct": round(
                        counts.get("settled_possession", 0) / total_time * 100, 1
                    )
                    if total_time
                    else 0,
                    "transition_pct": round(counts.get("transition", 0) / total_time * 100, 1)
                    if total_time
                    else 0,
                }
            )
        except Exception as e:
            logger.error(f"get_tactical_periods failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Phase 2.4 — Formation Analysis (from events)
    # ================================================================

    async def analyze_formation(self, match_id):
        try:
            from kawkab.core.formation_analysis import FormationAnalyzer

            match_id = SecurityValidator.validate_match_id(match_id)
            events = await self.storage_service.get_match_events(match_id)
            _ = await self.storage_service.get_match_players(match_id)

            home_positions = {}
            away_positions = {}
            for ev in events:
                meta = ev.get("metadata", {})
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = {}
                tid = ev.get("from_track_id") or ev.get("player_track_id", 0)
                team = ev.get("team", "home")
                x = meta.get("start_x_pct", 0.5) if meta else 0.5
                y = meta.get("start_y_pct", 0.5) if meta else 0.5
                if team == "home":
                    home_positions[tid] = (x, y)
                else:
                    away_positions[tid] = (x, y)

            analyzer = FormationAnalyzer()
            result = {}
            for side, pos in [("home", home_positions), ("away", away_positions)]:
                if not pos:
                    result[side] = {"formation": "unknown"}
                    continue
                positions = list(pos.values())
                frames = [
                    {
                        "timestamp": 0,
                        "home_positions": positions if side == "home" else [],
                        "away_positions": positions if side == "away" else [],
                        "possession": True,
                        "ball_pos": None,
                    }
                ]
                report = analyzer.analyze_team_shape(frames, team=side)
                result[side] = report.to_dict()

            return json.dumps({"success": True, **result})
        except Exception as e:
            logger.error(f"analyze_formation failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Tactical Shape Analysis
    # ================================================================

    async def analyze_tactical_shapes(self, match_id):
        try:
            from kawkab.core.tactical_shape_analyzer import TacticalShapeAnalyzer

            match_id = SecurityValidator.validate_match_id(match_id)
            events = await self.storage_service.get_match_events(match_id)
            if not events:
                return json.dumps({"home": {}, "away": {}})
            analyzer = TacticalShapeAnalyzer()
            home_report = analyzer.analyze_shapes(events, team="home")
            away_report = analyzer.analyze_shapes(events, team="away")
            return json.dumps(
                {"success": True, "home": home_report.to_dict(), "away": away_report.to_dict()}
            )
        except Exception as e:
            logger.error(f"analyze_tactical_shapes failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Pressing Classification
    # ================================================================

    async def classify_pressing(self, match_id):
        try:
            from kawkab.core.pressing_classifier import classify_pressing_system

            match_id = SecurityValidator.validate_match_id(match_id)
            events = await self.storage_service.get_match_events(match_id)
            if not events:
                return json.dumps({"home": {}, "away": {}})
            home_report = classify_pressing_system(events, team="home")
            away_report = classify_pressing_system(events, team="away")
            return json.dumps(
                {"success": True, "home": home_report.to_dict(), "away": away_report.to_dict()}
            )
        except Exception as e:
            logger.error(f"classify_pressing failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Comprehensive Tactical Report
    # ================================================================

    async def get_tactical_report(self, match_id):
        try:
            from kawkab.core.tactical_report import generate_tactical_report

            match_id = SecurityValidator.validate_match_id(match_id)
            events = await self.storage_service.get_match_events(match_id)
            if not events:
                return json.dumps({"error": "No events"})
            report = generate_tactical_report(events, match_id=match_id)
            return json.dumps({"success": True, **report.to_dict()})
        except Exception as e:
            logger.error(f"get_tactical_report failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Phase 3 — AI NL Query (ask a question about a match)
    # ================================================================

    async def ask_llm(self, match_id, question):
        try:
            self._check_rate_limit()
            match_id = SecurityValidator.validate_match_id(match_id)
            question = SecurityValidator.sanitize_string(question, max_length=500)
            events = await self.storage_service.get_match_events(match_id)
            match_data = await self.storage_service.get_match(match_id)

            event_summary = {}
            for ev in events:
                et = ev.get("event_type", "unknown")
                event_summary[et] = event_summary.get(et, 0) + 1

            home_team = (match_data or {}).get("home_team", "Home")
            away_team = (match_data or {}).get("away_team", "Away")

            context = (
                f"Match: {home_team} vs {away_team}\n"
                f"Total events: {len(events)}\n"
                f"Event breakdown: {json.dumps(event_summary)}\n"
            )
            system_prompt = (
                "You are a professional football analyst coach. Answer questions about a match "
                "based on the data provided. Be specific, tactical, and concise (max 3 paragraphs). "
                "If the data doesn't support a claim, say so."
            )
            full_prompt = f"{context}\n\nQuestion: {question}\n\nAnalysis:"
            answer = await self.llm_service.generate(full_prompt, system=system_prompt)
            return json.dumps({"success": True, "answer": answer, "event_summary": event_summary})
        except Exception as e:
            logger.error(f"ask_llm failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Phase 4 — Player Rating Index 0-100
    # ================================================================

    async def get_player_rating(self, match_id, track_id):
        try:
            self._check_rate_limit()
            match_id = SecurityValidator.validate_match_id(match_id)
            track_id = SecurityValidator.validate_match_id(track_id)
            events = await self.storage_service.get_match_events(match_id)
            _ = await self.storage_service.get_match_players(match_id)

            player_events = [
                e
                for e in events
                if e.get("from_track_id") == track_id or e.get("player_track_id") == track_id
            ]
            if not player_events:
                return json.dumps({"rating": 0, "components": {}})

            passes = sum(1 for e in player_events if e.get("event_type") == "pass")
            completed = sum(
                1 for e in player_events if e.get("event_type") == "pass" and e.get("completed")
            )
            shots = sum(1 for e in player_events if e.get("event_type") == "shot")
            tackles = sum(1 for e in player_events if e.get("event_type") == "tackle")
            _ = sum(1 for e in player_events if e.get("event_type") == "interception")
            carries = sum(1 for e in player_events if e.get("event_type") == "carry")
            dribbles = sum(1 for e in player_events if e.get("event_type") == "dribble")
            goals = sum(1 for e in player_events if e.get("event_type") == "goal")

            pass_acc = completed / max(passes, 1)
            shot_score = min(shots * 20, 100)
            tackle_score = min(tackles * 15, 100)
            carry_score = min(carries * 10, 100)
            dribble_score = min(dribbles * 15, 100)
            goal_score = min(goals * 30, 100)
            volume = min(len(player_events) * 2, 100)

            components = {
                "pass_accuracy": round(pass_acc * 100, 1),
                "shot_impact": shot_score,
                "tackle_effectiveness": tackle_score,
                "carry_progression": carry_score,
                "dribble_success": dribble_score,
                "goal_contribution": goal_score,
                "event_volume": volume,
            }
            rating = round(
                (
                    pass_acc * 30
                    + shot_score * 0.1
                    + tackle_score * 0.15
                    + carry_score * 0.1
                    + dribble_score * 0.15
                    + goal_score * 0.1
                    + volume * 0.1
                ),
                1,
            )
            return json.dumps({"rating": min(rating, 100), "components": components})
        except Exception as e:
            logger.error(f"get_player_rating failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Phase 4 — Squad Management
    # ================================================================

    async def get_squad_summary(self, match_id):
        try:
            self._check_rate_limit()
            match_id = SecurityValidator.validate_match_id(match_id)
            events = await self.storage_service.get_match_events(match_id)
            players = await self.storage_service.get_match_players(match_id)

            squad = {}
            for p in players or []:
                tid = p.get("track_id", p.get("id", 0))
                name = p.get("name", f"Player #{tid}")
                team = p.get("team", "unknown")
                if team not in squad:
                    squad[team] = []
                p_events = [
                    e
                    for e in events
                    if e.get("from_track_id") == tid or e.get("player_track_id") == tid
                ]
                squad[team].append(
                    {
                        "track_id": tid,
                        "name": name,
                        "jersey": p.get("jersey_number", ""),
                        "position": p.get("position", ""),
                        "events": len(p_events),
                        "passes": sum(1 for e in p_events if e.get("event_type") == "pass"),
                        "shots": sum(1 for e in p_events if e.get("event_type") == "shot"),
                        "tackles": sum(1 for e in p_events if e.get("event_type") == "tackle"),
                    }
                )

            return json.dumps(
                {"success": True, "squad": squad, "total_players": len(players or [])}
            )
        except Exception as e:
            logger.error(f"get_squad_summary failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Phase 6 Sprint 1 — Injury Risk
    # ================================================================
    # NOTE: this handler used to be defined TWICE (ruff F811) — the first
    # read workload_d1..28 player fields that are never populated anywhere,
    # the second fabricated ACWR from a synthetic sawtooth. Both deleted;
    # this canonical version uses real GPS-import data (same source as
    # get_squad_injury_report) with an honest insufficient-data category.

    async def get_injury_risk(self, match_id, track_id):
        try:
            self._check_rate_limit()
            match_id = SecurityValidator.validate_match_id(match_id)
            track_id = (
                SecurityValidator.validate_match_id(track_id)
                if isinstance(track_id, str)
                else int(track_id)
            )
            predictor = self.injury_risk_predictor
            players = await self.storage_service.get_match_players(match_id)
            player = None
            for p in players or []:
                if p.get("track_id") == track_id or p.get("id") == track_id:
                    player = p
                    break
            if not player:
                return json.dumps({"error": "Player not found"})

            position = str(player.get("position", "MID"))
            name = player.get("name", f"Player #{track_id}")

            # Real workload source: GPS-import ACWR history (acwr_daily).
            acwr_history = await self.storage_service.get_player_acwr(track_id, limit=1)
            if not acwr_history:
                return json.dumps(
                    {
                        "risk_score": 0.0,
                        "risk_category": "insufficient_data",
                        "acwr": 0.0,
                        "acwr_risk_level": "insufficient data",
                        "recovery_recommendation": "No GPS/workload data on file for this player",
                        "key_factors": [],
                        "player_name": name,
                        "position": position,
                    }
                )

            acwr = float(acwr_history[0].get("acwr", 1.0))
            risk = predictor.predict_injury_risk({"acwr": acwr, "position": position})
            rec = predictor.compute_recovery_recommendation(risk["risk_score"], position)
            return json.dumps(
                {
                    "risk_score": risk["risk_score"],
                    "risk_category": risk["risk_category"],
                    "risk_level": risk["risk_category"],
                    "acwr": round(acwr, 3),
                    "acwr_risk_level": "elevated"
                    if acwr > 1.3
                    else ("low" if acwr < 0.8 else "normal"),
                    "recovery_recommendation": rec,
                    "key_factors": risk["key_risk_factors"],
                    "factors": risk["key_risk_factors"],
                    "player_name": name,
                    "position": position,
                }
            )
        except Exception as e:
            logger.error(f"get_injury_risk failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_squad_injury_report(self, match_id):
        # Real per-player workload comes from GPS imports (storage_service
        # .get_player_acwr, backed by the acwr_daily table) -- the
        # workload_d1..28 player fields this used to read are never
        # populated anywhere in the codebase, and always evaluated to a
        # constant. A player with no GPS import gets an honest
        # "insufficient_data" category instead of a fabricated ACWR.
        try:
            self._check_rate_limit()
            match_id = SecurityValidator.validate_match_id(match_id)
            players = await self.storage_service.get_match_players(match_id)
            predictor = self.injury_risk_predictor
            home_players = []
            away_players = []
            total_risk_home = 0.0
            total_risk_away = 0.0
            high_risk_count = 0
            for p in players or []:
                tid = p.get("track_id")
                if tid is None:
                    continue
                team = p.get("team", "unknown")
                position = str(p.get("position", "MID"))
                name = p.get("name", f"Player #{tid}")
                jersey = p.get("jersey_number", "")
                acwr_history = await self.storage_service.get_player_acwr(tid, limit=1)
                if not acwr_history:
                    entry = {
                        "track_id": tid,
                        "name": name,
                        "jersey": jersey,
                        "position": position,
                        "risk_score": 0.0,
                        "risk_category": "insufficient_data",
                        "acwr": 0.0,
                        "recovery_recommendation": "No GPS/workload data on file for this player",
                        "key_factors": [],
                    }
                else:
                    acwr = float(acwr_history[0].get("acwr", 1.0))
                    risk = predictor.predict_injury_risk({"acwr": acwr, "position": position})
                    rec = predictor.compute_recovery_recommendation(risk["risk_score"], position)
                    entry = {
                        "track_id": tid,
                        "name": name,
                        "jersey": jersey,
                        "position": position,
                        "risk_score": risk["risk_score"],
                        "risk_category": risk["risk_category"],
                        "acwr": round(acwr, 3),
                        "recovery_recommendation": rec,
                        "key_factors": risk["key_risk_factors"],
                    }
                if team in ("home", "Home", "HOME"):
                    home_players.append(entry)
                    total_risk_home += entry["risk_score"]
                else:
                    away_players.append(entry)
                    total_risk_away += entry["risk_score"]
                if entry["risk_category"] in ("high", "critical"):
                    high_risk_count += 1
            avg_home = round(total_risk_home / len(home_players), 3) if home_players else 0
            avg_away = round(total_risk_away / len(away_players), 3) if away_players else 0
            return json.dumps(
                {
                    "success": True,
                    "home_players": home_players,
                    "away_players": away_players,
                    "avg_risk_home": avg_home,
                    "avg_risk_away": avg_away,
                    "high_risk_count": high_risk_count,
                    "total_players": len(home_players) + len(away_players),
                }
            )
        except Exception as e:
            logger.error(f"get_squad_injury_report failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Phase 6 Sprint 1 — Training Plan Auto-Generate
    # ================================================================

    # NOTE: this handler used to be defined TWICE in this class (ruff F811).
    # The first definition (a thin mock-diagnosis wrapper) was dead code —
    # Python kept only the second, richer event-driven version below.
    # The dead one is deleted; the event-driven one is canonical.

    async def _aggregate_season_stats(self) -> dict:
        """Shared match/event aggregation for get_season_summary and
        get_dashboard_stats — walks every match+its events exactly once;
        each caller reshapes the raw totals into its own response shape."""
        matches = await self.storage_service.get_all_matches()
        total = len(matches)
        total_events = 0
        home_wins = 0
        away_wins = 0
        draws = 0
        total_xg = 0.0
        total_shots = 0

        match_list = []
        for m in matches:
            mid = m.get("id", 0)
            events = await self.storage_service.get_match_events(mid) if mid else []
            ev_count = len(events)
            total_events += ev_count
            shots = sum(1 for e in events if e.get("event_type") == "shot")
            total_shots += shots
            xg = sum(
                e.get("metadata", {}).get("xg", 0) if isinstance(e.get("metadata"), dict) else 0
                for e in events
            )
            total_xg += xg
            home_goals = sum(
                1 for e in events if e.get("event_type") == "goal" and e.get("team") == "home"
            )
            away_goals = sum(
                1 for e in events if e.get("event_type") == "goal" and e.get("team") == "away"
            )
            if home_goals > away_goals:
                home_wins += 1
            elif away_goals > home_goals:
                away_wins += 1
            else:
                draws += 1
            match_list.append(
                {
                    "id": mid,
                    "name": m.get("name", ""),
                    "home_team": m.get("home_team", "Home"),
                    "away_team": m.get("away_team", "Away"),
                    "date": m.get("match_date", ""),
                    "events": ev_count,
                    "shots": shots,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "created_at": m.get("created_at", ""),
                }
            )

        return {
            "total_matches": total,
            "total_events": total_events,
            "total_shots": total_shots,
            "total_xg": total_xg,
            "home_wins": home_wins,
            "away_wins": away_wins,
            "draws": draws,
            "matches": match_list,
        }

    async def get_season_summary(self):
        try:
            self._check_rate_limit()
            agg = await self._aggregate_season_stats()
            if agg["total_matches"] == 0:
                return json.dumps({"total_matches": 0})

            return json.dumps(
                {
                    "total_matches": agg["total_matches"],
                    "total_events": agg["total_events"],
                    "total_shots": agg["total_shots"],
                    "avg_xg": round(agg["total_xg"] / max(agg["total_matches"], 1), 3),
                    "matches": agg["matches"],
                }
            )
        except Exception as e:
            logger.error(f"get_season_summary failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_dashboard_stats(self):
        """Lightweight KPI summary for the Dashboard section (see app.js loadDashboard()).

        total_xg is a raw sum across all matches (not pre-averaged) —
        the frontend divides by its own match_count to compute the
        average-per-match KPI shown on the dashboard card.
        """
        try:
            self._check_rate_limit()
            agg = await self._aggregate_season_stats()
            return json.dumps(
                {
                    "match_count": agg["total_matches"],
                    "total_events": agg["total_events"],
                    "total_xg": round(agg["total_xg"], 3),
                    "home_wins": agg["home_wins"],
                    "away_wins": agg["away_wins"],
                    "draws": agg["draws"],
                }
            )
        except Exception as e:
            logger.error(f"get_dashboard_stats failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_season_form(self):
        """Recent-form summary (streak, last-5 record, points-per-game)
        for the Dashboard, using core.form_analysis.FormAnalyzer.

        FormAnalyzer.compute_form_streak() is designed around real
        team-name matching (match["home_team"]/["away_team"] strings), but
        the coach's own team is not reliably the same free-text name
        match to match (sometimes entered as the literal club name,
        sometimes left as the "Home"/"Away" placeholder). This app's
        events already use a stable, reliable convention instead -- team
        "home" always means "the side being analysed" regardless of
        real-world venue (see _aggregate_season_stats's home_wins/
        away_wins, which uses the same convention) -- so synthetic
        matches are built with home_team="home"/away_team="away" and the
        already-computed goal counts, then FormAnalyzer is asked for
        that side's form. This reuses its streak/points/ppg calculation
        without inheriting its fragile name-matching requirement.

        get_all_matches() returns newest-first (ORDER BY created_at DESC);
        FormAnalyzer expects oldest-first (it reads matches[-1] as most
        recent), so the list is reversed before being passed in.
        """
        try:
            self._check_rate_limit()
            from kawkab.core.form_analysis import FormAnalyzer

            agg = await self._aggregate_season_stats()
            matches_oldest_first = list(reversed(agg["matches"]))
            synthetic = [
                {
                    "home_team": "home",
                    "away_team": "away",
                    "home_goals": m["home_goals"],
                    "away_goals": m["away_goals"],
                }
                for m in matches_oldest_first
            ]
            if not synthetic:
                return json.dumps(
                    {
                        "streak_type": "none",
                        "streak_length": 0,
                        "last_5_results": "",
                        "points_last_5": 0,
                        "ppg_last_5": 0.0,
                        "total_points": 0,
                    }
                )
            analyzer = FormAnalyzer()
            form = analyzer.compute_form_streak(synthetic, team="home")
            form.pop("goal_difference_trend", None)  # internal, not shown
            return json.dumps(form)
        except Exception as e:
            logger.error(f"get_season_form failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Wave C — Training Plan (get drills + save session)
    # ================================================================

    async def get_all_drills(self):
        try:
            self._check_rate_limit()
            await self.knowledge_service.initialize()
            drills = self.knowledge_service.drills
            drill_list = []
            for d in drills or []:
                drill_list.append(
                    {
                        "id": d.get("id", ""),
                        "name": d.get("name", "Unknown Drill"),
                        "category": d.get("category", "general"),
                        "difficulty": d.get("difficulty", "medium"),
                        "duration_min": d.get("duration_min", 15),
                        "description": d.get("description", ""),
                        "goals": d.get("goals", []),
                        "equipment": d.get("equipment", []),
                    }
                )
            return json.dumps({"drills": drill_list, "total": len(drill_list)})
        except Exception as e:
            logger.error(f"get_all_drills failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Sprint 1 — Injury Risk Dashboard
    # ================================================================
    # (the duplicate get_injury_risk that used to live here fabricated
    # ACWR from a synthetic sawtooth — deleted; the canonical GPS-backed
    # version is defined above, next to get_squad_injury_report)

    # ================================================================
    # Training Plan Auto-Generate (real reasoning engine — façade removed)
    # ================================================================

    async def generate_training_plan(self, match_id):
        # Facade fix: this handler used to fabricate five hardcoded
        # diagnoses (R001-R005) from raw event-type counts with constant
        # confidences, reference drill IDs (D001-D009) that never existed
        # in the knowledge base, and return a plan that was never
        # persisted. It now runs the real ReasoningService over the
        # stored events, flags any drill the KB cannot resolve, and
        # persists the plan so it survives a restart. Constant-firing
        # fabricated diagnoses are structurally impossible here: whatever
        # fires, fires on real stored events.
        try:
            self._check_rate_limit()
            match_id = SecurityValidator.validate_match_id(match_id)
            events = await self.storage_service.get_match_events(match_id, limit=5000)
            match_data = await self.storage_service.get_match(match_id) or {}

            from kawkab.services.knowledge_service import KnowledgeService
            from kawkab.services.reasoning_service import ReasoningService
            from kawkab.services.training_plan_service import TrainingPlanGenerator

            normalized_events = normalize_storage_events(events)
            analysis = build_match_analysis(match_id, match_data, normalized_events)

            kb = KnowledgeService()
            await kb.initialize()
            reasoner = ReasoningService(kb)
            report = await reasoner.diagnose_match(analysis, events=normalized_events)

            gen = TrainingPlanGenerator(kb)
            game_model = await self.storage_service.get_active_game_model_for_match(match_id)
            plan = await gen.generate_plan(report, game_model=game_model)

            plan_dict = gen.export_to_dict(plan)
            plan_dict["game_model_source"] = bool(game_model)

            # Honesty gate: a plan referencing drills the knowledge base
            # cannot resolve is flagged loudly — never silently rendered
            # as phantom content (the old D001-D009 failure mode).
            if plan_dict.get("unresolved_drills"):
                plan_dict["drills_warning"] = (
                    f"{len(plan_dict['unresolved_drills'])} referenced drill(s) are not in "
                    f"the knowledge base and will render as unresolved: "
                    f"{', '.join(plan_dict['unresolved_drills'])}"
                )

            # Persist (training_plans + plan_versions): the plan survives a
            # restart and can be assigned/executed/re-measured — the loop.
            priorities = [d.rule_name for d in report.diagnoses[:2]] or ["General development"]
            plan_db_id = await self.storage_service.save_training_plan(
                match_id=match_id,
                title=f"Plan: {' + '.join(priorities)}",
                payload=plan_dict,
                duration_weeks=plan.duration_weeks,
                priority_diagnoses=[
                    {
                        "rule_id": d.rule_id,
                        "rule_name": d.rule_name,
                        "confidence": d.confidence,
                    }
                    for d in report.diagnoses[:5]
                ],
                source="reasoning_engine",
            )

            persisted_id = plan_db_id if isinstance(plan_db_id, int) else None
            return json.dumps(
                {
                    "success": True,
                    "plan_id": persisted_id,
                    "plan": plan_dict,
                    "diagnosis_count": len(report.diagnoses),
                    "unresolved_drills": plan_dict.get("unresolved_drills", []),
                    "persisted": persisted_id is not None and persisted_id > 0,
                }
            )

        except Exception as e:
            logger.error(f"generate_training_plan failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def generate_training_plan_multi_match(self, match_ids_json: str):
        """Multi-match diagnosis -> morphocycle plan over a pooled event set.

        The KB rules declare min_matches thresholds; pooling several
        matches lets diagnoses reach 'confirmed' (see ReasoningService
        .diagnose_matches). Every event comes from real stored matches —
        the pooled report is persisted with full provenance
        (match_ids, confirmation tiers, hypothesis evaluations).
        """
        try:
            self._check_rate_limit()
            raw_ids = (
                json.loads(match_ids_json)
                if isinstance(match_ids_json, str)
                else list(match_ids_json or [])
            )
            if not isinstance(raw_ids, list) or not raw_ids:
                return json.dumps({"error": "match_ids must be a non-empty JSON array"})
            match_ids = [SecurityValidator.validate_match_id(m) for m in raw_ids]
            if len(set(match_ids)) < 2:
                return json.dumps(
                    {
                        "error": "multi-match mode requires at least 2 distinct match ids "
                        "(use generate_training_plan for a single match)"
                    }
                )

            from kawkab.services.knowledge_service import KnowledgeService
            from kawkab.services.reasoning_service import ReasoningService
            from kawkab.services.training_plan_service import TrainingPlanGenerator

            analyses = []
            pooled_events: list[dict] = []
            for mid in match_ids:
                events = await self.storage_service.get_match_events(mid, limit=5000)
                match_data = await self.storage_service.get_match(mid) or {}
                normalized = normalize_storage_events(events)
                analyses.append(build_match_analysis(mid, match_data, normalized))
                pooled_events.extend(normalized)

            kb = KnowledgeService()
            await kb.initialize()
            reasoner = ReasoningService(kb)
            report = await reasoner.diagnose_matches(analyses, match_ids=match_ids)

            gen = TrainingPlanGenerator(kb)
            game_model = await self.storage_service.get_active_game_model_for_match(match_ids[0])
            plan = await gen.generate_plan(report, game_model=game_model)
            plan_dict = gen.export_to_dict(plan)
            plan_dict["game_model_source"] = bool(game_model)

            if plan_dict.get("unresolved_drills"):
                plan_dict["drills_warning"] = (
                    f"{len(plan_dict['unresolved_drills'])} referenced drill(s) are not in "
                    f"the knowledge base and will render as unresolved: "
                    f"{', '.join(plan_dict['unresolved_drills'])}"
                )

            priorities = [d.rule_name for d in report.diagnoses[:2]] or ["General development"]
            plan_db_id = await self.storage_service.save_training_plan(
                match_id=match_ids[0],
                title=f"Multi-match plan ({len(match_ids)} matches): {' + '.join(priorities)}",
                payload=plan_dict,
                duration_weeks=plan.duration_weeks,
                priority_diagnoses=[
                    {
                        "rule_id": d.rule_id,
                        "rule_name": d.rule_name,
                        "confidence": d.confidence,
                        "confirmation": d.confirmation,
                        "match_count": d.match_count,
                    }
                    for d in report.diagnoses[:5]
                ],
                source="reasoning_engine_multi_match",
            )

            confirmed = sum(1 for d in report.diagnoses if d.confirmation == "confirmed")
            persisted_id = plan_db_id if isinstance(plan_db_id, int) else None
            return json.dumps(
                {
                    "success": True,
                    "plan_id": persisted_id,
                    "plan": plan_dict,
                    "diagnosis_count": len(report.diagnoses),
                    "confirmed_count": confirmed,
                    "match_ids": report.match_ids,
                    "unresolved_drills": plan_dict.get("unresolved_drills", []),
                    "persisted": persisted_id is not None and persisted_id > 0,
                }
            )
        except Exception as e:
            logger.error(f"generate_training_plan_multi_match failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_event_timestamp(self, match_id, event_id):
        try:
            self._check_rate_limit()
            events = await self.storage_service.get_match_events(match_id)
            for ev in events:
                eid = ev.get("id") or ev.get("event_id") or 0
                if eid == event_id:
                    return json.dumps(
                        {"timestamp": ev.get("timestamp", 0.0), "time": ev.get("timestamp", 0.0)}
                    )
            return json.dumps({"timestamp": 0.0, "time": 0.0})
        except Exception as e:
            logger.error(f"get_event_timestamp failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Roboflow Sports
    # ================================================================

    async def check_roboflow_sports_status(self):
        if self.roboflow_sports_service is None:
            return json.dumps({"available": False})
        return json.dumps(
            {
                "available": self.roboflow_sports_service.available,
                "has_team_classifier": self.roboflow_sports_service.has_team_classifier,
                "has_view_transformer": self.roboflow_sports_service.has_view_transformer,
            }
        )

    # ================================================================
    # Sprint 6 — Auto-Updater
    # ================================================================

    async def updater_check(self):
        try:
            svc = self._services.get("auto_updater_service")
            if svc is None:
                from kawkab.services.auto_updater_service import AutoUpdaterService

                svc = AutoUpdaterService()
                self._services["auto_updater_service"] = svc
            return svc.check_for_update()
        except Exception as e:
            logger.error(f"updater_check failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def updater_download(self, url, expected_digest=""):
        try:
            svc = self._services.get("auto_updater_service")
            if svc is None:
                return json.dumps({"error": "No updater service"})
            return svc.download_update(url, expected_digest)
        except Exception as e:
            logger.error(f"updater_download failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def updater_apply(self, path):
        try:
            svc = self._services.get("auto_updater_service")
            if svc is None:
                return json.dumps({"error": "No updater service"})
            return svc.apply_update(path)
        except Exception as e:
            logger.error(f"updater_apply failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def updater_version(self):
        try:
            svc = self._services.get("auto_updater_service")
            if svc is None:
                return json.dumps({"version": "0.13.0"})
            return svc.get_current_version()
        except Exception as e:
            logger.error(f"updater_version failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Sprint 7 — Sample Data & Utility
    # ================================================================

    async def load_sample_data(self):
        try:
            from kawkab.services.sample_data_generator import generate_sample_match

            data = generate_sample_match()
            match_data = data["match"]
            return json.dumps(
                {
                    "ok": True,
                    "match": f"{match_data['home_team']} {match_data['home_score']}-{match_data['away_score']} {match_data['away_team']}",
                    "events_count": data["total_events"],
                }
            )
        except Exception as e:
            logger.error(f"load_sample_data failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Sprint 2 — Wearable Import + Physiological Merge + Correlation
    # ================================================================

    async def import_wearable(self, file_path):
        try:
            # Validate before handing an arbitrary user path to a parser
            # (GPX/FIT/TCX are XML/zip formats the parsers open and read).
            # Wearable-specific allowlist: the generic data one would reject
            # every legitimate .gpx/.fit/.tcx import.
            SecurityValidator.validate_wearable_path(file_path)
            from kawkab.services.wearable_import_service import WearableImportService

            svc = WearableImportService()
            return svc.import_auto(file_path)
        except Exception as e:
            logger.error(f"import_wearable failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def merge_player_physiology(
        self, player_id, trajectory_json, wearable_json, body_mass_kg=75.0
    ):
        try:
            from kawkab.services.physiological_merge_service import PhysiologicalMergeService
            from kawkab.services.wearable_import_service import WearableDataPoint

            traj = (
                json.loads(trajectory_json) if isinstance(trajectory_json, str) else trajectory_json
            )
            w_data = json.loads(wearable_json) if isinstance(wearable_json, str) else wearable_json
            trajectory = [(t["t"], t.get("x", t.get("v", 0)), t.get("y", 0)) for t in traj]
            wearables = []
            for w in w_data:
                dp = WearableDataPoint(
                    timestamp_s=w.get("t", 0),
                    heart_rate_bpm=w.get("hr"),
                    speed_ms=w.get("spd", w.get("speed")),
                    distance_m=w.get("dist"),
                    acceleration_ms2=w.get("acc"),
                )
                wearables.append(dp)
            svc = PhysiologicalMergeService()
            return svc.merge(player_id, trajectory, wearables, body_mass_kg)
        except Exception as e:
            logger.error(f"merge_player_physiology failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def analyze_physio_tactical(
        self, events_json, speed_timeline_json, hr_timeline_json=None, window_s=5.0
    ):
        try:
            from kawkab.services.physio_tactical_correlation import PhysioTacticalCorrelationService

            events = json.loads(events_json) if isinstance(events_json, str) else events_json
            speed_tl = (
                json.loads(speed_timeline_json)
                if isinstance(speed_timeline_json, str)
                else speed_timeline_json
            )
            hr_tl = None
            if hr_timeline_json:
                hr_tl = (
                    json.loads(hr_timeline_json)
                    if isinstance(hr_timeline_json, str)
                    else hr_timeline_json
                )
            svc = PhysioTacticalCorrelationService()
            return svc.analyze(events, speed_tl, hr_tl, window_s)
        except Exception as e:
            logger.error(f"analyze_physio_tactical failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ── Sprint 2 — Advanced Visualizations ───────────────────────
