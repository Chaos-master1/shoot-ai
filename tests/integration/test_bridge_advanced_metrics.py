"""Integration test for Bridge advanced metrics wiring (v0.6.2).

Tests that Bridge.analyze_match correctly calls and stores results from:
- AdvancedEventDetectionService
- PhysicalLoadService
- PressureMetricsService
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from kawkab.core.security import SecurityValidator
from kawkab.services.advanced_event_detection_service import AdvancedEventDetectionService
from kawkab.services.analysis_service import AnalysisService
from kawkab.services.cv_service import Detection, FrameDetections, MatchTrackData
from kawkab.services.homography_service import HomographyMatrix
from kawkab.services.llm_service import LLMConfig
from kawkab.services.physical_load_service import PhysicalLoadService
from kawkab.services.pressure_metrics_service import PressureMetricsService
from kawkab.ui.bridge import Bridge


@pytest.fixture(autouse=True)
def _allow_temp_video_paths(monkeypatch):
    # SecurityValidator.validate_video_path only allows paths under
    # Documents/KawkabAI -- correct for the real app, but every test in
    # this file builds its fake video with tempfile.TemporaryDirectory(),
    # which is never under there. This file is about advanced-metrics/
    # frame_skip wiring, not path security, so bypass just the directory
    # check (still requires a real, existing, correctly-suffixed file).
    monkeypatch.setattr(
        SecurityValidator, "validate_video_path", staticmethod(lambda p: Path(p).resolve())
    )


def make_detection(track_id, class_name, x, y, w=20.0, h=40.0, confidence=0.9):
    return Detection(
        bbox=(x, y, x + w, y + h),
        confidence=confidence,
        class_id=0 if class_name == "person" else 32,
        class_name=class_name,
        track_id=track_id,
    )


def make_frame(frame_number, timestamp, detections, width=1280, height=720):
    return FrameDetections(
        frame_number=frame_number,
        timestamp=timestamp,
        detections=detections,
        image_width=width,
        image_height=height,
    )


def create_test_tracking_data() -> MatchTrackData:
    """Create synthetic tracking data for a short test match."""
    frames = []
    fps = 10.0
    duration = 3.0
    total_frames = int(duration * fps)

    for i in range(total_frames):
        ts = i / fps
        detections = []
        ball_x = 100 + (i % 50) * 5
        ball_y = 300 + 20 * (i % 10) / 10
        detections.append(
            make_detection(99, "sports ball", ball_x, ball_y, w=5, h=5, confidence=0.8)
        )
        for p in range(1, 23):
            px = 100 + (p * 40) + (i % 20) * 2
            py = 150 + (p % 3) * 150 + (i % 10) * 3
            detections.append(make_detection(p, "person", px, py, confidence=0.85))
        frames.append(make_frame(i, ts, detections))

    return MatchTrackData(
        match_id=1,
        fps=fps,
        total_frames=total_frames,
        duration_seconds=duration,
        frames=frames,
        track_registry={i: {"track_id": i} for i in range(1, 23)},
        player_teams={i: "home" if i <= 11 else "away" for i in range(1, 23)},
        tracking_metrics={
            "validated_player_tracks": 22,
            "raw_tracks_detected": 25,
            "fragmentation_rate": 1.2,
            "tracking_quality": "excellent",
            "frame_skip": 3,
            "team_detection": {"enabled": True, "assigned": 22, "n_clusters": 2},
        },
        match_type="full_match",
    )


def create_homography() -> HomographyMatrix:
    return HomographyMatrix(
        matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        pitch_length_m=105.0,
        pitch_width_m=68.0,
        confidence=0.95,
        error_px=2.5,
    )


class FakeCVService:
    def __init__(self, track_data):
        self._track_data = track_data

    async def process_video(
        self,
        video_path,
        progress_callback=None,
        frame_skip=3,
        enable_team_detection=True,
        checkpoint_interval=0,
        resume_checkpoint=None,
        storage_service=None,
        match_id=0,
    ):
        # Signature mirrors the real CVService.process_video (which the
        # analysis handler calls with match_id= and storage_service=).
        # The old 4-param stub raised TypeError on every analyze_match
        # call, silently degrading these tests to error-payload asserts.
        if progress_callback:
            await progress_callback(0.5, "Halfway")
            await progress_callback(1.0, "Done")
        return self._track_data


class FakeEnhancementService:
    async def preprocess_video(self, input_path, output_path):
        Path(output_path).write_bytes(b"fake")


class FakeLLMService:
    config = LLMConfig(provider="none")

    async def generate_coach_report(self, **kwargs):
        return "Test report"


class FakeKnowledgeService:
    stats = {"rules": 40, "drills": 24}

    async def initialize(self):
        pass


class FakeAudioService:
    pass


class FakeHomographyService:
    def load_calibration(self, match_id):
        return create_homography()

    def save_calibration(self, match_id, matrix):
        pass


class FakeStorageService:
    """Fake storage that doesn't need a real database."""

    def __init__(self):
        self._matches = {}
        self._players = {}
        self._events = []
        self._advanced_metrics = []
        self._next_id = 1

    async def initialize(self):
        pass

    async def save_match(self, name, video_path, home_team=None, away_team=None):
        mid = self._next_id
        self._next_id += 1
        self._matches[mid] = {"id": mid, "name": name, "video_path": video_path}
        return mid

    async def update_match_analysis(self, match_id, duration, fps, total_frames):
        pass

    async def save_player(self, match_id, player_data):
        self._players.setdefault(match_id, []).append(player_data)

    async def save_event(self, match_id, event):
        self._events.append({"match_id": match_id, **event})

    async def save_advanced_metrics(
        self,
        match_id,
        metric_name,
        metric_value,
        metric_category="",
        player_id=None,
        pitch_zone="",
        timestamp=None,
        metadata=None,
    ):
        self._advanced_metrics.append(
            {
                "match_id": match_id,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "metric_category": metric_category,
            }
        )

    async def get_match(self, match_id):
        return self._matches.get(match_id)

    async def get_all_matches(self):
        return list(self._matches.values())

    async def get_match_events(self, match_id):
        return [e for e in self._events if e["match_id"] == match_id]

    async def save_report(self, match_id, language, report_text, llm_provider):
        pass


@pytest.mark.asyncio
async def test_bridge_advanced_metrics_wiring():
    """Test that Bridge calls advanced metrics services and stores results."""
    track_data = create_test_tracking_data()

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / "test_video.mp4"
        video_path.write_bytes(b"fake mp4 data")

        # Create fake storage (no real DB needed)
        storage = FakeStorageService()
        await storage.initialize()

        # Save a match
        match_id = await storage.save_match(name="Test Match", video_path=str(video_path))

        # Create bridge with fake services and real advanced metrics
        bridge = Bridge(
            cv_service=FakeCVService(track_data),
            enhancement_service=FakeEnhancementService(),
            analysis_service=AnalysisService(),
            llm_service=FakeLLMService(),
            knowledge_service=FakeKnowledgeService(),
            storage_service=storage,
            audio_service=FakeAudioService(),
            homography_service=FakeHomographyService(),
            advanced_event_detection_service=AdvancedEventDetectionService(),
            physical_load_service=PhysicalLoadService(),
            pressure_metrics_service=PressureMetricsService(),
        )

        # Run analysis
        result_json = await bridge.analyze_match(match_id, str(video_path))
        import json

        result = json.loads(result_json)

        # Verify basic result
        assert "error" not in result, f"Error: {result.get('error')}"
        assert result["match_id"] == match_id
        assert result["player_count"] == 22
        assert result["event_count"] > 0

        # Verify advanced metrics in result
        assert "advanced_event_count" in result
        assert "advanced_metrics" in result
        assert "physical_load" in result["advanced_metrics"]
        assert "pressure" in result["advanced_metrics"]

        # Verify data was stored in fake storage
        assert len(storage._advanced_metrics) > 0, "No advanced metrics stored"

        metric_names = {m["metric_name"] for m in storage._advanced_metrics}
        assert "sprint_count" in metric_names or "ppda" in metric_names


@pytest.mark.asyncio
async def test_bridge_get_gpu_info():
    """Test that Bridge.get_gpu_info returns GPU info and recommendations."""

    bridge = Bridge(
        cv_service=FakeCVService(create_test_tracking_data()),
        enhancement_service=FakeEnhancementService(),
        analysis_service=AnalysisService(),
        llm_service=FakeLLMService(),
        knowledge_service=FakeKnowledgeService(),
        storage_service=FakeStorageService(),
        audio_service=FakeAudioService(),
        homography_service=FakeHomographyService(),
        advanced_event_detection_service=None,
        physical_load_service=None,
        pressure_metrics_service=None,
    )

    import json

    result_json = bridge.get_gpu_info()
    result = json.loads(result_json)

    assert "error" not in result
    assert "gpu_name" in result
    assert "tier" in result
    assert result["tier"] in {"high", "mid", "low", "unknown"}
    assert "recommendations" in result
    assert "current_settings" in result
    assert result["current_settings"]["model_size"] == "l"
    assert result["current_settings"]["frame_skip"] == bridge.frame_skip


@pytest.mark.asyncio
async def test_bridge_frame_skip_parameter():
    """Test that Bridge uses frame_skip parameter correctly."""
    track_data = create_test_tracking_data()
    bridge = Bridge(
        cv_service=FakeCVService(track_data),
        enhancement_service=FakeEnhancementService(),
        analysis_service=AnalysisService(),
        llm_service=FakeLLMService(),
        knowledge_service=FakeKnowledgeService(),
        storage_service=FakeStorageService(),
        audio_service=FakeAudioService(),
        homography_service=FakeHomographyService(),
        advanced_event_detection_service=None,
        physical_load_service=None,
        pressure_metrics_service=None,
        frame_skip=5,
    )
    assert bridge.frame_skip == 5

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / "test_video.mp4"
        video_path.write_bytes(b"fake mp4 data")
        storage = FakeStorageService()
        await storage.initialize()
        match_id = await storage.save_match(name="Test", video_path=str(video_path))
        bridge.storage_service = storage
        result_json = await bridge.analyze_match(match_id, str(video_path))
        import json

        result = json.loads(result_json)
        assert "error" not in result


@pytest.mark.asyncio
async def test_bridge_works_without_advanced_metrics_services():
    """Test that Bridge works when advanced metrics services are None."""
    track_data = create_test_tracking_data()

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / "test_video.mp4"
        video_path.write_bytes(b"fake mp4 data")

        storage = FakeStorageService()
        await storage.initialize()

        match_id = await storage.save_match(name="Test Match", video_path=str(video_path))

        bridge = Bridge(
            cv_service=FakeCVService(track_data),
            enhancement_service=FakeEnhancementService(),
            analysis_service=AnalysisService(),
            llm_service=FakeLLMService(),
            knowledge_service=FakeKnowledgeService(),
            storage_service=storage,
            audio_service=FakeAudioService(),
            homography_service=FakeHomographyService(),
            advanced_event_detection_service=None,
            physical_load_service=None,
            pressure_metrics_service=None,
        )

        result_json = await bridge.analyze_match(match_id, str(video_path))
        import json

        result = json.loads(result_json)

        assert "error" not in result
        assert result["advanced_event_count"] == 0
        assert result["advanced_metrics"]["physical_load"] == {}
        assert result["advanced_metrics"]["pressure"] == {}
        assert len(storage._advanced_metrics) == 0
