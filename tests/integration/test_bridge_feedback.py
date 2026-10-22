"""Tests for Bridge feedback methods (v0.8.0)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from kawkab.services.analysis_service import AnalysisService
from kawkab.services.cv_service import FrameDetections, MatchTrackData
from kawkab.services.feedback_service import CoachFeedback, FeedbackService
from kawkab.services.storage_service import StorageService
from kawkab.ui.bridge import Bridge


class FakeCVService:
    def __init__(self):
        self.model_size = "l"

    async def process_video(
        self, video_path, progress_callback=None, frame_skip=3, enable_team_detection=True
    ):
        return MatchTrackData(
            match_id=1,
            fps=10.0,
            total_frames=10,
            duration_seconds=1.0,
            frames=[FrameDetections(0, 0.0, [], 1280, 720)],
            track_registry={},
            player_teams={},
            tracking_metrics={},
            match_type="full_match",
        )


class FakeEnhancementService:
    async def preprocess_video(self, input_path, output_path):
        Path(output_path).write_bytes(b"fake")


class FakeLLMService:
    config = type("obj", (object,), {"provider": "none"})()

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
        return None

    def save_calibration(self, match_id, matrix):
        pass


@pytest.mark.asyncio
async def test_bridge_submit_feedback():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        storage = StorageService()
        storage._db_path = db_path
        await storage.initialize()

        feedback = FeedbackService(storage_service=storage)
        bridge = Bridge(
            cv_service=FakeCVService(),
            enhancement_service=FakeEnhancementService(),
            analysis_service=AnalysisService(),
            llm_service=FakeLLMService(),
            knowledge_service=FakeKnowledgeService(),
            storage_service=storage,
            audio_service=FakeAudioService(),
            homography_service=FakeHomographyService(),
            feedback_service=feedback,
        )

        import json

        result_json = await bridge.submit_feedback(
            json.dumps(
                {
                    "coach_id": "coach_001",
                    "match_id": 1,
                    "overall_rating": 5,
                    "tracking_rating": 4,
                    "comments": "Excellent tool!",
                }
            )
        )
        result = json.loads(result_json)
        assert "error" not in result
        assert "feedback_id" in result
        assert result["status"] == "saved"

        await storage.close()


@pytest.mark.asyncio
async def test_bridge_submit_issue():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        storage = StorageService()
        storage._db_path = db_path
        await storage.initialize()

        feedback = FeedbackService(storage_service=storage)
        bridge = Bridge(
            cv_service=FakeCVService(),
            enhancement_service=FakeEnhancementService(),
            analysis_service=AnalysisService(),
            llm_service=FakeLLMService(),
            knowledge_service=FakeKnowledgeService(),
            storage_service=storage,
            audio_service=FakeAudioService(),
            homography_service=FakeHomographyService(),
            feedback_service=feedback,
        )

        import json

        result_json = await bridge.submit_issue(
            json.dumps(
                {
                    "category": "tracking",
                    "severity": "high",
                    "description": "Lost player 7",
                    "match_id": 1,
                }
            )
        )
        result = json.loads(result_json)
        assert "error" not in result
        assert "issue_id" in result
        assert result["status"] == "saved"

        await storage.close()


@pytest.mark.asyncio
async def test_bridge_get_feedback_stats():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        storage = StorageService()
        storage._db_path = db_path
        await storage.initialize()

        feedback = FeedbackService(storage_service=storage)
        for i in range(3):
            await feedback.submit_feedback(
                CoachFeedback(coach_id=f"c{i}", match_id=i + 1, overall_rating=4 if i < 2 else 2)
            )

        bridge = Bridge(
            cv_service=FakeCVService(),
            enhancement_service=FakeEnhancementService(),
            analysis_service=AnalysisService(),
            llm_service=FakeLLMService(),
            knowledge_service=FakeKnowledgeService(),
            storage_service=storage,
            audio_service=FakeAudioService(),
            homography_service=FakeHomographyService(),
            feedback_service=feedback,
        )

        import json

        result_json = await bridge.get_feedback_stats()
        result = json.loads(result_json)
        assert "error" not in result
        assert result["total_feedback"] == 3
        assert result["average_rating"] == pytest.approx(3.33, 0.01)

        await storage.close()


@pytest.mark.asyncio
async def test_bridge_feedback_without_service():
    bridge = Bridge(
        cv_service=FakeCVService(),
        enhancement_service=FakeEnhancementService(),
        analysis_service=AnalysisService(),
        llm_service=FakeLLMService(),
        knowledge_service=FakeKnowledgeService(),
        storage_service=None,
        audio_service=FakeAudioService(),
        homography_service=FakeHomographyService(),
        feedback_service=None,
    )

    import json

    result_json = await bridge.submit_feedback(json.dumps({"overall_rating": 5}))
    result = json.loads(result_json)
    assert "error" in result
    assert "not available" in result["error"]
