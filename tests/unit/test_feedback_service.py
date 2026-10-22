"""Tests for FeedbackService (v0.8.0)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.services.feedback_service import (
    CoachFeedback,
    FeedbackService,
    IssueReport,
    UsageSession,
)
from kawkab.services.storage_service import StorageService


class TestFeedbackService:
    """Test feedback collection and storage."""

    @pytest.mark.asyncio
    async def test_submit_feedback_with_storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = StorageService()
            storage._db_path = db_path
            await storage.initialize()

            svc = FeedbackService(storage_service=storage)
            feedback = CoachFeedback(
                coach_id="coach_001",
                match_id=1,
                overall_rating=4,
                tracking_rating=5,
                events_rating=3,
                comments="Great tool!",
            )
            fid = await svc.submit_feedback(feedback)
            assert fid > 0

            all_feedback = await svc.get_all_feedback()
            assert len(all_feedback) == 1
            assert all_feedback[0]["overall_rating"] == 4
            assert all_feedback[0]["coach_id"] == "coach_001"

            await storage.close()

    @pytest.mark.asyncio
    async def test_submit_issue_with_storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = StorageService()
            storage._db_path = db_path
            await storage.initialize()

            svc = FeedbackService(storage_service=storage)
            issue = IssueReport(
                category="tracking",
                severity="high",
                description="Lost track of player 7",
                match_id=1,
            )
            iid = await svc.submit_issue(issue)
            assert iid > 0

            all_issues = await svc.get_all_issues()
            assert len(all_issues) == 1
            assert all_issues[0]["category"] == "tracking"
            assert all_issues[0]["severity"] == "high"

            await storage.close()

    @pytest.mark.asyncio
    async def test_record_session_with_storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = StorageService()
            storage._db_path = db_path
            await storage.initialize()

            svc = FeedbackService(storage_service=storage)
            session = UsageSession(
                session_id="sess_001",
                features_used=["analyze", "export_csv"],
                duration_seconds=120.0,
                match_count=1,
                gpu_tier="high",
                model_size="l",
            )
            sid = await svc.record_session(session)
            assert sid > 0

            await storage.close()

    @pytest.mark.asyncio
    async def test_summary_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = StorageService()
            storage._db_path = db_path
            await storage.initialize()

            svc = FeedbackService(storage_service=storage)
            for i in range(3):
                await svc.submit_feedback(
                    CoachFeedback(
                        coach_id=f"coach_{i}",
                        match_id=i + 1,
                        overall_rating=4 if i < 2 else 2,
                    )
                )
            await svc.submit_issue(
                IssueReport(category="ui", severity="medium", description="Button too small")
            )

            stats = await svc.get_summary_stats()
            assert stats["total_feedback"] == 3
            assert stats["average_rating"] == pytest.approx(3.33, 0.01)
            assert stats["rating_distribution"][4] == 2
            assert stats["rating_distribution"][2] == 1
            assert stats["total_issues"] == 1
            assert stats["issue_by_severity"]["medium"] == 1

            await storage.close()

    def test_pending_counts_without_storage(self):
        svc = FeedbackService(storage_service=None)
        feedback = CoachFeedback(coach_id="c1", match_id=1, overall_rating=5)
        import asyncio

        asyncio.run(svc.submit_feedback(feedback))

        assert svc.get_pending_counts()["feedback"] == 1
        assert svc.get_pending_counts()["issues"] == 0

    def test_dataclass_serialization(self):
        feedback = CoachFeedback(
            coach_id="c1",
            match_id=1,
            overall_rating=5,
            tracking_rating=4,
            comments="Test",
        )
        d = feedback.to_dict()
        assert d["coach_id"] == "c1"
        assert d["overall_rating"] == 5
        assert d["tracking_rating"] == 4
        assert "created_at" in d
