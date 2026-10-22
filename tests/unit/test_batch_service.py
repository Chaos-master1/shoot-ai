"""Tests for BatchService - multi-match batch processing queue."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.services.batch_service import BatchService, BatchStatus
from kawkab.services.storage_service import StorageService


class FakeBridge:
    """Fake bridge for batch testing."""

    def __init__(self, fail_match_id=None):
        self.fail_match_id = fail_match_id
        self.calls = []

    async def analyze_match(self, match_id, video_path):
        self.calls.append((match_id, video_path))
        if match_id == self.fail_match_id:
            return '{"error": "Simulated failure"}'
        return '{"match_id": ' + str(match_id) + "}"


class TestBatchService:
    """Test batch processing utilities."""

    @pytest.mark.asyncio
    async def test_create_job(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = StorageService()
            storage._db_path = db_path
            await storage.initialize()

            svc = BatchService()
            job = await svc.create_job(storage, "Weekend Matches", [1, 2, 3])
            assert job.id is not None
            assert job.name == "Weekend Matches"
            assert job.status == BatchStatus.PENDING
            assert job.total_matches == 3
            assert job.match_ids == [1, 2, 3]

            await storage.close()

    @pytest.mark.asyncio
    async def test_run_job_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = StorageService()
            storage._db_path = db_path
            await storage.initialize()

            # Create matches in DB
            for i in range(1, 4):
                await storage.save_match(f"Match {i}", f"/test/match{i}.mp4")

            svc = BatchService()
            bridge = FakeBridge()
            job = await svc.create_job(storage, "Test Batch", [1, 2, 3])

            completed = []

            async def progress_cb(done, total, match_id, status):
                completed.append((done, total, match_id, status))

            result = await svc.run_job(storage, bridge, job.id, progress_cb)
            assert result.status == BatchStatus.COMPLETED
            assert result.completed_matches == 3
            assert result.failed_matches == 0
            assert len(bridge.calls) == 3
            assert len(completed) == 3

            await storage.close()

    @pytest.mark.asyncio
    async def test_run_job_with_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = StorageService()
            storage._db_path = db_path
            await storage.initialize()

            for i in range(1, 4):
                await storage.save_match(f"Match {i}", f"/test/match{i}.mp4")

            svc = BatchService()
            bridge = FakeBridge(fail_match_id=2)
            job = await svc.create_job(storage, "Test Batch", [1, 2, 3])

            result = await svc.run_job(storage, bridge, job.id)
            assert result.status == BatchStatus.FAILED
            assert result.completed_matches == 2
            assert result.failed_matches == 1

            await storage.close()

    @pytest.mark.asyncio
    async def test_cancel_job(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = StorageService()
            storage._db_path = db_path
            await storage.initialize()

            await storage.save_match("Match 1", "/test/match1.mp4")

            svc = BatchService()
            bridge = FakeBridge()
            job = await svc.create_job(storage, "Test Batch", [1])

            # Cancel immediately
            svc.cancel()
            result = await svc.run_job(storage, bridge, job.id)
            assert result.status == BatchStatus.CANCELLED

            await storage.close()

    @pytest.mark.asyncio
    async def test_list_jobs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = StorageService()
            storage._db_path = db_path
            await storage.initialize()

            svc = BatchService()
            await svc.create_job(storage, "Job A", [1])
            await svc.create_job(storage, "Job B", [2, 3])

            jobs = await svc.list_jobs(storage)
            assert len(jobs) == 2
            assert jobs[0].name == "Job B"  # Most recent first

            await storage.close()
