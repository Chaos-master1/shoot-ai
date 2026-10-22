"""Tests for BenchmarkService - performance measurement and system detection."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest
from conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.services.benchmark_service import BenchmarkService
from kawkab.services.storage_service import StorageService


class TestBenchmarkService:
    """Test performance benchmarking utilities."""

    def test_init_detects_system_info(self):
        svc = BenchmarkService()
        assert svc._system_info["cpu_name"] != "unknown" or svc._system_info["ram_gb"] > 0

    def test_reset_clears_stages(self):
        svc = BenchmarkService()
        svc.start_stage("test")
        time.sleep(0.01)
        svc.end_stage("test")
        assert "test" in svc._stage_times
        svc.reset()
        assert svc._stage_times == {}
        assert svc._peak_memory_mb == 0.0

    def test_stage_timing(self):
        svc = BenchmarkService()
        svc.start_stage("sleep")
        time.sleep(0.05)
        svc.end_stage("sleep")
        assert svc._stage_times["sleep"] >= 0.05

    def test_build_result_computes_totals(self):
        svc = BenchmarkService()
        svc._stage_times = {
            "enhancement": 1.0,
            "detection": 2.0,
            "tracking": 1.5,
        }
        svc._peak_memory_mb = 512.0
        result = svc.build_result(
            match_id=42,
            video_path="/test.mp4",
            video_duration_seconds=60.0,
            total_frames=1800,
            model_size="l",
            frame_skip=3,
        )
        assert result.match_id == 42
        assert result.video_path == "/test.mp4"
        assert result.total_time_seconds == 4.5
        assert result.realtime_ratio == 4.5 / 60.0
        assert result.fps_effective == 1800 / 4.5
        assert result.stage_enhancement_seconds == 1.0
        assert result.stage_detection_seconds == 2.0
        assert result.stage_tracking_seconds == 1.5
        assert result.peak_memory_mb == 512.0

    def test_gpu_tier_classification(self):
        assert BenchmarkService.classify_gpu_tier("NVIDIA GeForce RTX 4090") == "high"
        assert BenchmarkService.classify_gpu_tier("NVIDIA GeForce RTX 4070") == "mid"
        assert BenchmarkService.classify_gpu_tier("NVIDIA GeForce RTX 3060") == "mid"
        assert BenchmarkService.classify_gpu_tier("NVIDIA GeForce RTX 3050") == "low"
        assert BenchmarkService.classify_gpu_tier("NVIDIA GeForce GTX 1650") == "low"
        assert BenchmarkService.classify_gpu_tier("Unknown GPU") == "unknown"

    def test_recommend_settings(self):
        high = BenchmarkService.recommend_settings("high")
        assert high["model_size"] == "l"
        assert high["frame_skip"] == 2
        assert high["gpu_enabled"] is True

        low = BenchmarkService.recommend_settings("low")
        assert low["model_size"] == "m"
        assert low["frame_skip"] == 4

        unknown = BenchmarkService.recommend_settings("unknown")
        assert unknown["model_size"] == "n"
        assert unknown["frame_skip"] == 5

    @pytest.mark.asyncio
    async def test_save_benchmark_to_database(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = StorageService()
            # Override default path
            storage._db_path = db_path
            await storage.initialize()

            cursor = storage._conn.cursor()
            cursor.execute(
                "INSERT INTO teams (id, name, short_name) VALUES (1, 'Test Team', 'TST')"
            )
            cursor.execute(
                "INSERT INTO matches (id, name, video_path) VALUES (1, 'Test Match', '/test.mp4')"
            )
            storage._conn.commit()

            svc = BenchmarkService()
            svc._stage_times = {"enhancement": 1.0, "detection": 2.0}
            svc._peak_memory_mb = 256.0

            result = svc.build_result(
                match_id=1,
                video_path="/test.mp4",
                video_duration_seconds=30.0,
                total_frames=900,
                model_size="l",
                frame_skip=3,
            )

            bench_id = await storage.save_benchmark(result)
            assert bench_id > 0

            # Verify retrieval
            recent = await storage.get_recent_benchmarks(limit=1)
            assert len(recent) == 1
            assert recent[0]["match_id"] == 1
            assert recent[0]["total_time_seconds"] == 3.0

            # Close connection so temp dir can be cleaned up on Windows
            await storage.close()
