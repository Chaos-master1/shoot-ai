"""Tests for ClipExtractionService video clip extraction."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()


@pytest.fixture(scope="module")
def clip_mod():
    return load_service_module("kawkab.services.clip_service", "clip_service.py")


class TestClipExtractionService:
    def test_init_creates_cache_dir(self, clip_mod, tmp_path):
        cache_dir = tmp_path / "clips"
        svc = clip_mod.ClipExtractionService(cache_dir=cache_dir)
        assert cache_dir.exists()
        assert svc.pre_pad == 2.0
        assert svc.post_pad == 2.0

    def test_init_custom_padding(self, clip_mod, tmp_path):
        svc = clip_mod.ClipExtractionService(
            cache_dir=tmp_path, pre_pad_seconds=5.0, post_pad_seconds=3.0
        )
        assert svc.pre_pad == 5.0
        assert svc.post_pad == 3.0

    @pytest.mark.asyncio
    async def test_extract_clip_missing_video(self, clip_mod, tmp_path):
        svc = clip_mod.ClipExtractionService(cache_dir=tmp_path)
        result = await svc.extract_clip(
            video_path=tmp_path / "nonexistent.mp4",
            start_time=10.0,
            end_time=20.0,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_clip_invalid_duration(self, clip_mod, tmp_path):
        (tmp_path / "input.mp4").write_text("fake")
        svc = clip_mod.ClipExtractionService(cache_dir=tmp_path)
        result = await svc.extract_clip(
            video_path=tmp_path / "input.mp4",
            start_time=20.0,
            end_time=10.0,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_clip_success(self, clip_mod, tmp_path):
        (tmp_path / "input.mp4").write_text("fake")
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        svc = clip_mod.ClipExtractionService(cache_dir=tmp_path)
        with patch.object(asyncio, "create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            result = await svc.extract_clip(
                video_path=tmp_path / "input.mp4",
                start_time=10.0,
                end_time=20.0,
            )
        assert result is not None
        assert result.name == "input_clip_8s_22s.mp4"

    @pytest.mark.asyncio
    async def test_extract_clip_ffmpeg_not_found(self, clip_mod, tmp_path):
        (tmp_path / "input.mp4").write_text("fake")
        svc = clip_mod.ClipExtractionService(cache_dir=tmp_path)
        with patch.object(
            asyncio, "create_subprocess_exec", AsyncMock(side_effect=FileNotFoundError)
        ):
            result = await svc.extract_clip(
                video_path=tmp_path / "input.mp4",
                start_time=10.0,
                end_time=20.0,
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_clip_ffmpeg_fails(self, clip_mod, tmp_path):
        (tmp_path / "input.mp4").write_text("fake")
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"ffmpeg error"))

        svc = clip_mod.ClipExtractionService(cache_dir=tmp_path)
        with patch.object(asyncio, "create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            result = await svc.extract_clip(
                video_path=tmp_path / "input.mp4",
                start_time=10.0,
                end_time=20.0,
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_clip_general_exception(self, clip_mod, tmp_path):
        (tmp_path / "input.mp4").write_text("fake")
        svc = clip_mod.ClipExtractionService(cache_dir=tmp_path)
        with patch.object(
            asyncio, "create_subprocess_exec", AsyncMock(side_effect=RuntimeError("boom"))
        ):
            result = await svc.extract_clip(
                video_path=tmp_path / "input.mp4",
                start_time=10.0,
                end_time=20.0,
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_clip_custom_output_name(self, clip_mod, tmp_path):
        (tmp_path / "input.mp4").write_text("fake")
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        svc = clip_mod.ClipExtractionService(cache_dir=tmp_path)
        with patch.object(asyncio, "create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            result = await svc.extract_clip(
                video_path=tmp_path / "input.mp4",
                start_time=10.0,
                end_time=20.0,
                output_name="custom_clip.mp4",
            )
        assert result is not None
        assert result.name == "custom_clip.mp4"

    @pytest.mark.asyncio
    async def test_extract_clip_quality_preset(self, clip_mod, tmp_path):
        (tmp_path / "input.mp4").write_text("fake")
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        svc = clip_mod.ClipExtractionService(cache_dir=tmp_path)
        with patch.object(asyncio, "create_subprocess_exec") as mock_exec:
            mock_exec.return_value = mock_proc
            result = await svc.extract_clip(
                video_path=tmp_path / "input.mp4",
                start_time=10.0,
                end_time=20.0,
                quality="high",
            )
            assert result is not None
            call_args = mock_exec.call_args[0]
            assert "-crf" in call_args
            crf_idx = call_args.index("-crf")
            assert call_args[crf_idx + 1] == "18"

    @pytest.mark.asyncio
    async def test_extract_evidence_clips(self, clip_mod, tmp_path):
        (tmp_path / "input.mp4").write_text("fake")
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        svc = clip_mod.ClipExtractionService(cache_dir=tmp_path)
        timestamps = [
            {"start": 10.0, "end": 20.0, "description": "Goal"},
            {"start": 30.0, "end": 40.0, "description": "Shot"},
        ]
        with patch.object(asyncio, "create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            results = await svc.extract_evidence_clips(
                video_path=tmp_path / "input.mp4",
                timestamps=timestamps,
            )
        assert len(results) == 2
        assert results[0]["description"] == "Goal"
        assert results[1]["description"] == "Shot"
        assert "path" in results[0]
        assert "filename" in results[0]

    @pytest.mark.asyncio
    async def test_extract_evidence_clips_all_fail(self, clip_mod, tmp_path):
        (tmp_path / "input.mp4").write_text("fake")
        svc = clip_mod.ClipExtractionService(cache_dir=tmp_path)
        timestamps = [
            {"start": 10.0, "end": 20.0, "description": "Goal"},
        ]
        with patch.object(
            asyncio, "create_subprocess_exec", AsyncMock(side_effect=FileNotFoundError)
        ):
            results = await svc.extract_evidence_clips(
                video_path=tmp_path / "input.mp4",
                timestamps=timestamps,
            )
        assert results == []

    @pytest.mark.asyncio
    async def test_extract_event_clips(self, clip_mod, tmp_path):
        (tmp_path / "input.mp4").write_text("fake")
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        svc = clip_mod.ClipExtractionService(cache_dir=tmp_path)
        events = [
            {"timestamp": 60.0, "type": "pass", "team": "home"},
            {"timestamp": 120.0, "type": "shot", "team": "away"},
        ]
        with patch.object(asyncio, "create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            results = await svc.extract_event_clips(
                video_path=tmp_path / "input.mp4",
                events=events,
                context_seconds=5.0,
            )
        assert len(results) == 2
        assert results[0]["event_type"] == "pass"
        assert results[0]["team"] == "home"
        assert results[1]["event_type"] == "shot"
        assert results[1]["timestamp"] == 120.0

    @pytest.mark.asyncio
    async def test_extract_event_clips_empty(self, clip_mod, tmp_path):
        svc = clip_mod.ClipExtractionService(cache_dir=tmp_path)
        results = await svc.extract_event_clips(
            video_path=tmp_path / "input.mp4",
            events=[],
        )
        assert results == []
