"""Tests for EnhancementService — video preprocessing and upscaling."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()

_mod = load_service_module("enhance_test", "enhancement_service.py")

EnhancementService = _mod.EnhancementService


@pytest.fixture
def fake_input(tmp_path) -> Path:
    p = tmp_path / "input.mp4"
    p.write_text("fake video")
    return p


@pytest.fixture
def fake_output(tmp_path) -> Path:
    return tmp_path / "output.mp4"


@pytest.fixture
def mock_ffmpeg():
    """Build a mock ffmpeg module with self-chaining stream."""
    ff = MagicMock()
    stream = MagicMock()
    stream.filter.return_value = stream
    ff.input.return_value = stream
    final = MagicMock()
    stream.output.return_value.overwrite_output.return_value = final
    with patch.dict("sys.modules", {"ffmpeg": ff}):
        yield ff, stream, final


# ===================================================================
# Init
# ===================================================================


class TestInit:
    def test_defaults(self):
        s = EnhancementService()
        assert s.enable_stabilization is True
        assert s.enable_denoising is True
        assert s.enable_sharpening is True
        assert s.enable_upscaling is False
        assert s.enable_interpolation is False
        assert s.gpu_enabled is True

    def test_custom_flags(self):
        s = EnhancementService(
            enable_stabilization=False,
            enable_denoising=False,
            enable_sharpening=False,
            enable_upscaling=True,
            enable_interpolation=True,
            gpu_enabled=False,
        )
        assert s.enable_stabilization is False
        assert s.enable_denoising is False
        assert s.enable_sharpening is False
        assert s.enable_upscaling is True
        assert s.enable_interpolation is True
        assert s.gpu_enabled is False


# ===================================================================
# PreprocessVideo
# ===================================================================


class TestPreprocessVideo:
    @pytest.mark.asyncio
    async def test_applies_all_filters(self, fake_input, fake_output, mock_ffmpeg):
        ff, stream, final = mock_ffmpeg
        s = EnhancementService(True, True, True)
        result = await s.preprocess_video(fake_input, fake_output)
        assert result == fake_output
        final.run.assert_called_once_with(quiet=True)

    @pytest.mark.asyncio
    async def test_disables_stabilization(self, fake_input, fake_output, mock_ffmpeg):
        ff, stream, final = mock_ffmpeg
        s = EnhancementService(False, True, True)
        result = await s.preprocess_video(fake_input, fake_output)
        assert result == fake_output
        final.run.assert_called_once_with(quiet=True)

    @pytest.mark.asyncio
    async def test_disables_denoising(self, fake_input, fake_output, mock_ffmpeg):
        ff, stream, final = mock_ffmpeg
        s = EnhancementService(True, False, True)
        result = await s.preprocess_video(fake_input, fake_output)
        assert result == fake_output
        final.run.assert_called_once_with(quiet=True)

    @pytest.mark.asyncio
    async def test_disables_sharpening(self, fake_input, fake_output, mock_ffmpeg):
        ff, stream, final = mock_ffmpeg
        s = EnhancementService(True, True, False)
        result = await s.preprocess_video(fake_input, fake_output)
        assert result == fake_output
        final.run.assert_called_once_with(quiet=True)

    @pytest.mark.asyncio
    async def test_all_disabled_no_filters(self, fake_input, fake_output, mock_ffmpeg):
        ff, stream, final = mock_ffmpeg
        s = EnhancementService(False, False, False)
        result = await s.preprocess_video(fake_input, fake_output)
        assert result == fake_output
        final.run.assert_called_once_with(quiet=True)

    @pytest.mark.asyncio
    async def test_ffmpeg_run_failure(self, fake_input, fake_output, mock_ffmpeg):
        ff, stream, final = mock_ffmpeg
        final.run.side_effect = RuntimeError("ffmpeg crashed")
        s = EnhancementService(False, False, False)
        with pytest.raises(RuntimeError, match="ffmpeg crashed"):
            await s.preprocess_video(fake_input, fake_output)

    @pytest.mark.asyncio
    async def test_filter_chain_order(self, fake_input, fake_output, mock_ffmpeg):
        ff, stream, final = mock_ffmpeg
        s = EnhancementService(True, True, True)
        await s.preprocess_video(fake_input, fake_output)
        filter_calls = [c[0][0] for c in stream.filter.call_args_list]
        assert filter_calls == ["vidstabdetect", "vidstabtransform", "hqdn3d", "unsharp", "scale"]

    @pytest.mark.asyncio
    async def test_output_codec_config(self, fake_input, fake_output, mock_ffmpeg):
        ff, stream, final = mock_ffmpeg
        s = EnhancementService(False, False, False)
        await s.preprocess_video(fake_input, fake_output)
        stream.output.assert_called_once()
        _, kwargs = stream.output.call_args
        assert kwargs["vcodec"] == "libx264"
        assert kwargs["preset"] == "fast"
        assert kwargs["crf"] == 23


# ===================================================================
# UpscaleVideo
# ===================================================================


class TestUpscaleVideo:
    @pytest.mark.asyncio
    async def test_returns_input_when_disabled(self, fake_input, fake_output):
        s = EnhancementService(enable_upscaling=False)
        result = await s.upscale_video(fake_input, fake_output)
        assert result == fake_input

    @pytest.mark.asyncio
    async def test_returns_input_when_missing_deps(self, fake_input, fake_output):
        s = EnhancementService(enable_upscaling=True)
        result = await s.upscale_video(fake_input, fake_output)
        assert result == fake_input


# ===================================================================
# InterpolateVideo
# ===================================================================


class TestInterpolateVideo:
    @pytest.mark.asyncio
    async def test_returns_input_when_disabled(self, fake_input, fake_output):
        s = EnhancementService(enable_interpolation=False)
        result = await s.interpolate_video(fake_input, fake_output)
        assert result == fake_input

    @pytest.mark.asyncio
    async def test_returns_input_when_missing_deps(self, fake_input, fake_output):
        s = EnhancementService(enable_interpolation=True)
        result = await s.interpolate_video(fake_input, fake_output)
        assert result == fake_input


# ===================================================================
# Error handling
# ===================================================================


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_ffmpeg_not_installed(self, fake_input, fake_output):
        s = EnhancementService(False, False, False)
        with patch.dict("sys.modules", {"ffmpeg": None}), pytest.raises(ModuleNotFoundError):
            await s.preprocess_video(fake_input, fake_output)

    @pytest.mark.asyncio
    async def test_upscale_scales_on_cpu_when_no_cuda(self, fake_input, fake_output):
        s = EnhancementService(enable_upscaling=True, gpu_enabled=True)
        result = await s.upscale_video(fake_input, fake_output)
        assert result == fake_input

    @pytest.mark.asyncio
    async def test_upscale_scales_on_cpu_when_gpu_off(self, fake_input, fake_output):
        s = EnhancementService(enable_upscaling=True, gpu_enabled=False)
        result = await s.upscale_video(fake_input, fake_output)
        assert result == fake_input
