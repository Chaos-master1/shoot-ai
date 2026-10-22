"""Tests for AudioService."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()

_mod = load_service_module("kawkab.services.audio_service", "audio_service.py")
AudioService = _mod.AudioService


def _install_fake_faster_whisper(monkeypatch, fake_model):
    """faster_whisper is imported lazily inside AudioService.initialize()
    (`from faster_whisper import WhisperModel`), not at module level -- so
    it must be faked via sys.modules, not monkeypatch.setattr on the
    kawkab module (which never has a WhisperModel attribute to patch)."""
    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = lambda *a, **kw: fake_model
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)


def _install_fake_librosa(monkeypatch, fake_librosa=None):
    """librosa is imported lazily inside detect_whistles()/analyze_crowd_noise()
    (`import librosa`), not at module level -- same reasoning as above."""
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa or MagicMock())


class TestAudioService:
    @pytest.mark.asyncio
    async def test_initialize_skips_if_disabled(self):
        service = AudioService(enable_transcription=False, enable_whistle_detection=False)
        await service.initialize()
        assert service._model is None

    @pytest.mark.asyncio
    async def test_initialize_loads_whisper(self, monkeypatch):
        fake_model = MagicMock()
        _install_fake_faster_whisper(monkeypatch, fake_model)
        service = AudioService(enable_transcription=True, enable_whistle_detection=False)
        await service.initialize()
        assert service._model is fake_model

    @pytest.mark.asyncio
    async def test_transcribe(self):
        fake_segment = MagicMock()
        fake_segment.text = "hello world"
        fake_segment.start = 0.0
        fake_segment.end = 2.0
        fake_info = MagicMock()
        fake_info.language = "en"
        fake_model = MagicMock()
        fake_model.transcribe.return_value = ([fake_segment], fake_info)
        service = AudioService(enable_transcription=True, enable_whistle_detection=False)
        service._model = fake_model
        result = await service.transcribe_video(Path("test.mp3"))
        assert len(result) == 1
        assert result[0]["text"] == "hello world"
        assert result[0]["language"] == "en"

    @pytest.mark.asyncio
    async def test_transcribe_no_model(self):
        service = AudioService(enable_transcription=False)
        result = await service.transcribe_video(Path("test.mp3"))
        assert result == []

    @pytest.mark.asyncio
    async def test_detect_whistles(self, monkeypatch):
        _install_fake_librosa(monkeypatch)
        fake_model = MagicMock()
        service = AudioService(enable_transcription=False, enable_whistle_detection=True)
        service._model = fake_model
        # A bare-mock librosa can't produce a real signal, so the function's
        # own exception handling degrades to []; this is the resilience
        # path (no unhandled exception), which is exactly what's under test.
        result = await service.detect_whistles(Path("test.mp3"))
        assert isinstance(result, list)
        assert result == []

    @pytest.mark.asyncio
    async def test_detect_whistles_disabled(self):
        service = AudioService(enable_whistle_detection=False)
        result = await service.detect_whistles(Path("test.mp3"))
        assert result == []

    @pytest.mark.asyncio
    async def test_analyze_crowd_noise(self, monkeypatch):
        _install_fake_librosa(monkeypatch)
        service = AudioService(enable_transcription=False, enable_crowd_analysis=True)
        result = await service.analyze_crowd_noise(Path("test.mp3"))
        # Same resilience path as detect_whistles: a bare-mock librosa
        # can't produce real audio data, so this exercises the "always
        # returns this exact shape" contract rather than real numbers.
        assert "avg_intensity" in result
        assert result == {
            "avg_intensity": 0.0,
            "peak_intensity": 0.0,
            "peak_time": 0.0,
            "samples": 0,
        }

    @pytest.mark.asyncio
    async def test_analyze_crowd_noise_disabled(self):
        service = AudioService(enable_crowd_analysis=False)
        result = await service.analyze_crowd_noise(Path("test.mp3"))
        assert result["avg_intensity"] == 0.0

    @pytest.mark.asyncio
    async def test_analyze_crowd_noise_real_signal(self, monkeypatch):
        """Happy path with a real (synthetic) signal, not a bare mock --
        verifies the actual RMS-intensity computation, not just the
        degrade-gracefully fallback the other crowd-noise test exercises."""
        import numpy as np

        sr = 8000
        y = (np.sin(2 * np.pi * 440 * np.arange(sr) / sr) * 0.5).astype("float32")
        fake_librosa = MagicMock()
        fake_librosa.load.return_value = (y, sr)
        fake_librosa.feature.rms.return_value = np.array([[0.1, 0.5, 0.3, 0.5]])
        fake_librosa.frames_to_time.return_value = np.array([0.0, 1.0, 2.0, 3.0])
        _install_fake_librosa(monkeypatch, fake_librosa)

        service = AudioService(enable_crowd_analysis=True)
        result = await service.analyze_crowd_noise(Path("test.mp3"))

        assert result["samples"] == 4
        assert result["avg_intensity"] == pytest.approx(0.35)
        assert result["peak_intensity"] == pytest.approx(0.5)
        assert result["peak_time"] in (1.0, 3.0)  # first argmax hit

    @pytest.mark.asyncio
    async def test_initialize_handles_import_error(self, monkeypatch):
        """Simulates faster-whisper being absent/broken by providing a fake
        faster_whisper module that deliberately has no WhisperModel
        attribute -- `from faster_whisper import WhisperModel` then raises
        a genuine ImportError via Python's own import machinery, which
        initialize() must catch, leaving _model as None."""
        fake_module = types.ModuleType("faster_whisper")  # no WhisperModel defined
        monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

        service = AudioService(enable_transcription=True)
        await service.initialize()
        assert service._model is None
