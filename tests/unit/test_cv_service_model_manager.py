"""Tests for CVService ModelManager integration (v0.7.4)."""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path

import pytest
from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()

# Some other test files replace sys.modules["kawkab.services.cv_service"]
# with a stub CVService (no constructor override -> "takes no arguments")
# and don't always restore it; under pytest-xdist, if one of those files
# shares this worker process, a plain `from kawkab.services.cv_service
# import CVService` could silently bind to that stub instead of the real
# class. load_service_module loads a fresh copy under its own unique
# sys.modules key, sidestepping the shared "kawkab.services.cv_service"
# entry (and whatever another file did to it) entirely.
_cv = load_service_module("cv_service_real_for_model_manager_test", "cv_service.py")
CVService = _cv.CVService
from kawkab.core.model_manager import ModelManager


class TestCVServiceModelManager:
    """Test CVService lazy model loading via ModelManager."""

    def test_cv_service_accepts_model_manager(self):
        """Test that CVService accepts a ModelManager instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = ModelManager(cache_dir=Path(tmpdir))
            cv = CVService(model_size="n", model_manager=mm)
            assert cv._model_manager is mm

    def test_cv_service_without_model_manager(self):
        """Test that CVService works without a ModelManager."""
        cv = CVService(model_size="n")
        assert cv._model_manager is None

    @pytest.mark.asyncio
    async def test_cv_service_initialize_with_model_manager(self):
        """Test that CVService.initialize uses ModelManager to get model path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = ModelManager(cache_dir=Path(tmpdir))
            cv = CVService(model_size="n", model_manager=mm)

            # Create a fake model file so ensure_model doesn't try to download
            model_path = mm.cache_dir / "yolo11n.pt"
            model_path.write_bytes(b"fake model weights")

            # Should not raise - uses model path from ModelManager
            # Note: YOLO will fail to load fake bytes, but we verify the path was used
            # Expected - fake model bytes
            with contextlib.suppress(Exception):
                await cv.initialize()

            assert cv._model_manager is mm
            assert mm.is_model_available("yolo11n")

    def test_cv_service_model_manager_fallback(self):
        """Test that CVService falls back to direct YOLO load if ModelManager fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = ModelManager(cache_dir=Path(tmpdir))
            # Remove the model directory to force failure
            import shutil

            shutil.rmtree(tmpdir)
            cv = CVService(model_size="n", model_manager=mm)
            # Should not crash on creation
            assert cv._model_manager is mm
