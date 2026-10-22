"""End-to-end video pipeline test — processes a real match clip.

Tests the full pipeline: video open → CVService → MatchTrackData → basic stats.
Skipped when GPU-accelerated dependencies (ultralytics, boxmot) are not installed.

CI-friendly path: verifies video structure with OpenCV only.
Full path: requires ultralytics + boxmot + torch.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from kawkab.core.gpu_acceleration import detect_gpu_tier

TESTS_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = TESTS_DIR.parent

# Smallest real-match clip in the repo — 4.2 MB, ~89 seconds
REAL_MATCH_VIDEO = PROJECT_ROOT / "data" / "real_match.mp4"

# Reference clip used in STATUS.md — 50.7 MB, 60 seconds
SWEDEN_TEST_60S = PROJECT_ROOT / "data" / "sweden_test_60s.mp4"

_ULTALYTICS_AVAILABLE = False
try:
    import ultralytics  # noqa: F401

    _ULTALYTICS_AVAILABLE = True
except ImportError:
    pass


# ── CI-safe: video structure checks (no GPU required) ────────────────


class TestVideoStructure:
    """Fast, zero-dependency checks that video files are valid."""

    def test_real_match_video_exists(self):
        assert REAL_MATCH_VIDEO.exists(), f"Missing: {REAL_MATCH_VIDEO}"
        assert REAL_MATCH_VIDEO.stat().st_size > 100_000

    def test_sweden_video_exists(self):
        assert SWEDEN_TEST_60S.exists(), f"Missing: {SWEDEN_TEST_60S}"
        assert SWEDEN_TEST_60S.stat().st_size > 1_000_000

    def test_real_match_duration(self):
        cap = cv2.VideoCapture(str(REAL_MATCH_VIDEO))
        assert cap.isOpened(), "Cannot open real_match.mp4"
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert fps > 0, "Zero FPS"
        assert total > 0, "Zero frames"
        duration = total / fps
        assert 1 <= duration <= 300, f"Unexpected duration: {duration:.1f}s"

    def test_real_match_frame_dimensions(self):
        cap = cv2.VideoCapture(str(REAL_MATCH_VIDEO))
        assert cap.isOpened()
        ret, frame = cap.read()
        cap.release()
        assert ret, "Could not read first frame"
        h, w = frame.shape[:2]
        assert w >= 320 and h >= 240, f"Frame too small: {w}x{h}"
        assert frame.shape[2] == 3, "Not a 3-channel BGR frame"

    def test_sweden_video_structure(self):
        cap = cv2.VideoCapture(str(SWEDEN_TEST_60S))
        assert cap.isOpened()
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = cap.read()
        cap.release()
        assert ret and frame is not None
        h, w = frame.shape[:2]
        assert w >= 640 and h >= 360
        duration = total / fps
        assert 55 <= duration <= 120


# ── Full pipeline: requires GPU deps ─────────────────────────────────


@pytest.mark.skipif(
    not _ULTALYTICS_AVAILABLE,
    reason="ultralytics not installed — cannot test full pipeline",
)
class TestFullPipeline:
    """Full CV pipeline test. Requires ultralytics + torch + boxmot."""

    @pytest.fixture(autouse=True)
    def _ensure_real_cv_service(self, monkeypatch):
        # Several other test files replace sys.modules
        # ["kawkab.services.cv_service"] with a stub CVService (no
        # constructor override -> "takes no arguments") and don't always
        # restore it; under pytest-xdist, if one of those files shares this
        # worker process, the `from ... import CVService` below could
        # silently bind to that stub. monkeypatch.delitem (unlike a bare
        # sys.modules.pop) auto-restores whatever was there right after
        # this test, so it doesn't itself become a new source of leakage
        # for some other file relying on that entry persisting.
        import sys

        monkeypatch.delitem(sys.modules, "kawkab.services.cv_service", raising=False)
        yield

    @pytest.mark.asyncio
    async def test_cv_service_initializes(self):
        from kawkab.services.cv_service import CVService

        svc = CVService(model_size="n", gpu_enabled=False)
        await svc.initialize()
        assert svc._initialized
        assert svc._model is not None
        await svc.shutdown()

    @pytest.mark.asyncio
    async def test_detect_single_frame(self):
        from kawkab.services.cv_service import CVService

        svc = CVService(model_size="n", gpu_enabled=False)
        await svc.initialize()
        cap = cv2.VideoCapture(str(REAL_MATCH_VIDEO))
        ret, frame = cap.read()
        cap.release()
        assert ret
        result = await svc.detect_frame(frame, frame_number=0, timestamp=0.0)
        assert result.frame_number == 0
        assert result.image_width > 0
        assert result.image_height > 0
        await svc.shutdown()

    @pytest.mark.asyncio
    async def test_process_video_returns_match_data(self):
        from kawkab.services.cv_service import CVService

        svc = CVService(model_size="n", gpu_enabled=False)
        await svc.initialize()
        match_data = await svc.process_video(
            REAL_MATCH_VIDEO,
            frame_skip=30,
            enable_team_detection=False,
        )
        assert match_data.fps > 0
        assert match_data.total_frames > 0
        assert len(match_data.frames) > 0
        await svc.shutdown()

    @pytest.mark.asyncio
    async def test_track_registry_is_populated(self):
        from kawkab.services.cv_service import CVService

        svc = CVService(model_size="n", gpu_enabled=False)
        await svc.initialize()
        match_data = await svc.process_video(
            REAL_MATCH_VIDEO,
            frame_skip=30,
            enable_team_detection=False,
        )
        if match_data.track_registry:
            sample_tid = list(match_data.track_registry.keys())[0]
            entry = match_data.track_registry[sample_tid]
            assert "track_id" in entry
            assert "frames_tracked" in entry
        await svc.shutdown()

    @pytest.mark.asyncio
    async def test_no_exception_on_short_clip(self):
        from kawkab.services.cv_service import CVService

        svc = CVService(model_size="n", gpu_enabled=False)
        await svc.initialize()
        match_data = await svc.process_video(
            SWEDEN_TEST_60S,
            frame_skip=30,
            enable_team_detection=False,
        )
        assert match_data is not None
        await svc.shutdown()

    @pytest.mark.asyncio
    async def test_gpu_tier_detected(self):
        tier = detect_gpu_tier()
        assert tier in ("low", "medium", "high", "ultra", "cpu")
