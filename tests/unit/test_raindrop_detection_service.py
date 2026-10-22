"""Tests for RaindropDetectionService — raindrop detection in video frames."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()
_mod = load_service_module("raindrop_detection_test", "raindrop_detection_service.py")
RaindropDetectionService = _mod.RaindropDetectionService
RaindropDetection = _mod.RaindropDetection


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def svc():
    return RaindropDetectionService()


@pytest.fixture
def clear_frame():
    """Uniform gray frame — no edges, no raindrops."""
    return np.ones((240, 320, 3), dtype=np.uint8) * 127


@pytest.fixture
def edge_frame():
    """Frame with a small circle that looks like a raindrop."""
    frame = np.ones((240, 320, 3), dtype=np.uint8) * 200
    cv2.circle(frame, (80, 60), 8, (50, 50, 50), -1)
    cv2.circle(frame, (200, 150), 6, (30, 30, 30), -1)
    return frame


# ===========================================================================
# Init
# ===========================================================================


class TestInit:
    def test_default_init(self, svc):
        assert svc.available is True
        assert svc.has_cnn is False
        assert svc._cnn_model is None

    def test_init_with_model_path(self):
        svc = RaindropDetectionService(model_path="/nonexistent/model.pth")
        assert svc.has_cnn is False  # file doesn't exist, gracefully fails


# ===========================================================================
# Sliding window generator
# ===========================================================================


class TestSlidingWindow:
    def test_sliding_window_count(self, svc):
        image = np.ones((100, 100, 3), dtype=np.uint8)
        windows = list(svc._sliding_window(image))
        # (100 - 30 + 1) / 10 = 7.1 -> 7 steps per dimension
        expected = ((100 - 30) // 10 + 1) ** 2
        assert len(windows) == expected

    def test_sliding_window_window_size(self, svc):
        image = np.ones((100, 100, 3), dtype=np.uint8)
        for _x, _y, window in svc._sliding_window(image):
            assert window.shape == (30, 30, 3)
            break

    def test_sliding_window_empty(self, svc):
        image = np.ones((20, 20, 3), dtype=np.uint8)
        windows = list(svc._sliding_window(image))
        assert len(windows) == 0


# ===========================================================================
# Merge overlapping
# ===========================================================================


class TestMergeOverlapping:
    def test_empty_detections(self, svc):
        assert svc._merge_overlapping([]) == []

    def test_single_detection_dropped(self, svc):
        """Single isolated rect is dropped when GROUP_THRESHOLD=1."""
        merged = svc._merge_overlapping([(10, 10, 0.8)])
        assert len(merged) == 0

    def test_overlapping_rectangles_merged(self, svc):
        """Two overlapping detections merge into one group."""
        detections = [(10, 10, 0.8), (12, 10, 0.7)]
        merged = svc._merge_overlapping(detections)
        assert len(merged) == 1


# ===========================================================================
# OpenCV heuristic classify
# ===========================================================================


class TestOpenCVClassifyWindows:
    def test_clear_frame_zero_detections(self, svc, clear_frame):
        detections = svc._opencv_classify_windows(clear_frame)
        assert len(detections) == 0

    def test_grayscale_input(self, svc):
        gray = np.ones((240, 320), dtype=np.uint8) * 127
        detections = svc._opencv_classify_windows(gray)
        assert isinstance(detections, list)

    def test_small_image_no_detections(self, svc):
        small = np.ones((20, 20, 3), dtype=np.uint8) * 127
        detections = svc._opencv_classify_windows(small)
        assert len(detections) == 0


# ===========================================================================
# CNN classify windows (mocked)
# ===========================================================================


class TestCNNClassifyWindows:
    def test_cnn_not_available_returns_empty(self, svc, clear_frame):
        assert svc._cnn_available is False
        detections = svc._cnn_classify_windows(clear_frame)
        assert detections == []

    def test_cnn_classify_with_mocked_torch(self, monkeypatch, clear_frame):
        mock_torch = MagicMock()
        mock_nn = MagicMock()
        mock_model = MagicMock()

        svc = RaindropDetectionService()
        svc._torch = mock_torch
        svc._nn = mock_nn
        svc._cnn_model = mock_model
        svc._cnn_available = True

        # Mock the sliding window to yield a single known window
        def fake_sliding(img):
            yield (5, 5, np.ones((30, 30, 3), dtype=np.float32))

        svc._sliding_window = fake_sliding

        # Mock CNN output -> softmax -> two-class probs where raindrop (index 1) > 0.5
        mock_raw = MagicMock()
        mock_raw.squeeze.return_value = MagicMock()
        mock_model.return_value = mock_raw

        # softmax returns 2D tensor: [[p_clear, p_rain], ...]
        # Code does: probs = softmax(...)[0]; raindrop_prob = float(probs[1])
        class FakeProbs2D:
            def __getitem__(self, _):
                class FakeProbs1D:
                    def __getitem__(self, j):
                        return [0.1, 0.9][j]

                return FakeProbs1D()

        mock_torch.softmax = lambda *a, **kw: FakeProbs2D()
        mock_torch.no_grad.return_value.__enter__.return_value = None
        mock_torch.from_numpy.return_value = MagicMock()

        detections = svc._cnn_classify_windows(clear_frame)
        assert len(detections) == 1
        x, y, conf = detections[0]
        assert x == 5
        assert y == 5
        assert conf == 0.9


# ===========================================================================
# Detect — core method
# ===========================================================================


class TestDetect:
    def test_empty_frames(self, svc):
        result = svc.detect([])
        assert result.frame_count == 0
        assert result.raindrop_count == 0
        assert result.is_rainy is False

    def test_empty_frames_list_contains_none(self, svc):
        result = svc.detect([None])
        assert result.frame_count == 0

    def test_empty_frames_list_contains_empty_array(self, svc):
        empty = np.array([], dtype=np.uint8)
        result = svc.detect([empty])
        assert result.frame_count == 0

    def test_clear_single_frame(self, svc, clear_frame):
        result = svc.detect([clear_frame])
        assert result.frame_count == 1
        assert result.raindrop_count == 0
        assert result.is_rainy is False
        assert result.method == "opencv_heuristic"

    def test_multiple_clear_frames(self, svc, clear_frame):
        result = svc.detect([clear_frame, clear_frame, clear_frame])
        assert result.frame_count == 3
        assert result.raindrop_count == 0

    def test_raindrop_density_zero_for_clear(self, svc, clear_frame):
        result = svc.detect([clear_frame])
        assert result.raindrop_density == 0.0

    def test_average_confidence_zero_for_clear(self, svc, clear_frame):
        result = svc.detect([clear_frame])
        assert result.avg_confidence == 0.0

    def test_single_pixel_frame_no_raindrops(self, svc):
        tiny = np.ones((1, 1, 3), dtype=np.uint8) * 127
        result = svc.detect([tiny])
        # Tiny frame — no sliding windows fit, no detections
        assert result.frame_count == 1
        assert result.raindrop_count == 0


# ===========================================================================
# Detect — mocked opencv classify (force raindrop results)
# ===========================================================================


class TestDetectWithMockedDetections:
    @pytest.fixture
    def svc_with_fake_detections(self):
        svc = RaindropDetectionService()

        # Use overlapping fake detections so groupRectangles keeps them
        def fake_classify(img):
            # All detections at same position so they overlap and merge
            return [(10, 10, 0.8), (10, 10, 0.7), (10, 10, 0.9)]

        svc._opencv_classify_windows = fake_classify
        return svc

    def test_fake_detections_counted(self, svc_with_fake_detections, clear_frame):
        result = svc_with_fake_detections.detect([clear_frame])
        # Three overlapping detections merge into one group
        assert result.raindrop_count >= 1

    def test_is_rainy_with_many_detections(self, svc_with_fake_detections, clear_frame):
        result = svc_with_fake_detections.detect([clear_frame] * 5)
        assert result.is_rainy is True

    def test_avg_confidence_reported(self, svc_with_fake_detections, clear_frame):
        result = svc_with_fake_detections.detect([clear_frame])
        assert result.avg_confidence > 0


# ===========================================================================
# Detect from video file
# ===========================================================================


class TestDetectFromVideoFile:
    def test_video_file_not_found(self, svc):
        result = svc.detect_from_video_file("/nonexistent/video.mp4")
        assert result.frame_count == 0
        assert result.is_rainy is False

    def test_video_file_with_mocked_capture(self, svc, clear_frame):
        """Mock VideoCapture to return a single clear frame."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        # Return one frame, then no more
        mock_cap.read.side_effect = [(True, clear_frame), (False, None)]
        with patch("cv2.VideoCapture", return_value=mock_cap):
            result = svc.detect_from_video_file("fake.mp4", sample_every_n_frames=1, max_frames=5)
        assert result.frame_count == 1
        assert result.raindrop_count == 0

    def test_video_file_unopenable(self, svc):
        with patch("cv2.VideoCapture") as mock_vc:
            mock_vc.return_value.isOpened.return_value = False
            result = svc.detect_from_video_file("unopenable.mp4")
        assert result.frame_count == 0
        assert result.is_rainy is False
