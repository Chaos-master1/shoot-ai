"""Tests for BallTracker, TrackSmoother, and CameraCutDetector."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from conftest import install_kawkab_stubs

install_kawkab_stubs()


# ── Module-level cv2 stub ────────────────────────────────────────────────


def _install_cv2_stub() -> None:
    if "cv2" in sys.modules:
        return
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.COLOR_BGR2HSV = 40
    cv2_stub.RETR_EXTERNAL = 0
    cv2_stub.CHAIN_APPROX_SIMPLE = 1
    cv2_stub.HISTCMP_BHATTACHARYYA = 3
    cv2_stub.NORM_MINMAX = 32
    cv2_stub.CAP_PROP_FPS = 5
    cv2_stub.CAP_PROP_FRAME_COUNT = 7
    for fn in (
        "cvtColor",
        "inRange",
        "erode",
        "dilate",
        "findContours",
        "minEnclosingCircle",
        "contourArea",
        "arcLength",
        "compareHist",
        "calcHist",
        "normalize",
        "KalmanFilter",
        "VideoCapture",
    ):
        setattr(cv2_stub, fn, MagicMock())
    sys.modules["cv2"] = cv2_stub


_install_cv2_stub()

from kawkab.services.ball_tracker import BallDetection, BallTracker
from kawkab.services.camera_cut_detector import CameraCutDetector
from kawkab.services.track_smoother import TrackSmoother

# ═══════════════════════════════════════════════════════════════════════════
# BallTracker
# ═══════════════════════════════════════════════════════════════════════════


class TestBallTracker:
    """BallTracker — HSV + Kalman ball detection."""

    def _kalman_mock(self) -> MagicMock:
        kf = MagicMock()
        kf.statePost = np.array([100.0, 50.0, 8.0, 0.0, 0.0, 0.0], dtype=np.float32)
        kf.predict.return_value = kf.statePost
        kf.correct.return_value = None
        kf.errorCovPost = np.eye(6, dtype=np.float32)
        return kf

    # -- Init ----------------------------------------------------------------

    def test_init_default(self):
        with patch("cv2.KalmanFilter", return_value=self._kalman_mock()):
            tracker = BallTracker()
        assert tracker.fps == 24.0
        assert tracker.dt == pytest.approx(1.0 / 24.0)
        assert tracker.initialized is False
        assert tracker.missed_frames == 0
        assert tracker.confidence == 0.0
        assert len(tracker.trail) == 0

    def test_init_custom_fps(self):
        with patch("cv2.KalmanFilter", return_value=self._kalman_mock()):
            tracker = BallTracker(fps=50.0)
        assert tracker.dt == pytest.approx(1.0 / 50.0)

    def test_init_clamps_fps_to_minimum(self):
        with patch("cv2.KalmanFilter", return_value=self._kalman_mock()):
            tracker = BallTracker(fps=0.1)
        assert tracker.dt == 1.0

    # -- Update --------------------------------------------------------------

    def test_first_update_initializes_kalman(self):
        mock_kf = self._kalman_mock()
        with patch("cv2.KalmanFilter", return_value=mock_kf):
            tracker = BallTracker()
            candidate = {
                "x": 100.0,
                "y": 50.0,
                "radius": 8.0,
                "circularity": 0.85,
                "label": "white",
            }
            with patch(
                "kawkab.services.ball_tracker._find_hsv_candidates", return_value=[candidate]
            ):
                det = tracker.update(np.zeros((10, 10, 3), dtype=np.uint8), 0, 0.0)

        assert det is not None
        assert det.frame == 0
        assert det.timestamp == 0.0
        assert det.x == 100.0
        assert det.y == 50.0
        # HSV-fallback confidence: 0.1 + (circularity - 0.6) * 0.5 = 0.225 for
        # circularity 0.85, and MUST stay below the 0.3 recording gate
        # (CVService records ball detections at conf > 0.3). The old formula
        # (0.3 + circularity * 0.4 = 0.64) crossed the gate by construction,
        # letting every white blob through while marginal true YOLO
        # detections were dropped.
        assert det.conf == pytest.approx(0.225)
        assert det.conf < 0.3
        assert det.is_prediction is False
        assert tracker.initialized is True

    def test_update_subsequent_invokes_predict_and_correct(self):
        mock_kf = self._kalman_mock()
        with patch("cv2.KalmanFilter", return_value=mock_kf):
            tracker = BallTracker()
            candidate = {
                "x": 100.0,
                "y": 50.0,
                "radius": 8.0,
                "circularity": 0.85,
                "label": "white",
            }
            with patch(
                "kawkab.services.ball_tracker._find_hsv_candidates", return_value=[candidate]
            ):
                tracker.update(np.zeros((10, 10, 3), dtype=np.uint8), 0, 0.0)

            mock_kf.reset_mock()
            mock_kf.statePost = np.array([101.0, 51.0, 8.0, 0.0, 0.0, 0.0], dtype=np.float32)
            mock_kf.predict.return_value = mock_kf.statePost

            with patch(
                "kawkab.services.ball_tracker._find_hsv_candidates", return_value=[candidate]
            ):
                det = tracker.update(np.zeros((10, 10, 3), dtype=np.uint8), 1, 0.04)

        assert det is not None
        mock_kf.predict.assert_called_once()
        mock_kf.correct.assert_called_once()

    def test_update_prediction_mode_when_no_candidate(self):
        mock_kf = self._kalman_mock()
        with patch("cv2.KalmanFilter", return_value=mock_kf):
            tracker = BallTracker()
            candidate = {
                "x": 100.0,
                "y": 50.0,
                "radius": 8.0,
                "circularity": 0.85,
                "label": "white",
            }
            with patch(
                "kawkab.services.ball_tracker._find_hsv_candidates", return_value=[candidate]
            ):
                tracker.update(np.zeros((10, 10, 3), dtype=np.uint8), 0, 0.0)

            mock_kf.reset_mock()
            with patch("kawkab.services.ball_tracker._find_hsv_candidates", return_value=[]):
                det = tracker.update(np.zeros((10, 10, 3), dtype=np.uint8), 1, 1.0 / 24.0)

        assert det is not None
        assert det.is_prediction is True
        assert det.conf < 0.3
        mock_kf.predict.assert_called_once()

    def test_update_returns_none_after_missed_limit(self):
        mock_kf = self._kalman_mock()
        with patch("cv2.KalmanFilter", return_value=mock_kf):
            tracker = BallTracker()
            candidate = {
                "x": 100.0,
                "y": 50.0,
                "radius": 8.0,
                "circularity": 0.85,
                "label": "white",
            }
            with patch(
                "kawkab.services.ball_tracker._find_hsv_candidates", return_value=[candidate]
            ):
                tracker.update(np.zeros((10, 10, 3), dtype=np.uint8), 0, 0.0)

            with patch("kawkab.services.ball_tracker._find_hsv_candidates", return_value=[]):
                det = None
                for i in range(32):
                    det = tracker.update(
                        np.zeros((10, 10, 3), dtype=np.uint8), i + 1, (i + 1) / 24.0
                    )

        assert det is None
        assert tracker.initialized is False

    def test_update_no_candidate_uninitialized_returns_none(self):
        with patch("cv2.KalmanFilter", return_value=self._kalman_mock()):
            tracker = BallTracker()
            with patch("kawkab.services.ball_tracker._find_hsv_candidates", return_value=[]):
                det = tracker.update(np.zeros((10, 10, 3), dtype=np.uint8), 0, 0.0)
        assert det is None

    # -- Reset ---------------------------------------------------------------

    def test_reset_clears_state(self):
        mock_kf = self._kalman_mock()
        with patch("cv2.KalmanFilter", return_value=mock_kf):
            tracker = BallTracker()
            candidate = {
                "x": 100.0,
                "y": 50.0,
                "radius": 8.0,
                "circularity": 0.85,
                "label": "white",
            }
            with patch(
                "kawkab.services.ball_tracker._find_hsv_candidates", return_value=[candidate]
            ):
                tracker.update(np.zeros((10, 10, 3), dtype=np.uint8), 0, 0.0)

        tracker.reset()
        assert tracker.initialized is False
        assert tracker.missed_frames == 0
        assert tracker.confidence == 0.0
        assert len(tracker.trail) == 0

    # -- Trail ---------------------------------------------------------------

    def test_get_trail_returns_empty_when_no_trail(self):
        with patch("cv2.KalmanFilter", return_value=self._kalman_mock()):
            tracker = BallTracker()
        assert tracker.get_trail() == []

    def test_get_trail_filters_by_max_age(self):
        mock_kf = self._kalman_mock()
        with patch("cv2.KalmanFilter", return_value=mock_kf):
            tracker = BallTracker()
            candidate = {
                "x": 100.0,
                "y": 50.0,
                "radius": 8.0,
                "circularity": 0.85,
                "label": "white",
            }
            with patch(
                "kawkab.services.ball_tracker._find_hsv_candidates", return_value=[candidate]
            ):
                tracker.update(np.zeros((10, 10, 3), dtype=np.uint8), 0, 0.0)
                tracker.update(np.zeros((10, 10, 3), dtype=np.uint8), 1, 1.0)
                tracker.update(np.zeros((10, 10, 3), dtype=np.uint8), 2, 2.0)

        cutoff = 2.0 - 1.0
        trail = tracker.get_trail(max_age=1.0)
        assert len(trail) == 2
        for b in trail:
            assert b.timestamp >= cutoff

    def test_trail_trims_at_1000_entries(self):
        mock_kf = self._kalman_mock()
        with patch("cv2.KalmanFilter", return_value=mock_kf):
            tracker = BallTracker()
            candidate = {
                "x": 100.0,
                "y": 50.0,
                "radius": 8.0,
                "circularity": 0.85,
                "label": "white",
            }
            with patch(
                "kawkab.services.ball_tracker._find_hsv_candidates", return_value=[candidate]
            ):
                for i in range(1001):
                    tracker.update(np.zeros((10, 10, 3), dtype=np.uint8), i, i / 24.0)

        assert len(tracker.trail) == 500


class TestBallDetection:
    def test_fields(self):
        bd = BallDetection(
            frame=1, timestamp=0.04, x=100.0, y=50.0, conf=0.9, is_prediction=False, radius=8.0
        )
        assert bd.frame == 1
        assert bd.x == 100.0
        assert bd.y == 50.0
        assert bd.conf == 0.9
        assert bd.is_prediction is False
        assert bd.radius == 8.0

    def test_defaults(self):
        bd = BallDetection(frame=0, timestamp=0.0, x=0.0, y=0.0, conf=0.0)
        assert bd.is_prediction is False
        assert bd.radius == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# TrackSmoother
# ═══════════════════════════════════════════════════════════════════════════


class TestTrackSmoother:
    """TrackSmoother — RTS Kalman smoother for track positions."""

    def test_init_default(self):
        smoother = TrackSmoother()
        assert smoother.dt == pytest.approx(1.0 / 24.0)
        assert smoother.F.shape == (4, 4)
        assert smoother.H.shape == (2, 4)

    def test_init_custom(self):
        smoother = TrackSmoother(dt=1.0 / 30.0, process_noise=1e-2, measurement_noise=1e-1)
        assert smoother.dt == pytest.approx(1.0 / 30.0)
        expected_q = np.eye(4, dtype=np.float64) * 1e-2
        expected_r = np.eye(2, dtype=np.float64) * 1e-1
        np.testing.assert_array_almost_equal(smoother.Q, expected_q)
        np.testing.assert_array_almost_equal(smoother.R, expected_r)

    def test_smooth_three_or_more(self):
        smoother = TrackSmoother(dt=1.0 / 24.0)
        frames = [0, 1, 2]
        positions = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
        result = smoother.smooth(frames, positions)
        assert len(result) == 3
        for x, y in result:
            assert isinstance(x, float)
            assert isinstance(y, float)

    def test_smooth_two_points_returns_original(self):
        smoother = TrackSmoother()
        frames = [0, 1]
        positions = [(10.0, 20.0), (11.0, 21.0)]
        result = smoother.smooth(frames, positions)
        assert result == positions

    def test_smooth_one_point_returns_original(self):
        smoother = TrackSmoother()
        frames = [0]
        positions = [(42.0, 17.0)]
        result = smoother.smooth(frames, positions)
        assert result == positions

    def test_smooth_empty_returns_original(self):
        smoother = TrackSmoother()
        result = smoother.smooth([], [])
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════
# CameraCutDetector
# ═══════════════════════════════════════════════════════════════════════════


def _make_cap_mock(
    fps: float = 30.0, total_frames: int = 100, read_success: bool = True
) -> MagicMock:
    """Helper: build a VideoCapture mock."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        5: fps,
        7: float(total_frames),
    }.get(prop, 0.0)

    if not read_success:
        cap.read.return_value = (False, None)
    return cap


class TestCameraCutDetector:
    """CameraCutDetector — broadcast scene-change detection."""

    def test_init_default(self):
        detector = CameraCutDetector()
        assert detector.hue_bins == 32
        assert detector.sat_bins == 8
        assert detector.threshold == 0.35
        assert detector.min_cut_interval == 0.5

    def test_init_custom(self):
        detector = CameraCutDetector(hue_bins=16, sat_bins=4, threshold=0.5, min_cut_interval=1.0)
        assert detector.hue_bins == 16
        assert detector.sat_bins == 4
        assert detector.threshold == 0.5
        assert detector.min_cut_interval == 1.0

    # -- detect_cuts_fast ----------------------------------------------------

    def test_detect_cuts_fast_delegates(self):
        detector = CameraCutDetector()
        with patch.object(detector, "detect_cuts", return_value=[{"frame": 5}]) as mock_dc:
            result = detector.detect_cuts_fast(Path("dummy.mp4"))

        mock_dc.assert_called_once_with(Path("dummy.mp4"), sample_every_n=6, max_frames=0)
        assert result == [{"frame": 5}]

    # -- detect_cuts ---------------------------------------------------------

    def test_detect_cuts_video_not_opened_returns_empty(self):
        cap = _make_cap_mock()
        cap.isOpened.return_value = False
        with patch("cv2.VideoCapture", return_value=cap):
            detector = CameraCutDetector()
            cuts = detector.detect_cuts(Path("nonexistent.mp4"))

        assert cuts == []

    def test_detect_cuts_detects_cut(self):
        cap = _make_cap_mock(fps=30.0, total_frames=100)
        cap.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None),
        ]

        with patch("cv2.VideoCapture", return_value=cap):
            detector = CameraCutDetector(threshold=0.1)
            hists = [np.array([0.1, 0.9]), np.array([0.9, 0.1]), np.array([0.8, 0.2])]
            with (
                patch.object(detector, "_compute_hsv_hist", side_effect=hists),
                patch("cv2.compareHist", side_effect=[0.5, 0.05]),
            ):
                cuts = detector.detect_cuts(Path("dummy.mp4"), sample_every_n=1, max_frames=3)

        assert len(cuts) == 1
        assert cuts[0]["frame"] == 1
        assert cuts[0]["diff_score"] == 0.5

    def test_detect_cuts_no_cuts(self):
        cap = _make_cap_mock(fps=30.0, total_frames=100)
        cap.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None),
        ]

        with patch("cv2.VideoCapture", return_value=cap):
            detector = CameraCutDetector(threshold=0.5)
            with (
                patch.object(detector, "_compute_hsv_hist", return_value=np.array([0.5, 0.5])),
                patch("cv2.compareHist", return_value=0.0),
            ):
                cuts = detector.detect_cuts(Path("dummy.mp4"), sample_every_n=1, max_frames=2)

        assert len(cuts) == 0

    # -- _cuts_to_segments ---------------------------------------------------

    def test_cuts_to_segments(self):
        detector = CameraCutDetector()
        cap = _make_cap_mock(fps=30.0, total_frames=200)
        with patch("cv2.VideoCapture", return_value=cap):
            cuts = [
                {"frame": 30, "timestamp": 1.0, "diff_score": 0.5},
                {"frame": 90, "timestamp": 3.0, "diff_score": 0.6},
            ]
            segments = detector._cuts_to_segments(cuts, Path("dummy.mp4"))

        assert len(segments) == 3
        assert segments[0]["start_frame"] == 0
        assert segments[0]["end_frame"] == 30
        assert segments[1]["start_frame"] == 30
        assert segments[1]["end_frame"] == 90
        assert segments[2]["start_frame"] == 90
        assert segments[2]["end_frame"] == 200

    def test_cuts_to_segments_skips_small_gaps(self):
        detector = CameraCutDetector()
        cap = _make_cap_mock(fps=30.0, total_frames=100)
        with patch("cv2.VideoCapture", return_value=cap):
            cuts = [{"frame": 3, "timestamp": 0.1, "diff_score": 0.5}]
            segments = detector._cuts_to_segments(cuts, Path("dummy.mp4"))

        assert len(segments) == 1
        assert segments[0]["start_frame"] == 3
        assert segments[0]["end_frame"] == 100

    def test_cuts_to_segments_empty_cuts(self):
        detector = CameraCutDetector()
        cap = _make_cap_mock(fps=30.0, total_frames=100)
        with patch("cv2.VideoCapture", return_value=cap):
            segments = detector._cuts_to_segments([], Path("dummy.mp4"))

        assert len(segments) == 1
        assert segments[0]["start_frame"] == 0
        assert segments[0]["end_frame"] == 100

    # -- get_camera_segments -------------------------------------------------

    def test_get_camera_segments_integrates(self):
        detector = CameraCutDetector()
        cuts_result = [{"frame": 30, "timestamp": 1.0, "diff_score": 0.5}]

        cap = _make_cap_mock(fps=30.0, total_frames=200)
        with (
            patch("cv2.VideoCapture", return_value=cap),
            patch.object(detector, "detect_cuts", return_value=cuts_result),
        ):
            segments = detector.get_camera_segments(Path("dummy.mp4"), sample_every_n=1)

        assert len(segments) == 2
        assert segments[0]["start_frame"] == 0
        assert segments[0]["end_frame"] == 30
        assert segments[1]["start_frame"] == 30
        assert segments[1]["end_frame"] == 200
