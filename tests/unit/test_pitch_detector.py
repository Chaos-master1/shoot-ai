"""Tests for CV-based pitch / line detection."""

from __future__ import annotations

from unittest.mock import patch

from conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.services.pitch_detector import CalibrationGuess, PitchDetector


class TestAvailable:
    def test_available_true_when_opencv_installed(self):
        pd = PitchDetector()
        assert pd.available is True

    @patch("kawkab.services.pitch_detector.PitchDetector._check_opencv", return_value=False)
    def test_available_false_when_opencv_missing(self, mock_check):
        pd = PitchDetector()
        assert pd.available is False


class TestDefaultGuess:
    def test_default_guess_with_dimensions(self):
        pd = PitchDetector()
        guess = pd._default_guess(1280, 720)
        assert isinstance(guess, CalibrationGuess)
        assert guess.image_width == 1280
        assert guess.image_height == 720
        assert guess.confidence == 0.0
        assert "tl" in guess.corners
        assert "br" in guess.corners
        assert guess.corners["tl"] == (128.0, 108.0)
        assert guess.corners["br"] == (1152.0, 612.0)

    def test_default_guess_with_zero_dimensions(self):
        pd = PitchDetector()
        guess = pd._default_guess(0, 0)
        assert guess.image_width == 0
        assert guess.image_height == 0
        assert guess.corners == {}
        assert guess.notes == ["no input"]

    def test_default_guess_with_custom_notes(self):
        pd = PitchDetector()
        guess = pd._default_guess(640, 480, notes=["custom note"])
        assert "custom note" in guess.notes


class TestClassifyLines:
    def test_classifies_horizontal_lines(self):
        pd = PitchDetector()
        lines = [[(0, 100, 200, 102)], [(50, 200, 250, 198)]]
        h, v = pd._classify_lines(lines, 640, 480)
        assert len(h) == 2
        assert len(v) == 0

    def test_classifies_vertical_lines(self):
        pd = PitchDetector()
        lines = [[(100, 0, 102, 200)], [(200, 50, 198, 250)]]
        h, v = pd._classify_lines(lines, 640, 480)
        assert len(h) == 0
        assert len(v) == 2

    def test_classifies_mixed_lines(self):
        pd = PitchDetector()
        lines = [[(0, 100, 200, 100)], [(100, 0, 100, 200)]]
        h, v = pd._classify_lines(lines, 640, 480)
        assert len(h) == 1
        assert len(v) == 1

    def test_vertical_with_x2_equal_x1_is_vertical(self):
        pd = PitchDetector()
        lines = [[(100, 0, 100, 200)]]
        h, v = pd._classify_lines(lines, 640, 480)
        assert len(v) == 1
        assert len(h) == 0


class TestEstimateVanishingPoints:
    def test_vanishing_point_from_h_lines(self):
        pd = PitchDetector()
        h_lines = [(0, 100, 200, 100), (50, 200, 250, 200)]
        vps = pd._estimate_vanishing_points(h_lines, [], 640, 480)
        assert len(vps) >= 1

    def test_vanishing_point_from_v_lines(self):
        pd = PitchDetector()
        v_lines = [(100, 0, 100, 200), (200, 50, 200, 250)]
        vps = pd._estimate_vanishing_points([], v_lines, 640, 480)
        assert len(vps) >= 1

    def test_fallback_when_no_lines(self):
        pd = PitchDetector()
        vps = pd._estimate_vanishing_points([], [], 640, 480)
        assert len(vps) == 1
        assert vps[0] == (320.0, 240.0)


class TestCornersFromLines:
    def test_corners_with_lines(self):
        pd = PitchDetector()
        h_lines = [(0, 100, 200, 100), (0, 300, 200, 300)]
        v_lines = [(100, 0, 100, 400), (300, 0, 300, 400)]
        corners = pd._corners_from_lines(h_lines, v_lines, 640, 480, [])
        assert "tl" in corners
        assert corners["tl"] == (100.0, 100.0)
        assert corners["br"] == (300.0, 300.0)

    def test_corners_fallback_when_no_lines(self):
        pd = PitchDetector()
        corners = pd._corners_from_lines([], [], 640, 480, [])
        assert corners["tl"] == (64.0, 72.0)
        assert corners["br"] == (576.0, 408.0)


class TestScoreConfidence:
    def test_no_lines_zero_confidence(self):
        assert PitchDetector._score_confidence(0, 0) == 0.0

    def test_score_increases_with_line_count(self):
        low = PitchDetector._score_confidence(2, 2)
        high = PitchDetector._score_confidence(10, 10)
        assert high > low

    def test_bonus_for_both_horizontal_and_vertical(self):
        both = PitchDetector._score_confidence(3, 3)
        only_h = PitchDetector._score_confidence(6, 0)
        assert both > only_h

    def test_score_capped_at_1(self):
        score = PitchDetector._score_confidence(20, 20)
        assert score <= 1.0


class TestDetectNoOpenCV:
    @patch("kawkab.services.pitch_detector.PitchDetector._check_opencv", return_value=False)
    def test_detect_returns_default_when_opencv_missing(self, mock_check):
        pd = PitchDetector()
        guess = pd.detect("not_a_frame")
        assert guess.confidence == 0.0
        assert "opencv not installed" in guess.notes

    @patch("kawkab.services.pitch_detector.PitchDetector._check_opencv", return_value=False)
    def test_detect_with_zero_dims(self, mock_check):
        pd = PitchDetector()
        guess = pd.detect(b"")
        assert guess.confidence == 0.0


class TestDetectWithOpenCV:
    def test_detect_with_empty_bytes(self):
        pd = PitchDetector()
        guess = pd.detect(b"")
        assert guess.confidence == 0.0

    def test_detect_with_none(self):
        pd = PitchDetector()
        guess = pd.detect(None)
        assert guess.confidence == 0.0

    def test_init_params_stored(self):
        pd = PitchDetector(min_line_length=100, max_line_gap=15, canny_low=40, canny_high=160)
        assert pd.min_line_length == 100
        assert pd.max_line_gap == 15
        assert pd.canny_low == 40
        assert pd.canny_high == 160
