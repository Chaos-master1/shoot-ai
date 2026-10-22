"""Tests for homography service operations."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()

import numpy as np
import pytest


def _install_stubs():
    if hasattr(sys.modules.get("kawkab.services", None), "homography_service"):
        return
    import kawkab

    if not hasattr(kawkab, "services"):
        services_mod = types.ModuleType("kawkab.services")
        sys.modules["kawkab.services"] = services_mod
    cv_mod = types.ModuleType("kawkab.services.cv_service")

    class FrameDetections:
        pass

    class MatchTrackData:
        pass

    cv_mod.FrameDetections = FrameDetections
    cv_mod.MatchTrackData = MatchTrackData
    sys.modules["kawkab.services.cv_service"] = cv_mod


_install_stubs()
_h = load_service_module("h_test", "homography_service.py")
HomographyService = _h.HomographyService
HomographyMatrix = _h.HomographyMatrix


class TestHomographyService:
    """12 tests for homography computation and error handling."""

    def test_service_init(self):
        hs = HomographyService()
        assert hs is not None
        assert hasattr(hs, "_calibrations")

    def test_save_load_calibration(self):
        hs = HomographyService()
        matrix_data = np.eye(3, dtype=np.float64)
        hm = HomographyMatrix(matrix=matrix_data.tolist(), confidence=0.95, error_px=1.2)
        hs.save_calibration(1, hm)
        loaded = hs.load_calibration(1)
        assert loaded is not None
        assert loaded.confidence == 0.95
        assert loaded.error_px == 1.2

    def test_load_nonexistent(self):
        hs = HomographyService()
        result = hs.load_calibration(999)
        assert result is None

    def test_compute_homography_from_corners(self):
        with patch.object(_h, "cv2_find_homography") as mock_find_h:
            mock_find_h.return_value = (np.eye(3, dtype=np.float64), None)
            hs = HomographyService()
            pixel_corners = [(100, 200), (500, 200), (500, 400), (100, 400)]
            matrix = hs.compute_homography_from_corners(
                pixel_corners=pixel_corners,
                pitch_length_m=105.0,
                pitch_width_m=68.0,
            )
            assert matrix is not None
            assert len(matrix.matrix) == 3
            assert all(len(row) == 3 for row in matrix.matrix)
            assert 0 <= matrix.confidence <= 1.0

    def test_homography_matrix_creation(self):
        m = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        hm = HomographyMatrix(matrix=m, confidence=0.8, error_px=0.5)
        assert hm.pitch_length_m == 105.0
        assert hm.pitch_width_m == 68.0

    def test_homography_matrix_min_corners(self):
        hs = HomographyService()
        with pytest.raises(ValueError):
            hs.compute_homography_from_corners(
                pixel_corners=[(0, 0)],
                pitch_length_m=105.0,
                pitch_width_m=68.0,
            )

    def test_homography_negative_confidence(self):
        m = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        hm = HomographyMatrix(matrix=m, confidence=-0.1, error_px=0.5)
        assert hm.confidence == -0.1

    def test_homography_service_default_state(self):
        hs = HomographyService()
        assert len(hs._calibrations) == 0

    def test_compute_homography_empty_corners(self):
        hs = HomographyService()
        with pytest.raises(ValueError):
            hs.compute_homography_from_corners(
                pixel_corners=[],
                pitch_length_m=105.0,
                pitch_width_m=68.0,
            )

    def test_bad_corner_format_three_points(self):
        hs = HomographyService()
        with pytest.raises(ValueError):
            hs.compute_homography_from_corners(
                pixel_corners=[(0, 0), (1, 1), (2, 2)],
                pitch_length_m=105.0,
                pitch_width_m=68.0,
            )

    def test_overwrite_calibration(self):
        hs = HomographyService()
        m1 = HomographyMatrix(
            matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], confidence=0.5, error_px=2.0
        )
        hs.save_calibration(1, m1)
        m2 = HomographyMatrix(
            matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 1]], confidence=0.9, error_px=0.5
        )
        hs.save_calibration(1, m2)
        loaded = hs.load_calibration(1)
        assert loaded.confidence == 0.9
        assert loaded.error_px == 0.5

    def test_compute_homography_returns_homographymatrix(self):
        with patch.object(_h, "cv2_find_homography") as mock_find_h:
            mock_find_h.return_value = (np.eye(3, dtype=np.float64), None)
            hs = HomographyService()
            corners = [(0, 0), (640, 0), (640, 480), (0, 480)]
            matrix = hs.compute_homography_from_corners(
                pixel_corners=corners,
                pitch_length_m=105.0,
                pitch_width_m=68.0,
            )
            assert isinstance(matrix, HomographyMatrix)
            assert matrix.source == "manual"
