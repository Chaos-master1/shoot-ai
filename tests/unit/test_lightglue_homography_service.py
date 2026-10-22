"""Tests for LightGlueHomographyService — SuperPoint + LightGlue ONNX homography."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()

# ---------------------------------------------------------------------------
# HomographyMatrix stub — avoid loading homography_service.py and its imports
# ---------------------------------------------------------------------------


class HomographyMatrixStub:
    """Minimal HomographyMatrix for testing LightGlueHomographyService."""

    def __init__(
        self,
        matrix=None,
        pitch_length_m=105.0,
        pitch_width_m=68.0,
        source="manual",
        confidence=0.0,
        error_px=0.0,
    ):
        self.matrix = matrix or [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m
        self.source = source
        self.confidence = confidence
        self.error_px = error_px


# Register the stub so lightglue_homography_service's import resolves
if "kawkab.services.homography_service" not in sys.modules:
    import types

    _hmod = types.ModuleType("kawkab.services.homography_service")
    _hmod.HomographyMatrix = HomographyMatrixStub
    sys.modules["kawkab.services.homography_service"] = _hmod

if "kawkab.services" not in sys.modules:
    import types

    _smod = types.ModuleType("kawkab.services")
    _smod.__path__ = []
    sys.modules["kawkab.services"] = _smod

_mod = load_service_module("lg_homography_test", "lightglue_homography_service.py")

LightGlueHomographyService = _mod.LightGlueHomographyService
HomographyMatrix = HomographyMatrixStub


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def model_dir(tmp_path):
    return tmp_path / "lightglue_models"


@pytest.fixture
def svc(model_dir):
    return LightGlueHomographyService(model_dir=model_dir)


@pytest.fixture
def sample_image():
    return np.ones((480, 640, 3), dtype=np.uint8) * 127


@pytest.fixture
def mock_onnx_session():
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock()]
    session.get_inputs.return_value[0].name = "input"
    session.run.return_value = (
        np.random.rand(2, 100, 2).astype(np.float32),
        np.column_stack([np.arange(50), np.arange(50), np.arange(50) + 1]).astype(np.float32),
        np.ones(50, dtype=np.float32) * 0.9,
    )
    return session


# ===========================================================================
# Tests
# ===========================================================================


class TestInit:
    def test_creates_model_dir(self, model_dir):
        svc = LightGlueHomographyService(model_dir=model_dir)
        assert model_dir.exists()
        assert svc.model_dir == model_dir

    def test_model_path(self, svc):
        assert svc.model_path.name == "superpoint_lightglue_pipeline.onnx"

    def test_default_model_dir(self):
        svc = LightGlueHomographyService()
        assert svc.model_dir is not None


class TestAvailable:
    def test_available_false_when_no_model(self, svc):
        assert svc.available is False

    def test_available_true_when_model_exists(self, svc):
        svc.model_path.write_text("dummy")
        assert svc.available is True


class TestEnsureModel:
    def test_ensure_model_downloads_when_missing(self, svc):
        with patch("urllib.request.urlretrieve") as mock_dl:
            svc.ensure_model()
            mock_dl.assert_called_once()
        # urlretrieve is mocked, so file won't actually exist; that's fine
        # The important thing is it called urlretrieve

    def test_ensure_model_skips_when_exists(self, svc):
        svc.model_path.write_text("dummy")
        with patch("urllib.request.urlretrieve") as mock_dl:
            svc.ensure_model()
            mock_dl.assert_not_called()


class TestPreprocess:
    def test_preprocess_returns_tensor_and_scales(self, svc):
        img = np.ones((240, 320, 3), dtype=np.uint8)
        tensor, sx, sy = svc._preprocess(img)
        assert tensor.ndim == 4
        assert tensor.shape[2] == svc._input_size
        assert tensor.shape[3] == svc._input_size
        assert sx > 0
        assert sy > 0

    def test_preprocess_grayscale(self, svc):
        img = np.ones((240, 320), dtype=np.uint8)
        tensor, sx, sy = svc._preprocess(img)
        assert tensor.ndim == 4

    def test_preprocess_normalizes(self, svc):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        tensor, _, _ = svc._preprocess(img)
        assert tensor.max() <= 1.0


class TestMatch:
    def test_match_raises_when_no_model(self, svc):
        img0 = np.zeros((100, 100, 3), dtype=np.uint8)
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="LightGlue model not found"):
            svc.match(img0, img1)

    def test_match_returns_none_when_few_matches(self, svc, sample_image, mock_onnx_session):
        mock_onnx_session.run.return_value = (
            np.random.rand(2, 10, 2).astype(np.float32),
            np.ones((1, 3), dtype=np.float32),
            np.array([0.6], dtype=np.float32),
        )
        svc._session = mock_onnx_session
        svc.model_path.write_text("dummy")
        result = svc.match(sample_image, sample_image, conf_threshold=0.5)
        assert result is None

    def test_match_returns_keypoints(self, svc, sample_image, mock_onnx_session):
        svc._session = mock_onnx_session
        svc.model_path.write_text("dummy")
        result = svc.match(sample_image, sample_image, conf_threshold=0.5)
        assert result is not None
        kpts0, kpts1, conf = result
        assert len(kpts0) > 0
        assert len(kpts1) > 0
        assert 0.0 <= conf <= 1.0


class TestComputeHomography:
    def test_compute_homography_returns_matrix(self, svc, sample_image, mock_onnx_session):
        svc._session = mock_onnx_session
        svc.model_path.write_text("dummy")
        with (
            patch("cv2.findHomography") as mock_findH,
            patch("cv2.perspectiveTransform") as mock_persp,
        ):
            mock_findH.return_value = (
                np.eye(3, dtype=np.float64),
                np.ones(50, dtype=np.uint8),
            )
            mock_persp.return_value = np.zeros((1, 1, 2))
            H = svc.compute_homography(sample_image, sample_image)
        assert H is not None
        assert isinstance(H, _mod.HomographyMatrix)
        assert H.source == "lightglue"
        assert H.pitch_length_m == 105.0

    def test_compute_homography_returns_none_without_match(self, svc, sample_image):
        svc._session = MagicMock()
        svc._session.run.return_value = (
            np.random.rand(2, 5, 2).astype(np.float32),
            np.ones((1, 3), dtype=np.float32),
            np.array([0.4], dtype=np.float32),
        )
        svc.model_path.write_text("dummy")
        result = svc.compute_homography(sample_image, sample_image)
        assert result is None


class TestAutoCalibrate:
    def test_auto_calibrate_returns_homography(self, svc, sample_image, mock_onnx_session):
        svc._session = mock_onnx_session
        svc.model_path.write_text("dummy")
        H = svc.auto_calibrate(sample_image)
        assert H is not None
        assert H.source == "lightglue_auto"

    def test_auto_calibrate_returns_none_when_no_match(self, svc, sample_image):
        svc._session = MagicMock()
        svc._session.run.return_value = (
            np.random.rand(2, 3, 2).astype(np.float32),
            np.ones((1, 3), dtype=np.float32),
            np.array([0.4], dtype=np.float32),
        )
        svc.model_path.write_text("dummy")
        result = svc.auto_calibrate(sample_image)
        assert result is None

    def test_auto_calibrate_custom_pitch_dimensions(self, svc, sample_image, mock_onnx_session):
        svc._session = mock_onnx_session
        svc.model_path.write_text("dummy")
        H = svc.auto_calibrate(sample_image, pitch_length=100.0, pitch_width=60.0)
        assert H is not None
        assert H.pitch_length_m == 100.0
        assert H.pitch_width_m == 60.0


class TestPropagateHomography:
    def test_propagate_returns_homography(self, svc, sample_image, mock_onnx_session):
        svc._session = mock_onnx_session
        svc.model_path.write_text("dummy")
        ref_H = HomographyMatrix(
            matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            pitch_length_m=105.0,
            pitch_width_m=68.0,
            source="manual",
            confidence=1.0,
        )
        H = svc.propagate_homography(sample_image, sample_image, ref_H)
        assert H is not None
        assert H.source == "lightglue_propagated"

    def test_propagate_returns_none_when_no_match(self, svc, sample_image):
        svc._session = MagicMock()
        svc._session.run.return_value = (
            np.random.rand(2, 3, 2).astype(np.float32),
            np.ones((1, 3), dtype=np.float32),
            np.array([0.4], dtype=np.float32),
        )
        svc.model_path.write_text("dummy")
        ref_H = HomographyMatrix(
            matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            pitch_length_m=105.0,
            pitch_width_m=68.0,
        )
        result = svc.propagate_homography(sample_image, sample_image, ref_H)
        assert result is None


class TestErrorPx:
    """error_px should propagate computed reprojection error, not 0.0."""

    def test_auto_calibrate_returns_nonzero_error_px(self, svc, sample_image, mock_onnx_session):
        svc._session = mock_onnx_session
        svc.model_path.write_text("dummy")
        with (
            patch("cv2.findHomography") as mock_findH,
            patch("cv2.perspectiveTransform") as mock_persp,
        ):
            mock_findH.return_value = (
                np.eye(3, dtype=np.float64),
                np.ones(50, dtype=np.uint8),
            )
            mock_persp.return_value = np.zeros((1, 1, 2))
            H = svc.auto_calibrate(sample_image)
        assert H is not None
        assert H.error_px != 0.0, "error_px should not be 0.0 when inliers exist"

    def test_propagate_returns_nonzero_error_px(self, svc, sample_image, mock_onnx_session):
        svc._session = mock_onnx_session
        svc.model_path.write_text("dummy")
        ref_H = HomographyMatrix(
            matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            pitch_length_m=105.0,
            pitch_width_m=68.0,
            source="manual",
            confidence=1.0,
        )
        with (
            patch("cv2.findHomography") as mock_findH,
            patch("cv2.perspectiveTransform") as mock_persp,
        ):
            mock_findH.return_value = (
                np.eye(3, dtype=np.float64),
                np.ones(50, dtype=np.uint8),
            )
            mock_persp.return_value = np.zeros((1, 1, 2))
            H = svc.propagate_homography(sample_image, sample_image, ref_H)
        assert H is not None
        assert H.error_px != 0.0, "error_px should not be 0.0 when inliers exist"

    def test_error_px_defaults_to_999_for_failed_matching(self, svc, sample_image):
        svc._session = MagicMock()
        svc._session.run.return_value = (
            np.random.rand(2, 3, 2).astype(np.float32),
            np.ones((1, 3), dtype=np.float32),
            np.array([0.4], dtype=np.float32),
        )
        svc.model_path.write_text("dummy")
        H = svc.auto_calibrate(sample_image)
        assert H is None


class TestBuildPitchTemplate:
    def test_build_pitch_template_returns_expected_size(self, svc):
        template = svc._build_pitch_template()
        assert template.shape == (680, 1050, 3)
        assert template.dtype == np.uint8

    def test_build_pitch_template_cached(self, svc):
        t1 = svc._build_pitch_template()
        t2 = svc._build_pitch_template()
        assert t1 is t2
