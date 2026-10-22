"""Tests for JerseyNumberService."""

from unittest.mock import MagicMock

import numpy as np
from conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.services.jersey_service import JerseyNumberService


class TestJerseyNumberService:
    def _patch_cv2_for_ocr(self, monkeypatch):
        """Mock cv2 functions used by _preprocess_for_ocr and _detect_ocr."""
        monkeypatch.setattr("kawkab.services.jersey_service.cv2.cvtColor", lambda img, code: img)
        monkeypatch.setattr("kawkab.services.jersey_service.cv2.COLOR_BGR2LAB", 1)
        monkeypatch.setattr("kawkab.services.jersey_service.cv2.COLOR_LAB2BGR", 2)
        monkeypatch.setattr("kawkab.services.jersey_service.cv2.COLOR_BGR2GRAY", 6)
        monkeypatch.setattr("kawkab.services.jersey_service.cv2.COLOR_GRAY2BGR", 4)
        monkeypatch.setattr(
            "kawkab.services.jersey_service.cv2.bilateralFilter", lambda img, *a, **kw: img
        )
        monkeypatch.setattr("kawkab.services.jersey_service.cv2.resize", lambda img, *a, **kw: img)
        # Mock createCLAHE
        fake_clahe = MagicMock()
        fake_clahe.apply = lambda img: img
        monkeypatch.setattr(
            "kawkab.services.jersey_service.cv2.createCLAHE", lambda **kw: fake_clahe
        )
        # Mock split/merge
        monkeypatch.setattr("kawkab.services.jersey_service.cv2.split", lambda img: [img, img, img])
        monkeypatch.setattr(
            "kawkab.services.jersey_service.cv2.merge", lambda channels: channels[0]
        )

    def test_detect_ocr_success(self, monkeypatch):
        self._patch_cv2_for_ocr(monkeypatch)
        fake_reader = MagicMock()
        fake_reader.readtext.return_value = [([[0, 0], [10, 0], [10, 10], [0, 10]], "7", 0.85)]
        service = JerseyNumberService(reader="ocr")
        service._ocr_reader = fake_reader
        result = service.detect(np.zeros((30, 30, 3), dtype=np.uint8))
        assert result["jersey_number"] == 7
        assert result["source"] == "ocr"

    def test_detect_cnn_falls_to_pixel(self, monkeypatch):
        monkeypatch.setattr(
            "kawkab.services.jersey_service.cv2.cvtColor",
            lambda img, code: np.mean(img, axis=2).astype(np.uint8),
        )
        monkeypatch.setattr("kawkab.services.jersey_service.cv2.COLOR_BGR2GRAY", 6)
        service = JerseyNumberService(reader="cnn")
        result = service.detect(np.ones((28, 28, 3), dtype=np.uint8) * 255)
        assert isinstance(result, dict)
        assert result["source"] == "pixel"

    def test_detect_fallback_to_pixel(self, monkeypatch):
        monkeypatch.setattr(
            "kawkab.services.jersey_service.cv2.cvtColor",
            lambda img, code: np.mean(img, axis=2).astype(np.uint8),
        )
        monkeypatch.setattr("kawkab.services.jersey_service.cv2.COLOR_BGR2GRAY", 6)
        monkeypatch.setattr("kawkab.services.jersey_service.cv2.resize", lambda img, *a, **kw: img)
        service = JerseyNumberService(reader="ocr")
        result = service.detect(np.ones((30, 30, 3), dtype=np.uint8) * 255)
        assert isinstance(result, dict)
