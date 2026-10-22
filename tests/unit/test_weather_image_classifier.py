"""Dedicated tests for WeatherImageClassifier — initialization, classify, batch, feature extraction."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()
_mod = load_service_module("weather_clf_test", "weather_image_classifier.py")
WeatherImageClassifier = _mod.WeatherImageClassifier
WeatherClassification = _mod.WeatherClassification
WEATHER_CLASSES = _mod.WEATHER_CLASSES
compute_features = _mod.compute_features

import numpy as np
import pytest


@pytest.fixture
def svc() -> WeatherImageClassifier:
    return WeatherImageClassifier()


class TestClassifierInit:
    def test_available(self, svc: WeatherImageClassifier) -> None:
        assert svc.available

    def test_no_cnn_by_default(self, svc: WeatherImageClassifier) -> None:
        assert not svc.has_cnn

    def test_cnn_model_none_without_path(self, svc: WeatherImageClassifier) -> None:
        assert svc._cnn_model is None

    def test_weathe_classes_defined(self) -> None:
        assert "sunny" in WEATHER_CLASSES
        assert "rainy" in WEATHER_CLASSES
        assert "cloudy" in WEATHER_CLASSES
        assert "snowy" in WEATHER_CLASSES
        assert "foggy" in WEATHER_CLASSES
        assert len(WEATHER_CLASSES) == 5


class TestClassify:
    def test_classify_returns_weather_classification(self, svc: WeatherImageClassifier) -> None:
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = svc.classify(frame)
        assert isinstance(result, WeatherClassification)

    def test_classify_has_predicted_class(self, svc: WeatherImageClassifier) -> None:
        frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        result = svc.classify(frame)
        assert result.predicted_class in WEATHER_CLASSES

    def test_classify_method_feature_based(self, svc: WeatherImageClassifier) -> None:
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        result = svc.classify(frame)
        assert result.method == "feature_based"

    def test_classify_bright_frame(self, svc: WeatherImageClassifier) -> None:
        frame = np.full((480, 640, 3), 220, dtype=np.uint8)
        result = svc.classify(frame)
        assert result.brightness > 150

    def test_classify_dark_frame(self, svc: WeatherImageClassifier) -> None:
        frame = np.full((480, 640, 3), 30, dtype=np.uint8)
        result = svc.classify(frame)
        assert result.brightness < 90

    def test_classify_probabilities_sum_to_one(self, svc: WeatherImageClassifier) -> None:
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        result = svc.classify(frame)
        total = sum(result.class_probabilities.values())
        assert abs(total - 1.0) < 0.01

    def test_classify_confidence_in_range(self, svc: WeatherImageClassifier) -> None:
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        result = svc.classify(frame)
        assert 0.0 <= result.confidence <= 1.0

    def test_classify_edge_density_non_negative(self, svc: WeatherImageClassifier) -> None:
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = svc.classify(frame)
        assert result.edge_density >= 0.0

    def test_classify_none_frame(self, svc: WeatherImageClassifier) -> None:
        result = svc.classify(None)
        assert result.predicted_class == "sunny"
        assert result.confidence == 0.0
        assert result.method == "feature_based"

    def test_classify_empty_frame(self, svc: WeatherImageClassifier) -> None:
        frame = np.zeros((0, 0, 3), dtype=np.uint8)
        result = svc.classify(frame)
        assert result.predicted_class == "sunny"

    def test_classify_blue_sky(self, svc: WeatherImageClassifier) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = 180  # blue channel high
        frame[:, :, 2] = 80  # red channel low
        result = svc.classify(frame)
        assert result.blue_dominance > 0


class TestClassifyBatch:
    def test_batch_empty(self, svc: WeatherImageClassifier) -> None:
        result = svc.classify_batch([])
        assert result.predicted_class == "sunny"
        assert result.confidence == 0.0

    def test_batch_single(self, svc: WeatherImageClassifier) -> None:
        frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        result = svc.classify_batch([frame])
        assert result.predicted_class in WEATHER_CLASSES
        assert result.method == "feature_based"

    def test_batch_multiple(self, svc: WeatherImageClassifier) -> None:
        frames = [
            np.full((480, 640, 3), 200, dtype=np.uint8),
            np.full((480, 640, 3), 180, dtype=np.uint8),
        ]
        result = svc.classify_batch(frames)
        assert result.predicted_class in WEATHER_CLASSES
        assert result.brightness > 0

    def test_batch_all_none(self, svc: WeatherImageClassifier) -> None:
        result = svc.classify_batch([np.zeros((0, 0, 3), dtype=np.uint8)])
        assert result.predicted_class == "sunny"

    def test_batch_mixed_valid_invalid(self, svc: WeatherImageClassifier) -> None:
        frames = [
            np.full((480, 640, 3), 200, dtype=np.uint8),
            np.zeros((0, 0, 3), dtype=np.uint8),
        ]
        result = svc.classify_batch(frames)
        assert result.predicted_class in WEATHER_CLASSES


class TestComputeFeatures:
    def test_returns_tuple(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        brightness, edges, blue = compute_features(frame)
        assert isinstance(brightness, float)
        assert isinstance(edges, float)
        assert isinstance(blue, float)

    def test_white_frame(self) -> None:
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        brightness, edges, blue = compute_features(frame)
        assert brightness > 240
        assert edges < 0.05

    def test_black_frame(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        brightness, _, _ = compute_features(frame)
        assert brightness < 5

    def test_blue_dominance_positive(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = 200  # blue
        frame[:, :, 2] = 50  # red
        _, _, blue = compute_features(frame)
        assert blue > 0

    def test_red_dominance_negative_blue(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = 50  # blue
        frame[:, :, 2] = 200  # red
        _, _, blue = compute_features(frame)
        assert blue < 0

    def test_grayscale_frame(self) -> None:
        frame = np.full((100, 100), 128, dtype=np.uint8)
        brightness, edges, blue = compute_features(frame)
        assert blue == 0.0
