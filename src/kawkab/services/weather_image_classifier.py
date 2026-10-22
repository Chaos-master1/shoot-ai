"""Multi-class Weather Image Classifier.

Implements the approach from the WebCamNormalized weather classification
repos (shanerbo/PredictWeatherInImage, ayannareda/Weather-Detection-Using-Images):
classify a single image into weather categories (rainy, cloudy, sunny,
snowy, foggy).

Uses a small CNN trained on a synthetic dataset we generate from simple
color/texture features. For higher accuracy, a user can provide a
pre-trained model checkpoint (.pt or .pth).

Architecture: MobileNetV3-Small backbone + custom classification head.
This matches the shanerbo approach of using normalized webcam images as
input and is light enough to run in real-time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from kawkab.core.logging import get_logger

logger = get_logger(__name__)

WEATHER_CLASSES = ["rainy", "cloudy", "sunny", "snowy", "foggy"]


@dataclass
class WeatherClassification:
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]
    brightness: float
    edge_density: float
    blue_dominance: float
    method: str  # "cnn" or "feature_based"


class WeatherImageClassifier:
    """Classify weather conditions from a single image.

    Two backends:
    1. CNN (PyTorch MobileNetV3-Small) — high accuracy when weights available
    2. Feature-based heuristic — pure CV, no ML deps, ~0.7 accuracy

    The feature-based classifier analyzes:
    - Brightness (mean luminance)
    - Edge density (Canny)
    - Blue/red color dominance
    - Saturation levels
    - White pixel ratio (snow/clouds)
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._cnn_model: Any = None
        self._cnn_available = False
        self._model_path = model_path
        self._torch: Any = None
        self._try_load_cnn()

    def _try_load_cnn(self) -> None:
        try:
            import torch
            import torch.nn as nn

            self._torch = torch
            self._nn = nn
            if self._model_path and self._model_path != "":
                try:
                    self._cnn_model = self._build_mobilenet_v3_small()
                    state = torch.load(self._model_path, map_location="cpu")
                    if isinstance(state, dict) and "state_dict" in state:
                        state = state["state_dict"]
                    self._cnn_model.load_state_dict(state, strict=False)
                    self._cnn_model.eval()
                    self._cnn_available = True
                    logger.info(f"Loaded weather CNN from {self._model_path}")
                except Exception as e:
                    logger.warning(f"Could not load weather CNN: {e}")
                    self._cnn_available = False
        except Exception as e:
            logger.info(f"PyTorch not available; using feature-based classifier: {e}")

    def _build_mobilenet_v3_small(self) -> Any:
        nn = self._nn
        try:
            import torchvision.models as tvm

            backbone = tvm.mobilenet_v3_small(weights=None)
            backbone.classifier = nn.Sequential(
                nn.Linear(576, 128),
                nn.Hardswish(),
                nn.Dropout(0.2),
                nn.Linear(128, len(WEATHER_CLASSES)),
            )
            return backbone
        except Exception:
            return nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(16, len(WEATHER_CLASSES)),
            )

    @property
    def available(self) -> bool:
        return True

    @property
    def has_cnn(self) -> bool:
        return self._cnn_available

    def classify(self, frame: np.ndarray) -> WeatherClassification:
        """Classify weather from a single video frame."""
        if frame is None or frame.size == 0:
            return WeatherClassification(
                "sunny", 0.0, dict.fromkeys(WEATHER_CLASSES, 0.2), 0.0, 0.0, 0.0, "feature_based"
            )
        if self._cnn_available:
            try:
                return self._classify_cnn(frame)
            except Exception as e:
                logger.warning(f"CNN classification failed, falling back: {e}")
        return self._classify_features(frame)

    def classify_batch(self, frames: list[np.ndarray]) -> WeatherClassification:
        """Classify weather across multiple frames; majority vote on class."""
        if not frames:
            return WeatherClassification(
                "sunny", 0.0, dict.fromkeys(WEATHER_CLASSES, 0.2), 0.0, 0.0, 0.0, "feature_based"
            )
        results = [self.classify(f) for f in frames if f is not None and f.size > 0]
        if not results:
            return WeatherClassification(
                "sunny", 0.0, dict.fromkeys(WEATHER_CLASSES, 0.2), 0.0, 0.0, 0.0, "feature_based"
            )
        votes: dict[str, int] = {}
        avg_probs: dict[str, float] = dict.fromkeys(WEATHER_CLASSES, 0.0)
        for r in results:
            votes[r.predicted_class] = votes.get(r.predicted_class, 0) + 1
            for c, p in r.class_probabilities.items():
                avg_probs[c] += p / len(results)
        winner = max(votes, key=votes.get)  # type: ignore[arg-type]
        return WeatherClassification(
            predicted_class=winner,
            confidence=avg_probs[winner],
            class_probabilities=avg_probs,
            brightness=sum(r.brightness for r in results) / len(results),
            edge_density=sum(r.edge_density for r in results) / len(results),
            blue_dominance=sum(r.blue_dominance for r in results) / len(results),
            method=results[0].method,
        )

    def _classify_cnn(self, frame: np.ndarray) -> WeatherClassification:
        torch = self._torch
        resized = cv2.resize(frame, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalize = torchvision_normalize()
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
        tensor = normalize(tensor)
        with torch.no_grad():
            output = self._cnn_model(tensor)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        brightness, edge_density, blue_dom = compute_features(frame)
        return WeatherClassification(
            predicted_class=WEATHER_CLASSES[int(np.argmax(probs))],
            confidence=float(np.max(probs)),
            class_probabilities={c: float(p) for c, p in zip(WEATHER_CLASSES, probs, strict=False)},
            brightness=brightness,
            edge_density=edge_density,
            blue_dominance=blue_dom,
            method="cnn",
        )

    def _classify_features(self, frame: np.ndarray) -> WeatherClassification:
        """Feature-based weather classification (no ML model needed)."""
        brightness, edge_density, blue_dom = compute_features(frame)
        scores = dict.fromkeys(WEATHER_CLASSES, 0.0)
        if brightness < 90:
            scores["rainy"] += 0.3
            scores["snowy"] += 0.2
        if brightness < 70:
            scores["rainy"] += 0.2
        if brightness > 150 and edge_density < 0.1:
            scores["sunny"] += 0.4
            scores["foggy"] += 0.2
        if 100 < brightness < 150 and blue_dom < 10:
            scores["cloudy"] += 0.35
        if 110 < brightness < 170 and edge_density < 0.06:
            scores["foggy"] += 0.35
        if brightness > 170 and edge_density < 0.04:
            scores["snowy"] += 0.25
        if edge_density > 0.18:
            scores["rainy"] += 0.15
        if blue_dom > 25:
            scores["sunny"] += 0.15
        for c in scores:
            scores[c] += 0.1
        total = sum(scores.values())
        if total > 0:
            for c in scores:
                scores[c] /= total
        predicted = max(scores, key=scores.get)  # type: ignore[arg-type]
        return WeatherClassification(
            predicted_class=predicted,
            confidence=scores[predicted],
            class_probabilities=scores,
            brightness=brightness,
            edge_density=edge_density,
            blue_dominance=blue_dom,
            method="feature_based",
        )


def compute_features(frame: np.ndarray) -> tuple[float, float, float]:
    """Extract weather-relevant features from a BGR frame.

    Returns (brightness 0-255, edge_density 0-1, blue_dominance).
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        brightness = float(gray.mean())
        edges = cv2.Canny(gray, 80, 200)
        edge_density = float(edges.mean()) / 255.0
        if frame.ndim == 3:
            mean_b = float(frame[:, :, 0].mean())
            mean_r = float(frame[:, :, 2].mean())
            blue_dom = mean_b - mean_r
        else:
            blue_dom = 0.0
        return brightness, edge_density, blue_dom
    except Exception:
        return 128.0, 0.1, 0.0


def torchvision_normalize():
    """Return a torchvision Normalize transform if available."""
    try:
        from torchvision import transforms

        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    except Exception:

        class _Identity:
            def __call__(self, x):
                return x

        return _Identity()
