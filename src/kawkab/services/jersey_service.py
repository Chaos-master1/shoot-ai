"""Jersey Number Detection Service.

Provides a unified interface for jersey number recognition with:
- Primary: EasyOCR (current production reader)
- Fallback: Pixel-based estimation (no ML deps)
- Future: CNN-based detection (PyTorch model, not yet trained)

Architecture matches the cnn-number-detection repo for future model loading.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default CNN architecture from cnn-number-detection repo (ModelGNetDeep)
# For when a trained model is available:
#   3x Conv2D(16,32,64) → 3x MaxPool → Flatten → Dense(128) → Dense(10)
CNN_INPUT_SIZE = 28
CNN_CATEGORIES = ["-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


class JerseyNumberService:
    """Jersey number detector with multiple backends.

    Usage:
        svc = JerseyNumberService()
        result = svc.detect(torso_crop)  # returns {"number": 7, "confidence": 0.92, ...}
    """

    def __init__(self, reader: str = "auto", gpu_enabled: bool = False) -> None:
        self._reader = reader
        self._gpu = gpu_enabled
        self._ocr_reader: Any = None
        self._cnn_model: Any = None

    def detect(self, torso: np.ndarray) -> dict[str, Any]:
        """Detect jersey number from a torso crop image.

        Args:
            torso: BGR numpy array of the torso/chest region

        Returns:
            dict with keys:
                - "jersey_number": int or None
                - "confidence": float (0-1)
                - "candidates": list of (number, count) top votes
                - "source": str ("ocr", "pixel", "cnn", or "none")
        """
        if self._reader == "cnn" and self._cnn_model is not None:
            return self._detect_cnn(torso)
        if self._reader in ("auto", "ocr"):
            result = self._detect_ocr(torso)
            if result["jersey_number"] is not None and result["confidence"] > 0.3:
                return result
        # Fallback
        return self._detect_pixel(torso)

    # ------------------------------------------------------------------
    # EasyOCR backend with enhanced preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess_for_ocr(torso: np.ndarray) -> np.ndarray:
        """Enhance torso crop for better OCR on small jersey numbers.

        Steps:
          1. CLAHE contrast enhancement on luminance channel
          2. Bilateral filter for edge-preserving smooth
          3. Adaptive thresholding to isolate digits
          4. 2x upscale via INTER_CUBIC for tiny (8-20px) numbers
        """
        h, w = torso.shape[:2]
        if min(h, w) < 10:
            return torso
        lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)
        lum, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        lum = clahe.apply(lum)
        enhanced = cv2.merge([lum, a_ch, b_ch])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
        if max(h, w) < 60:
            scale = max(2.0, 60.0 / max(h, w))
            enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return enhanced

    def _detect_ocr(self, torso: np.ndarray) -> dict[str, Any]:
        """Detect jersey number using EasyOCR with enhanced preprocessing."""
        try:
            if self._ocr_reader is None:
                import easyocr

                self._ocr_reader = easyocr.Reader(["en"], gpu=self._gpu, verbose=False)
            processed = self._preprocess_for_ocr(torso)
            results = self._ocr_reader.readtext(processed, allowlist="0123456789")
            if not results:
                return {
                    "jersey_number": None,
                    "confidence": 0.0,
                    "candidates": [],
                    "source": "none",
                }

            digits = "".join(r[1] for r in results if r[2] > 0.2)
            if not digits:
                return {
                    "jersey_number": None,
                    "confidence": 0.0,
                    "candidates": [],
                    "source": "none",
                }

            try:
                num = int(digits[:2])
                if 0 <= num <= 99:
                    conf = float(np.mean([r[2] for r in results if r[2] > 0.2]))
                    return {
                        "jersey_number": num,
                        "confidence": round(conf, 3),
                        "candidates": [(num, 1)],
                        "source": "ocr",
                    }
            except ValueError:
                pass
        except ImportError:
            logger.warning("EasyOCR not installed")
        except Exception as e:
            logger.debug(f"OCR failed: {e}")

        return {"jersey_number": None, "confidence": 0.0, "candidates": [], "source": "none"}

    # ------------------------------------------------------------------
    # Pixel-based fallback backend
    # ------------------------------------------------------------------

    def _detect_pixel(self, torso: np.ndarray) -> dict[str, Any]:
        """Estimate jersey number from white pixel ratio."""
        h, w = torso.shape[:2]
        gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
        white_pixels = np.sum(gray > 200)
        ratio = white_pixels / (w * h)
        # Heuristic: map ratio to 1-99 range
        estimated = int(1 + (ratio * 100) % 99)
        return {
            "jersey_number": estimated,
            "confidence": round(float(min(float(ratio) * 2, 0.5)), 3),
            "candidates": [(estimated, 1)],
            "source": "pixel",
        }

    # ------------------------------------------------------------------
    # CNN backend (future — model not yet trained for jersey numbers)
    # ------------------------------------------------------------------

    def _detect_cnn(self, torso: np.ndarray) -> dict[str, Any]:
        """Detect jersey number using CNN classifier.

        NOTE: Requires a trained model. Currently returns None until
        training data is available. See:
        https://github.com/FabianGroeger96/cnn-number-detection

        The architecture follows ModelGNetDeep from the repo:
        3x Conv2D(16,32,64) → 3x MaxPool(3x3) → Dropout(0.25) → Flatten
        → Dense(128, relu) → Dropout(0.5) → Dense(10, softmax)
        """
        if self._cnn_model is None:
            return {"jersey_number": None, "confidence": 0.0, "candidates": [], "source": "none"}

        try:
            # Isolate digits using contour detection (cnn-number-detection style)
            digits = self._isolate_digits(torso)
            if not digits:
                return {
                    "jersey_number": None,
                    "confidence": 0.0,
                    "candidates": [],
                    "source": "none",
                }

            # Classify each digit
            import torch

            number_str = ""
            confidences = []
            for digit_patch in digits:
                # Resize to 28x28 grayscale
                patch = cv2.resize(digit_patch, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                patch = patch.astype(np.float32) / 255.0

                # Model inference
                tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    outputs = self._cnn_model(tensor)
                    probs = torch.softmax(outputs, dim=1)[0]
                    pred = int(torch.argmax(probs))
                    conf = float(probs[pred])

                if pred == 0:  # category -1 = not a number
                    continue
                number_str += str(pred - 1) if pred > 0 else ""
                confidences.append(conf)

            if not number_str:
                return {
                    "jersey_number": None,
                    "confidence": 0.0,
                    "candidates": [],
                    "source": "none",
                }

            num = int(number_str[:2])
            avg_conf = float(np.mean(confidences)) if confidences else 0.0
            return {
                "jersey_number": num,
                "confidence": round(avg_conf, 3),
                "candidates": [(num, len(confidences))],
                "source": "cnn",
            }
        except Exception as e:
            logger.debug(f"CNN detection failed: {e}")

        return {"jersey_number": None, "confidence": 0.0, "candidates": [], "source": "none"}

    # ------------------------------------------------------------------
    # Digit isolation (from cnn-number-detection Isolator pattern)
    # ------------------------------------------------------------------

    @staticmethod
    def _isolate_digits(img: np.ndarray) -> list[np.ndarray]:
        """Isolate individual digit regions using contour detection.

        Adapted from cnn-number-detection Isolator.
        Returns list of cropped digit patches.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Sobel(blurred, cv2.CV_8U, 1, 0, ksize=3)
        _, thresh = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        digits = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h < 8 or w < 4:
                continue
            if h > img.shape[0] * 0.9 or w > img.shape[1] * 0.8:
                continue
            aspect = w / max(h, 1)
            if aspect < 0.3 or aspect > 1.0:
                continue
            digit = img[y : y + h, x : x + w]
            digits.append(digit)

        # Sort left-to-right (key = leftmost contour x; the previous
        # implementation re-ran threshold+findContours twice per element).
        def _leftmost_x(patch: np.ndarray) -> float:
            thresh = cv2.threshold(
                cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY
            )[1]
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if not contours:
                return 0.0
            return float(cv2.boundingRect(contours[0])[0])

        digits.sort(key=_leftmost_x)

        return digits[:3]  # max 3 digits

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def load_cnn_model(self, model_path: str | Path) -> bool:
        """Load a trained PyTorch CNN model for digit classification.

        Expected architecture (ModelGNetDeep):
        3x Conv2D(16,32,64, kernel=3) → ReLU → MaxPool(3, stride=2)
        → Dropout(0.25) → Flatten → Dense(128, relu) → Dropout(0.5)
        → Dense(11, softmax)  # categories: -1, 0-9
        """
        try:
            import torch
            import torch.nn as nn

            class GNetDeep(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.conv1 = nn.Conv2d(1, 16, 3, padding="same")
                    self.pool1 = nn.MaxPool2d(3, stride=2)
                    self.conv2 = nn.Conv2d(16, 32, 3, padding="same")
                    self.pool2 = nn.MaxPool2d(3, stride=2)
                    self.conv3 = nn.Conv2d(32, 64, 3, padding="same")
                    self.pool3 = nn.MaxPool2d(3, stride=2)
                    self.drop1 = nn.Dropout(0.25)
                    self.fc1 = nn.Linear(64 * 3 * 3, 128)
                    self.drop2 = nn.Dropout(0.5)
                    self.fc2 = nn.Linear(128, 11)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    x = torch.relu(self.conv1(x))
                    x = self.pool1(x)
                    x = torch.relu(self.conv2(x))
                    x = self.pool2(x)
                    x = torch.relu(self.conv3(x))
                    x = self.pool3(x)
                    x = self.drop1(x)
                    x = x.view(x.size(0), -1)
                    x = torch.relu(self.fc1(x))
                    x = self.drop2(x)
                    x = self.fc2(x)
                    return x

            model = GNetDeep()
            model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
            model.eval()
            self._cnn_model = model
            logger.info(f"CNN jersey model loaded from {model_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load CNN model: {e}")
            return False
