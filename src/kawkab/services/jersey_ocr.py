"""Jersey number OCR for player identification in broadcast football.

Reads jersey numbers from player crop images using EasyOCR.
Maps numbers to track IDs and optionally to known squad rosters.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from kawkab.core.logging import get_logger

logger = get_logger(__name__)

_ocr_reader = None


def _get_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr

        _ocr_reader = easyocr.Reader(["en"], gpu=True)
    return _ocr_reader


class JerseyOCR:
    """Read jersey numbers from player crop images using EasyOCR."""

    def __init__(self, squad_roster: dict[str, list[dict]] | None = None):
        """Initialize with optional squad roster.

        Args:
            squad_roster: {team_name: [{name, number}, ...]}
        """
        self.reader = _get_reader()
        self.squad_roster = squad_roster or {}
        self._name_by_number: dict[str, dict[int, str]] = {}
        for team, players in self.squad_roster.items():
            self._name_by_number[team] = {p["number"]: p["name"] for p in players}

    def read_number_from_crop(self, crop: np.ndarray) -> int | None:
        """Read jersey number from a player crop image.

        Returns the number as int, or None if no valid number detected.
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 10:
            return None

        results = self.reader.readtext(crop, allowlist="0123456789", paragraph=False)
        digits = []
        for bbox, text, conf in results:
            text = text.strip()
            if not text.isdigit():
                continue
            if conf < 0.3:
                continue
            digits.append((int(text), conf, bbox))

        if not digits:
            return None

        best = max(digits, key=lambda x: x[1])
        return best[0]

    def read_number_from_torso(
        self, frame: np.ndarray, bbox: tuple[float, float, float, float]
    ) -> int | None:
        """Extract torso from frame and read jersey number.

        Args:
            frame: Full video frame (BGR).
            bbox: (x1, y1, x2, y2) bounding box.

        Returns:
            Jersey number or None.
        """
        torso = self._extract_jersey_region(frame, bbox)
        if torso is None:
            return None
        return self.read_number_from_crop(torso)

    def identify_track(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
        team_name: str | None = None,
    ) -> dict[str, Any]:
        """Identify a player by reading their jersey number.

        Args:
            frame: Full video frame.
            bbox: Player bounding box.
            team_name: Optional team name for name lookup.

        Returns:
            {number, name, confidence}
        """
        number = self.read_number_from_torso(frame, bbox)
        result: dict[str, Any] = {"number": number, "name": None, "confidence": 0.0}

        if number is None:
            return result

        if team_name and team_name in self._name_by_number:
            name = self._name_by_number[team_name].get(number)
            result["name"] = name

        return result

    @staticmethod
    def _extract_jersey_region(
        frame: np.ndarray, bbox: tuple[float, float, float, float]
    ) -> np.ndarray | None:
        """Crop the jersey number region from a player bbox.

        Strategy: take the upper 30-70% of the bbox (chest/back area).
        For broadcast football, numbers are typically on the back
        (visible when player faces away) or front (when facing toward).
        """
        x1, y1, x2, y2 = map(int, bbox)
        h = y2 - y1
        if h < 30:
            return None
        jersey_top = y1 + int(h * 0.25)
        jersey_bot = y1 + int(h * 0.65)
        jersey_left = max(0, x1)
        jersey_right = min(frame.shape[1], x2)
        if jersey_bot - jersey_top < 15 or jersey_right - jersey_left < 10:
            return None
        crop = frame[jersey_top:jersey_bot, jersey_left:jersey_right]
        if crop.size == 0:
            return None
        if max(crop.shape[0], crop.shape[1]) < 60:
            crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        return crop

    @staticmethod
    def preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
        """Enhance crop for better OCR: grayscale, contrast, sharpen."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
