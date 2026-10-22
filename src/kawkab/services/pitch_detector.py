"""CV-based pitch / line detection for auto-calibration.

Given a video frame, attempts to detect the football pitch lines and
return candidate homography calibration points. Uses classical CV
(line detection via Hough transform + vanishing-point analysis) so
it works without GPU and without neural networks.

Output: 4 corner points in image coordinates that the user can fine-
tune via drag handles in the calibration UI. The user is always
in the loop — this only provides an initial guess.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CalibrationGuess:
    """Initial calibration guess from the pitch detector."""

    image_width: int
    image_height: int
    corners: dict[str, tuple[float, float]]
    confidence: float
    vanishing_points: list[tuple[float, float]]
    detected_lines: int
    notes: list[str]


class PitchDetector:
    """Detect pitch lines and guess the four corner calibration points.

    The class never raises. When OpenCV or numpy are unavailable, or
    the frame is unsuitable, it returns a default centered guess with
    a confidence of 0 — the UI then knows to ask the user to calibrate
    manually.

    Args:
        min_line_length: Minimum Hough line length in pixels.
        max_line_gap: Maximum gap between line segments to merge.
        canny_low: Lower Canny edge threshold.
        canny_high: Upper Canny edge threshold.
    """

    def __init__(
        self,
        min_line_length: int = 80,
        max_line_gap: int = 12,
        canny_low: int = 50,
        canny_high: int = 150,
    ) -> None:
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.canny_low = canny_low
        self.canny_high = canny_high
        self._available = self._check_opencv()

    @property
    def available(self) -> bool:
        return self._available

    def _check_opencv(self) -> bool:
        try:
            import cv2  # noqa: F401

            return True
        except ImportError:
            return False

    def detect(self, frame: Any) -> CalibrationGuess:
        """Detect pitch lines and propose corner calibration points.

        Args:
            frame: BGR image (numpy array) or anything cv2 can read.
        """
        notes: list[str] = []
        if not self._available:
            return self._default_guess(0, 0, notes=["opencv not installed"])
        try:
            import cv2
            import numpy as np

            if isinstance(frame, (bytes, bytearray)):
                arr = np.frombuffer(frame, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            else:
                img = frame
            if img is None or img.size == 0:
                return self._default_guess(0, 0, notes=["empty frame"])
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=math.pi / 360,
                threshold=80,
                minLineLength=self.min_line_length,
                maxLineGap=self.max_line_gap,
            )
            if lines is None or len(lines) == 0:
                return self._default_guess(w, h, notes=["no lines detected"])
            h_lines, v_lines = self._classify_lines(lines, w, h)
            vps = self._estimate_vanishing_points(h_lines, v_lines, w, h)
            corners = self._corners_from_lines(h_lines, v_lines, w, h, vps)
            confidence = self._score_confidence(len(h_lines), len(v_lines))
            notes.append(f"detected {len(h_lines)} horizontal, {len(v_lines)} vertical lines")
            return CalibrationGuess(
                image_width=w,
                image_height=h,
                corners=corners,
                confidence=round(confidence, 2),
                vanishing_points=vps,
                detected_lines=len(h_lines) + len(v_lines),
                notes=notes,
            )
        except Exception as e:
            logger.warning("Pitch detection failed: %s", e)
            return self._default_guess(0, 0, notes=[f"error: {e}"])

    def _default_guess(self, w: int, h: int, notes: list[str] | None = None) -> CalibrationGuess:
        if w == 0 or h == 0:
            return CalibrationGuess(
                image_width=0,
                image_height=0,
                corners={},
                confidence=0.0,
                vanishing_points=[],
                detected_lines=0,
                notes=notes or ["no input"],
            )
        inset_x = w * 0.1
        inset_y = h * 0.15
        return CalibrationGuess(
            image_width=w,
            image_height=h,
            corners={
                "tl": (inset_x, inset_y),
                "tr": (w - inset_x, inset_y),
                "bl": (inset_x, h - inset_y),
                "br": (w - inset_x, h - inset_y),
            },
            confidence=0.0,
            vanishing_points=[],
            detected_lines=0,
            notes=notes or ["using default centered guess"],
        )

    def _classify_lines(self, lines: Any, w: int, h: int) -> tuple[list[Any], list[Any]]:
        h_lines: list[Any] = []
        v_lines: list[Any] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = 90.0 if x2 == x1 else abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            if angle < 20 or angle > 160:
                h_lines.append((x1, y1, x2, y2))
            elif 70 < angle < 110:
                v_lines.append((x1, y1, x2, y2))
        return h_lines, v_lines

    def _estimate_vanishing_points(
        self,
        h_lines: list[tuple[int, int, int, int]],
        v_lines: list[tuple[int, int, int, int]],
        w: int,
        h: int,
    ) -> list[tuple[float, float]]:
        vps: list[tuple[float, float]] = []
        if len(h_lines) >= 2:
            mids = [((x1 + x2) / 2.0, (y1 + y2) / 2.0) for x1, y1, x2, y2 in h_lines]
            avg_x = sum(m[0] for m in mids) / len(mids)
            avg_y = sum(m[1] for m in mids) / len(mids)
            vps.append((avg_x, avg_y))
        if len(v_lines) >= 2:
            mids = [((x1 + x2) / 2.0, (y1 + y2) / 2.0) for x1, y1, x2, y2 in v_lines]
            avg_x = sum(m[0] for m in mids) / len(mids)
            avg_y = sum(m[1] for m in mids) / len(mids)
            vps.append((avg_x, avg_y))
        if not vps:
            vps.append((w / 2.0, h / 2.0))
        return vps

    def _corners_from_lines(
        self,
        h_lines: list[tuple[int, int, int, int]],
        v_lines: list[tuple[int, int, int, int]],
        w: int,
        h: int,
        vps: list[tuple[float, float]],
    ) -> dict[str, tuple[float, float]]:
        if not h_lines or not v_lines:
            return {
                "tl": (w * 0.1, h * 0.15),
                "tr": (w * 0.9, h * 0.15),
                "bl": (w * 0.1, h * 0.85),
                "br": (w * 0.9, h * 0.85),
            }
        h_avg = [((x1 + x2) / 2.0, (y1 + y2) / 2.0) for x1, y1, x2, y2 in h_lines]
        v_avg = [((x1 + x2) / 2.0, (y1 + y2) / 2.0) for x1, y1, x2, y2 in v_lines]
        h_top = min(h_avg, key=lambda p: p[1])
        h_bot = max(h_avg, key=lambda p: p[1])
        v_left = min(v_avg, key=lambda p: p[0])
        v_right = max(v_avg, key=lambda p: p[0])
        return {
            "tl": (v_left[0], h_top[1]),
            "tr": (v_right[0], h_top[1]),
            "bl": (v_left[0], h_bot[1]),
            "br": (v_right[0], h_bot[1]),
        }

    @staticmethod
    def _score_confidence(n_h: int, n_v: int) -> float:
        if n_h == 0 and n_v == 0:
            return 0.0
        score = min(1.0, (n_h + n_v) / 20.0)
        if n_h >= 2 and n_v >= 2:
            score = min(1.0, score + 0.2)
        return score
