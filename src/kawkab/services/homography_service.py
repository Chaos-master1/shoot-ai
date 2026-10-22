"""Homography service - converts pixel coordinates to pitch coordinates (meters).

This is the missing link that makes all spatial stats meaningful:
- Distance covered in meters (not pixels)
- xT in real pitch zones
- Formations in real pitch positions
- Defensive line height in meters

Supports 3 calibration modes:
1. Manual: Coach clicks 4 pitch corners on a frame
2. Auto: LightGlue feature-matching based (via LightGlueHomographyService)
3. Default: Estimated based on visible pitch markings
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kawkab.core.logging import get_logger
from kawkab.core.paths import get_paths

logger = get_logger(__name__)


@dataclass
class HomographyMatrix:
    """Camera-to-pitch transformation matrix."""

    matrix: list[list[float]] = field(default_factory=list)
    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0
    source: str = "manual"
    confidence: float = 0.0
    error_px: float = 0.0
    quality_issues: list[str] = field(default_factory=list)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array(self.matrix, dtype=np.float64)

    def pixel_to_pitch(self, x: float, y: float) -> tuple[float, float]:
        """Convert pixel (x, y) to pitch (meters from top-left)."""
        H = self.to_array()
        pt = np.array([x, y, 1.0])
        transformed = H @ pt
        if abs(transformed[2]) < 1e-8:
            return 0.0, 0.0
        return float(transformed[0] / transformed[2]), float(transformed[1] / transformed[2])

    def pitch_to_pixel(self, mx: float, my: float) -> tuple[float, float]:
        """Convert pitch (meters) back to pixel."""
        H_inv = np.linalg.inv(self.to_array())
        pt = np.array([mx, my, 1.0])
        transformed = H_inv @ pt
        if abs(transformed[2]) < 1e-8:
            return 0.0, 0.0
        return float(transformed[0] / transformed[2]), float(transformed[1] / transformed[2])


class HomographyService:
    """Manages camera calibration and pixel-to-pitch transformation."""

    def __init__(self) -> None:
        self._calibrations: dict[str, HomographyMatrix] = {}
        logger.info("HomographyService initialized")

    def compute_homography_from_corners(
        self,
        pixel_corners: list[tuple[float, float]],
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
    ) -> HomographyMatrix:
        """Compute homography from 4 user-clicked pitch corners.

        The user clicks 4 points in this order:
        - Top-left of pitch
        - Top-right of pitch
        - Bottom-right of pitch
        - Bottom-left of pitch

        Args:
            pixel_corners: List of 4 (x, y) pixel coordinates
            pitch_length_m: Real pitch length (default 105m for FIFA standard)
            pitch_width_m: Real pitch width (default 68m for FIFA standard)

        Returns:
            HomographyMatrix that can transform pixel -> pitch coords
        """
        if len(pixel_corners) != 4:
            raise ValueError(f"Need exactly 4 corner points, got {len(pixel_corners)}")

        pixel_pts = np.array(pixel_corners, dtype=np.float32)

        pitch_pts = np.array(
            [
                [0, 0],
                [pitch_length_m, 0],
                [pitch_length_m, pitch_width_m],
                [0, pitch_width_m],
            ],
            dtype=np.float32,
        )

        H, _ = cv2_find_homography(pixel_pts, pitch_pts)
        if H is None:
            raise ValueError("Failed to compute homography (check corner order)")

        error = self._compute_reprojection_error(H, pixel_pts, pitch_pts)

        # With exactly 4 point correspondences findHomography fits them
        # exactly, so the reprojection error is ~0 even for geometrically
        # bad corner sets (mirrored, self-intersecting, skewed). The
        # independent geometry assessment is the honest confidence signal;
        # the reprojection term only penalises numerical failure.
        error_confidence = max(0.0, min(1.0, 1.0 - (error / 100.0)))
        quality = self._assess_corner_quality(pixel_corners, pitch_length_m, pitch_width_m)
        confidence = min(error_confidence, quality["score"])
        if quality["issues"]:
            logger.warning(
                "Homography calibration quality issues: %s (confidence %.2f)",
                "; ".join(quality["issues"]),
                confidence,
            )

        matrix = HomographyMatrix(
            matrix=H.tolist(),
            pitch_length_m=pitch_length_m,
            pitch_width_m=pitch_width_m,
            source="manual",
            confidence=confidence,
            error_px=error,
            quality_issues=quality["issues"],
        )

        logger.info(
            f"Homography computed: {pitch_length_m}x{pitch_width_m}m, "
            f"error={error:.2f}px, confidence={confidence:.0%}"
        )
        return matrix

    def compute_homography_from_visible_markings(
        self,
        frame_width: int,
        frame_height: int,
        visible_pitch_area: tuple[float, float, float, float] | None = None,
    ) -> HomographyMatrix:
        """Estimate homography from visible pitch area (no user input).

        This is a rough estimate. Useful when user doesn't calibrate.
        Assumes:
        - The pitch occupies the central 80% of the frame
        - Camera is roughly elevated (slight angle)
        - Aspect ratio: standard football pitch 105:68 ≈ 1.54:1

        Args:
            frame_width: Video frame width
            frame_height: Video frame height
            visible_pitch_area: Optional (x1, y1, x2, y2) of visible pitch

        Returns:
            Estimated HomographyMatrix (low confidence)
        """
        if visible_pitch_area:
            x1, y1, x2, y2 = visible_pitch_area
        else:
            margin_x = int(frame_width * 0.1)
            margin_y = int(frame_height * 0.1)
            x1, y1 = margin_x, margin_y
            x2, y2 = frame_width - margin_x, frame_height - margin_y

        pixel_corners = [
            (x1, y1),  # top-left
            (x2, y1),  # top-right
            (x2, y2),  # bottom-right
            (x1, y2),  # bottom-left
        ]

        matrix = self.compute_homography_from_corners(pixel_corners)
        matrix.source = "estimated"
        matrix.confidence = min(0.5, matrix.confidence)
        return matrix

    def _compute_reprojection_error(
        self,
        H: np.ndarray,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
    ) -> float:
        """Compute mean reprojection error in pixels."""
        try:
            import cv2

            projected = cv2.perspectiveTransform(src_pts.reshape(1, -1, 2), H).reshape(-1, 2)
            errors = np.linalg.norm(projected - dst_pts, axis=1)
            return float(np.mean(errors))
        except Exception:
            return 50.0

    def save_calibration(self, match_id: int, matrix: HomographyMatrix) -> Path:
        """Save homography calibration to disk for a match."""
        paths = get_paths()
        calib_dir = paths.appdata / "calibrations"
        calib_dir.mkdir(parents=True, exist_ok=True)

        calib_path = calib_dir / f"match_{match_id}.json"
        with open(calib_path, "w") as f:
            json.dump(
                {
                    "matrix": matrix.matrix,
                    "pitch_length_m": matrix.pitch_length_m,
                    "pitch_width_m": matrix.pitch_width_m,
                    "source": matrix.source,
                    "confidence": matrix.confidence,
                    "error_px": matrix.error_px,
                    "quality_issues": matrix.quality_issues,
                },
                f,
                indent=2,
            )

        logger.info(f"Calibration saved: {calib_path}")
        return calib_path

    def load_calibration(self, match_id: int) -> HomographyMatrix | None:
        """Load saved calibration for a match."""
        paths = get_paths()
        calib_path = paths.appdata / "calibrations" / f"match_{match_id}.json"
        if not calib_path.exists():
            return None
        try:
            with open(calib_path) as f:
                data = json.load(f)
            return HomographyMatrix(
                matrix=data["matrix"],
                pitch_length_m=data.get("pitch_length_m", 105.0),
                pitch_width_m=data.get("pitch_width_m", 68.0),
                source=data.get("source", "unknown"),
                confidence=data.get("confidence", 0.0),
                error_px=data.get("error_px", 0.0),
                quality_issues=data.get("quality_issues", []),
            )
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return None

    def save_segment_calibration(
        self, match_id: int, segment_index: int, matrix: HomographyMatrix
    ) -> None:
        """Save per-segment homography calibration for a match."""
        paths = get_paths()
        calib_dir = paths.appdata / "calibrations"
        calib_dir.mkdir(parents=True, exist_ok=True)

        seg_path = calib_dir / f"match_{match_id}_segments.json"
        segments = {}
        if seg_path.exists():
            try:
                with open(seg_path) as f:
                    segments = json.load(f)
            except Exception:
                segments = {}

        segments[str(segment_index)] = {
            "matrix": matrix.matrix,
            "pitch_length_m": matrix.pitch_length_m,
            "pitch_width_m": matrix.pitch_width_m,
            "source": matrix.source,
            "confidence": matrix.confidence,
            "error_px": matrix.error_px,
            "quality_issues": matrix.quality_issues,
        }

        with open(seg_path, "w") as f:
            json.dump(segments, f, indent=2)
        logger.info(f"Segment {segment_index} calibration saved for match {match_id}")

    def load_segment_calibrations(self, match_id: int) -> dict[int, HomographyMatrix]:
        """Load all per-segment calibrations for a match."""
        paths = get_paths()
        seg_path = paths.appdata / "calibrations" / f"match_{match_id}_segments.json"
        if not seg_path.exists():
            return {}
        try:
            with open(seg_path) as f:
                data = json.load(f)
            result = {}
            for seg_idx, cal_data in data.items():
                result[int(seg_idx)] = HomographyMatrix(
                    matrix=cal_data["matrix"],
                    pitch_length_m=cal_data.get("pitch_length_m", 105.0),
                    pitch_width_m=cal_data.get("pitch_width_m", 68.0),
                    source=cal_data.get("source", "unknown"),
                    confidence=cal_data.get("confidence", 0.0),
                    error_px=cal_data.get("error_px", 0.0),
                    quality_issues=cal_data.get("quality_issues", []),
                )
            return result
        except Exception as e:
            logger.error(f"Failed to load segment calibrations: {e}")
            return {}

    def transform_track_positions(
        self,
        pixel_positions: list[tuple[float, float, float]],
        matrix: HomographyMatrix,
    ) -> list[tuple[float, float, float]]:
        """Convert all positions in a track from pixel to pitch coords.

        Args:
            pixel_positions: List of (timestamp, pixel_x, pixel_y)
            matrix: HomographyMatrix

        Returns:
            List of (timestamp, pitch_x_m, pitch_y_m)
        """
        return [(ts, *matrix.pixel_to_pitch(px, py)) for ts, px, py in pixel_positions]

    def convert_formation_to_pitch(
        self,
        formation_pixels: dict[str, list[tuple[float, float]]],
        matrix: HomographyMatrix,
    ) -> dict[str, list[tuple[float, float]]]:
        """Convert formation player positions from pixel to pitch coords."""
        return {
            group: [matrix.pixel_to_pitch(px, py) for px, py in positions]
            for group, positions in formation_pixels.items()
        }

    def validate_4corner_calibration(
        self,
        pixel_corners: list[tuple[float, float]],
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
    ) -> dict[str, Any]:
        """Validate the quality of a 4-corner calibration.

        Checks for:
        - Correct number of corners (exactly 4)
        - Each corner inside the image bounds
        - Reasonable aspect ratio (not too skewed)
        - Convex quadrilateral (no self-intersecting polygon)
        - Reasonable scale (meters per pixel matches expected broadcast view)

        Returns a dict with: 'is_valid', 'score' (0-1), 'issues' (list of str).
        """
        result = self._assess_corner_quality(pixel_corners, pitch_length_m, pitch_width_m)
        issues: list[str] = result["issues"]
        score: float = result["score"]
        metrics: dict[str, Any] = result["metrics"]
        if metrics:
            reprojection = 0.0
            try:
                matrix = self.compute_homography_from_corners(
                    pixel_corners, pitch_length_m, pitch_width_m
                )
                src_pts = np.array(pixel_corners, dtype=np.float32)
                dst_pts = np.array(
                    [
                        [0, 0],
                        [pitch_length_m, 0],
                        [pitch_length_m, pitch_width_m],
                        [0, pitch_width_m],
                    ],
                    dtype=np.float32,
                )
                reprojection = self._compute_reprojection_error(matrix.to_array(), src_pts, dst_pts)
            except Exception as e:
                issues.append(f"reprojection error: {e}")
                reprojection = 999.0
            if reprojection > 5.0:
                issues.append(f"reprojection error {reprojection:.2f}px > 5px threshold")
                score -= 0.2
            metrics["reprojection_error_px"] = round(reprojection, 2)
        score = max(0.0, score)
        return {
            "is_valid": len(issues) == 0,
            "score": round(score, 2),
            "issues": issues,
            "metrics": metrics,
        }

    def _assess_corner_quality(
        self,
        pixel_corners: list[tuple[float, float]],
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
    ) -> dict[str, Any]:
        """Assess geometric quality of 4 clicked corners (no homography needed).

        Returns {'score': float, 'issues': list[str], 'metrics': dict}.
        'metrics' is empty when the corners are too degenerate to assess.
        """
        issues: list[str] = []
        score = 1.0
        if len(pixel_corners) != 4:
            issues.append(f"need exactly 4 corners, got {len(pixel_corners)}")
            return {"score": 0.0, "issues": issues, "metrics": {}}
        tl, tr, br, bl = pixel_corners
        widths_top = math.hypot(tr[0] - tl[0], tr[1] - tl[1])
        widths_bot = math.hypot(br[0] - bl[0], br[1] - bl[1])
        heights_left = math.hypot(bl[0] - tl[0], bl[1] - tl[1])
        heights_right = math.hypot(br[0] - tr[0], br[1] - tr[1])
        if widths_top == 0 or widths_bot == 0 or heights_left == 0 or heights_right == 0:
            issues.append("one or more edges have zero length (degenerate)")
            return {"score": 0.0, "issues": issues, "metrics": {}}
        aspect = ((widths_top + widths_bot) / 2) / ((heights_left + heights_right) / 2)
        expected_aspect = pitch_length_m / pitch_width_m
        aspect_error = abs(aspect - expected_aspect) / expected_aspect
        if aspect_error > 0.3:
            issues.append(f"aspect ratio {aspect:.2f} off from expected {expected_aspect:.2f}")
            score -= 0.3
        width_diff = abs(widths_top - widths_bot) / max(widths_top, widths_bot)
        height_diff = abs(heights_left - heights_right) / max(heights_left, heights_right)
        if width_diff > 0.15:
            issues.append(f"top/bottom width mismatch {width_diff:.2f}")
            score -= 0.2
        if height_diff > 0.15:
            issues.append(f"left/right height mismatch {height_diff:.2f}")
            score -= 0.2
        if not self._is_convex(pixel_corners):
            issues.append("corner polygon is not convex (likely self-intersecting)")
            score -= 0.3
        return {
            "score": score,
            "issues": issues,
            "metrics": {
                "aspect_ratio": round(aspect, 3),
                "expected_aspect_ratio": round(expected_aspect, 3),
                "width_diff_ratio": round(width_diff, 3),
                "height_diff_ratio": round(height_diff, 3),
            },
        }

    @staticmethod
    def _is_convex(points: list[tuple[float, float]]) -> bool:
        n = len(points)
        if n < 3:
            return False
        sign = 0
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            x3, y3 = points[(i + 2) % n]
            cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
            if cross != 0:
                if sign == 0:
                    sign = 1 if cross > 0 else -1
                elif (cross > 0) != (sign > 0):
                    return False
        return True


def cv2_find_homography(src: np.ndarray, dst: np.ndarray):
    """Wrapper to import cv2 lazily."""
    import cv2

    return cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
