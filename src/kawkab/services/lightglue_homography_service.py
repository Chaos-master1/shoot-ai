"""LightGlue-ONNX homography service.

Uses SuperPoint + LightGlue feature matching (via ONNX Runtime) to:
1. Auto-calibrate: match a video frame against a synthetic pitch template
2. Propagate: match frame N against a calibrated reference frame

Requires: onnxruntime-gpu (optional dep, extras=[lightglue])
"""

from __future__ import annotations

import logging
import math
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from kawkab.core.paths import get_paths
from kawkab.services.homography_service import HomographyMatrix

logger = logging.getLogger(__name__)

MODEL_RELEASE_URL = "https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v2.0"
MODEL_FILENAME = "superpoint_lightglue_pipeline.onnx"
MODEL_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

TEMPLATE_WIDTH = 1050
TEMPLATE_HEIGHT = 680


class LightGlueHomographyService:
    """Auto homography via SuperPoint + LightGlue feature matching (ONNX)."""

    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = model_dir or (get_paths().models / "lightglue")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._session: Any = None
        self._input_size = 1024
        self._pitch_template: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    @property
    def model_path(self) -> Path:
        return self.model_dir / MODEL_FILENAME

    @property
    def available(self) -> bool:
        return self.model_path.exists()

    def ensure_model(self) -> None:
        if not self.model_path.exists():
            url = f"{MODEL_RELEASE_URL}/{MODEL_FILENAME}"
            logger.info("Downloading LightGlue ONNX model from %s ...", url)
            urllib.request.urlretrieve(url, str(self.model_path))
            logger.info("Model downloaded to %s", self.model_path)

    def _load_session(self) -> None:
        if self._session is not None:
            return
        import onnxruntime

        path = str(self.model_path)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        opts = onnxruntime.SessionOptions()
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = onnxruntime.InferenceSession(path, opts, providers=providers)

    def _ensure_inputs(self) -> None:
        if not self.available:
            raise RuntimeError("LightGlue model not found. Call ensure_model() first.")
        self._load_session()

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Resize + pad to square, return (tensor, scale_x, scale_y)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        h, w = gray.shape[:2]
        scale = self._input_size / max(h, w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((self._input_size, self._input_size), dtype=np.float32)
        canvas[:new_h, :new_w] = resized.astype(np.float32) / 255.0
        scale_x = w / new_w if new_w else 1.0
        scale_y = h / new_h if new_h else 1.0
        return canvas[np.newaxis, np.newaxis, ...], scale_x, scale_y

    # ------------------------------------------------------------------
    # Core matching
    # ------------------------------------------------------------------

    def match(
        self, img0: np.ndarray, img1: np.ndarray, conf_threshold: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Run SuperPoint + LightGlue, return (kpts0, kpts1, confidence)."""
        self._ensure_inputs()
        inp0, sx0, sy0 = self._preprocess(img0)
        inp1, sx1, sy1 = self._preprocess(img1)
        batch = np.concatenate([inp0, inp1], axis=0)

        input_name = self._session.get_inputs()[0].name
        kpts_out, matches_out, scores_out = self._session.run(None, {input_name: batch})

        good = scores_out > conf_threshold
        matches_out = matches_out[good]
        if len(matches_out) < 4:
            return None

        kpts0_raw = kpts_out[0]
        kpts1_raw = kpts_out[1]
        idx0 = matches_out[:, 1].astype(int)
        idx1 = matches_out[:, 2].astype(int)

        kpts0 = kpts0_raw[idx0] * [sx0, sy0]
        kpts1 = kpts1_raw[idx1] * [sx1, sy1]

        conf = float(min(1.0, len(matches_out) / 100.0))
        return kpts0, kpts1, conf

    # ------------------------------------------------------------------
    # Homography computation
    # ------------------------------------------------------------------

    def compute_homography(
        self,
        img0: np.ndarray,
        img1: np.ndarray,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
    ) -> HomographyMatrix | None:
        """Match two images and return HomographyMatrix (img0 -> img1)."""
        result = self.match(img0, img1)
        if result is None:
            return None
        kpts0, kpts1, conf = result

        H, mask = cv2.findHomography(kpts0, kpts1, cv2.RANSAC, 5.0)
        if H is None or mask is None:
            return None

        inliers = int(mask.sum())
        error = float(
            np.mean(
                [
                    np.linalg.norm(
                        cv2.perspectiveTransform(
                            kpts0[mask.flatten().astype(bool)][i : i + 1],
                            H,
                        )[0, 0]
                        - kpts1[mask.flatten().astype(bool)][i]
                    )
                    for i in range(min(100, inliers))
                ]
            )
            if inliers > 0
            else 999
        )

        return HomographyMatrix(
            matrix=H.tolist(),
            pitch_length_m=pitch_length,
            pitch_width_m=pitch_width,
            source="lightglue",
            confidence=min(conf, inliers / max(len(kpts0), 1)),
            error_px=round(error, 2),
        )

    # ------------------------------------------------------------------
    # Auto calibration (video frame -> pitch template)
    # ------------------------------------------------------------------

    def _build_pitch_template(self) -> np.ndarray:
        """Render a synthetic football pitch at 1050x680."""
        if self._pitch_template is not None:
            return self._pitch_template

        w, h = TEMPLATE_WIDTH, TEMPLATE_HEIGHT
        pitch = np.full((h, w, 3), (50, 160, 50), dtype=np.uint8)
        white = (255, 255, 255)

        def line(x1, y1, x2, y2, thickness=2):
            cv2.line(pitch, (x1, y1), (x2, y2), white, thickness)

        margin = 10
        pw = w - 2 * margin
        ph = h - 2 * margin

        # Outer boundary
        cv2.rectangle(pitch, (margin, margin), (w - margin, h - margin), white, 2)

        # Halfway line
        line(w // 2, margin, w // 2, h - margin)

        # Center circle
        cx, cy = w // 2, h // 2
        cv2.circle(pitch, (cx, cy), int(ph * 0.15), white, 2)

        # Penalty areas
        pa_w = int(pw * 0.17)
        pa_h = int(ph * 0.44)
        pa_top = (h - pa_h) // 2
        # Left
        cv2.rectangle(pitch, (margin, pa_top), (margin + pa_w, pa_top + pa_h), white, 2)
        # Right
        cv2.rectangle(pitch, (w - margin - pa_w, pa_top), (w - margin, pa_top + pa_h), white, 2)

        # Goal areas (6-yard box)
        ga_w = int(pw * 0.07)
        ga_h = int(ph * 0.26)
        ga_top = (h - ga_h) // 2
        cv2.rectangle(pitch, (margin, ga_top), (margin + ga_w, ga_top + ga_h), white, 2)
        cv2.rectangle(pitch, (w - margin - ga_w, ga_top), (w - margin - 1, ga_top + ga_h), white, 2)

        # Goal posts (small white rectangles at each end)
        gp_w = 3
        gp_h = int(ph * 0.10)
        gp_top = (h - gp_h) // 2
        cv2.rectangle(pitch, (margin - gp_w, gp_top), (margin, gp_top + gp_h), white, -1)
        cv2.rectangle(pitch, (w - margin, gp_top), (w - margin + gp_w, gp_top + gp_h), white, -1)

        # Penalty spots
        ps_d = 4
        for spot_x in [margin + int(pa_w * 0.7), w - margin - int(pa_w * 0.7)]:
            cv2.circle(pitch, (spot_x, cy), ps_d, white, -1)

        # Corner arcs
        for cx2, cy2, start_a, end_a in [
            (margin, margin, 0, 90),
            (w - margin, margin, 90, 180),
            (w - margin, h - margin, 180, 270),
            (margin, h - margin, 270, 360),
        ]:
            cv2.ellipse(pitch, (cx2, cy2), (15, 15), 0, start_a, end_a, white, 2)

        self._pitch_template = pitch
        return pitch

    def auto_calibrate(
        self,
        frame: np.ndarray,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
    ) -> HomographyMatrix | None:
        """Match frame against synthetic pitch template."""
        template = self._build_pitch_template()
        result = self.match(frame, template)
        if result is None:
            return None
        kpts_frame, kpts_template, conf = result

        template_corners = np.array(
            [
                [0, 0],
                [TEMPLATE_WIDTH, 0],
                [TEMPLATE_WIDTH, TEMPLATE_HEIGHT],
                [0, TEMPLATE_HEIGHT],
            ],
            dtype=np.float32,
        )
        pitch_corners = np.array(
            [
                [0, 0],
                [pitch_length, 0],
                [pitch_length, pitch_width],
                [0, pitch_width],
            ],
            dtype=np.float32,
        )

        H_template_to_pitch = cv2.getPerspectiveTransform(template_corners, pitch_corners)

        H_frame_to_template, mask = cv2.findHomography(kpts_frame, kpts_template, cv2.RANSAC, 5.0)
        if H_frame_to_template is None:
            return None

        H_frame_to_pitch = H_template_to_pitch @ H_frame_to_template
        inliers = int(mask.sum())

        # Compute reprojection error in pixels
        if inliers >= 4:
            try:
                mask_bool = mask.flatten().astype(bool)
                pts_frame = kpts_frame[mask_bool][:100]
                pts_template = kpts_template[mask_bool][:100]
                projected = cv2.perspectiveTransform(
                    pts_frame.reshape(-1, 1, 2).astype(np.float32),
                    H_frame_to_template,
                )
                errors = np.linalg.norm(projected.reshape(-1, 2) - pts_template, axis=1)
                error = float(np.mean(errors))
            except Exception:
                error = 999.0
        else:
            error = 999.0
        if math.isnan(error) or error < 0:
            error = 999.0

        return HomographyMatrix(
            matrix=H_frame_to_pitch.tolist(),
            pitch_length_m=pitch_length,
            pitch_width_m=pitch_width,
            source="lightglue_auto",
            confidence=min(conf, inliers / max(len(kpts_frame), 1)),
            error_px=round(error, 2),
        )

    # ------------------------------------------------------------------
    # Frame-to-frame propagation
    # ------------------------------------------------------------------

    def propagate_homography(
        self,
        frame: np.ndarray,
        ref_frame: np.ndarray,
        ref_homography: HomographyMatrix,
    ) -> HomographyMatrix | None:
        """Match current frame against a calibrated reference frame."""
        result = self.match(frame, ref_frame)
        if result is None:
            return None
        kpts_frame, kpts_ref, conf = result

        H_frame_to_ref, mask = cv2.findHomography(kpts_frame, kpts_ref, cv2.RANSAC, 5.0)
        if H_frame_to_ref is None:
            return None

        H_ref = np.array(ref_homography.matrix, dtype=np.float64)
        H_frame_to_pitch = H_ref @ H_frame_to_ref
        inliers = int(mask.sum())

        # Compute reprojection error in pixels
        if inliers >= 4:
            try:
                mask_bool = mask.flatten().astype(bool)
                pts_frame = kpts_frame[mask_bool][:100]
                pts_ref = kpts_ref[mask_bool][:100]
                projected = cv2.perspectiveTransform(
                    pts_frame.reshape(-1, 1, 2).astype(np.float32),
                    H_frame_to_ref,
                )
                errors = np.linalg.norm(projected.reshape(-1, 2) - pts_ref, axis=1)
                error = float(np.mean(errors))
            except Exception:
                error = 999.0
        else:
            error = 999.0
        if math.isnan(error) or error < 0:
            error = 999.0

        return HomographyMatrix(
            matrix=H_frame_to_pitch.tolist(),
            pitch_length_m=ref_homography.pitch_length_m,
            pitch_width_m=ref_homography.pitch_width_m,
            source="lightglue_propagated",
            confidence=min(conf, inliers / max(len(kpts_frame), 1)),
            error_px=round(error, 2),
        )
