"""Dedicated ball tracking module with CNN + Kalman filter.

Architecture:
- Mode 1: YOLO-based ball detection (ultralytics YOLO11, fine-tuned for ball)
- Mode 2: HSV fallback when YOLO unavailable or no ball detected
- 8-state Kalman filter with ballistic motion model (constant acceleration)
- Bounce detection via vy sign reversal with coefficient of restitution
- Confidence calibration from YOLO confidence scores
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("ball_tracker")

# ── Constants ─────────────────────────────────────────────────────
BALL_MIN_RADIUS = 4
BALL_MAX_RADIUS = 12
BALL_CIRCULARITY_MIN = 0.6
WHITE_LOWER = (0, 0, 200)
WHITE_UPPER = (180, 30, 255)
DARK_LOWER = (0, 0, 0)
DARK_UPPER = (180, 255, 50)
MISSED_FRAME_LIMIT = 30
CONFIDENCE_VISIBLE = 0.9
CONFIDENCE_PREDICTED = 0.3
CONFIDENCE_LOW = 0.3
CONFIDENCE_MEDIUM = 0.5
CONFIDENCE_HIGH = 0.7

YOLO_CONF_THRESHOLD = 0.15
YOLO_IOU_THRESHOLD = 0.5

COEFFICIENT_OF_RESTITUTION = 0.7


@dataclass
class BallDetection:
    frame: int
    timestamp: float
    x: float
    y: float
    conf: float
    is_prediction: bool = False
    radius: float = 0.0


def _detect_gpu() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _calibrate_confidence(raw_conf: float) -> float:
    """Map YOLO confidence (0-1) to calibrated ball confidence."""
    if raw_conf >= CONFIDENCE_HIGH:
        return 0.90 + (raw_conf - CONFIDENCE_HIGH) * 0.3
    elif raw_conf >= CONFIDENCE_MEDIUM:
        return 0.50 + (raw_conf - CONFIDENCE_MEDIUM) * 2.0
    elif raw_conf >= CONFIDENCE_LOW:
        return 0.15 + (raw_conf - CONFIDENCE_LOW) * 3.5
    return raw_conf * 0.5


def _load_yolo_model(model_path: str | Path | None = None) -> Any | None:
    """Lazy-load YOLO model for ball detection, GPU if available."""
    try:
        from ultralytics import YOLO

        path = model_path or "yolo11n.pt"
        model = YOLO(str(path))
        if _detect_gpu():
            try:
                model.to("cuda")
                logger.debug("Ball YOLO model on GPU")
            except Exception:
                pass
        logger.info(f"Ball YOLO model loaded: {path}")
        return model
    except Exception as e:
        logger.warning(f"Failed to load YOLO for ball detection: {e}")
        return None


def _detect_yolo(model: Any, frame: np.ndarray) -> dict | None:
    """Run YOLO inference for ball detection (COCO class 32 = sports ball).

    Returns dict with x, y, radius, confidence, raw_conf or None.
    """
    if model is None:
        return None
    try:
        results = model(
            frame,
            conf=YOLO_CONF_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            classes=[32],
            imgsz=1280,
            verbose=False,
        )
        if not results or len(results) == 0:
            return None
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None
        best_idx = int(boxes.conf.argmax())
        raw_conf = float(boxes.conf[best_idx])
        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        radius = max(x2 - x1, y2 - y1) / 2.0
        return {
            "x": cx,
            "y": cy,
            "radius": radius,
            "confidence": _calibrate_confidence(raw_conf),
            "raw_conf": raw_conf,
        }
    except Exception as e:
        logger.debug(f"YOLO ball detection failed: {e}")
        return None


def _find_hsv_candidates(frame: np.ndarray) -> list[dict]:
    """HSV-based ball candidate detection (original fallback)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    candidates = []
    for lower, upper, label in [
        (WHITE_LOWER, WHITE_UPPER, "white"),
        (DARK_LOWER, DARK_UPPER, "dark"),
    ]:
        mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
        # None-kernel works at runtime but is mistyped in cv2 stubs; an
        # explicit empty kernel is equivalent (no erosion/dilation) and typed.
        empty_kernel = np.zeros((1, 1), dtype=np.uint8)
        mask = cv2.erode(mask, empty_kernel, iterations=1)
        mask = cv2.dilate(mask, empty_kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            if radius < BALL_MIN_RADIUS or radius > BALL_MAX_RADIUS:
                continue
            area = cv2.contourArea(cnt)
            if area == 0:
                continue
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            if circularity < BALL_CIRCULARITY_MIN:
                continue
            candidates.append(
                {"x": x, "y": y, "radius": radius, "circularity": circularity, "label": label}
            )
    candidates.sort(key=lambda c: c["circularity"], reverse=True)
    return candidates[:3]


class BallTracker:
    """Ball tracker with CNN (YOLO) + HSV fallback + 8-state ballistic Kalman.

    API (backward compatible):
        update(frame, frame_number, timestamp) -> BallDetection | None
        predict() -> BallDetection | None
        reset()
        get_trail(max_age=5.0) -> list[BallDetection]
    """

    def __init__(self, fps: float = 24.0, model_path: str | Path | None = None):
        self.fps = fps
        self.dt = 1.0 / max(fps, 1)
        # Effective dt between consecutive update() calls. CVService only
        # calls update() on detection frames (every frame_skip-th frame),
        # so the *real* time step is frame_skip/fps. The Kalman filter's
        # velocities/accelerations (and bounce detection, which thresholds
        # on |vy| > 2.0 px/frame-step) are all per-dt; a dt that is
        # frame_skip times too small understates ball speed by the same
        # factor and silently disables bounce detection at frame_skip >= 2.
        self._dt_override: float | None = None

        # 8-state Kalman: [x, y, r, vx, vy, vr, ax, ay]
        self.kalman = cv2.KalmanFilter(8, 3)
        self._init_kalman(self.dt)

        self.initialized = False
        self.missed_frames = 0
        self.last_frame = -1
        self.last_timestamp = 0.0
        self.confidence = 0.0
        self.trail: list[BallDetection] = []
        self._last_vy = 0.0

        # YOLO model (shared singleton across instances)
        self._model = None
        self._model_path = model_path
        self._use_yolo = True

        # Mode tracking (for diagnostics)
        self._current_mode: str = "unknown"

    def _init_kalman(self, dt: float):
        """8-state constant-acceleration ballistic motion model.

        State:  [x, y, r, vx, vy, vr, ax, ay]
        Meas:   [x, y, r]
        """
        dt2 = 0.5 * dt * dt
        self.kalman.transitionMatrix = np.array(
            [
                [1, 0, 0, dt, 0, 0, dt2, 0],
                [0, 1, 0, 0, dt, 0, 0, dt2],
                [0, 0, 1, 0, 0, dt, 0, 0],
                [0, 0, 0, 1, 0, 0, dt, 0],
                [0, 0, 0, 0, 1, 0, 0, dt],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        self.kalman.measurementMatrix = np.hstack([np.eye(3, 3), np.zeros((3, 5))]).astype(
            np.float32
        )

        dt_scale = max(dt / (1.0 / 24.0), 0.01)
        self.kalman.processNoiseCov = np.diag(
            np.array(
                [
                    1e-3,
                    1e-3,
                    1e-3,
                    1e-2,
                    1e-2,
                    1e-2,
                    1e-1,
                    1e-1,
                ],
                dtype=np.float32,
            )
            * dt_scale
        )

        self.kalman.measurementNoiseCov = np.diag(
            np.array([1e-1, 1e-1, 1e-1], dtype=np.float32) / dt_scale
        )
        self.kalman.errorCovPost = np.eye(8, dtype=np.float32)

    def _detect_bounce(self, vy: float) -> bool:
        """Detect bounce: vy sign reversal from positive to negative."""
        bounced = self._last_vy > 2.0 and vy < -2.0
        self._last_vy = vy
        return bounced

    def _apply_bounce(self):
        """Invert vy with coefficient of restitution."""
        state = self.kalman.statePost.ravel()
        state[4] *= -COEFFICIENT_OF_RESTITUTION
        state[7] *= -COEFFICIENT_OF_RESTITUTION
        self.kalman.statePost = state.reshape(-1, 1)
        logger.debug(f"Ball bounce, vy={state[4]:.1f}")

    def _measurement_update(self, best: dict, frame_number: int, timestamp: float) -> BallDetection:
        self.missed_frames = 0
        meas = np.array([[best["x"]], [best["y"]], [best["radius"]]], dtype=np.float32)

        if not self.initialized:
            # Column vector (8, 1) -- a 1-D assignment happened to work on
            # OpenCV 4 but raises a gemm shape assertion on OpenCV 5+.
            self.kalman.statePost = np.array(
                [[best["x"]], [best["y"]], [best["radius"]], [0], [0], [0], [0], [0]],
                dtype=np.float32,
            )
            self.kalman.errorCovPost = np.eye(8, dtype=np.float32)
            self.initialized = True
            self.confidence = best["confidence"]
        else:
            self.kalman.predict()
            vy_before = float(self.kalman.statePost.ravel()[4])
            self.kalman.correct(meas)
            vy_after = float(self.kalman.statePost.ravel()[4])
            if self._detect_bounce(vy_after) or (vy_before > 2.0 and vy_after < -2.0):
                self._apply_bounce()
            self.confidence = best["confidence"]

        state = self.kalman.statePost.ravel()
        return BallDetection(
            frame=frame_number,
            timestamp=timestamp,
            x=float(state[0]),
            y=float(state[1]),
            conf=self.confidence,
            is_prediction=False,
            radius=float(state[2]),
        )

    def _prediction_update(self, frame_number: int, timestamp: float) -> BallDetection | None:
        if not self.initialized or self.missed_frames >= MISSED_FRAME_LIMIT:
            self.initialized = False
            self.confidence = 0.0
            self._current_mode = "unknown"
            return None

        self.missed_frames += 1
        self._current_mode = "prediction"
        pred = self.kalman.predict().ravel()
        vy = float(pred[4])
        if self._detect_bounce(vy):
            self._apply_bounce()
            pred = self.kalman.statePost.ravel()

        decay = max(0.1, 1.0 - self.missed_frames / MISSED_FRAME_LIMIT)
        self.confidence = CONFIDENCE_PREDICTED * decay

        return BallDetection(
            frame=frame_number,
            timestamp=timestamp,
            x=float(pred[0]),
            y=float(pred[1]),
            conf=self.confidence,
            is_prediction=True,
            radius=float(pred[2]),
        )

    def set_effective_dt(self, dt: float) -> None:
        """Set the real time step between consecutive update() calls.

        Use when update() is not called on every video frame (e.g. CVService
        calls it every frame_skip-th frame). Re-initializes the Kalman
        transition model with the correct dt so velocities and bounce
        detection are estimated in consistent units. Safe to call before
        or during tracking; existing trail/history stays valid.
        """
        if dt <= 0:
            return
        self._dt_override = dt
        self._init_kalman(dt)

    def _current_dt(self) -> float:
        return self._dt_override if self._dt_override is not None else self.dt

    def update(
        self, frame: np.ndarray, frame_number: int, timestamp: float
    ) -> BallDetection | None:
        """Process a new frame: YOLO -> HSV fallback -> Kalman prediction.

        Returns BallDetection or None if tracking has been lost.
        """
        # Lazy-load YOLO on first call
        if self._model is None and self._use_yolo:
            self._model = _load_yolo_model(self._model_path)
            if self._model is None:
                self._use_yolo = False

        # Mode 1: YOLO
        best = _detect_yolo(self._model, frame) if self._use_yolo else None

        # Mode 2: HSV fallback
        if best is None:
            candidates = _find_hsv_candidates(frame)
            if candidates:
                # HSV conf must stay strictly BELOW the 0.3 recording gate
                # (CVService records ball detections at conf > 0.3). The old
                # formula (0.3 + circularity * 0.4) was >= 0.3 by
                # construction, so every white/dark circular blob -- pitch
                # lines, boots, socks -- passed the gate while marginal true
                # YOLO detections were dropped. Only high-circularity,
                # motion-consistent candidates are considered at all.
                def _hsv_conf(cand: dict) -> float:
                    circularity = cand.get("circularity", 0.5)
                    return min(0.1 + (circularity - BALL_CIRCULARITY_MIN) * 0.5, 0.29)

                if self.initialized:
                    # Motion-consistency gate: only accept a candidate within
                    # a loose radius of the Kalman prediction (~5x typical
                    # per-step ball travel). A white line blob far from the
                    # predicted ball position is noise, not the ball.
                    pred = self.kalman.statePost.ravel()
                    max_dist = max(40.0, 400.0 * self._current_dt())
                    chosen = next(
                        (
                            c
                            for c in candidates
                            if math.hypot(c["x"] - float(pred[0]), c["y"] - float(pred[1]))
                            <= max_dist
                        ),
                        None,
                    )
                else:
                    # No track yet: take the best (most circular) candidate.
                    chosen = candidates[0]

                if chosen is not None:
                    best = {
                        "x": chosen["x"],
                        "y": chosen["y"],
                        "radius": chosen["radius"],
                        "confidence": _hsv_conf(chosen),
                    }

        if best is not None:
            self._current_mode = "yolo" if self._use_yolo and best.get("raw_conf", 0) > 0 else "hsv"
            det: BallDetection | None = self._measurement_update(best, frame_number, timestamp)
        else:
            # Mode 3: prediction only
            det = self._prediction_update(frame_number, timestamp)

        self.last_frame = frame_number
        self.last_timestamp = timestamp
        if det is not None:
            self.trail.append(det)
            if len(self.trail) > 1000:
                self.trail = self.trail[-500:]
        return det

    def predict(self) -> BallDetection | None:
        """Advance Kalman without a measurement frame."""
        self._current_mode = "prediction"
        return self._prediction_update(
            self.last_frame + 1, self.last_timestamp + self._current_dt()
        )

    def reset(self):
        self.initialized = False
        self.missed_frames = 0
        self.confidence = 0.0
        self.trail.clear()
        self._last_vy = 0.0
        self._current_mode = "unknown"

    def get_trail(self, max_age: float = 5.0) -> list[BallDetection]:
        if not self.trail:
            return []
        cutoff = self.last_timestamp - max_age
        return [b for b in self.trail if b.timestamp >= cutoff]
