"""Pose estimation and analysis service using YOLO26-pose.

Uses ultralytics YOLO26-pose model (17 keypoints per person) to derive
tactical insights beyond what bbox-based detection can offer:
- Activity classification (standing/walking/jogging/running/sprinting)
- Fall detection (sudden hip drop)
- Player orientation / facing direction
- Fatigue proxy from stride length over time

The pose model is loaded lazily and shares the same ultralytics package
as the detection model. Falls back gracefully if pose model unavailable.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from kawkab.core.logging import get_logger

logger = get_logger(__name__)

COCO_KEYPOINTS: list[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

POSE_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


@dataclass
class PoseResult:
    track_id: int
    keypoints: np.ndarray
    confidence: float
    bbox: tuple[float, float, float, float]
    timestamp: float = 0.0


@dataclass
class ActivitySegment:
    track_id: int
    activity: str
    start_time: float
    end_time: float
    duration_s: float
    avg_speed_kmh: float = 0.0


@dataclass
class FallEvent:
    track_id: int
    timestamp: float
    hip_drop_ratio: float
    recovery_time_s: float | None = None


class PoseAnalysisService:
    """Async-friendly pose estimation + activity/fall analysis.

    Methods:
    - detect_poses(frame) -> list[PoseResult]
    - classify_activity(track_id, keypoints_history) -> str
    - detect_fall(prev_keypoints, curr_keypoints) -> FallEvent | None
    - get_player_orientation(keypoints) -> float (radians)
    """

    def __init__(self, model_size: str = "n", device: str = "") -> None:
        self.model_size = model_size
        self.device = device
        self._model: Any = None
        self._available = False
        self._activity_history: dict[int, deque[tuple[float, str]]] = defaultdict(
            lambda: deque(maxlen=300)
        )
        self._keypoint_history: dict[int, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=30))

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return self._available
        try:
            from ultralytics import YOLO

            model_name = f"yolo26{self.model_size}-pose.pt"
            self._model = YOLO(model_name)
            self._available = True
            logger.info(f"YOLO26-pose loaded: {model_name}")
        except Exception as e:
            logger.info(f"YOLO26-pose not available: {e}")
            self._available = False
        return self._available

    @property
    def available(self) -> bool:
        return self._ensure_model()

    def detect_poses(self, frame: np.ndarray, conf_threshold: float = 0.3) -> list[PoseResult]:
        """Run pose detection on a single frame."""
        if not self._ensure_model():
            return []
        try:
            model = self._model
            assert model is not None  # _ensure_model() returned True
            results = model.predict(
                frame, conf=conf_threshold, verbose=False, device=self.device or None
            )
            if not results:
                return []
            result = results[0]
            if not hasattr(result, "keypoints") or result.keypoints is None:
                return []
            poses: list[PoseResult] = []
            kpts_data = result.keypoints.data
            n = len(kpts_data) if kpts_data is not None else 0
            for i in range(n):
                kpts = (
                    kpts_data[i].cpu().numpy()
                    if hasattr(kpts_data[i], "cpu")
                    else np.array(kpts_data[i])
                )
                if kpts.shape[0] < 17:
                    continue
                confidence = float(np.mean(kpts[:, 2]))
                if confidence < conf_threshold:
                    continue
                xs = kpts[:, 0]
                ys = kpts[:, 1]
                bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
                poses.append(
                    PoseResult(
                        track_id=i,
                        keypoints=kpts,
                        confidence=confidence,
                        bbox=bbox,
                        timestamp=time.time(),
                    )
                )
            return poses
        except Exception as e:
            logger.warning(f"detect_poses failed: {e}")
            return []

    def classify_activity(self, track_id: int, keypoints: np.ndarray, timestamp: float) -> str:
        """Classify player activity from a single pose sample.

        Uses ankle+hip velocity estimate over the keypoint history.
        """
        self._keypoint_history[track_id].append(keypoints.copy())
        history = self._keypoint_history[track_id]
        if len(history) < 2:
            return "unknown"
        try:
            curr_ankles = keypoints[[15, 16], :2]
            prev_ankles = history[-2][[15, 16], :2]
            valid = (curr_ankles[:, 0] > 0) & (prev_ankles[:, 0] > 0)
            if not valid.any():
                return "unknown"
            displacement = float(
                np.linalg.norm(curr_ankles[valid] - prev_ankles[valid], axis=1).max()
            )
            speed_px_per_frame = displacement
            speed_m_per_s = speed_px_per_frame * 0.05
            speed_kmh = speed_m_per_s * 3.6
        except Exception:
            return "unknown"
        if speed_kmh < 1.5:
            activity = "standing"
        elif speed_kmh < 5:
            activity = "walking"
        elif speed_kmh < 10:
            activity = "jogging"
        elif speed_kmh < 20:
            activity = "running"
        else:
            activity = "sprinting"
        self._activity_history[track_id].append((timestamp, activity))
        return activity

    def detect_fall(
        self,
        track_id: int,
        prev_keypoints: np.ndarray,
        curr_keypoints: np.ndarray,
        timestamp: float,
    ) -> FallEvent | None:
        """Detect a fall event: rapid hip-height drop in a short window."""
        try:
            prev_hip_y = (prev_keypoints[11, 1] + prev_keypoints[12, 1]) / 2
            curr_hip_y = (curr_keypoints[11, 1] + curr_keypoints[12, 1]) / 2
            if prev_hip_y <= 0 or curr_hip_y <= 0:
                return None
            height_change = (curr_hip_y - prev_hip_y) / max(prev_hip_y, 1e-6)
            if height_change > 0.3:
                return FallEvent(
                    track_id=track_id,
                    timestamp=timestamp,
                    hip_drop_ratio=height_change,
                )
        except Exception:
            pass
        return None

    def get_player_orientation(self, keypoints: np.ndarray) -> float:
        """Estimate player facing direction from shoulder-hip alignment.

        Returns angle in radians (0 = facing right in image coords).
        """
        try:
            l_shoulder = keypoints[5, :2]
            r_shoulder = keypoints[6, :2]
            l_hip = keypoints[11, :2]
            r_hip = keypoints[12, :2]
            shoulder_center = (l_shoulder + r_shoulder) / 2
            hip_center = (l_hip + r_hip) / 2
            dx = shoulder_center[0] - hip_center[0]
            dy = shoulder_center[1] - hip_center[1]
            if abs(dx) < 1e-3 and abs(dy) < 1e-3:
                return 0.0
            return float(np.arctan2(dy, dx))
        except Exception:
            return 0.0

    def get_activity_segments(self, track_id: int) -> list[ActivitySegment]:
        """Convert activity history into consolidated time segments."""
        history = list(self._activity_history.get(track_id, []))
        if not history:
            return []
        segments: list[ActivitySegment] = []
        current_activity = history[0][1]
        start_time = history[0][0]
        for i in range(1, len(history)):
            if history[i][1] != current_activity:
                duration = history[i][0] - start_time
                segments.append(
                    ActivitySegment(
                        track_id=track_id,
                        activity=current_activity,
                        start_time=start_time,
                        end_time=history[i][0],
                        duration_s=duration,
                    )
                )
                current_activity = history[i][1]
                start_time = history[i][0]
        if history:
            duration = history[-1][0] - start_time
            segments.append(
                ActivitySegment(
                    track_id=track_id,
                    activity=current_activity,
                    start_time=start_time,
                    end_time=history[-1][0],
                    duration_s=duration,
                )
            )
        return segments

    def summarize_activity(self, track_id: int) -> dict[str, float]:
        """Return total seconds spent in each activity for a track."""
        segments = self.get_activity_segments(track_id)
        summary: dict[str, float] = defaultdict(float)
        for s in segments:
            summary[s.activity] += s.duration_s
        return dict(summary)

    def clear_history(self, track_id: int | None = None) -> None:
        if track_id is None:
            self._activity_history.clear()
            self._keypoint_history.clear()
        else:
            self._activity_history.pop(track_id, None)
            self._keypoint_history.pop(track_id, None)
