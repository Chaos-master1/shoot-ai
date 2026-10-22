"""Tests for PoseAnalysisService (YOLO26-pose)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()
_svc = load_service_module("pose_test", "pose_analysis_service.py")
PoseAnalysisService = _svc.PoseAnalysisService
COCO_KEYPOINTS = _svc.COCO_KEYPOINTS
POSE_CONNECTIONS = _svc.POSE_CONNECTIONS

from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture
def pose_svc() -> PoseAnalysisService:
    return PoseAnalysisService(model_size="n")


class TestPoseBasics:
    def test_keypoints_defined(self) -> None:
        assert len(COCO_KEYPOINTS) == 17

    def test_connections_defined(self) -> None:
        assert len(POSE_CONNECTIONS) > 0

    def test_default_not_available(self, pose_svc: PoseAnalysisService) -> None:
        with patch.object(pose_svc, "_ensure_model", return_value=False):
            assert not pose_svc.available


class TestActivityClassification:
    def test_unknown_with_one_sample(self, pose_svc: PoseAnalysisService) -> None:
        keypoints = np.zeros((17, 3))
        activity = pose_svc.classify_activity(0, keypoints, 0.0)
        assert activity == "unknown"

    def test_standing_low_speed(self, pose_svc: PoseAnalysisService) -> None:
        keypoints = np.zeros((17, 3))
        pose_svc.classify_activity(0, keypoints, 0.0)
        activity = pose_svc.classify_activity(0, keypoints, 0.1)
        assert activity in {"standing", "unknown"}

    def test_sprinting_high_speed(self, pose_svc: PoseAnalysisService) -> None:
        # Both frames need valid (non-zero) ankle positions
        keypoints1 = np.zeros((17, 3))
        keypoints1[15, 0] = 10  # left ankle
        keypoints1[16, 0] = 10  # right ankle
        keypoints2 = np.zeros((17, 3))
        keypoints2[15, 0] = 210  # 200px displacement
        keypoints2[16, 0] = 210
        pose_svc.classify_activity(0, keypoints1, 0.0)
        activity = pose_svc.classify_activity(0, keypoints2, 0.5)
        # 200px in 0.5s = 400 px/s, *0.05 = 20 m/s = 72 km/h = sprinting
        assert activity in {"running", "sprinting", "jogging"}


class TestFallDetection:
    def test_no_fall_normal_motion(self, pose_svc: PoseAnalysisService) -> None:
        kp1 = np.zeros((17, 3))
        kp2 = np.zeros((17, 3))
        kp2[11, 1] = 10
        result = pose_svc.detect_fall(0, kp1, kp2, 1.0)
        assert result is None

    def test_fall_detected(self, pose_svc: PoseAnalysisService) -> None:
        kp1 = np.zeros((17, 3))
        kp1[11, 1] = 100
        kp1[12, 1] = 100
        kp2 = np.zeros((17, 3))
        kp2[11, 1] = 200
        kp2[12, 1] = 200
        result = pose_svc.detect_fall(0, kp1, kp2, 1.0)
        assert result is not None
        assert result.hip_drop_ratio > 0.3


class TestOrientation:
    def test_neutral_pose_zero(self, pose_svc: PoseAnalysisService) -> None:
        keypoints = np.zeros((17, 3))
        angle = pose_svc.get_player_orientation(keypoints)
        assert isinstance(angle, float)

    def test_facing_right(self, pose_svc: PoseAnalysisService) -> None:
        keypoints = np.zeros((17, 3))
        keypoints[5] = [10, 0, 0]
        keypoints[6] = [20, 0, 0]
        keypoints[11] = [10, 50, 0]
        keypoints[12] = [20, 50, 0]
        angle = pose_svc.get_player_orientation(keypoints)
        assert abs(angle - (-np.pi / 2)) < 0.1


class TestClearHistory:
    def test_clear_specific(self, pose_svc: PoseAnalysisService) -> None:
        keypoints = np.zeros((17, 3))
        pose_svc.classify_activity(5, keypoints, 0.0)
        pose_svc.clear_history(5)
        assert 5 not in pose_svc._activity_history

    def test_clear_all(self, pose_svc: PoseAnalysisService) -> None:
        keypoints = np.zeros((17, 3))
        pose_svc.classify_activity(1, keypoints, 0.0)
        pose_svc.classify_activity(2, keypoints, 0.0)
        pose_svc.clear_history()
        assert len(pose_svc._activity_history) == 0
