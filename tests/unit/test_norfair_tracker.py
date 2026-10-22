"""Tests for NorfairTracker wrapper."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()


def _install_norfair_stub() -> None:
    # Deliberately unconditional: this suite unit-tests NorfairTracker's own
    # wrapping logic (dict-in/dict-out, label routing), not norfair's real
    # tracking/matching algorithm -- so the real package must never leak in.
    # An earlier "if already imported, skip" guard here meant that whenever
    # something upstream (install_kawkab_stubs() eagerly importing
    # kawkab.services, which imports cv_service, which imports norfair)
    # had already imported the real norfair package, this stub silently
    # became a no-op and the test exercised real norfair's real
    # initialization_delay/hit_counter confirmation timing instead of the
    # wrapper logic -- passing or failing depending on import order and
    # which norfair version happened to be installed, not on this file's
    # own correctness. Verified via a clean `pip install -e ".[test,cloud]"`
    # (real norfair 2.3.0 installed): every test below failed for exactly
    # this reason before this fix.
    norfair_mod = types.ModuleType("norfair")

    class TrackedObject:
        def __init__(self, detection=None, track_id=None):
            self.global_id = track_id or id(self)
            self.last_detection = detection
            self.id = self.global_id

    class Detection:
        def __init__(self, points, scores=None, label=None, data=None, embedding=None):
            self.points = points
            self.scores = scores
            self.label = label
            self.data = data or {}
            self.embedding = embedding

    class Tracker:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._next_id = 1

        def update(self, detections, coord_transformations=None, period=1):
            results = []
            for det in detections:
                tid = self._next_id
                self._next_id += 1
                obj = TrackedObject(detection=det, track_id=tid)
                results.append(obj)
            return results

    norfair_mod.Detection = Detection
    norfair_mod.Tracker = Tracker
    sys.modules["norfair"] = norfair_mod

    cm_mod = types.ModuleType("norfair.camera_motion")

    class MotionEstimator:
        def __init__(self):
            self._update_count = 0

        def update(self, frame):
            self._update_count += 1
            return {"transformation_matrix": np.eye(3, dtype=np.float32)}

    cm_mod.MotionEstimator = MotionEstimator
    sys.modules["norfair.camera_motion"] = cm_mod

    tr_mod = types.ModuleType("norfair.tracker")
    tr_mod.TrackedObject = TrackedObject
    sys.modules["norfair.tracker"] = tr_mod


def _install_cv2_stub() -> None:
    # Same reasoning as _install_norfair_stub(): must win unconditionally,
    # or a real cv2 already imported upstream silently defeats the stub.
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.COLOR_BGR2HSV = 40
    cv2_stub.cvtColor = MagicMock(side_effect=lambda img, code: img)
    cv2_stub.calcHist = MagicMock(return_value=np.ones((32, 32), dtype=np.float32) * 0.01)
    cv2_stub.normalize = MagicMock(return_value=None)
    sys.modules["cv2"] = cv2_stub


@pytest.fixture(scope="module", autouse=True)
def _stubbed_norfair_and_cv2():
    """Install the norfair/cv2 stubs for every test in this module, then
    restore whatever was in sys.modules beforehand -- without this,
    stubbing "cv2"/"norfair" here would leak into other test files that
    run afterward in the same session and need the real packages."""
    keys = ["norfair", "norfair.camera_motion", "norfair.tracker", "cv2"]
    saved = {k: sys.modules.get(k) for k in keys}
    _install_norfair_stub()
    _install_cv2_stub()
    yield
    for k in keys:
        if saved[k] is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = saved[k]


@pytest.fixture(scope="module")
def nf_mod(_stubbed_norfair_and_cv2):
    return load_service_module("kawkab.services.norfair_tracker", "norfair_tracker.py")


@pytest.fixture
def fake_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestNorfairTracker:
    def test_init(self, nf_mod):
        tracker = nf_mod.NorfairTracker()
        assert tracker._initialized is False
        assert hasattr(tracker, "_person_tracker")
        assert hasattr(tracker, "_ball_tracker")
        assert hasattr(tracker, "_motion_estimator")

    def test_reset(self, nf_mod):
        tracker = nf_mod.NorfairTracker()
        old_person = tracker._person_tracker
        old_ball = tracker._ball_tracker
        old_motion = tracker._motion_estimator
        tracker.reset()
        assert tracker._person_tracker is not old_person
        assert tracker._ball_tracker is not old_ball
        assert tracker._motion_estimator is not old_motion

    def test_update_returns_list(self, nf_mod, fake_frame):
        tracker = nf_mod.NorfairTracker()
        result = tracker.update(fake_frame, [])
        assert isinstance(result, list)

    def test_update_with_person_detection(self, nf_mod, fake_frame):
        tracker = nf_mod.NorfairTracker()
        dets = [{"bbox": (100, 100, 200, 300), "confidence": 0.9, "label": "person"}]
        result = tracker.update(fake_frame, dets)
        assert len(result) == 1
        assert result[0]["label"] == "person"
        assert result[0]["track_id"] > 0
        assert result[0]["bbox"] == (100, 100, 200, 300)
        assert result[0]["confidence"] == 0.9

    def test_update_with_multiple_persons(self, nf_mod, fake_frame):
        tracker = nf_mod.NorfairTracker()
        dets = [
            {"bbox": (100, 100, 200, 300), "confidence": 0.9, "label": "person"},
            {"bbox": (300, 100, 400, 300), "confidence": 0.8, "label": "person"},
            {"bbox": (500, 100, 600, 300), "confidence": 0.7, "label": "person"},
        ]
        result = tracker.update(fake_frame, dets)
        assert len(result) == 3
        ids = [r["track_id"] for r in result]
        assert len(set(ids)) == 3

    def test_update_with_ball_detection(self, nf_mod, fake_frame):
        tracker = nf_mod.NorfairTracker()
        dets = [{"bbox": (300, 200, 310, 210), "confidence": 0.6, "label": "sports ball"}]
        result = tracker.update(fake_frame, dets)
        assert len(result) == 1
        assert result[0]["label"] == "sports ball"

    def test_update_mixed_detections(self, nf_mod, fake_frame):
        tracker = nf_mod.NorfairTracker()
        dets = [
            {"bbox": (100, 100, 200, 300), "confidence": 0.9, "label": "person"},
            {"bbox": (300, 200, 310, 210), "confidence": 0.6, "label": "sports ball"},
        ]
        result = tracker.update(fake_frame, dets)
        assert len(result) == 2
        labels = {r["label"] for r in result}
        assert labels == {"person", "sports ball"}

    def test_update_ignores_unknown_labels(self, nf_mod, fake_frame):
        tracker = nf_mod.NorfairTracker()
        dets = [{"bbox": (0, 0, 10, 10), "confidence": 0.5, "label": "car"}]
        result = tracker.update(fake_frame, dets)
        assert result == []

    def test_multiple_frames_consistency(self, nf_mod, fake_frame):
        tracker = nf_mod.NorfairTracker()
        dets = [{"bbox": (100, 100, 200, 300), "confidence": 0.9, "label": "person"}]
        r1 = tracker.update(fake_frame, dets)
        r2 = tracker.update(fake_frame, dets)
        assert len(r1) == 1
        assert len(r2) == 1

    def test_update_with_period(self, nf_mod, fake_frame):
        tracker = nf_mod.NorfairTracker()
        dets = [{"bbox": (100, 100, 200, 300), "confidence": 0.9, "label": "person"}]
        result = tracker.update(fake_frame, dets, period=3)
        assert len(result) == 1

    def test_confidence_in_output(self, nf_mod, fake_frame):
        tracker = nf_mod.NorfairTracker()
        dets = [{"bbox": (50, 60, 150, 260), "confidence": 0.75, "label": "person"}]
        result = tracker.update(fake_frame, dets)
        assert result[0]["confidence"] == 0.75

    def test_bbox_integrity(self, nf_mod, fake_frame):
        tracker = nf_mod.NorfairTracker()
        bbox = (50, 60, 150, 260)
        dets = [{"bbox": bbox, "confidence": 0.9, "label": "person"}]
        result = tracker.update(fake_frame, dets)
        assert result[0]["bbox"] == bbox
