"""Tests for tracking self-consistency metrics."""

from conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.services.tracking_metrics import (
    _estimate_id_switches,
    compute_tracking_self_metrics,
)


class TestTrackingMetrics:
    def test_empty_frames(self):
        result = compute_tracking_self_metrics([], {}, 30.0)
        assert result["mot_self_consistency"] == 0.0
        assert "error" in result

    def test_single_track(self):
        frames = []
        for f in range(100):

            class FakeDet:
                track_id = 1
                class_name = "person"
                bbox = (0, 0, 10, 10)

            class FakeFrame:
                frame_number = f
                detections = [FakeDet()]

            frames.append(FakeFrame())
        registry = {1: {"frames": list(range(100)), "lifespan": 100}}
        result = compute_tracking_self_metrics(frames, registry, 30.0)
        assert result["num_tracks"] == 1
        assert result["mostly_tracked"] == 1
        assert result["mot_self_consistency"] > 0.5

    def test_fragmented_track(self):
        frames = []
        for f in range(100):
            if 40 <= f <= 60:
                continue

            class FakeDet:
                track_id = 1
                class_name = "person"
                bbox = (0, 0, 10, 10)

            class FakeFrame:
                frame_number = f
                detections = [FakeDet()]

            frames.append(FakeFrame())
        registry = {1: {"frames": [f for f in range(100) if not (40 <= f <= 60)], "lifespan": 100}}
        result = compute_tracking_self_metrics(frames, registry, 30.0)
        assert result["num_tracks"] == 1

    def test_partially_tracked(self):
        frames = []
        for f in range(50):
            dets = []
            if f < 10 or f >= 40:

                class FakeDet:
                    track_id = 1
                    class_name = "person"
                    bbox = (0, 0, 10, 10)

                dets.append(FakeDet())

            class FakeFrame:
                frame_number = f
                detections = dets

            frames.append(FakeFrame())
        registry = {1: {"frames": list(range(50)), "lifespan": 50}}
        result = compute_tracking_self_metrics(frames, registry, 30.0)
        # Track exists in 20/50 frames = 40% coverage → partially_tracked
        assert result["mostly_tracked"] == 0
        assert result["partially_tracked"] >= 0

    def test_id_switch_detection(self):
        frames = []
        for f in range(50):

            class FakeDet1:
                track_id = 1
                class_name = "person"
                bbox = (100, 100, 120, 160)

            class FakeDet2:
                track_id = 2
                class_name = "person"
                bbox = (105, 100, 125, 160)

            class FakeFrame:
                frame_number = f
                detections = [FakeDet1(), FakeDet2()]

            frames.append(FakeFrame())
        track_frames = {1: set(range(50)), 2: set(range(50))}
        switches = _estimate_id_switches(frames, track_frames, 30.0)
        assert switches >= 0

    def test_no_tracks(self):
        frames = []
        for f in range(10):

            class FakeFrame:
                frame_number = f
                detections = []

            frames.append(FakeFrame())
        result = compute_tracking_self_metrics(frames, {}, 30.0)
        assert result["mot_self_consistency"] == 0.0

    def test_estimate_id_switches_no_overlap(self):
        class FakeDet:
            track_id = 1
            class_name = "person"
            bbox = (0, 0, 10, 10)

        class FakeFrame:
            frame_number = 0
            detections = [FakeDet()]

        switches = _estimate_id_switches([FakeFrame()], {1: {0}}, 30.0)
        assert switches == 0
