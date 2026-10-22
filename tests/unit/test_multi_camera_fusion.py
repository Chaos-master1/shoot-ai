"""Tests for multi-camera fusion — merging tracks from multiple calibrated cameras."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()

import numpy as np

_mod = load_service_module("kawkab.services.multi_camera_service", "multi_camera_service.py")
MultiCameraFusion = _mod.MultiCameraFusion
CameraView = _mod.CameraView
FusedTrack = _mod.FusedTrack


def _pitch_homography():
    """Scales pixel coords (640x480) to valid pitch coords (105x68 m)."""
    H = np.eye(3, dtype=np.float64)
    H[0, 0] = 105.0 / 640.0
    H[1, 1] = 68.0 / 480.0
    return H


def _offset_homography(dx: float = 0.0, dy: float = 0.0):
    """Homography with pitch-space translation offset in meters.
    Offset simulates different camera angles producing slightly different
    pitch projections for the same physical position."""
    H = _pitch_homography()
    H[0, 2] = dx
    H[1, 2] = dy
    return H


def _make_track(track_id: int, bbox: tuple[float, float, float, float], confidence: float = 0.9):
    return {
        "track_id": track_id,
        "bbox": bbox,
        "confidence": confidence,
        "class_name": "person",
    }


class TestMultiCameraFusion:
    """15+ tests for multi-camera fusion service."""

    def test_initialization(self):
        fusion = MultiCameraFusion()
        assert fusion.max_distance == 3.0
        assert fusion.min_cameras == 1
        assert fusion.max_occlusion_frames == 90
        assert fusion.cameras == {}
        assert fusion.fused_tracks == {}

    def test_register_camera(self):
        fusion = MultiCameraFusion()
        H = _pitch_homography()
        fusion.register_camera("cam1", H, confidence=0.95)
        assert "cam1" in fusion.cameras
        assert fusion.cameras["cam1"].confidence == 0.95
        assert np.allclose(fusion.cameras["cam1"].homography, H)

    def test_register_multiple_cameras(self):
        fusion = MultiCameraFusion()
        fusion.register_camera("cam1", _pitch_homography(), 0.9)
        fusion.register_camera("cam2", _offset_homography(10, 5), 0.8)
        fusion.register_camera("cam3", _offset_homography(-10, -5), 0.7)
        assert len(fusion.cameras) == 3

    def test_update_unregistered_camera(self):
        fusion = MultiCameraFusion()
        result = fusion.update("unknown_cam", [], 0.0)
        assert result == []

    def test_single_camera_single_track(self):
        fusion = MultiCameraFusion()
        H = _pitch_homography()
        fusion.register_camera("cam1", H, 1.0)
        tracks = [_make_track(1, (100, 200, 120, 240), 0.9)]
        result = fusion.update("cam1", tracks, 1.0)
        assert len(result) == 1
        assert result[0].global_id == 1
        assert 0 <= result[0].pitch_x <= 105.0
        assert 0 <= result[0].pitch_y <= 68.0

    def test_two_cameras_overlapping(self):
        fusion = MultiCameraFusion(max_distance=5.0)
        H1 = _pitch_homography()
        H2 = _offset_homography(2.0, 1.0)
        fusion.register_camera("cam1", H1, 0.9)
        fusion.register_camera("cam2", H2, 0.8)

        tracks1 = [_make_track(1, (100, 200, 120, 240), 0.9)]
        tracks2 = [_make_track(1, (110, 210, 130, 250), 0.85)]

        result = fusion.update("cam1", tracks1, 1.0)
        assert len(result) == 1

        result = fusion.update("cam2", tracks2, 1.0)
        assert len(result) == 1
        assert len(result[0].camera_sources) >= 2

    def test_occlusion_handling(self):
        fusion = MultiCameraFusion(max_occlusion_frames=5)
        H1 = _pitch_homography()
        H2 = _offset_homography(0.3, 0.2)
        fusion.register_camera("cam1", H1, 0.9)
        fusion.register_camera("cam2", H2, 0.8)

        tracks1 = [_make_track(1, (100, 200, 120, 240), 0.9)]
        tracks2 = [_make_track(1, (110, 210, 130, 250), 0.85)]

        result = fusion.update("cam1", tracks1, 1.0)
        assert len(result) == 1
        gid = result[0].global_id

        result = fusion.update("cam2", tracks2, 1.0)
        assert len(result) == 1
        assert gid in [ft.global_id for ft in result]

        result = fusion.update("cam1", [], 2.0)
        assert len(result) == 1
        assert result[0].global_id == gid

        result = fusion.update("cam2", [], 3.0)
        assert len(result) == 1
        assert result[0].global_id == gid

    def test_occlusion_track_persistence(self):
        fusion = MultiCameraFusion(max_occlusion_frames=10)
        H = _pitch_homography()
        fusion.register_camera("cam1", H, 0.9)

        tracks = [_make_track(1, (100, 200, 120, 240), 0.9)]
        result = fusion.update("cam1", tracks, 1.0)
        assert len(result) == 1
        gid = result[0].global_id

        for t in range(2, 6):
            result = fusion.update("cam1", [], float(t))
            assert len(result) == 1
            assert result[0].global_id == gid

    def test_stale_track_culling(self):
        fusion = MultiCameraFusion(max_occlusion_frames=3)
        H = _pitch_homography()
        fusion.register_camera("cam1", H, 0.9)

        tracks = [_make_track(1, (100, 200, 120, 240), 0.9)]
        result = fusion.update("cam1", tracks, 1.0)
        assert len(result) == 1

        for t in range(2, 6):
            result = fusion.update("cam1", [], float(t))

        assert len(fusion.fused_tracks) == 0

    def test_camera_confidence_weighting(self):
        fusion = MultiCameraFusion(max_distance=5.0)
        H1 = _pitch_homography()
        H2 = _offset_homography(1.0, 0.5)
        fusion.register_camera("cam1", H1, 1.0)
        fusion.register_camera("cam2", H2, 0.5)

        tracks1 = [_make_track(1, (100, 200, 120, 240), 0.9)]
        tracks2 = [_make_track(1, (105, 205, 125, 245), 0.8)]

        fusion.update("cam1", tracks1, 1.0)
        result = fusion.update("cam2", tracks2, 1.0)
        assert len(result) == 1
        ft = result[0]
        assert len(ft.camera_sources) >= 2

    def test_track_persistence_across_camera_switches(self):
        fusion = MultiCameraFusion(max_distance=5.0)
        H1 = _pitch_homography()
        H2 = _offset_homography(3.0, 2.0)
        fusion.register_camera("cam1", H1, 0.9)
        fusion.register_camera("cam2", H2, 0.8)

        tracks1 = [_make_track(1, (100, 200, 120, 240), 0.9)]
        result = fusion.update("cam1", tracks1, 1.0)
        gid = result[0].global_id

        tracks2 = [_make_track(1, (110, 210, 130, 250), 0.85)]
        result = fusion.update("cam2", tracks2, 2.0)
        assert result[0].global_id == gid

        tracks1_new = [_make_track(2, (200, 300, 220, 340), 0.9)]
        result = fusion.update("cam1", tracks1_new, 3.0)
        assert result[0].global_id == gid

    def test_empty_frame_handling(self):
        fusion = MultiCameraFusion()
        H = _pitch_homography()
        fusion.register_camera("cam1", H, 0.9)
        result = fusion.update("cam1", [], 1.0)
        assert result == []

    def test_multiple_tracks_per_camera(self):
        fusion = MultiCameraFusion(max_distance=5.0)
        H1 = _pitch_homography()
        H2 = _offset_homography(2.0, 1.0)
        fusion.register_camera("cam1", H1, 0.9)
        fusion.register_camera("cam2", H2, 0.8)

        tracks1 = [
            _make_track(1, (100, 200, 120, 240), 0.9),
            _make_track(2, (300, 200, 320, 240), 0.85),
        ]
        tracks2 = [
            _make_track(1, (110, 210, 130, 250), 0.8),
            _make_track(2, (310, 210, 330, 250), 0.75),
        ]

        result = fusion.update("cam1", tracks1, 1.0)
        assert len(result) == 2

        result = fusion.update("cam2", tracks2, 1.0)
        assert len(result) == 2
        assert len(result[0].camera_sources) >= 2
        assert len(result[1].camera_sources) >= 2

    def test_state_save_load_round_trip(self, tmp_path):
        fusion = MultiCameraFusion(max_distance=4.0, min_cameras=2, max_occlusion_frames=60)
        H1 = _pitch_homography()
        H2 = _offset_homography(2.0, 1.0)
        fusion.register_camera("cam1", H1, 0.9)
        fusion.register_camera("cam2", H2, 0.8)

        tracks1 = [_make_track(1, (100, 200, 120, 240), 0.9)]
        tracks2 = [_make_track(1, (110, 210, 130, 250), 0.85)]
        fusion.update("cam1", tracks1, 1.0)
        fusion.update("cam2", tracks2, 1.0)

        save_path = tmp_path / "fusion_state.json"
        fusion.save_state(save_path)
        assert save_path.exists()

        fusion2 = MultiCameraFusion()
        fusion2.load_state(save_path)
        assert fusion2.max_distance == fusion.max_distance
        assert fusion2.min_cameras == fusion.min_cameras
        assert fusion2.max_occlusion_frames == fusion.max_occlusion_frames
        assert len(fusion2.cameras) == len(fusion.cameras)
        assert len(fusion2.fused_tracks) == len(fusion.fused_tracks)
        for gid, ft in fusion.fused_tracks.items():
            assert gid in fusion2.fused_tracks
            ft2 = fusion2.fused_tracks[gid]
            assert abs(ft.pitch_x - ft2.pitch_x) < 1e-6
            assert abs(ft.pitch_y - ft2.pitch_y) < 1e-6
            assert ft.camera_sources == ft2.camera_sources

    def test_pitch_coverage_reporting(self):
        fusion = MultiCameraFusion()
        H1 = _pitch_homography()
        H2 = _offset_homography(20.0, 10.0)
        fusion.register_camera("cam1", H1, 0.9)
        fusion.register_camera("cam2", H2, 0.8)

        tracks1 = [
            _make_track(1, (100, 200, 120, 240), 0.9),
            _make_track(2, (300, 200, 320, 240), 0.85),
        ]
        fusion.update("cam1", tracks1, 1.0)

        coverage = fusion.get_pitch_coverage()
        assert "cam1" in coverage
        assert coverage["cam1"]["track_count"] == 2
        assert coverage["cam1"]["avg_confidence"] > 0

    def test_camera_contributions(self):
        fusion = MultiCameraFusion(max_distance=5.0)
        H1 = _pitch_homography()
        H2 = _offset_homography(2.0, 1.0)
        fusion.register_camera("cam1", H1, 0.9)
        fusion.register_camera("cam2", H2, 0.8)

        tracks1 = [_make_track(1, (100, 200, 120, 240), 0.9)]
        tracks2 = [_make_track(1, (110, 210, 130, 250), 0.85)]
        fusion.update("cam1", tracks1, 1.0)
        fusion.update("cam2", tracks2, 1.0)

        contributions = fusion.get_camera_contributions()
        assert "cam1" in contributions
        assert "cam2" in contributions
        assert contributions["cam1"] > 0
        assert contributions["cam2"] > 0

    def test_reset(self):
        fusion = MultiCameraFusion()
        H = _pitch_homography()
        fusion.register_camera("cam1", H, 0.9)
        tracks = [_make_track(1, (100, 200, 120, 240), 0.9)]
        fusion.update("cam1", tracks, 1.0)
        assert len(fusion.fused_tracks) == 1

        fusion.reset()
        assert fusion.fused_tracks == {}
        assert fusion._track_history == {}
        assert fusion._next_global_id == 1
        assert fusion._frame_count == 0

    def test_three_cameras_fusion(self):
        fusion = MultiCameraFusion(max_distance=5.0)
        H1 = _pitch_homography()
        H2 = _offset_homography(1.0, 0.5)
        H3 = _offset_homography(-1.0, -0.5)
        fusion.register_camera("cam1", H1, 0.95)
        fusion.register_camera("cam2", H2, 0.85)
        fusion.register_camera("cam3", H3, 0.75)

        tracks1 = [_make_track(1, (100, 200, 120, 240), 0.9)]
        tracks2 = [_make_track(1, (105, 205, 125, 245), 0.85)]
        tracks3 = [_make_track(1, (95, 195, 115, 235), 0.8)]

        fusion.update("cam1", tracks1, 1.0)
        fusion.update("cam2", tracks2, 1.0)
        result = fusion.update("cam3", tracks3, 1.0)
        assert len(result) == 1
        assert len(result[0].camera_sources) >= 3

    def test_camera_contributions_empty(self):
        fusion = MultiCameraFusion()
        H = _pitch_homography()
        fusion.register_camera("cam1", H, 0.9)
        contributions = fusion.get_camera_contributions()
        assert contributions == {"cam1": 0.0}

    def test_pitch_coverage_empty(self):
        fusion = MultiCameraFusion()
        H = _pitch_homography()
        fusion.register_camera("cam1", H, 0.9)
        coverage = fusion.get_pitch_coverage()
        assert coverage["cam1"]["track_count"] == 0
        assert coverage["cam1"]["covered_area_pct"] == 0.0

    def test_state_save_load_empty(self, tmp_path):
        fusion = MultiCameraFusion()
        H = _pitch_homography()
        fusion.register_camera("cam1", H, 0.9)

        save_path = tmp_path / "empty_state.json"
        fusion.save_state(save_path)

        fusion2 = MultiCameraFusion()
        fusion2.load_state(save_path)
        assert len(fusion2.cameras) == 1
        assert len(fusion2.fused_tracks) == 0

    def test_load_nonexistent_state(self, tmp_path):
        fusion = MultiCameraFusion()
        fusion.load_state(tmp_path / "nonexistent.json")
        assert len(fusion.cameras) == 0

    def test_reset_clears_all_state(self):
        fusion = MultiCameraFusion()
        H = _pitch_homography()
        fusion.register_camera("cam1", H, 0.9)
        fusion.register_camera("cam2", H, 0.8)
        tracks = [_make_track(1, (100, 200, 120, 240), 0.9)]
        fusion.update("cam1", tracks, 1.0)
        fusion.update("cam2", tracks, 1.0)
        assert len(fusion.fused_tracks) == 1
        fusion.reset()
        assert len(fusion.fused_tracks) == 0
        assert fusion._next_global_id == 1
        assert fusion._frame_count == 0

    def test_three_cameras_no_overlap(self):
        fusion = MultiCameraFusion(max_distance=3.0)
        H1 = _pitch_homography()
        H2 = _offset_homography(50.0, 30.0)
        H3 = _offset_homography(-50.0, -30.0)
        fusion.register_camera("cam1", H1, 0.9)
        fusion.register_camera("cam2", H2, 0.8)
        fusion.register_camera("cam3", H3, 0.7)

        tracks1 = [_make_track(1, (100, 200, 120, 240), 0.9)]
        tracks2 = [_make_track(1, (100, 200, 120, 240), 0.8)]
        tracks3 = [_make_track(1, (100, 200, 120, 240), 0.7)]

        fusion.update("cam1", tracks1, 1.0)
        fusion.update("cam2", tracks2, 1.0)
        result = fusion.update("cam3", tracks3, 1.0)
        assert len(result) == 3

    def test_confidence_weighted_position(self):
        fusion = MultiCameraFusion(max_distance=10.0)
        H1 = _pitch_homography()
        H2 = _offset_homography(0.0, 0.0)
        fusion.register_camera("cam1", H1, 1.0)
        fusion.register_camera("cam2", H2, 0.5)

        tracks1 = [_make_track(1, (100, 200, 120, 240), 0.9)]
        tracks2 = [_make_track(1, (104, 204, 124, 244), 0.8)]

        fusion.update("cam1", tracks1, 1.0)
        result = fusion.update("cam2", tracks2, 1.0)
        assert len(result) == 1
        ft = result[0]
        cx1, cy1 = 110.0, 220.0
        cx2, cy2 = 114.0, 224.0
        expected_x = ((cx1 * 105.0 / 640.0) * 1.0 + (cx2 * 105.0 / 640.0) * 0.5) / 1.5
        expected_y = ((cy1 * 68.0 / 480.0) * 1.0 + (cy2 * 68.0 / 480.0) * 0.5) / 1.5
        assert abs(ft.pitch_x - expected_x) < 0.5
        assert abs(ft.pitch_y - expected_y) < 0.5

    def test_global_id_persistence(self):
        fusion = MultiCameraFusion(max_distance=5.0)
        H1 = _pitch_homography()
        H2 = _offset_homography(2.0, 1.0)
        fusion.register_camera("cam1", H1, 0.9)
        fusion.register_camera("cam2", H2, 0.8)

        tracks1 = [_make_track(1, (100, 200, 120, 240), 0.9)]
        result = fusion.update("cam1", tracks1, 1.0)
        gid1 = result[0].global_id

        tracks2 = [_make_track(1, (110, 210, 130, 250), 0.85)]
        result = fusion.update("cam2", tracks2, 1.0)
        assert result[0].global_id == gid1

        tracks1_new = [_make_track(2, (400, 300, 420, 340), 0.9)]
        result = fusion.update("cam1", tracks1_new, 2.0)
        assert len(result) == 2
        gids = [ft.global_id for ft in result]
        assert gid1 in gids

    def test_three_cameras_all_overlapping(self):
        fusion = MultiCameraFusion(max_distance=5.0)
        H1 = _pitch_homography()
        H2 = _offset_homography(1.0, 0.5)
        H3 = _offset_homography(-0.5, -0.3)
        fusion.register_camera("cam1", H1, 0.95)
        fusion.register_camera("cam2", H2, 0.85)
        fusion.register_camera("cam3", H3, 0.75)

        tracks1 = [_make_track(1, (100, 200, 120, 240), 0.9)]
        tracks2 = [_make_track(1, (105, 205, 125, 245), 0.85)]
        tracks3 = [_make_track(1, (98, 198, 118, 238), 0.8)]

        fusion.update("cam1", tracks1, 1.0)
        fusion.update("cam2", tracks2, 1.0)
        result = fusion.update("cam3", tracks3, 1.0)
        assert len(result) == 1
        assert len(result[0].camera_sources) == 3

    def test_occlusion_then_reappearance(self):
        fusion = MultiCameraFusion(max_occlusion_frames=10)
        H1 = _pitch_homography()
        H2 = _offset_homography(2.0, 1.0)
        fusion.register_camera("cam1", H1, 0.9)
        fusion.register_camera("cam2", H2, 0.8)

        tracks1 = [_make_track(1, (100, 200, 120, 240), 0.9)]
        result = fusion.update("cam1", tracks1, 1.0)
        gid = result[0].global_id

        for t in range(2, 6):
            fusion.update("cam1", [], float(t))

        assert gid in fusion.fused_tracks

        tracks1_new = [_make_track(1, (150, 250, 170, 290), 0.9)]
        result = fusion.update("cam1", tracks1_new, 6.0)
        assert result[0].global_id == gid

    def test_save_load_with_tracks(self, tmp_path):
        fusion = MultiCameraFusion(max_distance=4.0)
        H1 = _pitch_homography()
        H2 = _offset_homography(0.5, 0.3)
        fusion.register_camera("cam1", H1, 0.9)
        fusion.register_camera("cam2", H2, 0.8)

        tracks1 = [_make_track(1, (100, 200, 120, 240), 0.9)]
        tracks2 = [_make_track(1, (105, 205, 125, 245), 0.85)]
        fusion.update("cam1", tracks1, 1.0)
        fusion.update("cam2", tracks2, 1.0)

        save_path = tmp_path / "fusion_state.json"
        fusion.save_state(save_path)

        fusion2 = MultiCameraFusion()
        fusion2.load_state(save_path)
        assert len(fusion2.cameras) == 2
        assert len(fusion2.fused_tracks) == 1
        ft = list(fusion2.fused_tracks.values())[0]
        assert ft.pitch_x > 0
        assert ft.pitch_y > 0
        assert len(ft.camera_sources) >= 2

    def test_occlusion_velocity_extrapolation(self):
        fusion = MultiCameraFusion(max_occlusion_frames=5)
        H = _pitch_homography()
        fusion.register_camera("cam1", H, 1.0)

        tracks1 = [_make_track(1, (100, 200, 120, 240), 0.9)]
        result = fusion.update("cam1", tracks1, 1.0)
        gid = result[0].global_id
        x0 = result[0].pitch_x
        y0 = result[0].pitch_y

        tracks2 = [_make_track(1, (150, 250, 170, 290), 0.9)]
        result = fusion.update("cam1", tracks2, 2.0)
        x1 = result[0].pitch_x
        y1 = result[0].pitch_y

        result = fusion.update("cam1", [], 3.0)
        assert result[0].global_id == gid
        dx = x1 - x0
        dy = y1 - y0
        assert abs(result[0].pitch_x - (x1 + dx)) < 1.0
        assert abs(result[0].pitch_y - (y1 + dy)) < 1.0

    def test_confidence_decay_during_occlusion(self):
        fusion = MultiCameraFusion(max_occlusion_frames=10)
        H = _pitch_homography()
        fusion.register_camera("cam1", H, 1.0)

        tracks = [_make_track(1, (100, 200, 120, 240), 0.9)]
        result = fusion.update("cam1", tracks, 1.0)
        initial_conf = result[0].confidence

        for t in range(2, 6):
            result = fusion.update("cam1", [], float(t))
            assert result[0].confidence < initial_conf

    def test_tracks_outside_max_distance_not_matched(self):
        fusion = MultiCameraFusion(max_distance=2.0)
        H1 = _pitch_homography()
        H2 = _offset_homography(0.0, 0.0)
        fusion.register_camera("cam1", H1, 0.9)
        fusion.register_camera("cam2", H2, 0.8)

        tracks1 = [_make_track(1, (100, 200, 120, 240), 0.9)]
        tracks2 = [_make_track(1, (500, 400, 520, 440), 0.8)]

        fusion.update("cam1", tracks1, 1.0)
        result = fusion.update("cam2", tracks2, 1.0)
        assert len(result) == 2

    def test_camera_confidence_weighting_merges_correctly(self):
        fusion = MultiCameraFusion(max_distance=10.0)
        H1 = _pitch_homography()
        H2 = _offset_homography(0.0, 0.0)
        fusion.register_camera("cam1", H1, 1.0)
        fusion.register_camera("cam2", H2, 0.5)

        tracks1 = [_make_track(1, (100, 200, 120, 240), 0.9)]
        tracks2 = [_make_track(1, (104, 204, 124, 244), 0.8)]

        fusion.update("cam1", tracks1, 1.0)
        result = fusion.update("cam2", tracks2, 1.0)
        ft = result[0]
        cx1, cy1 = 110.0, 220.0
        cx2, cy2 = 114.0, 224.0
        expected_x = ((cx1 * 105.0 / 640.0) * 1.0 + (cx2 * 105.0 / 640.0) * 0.5) / 1.5
        expected_y = ((cy1 * 68.0 / 480.0) * 1.0 + (cy2 * 68.0 / 480.0) * 0.5) / 1.5
        assert abs(ft.pitch_x - expected_x) < 0.5
        assert abs(ft.pitch_y - expected_y) < 0.5

    def test_multiple_tracks_per_camera_no_match(self):
        fusion = MultiCameraFusion(max_distance=2.0)
        H1 = _pitch_homography()
        H2 = _offset_homography(0.0, 0.0)
        fusion.register_camera("cam1", H1, 0.9)
        fusion.register_camera("cam2", H2, 0.8)

        tracks1 = [
            _make_track(1, (100, 200, 120, 240), 0.9),
            _make_track(2, (300, 200, 320, 240), 0.85),
        ]
        tracks2 = [
            _make_track(1, (500, 400, 520, 440), 0.8),
            _make_track(2, (600, 400, 620, 440), 0.75),
        ]

        fusion.update("cam1", tracks1, 1.0)
        result = fusion.update("cam2", tracks2, 1.0)
        assert len(result) == 4

    def test_save_load_preserves_track_history(self, tmp_path):
        fusion = MultiCameraFusion(max_distance=3.0)
        H = _pitch_homography()
        fusion.register_camera("cam1", H, 0.9)

        tracks = [_make_track(1, (100, 200, 120, 240), 0.9)]
        fusion.update("cam1", tracks, 1.0)
        fusion.update("cam1", tracks, 2.0)

        save_path = tmp_path / "fusion_hist.json"
        fusion.save_state(save_path)

        fusion2 = MultiCameraFusion()
        fusion2.load_state(save_path)
        assert len(fusion2.fused_tracks) == 1
        ft = list(fusion2.fused_tracks.values())[0]
        assert ft.pitch_x > 0

    def test_register_camera_overwrite(self):
        fusion = MultiCameraFusion()
        H1 = _pitch_homography()
        H2 = _offset_homography(10.0, 5.0)
        fusion.register_camera("cam1", H1, 0.9)
        fusion.register_camera("cam1", H2, 0.8)
        assert len(fusion.cameras) == 1
        assert np.allclose(fusion.cameras["cam1"].homography, H2)
        assert fusion.cameras["cam1"].confidence == 0.8

    def test_track_with_reid_embedding(self):
        fusion = MultiCameraFusion(max_distance=5.0)
        H1 = _pitch_homography()
        H2 = _offset_homography(2.0, 1.0)
        fusion.register_camera("cam1", H1, 0.9)
        fusion.register_camera("cam2", H2, 0.8)

        reid_emb = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        tracks1 = [
            {
                "track_id": 1,
                "bbox": (100, 200, 120, 240),
                "confidence": 0.9,
                "class_name": "person",
                "reid_embedding": reid_emb,
            }
        ]
        tracks2 = [
            {
                "track_id": 1,
                "bbox": (110, 210, 130, 250),
                "confidence": 0.85,
                "class_name": "person",
                "reid_embedding": reid_emb,
            }
        ]

        fusion.update("cam1", tracks1, 1.0)
        result = fusion.update("cam2", tracks2, 1.0)
        assert len(result) == 1
        assert len(result[0].camera_sources) >= 2

    def test_sequential_updates_same_camera(self):
        fusion = MultiCameraFusion()
        H = _pitch_homography()
        fusion.register_camera("cam1", H, 0.9)

        tracks1 = [_make_track(1, (100, 200, 120, 240), 0.9)]
        result = fusion.update("cam1", tracks1, 1.0)
        assert len(result) == 1
        gid = result[0].global_id

        tracks2 = [_make_track(1, (110, 210, 130, 250), 0.9)]
        result = fusion.update("cam1", tracks2, 2.0)
        assert len(result) == 1
        assert result[0].global_id == gid

    def test_pitch_coverage_with_tracks(self):
        fusion = MultiCameraFusion()
        H = _pitch_homography()
        fusion.register_camera("cam1", H, 0.9)

        tracks = [
            _make_track(1, (100, 200, 120, 240), 0.9),
            _make_track(2, (300, 250, 320, 290), 0.85),
            _make_track(3, (500, 300, 520, 340), 0.8),
        ]
        fusion.update("cam1", tracks, 1.0)
        coverage = fusion.get_pitch_coverage()
        assert coverage["cam1"]["track_count"] == 3
        assert coverage["cam1"]["covered_area_pct"] > 0
        assert coverage["cam1"]["avg_confidence"] > 0

    def test_camera_contributions_after_fusion(self):
        fusion = MultiCameraFusion(max_distance=5.0)
        H1 = _pitch_homography()
        H2 = _offset_homography(2.0, 1.0)
        fusion.register_camera("cam1", H1, 0.9)
        fusion.register_camera("cam2", H2, 0.8)

        tracks1 = [_make_track(1, (100, 200, 120, 240), 0.9)]
        tracks2 = [_make_track(1, (110, 210, 130, 250), 0.85)]
        fusion.update("cam1", tracks1, 1.0)
        fusion.update("cam2", tracks2, 1.0)

        contributions = fusion.get_camera_contributions()
        assert contributions["cam1"] == 1.0
        assert contributions["cam2"] == 1.0

    def test_sequential_frames_same_camera(self):
        fusion = MultiCameraFusion()
        H = _pitch_homography()
        fusion.register_camera("cam1", H, 0.9)

        for i in range(5):
            tracks = [_make_track(1, (100 + i * 10, 200, 120 + i * 10, 240), 0.9)]
            result = fusion.update("cam1", tracks, float(i + 1))
            assert len(result) == 1
            assert result[0].global_id == 1

    def test_ball_track_filtered_out(self):
        fusion = MultiCameraFusion()
        H = _pitch_homography()
        fusion.register_camera("cam1", H, 0.9)

        tracks = [
            _make_track(1, (100, 200, 120, 240), 0.9),
            {
                "track_id": 2,
                "bbox": (300, 300, 310, 310),
                "confidence": 0.5,
                "class_name": "sports ball",
            },
        ]
        result = fusion.update("cam1", tracks, 1.0)
        assert len(result) == 1

    def test_camera_contributions_no_tracks(self):
        fusion = MultiCameraFusion()
        H = _pitch_homography()
        fusion.register_camera("cam1", H, 0.9)
        fusion.register_camera("cam2", H, 0.8)
        contributions = fusion.get_camera_contributions()
        assert contributions["cam1"] == 0.0
        assert contributions["cam2"] == 0.0

    def test_pitch_coverage_single_track(self):
        fusion = MultiCameraFusion()
        H = _pitch_homography()
        fusion.register_camera("cam1", H, 0.9)
        tracks = [_make_track(1, (100, 200, 120, 240), 0.9)]
        fusion.update("cam1", tracks, 1.0)
        coverage = fusion.get_pitch_coverage()
        assert coverage["cam1"]["track_count"] == 1
        assert coverage["cam1"]["covered_area_pct"] == 0.0

    def test_occlusion_velocity_extrapolation_moves_track(self):
        fusion = MultiCameraFusion(max_occlusion_frames=5)
        H = _pitch_homography()
        fusion.register_camera("cam1", H, 1.0)

        tracks1 = [_make_track(1, (100, 200, 120, 240), 0.9)]
        result = fusion.update("cam1", tracks1, 1.0)
        x0, y0 = result[0].pitch_x, result[0].pitch_y

        tracks2 = [_make_track(1, (200, 300, 220, 340), 0.9)]
        result = fusion.update("cam1", tracks2, 2.0)
        x1, y1 = result[0].pitch_x, result[0].pitch_y

        result = fusion.update("cam1", [], 3.0)
        dx = x1 - x0
        dy = y1 - y0
        assert abs(result[0].pitch_x - (x1 + dx)) < 1.0
        assert abs(result[0].pitch_y - (y1 + dy)) < 1.0

    def test_cull_stale_tracks_removes_old(self):
        fusion = MultiCameraFusion(max_occlusion_frames=2)
        H = _pitch_homography()
        fusion.register_camera("cam1", H, 0.9)

        tracks = [_make_track(1, (100, 200, 120, 240), 0.9)]
        fusion.update("cam1", tracks, 1.0)
        assert len(fusion.fused_tracks) == 1

        for t in range(2, 6):
            fusion.update("cam1", [], float(t))

        assert len(fusion.fused_tracks) == 0

    def test_save_load_round_trip_full(self, tmp_path):
        fusion = MultiCameraFusion(max_distance=3.0, min_cameras=2, max_occlusion_frames=60)
        H1 = _pitch_homography()
        H2 = _offset_homography(0.5, 0.3)
        fusion.register_camera("cam1", H1, 0.95)
        fusion.register_camera("cam2", H2, 0.85)

        tracks1 = [_make_track(1, (100, 200, 120, 240), 0.9)]
        tracks2 = [_make_track(1, (105, 205, 125, 245), 0.85)]
        fusion.update("cam1", tracks1, 1.0)
        fusion.update("cam2", tracks2, 1.0)

        save_path = tmp_path / "fusion_state.json"
        fusion.save_state(save_path)

        fusion2 = MultiCameraFusion()
        fusion2.load_state(save_path)
        assert fusion2.max_distance == 3.0
        assert fusion2.min_cameras == 2
        assert fusion2.max_occlusion_frames == 60
        assert len(fusion2.cameras) == 2
        assert len(fusion2.fused_tracks) == 1
        ft = list(fusion2.fused_tracks.values())[0]
        assert ft.pitch_x > 0
        assert ft.pitch_y > 0
        assert "cam1" in ft.camera_sources
        assert "cam2" in ft.camera_sources

    def test_sequential_updates_preserve_global_id(self):
        fusion = MultiCameraFusion()
        H = _pitch_homography()
        fusion.register_camera("cam1", H, 0.9)

        for i in range(10):
            tracks = [_make_track(1, (100 + i * 5, 200, 120 + i * 5, 240), 0.9)]
            result = fusion.update("cam1", tracks, float(i + 1))
            assert len(result) == 1
            assert result[0].global_id == 1

    def test_camera_switch_preserves_id(self):
        fusion = MultiCameraFusion(max_distance=5.0)
        H1 = _pitch_homography()
        H2 = _offset_homography(2.0, 1.0)
        fusion.register_camera("cam1", H1, 0.9)
        fusion.register_camera("cam2", H2, 0.8)

        tracks1 = [_make_track(1, (100, 200, 120, 240), 0.9)]
        result = fusion.update("cam1", tracks1, 1.0)
        gid = result[0].global_id

        tracks2 = [_make_track(1, (110, 210, 130, 250), 0.85)]
        result = fusion.update("cam2", tracks2, 2.0)
        assert result[0].global_id == gid

        tracks1_new = [_make_track(2, (200, 300, 220, 340), 0.9)]
        result = fusion.update("cam1", tracks1_new, 3.0)
        assert result[0].global_id == gid

    def test_camera_contributions_partial(self):
        fusion = MultiCameraFusion(max_distance=5.0)
        H1 = _pitch_homography()
        H2 = _offset_homography(50.0, 30.0)
        fusion.register_camera("cam1", H1, 0.9)
        fusion.register_camera("cam2", H2, 0.8)

        tracks1 = [_make_track(1, (100, 200, 120, 240), 0.9)]
        tracks2 = [_make_track(1, (100, 200, 120, 240), 0.8)]

        fusion.update("cam1", tracks1, 1.0)
        fusion.update("cam2", tracks2, 1.0)

        contributions = fusion.get_camera_contributions()
        assert contributions["cam1"] > 0
        assert contributions["cam2"] > 0

    def test_reset_clears_camera_tracks(self):
        fusion = MultiCameraFusion()
        H = _pitch_homography()
        fusion.register_camera("cam1", H, 0.9)
        tracks = [_make_track(1, (100, 200, 120, 240), 0.9)]
        fusion.update("cam1", tracks, 1.0)
        assert fusion.cameras["cam1"].tracks != []
        fusion.reset()
        assert fusion.cameras["cam1"].tracks == []
        assert fusion.cameras["cam1"].timestamp == 0.0

    def test_save_load_no_tracks(self, tmp_path):
        fusion = MultiCameraFusion()
        H = _pitch_homography()
        fusion.register_camera("cam1", H, 0.9)
        fusion.register_camera("cam2", H, 0.8)

        save_path = tmp_path / "empty_fusion.json"
        fusion.save_state(save_path)

        fusion2 = MultiCameraFusion()
        fusion2.load_state(save_path)
        assert len(fusion2.cameras) == 2
        assert len(fusion2.fused_tracks) == 0

    def test_pitch_coverage_multiple_cameras(self):
        fusion = MultiCameraFusion()
        H1 = _pitch_homography()
        H2 = _offset_homography(0.5, 0.3)
        fusion.register_camera("cam1", H1, 0.9)
        fusion.register_camera("cam2", H2, 0.8)

        tracks1 = [
            _make_track(1, (100, 200, 120, 240), 0.9),
            _make_track(2, (500, 300, 520, 340), 0.85),
        ]
        tracks2 = [
            _make_track(1, (100, 200, 120, 240), 0.8),
            _make_track(2, (500, 300, 520, 340), 0.75),
        ]

        fusion.update("cam1", tracks1, 1.0)
        fusion.update("cam2", tracks2, 1.0)

        coverage = fusion.get_pitch_coverage()
        assert coverage["cam1"]["track_count"] == 2
        assert coverage["cam2"]["track_count"] == 2
        assert coverage["cam1"]["covered_area_pct"] > 0
        assert coverage["cam2"]["covered_area_pct"] > 0

    def test_heading_and_speed_defaults(self):
        fusion = MultiCameraFusion()
        H = _pitch_homography()
        fusion.register_camera("cam1", H, 0.9)
        tracks = [_make_track(1, (100, 200, 120, 240), 0.9)]
        result = fusion.update("cam1", tracks, 1.0)
        assert result[0].speed == 0.0
        assert result[0].heading == 0.0

    def test_fused_track_dataclass(self):
        ft = FusedTrack(
            global_id=1,
            pitch_x=52.5,
            pitch_y=34.0,
            speed=4.5,
            heading=90.0,
            confidence=0.85,
            camera_sources=["cam1", "cam2"],
            last_seen_camera="cam2",
            last_seen_timestamp=42.0,
        )
        assert ft.global_id == 1
        assert ft.pitch_x == 52.5
        assert ft.pitch_y == 34.0
        assert ft.speed == 4.5
        assert ft.heading == 90.0
        assert ft.confidence == 0.85
        assert ft.camera_sources == ["cam1", "cam2"]
        assert ft.last_seen_camera == "cam2"
        assert ft.last_seen_timestamp == 42.0

    def test_camera_view_dataclass(self):
        H = _pitch_homography()
        cv = CameraView(camera_id="test_cam", homography=H, confidence=0.8, timestamp=1.5)
        assert cv.camera_id == "test_cam"
        assert np.allclose(cv.homography, H)
        assert cv.confidence == 0.8
        assert cv.timestamp == 1.5
        assert cv.tracks == []

    def test_fused_track_defaults(self):
        ft = FusedTrack(global_id=1, pitch_x=50.0, pitch_y=30.0)
        assert ft.speed == 0.0
        assert ft.heading == 0.0
        assert ft.confidence == 0.0
        assert ft.camera_sources == []
        assert ft.last_seen_camera == ""
        assert ft.last_seen_timestamp == 0.0

    def test_ball_track_not_in_fused_output(self):
        fusion = MultiCameraFusion()
        H = _pitch_homography()
        fusion.register_camera("cam1", H, 0.9)

        tracks = [
            _make_track(1, (100, 200, 120, 240), 0.9),
            {
                "track_id": 2,
                "bbox": (300, 300, 310, 310),
                "confidence": 0.5,
                "class_name": "sports ball",
            },
        ]
        result = fusion.update("cam1", tracks, 1.0)
        assert len(result) == 1
