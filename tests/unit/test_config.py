"""Tests for core configuration module (Sprint 1)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from kawkab.core.config import (
    CameraCutConfig,
    ColorConfig,
    DetectionConfig,
    EventDetectionConfig,
    FilterConfig,
    PerformanceConfig,
    PitchDetectionConfig,
    StitchConfig,
    TrackingConfig,
    TrackingConfigRoot,
    _load_yaml,
    _parse_value,
)

# ── DetectionConfig ───────────────────────────────────────────────────────


class TestDetectionConfig:
    def test_default_values(self):
        c = DetectionConfig()
        assert c.confidence_threshold == 0.4
        assert c.ball_confidence_threshold == 0.15
        assert c.iou_threshold == 0.5
        assert c.max_bbox_area_ratio == 0.15
        assert c.min_bbox_area_ratio == 0.002
        assert c.classes == [0, 32]
        assert c.ball_size_min_px == 4
        assert c.ball_size_max_px == 12
        assert c.ball_circularity_min == 0.6
        assert c.tile_overlap == 0.2
        assert c.enable_tiling is False

    def test_override_values(self):
        c = DetectionConfig(confidence_threshold=0.8, enable_tiling=True)
        assert c.confidence_threshold == 0.8
        assert c.enable_tiling is True
        assert c.iou_threshold == 0.5  # unchanged


# ── TrackingConfig ────────────────────────────────────────────────────────


class TestTrackingConfig:
    def test_default_values(self):
        c = TrackingConfig()
        assert c.max_age == 30
        assert c.min_hits == 3
        assert c.w_association_emb == 0.75
        assert c.iou_match_thresh == 0.8
        assert c.reid_embedding_cap == 8
        assert c.reid_sample_rate == 30
        assert c.enable_camera_motion_compensation is False

    def test_override_values(self):
        c = TrackingConfig(max_age=60, reid_sample_rate=15)
        assert c.max_age == 60
        assert c.reid_sample_rate == 15
        assert c.min_hits == 3


# ── FilterConfig ──────────────────────────────────────────────────────────


class TestFilterConfig:
    def test_default_values(self):
        c = FilterConfig()
        assert c.expected_player_count == 22
        assert c.max_keep_top_n == 28
        assert c.min_track_lifetime_frames == 30
        assert c.broadcast_frag_ratio_threshold == 0.2
        assert c.broadcast_stage1_divisor == 3000
        assert c.broadcast_stage1_pct == 0.02
        assert c.singlecam_stage1_pct == 1.0
        assert c.broadcast_stage3_min_segments == 2
        assert c.broadcast_stage3_min_pct == 0.15
        assert c.broadcast_stage3_top_k_buffer == 3


# ── StitchConfig ──────────────────────────────────────────────────────────


class TestStitchConfig:
    def test_default_values(self):
        c = StitchConfig()
        assert c.spatial_threshold_px == 50.0
        assert c.temporal_gap_max == 2.0
        assert c.gap_multiplier == 1.5
        assert c.overlap_ratio == 0.3
        assert c.color_distance_threshold == 70.0
        assert c.face_distance_threshold == 0.6
        assert c.reid_similarity_threshold == 0.6
        assert c.reid_few_emb_threshold == 0.7
        assert c.reid_many_emb_threshold == 0.65
        assert c.color_few_samples_threshold == 70.0
        assert c.color_many_samples_threshold == 55.0


# ── CameraCutConfig ───────────────────────────────────────────────────────


class TestCameraCutConfig:
    def test_default_values(self):
        c = CameraCutConfig()
        assert c.hue_bins == 32
        assert c.sat_bins == 8
        assert c.threshold == 0.35
        assert c.min_cut_interval == 0.5
        assert c.sample_every_n == 6
        assert c.segment_min_frames == 6

    def test_override(self):
        c = CameraCutConfig(threshold=0.5, sample_every_n=10)
        assert c.threshold == 0.5
        assert c.sample_every_n == 10


# ── PitchDetectionConfig ──────────────────────────────────────────────────


class TestPitchDetectionConfig:
    def test_default_values(self):
        c = PitchDetectionConfig()
        assert c.min_line_length == 80
        assert c.max_line_gap == 12
        assert c.canny_low == 50
        assert c.canny_high == 150
        assert c.hough_threshold == 80
        assert c.min_confidence == 0.15


# ── ColorConfig ───────────────────────────────────────────────────────────


class TestColorConfig:
    def test_default_values(self):
        c = ColorConfig()
        assert c.pitch_green_lower == [25, 40, 40]
        assert c.pitch_green_upper == [90, 255, 255]
        assert c.n_clusters == 3
        assert c.white_threshold == 230
        assert c.black_threshold == 30
        assert c.min_color_samples == 3
        assert c.color_sample_rate_hz == 2.0
        assert c.jpeg_ocr_sample_rate == 30


# ── PerformanceConfig ─────────────────────────────────────────────────────


class TestPerformanceConfig:
    def test_default_values(self):
        c = PerformanceConfig()
        assert c.frame_skip == 6
        assert c.checkpoint_interval == 500
        assert c.enable_checkpoint is False
        assert c.gpu_enabled is True
        assert c.half_precision is True
        assert c.enable_streaming is False
        assert c.batch_reid is True

    def test_override(self):
        c = PerformanceConfig(frame_skip=3, gpu_enabled=False)
        assert c.frame_skip == 3
        assert c.gpu_enabled is False


# ── EventDetectionConfig ──────────────────────────────────────────────────


class TestEventDetectionConfig:
    def test_default_values(self):
        c = EventDetectionConfig()
        assert c.goal_line_x_ratio == 0.05
        assert c.min_pass_duration == 0.3
        assert c.max_pass_duration == 6.0
        assert c.min_pass_px == 50.0
        assert c.min_pass_straightness == 0.5
        assert c.min_shot_px == 60.0
        assert c.max_shot_duration == 1.5
        assert c.min_shot_straightness == 0.3
        assert c.ball_conf_min == 0.3
        assert c.segment_gap_time == 0.5
        assert c.segment_max_jump_px == 300.0
        assert c.shot_dedup_window == 2.0
        assert c.pass_dedup_window == 1.0


# ── TrackingConfigRoot ────────────────────────────────────────────────────


class TestTrackingConfigRoot:
    def test_default_root(self):
        root = TrackingConfigRoot()
        assert isinstance(root.detection, DetectionConfig)
        assert isinstance(root.tracking, TrackingConfig)
        assert isinstance(root.filter, FilterConfig)
        assert isinstance(root.stitch, StitchConfig)
        assert isinstance(root.camera_cut, CameraCutConfig)
        assert isinstance(root.pitch, PitchDetectionConfig)
        assert isinstance(root.color, ColorConfig)
        assert isinstance(root.event, EventDetectionConfig)
        assert isinstance(root.performance, PerformanceConfig)

    def test_from_dict_override_detection(self):
        root = TrackingConfigRoot._from_dict(
            {"detection": {"confidence_threshold": 0.9, "enable_tiling": True}}
        )
        assert root.detection.confidence_threshold == 0.9
        assert root.detection.enable_tiling is True
        assert root.detection.iou_threshold == 0.5  # unchanged

    def test_from_dict_override_tracking(self):
        root = TrackingConfigRoot._from_dict(
            {"tracking": {"max_age": 100, "iou_match_thresh": 0.6}}
        )
        assert root.tracking.max_age == 100
        assert root.tracking.iou_match_thresh == 0.6

    def test_from_dict_override_performance(self):
        root = TrackingConfigRoot._from_dict(
            {"performance": {"frame_skip": 2, "gpu_enabled": False}}
        )
        assert root.performance.frame_skip == 2
        assert root.performance.gpu_enabled is False

    def test_from_dict_empty(self):
        root = TrackingConfigRoot._from_dict({})
        assert root.detection.confidence_threshold == 0.4

    def test_from_dict_unknown_section_ignored(self):
        root = TrackingConfigRoot._from_dict({"unknown_section": {"a": 1}})
        assert root.detection.confidence_threshold == 0.4

    def test_from_dict_partial_override(self):
        root = TrackingConfigRoot._from_dict(
            {
                "filter": {"expected_player_count": 10},
                "stitch": {"spatial_threshold_px": 30.0},
            }
        )
        assert root.filter.expected_player_count == 10
        assert root.filter.max_keep_top_n == 28
        assert root.stitch.spatial_threshold_px == 30.0

    def test_load_json(self):
        data = {"detection": {"confidence_threshold": 0.7}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            p = Path(f.name)
        try:
            root = TrackingConfigRoot.load(p)
            assert root.detection.confidence_threshold == 0.7
        finally:
            p.unlink(missing_ok=True)

    def test_load_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            TrackingConfigRoot.load(Path("/nonexistent/config.json"))


# ── _load_yaml ────────────────────────────────────────────────────────────


class TestLoadYaml:
    def test_valid_yaml(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write("# Config\n")
            f.write("detection:\n")
            f.write("  confidence_threshold: 0.8\n")
            f.write("  enable_tiling: true\n")
            f.write("tracking:\n")
            f.write("  max_age: 60\n")
            f.flush()
            p = Path(f.name)
        try:
            data = _load_yaml(p)
            assert "detection" in data
            assert "tracking" in data
            assert data["detection"]["confidence_threshold"] == 0.8
            assert data["detection"]["enable_tiling"] is True
            assert data["tracking"]["max_age"] == 60
        finally:
            p.unlink(missing_ok=True)

    def test_empty_yaml(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write("# just a comment\n")
            f.flush()
            p = Path(f.name)
        try:
            data = _load_yaml(p)
            assert data == {}
        finally:
            p.unlink(missing_ok=True)

    def test_yaml_list_values(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write("detection:\n")
            f.write("  classes: [0, 32, 42]\n")
            f.flush()
            p = Path(f.name)
        try:
            data = _load_yaml(p)
            assert data["detection"]["classes"] == [0, 32, 42]
        finally:
            p.unlink(missing_ok=True)

    def test_yaml_float_values(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write("stitch:\n")
            f.write("  spatial_threshold_px: 75.5\n")
            f.flush()
            p = Path(f.name)
        try:
            data = _load_yaml(p)
            assert data["stitch"]["spatial_threshold_px"] == 75.5
        finally:
            p.unlink(missing_ok=True)


# ── _parse_value ──────────────────────────────────────────────────────────


class TestParseValue:
    def test_boolean_true(self):
        assert _parse_value("true") is True
        assert _parse_value("True") is True
        assert _parse_value("yes") is True

    def test_boolean_false(self):
        assert _parse_value("false") is False
        assert _parse_value("False") is False
        assert _parse_value("no") is False

    def test_integer(self):
        assert _parse_value("42") == 42
        assert _parse_value("0") == 0
        assert _parse_value("-5") == -5

    def test_float(self):
        assert _parse_value("3.14") == 3.14
        assert _parse_value("0.5") == 0.5

    def test_list(self):
        assert _parse_value("[1, 2, 3]") == [1, 2, 3]
        assert _parse_value("[true, false]") == [True, False]
        assert _parse_value("[1.5, 2.5]") == [1.5, 2.5]

    def test_string_fallback(self):
        assert _parse_value("hello") == "hello"
        assert _parse_value("some text") == "some text"
