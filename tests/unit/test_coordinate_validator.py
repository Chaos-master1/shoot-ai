"""Tests for CoordinateValidator."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.core.coordinate_validator import CoordinateValidator


class TestCoordinateValidator:
    def test_valid_coordinates_pass(self):
        r = CoordinateValidator.validate_point(50, 34)
        assert r.valid
        assert r.errors == []
        assert not r.clamped

    def test_negative_x_clamped_to_zero(self):
        r = CoordinateValidator.validate_x(-5)
        assert r.valid
        assert any("clamped" in w.lower() for w in r.warnings)
        assert r.clamped
        assert CoordinateValidator.clamp_x(-5) == 0.0

    def test_x_above_105_clamped(self):
        r = CoordinateValidator.validate_x(200)
        assert r.valid
        assert r.clamped
        assert CoordinateValidator.clamp_x(200) == 105.0

    def test_negative_y_clamped_to_zero(self):
        r = CoordinateValidator.validate_y(-10)
        assert r.valid
        assert r.clamped
        assert CoordinateValidator.clamp_y(-10) == 0.0

    def test_y_above_68_clamped(self):
        r = CoordinateValidator.validate_y(100)
        assert r.valid
        assert r.clamped
        assert CoordinateValidator.clamp_y(100) == 68.0

    def test_non_numeric_coordinates_fail(self):
        r = CoordinateValidator.validate_point("abc", 34)
        assert not r.valid
        assert len(r.errors) > 0

    def test_event_with_all_spatial_fields(self):
        event = {"x": 50, "y": 34, "end_x": 105, "end_y": 68, "start_x": 0, "start_y": 0}
        r = CoordinateValidator.validate_event_spatial(event)
        assert r.valid
        assert not r.clamped

    def test_event_with_no_spatial_fields_passes(self):
        event = {"type": "pass", "team": "home"}
        r = CoordinateValidator.validate_event_spatial(event)
        assert r.valid

    def test_event_out_of_bounds_clamped(self):
        event = {"x": -1, "y": 200}
        r = CoordinateValidator.validate_event_spatial(event)
        assert r.valid
        assert r.clamped
        assert event["x"] == 0.0
        assert event["y"] == 68.0

    def test_event_non_numeric_field_fails(self):
        event = {"x": "invalid", "y": 34}
        r = CoordinateValidator.validate_event_spatial(event)
        assert not r.valid
