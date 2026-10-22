"""Tests for pitch coordinate utilities."""

import pytest

from kawkab.core.coords import (
    FINAL_THIRD_X,
    STANDARD_PITCH,
    PitchConfig,
    clamp_pitch,
    euclidean_distance_m,
    half_space,
    is_normalized,
    meters_to_pixel_fraction,
    norm_to_meters,
    pitch_third,
    zone_label,
)


class TestPitchConfig:
    def test_default_pitch(self):
        p = PitchConfig()
        assert p.length_m == 105.0
        assert p.width_m == 68.0
        assert p.half_length == 52.5
        assert p.half_width == 34.0

    def test_custom_pitch(self):
        p = PitchConfig(length_m=90, width_m=60)
        assert p.half_length == 45.0
        assert p.diagonal_m == pytest.approx((90**2 + 60**2) ** 0.5)

    def test_third_x(self):
        p = PitchConfig()
        assert p.third_x("def") == 35.0
        assert p.third_x("att") == 70.0
        assert p.third_x("mid") == 52.5

    def test_standard_pitch_constant(self):
        assert STANDARD_PITCH.length_m == 105.0
        assert FINAL_THIRD_X == 70.0


class TestNormToMeters:
    def test_normalized_values(self):
        assert norm_to_meters(0.0, 105.0) == 0.0
        assert norm_to_meters(1.0, 105.0) == 105.0
        assert norm_to_meters(0.5, 68.0) == 34.0

    def test_already_meters(self):
        assert norm_to_meters(105.0, 105.0) == 105.0
        assert norm_to_meters(50.0, 105.0) == 50.0

    def test_is_normalized(self):
        assert is_normalized(0.5)
        assert is_normalized(0.0)
        assert is_normalized(1.5)
        assert not is_normalized(2.0)
        assert not is_normalized(-0.1)


class TestClampPitch:
    def test_inside(self):
        assert clamp_pitch(50, 34) == (50, 34)

    def test_clamp_negative(self):
        assert clamp_pitch(-10, -5) == (0, 0)

    def test_clamp_overflow(self):
        assert clamp_pitch(200, 100) == (105, 68)


class TestPitchThird:
    def test_defensive_third(self):
        assert pitch_third(0) == "defensive"
        assert pitch_third(34.9) == "defensive"

    def test_middle_third(self):
        assert pitch_third(36) == "middle"
        assert pitch_third(52.5) == "middle"
        assert pitch_third(70) == "middle"
        assert pitch_third(35) == "middle"

    def test_attacking_third(self):
        assert pitch_third(70.1) == "attacking"
        assert pitch_third(105) == "attacking"


class TestHalfSpace:
    def test_central(self):
        assert half_space(50, 34) == "central"
        assert half_space(50, 30) == "central"
        assert half_space(50, 38) == "central"

    def test_left_halfspace(self):
        assert half_space(50, 5) == "left_halfspace"
        assert half_space(50, 13) == "left_halfspace"

    def test_right_halfspace(self):
        assert half_space(50, 55) == "right_halfspace"
        assert half_space(50, 60) == "right_halfspace"


class TestZoneLabel:
    def test_defensive_central(self):
        assert zone_label(10, 34) == "defensive_central"

    def test_attacking_left(self):
        assert zone_label(80, 10) == "attacking_left"

    def test_middle_right(self):
        assert zone_label(50, 55) == "middle_right"


class TestEuclideanDistanceM:
    def test_same_point(self):
        assert euclidean_distance_m(50, 34, 50, 34) == 0.0

    def test_horizontal_distance(self):
        d = euclidean_distance_m(50, 34, 80, 34)
        assert d == pytest.approx(30.0)

    def test_with_normalized_inputs(self):
        d = euclidean_distance_m(0.5, 0.5, 1.0, 0.5, PitchConfig(100, 50))
        assert d == pytest.approx(50.0)

    def test_custom_pitch(self):
        d = euclidean_distance_m(0, 0, 10, 0, PitchConfig(50, 50))
        assert d == pytest.approx(10.0)


class TestMetersToPixelFraction:
    def test_ten_meters(self):
        px = meters_to_pixel_fraction(10.0)
        assert px == pytest.approx(121.9, rel=0.01)

    def test_full_pitch(self):
        px = meters_to_pixel_fraction(105.0)
        assert px == pytest.approx(1280.0)

    def test_custom_viewport(self):
        px = meters_to_pixel_fraction(52.5, view_width_px=640)
        assert px == pytest.approx(320.0)
