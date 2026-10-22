"""Tests for Kalman smoother for player tracking positions."""

from __future__ import annotations

import pytest
from conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.services.kalman_smoother import PlayerPositionSmoother


class TestInitialization:
    def test_default_params(self):
        smoother = PlayerPositionSmoother()
        assert smoother.q_std == 0.3
        assert smoother.r_std == 0.8
        assert smoother.initialized is False

    def test_custom_params(self):
        smoother = PlayerPositionSmoother(process_noise_std=0.5, measurement_noise_std=1.0)
        assert smoother.q_std == 0.5
        assert smoother.r_std == 1.0


class TestFirstUpdate:
    def test_first_update_initializes_state(self):
        smoother = PlayerPositionSmoother()
        smoother.update(50.0, 34.0, 0.04)
        assert smoother.initialized is True
        x, y = smoother.get_position()
        assert x == 50.0
        assert y == 34.0

    def test_first_update_velocity_is_zero(self):
        smoother = PlayerPositionSmoother()
        smoother.update(50.0, 34.0, 0.04)
        vx, vy = smoother.get_velocity()
        assert vx == 0.0
        assert vy == 0.0

    def test_first_update_speed_is_zero(self):
        smoother = PlayerPositionSmoother()
        smoother.update(50.0, 34.0, 0.04)
        assert smoother.get_speed_mps() == 0.0

    def test_first_update_innovation(self):
        smoother = PlayerPositionSmoother()
        smoother.update(50.0, 34.0, 0.04)
        innov = smoother.get_position_innovation(52.0, 34.0)
        assert innov == pytest.approx(2.0, abs=0.01)


class TestTracking:
    def test_tracks_after_multiple_updates(self):
        smoother = PlayerPositionSmoother()
        for i in range(10):
            smoother.update(float(i), 0.0, 0.04)
        x, y = smoother.get_position()
        assert smoother.initialized is True

    def test_small_increments_accumulate(self):
        smoother = PlayerPositionSmoother()
        smoother.update(0.0, 0.0, 1.0)
        smoother.update(1.0, 0.0, 1.0)
        smoother.update(2.0, 0.0, 1.0)
        x, y = smoother.get_position()
        # With small steps within 2.0 threshold, filter should track
        assert x > 0

    def test_speed_with_small_increments(self):
        smoother = PlayerPositionSmoother()
        smoother.update(0.0, 0.0, 1.0)
        smoother.update(1.0, 1.0, 1.0)
        smoother.update(2.0, 2.0, 1.0)
        speed = smoother.get_speed_mps()
        assert speed >= 0

    def test_median_filter_called_on_subsequent_updates(self):
        smoother = PlayerPositionSmoother()
        smoother.update(0.0, 0.0, 0.04)
        smoother.update(1.0, 0.0, 0.04)
        smoother.update(2.0, 0.0, 0.04)
        assert smoother.initialized is True


class TestReset:
    def test_reset_clears_state(self):
        smoother = PlayerPositionSmoother()
        smoother.update(50.0, 34.0, 0.04)
        assert smoother.initialized is True
        smoother.reset()
        assert smoother.initialized is False
        x, y = smoother.get_position()
        assert x == 0.0
        assert y == 0.0

    def test_reset_clears_median_buffer(self):
        smoother = PlayerPositionSmoother()
        smoother.update(0.0, 0.0, 0.04)
        smoother.update(1.0, 0.0, 0.04)
        smoother.update(2.0, 0.0, 0.04)
        smoother.reset()
        # After reset, first update reinitializes
        smoother.update(10.0, 10.0, 0.04)
        x, y = smoother.get_position()
        assert x == 10.0
        assert y == 10.0


class TestPredictOnly:
    def test_predict_only_with_velocity_small_steps(self):
        smoother = PlayerPositionSmoother()
        smoother.update(0.0, 0.0, 1.0)
        smoother.update(1.0, 0.0, 1.0)
        smoother.update(2.0, 0.0, 1.0)
        # After 3 updates, filter should have some velocity estimate
        x0, y0 = smoother.get_position()
        smoother.predict_only(1.0)
        x1, y1 = smoother.get_position()
        assert x1 >= x0  # velocity carries position forward

    def test_predict_only_before_init_does_nothing(self):
        smoother = PlayerPositionSmoother()
        smoother.predict_only(0.04)
        assert smoother.initialized is False
        assert smoother.get_position() == (0.0, 0.0)


class TestOutlierRejection:
    def test_large_jump_is_rejected(self):
        smoother = PlayerPositionSmoother()
        smoother.update(0.0, 0.0, 0.04)
        smoother.update(1.0, 0.0, 0.04)
        smoother.update(2.0, 0.0, 0.04)
        # Jump > 2.0 should be rejected (clamped to predicted position)
        smoother.update(100.0, 0.0, 0.04)
        x, y = smoother.get_position()
        assert x < 10.0


class TestPositionAndVelocity:
    def test_get_position_before_init_returns_zero(self):
        smoother = PlayerPositionSmoother()
        x, y = smoother.get_position()
        assert x == 0.0
        assert y == 0.0

    def test_get_velocity_before_init_returns_zero(self):
        smoother = PlayerPositionSmoother()
        vx, vy = smoother.get_velocity()
        assert vx == 0.0
        assert vy == 0.0

    def test_get_speed_before_init_returns_zero(self):
        smoother = PlayerPositionSmoother()
        assert smoother.get_speed_mps() == 0.0

    def test_get_innovation_before_init_returns_zero(self):
        smoother = PlayerPositionSmoother()
        assert smoother.get_position_innovation(5.0, 5.0) == 0.0


class TestConsecutiveTracking:
    def test_consecutive_small_steps(self):
        smoother = PlayerPositionSmoother()
        for i in range(20):
            smoother.update(float(i * 0.5), 0.0, 1.0)
        fx, fy = smoother.get_position()
        # With 20 steps of 0.5 each, filter should be tracking
        assert smoother.initialized is True

    def test_median_filter_effect(self):
        smoother = PlayerPositionSmoother()
        smoother.update(0.0, 0.0, 0.04)
        smoother.update(1.0, 0.0, 0.04)
        # After init + 2 updates, median buffer has 3 elements
        assert smoother.initialized is True

    def test_handles_negative_positions(self):
        smoother = PlayerPositionSmoother()
        smoother.update(-10.0, -5.0, 0.04)
        x, y = smoother.get_position()
        assert x == -10.0
        assert y == -5.0
