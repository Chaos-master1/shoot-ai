"""Tests for game constants."""

import pytest

from kawkab.core.game_constants import GAME, GameConstants


class TestGameConstants:
    def test_default_values(self):
        g = GameConstants()
        assert g.PITCH_LENGTH_M == 105.0
        assert g.PITCH_WIDTH_M == 68.0
        assert g.PRESS_THRESHOLD_M == 5.0
        assert g.POSSESSION_CHANGE_DIST_M == 2.0
        assert g.FINAL_THIRD_PCT == 2 / 3
        assert g.CARRY_PIXEL_TO_METER_RATIO == 0.015

    def test_custom_values(self):
        g = GameConstants(PITCH_LENGTH_M=90.0, PRESS_THRESHOLD_M=3.0)
        assert g.PITCH_LENGTH_M == 90.0
        assert g.PRESS_THRESHOLD_M == 3.0
        assert g.PITCH_WIDTH_M == 68.0  # default

    def test_is_frozen(self):
        g = GameConstants()
        with pytest.raises(AttributeError):
            g.PITCH_LENGTH_M = 100.0

    def test_momentum_window_default(self):
        g = GameConstants()
        assert g.MOMENTUM_WINDOW_MIN == 5.0

    def test_new_constants(self):
        g = GameConstants()
        assert g.HEATMAP_GRID_COLS == 40
        assert g.HEATMAP_GRID_ROWS == 60
        assert g.HEATMAP_KERNEL_SIZE == 3
        assert g.PPDA_ATTACKING_THRESHOLD_PCT == 0.4
        assert g.MOMENTUM_WINDOW_MINUTES == 5
        assert g.PROGRESSIVE_MIN_PROGRESSION_RATIO == 0.25
        assert g.PROGRESSIVE_ATTACKING_THIRD_FRACTION == 0.6
        assert g.PROGRESSIVE_MIN_CARRY_M == 5.0
        assert g.PRESSING_TRAP_ZONE_BOUNDARY_PCT == (0.25, 0.75)

    def test_singleton(self):
        assert GAME.PITCH_LENGTH_M == 105.0
        assert GAME.PRESS_THRESHOLD_M == 5.0
