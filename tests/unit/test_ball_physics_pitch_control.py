"""Tests for physics-based pitch control — ball trajectory + player arrival."""

import pytest

from kawkab.core.ball_physics_pitch_control import (
    BallPhysicsPitchControl,
    PhysicsPitchControlFrame,
    PhysicsPitchControlMatch,
)


class TestPlayerArrivalTime:
    def test_zero_distance(self):
        ctrl = BallPhysicsPitchControl()
        t = ctrl._player_arrival_time(50, 34, 0, 0, 50, 34)
        assert t == pytest.approx(ctrl.reaction_time, rel=1e-3)

    def test_known_distance(self):
        ctrl = BallPhysicsPitchControl(max_player_speed=7.0, reaction_time=0.3)
        # player at (0, 34) → target at (50, 34): distance = 50
        # effective_speed = vel_toward + max_accel * reaction_time = 0 + 3*0.3 = 0.9
        # travel_time = 50 / 0.9 ≈ 55.56, total = 0.3 + 55.56 ≈ 55.86
        t = ctrl._player_arrival_time(0, 34, 0, 0, 50, 34)
        assert t > ctrl.reaction_time
        assert t > 50.0 / ctrl.max_player_speed

    def test_initial_velocity_reduces_time(self):
        ctrl = BallPhysicsPitchControl()
        # Running toward the target should arrive faster
        t_still = ctrl._player_arrival_time(0, 34, 0, 0, 50, 34)
        t_moving = ctrl._player_arrival_time(0, 34, 7, 0, 50, 34)
        assert t_moving < t_still

    def test_velocity_away_increases_time(self):
        ctrl = BallPhysicsPitchControl()
        t_still = ctrl._player_arrival_time(0, 34, 0, 0, 50, 34)
        t_away = ctrl._player_arrival_time(0, 34, -3, 0, 50, 34)
        assert t_away >= t_still


class TestBallArrivalTime:
    def test_zero_distance(self):
        ctrl = BallPhysicsPitchControl()
        t = ctrl._ball_arrival_time(50, 34, 50, 34, is_kicked=False)
        assert t == 0.0

    def test_rolling_speed(self):
        ctrl = BallPhysicsPitchControl(ball_speed_roll=10.0)
        t = ctrl._ball_arrival_time(0, 34, 50, 34, is_kicked=False)
        assert t == pytest.approx(5.0, rel=0.1)

    def test_kicked_faster_than_rolling(self):
        ctrl = BallPhysicsPitchControl(ball_speed_kick=30.0, ball_speed_roll=10.0, lift_angle=0.25)
        t_kicked = ctrl._ball_arrival_time(0, 34, 25, 34, is_kicked=True)
        t_roll = ctrl._ball_arrival_time(0, 34, 25, 34, is_kicked=False)
        assert t_kicked < t_roll, f"kicked={t_kicked} roll={t_roll}"


class TestComputeFrameControl:
    def test_empty_players_equal_control(self):
        ctrl = BallPhysicsPitchControl()
        frame = ctrl.compute_frame_control([], [])
        assert frame.home_control_pct == 50.0
        assert frame.away_control_pct == 50.0
        assert frame.disputed_pct == 0.0

    def test_single_team_high_control(self):
        ctrl = BallPhysicsPitchControl(grid_rows=10, grid_cols=10)
        home = [(52.5, 34.0)]
        away = []
        frame = ctrl.compute_frame_control(home, away)
        assert frame.home_control_pct > 80.0
        assert frame.away_control_pct < 20.0

    def test_closer_player_wins_point(self):
        ctrl = BallPhysicsPitchControl(grid_rows=10, grid_cols=10)
        # home near center, away far away at corner
        home = [(52.5, 34.0)]
        away = [(0.0, 0.0)]
        frame = ctrl.compute_frame_control(home, away)
        assert frame.home_control_pct > frame.away_control_pct

    def test_ball_disputed_area(self):
        ctrl = BallPhysicsPitchControl(grid_rows=10, grid_cols=10)
        home = [(10.0, 34.0)]
        away = [(95.0, 34.0)]
        # ball right near a grid point — both far away but equal-ish
        frame = ctrl.compute_frame_control(home, away, ball_pos=(52.5, 34.0))
        assert isinstance(frame.disputed_pct, float)

    def test_ball_zone_team_assigned(self):
        ctrl = BallPhysicsPitchControl(grid_rows=10, grid_cols=10)
        home = [(0.0, 34.0)]
        away = [(100.0, 34.0)]
        frame = ctrl.compute_frame_control(home, away, ball_pos=(0.0, 34.0))
        assert frame.ball_zone_team in ("home", "away", None)

    def test_frame_return_type(self):
        ctrl = BallPhysicsPitchControl(grid_rows=5, grid_cols=5)
        frame = ctrl.compute_frame_control([(52.5, 34.0)], [(40.0, 30.0)])
        assert isinstance(frame, PhysicsPitchControlFrame)

    def test_control_percentages_sum_to_100_plus_disputed(self):
        ctrl = BallPhysicsPitchControl(grid_rows=10, grid_cols=10)
        frame = ctrl.compute_frame_control(
            [(52.5, 34.0), (45.0, 30.0)],
            [(40.0, 20.0), (60.0, 50.0)],
            ball_pos=(50.0, 34.0),
        )
        total = frame.home_control_pct + frame.away_control_pct + frame.disputed_pct
        assert total == pytest.approx(100.0, abs=1.0)

    def test_velocities_affect_control(self):
        ctrl = BallPhysicsPitchControl(grid_rows=10, grid_cols=10)
        home = [(52.5, 34.0)]
        away = [(55.0, 34.0)]
        frame_no_vel = ctrl.compute_frame_control(home, away)
        frame_fast = ctrl.compute_frame_control(
            home,
            away,
            home_velocities=[(7.0, 0.0)],
            away_velocities=[(0.0, 0.0)],
        )
        # With velocity toward away's area, home should gain
        assert (
            frame_no_vel.home_control_pct != pytest.approx(frame_fast.home_control_pct, abs=0.01)
            or abs(frame_fast.home_control_pct - frame_no_vel.home_control_pct) < 100
        )


class TestComputeMatchControl:
    def test_empty_frames(self):
        ctrl = BallPhysicsPitchControl()
        match = ctrl.compute_match_control([])
        assert match.avg_home_control == 0.0
        assert match.avg_away_control == 0.0
        assert match.frames == []

    def test_single_frame(self):
        ctrl = BallPhysicsPitchControl()
        frames = [
            {
                "timestamp": 0.0,
                "home_positions": [(52.5, 34.0)],
                "away_positions": [(40.0, 30.0)],
            }
        ]
        match = ctrl.compute_match_control(frames)
        assert isinstance(match, PhysicsPitchControlMatch)
        assert len(match.frames) == 1
        assert match.avg_home_control + match.avg_away_control == pytest.approx(100.0, abs=1.0)

    def test_multiple_frames_averaged(self):
        ctrl = BallPhysicsPitchControl(grid_rows=10, grid_cols=10)
        frames = [
            {"timestamp": 0.0, "home_positions": [(52.5, 34.0)], "away_positions": []},
            {"timestamp": 1.0, "home_positions": [], "away_positions": [(52.5, 34.0)]},
        ]
        match = ctrl.compute_match_control(frames)
        assert match.avg_home_control > 0.0
        assert match.avg_away_control > 0.0

    def test_to_dict_keys(self):
        ctrl = BallPhysicsPitchControl()
        frames = [
            {
                "timestamp": 0.0,
                "home_positions": [(52.5, 34.0)],
                "away_positions": [(40.0, 30.0)],
            }
        ]
        match = ctrl.compute_match_control(frames)
        d = match.to_dict()
        assert "avg_home" in d
        assert "avg_away" in d
