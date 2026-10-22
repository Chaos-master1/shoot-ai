"""Tests for Off-Ball Value (OBV) — space creation, defense, support, decoy."""

import pytest

from kawkab.core.obv import OBVMatchReport, OBVPlayerResult, OffBallValuator


def _simple_frame(t, possession, ball, home_positions, away_positions):
    return {
        "timestamp": float(t),
        "possession": possession,
        "ball_pos": ball,
        "home_positions": home_positions,
        "away_positions": away_positions,
    }


class TestOffBallValuator:
    def test_empty_frames_returns_team_report(self):
        obv = OffBallValuator()
        report = obv.compute_obv([], team="home")
        assert isinstance(report, OBVMatchReport)
        assert report.team == "home"
        assert report.team_obv == 0.0
        assert report.players == {}

    def test_no_player_data_returns_zero(self):
        obv = OffBallValuator()
        frames = [
            _simple_frame(0.0, True, (50, 34), [], []),
            _simple_frame(1.0, True, (55, 34), [], []),
        ]
        report = obv.compute_obv(frames, team="home")
        assert report.team_obv == 0.0

    def test_static_player_zero_obv(self):
        obv = OffBallValuator()
        frames = [
            _simple_frame(0.0, True, (50, 34), [(50, 34, 1)], []),
            _simple_frame(1.0, True, (50, 34), [(50, 34, 1)], []),
            _simple_frame(2.0, True, (50, 34), [(50, 34, 1)], []),
        ]
        report = obv.compute_obv(frames, team="home")
        assert 1 in report.players
        assert report.players[1].total_obv == 0.0

    def test_fewer_than_3_frames_returns_zero(self):
        obv = OffBallValuator()
        frames = [
            _simple_frame(0.0, True, (50, 34), [(50, 34, 1)], []),
            _simple_frame(1.0, True, (50, 34), [(50, 34, 1)], []),
        ]
        report = obv.compute_obv(frames, team="home")
        assert report.players[1].total_obv == 0.0

    def test_movement_with_possession_gives_space_creation(self):
        obv = OffBallValuator()
        frames = []
        for i in range(6):
            x = 40.0 + i * 5.0  # 5 m/s >= SPACE_CREATION_SPEED_MS (4.0)
            frames.append(
                _simple_frame(
                    t=float(i),
                    possession=True,
                    ball=(20.0, 34.0),
                    home_positions=[(x, 34.0, 1)],
                    away_positions=[],
                )
            )
        report = obv.compute_obv(frames, team="home")
        assert report.players[1].space_creation_value > 0.0
        assert report.players[1].total_obv > 0.0

    def test_defensive_value_during_opponent_possession(self):
        obv = OffBallValuator()
        frames = []
        for i in range(4):
            frames.append(
                _simple_frame(
                    t=float(i),
                    possession=False,
                    ball=(60.0, 34.0),
                    home_positions=[(50.0, 34.0, 1)],
                    away_positions=[(70.0, 34.0, 10)],
                )
            )
        report = obv.compute_obv(frames, team="home")
        assert report.players[1].def_positioning_value > 0.0

    def test_support_value_during_possession(self):
        obv = OffBallValuator()
        frames = []
        for i in range(4):
            x = 80.0  # ahead of ball, good passing angle
            frames.append(
                _simple_frame(
                    t=float(i),
                    possession=True,
                    ball=(50.0, 34.0),
                    home_positions=[(x, 34.0, 1)],
                    away_positions=[],
                )
            )
        report = obv.compute_obv(frames, team="home")
        assert report.players[1].support_value > 0.0

    def test_decoy_run_value(self):
        obv = OffBallValuator()
        frames = []
        for i in range(5):
            x = 30.0 + i * 4.0
            frames.append(
                _simple_frame(
                    t=float(i),
                    possession=False,
                    ball=(50.0, 34.0),
                    home_positions=[(x, 34.0, 1)],
                    away_positions=[(x + 2.0, 34.0, 10)],
                )
            )
        report = obv.compute_obv(frames, team="home")
        assert report.players[1].decoy_run_value >= 0.0

    def test_team_obv_is_sum_of_players(self):
        obv = OffBallValuator()
        frames = []
        for i in range(4):
            frames.append(
                _simple_frame(
                    t=float(i),
                    possession=True,
                    ball=(20.0, 34.0),
                    home_positions=[(40.0 + i * 3.0, 34.0, 1), (35.0 + i * 2.0, 40.0, 2)],
                    away_positions=[],
                )
            )
        report = obv.compute_obv(frames, team="home")
        expected_sum = sum(r.total_obv for r in report.players.values())
        assert report.team_obv == pytest.approx(expected_sum, abs=1e-9)

    def test_to_dict_keys(self):
        obv = OffBallValuator()
        report = obv.compute_obv([], team="away")
        d = report.to_dict()
        assert "team" in d
        assert "players" in d
        assert "team_obv" in d

    def test_player_result_to_dict(self):
        pr = OBVPlayerResult(track_id=7, space_creation_value=0.5, total_obv=0.8)
        d = pr.to_dict()
        assert d["tid"] == 7
        assert d["space_creation"] == 0.5
        assert d["total"] == 0.8
        assert "def_pos" in d
        assert "support" in d
        assert "decoy" in d

    def test_obv_component_non_negative(self):
        obv = OffBallValuator()
        frames = []
        for i in range(4):
            frames.append(
                _simple_frame(
                    t=float(i),
                    possession=True,
                    ball=(50.0, 34.0),
                    home_positions=[(40.0, 34.0, 1)],
                    away_positions=[],
                )
            )
        report = obv.compute_obv(frames, team="home")
        for _pid, pr in report.players.items():
            assert pr.space_creation_value >= 0.0
            assert pr.def_positioning_value >= 0.0
            assert pr.support_value >= 0.0
            assert pr.decoy_run_value >= 0.0

    def test_space_creation_speed_threshold(self):
        obv = OffBallValuator()
        # slow movement below threshold should not count as space creation
        frames = []
        for i in range(4):
            x = 40.0 + i * 0.5  # 0.5 m/s
            frames.append(
                _simple_frame(
                    t=float(i),
                    possession=True,
                    ball=(20.0, 34.0),
                    home_positions=[(x, 34.0, 1)],
                    away_positions=[],
                )
            )
        report = obv.compute_obv(frames, team="home")
        assert report.players[1].space_creation_value == 0.0

    def test_compute_passing_lane_out_of_range(self):
        obv = OffBallValuator()
        val = obv._compute_passing_lane_value(
            (10.0, 34.0),
            (50.0, 34.0),
            {},
            1,
            0,
        )
        assert 0.0 <= val <= 1.0

    def test_compute_defensive_value_out_of_range(self):
        obv = OffBallValuator()
        val = obv._compute_defensive_value(
            (10.0, 34.0),
            (50.0, 34.0),
            {},
            0,
        )
        assert 0.0 <= val <= 1.0

    def test_compute_decoy_value_zero_when_no_opponents(self):
        obv = OffBallValuator()
        val = obv._compute_decoy_value(
            (50.0, 34.0),
            (30.0, 34.0),
            {},
            0,
        )
        assert val == 0.0

    def test_compute_decoy_value_with_defenders(self):
        obv = OffBallValuator()
        opp_trajs = {
            10: [(0.0, 50.0, 34.0), (1.0, 52.0, 34.0), (2.0, 54.0, 34.0)],
            20: [(0.0, 60.0, 34.0), (1.0, 58.0, 34.0), (2.0, 56.0, 34.0)],
        }
        val = obv._compute_decoy_value(
            (52.0, 34.0),
            (30.0, 34.0),
            opp_trajs,
            1,
        )
        assert val > 0.0

    def test_away_team_obv(self):
        obv = OffBallValuator()
        frames = []
        for i in range(4):
            frames.append(
                _simple_frame(
                    t=float(i),
                    possession=False,
                    ball=(50.0, 34.0),
                    home_positions=[(40.0, 34.0, 1)],
                    away_positions=[(60.0, 34.0, 10)],
                )
            )
        report = obv.compute_obv(frames, team="away")
        assert report.team == "away"
        assert len(report.players) > 0

    def test_malformed_position_entry_skipped(self):
        obv = OffBallValuator()
        frames = [
            {
                "timestamp": 0.0,
                "possession": True,
                "ball_pos": (50, 34),
                "home_positions": [(50, 34)],  # missing track_id
                "away_positions": [],
            },
            {
                "timestamp": 1.0,
                "possession": True,
                "ball_pos": (55, 34),
                "home_positions": [(55, 34)],
                "away_positions": [],
            },
        ]
        report = obv.compute_obv(frames, team="home")
        assert report.team_obv == 0.0
