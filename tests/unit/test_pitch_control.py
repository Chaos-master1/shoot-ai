"""Tests for Voronoi pitch control."""

from kawkab.core.pitch_control import (
    MatchPitchControl,
    PitchControlFrame,
    VoronoiPitchControl,
    WeightedPitchControl,
)


class TestVoronoiPitchControl:
    def test_single_team_full_control(self):
        pc = VoronoiPitchControl(grid_rows=10, grid_cols=15)
        home = [(52.5, 34.0)]
        away = []
        result = pc.compute_frame_control(home, away)
        assert result.home_control_pct > 95.0
        assert result.away_control_pct < 5.0

    def test_two_teams_split_control(self):
        pc = VoronoiPitchControl(grid_rows=10, grid_cols=15)
        home = [(26.0, 34.0)]
        away = [(79.0, 34.0)]
        result = pc.compute_frame_control(home, away)
        assert 30.0 < result.home_control_pct < 70.0
        assert 30.0 < result.away_control_pct < 70.0

    def test_no_players_returns_equal(self):
        pc = VoronoiPitchControl(grid_rows=10, grid_cols=15)
        result = pc.compute_frame_control([], [])
        assert result.home_control_pct == 50.0
        assert result.away_control_pct == 50.0

    def test_ball_zone_attribution_home(self):
        pc = VoronoiPitchControl(grid_rows=10, grid_cols=15)
        home = [(20, 34)]
        away = [(80, 34)]
        result = pc.compute_frame_control(home, away, ball_pos=(25, 34))
        assert result.ball_zone_team == "home"

    def test_ball_zone_attribution_away(self):
        pc = VoronoiPitchControl(grid_rows=10, grid_cols=15)
        home = [(20, 34)]
        away = [(80, 34)]
        result = pc.compute_frame_control(home, away, ball_pos=(75, 34))
        assert result.ball_zone_team == "away"

    def test_ball_zone_none_when_no_players(self):
        pc = VoronoiPitchControl(grid_rows=10, grid_cols=15)
        result = pc.compute_frame_control([], [], ball_pos=(50, 34))
        assert result.ball_zone_team is None

    def test_match_control_aggregates_empty(self):
        pc = VoronoiPitchControl()
        result = pc.compute_match_control([])
        assert result.avg_home_control == 0.0
        assert result.avg_away_control == 0.0

    def test_match_control_aggregates_frames(self):
        pc = VoronoiPitchControl(grid_rows=10, grid_cols=15)
        frames = [
            {
                "timestamp": 0.0,
                "home_positions": [(26, 34)],
                "away_positions": [(79, 34)],
                "ball_pos": None,
            },
            {
                "timestamp": 10.0,
                "home_positions": [(26, 34)],
                "away_positions": [(79, 34)],
                "ball_pos": (30, 34),
            },
        ]
        result = pc.compute_match_control(frames)
        assert 30.0 < result.avg_home_control < 70.0
        assert 30.0 < result.avg_away_control < 70.0
        assert len(result.frames) == 2

    def test_match_control_ball_distribution(self):
        pc = VoronoiPitchControl(grid_rows=10, grid_cols=15)
        frames = [
            {
                "timestamp": 0.0,
                "home_positions": [(20, 34)],
                "away_positions": [(80, 34)],
                "ball_pos": (90, 34),
            },
            {
                "timestamp": 10.0,
                "home_positions": [(20, 34)],
                "away_positions": [(80, 34)],
                "ball_pos": (10, 34),
            },
            {
                "timestamp": 20.0,
                "home_positions": [(20, 34)],
                "away_positions": [(80, 34)],
                "ball_pos": (50, 34),
            },
        ]
        result = pc.compute_match_control(frames)
        assert result.ball_in_away_third > 0

    def test_frame_to_dict(self):
        frame = PitchControlFrame(timestamp=10.0, home_control_pct=55.0, away_control_pct=45.0)
        d = frame.to_dict()
        assert d["t"] == 10.0
        assert d["h"] == 55.0
        assert d["a"] == 45.0

    def test_match_to_dict(self):
        mc = MatchPitchControl(avg_home_control=55.0, avg_away_control=45.0)
        d = mc.to_dict()
        assert d["avg_home_control"] == 55.0
        assert d["avg_away_control"] == 45.0


class TestWeightedPitchControl:
    def test_single_team_full_control(self):
        pc = WeightedPitchControl(grid_rows=10, grid_cols=15)
        home = [(52.5, 34.0)]
        away = []
        result = pc.compute_frame_control(home, away)
        assert result.home_control_pct > 90.0
        assert result.away_control_pct < 5.0

    def test_two_teams_split_control(self):
        pc = WeightedPitchControl(grid_rows=10, grid_cols=15)
        home = [(26.0, 34.0)]
        away = [(79.0, 34.0)]
        result = pc.compute_frame_control(home, away)
        assert 30.0 < result.home_control_pct < 70.0

    def test_no_players_returns_equal(self):
        pc = WeightedPitchControl(grid_rows=10, grid_cols=15)
        result = pc.compute_frame_control([], [])
        assert result.home_control_pct == 50.0
        assert result.away_control_pct == 50.0

    def test_speed_increases_influence_radius(self):
        pc = WeightedPitchControl(grid_rows=10, grid_cols=15, time_horizon=3.0)
        home = [(26.0, 34.0)]
        away = [(79.0, 34.0)]
        result = pc.compute_frame_control(home, away, home_speeds=[8.0], away_speeds=[2.0])
        fast_home = result.home_control_pct
        pc_slow = WeightedPitchControl(grid_rows=10, grid_cols=15, time_horizon=3.0)
        result_slow = pc_slow.compute_frame_control(
            home, away, home_speeds=[2.0], away_speeds=[8.0]
        )
        assert fast_home > result_slow.home_control_pct

    def test_ball_zone_attribution(self):
        pc = WeightedPitchControl(grid_rows=10, grid_cols=15)
        home = [(20, 34)]
        away = [(80, 34)]
        result = pc.compute_frame_control(home, away, ball_pos=(25, 34))
        assert result.ball_zone_team == "home"

    def test_match_control_aggregates(self):
        pc = WeightedPitchControl(grid_rows=10, grid_cols=15)
        frames = [
            {
                "timestamp": 0.0,
                "home_positions": [(26, 34)],
                "away_positions": [(79, 34)],
                "ball_pos": None,
            },
            {
                "timestamp": 10.0,
                "home_positions": [(26, 34)],
                "away_positions": [(79, 34)],
                "ball_pos": (30, 34),
            },
        ]
        result = pc.compute_match_control(frames)
        assert 30.0 < result.avg_home_control < 70.0
        assert len(result.frames) == 2

    def test_match_control_empty(self):
        pc = WeightedPitchControl()
        result = pc.compute_match_control([])
        assert result.avg_home_control == 0.0

    def test_voronoi_no_ball_in_any_frame(self):
        pc = VoronoiPitchControl(grid_rows=10, grid_cols=15)
        frames = [
            {
                "timestamp": 0.0,
                "home_positions": [(20, 34)],
                "away_positions": [(80, 34)],
                "ball_pos": None,
            },
            {
                "timestamp": 10.0,
                "home_positions": [(20, 34)],
                "away_positions": [(80, 34)],
                "ball_pos": None,
            },
        ]
        result = pc.compute_match_control(frames)
        assert result.ball_in_home_third == 0
        assert result.ball_in_away_third == 0
        assert result.ball_in_middle_third == 0

    def test_weighted_control_with_away_speeds_only(self):
        pc = WeightedPitchControl(grid_rows=10, grid_cols=15)
        home = [(30, 34), (40, 34)]
        away = [(70, 34)]
        result = pc.compute_frame_control(home, away, away_speeds=[8.0])
        assert 0.0 < result.home_control_pct < 100.0

    def test_weighted_control_varying_player_counts(self):
        pc = WeightedPitchControl(grid_rows=10, grid_cols=15)
        frames = [
            {
                "timestamp": 0.0,
                "home_positions": [(20, 34)],
                "away_positions": [(80, 34)],
                "ball_pos": (50, 34),
            },
            {
                "timestamp": 10.0,
                "home_positions": [(20, 34), (30, 40)],
                "away_positions": [(80, 34)],
                "ball_pos": (50, 34),
            },
        ]
        result = pc.compute_match_control(frames)
        assert len(result.frames) == 2

    def test_weighted_match_ball_third_distribution(self):
        pc = WeightedPitchControl(grid_rows=10, grid_cols=15)
        frames = [
            {
                "timestamp": 0.0,
                "home_positions": [(20, 34)],
                "away_positions": [(80, 34)],
                "ball_pos": (90, 34),
            },
            {
                "timestamp": 10.0,
                "home_positions": [(20, 34)],
                "away_positions": [(80, 34)],
                "ball_pos": (10, 34),
            },
        ]
        result = pc.compute_match_control(frames)
        assert result.ball_in_away_third > 0
        assert result.ball_in_home_third > 0
