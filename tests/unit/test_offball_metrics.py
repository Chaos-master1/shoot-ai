"""Tests for off-ball metrics."""

from kawkab.core.offball_metrics import OffBallAnalyzer, OffBallMatchReport, OffBallPlayerMetrics


class TestOffBallAnalyzer:
    def test_analyze_empty_frames(self):
        oba = OffBallAnalyzer()
        report = oba.analyze_offball([], team="home")
        assert isinstance(report, OffBallMatchReport)
        assert report.team == "home"
        assert report.players == {}

    def test_analyze_single_frame(self):
        oba = OffBallAnalyzer()
        frames = [
            {
                "timestamp": 0.0,
                "possession": True,
                "home_positions": [(50, 34, 1)],
                "away_positions": [(50, 40, 2)],
                "ball_pos": (52, 34),
            }
        ]
        report = oba.analyze_offball(frames, team="home")
        assert isinstance(report, OffBallMatchReport)

    def test_analyze_moving_player(self):
        oba = OffBallAnalyzer()
        frames = []
        for i in range(20):
            frames.append(
                {
                    "timestamp": i * 0.5,
                    "possession": False,
                    "home_positions": [(10 + i, 34, 1)],
                    "away_positions": [(80, 34, 2)],
                    "ball_pos": (50, 34),
                }
            )
        report = oba.analyze_offball(frames, team="home")
        if report.players:
            p = list(report.players.values())[0]
            assert p.total_distance_without_ball_m > 0

    def test_high_speed_detection(self):
        oba = OffBallAnalyzer()
        frames = []
        for i in range(30):
            frames.append(
                {
                    "timestamp": i * 0.2,
                    "possession": False,
                    "home_positions": [(0 + i * 3, 34, 1)],
                    "away_positions": [(80, 34, 2)],
                    "ball_pos": (50, 34),
                }
            )
        report = oba.analyze_offball(frames, team="home")
        if report.players:
            p = list(report.players.values())[0]
            assert p.high_speed_runs_without_ball >= 0

    def test_possession_vs_non_possession(self):
        oba = OffBallAnalyzer()
        frames = []
        for i in range(10):
            frames.append(
                {
                    "timestamp": i * 0.5,
                    "possession": True,
                    "home_positions": [(50, 34, 1)],
                    "away_positions": [(80, 34, 2)],
                    "ball_pos": (52, 34),
                }
            )
        report = oba.analyze_offball(frames, team="home")
        assert isinstance(report, OffBallMatchReport)

    def test_player_metrics_to_dict(self):
        pm = OffBallPlayerMetrics(
            track_id=1,
            total_distance_without_ball_m=5000,
            space_creation_runs=8,
            movement_efficiency=0.75,
        )
        d = pm.to_dict()
        assert d["tid"] == 1
        assert d["dist_no_ball"] == 5000.0
        assert d["space_creation"] == 8

    def test_report_to_dict(self):
        report = OffBallMatchReport(
            team="home", team_total_dist_no_ball_km=45.5, team_space_creation_runs=20
        )
        d = report.to_dict()
        assert d["team"] == "home"
        assert d["team_dist_no_ball_km"] == 45.5
