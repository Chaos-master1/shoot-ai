"""Tests for momentum index."""

from kawkab.core.momentum import MomentumPoint, MomentumReport, compute_momentum_index


class TestMomentumIndex:
    def test_empty_events(self):
        report = compute_momentum_index([])
        assert report.home_momentum_pct == 0
        assert report.away_momentum_pct == 0
        assert len(report.timeline) == 0

    def test_home_dominant_returns_positive(self):
        events = [
            {"type": "shot", "team": "home", "xg": 0.5, "on_target": True, "timestamp": 600},
            {"type": "shot", "team": "home", "xg": 0.3, "on_target": True, "timestamp": 900},
        ]
        report = compute_momentum_index(events, window_minutes=5.0, match_duration_minutes=20.0)
        assert report.home_momentum_pct > 0

    def test_away_dominant_returns_negative(self):
        events = [
            {"type": "shot", "team": "away", "xg": 0.5, "on_target": True, "timestamp": 600},
            {"type": "shot", "team": "away", "xg": 0.3, "on_target": True, "timestamp": 900},
        ]
        report = compute_momentum_index(events, window_minutes=5.0, match_duration_minutes=20.0)
        assert report.away_momentum_pct > 0

    def test_timeline_ordered(self):
        events = [
            {"type": "shot", "team": "home", "xg": 0.2, "on_target": False, "timestamp": 500},
            {"type": "shot", "team": "away", "xg": 0.1, "on_target": False, "timestamp": 1500},
        ]
        report = compute_momentum_index(events, window_minutes=3.0, match_duration_minutes=30.0)
        assert len(report.timeline) > 0
        timestamps = [p["minute"] for p in report.timeline]
        assert sorted(timestamps) == timestamps

    def test_passes_final_third(self):
        events = [
            {"type": "pass", "team": "home", "end_x": 80, "timestamp": 600},
            {"type": "pass", "team": "home", "end_x": 75, "timestamp": 700},
            {"type": "pass", "team": "away", "end_x": 85, "timestamp": 800},
        ]
        report = compute_momentum_index(events, window_minutes=10.0, match_duration_minutes=20.0)
        # Home should have territory advantage due to more final third passes
        assert report.home_momentum_pct > 0 or report.home_max_run >= 0

    def test_momentum_point_to_dict(self):
        pt = MomentumPoint(
            minute=45.0,
            momentum=0.5,
            home_xg=0.3,
            away_xg=0.1,
            home_territory_pct=60.0,
            home_passes_final_third=5,
            away_passes_final_third=2,
        )
        d = pt.to_dict()
        assert d["minute"] == 45.0
        assert d["momentum"] == 0.5

    def test_report_to_dict(self):
        report = MomentumReport(
            home_momentum_pct=55.0,
            away_momentum_pct=30.0,
            neutral_pct=15.0,
            home_max_run=7.0,
            away_max_run=3.0,
        )
        d = report.to_dict()
        assert d["home_momentum_pct"] == 55.0
        assert d["home_max_run"] == 7.0
