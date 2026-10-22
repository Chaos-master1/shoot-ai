"""Tests for match timeline / xG flow."""

from kawkab.core.match_timeline import XGFlowReport, compute_xg_timeline


class TestXGFlow:
    def test_empty_events(self):
        report = compute_xg_timeline([], 90)
        assert report.home_total == 0
        assert len(report.points) == 2  # kick-off + full-time

    def test_single_shot(self):
        events = [{"type": "shot", "team": "home", "xg": 0.5, "timestamp": 1800, "is_goal": False}]
        report = compute_xg_timeline(events, 90)
        assert report.home_total == 0.5
        assert report.away_total == 0
        assert len(report.points) == 3

    def test_goal_increments_count(self):
        events = [{"type": "shot", "team": "home", "xg": 0.3, "timestamp": 2700, "is_goal": True}]
        report = compute_xg_timeline(events, 90)
        assert report.home_goals == 1
        assert report.home_total == 0.3

    def test_cumulative_values(self):
        events = [
            {"type": "shot", "team": "home", "xg": 0.2, "timestamp": 600, "is_goal": False},
            {"type": "shot", "team": "home", "xg": 0.6, "timestamp": 1800, "is_goal": False},
        ]
        report = compute_xg_timeline(events, 90)
        # Check last point
        last = report.points[-1]
        assert last["home_cumulative"] == 0.8

    def test_to_dict(self):
        report = XGFlowReport(home_total=1.5, away_total=0.5, home_goals=2, away_goals=0)
        d = report.to_dict()
        assert d["home_total"] == 1.5
        assert d["home_goals"] == 2
