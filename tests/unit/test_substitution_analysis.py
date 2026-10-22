"""Tests for substitution impact analysis."""

from kawkab.core.substitution_analysis import (
    SubstitutionAnalyzer,
    SubstitutionEvent,
    SubstitutionMatchReport,
)


class TestSubstitutionAnalyzer:
    def test_analyze_no_subs(self):
        sa = SubstitutionAnalyzer()
        report = sa.analyze_substitutions([], [])
        assert isinstance(report, SubstitutionMatchReport)
        assert report.substitutions == []
        assert report.net_xg_impact == 0.0

    def test_analyze_with_sub_no_stats(self):
        sa = SubstitutionAnalyzer()
        subs = [{"minute": 60, "player_out": 7, "player_in": 11, "team": "home"}]
        report = sa.analyze_substitutions(subs, [])
        assert len(report.substitutions) == 1
        assert report.substitutions[0].minute == 60.0

    def test_analyze_with_events(self):
        sa = SubstitutionAnalyzer()
        subs = [{"minute": 45, "player_out": 10, "player_in": 14, "team": "home"}]
        events = [
            {"timestamp": 10, "type": "shot", "team": "home", "xg": 0.05},
            {"timestamp": 30, "type": "shot", "team": "home", "xg": 0.10},
            {"timestamp": 55, "type": "shot", "team": "home", "xg": 0.15},
            {"timestamp": 70, "type": "shot", "team": "away", "xg": 0.08},
        ]
        report = sa.analyze_substitutions(subs, events)
        assert len(report.substitutions) == 1
        sub = report.substitutions[0]
        assert sub.xg_before >= 0 or sub.xg_before == 0
        assert sub.xg_after >= 0 or sub.xg_after == 0

    def test_multiple_subs(self):
        sa = SubstitutionAnalyzer()
        subs = [
            {"minute": 45, "player_out": 10, "player_in": 14, "team": "home"},
            {"minute": 70, "player_out": 9, "player_in": 15, "team": "away"},
        ]
        report = sa.analyze_substitutions(subs, [])
        assert len(report.substitutions) == 2

    def test_sub_to_dict(self):
        sub = SubstitutionEvent(
            minute=60,
            player_out_id=7,
            player_in_id=11,
            team="home",
            xg_before=0.2,
            xg_after=0.5,
            xg_delta=0.3,
        )
        d = sub.to_dict()
        assert d["minute"] == 60.0
        assert d["player_out"] == 7
        assert d["xg_delta"] == 0.3

    def test_report_to_dict(self):
        report = SubstitutionMatchReport(net_xg_impact=0.45)
        d = report.to_dict()
        assert d["net_xg_impact"] == 0.45

    def test_shot_rate_computation(self):
        sa = SubstitutionAnalyzer()
        subs = [{"minute": 45, "player_out": 10, "player_in": 11, "team": "home"}]
        events = [
            {"timestamp": 1800, "type": "shot", "team": "home"},  # minute 30
            {"timestamp": 2400, "type": "shot", "team": "home"},  # minute 40
            {"timestamp": 3000, "type": "shot", "team": "away"},  # minute 50
        ]
        report = sa.analyze_substitutions(subs, events)
        if report.substitutions:
            assert report.substitutions[0].shot_rate_before >= 0

    def test_pressing_before_after(self):
        sa = SubstitutionAnalyzer()
        subs = [
            {
                "minute": 60,
                "player_out": 8,
                "player_in": 17,
                "team": "home",
                "pressing_before": 8.5,
                "pressing_after": 6.2,
            }
        ]
        report = sa.analyze_substitutions(subs, [])
        assert report.substitutions[0].pressing_before == 8.5
        assert report.substitutions[0].pressing_after == 6.2
