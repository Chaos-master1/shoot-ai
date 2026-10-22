"""Tests for defensive actions and final third entry analysis."""

from kawkab.core.defensive_actions import (
    DefensiveAction,
    DefensiveHeatmap,
    FinalThirdReport,
    analyze_final_third_entries,
    build_defensive_heatmap,
    extract_defensive_actions,
)


class TestDefensiveActions:
    def test_extract_empty(self):
        actions = extract_defensive_actions([], "home")
        assert len(actions) == 0

    def test_extract_tackle(self):
        events = [
            {
                "type": "tackle",
                "team": "home",
                "start_x": 40,
                "start_y": 30,
                "completed": True,
                "timestamp": 600,
            },
        ]
        actions = extract_defensive_actions(events, "home")
        assert len(actions) == 1
        assert actions[0].action_type == "tackle"
        assert actions[0].success is True

    def test_extract_interception(self):
        events = [
            {
                "type": "interception",
                "team": "away",
                "start_x": 50,
                "start_y": 34,
                "completed": True,
                "timestamp": 900,
            },
        ]
        actions = extract_defensive_actions(events, "away")
        assert len(actions) == 1
        assert actions[0].action_type == "interception"

    def test_extract_pressure(self):
        events = [
            {
                "type": "pressure",
                "team": "home",
                "start_x": 60,
                "start_y": 34,
                "completed": False,
                "timestamp": 1200,
            },
        ]
        actions = extract_defensive_actions(events, "home")
        assert len(actions) == 1
        assert actions[0].action_type == "pressure"

    def test_build_heatmap_empty(self):
        hm = build_defensive_heatmap([])
        assert hm.total_actions == 0

    def test_build_heatmap_single(self):
        actions = [DefensiveAction(x=52.5, y=34.0, team="home", action_type="tackle")]
        hm = build_defensive_heatmap(actions, grid_rows=5, grid_cols=5)
        assert hm.total_actions == 1
        assert hm.team == "home"
        assert len(hm.grid) == 5
        assert len(hm.grid[0]) == 5

    def test_action_to_dict(self):
        a = DefensiveAction(
            timestamp=600, team="home", action_type="tackle", x=40, y=30, success=True
        )
        d = a.to_dict()
        assert d["action_type"] == "tackle"

    def test_heatmap_to_dict(self):
        hm = DefensiveHeatmap(team="home", grid=[[0.5]], total_actions=1)
        d = hm.to_dict()
        assert d["team"] == "home"
        assert d["total_actions"] == 1


class TestFinalThirdEntry:
    def test_empty_events(self):
        report = analyze_final_third_entries([])
        assert report.home_entries == 0
        assert report.away_entries == 0

    def test_home_entry_via_pass(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "start_x": 60,
                "end_x": 75,
                "end_y": 30,
                "completed": True,
                "timestamp": 600,
            },
        ]
        report = analyze_final_third_entries(events)
        assert report.home_entries == 1
        assert report.home_success_pct == 100.0

    def test_away_entry_via_cross(self):
        events = [
            {
                "type": "pass",
                "team": "away",
                "pass_type": "cross",
                "start_x": 50,
                "end_x": 80,
                "end_y": 34,
                "completed": False,
                "timestamp": 900,
            },
        ]
        report = analyze_final_third_entries(events)
        assert report.away_entries == 1
        assert report.away_by_type.get("cross", 0) == 1

    def test_entry_not_counted_when_start_in_final_third(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "start_x": 80,
                "end_x": 90,
                "completed": True,
                "timestamp": 600,
            },
        ]
        report = analyze_final_third_entries(events)
        assert report.home_entries == 0

    def test_success_percentage(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "start_x": 60,
                "end_x": 75,
                "completed": True,
                "timestamp": 600,
            },
            {
                "type": "pass",
                "team": "home",
                "start_x": 60,
                "end_x": 75,
                "completed": False,
                "timestamp": 700,
            },
        ]
        report = analyze_final_third_entries(events)
        assert report.home_entries == 2
        assert report.home_success_pct == 50.0

    def test_report_to_dict(self):
        report = FinalThirdReport(home_entries=10, away_entries=5)
        d = report.to_dict()
        assert d["home_entries"] == 10
        assert d["total_entries"] == 15
