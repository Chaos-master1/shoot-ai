"""Tests for Expected Possession Value (EPV) module."""

from kawkab.core.epv import (
    EPVModel,
    EPVReport,
    EPVResult,
    _extract_possessions,
    _possession_switching_events,
    _to_zone,
)


class TestUtil:
    def test_zone_mapping(self):
        assert _to_zone(0, 0) == (0, 0)
        assert _to_zone(105, 68) == (15, 11)

    def test_possession_switching_set(self):
        s = _possession_switching_events()
        assert "tackle" in s
        assert "interception" in s

    def test_extract_empty(self):
        assert _extract_possessions([]) == []

    def test_extract_single_possession(self):
        events = [
            {"type": "pass", "timestamp": 1.0, "team": "home", "x": 50, "y": 34},
            {"type": "pass", "timestamp": 2.0, "team": "home", "x": 60, "y": 34},
        ]
        phases = _extract_possessions(events)
        assert len(phases) == 1

    def test_extract_switch_on_tackle(self):
        events = [
            {"type": "pass", "timestamp": 1.0, "team": "home", "x": 50, "y": 34},
            {"type": "tackle", "timestamp": 2.0, "team": "away", "x": 55, "y": 34},
            {"type": "pass", "timestamp": 3.0, "team": "away", "x": 40, "y": 34},
        ]
        phases = _extract_possessions(events)
        assert len(phases) >= 2

    def test_extract_switch_on_shot(self):
        events = [
            {"type": "pass", "timestamp": 1.0, "team": "home", "x": 50, "y": 34},
            {"type": "shot", "timestamp": 2.0, "team": "away", "x": 80, "y": 34, "is_goal": False},
        ]
        phases = _extract_possessions(events)
        assert len(phases) >= 2


class TestEPVModel:
    def test_empty_possession(self):
        model = EPVModel()
        result = model.compute_possession_epv([])
        assert result.value == 0.0

    def test_single_pass(self):
        model = EPVModel()
        events = [{"type": "pass", "timestamp": 1.0, "team": "home", "x": 50, "y": 34}]
        result = model.compute_possession_epv(events)
        assert -0.5 <= result.value <= 1.5
        assert result.team == "home"

    def test_goal_possession_higher(self):
        model = EPVModel()
        goal_events = [
            {"type": "pass", "timestamp": 1.0, "team": "home", "x": 50, "y": 34},
            {"type": "shot", "timestamp": 2.0, "team": "home", "x": 95, "y": 34, "is_goal": True},
        ]
        no_goal_events = [
            {"type": "pass", "timestamp": 1.0, "team": "home", "x": 50, "y": 34},
        ]
        g = model.compute_possession_epv(goal_events)
        ng = model.compute_possession_epv(no_goal_events)
        assert g.value > ng.value

    def test_progressive_possession_higher(self):
        model = EPVModel()
        deep = [
            {
                "type": "pass",
                "timestamp": 1.0,
                "team": "home",
                "x": 30,
                "y": 34,
                "end_x": 90,
                "end_y": 34,
            }
        ]
        shallow = [
            {
                "type": "pass",
                "timestamp": 1.0,
                "team": "home",
                "x": 30,
                "y": 34,
                "end_x": 35,
                "end_y": 34,
            }
        ]
        d = model.compute_possession_epv(deep)
        s = model.compute_possession_epv(shallow)
        assert d.value >= s.value

    def test_shot_bonus_applied(self):
        model = EPVModel()
        with_shot = [
            {"type": "pass", "timestamp": 1.0, "team": "home", "x": 50, "y": 34},
            {"type": "shot", "timestamp": 2.0, "team": "home", "x": 90, "y": 34, "is_goal": False},
        ]
        no_shot = [
            {"type": "pass", "timestamp": 1.0, "team": "home", "x": 50, "y": 34},
        ]
        ws = model.compute_possession_epv(with_shot)
        ns = model.compute_possession_epv(no_shot)
        assert ws.value > ns.value

    def test_zone_value_higher_near_goal(self):
        model = EPVModel()
        near = model._zone_value(95, 34)
        far = model._zone_value(30, 34)
        assert near > far


class TestEPVReport:
    def test_empty_report(self):
        model = EPVModel()
        report = model.compute_match_epv([])
        assert report.total_possessions == 0
        assert report.home_total == 0
        assert report.away_total == 0

    def test_single_team_report(self):
        model = EPVModel()
        events = [
            {"type": "pass", "timestamp": 1.0, "team": "home", "x": 50, "y": 34},
            {"type": "pass", "timestamp": 2.0, "team": "home", "x": 60, "y": 34},
        ]
        report = model.compute_match_epv(events)
        assert report.total_possessions == 1
        assert report.home_total > 0
        assert report.away_total == 0

    def test_both_teams(self):
        model = EPVModel()
        events = [
            {"type": "pass", "timestamp": 1.0, "team": "home", "x": 50, "y": 34},
            {"type": "tackle", "timestamp": 2.0, "team": "away", "x": 55, "y": 34},
            {"type": "pass", "timestamp": 3.0, "team": "away", "x": 40, "y": 34},
        ]
        report = model.compute_match_epv(events)
        assert report.home_total > 0
        assert report.away_total >= 0
        assert report.total_possessions >= 2

    def test_report_to_dict(self):
        report = EPVReport(home_total=1.5, away_total=0.5, total_possessions=10)
        d = report.to_dict()
        assert d["home_total"] == 1.5
        assert d["total_possessions"] == 10

    def test_per_possession_values(self):
        model = EPVModel()
        events = [
            {"type": "pass", "timestamp": 1.0, "team": "home", "x": 50, "y": 34},
            {"type": "shot", "timestamp": 2.0, "team": "home", "x": 90, "y": 34, "is_goal": False},
            {"type": "tackle", "timestamp": 3.0, "team": "away", "x": 50, "y": 34},
            {"type": "pass", "timestamp": 4.0, "team": "away", "x": 35, "y": 34},
        ]
        report = model.compute_match_epv(events)
        assert report.home_per_possession > 0
        assert report.away_per_possession >= 0

    def test_epv_result_to_dict(self):
        r = EPVResult(value=0.35, events=3, team="home", has_shot=True, is_goal=False)
        d = r.to_dict()
        assert d["value"] == 0.35
        assert d["has_shot"] is True

    def test_possessions_list_present(self):
        model = EPVModel()
        events = [
            {"type": "pass", "timestamp": 1.0, "team": "home", "x": 50, "y": 34},
        ]
        report = model.compute_match_epv(events)
        assert len(report.possessions) == 1
        assert isinstance(report.possessions[0], EPVResult)
