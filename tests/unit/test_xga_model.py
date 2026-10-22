"""Tests for Expected Goals Against (xGA) model."""

import pytest

from kawkab.core.xga_model import (
    ExpectedGoalsAgainstModel,
    XGAReport,
    _to_zone,
)


class TestXgaUtil:
    def test_to_zone_zero(self):
        zx, zy = _to_zone(0, 0, 5, 4)
        assert zx == 0
        assert zy == 0

    def test_to_zone_corner(self):
        zx, zy = _to_zone(105, 68, 5, 4)
        assert zx == 4
        assert zy == 3

    def test_to_zone_mid(self):
        zx, zy = _to_zone(52.5, 34, 5, 4)
        assert zx == 2
        assert zy == 2


class TestXgaCompute:
    def test_three_shots(self):
        model = ExpectedGoalsAgainstModel()
        events = [
            {
                "type": "shot",
                "team": "away",
                "xg": 0.5,
                "start_x": 90,
                "start_y": 34,
                "is_goal": False,
            },
            {
                "type": "shot",
                "team": "away",
                "xg": 0.5,
                "start_x": 85,
                "start_y": 20,
                "is_goal": False,
            },
            {
                "type": "shot",
                "team": "away",
                "xg": 0.5,
                "start_x": 80,
                "start_y": 50,
                "is_goal": True,
            },
        ]
        xga = model.compute_xga(events, "home")
        assert xga == pytest.approx(1.5, rel=0.01)

    def test_no_shots_faced(self):
        model = ExpectedGoalsAgainstModel()
        events = [
            {"type": "pass", "team": "home"},
            {"type": "pass", "team": "away"},
        ]
        xga = model.compute_xga(events, "home")
        assert xga == 0.0

    def test_own_shots_not_counted(self):
        model = ExpectedGoalsAgainstModel()
        events = [
            {
                "type": "shot",
                "team": "home",
                "xg": 0.8,
                "start_x": 90,
                "start_y": 34,
                "is_goal": False,
            },
        ]
        xga = model.compute_xga(events, "home")
        assert xga == 0.0

    def test_empty_events(self):
        model = ExpectedGoalsAgainstModel()
        xga = model.compute_xga([], "home")
        assert xga == 0.0


class TestXgaByZone:
    def test_zone_distribution(self):
        model = ExpectedGoalsAgainstModel()
        events = [
            {
                "type": "shot",
                "team": "away",
                "xg": 0.3,
                "start_x": 90,
                "start_y": 10,
                "is_goal": False,
            },
            {
                "type": "shot",
                "team": "away",
                "xg": 0.6,
                "start_x": 95,
                "start_y": 55,
                "is_goal": False,
            },
            {
                "type": "shot",
                "team": "away",
                "xg": 0.1,
                "start_x": 50,
                "start_y": 34,
                "is_goal": False,
            },
        ]
        zones = model.compute_xga_by_zone(events, "home")
        assert len(zones) >= 2
        total = sum(zones.values())
        assert total == pytest.approx(1.0, rel=0.01)

    def test_zone_no_shots(self):
        model = ExpectedGoalsAgainstModel()
        zones = model.compute_xga_by_zone([], "home")
        assert zones == {}


class TestXgaByType:
    def test_type_split(self):
        model = ExpectedGoalsAgainstModel()
        events = [
            {
                "type": "shot",
                "team": "away",
                "xg": 0.5,
                "body_part": "head",
                "start_x": 90,
                "start_y": 34,
                "metadata": {},
                "is_goal": False,
            },
            {
                "type": "shot",
                "team": "away",
                "xg": 0.3,
                "body_part": "right_foot",
                "start_x": 85,
                "start_y": 34,
                "metadata": {},
                "is_goal": False,
            },
        ]
        type_bd, sit_bd = model.compute_xga_by_type(events, "home")
        assert "head" in type_bd
        assert "right_foot" in type_bd
        assert type_bd["head"] == pytest.approx(0.5, rel=0.01)

    def test_type_situation_split(self):
        model = ExpectedGoalsAgainstModel()
        events = [
            {
                "type": "shot",
                "team": "away",
                "xg": 0.4,
                "body_part": "right_foot",
                "start_x": 90,
                "start_y": 34,
                "metadata": {"set_piece": "corner_kick"},
                "is_goal": False,
            },
            {
                "type": "shot",
                "team": "away",
                "xg": 0.76,
                "body_part": "right_foot",
                "start_x": 95,
                "start_y": 34,
                "metadata": {"set_piece": "penalty"},
                "is_goal": True,
            },
        ]
        type_bd, sit_bd = model.compute_xga_by_type(events, "home")
        assert "set_piece" in sit_bd or "penalty" in sit_bd
        total_sit = sum(sit_bd.values())
        assert total_sit == pytest.approx(1.16, rel=0.01)

    def test_type_empty(self):
        model = ExpectedGoalsAgainstModel()
        type_bd, sit_bd = model.compute_xga_by_type([], "home")
        assert type_bd == {}
        assert sit_bd == {}


class TestXgaSavePct:
    def test_save_pct_calculation(self):
        model = ExpectedGoalsAgainstModel()
        events = [
            {
                "type": "shot",
                "team": "away",
                "xg": 1.0,
                "start_x": 90,
                "start_y": 34,
                "is_goal": True,
            },
            {
                "type": "shot",
                "team": "away",
                "xg": 1.0,
                "start_x": 85,
                "start_y": 20,
                "is_goal": True,
            },
            {
                "type": "shot",
                "team": "away",
                "xg": 1.0,
                "start_x": 80,
                "start_y": 50,
                "is_goal": False,
            },
        ]
        sp = model.compute_xga_save_pct(events, "home")
        assert sp == pytest.approx(0.333, rel=0.01)

    def test_save_pct_no_shots(self):
        model = ExpectedGoalsAgainstModel()
        sp = model.compute_xga_save_pct([], "home")
        assert sp == 0.0

    def test_save_pct_all_saved(self):
        model = ExpectedGoalsAgainstModel()
        events = [
            {
                "type": "shot",
                "team": "away",
                "xg": 0.5,
                "start_x": 90,
                "start_y": 34,
                "is_goal": False,
            },
            {
                "type": "shot",
                "team": "away",
                "xg": 0.3,
                "start_x": 85,
                "start_y": 20,
                "is_goal": False,
            },
        ]
        sp = model.compute_xga_save_pct(events, "home")
        assert sp == pytest.approx(1.0, rel=0.01)

    def test_save_pct_all_conceded(self):
        model = ExpectedGoalsAgainstModel()
        events = [
            {
                "type": "shot",
                "team": "away",
                "xg": 0.5,
                "start_x": 90,
                "start_y": 34,
                "is_goal": True,
            },
            {
                "type": "shot",
                "team": "away",
                "xg": 0.5,
                "start_x": 85,
                "start_y": 20,
                "is_goal": True,
            },
        ]
        sp = model.compute_xga_save_pct(events, "home")
        assert sp == pytest.approx(0.0, rel=0.01)


class TestXgaFullReport:
    def test_full_report_structure(self):
        model = ExpectedGoalsAgainstModel()
        events = [
            {
                "type": "shot",
                "team": "away",
                "xg": 0.5,
                "body_part": "head",
                "start_x": 90,
                "start_y": 34,
                "metadata": {},
                "is_goal": True,
            },
            {
                "type": "shot",
                "team": "away",
                "xg": 0.3,
                "body_part": "right_foot",
                "start_x": 50,
                "start_y": 34,
                "metadata": {},
                "is_goal": False,
            },
        ]
        report = model.compute_full_report(events, "home")
        assert isinstance(report, XGAReport)
        assert report.total_xga == pytest.approx(0.8, rel=0.01)
        assert report.shots_faced == 2
        assert report.actual_goals_conceded == 1

    def test_full_report_empty(self):
        model = ExpectedGoalsAgainstModel()
        report = model.compute_full_report([], "home")
        assert report.total_xga == 0.0
        assert report.shots_faced == 0
        assert report.save_pct == 0.0

    def test_full_report_to_dict(self):
        model = ExpectedGoalsAgainstModel()
        events = [
            {
                "type": "shot",
                "team": "away",
                "xg": 0.4,
                "body_part": "left_foot",
                "start_x": 88,
                "start_y": 30,
                "metadata": {},
                "is_goal": False,
            },
        ]
        report = model.compute_full_report(events, "home")
        d = report.to_dict()
        assert d["total_xga"] == pytest.approx(0.4, rel=0.01)
        assert d["shots_faced"] == 1
        assert "type_breakdown" in d
        assert "zone_breakdown" in d
