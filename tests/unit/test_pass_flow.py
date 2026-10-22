"""Tests for pass flow analysis."""

from kawkab.core.pass_flow import PassFlowLink, compute_pass_flow


class TestPassFlow:
    def test_empty_events(self):
        result = compute_pass_flow([], "home")
        assert result == []

    def test_single_pass(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "start_x": 10,
                "start_y": 34,
                "end_x": 50,
                "end_y": 40,
                "completed": True,
                "timestamp": 10,
            },
        ]
        result = compute_pass_flow(events, "home")
        assert len(result) == 1
        assert result[0]["count"] == 1
        assert result[0]["completed"] == 1

    def test_filters_other_team(self):
        events = [
            {
                "type": "pass",
                "team": "away",
                "start_x": 10,
                "start_y": 34,
                "end_x": 50,
                "end_y": 40,
                "completed": True,
                "timestamp": 10,
            },
        ]
        result = compute_pass_flow(events, "home")
        assert result == []

    def test_aggregates_same_zone(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "start_x": 12,
                "start_y": 34,
                "end_x": 52,
                "end_y": 40,
                "completed": True,
                "timestamp": 10,
            },
            {
                "type": "pass",
                "team": "home",
                "start_x": 13,
                "start_y": 33,
                "end_x": 53,
                "end_y": 39,
                "completed": False,
                "timestamp": 20,
            },
        ]
        result = compute_pass_flow(events, "home", grid_cells=5)
        assert len(result) >= 1
        assert result[0]["count"] == 2
        assert result[0]["completed"] == 1

    def test_link_to_dict(self):
        link = PassFlowLink(origin_x=10, origin_y=34, dest_x=50, dest_y=40, count=2, completed=1)
        d = link.to_dict()
        assert d["count"] == 2
        assert d["accuracy"] == 0.5
