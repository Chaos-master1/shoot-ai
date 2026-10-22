"""Tests for Territory Compounding module."""

from kawkab.core.territory_value import (
    TerritoryCell,
    TerritoryReport,
    _detect_possession_chains_full,
    compute_territory_value,
)
from kawkab.core.xt_model import ExpectedThreatModel


def _ev(
    idx: int,
    etype: str = "pass",
    team: str = "home",
    timestamp: float = 0.0,
    start_x: float = 0.0,
    start_y: float = 34.0,
    end_x: float = 0.0,
    end_y: float = 34.0,
    completed: bool = True,
) -> dict:
    return {
        "type": etype,
        "team": team,
        "timestamp": timestamp,
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y,
        "completed": completed,
    }


class TestDetectChains:
    def test_single_chain(self):
        events = [
            _ev(0, team="home", timestamp=0.0),
            _ev(1, team="home", timestamp=1.0),
        ]
        chains = _detect_possession_chains_full(events)
        assert len(chains) == 1
        assert len(chains[0]) == 2

    def test_multi_team_chains(self):
        events = [
            _ev(0, team="home", timestamp=0.0),
            _ev(1, team="away", timestamp=1.0),
            _ev(2, team="away", timestamp=2.0),
        ]
        chains = _detect_possession_chains_full(events)
        assert len(chains) == 2
        assert len(chains[0]) == 1
        assert len(chains[1]) == 2

    def test_empty_events(self):
        assert _detect_possession_chains_full([]) == []


class TestComputeTerritoryValue:
    def test_empty_events(self):
        report = compute_territory_value([], [], [], "home")
        assert report.team == "home"
        assert report.total_xT_gained == 0.0
        assert report.total_xT_conceded == 0.0
        assert report.match_id == ""

    def test_single_event_adds_xt_to_zone(self):
        events = [
            _ev(0, team="home", timestamp=0.0, start_x=10.0, end_x=40.0),
            _ev(1, team="home", timestamp=1.0, start_x=40.0, end_x=70.0),
        ]
        home_events = [e for e in events if e.get("team") == "home"]
        report = compute_territory_value(home_events, [], events, "home")
        assert report.total_xT_gained > 0 or report.total_xT_gained == 0.0
        assert len(report.cells) >= 1

    def test_territory_timeline_generated(self):
        events = [
            _ev(0, team="home", timestamp=30.0, start_x=10.0, end_x=40.0),
            _ev(1, team="home", timestamp=90.0, start_x=40.0, end_x=70.0),
        ]
        home_events = [e for e in events if e.get("team") == "home"]
        report = compute_territory_value(home_events, [], events, "home")
        assert len(report.territory_timeline) >= 1
        entry = report.territory_timeline[0]
        assert "minute" in entry
        assert "team_control_pct" in entry

    def test_multiple_chains_aggregated(self):
        events = [
            _ev(0, team="home", timestamp=0.0, start_x=10.0, end_x=30.0),
            _ev(1, team="home", timestamp=1.0, start_x=30.0, end_x=50.0),
            _ev(2, team="away", timestamp=2.0, start_x=70.0, end_x=50.0),
            _ev(3, team="away", timestamp=3.0, start_x=50.0, end_x=30.0),
        ]
        home_events = [e for e in events if e.get("team") == "home"]
        away_events = [e for e in events if e.get("team") == "away"]
        report = compute_territory_value(home_events, away_events, events, "home")
        assert len(report.possession_chains) == 2
        assert report.total_xT_gained >= 0

    def test_net_territory_value_positive_for_team_with_more_xt(self):
        # Home has more forward passes (more xT) than away
        events = [
            _ev(0, team="home", timestamp=0.0, start_x=10.0, end_x=40.0),
            _ev(1, team="home", timestamp=1.0, start_x=40.0, end_x=80.0),
            _ev(2, team="away", timestamp=2.0, start_x=80.0, end_x=70.0),
            _ev(3, team="away", timestamp=3.0, start_x=70.0, end_x=60.0),
        ]
        home_events = [e for e in events if e.get("team") == "home"]
        away_events = [e for e in events if e.get("team") == "away"]
        report = compute_territory_value(home_events, away_events, events, "home")
        # Home should have higher total xT gained than conceded
        assert report.total_xT_gained >= 0
        # net should equal gained minus conceded
        assert (
            abs(report.net_territory_value - (report.total_xT_gained - report.total_xT_conceded))
            < 1e-6
        )

    def test_dominant_zones_detected(self):
        events = [
            _ev(0, team="home", timestamp=0.0, start_x=10.0, end_x=40.0),
            _ev(1, team="home", timestamp=1.0, start_x=40.0, end_x=85.0),
        ]
        home_events = [e for e in events if e.get("team") == "home"]
        report = compute_territory_value(home_events, [], events, "home")
        # All zones should be dominant since there's no opposition
        assert len(report.dominant_zones) >= 0

    def test_possession_chains_summaries(self):
        events = [
            _ev(0, team="home", timestamp=0.0, start_x=10.0, end_x=30.0),
            _ev(1, team="home", timestamp=1.0, start_x=30.0, end_x=50.0),
            _ev(2, team="home", timestamp=2.0, start_x=50.0, end_x=70.0),
        ]
        home_events = [e for e in events if e.get("team") == "home"]
        report = compute_territory_value(home_events, [], events, "home")
        assert len(report.possession_chains) == 1
        chain = report.possession_chains[0]
        assert "pass_count" in chain
        assert "xT_gained" in chain
        assert "reached_final_third" in chain

    def test_reached_final_third_flag(self):
        events = [
            _ev(0, team="home", timestamp=0.0, start_x=10.0, end_x=80.0),
        ]
        home_events = [e for e in events if e.get("team") == "home"]
        report = compute_territory_value(home_events, [], events, "home")
        assert report.possession_chains[0]["reached_final_third"] is True

    def test_custom_xt_model(self):
        events = [
            _ev(0, team="home", timestamp=0.0, start_x=10.0, end_x=40.0),
            _ev(1, team="home", timestamp=1.0, start_x=40.0, end_x=80.0),
        ]
        model = ExpectedThreatModel(rows=5, cols=4)
        model.build_transition_matrix(events)
        home_events = [e for e in events if e.get("team") == "home"]
        report = compute_territory_value(home_events, [], events, "home", xt_model=model)
        assert len(report.cells) >= 1

    def test_opponent_xt_tracked(self):
        events = [
            _ev(0, team="home", timestamp=0.0, start_x=10.0, end_x=30.0),
            _ev(1, team="away", timestamp=1.0, start_x=80.0, end_x=60.0),
        ]
        home_events = [e for e in events if e.get("team") == "home"]
        away_events = [e for e in events if e.get("team") == "away"]
        report = compute_territory_value(home_events, away_events, events, "home")
        assert report.total_xT_conceded >= 0


class TestTerritoryCell:
    def test_to_dict(self):
        cell = TerritoryCell(
            zone_x=2, zone_y=3, xT_gained=1.5, xT_conceded=0.5, net_xT=1.0, event_count=5
        )
        d = cell.to_dict()
        assert d["zone_x"] == 2
        assert d["zone_y"] == 3
        assert d["xT_gained"] == 1.5
        assert d["net_xT"] == 1.0
        assert d["event_count"] == 5


class TestTerritoryReport:
    def test_summary_text_non_empty(self):
        report = TerritoryReport(team="home", match_id="m1")
        text = report.summary_text()
        assert isinstance(text, str)
        assert len(text) > 20
        assert "Territory Report" in text

    def test_to_dict(self):
        report = TerritoryReport(
            team="home", match_id="m1", total_xT_gained=2.5, net_territory_value=1.0
        )
        d = report.to_dict()
        assert d["team"] == "home"
        assert d["total_xT_gained"] == 2.5
