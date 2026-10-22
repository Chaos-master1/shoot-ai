"""Tests for Build-Up Analysis module."""

from kawkab.core.build_up import (
    BuildUpReport,
    _classify_zone,
    _is_under_pressure,
    _lines_bypassed,
    _passes_through_line,
    analyze_build_up,
)


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
    under_pressure: bool = False,
    opponent_positions: list | None = None,
) -> dict:
    ev = {
        "type": etype,
        "team": team,
        "timestamp": timestamp,
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y,
        "completed": completed,
    }
    if under_pressure:
        ev["under_pressure"] = True
    if opponent_positions:
        ev["opponent_positions"] = opponent_positions
    return ev


class TestClassifyZone:
    def test_defensive_third(self):
        assert _classify_zone(10.0) == "defensive_third"

    def test_middle_third(self):
        assert _classify_zone(50.0) == "middle_third"

    def test_final_third(self):
        assert _classify_zone(80.0) == "final_third"

    def test_boundary_defensive(self):
        assert _classify_zone(34.0) == "middle_third"  # edge → next zone

    def test_boundary_final(self):
        assert _classify_zone(68.0) == "final_third"


class TestPassesThroughLine:
    def test_crosses_forward(self):
        assert _passes_through_line(20.0, 50.0, 34.0) is True

    def test_crosses_backward(self):
        assert _passes_through_line(50.0, 20.0, 34.0) is True

    def test_does_not_cross(self):
        assert _passes_through_line(10.0, 20.0, 34.0) is False

    def exact_on_line_start(self):
        # Start exactly on the line — does not exceed it
        assert _passes_through_line(34.0, 50.0, 34.0) is False


class TestLinesBypassed:
    def test_no_lines(self):
        assert _lines_bypassed(10.0, 20.0) == 0

    def test_one_line(self):
        assert _lines_bypassed(20.0, 50.0) == 1

    def test_two_lines(self):
        assert _lines_bypassed(20.0, 80.0) == 2

    def test_backward_pass(self):
        assert _lines_bypassed(80.0, 20.0) == 2


class TestIsUnderPressure:
    def test_explicit_flag(self):
        ev = _ev(0, under_pressure=True)
        assert _is_under_pressure(ev, [ev], 0) is True

    def test_opponent_positions_nearby(self):
        ev = _ev(
            0,
            start_x=50.0,
            start_y=34.0,
            opponent_positions=[{"x": 51.0, "y": 34.0}],
        )
        assert _is_under_pressure(ev, [ev], 0) is True

    def test_opponent_too_far(self):
        ev = _ev(
            0,
            start_x=50.0,
            start_y=34.0,
            opponent_positions=[{"x": 60.0, "y": 34.0}],
        )
        assert _is_under_pressure(ev, [ev], 0) is False

    def test_no_opponent_info(self):
        ev = _ev(0)
        assert _is_under_pressure(ev, [ev], 0) is False


class TestGoalKickPatterns:
    def test_short_goal_kick(self):
        events = [
            _ev(0, etype="goal_kick", team="home", timestamp=0.0, start_x=0.0, end_x=20.0),
        ]
        report = analyze_build_up(events, events, "home")
        assert report.goal_kick_patterns["short"]["attempts"] == 1
        assert report.goal_kick_patterns["long"]["attempts"] == 0

    def test_long_goal_kick(self):
        events = [
            _ev(0, etype="goal_kick", team="home", timestamp=0.0, start_x=0.0, end_x=40.0),
        ]
        report = analyze_build_up(events, events, "home")
        assert report.goal_kick_patterns["long"]["attempts"] == 1
        assert report.goal_kick_patterns["short"]["attempts"] == 0

    def test_goal_kick_middle_distance_not_counted(self):
        events = [
            _ev(0, etype="goal_kick", team="home", timestamp=0.0, start_x=0.0, end_x=30.0),
        ]
        report = analyze_build_up(events, events, "home")
        assert report.goal_kick_patterns["short"]["attempts"] == 0
        assert report.goal_kick_patterns["long"]["attempts"] == 0


class TestLineBreakingPasses:
    def test_pass_breaks_one_line(self):
        events = [
            _ev(0, team="home", timestamp=0.0, start_x=20.0, end_x=50.0),
        ]
        report = analyze_build_up(events, events, "home")
        assert len(report.line_breaking_passes) >= 1

    def test_pass_breaks_two_lines(self):
        events = [
            _ev(0, team="home", timestamp=0.0, start_x=20.0, end_x=80.0),
        ]
        report = analyze_build_up(events, events, "home")
        breaking = [
            p for p in report.line_breaking_passes if p.get("defensive_line_bypassed", 0) == 2
        ]
        assert len(breaking) >= 1

    def test_short_pass_no_lines_broken(self):
        events = [
            _ev(0, team="home", timestamp=0.0, start_x=10.0, end_x=20.0),
        ]
        report = analyze_build_up(events, events, "home")
        assert len(report.line_breaking_passes) == 0


class TestBuildOutUnderPressure:
    def test_pressure_detected(self):
        events = [
            _ev(0, team="home", timestamp=0.0, start_x=10.0, end_x=20.0, under_pressure=True),
        ]
        report = analyze_build_up(events, events, "home")
        assert report.build_out_under_pressure["attempts"] >= 1


class TestBuildUpEfficiency:
    def test_possession_reaches_final_third(self):
        events = [
            _ev(0, team="home", timestamp=0.0, start_x=5.0, end_x=15.0),
            _ev(1, team="home", timestamp=1.0, start_x=15.0, end_x=30.0),
            _ev(2, team="home", timestamp=2.0, start_x=30.0, end_x=50.0),
            _ev(3, team="home", timestamp=3.0, start_x=50.0, end_x=75.0),
        ]
        report = analyze_build_up(events, events, "home")
        assert report.build_up_efficiency > 0

    def test_no_events(self):
        report = analyze_build_up([], [], "home")
        assert report.build_up_efficiency == 0.0
        assert report.goal_kick_patterns["short"]["attempts"] == 0


class TestBuildUpReport:
    def test_summary_text_non_empty(self):
        report = BuildUpReport(team="home", match_id="m1")
        text = report.summary_text()
        assert isinstance(text, str)
        assert len(text) > 20
        assert "Build-Up Report" in text

    def test_to_dict(self):
        report = BuildUpReport(team="home", match_id="m1")
        d = report.to_dict()
        assert d["team"] == "home"
        assert d["match_id"] == "m1"
        assert "build_up_efficiency" in d

    def test_average_pass_sequence_length_computed(self):
        events = [
            _ev(0, team="home", timestamp=0.0, start_x=5.0, end_x=15.0),
            _ev(1, team="home", timestamp=1.0, start_x=15.0, end_x=30.0),
            _ev(2, team="home", timestamp=2.0, start_x=30.0, end_x=50.0),
        ]
        report = analyze_build_up(events, events, "home")
        assert report.average_pass_sequence_length > 0

    def test_zone_exit_stats_populated(self):
        events = [
            _ev(0, team="home", timestamp=0.0, start_x=5.0, end_x=15.0),  # def
            _ev(1, team="home", timestamp=1.0, start_x=15.0, end_x=50.0),  # def→mid
        ]
        report = analyze_build_up(events, events, "home")
        assert "defensive_third" in report.zone_exit_stats
        assert "middle_third" in report.zone_exit_stats

    def test_multiple_sequences(self):
        events = [
            _ev(0, team="home", timestamp=0.0, start_x=5.0, end_x=15.0),
            _ev(1, team="home", timestamp=1.0, start_x=15.0, end_x=30.0),
            _ev(2, team="away", timestamp=2.0, start_x=60.0, end_x=50.0),
            _ev(3, team="home", timestamp=3.0, start_x=10.0, end_x=25.0),
        ]
        report = analyze_build_up(events, events, "home")
        assert isinstance(report.build_up_actions, list)
        assert len(report.build_up_actions) >= 1

    def test_goal_kick_success_rate(self):
        events = [
            _ev(
                0,
                etype="goal_kick",
                team="home",
                timestamp=0.0,
                start_x=0.0,
                end_x=20.0,
                completed=True,
            ),
            _ev(
                1,
                etype="goal_kick",
                team="home",
                timestamp=1.0,
                start_x=0.0,
                end_x=15.0,
                completed=False,
            ),
        ]
        report = analyze_build_up(events, events, "home")
        assert report.goal_kick_patterns["short"]["attempts"] == 2
        assert report.goal_kick_patterns["short"]["success_pct"] == 50.0

    def test_shot_ends_sequence(self):
        events = [
            _ev(0, team="home", timestamp=0.0, start_x=5.0, end_x=30.0),
            _ev(1, team="home", timestamp=1.0, start_x=30.0, end_x=50.0),
            _ev(2, etype="shot", team="home", timestamp=2.0, start_x=50.0),
        ]
        report = analyze_build_up(events, events, "home")
        assert isinstance(report.build_up_efficiency, float)
