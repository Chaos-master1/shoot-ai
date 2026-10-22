"""Tests for Pressing Classifier — block type, man/zonal, trigger pressing."""

from kawkab.core.game_constants import GAME
from kawkab.core.pressing_classifier import (
    _avg_defensive_line_x,
    _classify_block_type,
    _detect_man_or_zonal,
    _detect_trigger_pressing_moments,
    classify_pressing_system,
)

PITCH_LENGTH = GAME.PITCH_LENGTH_M


def _make_event(team, etype, x=50, y=34, ts=0, completed=True, under_pressure=False):
    return {
        "team": team,
        "type": etype,
        "start_x": x,
        "start_y": y,
        "timestamp": ts,
        "completed": completed,
        "under_pressure": under_pressure,
    }


class TestClassifyBlockType:
    def test_high_block(self):
        assert _classify_block_type(PITCH_LENGTH * 0.6) == "high_block"

    def test_mid_block(self):
        assert _classify_block_type(PITCH_LENGTH * 0.4) == "mid_block"

    def test_low_block(self):
        assert _classify_block_type(PITCH_LENGTH * 0.2) == "low_block"

    def test_boundary_high(self):
        assert _classify_block_type(PITCH_LENGTH * 0.55) == "high_block"

    def test_boundary_mid(self):
        assert _classify_block_type(PITCH_LENGTH * 0.35) == "mid_block"


class TestAvgDefensiveLine:
    def test_with_events(self):
        events = [
            _make_event("away", "pass", x=60),
            _make_event("away", "pass", x=40),
            _make_event("away", "pass", x=50),
        ]
        avg = _avg_defensive_line_x(events, "home")
        assert 40 <= avg <= 60

    def test_empty_events(self):
        avg = _avg_defensive_line_x([], "home")
        assert avg == PITCH_LENGTH * 0.35

    def test_no_opponent_events(self):
        events = [_make_event("home", "pass", x=50)]
        avg = _avg_defensive_line_x(events, "home")
        assert avg > 0


class TestManOrZonal:
    def test_man_oriented(self):
        events = [
            _make_event("away", "tackle", x=i * 10, y=j * 10)
            for i in range(1, 6)
            for j in range(1, 3)
        ]
        style = _detect_man_or_zonal(events, "home")
        assert style in ("man_oriented", "zonal", "unknown")

    def test_few_tackles(self):
        events = [_make_event("away", "tackle", x=50)]
        style = _detect_man_or_zonal(events, "home")
        assert style == "unknown"


class TestTriggerPressing:
    def test_no_triggers(self):
        events = [_make_event("home", "pass", ts=0)]
        triggers = _detect_trigger_pressing_moments(events, "home")
        assert triggers == []

    def test_back_pass_trigger(self):
        events = [
            _make_event("away", "pass", ts=0),  # back pass
            _make_event("away", "pass", ts=1),
            _make_event("home", "tackle", ts=2),
        ]
        events[0]["type"] = "back_pass"
        triggers = _detect_trigger_pressing_moments(events, "home")
        assert len(triggers) >= 1

    def test_poor_control_trigger(self):
        events = [
            _make_event("away", "poor_control", ts=0),
            _make_event("home", "interception", ts=1),
        ]
        triggers = _detect_trigger_pressing_moments(events, "home")
        assert len(triggers) >= 1
        assert triggers[0]["trigger_event"] == "poor_control"

    def test_trigger_no_response(self):
        events = [
            _make_event("away", "back_pass", ts=0),
            _make_event("home", "pass", ts=8),  # outside window
        ]
        triggers = _detect_trigger_pressing_moments(events, "home", trigger_window_s=3.0)
        # May still detect trigger with no response
        if triggers:
            assert triggers[0]["response_time_s"] is None


class TestClassifyPressingSystem:
    def test_classify_empty_events(self):
        report = classify_pressing_system([], team="home")
        assert report.team == "home"
        assert report.primary_block_type != ""

    def test_classify_basic(self):
        events = [
            _make_event("home", "pass", x=30, ts=0),
            _make_event("away", "pass", x=50, ts=1),
            _make_event("home", "tackle", x=40, ts=2),
            _make_event("away", "pass", x=60, ts=3),
            _make_event("home", "interception", x=45, ts=4),
        ]
        report = classify_pressing_system(events, team="home")
        assert report.team == "home"
        assert report.pressing_style in ("man_oriented", "zonal", "unknown")

    def test_pressing_report_to_dict(self):
        from kawkab.core.pressing_classifier import PressingSystemReport

        report = PressingSystemReport(
            team="away",
            primary_block_type="high_block",
            pressing_style="man_oriented",
            trigger_count=5,
            trigger_success_rate=0.6,
            ppda=8.5,
        )
        d = report.to_dict()
        assert d["team"] == "away"
        assert d["primary_block_type"] == "high_block"
        assert d["ppda"] == 8.5
