"""Tests for Box Touches / Penalty Area Entries module."""

from kawkab.core.box_entries import (
    BoxEntryAnalyzer,
)


def _make_event(
    team: str,
    type: str,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    from_track_id: int = 1,
    track_id: int = None,
) -> dict:
    return {
        "type": type,
        "team": team,
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y,
        "from_track_id": from_track_id,
        "track_id": track_id,
    }


class TestDetectBoxTouch:
    def test_touch_in_box(self):
        bea = BoxEntryAnalyzer()
        ev = _make_event("home", "shot", 90, 34, 103, 34, 1)
        result = bea.detect_box_touch(ev)
        assert result["is_touch"] is True
        assert result["touch_type"] == "shot"

    def test_touch_outside_box(self):
        bea = BoxEntryAnalyzer()
        ev = _make_event("home", "pass", 50, 34, 80, 34, 1)
        result = bea.detect_box_touch(ev)
        assert result["is_touch"] is False

    def test_non_touch_event_type(self):
        bea = BoxEntryAnalyzer()
        ev = _make_event("home", "tackle", 90, 34, 103, 34, 1)
        result = bea.detect_box_touch(ev)
        assert result["is_touch"] is False

    def test_receive_in_box(self):
        bea = BoxEntryAnalyzer()
        ev = _make_event("home", "receive", 90, 34, 103, 34, 1)
        result = bea.detect_box_touch(ev)
        assert result["is_touch"] is True
        assert result["touch_type"] == "receive"


class TestDetectPenaltyAreaEntry:
    def test_entry_via_pass(self):
        bea = BoxEntryAnalyzer()
        ev = _make_event("home", "pass", 90, 34, 103, 34, 1)
        result = bea.detect_penalty_area_entry(ev)
        assert result["is_entry"] is True
        assert result["entry_type"] == "pass"

    def test_no_entry_already_in_box(self):
        bea = BoxEntryAnalyzer()
        ev = _make_event("home", "pass", 103, 34, 104, 34, 1)
        result = bea.detect_penalty_area_entry(ev)
        assert result["is_entry"] is False

    def test_no_entry_outside(self):
        bea = BoxEntryAnalyzer()
        ev = _make_event("home", "pass", 50, 34, 80, 34, 1)
        result = bea.detect_penalty_area_entry(ev)
        assert result["is_entry"] is False

    def test_entry_via_carry(self):
        bea = BoxEntryAnalyzer()
        ev = _make_event("home", "carry", 90, 34, 103, 34, 1)
        result = bea.detect_penalty_area_entry(ev)
        assert result["is_entry"] is True
        assert result["entry_type"] == "carry"


class TestAnalyzeBoxTouches:
    def test_per_team_stats(self):
        bea = BoxEntryAnalyzer()
        events = [
            _make_event("home", "shot", 90, 34, 103, 34, 1),
            _make_event("home", "pass", 90, 34, 104, 34, 2),
            _make_event("away", "dribble", 95, 34, 103, 34, 3),
        ]
        result = bea.analyze_box_touches(events)
        assert "home" in result
        assert "away" in result
        assert result["home"]["total_touches"] >= 1
        assert result["away"]["total_touches"] >= 1

    def test_empty_events(self):
        bea = BoxEntryAnalyzer()
        result = bea.analyze_box_touches([])
        assert result["home"]["total_touches"] == 0
        assert result["away"]["total_touches"] == 0


class TestAnalyzeBoxEntries:
    def test_per_team_entries(self):
        bea = BoxEntryAnalyzer()
        events = [
            _make_event("home", "pass", 90, 34, 103, 34, 1),
            _make_event("home", "carry", 90, 34, 104, 34, 2),
        ]
        result = bea.analyze_box_entries(events)
        assert result["home"]["total_entries"] >= 1

    def test_entry_leads_to_shot(self):
        bea = BoxEntryAnalyzer()
        events = [
            _make_event("home", "pass", 90, 34, 103, 34, 1),
            {
                "type": "shot",
                "team": "home",
                "start_x": 103,
                "start_y": 34,
                "end_x": 105,
                "end_y": 34,
                "is_goal": True,
                "xg": 0.3,
            },
        ]
        result = bea.analyze_box_entries(events)
        assert result["home"]["entries_leading_to_shots"] >= 1
        assert result["home"]["entries_leading_to_goals"] >= 1


class TestComputeEffectiveness:
    def test_conversion_rates(self):
        bea = BoxEntryAnalyzer()
        events = [
            _make_event("home", "shot", 90, 34, 103, 34, 1),
            {
                "type": "shot",
                "team": "home",
                "start_x": 103,
                "start_y": 34,
                "end_x": 105,
                "end_y": 34,
                "is_goal": True,
                "xg": 0.3,
            },
        ]
        result = bea.compute_effectiveness(events)
        assert "home" in result
        assert result["home"]["box_touch_to_shot_pct"] > 0

    def test_empty_events_safe_defaults(self):
        bea = BoxEntryAnalyzer()
        result = bea.compute_effectiveness([])
        assert result["home"]["box_touch_to_shot_pct"] == 0.0
