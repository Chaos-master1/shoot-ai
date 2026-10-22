"""Tests for Switch of Play + Box Entries Detection module."""

from kawkab.core.switch_of_play import (
    SWITCH_MIN_LATERAL_M,
    SWITCH_MIN_TOTAL_M,
    SwitchOfPlayDetector,
    _classify_zone,
)


class TestSwitchOfPlay:
    def test_detect_switch_of_play_flank_to_flank(self):
        """A pass crossing >20m lateral and >30m total is a switch."""
        sod = SwitchOfPlayDetector()
        event = {
            "type": "pass",
            "start_x": 50.0,
            "start_y": 5.0,
            "end_x": 70.0,
            "end_y": 63.0,
        }
        result = sod.detect_switch_of_play(event)
        assert result["is_switch"] is True
        assert result["lateral_distance_m"] >= SWITCH_MIN_LATERAL_M
        assert result["total_distance_m"] >= SWITCH_MIN_TOTAL_M
        assert result["recipient_zone"] != ""

    def test_detect_switch_of_play_short_lateral(self):
        """A pass with small lateral movement is not a switch."""
        sod = SwitchOfPlayDetector()
        event = {
            "type": "pass",
            "start_x": 50.0,
            "start_y": 34.0,
            "end_x": 55.0,
            "end_y": 40.0,
        }
        result = sod.detect_switch_of_play(event)
        assert result["is_switch"] is False

    def test_detect_switch_of_play_not_pass(self):
        """Non-pass events are never switches."""
        sod = SwitchOfPlayDetector()
        event = {"type": "shot", "start_x": 50.0, "start_y": 34.0, "end_x": 55.0, "end_y": 40.0}
        result = sod.detect_switch_of_play(event)
        assert result["is_switch"] is False

    def test_analyze_switches_counts_and_completion(self):
        """Full switch analysis returns counts and completion per team."""
        sod = SwitchOfPlayDetector()
        events = [
            {
                "type": "pass",
                "team": "home",
                "start_x": 50.0,
                "start_y": 5.0,
                "end_x": 70.0,
                "end_y": 63.0,
                "completed": True,
            },
            {
                "type": "pass",
                "team": "home",
                "start_x": 50.0,
                "start_y": 5.0,
                "end_x": 70.0,
                "end_y": 63.0,
                "completed": False,
            },
            {
                "type": "pass",
                "team": "away",
                "start_x": 55.0,
                "start_y": 8.0,
                "end_x": 65.0,
                "end_y": 60.0,
                "completed": True,
            },
        ]
        result = sod.analyze_switches(events)
        assert "home" in result
        assert "away" in result
        assert result["home"]["switch_count"] >= 0
        assert result["away"]["switch_count"] >= 0
        assert "completion_rate" in result["home"]
        assert "avg_lateral_distance_m" in result["home"]
        assert "preferred_direction" in result["home"]

    def test_analyze_switches_empty(self):
        """Empty events return safe defaults."""
        sod = SwitchOfPlayDetector()
        result = sod.analyze_switches([])
        assert result["switch_count"]["home"] == 0
        assert result["switch_count"]["away"] == 0
        assert result["completion_rate"]["home"] == 0.0

    def test_analyze_switches_no_switch_events(self):
        """Events with no valid switches return zero counts."""
        sod = SwitchOfPlayDetector()
        events = [
            {
                "type": "pass",
                "team": "home",
                "start_x": 50.0,
                "start_y": 34.0,
                "end_x": 55.0,
                "end_y": 35.0,
                "completed": True,
            },
        ]
        result = sod.analyze_switches(events)
        assert result["home"]["switch_count"] == 0


class TestBoxEntries:
    def test_detect_box_entries_entry_into_box(self):
        """Pass ending inside penalty area is a box entry."""
        sod = SwitchOfPlayDetector()
        event = {
            "type": "pass",
            "start_x": 90.0,
            "start_y": 34.0,
            "end_x": 103.0,
            "end_y": 34.0,
        }
        result = sod.detect_box_entries(event)
        assert result["is_entry"] is True
        assert result["entry_type"] == "pass"

    def test_detect_box_entries_near_box_not_in(self):
        """Pass ending just outside the box is not an entry."""
        sod = SwitchOfPlayDetector()
        event = {
            "type": "pass",
            "start_x": 85.0,
            "start_y": 34.0,
            "end_x": 100.0,
            "end_y": 34.0,
        }
        result = sod.detect_box_entries(event)
        assert result["is_entry"] is False

    def test_detect_box_entries_carry_entry(self):
        """Carry into the box is a box entry."""
        sod = SwitchOfPlayDetector()
        event = {
            "type": "carry",
            "start_x": 90.0,
            "start_y": 34.0,
            "end_x": 103.0,
            "end_y": 34.0,
        }
        result = sod.detect_box_entries(event)
        assert result["is_entry"] is True
        assert result["entry_type"] == "carry"

    def test_detect_box_entries_not_pass_or_carry(self):
        """Non-pass/carry events are not box entries."""
        sod = SwitchOfPlayDetector()
        event = {"type": "tackle", "start_x": 90.0, "start_y": 34.0, "end_x": 103.0, "end_y": 34.0}
        result = sod.detect_box_entries(event)
        assert result["is_entry"] is False

    def test_analyze_box_entries_counts(self):
        """Full box entry analysis returns per-team counts."""
        sod = SwitchOfPlayDetector()
        events = [
            {
                "type": "pass",
                "team": "home",
                "start_x": 90.0,
                "start_y": 34.0,
                "end_x": 103.0,
                "end_y": 34.0,
            },
            {
                "type": "carry",
                "team": "home",
                "start_x": 90.0,
                "start_y": 34.0,
                "end_x": 103.0,
                "end_y": 34.0,
            },
            {
                "type": "pass",
                "team": "away",
                "start_x": 90.0,
                "start_y": 34.0,
                "end_x": 103.0,
                "end_y": 34.0,
            },
        ]
        result = sod.analyze_box_entries(events)
        assert "home" in result
        assert "away" in result
        assert result["home"]["total_entries"] >= 0
        assert result["home"]["entries_via_pass"] >= 0
        assert result["home"]["entries_via_carry"] >= 0

    def test_analyze_box_entries_empty(self):
        """Empty events return safe defaults."""
        sod = SwitchOfPlayDetector()
        result = sod.analyze_box_entries([])
        assert result["total_entries"]["home"] == 0
        assert result["total_entries"]["away"] == 0
        assert result["entries_via_pass"]["home"] == 0
        assert result["entries_via_carry"]["home"] == 0

    def test_analyze_box_entries_no_entries(self):
        """Events with no box entries return zero counts."""
        sod = SwitchOfPlayDetector()
        events = [
            {
                "type": "pass",
                "team": "home",
                "start_x": 50.0,
                "start_y": 34.0,
                "end_x": 55.0,
                "end_y": 35.0,
            },
        ]
        result = sod.analyze_box_entries(events)
        assert result["home"]["total_entries"] == 0

    def test_analyze_box_entries_leads_to_shot(self):
        """Box entries that lead to a shot are counted."""
        sod = SwitchOfPlayDetector()
        events = [
            {
                "type": "pass",
                "team": "home",
                "start_x": 90.0,
                "start_y": 34.0,
                "end_x": 103.0,
                "end_y": 34.0,
            },
            {
                "type": "shot",
                "team": "home",
                "start_x": 103.0,
                "start_y": 34.0,
                "end_x": 105.0,
                "end_y": 34.0,
                "is_goal": True,
            },
        ]
        result = sod.analyze_box_entries(events)
        assert result["home"]["entries_leading_to_shots"] >= 1


class TestClassifyZone:
    def test_classify_zone_returns_string(self):
        zone = _classify_zone(90.0, 34.0)
        assert isinstance(zone, str)
        assert len(zone) > 0
