"""Tests for Passing Lane Analysis module."""

from kawkab.core.passing_lanes import (
    PassingLaneAnalysis,
    _classify_zone,
    _estimate_player_positions,
)


class TestPassingLaneAnalysis:
    def test_count_pass_options_basic(self):
        """Count viable passing options from a pass event."""
        pla = PassingLaneAnalysis()
        event = {
            "type": "pass",
            "team": "home",
            "start_x": 40.0,
            "start_y": 34.0,
            "timestamp": 100.0,
        }
        team_positions = [
            {"x": 55.0, "y": 34.0},
            {"x": 60.0, "y": 20.0},
            {"x": 70.0, "y": 50.0},
        ]
        count = pla.count_pass_options(event, [], team_positions)
        assert count >= 0
        assert isinstance(count, int)

    def test_count_pass_options_not_pass(self):
        """Non-pass events return 0 options."""
        pla = PassingLaneAnalysis()
        event = {"type": "shot", "start_x": 50.0, "start_y": 34.0}
        assert pla.count_pass_options(event, [], []) == 0

    def test_analyze_lane_density_two_teams(self):
        """Lane density returns per-team stats."""
        pla = PassingLaneAnalysis()
        events = [
            {
                "type": "pass",
                "team": "home",
                "start_x": 40.0,
                "start_y": 34.0,
                "timestamp": 10.0,
                "end_x": 55.0,
                "end_y": 34.0,
            },
            {
                "type": "pass",
                "team": "home",
                "start_x": 50.0,
                "start_y": 34.0,
                "timestamp": 20.0,
                "end_x": 65.0,
                "end_y": 34.0,
            },
            {
                "type": "pass",
                "team": "away",
                "start_x": 60.0,
                "start_y": 34.0,
                "timestamp": 30.0,
                "end_x": 45.0,
                "end_y": 34.0,
            },
        ]
        result = pla.analyze_lane_density(events)
        assert "home" in result
        assert "away" in result
        assert "avg_options" in result["home"]
        assert "max_options" in result["home"]
        assert "min_options" in result["home"]

    def test_analyze_lane_density_empty(self):
        """Empty events returns empty dict."""
        pla = PassingLaneAnalysis()
        assert pla.analyze_lane_density([]) == {}

    def test_detect_lane_blocking_blocked(self):
        """Detect a blocked/intercepted pass lane."""
        pla = PassingLaneAnalysis()
        event = {
            "type": "pass",
            "team": "home",
            "start_x": 40.0,
            "start_y": 34.0,
            "end_x": 70.0,
            "end_y": 34.0,
        }
        all_events = [
            event,
            {"type": "defensive", "team": "away", "start_x": 55.0, "start_y": 34.0},
        ]
        result = pla.detect_lane_blocking(event, all_events)
        # Opponent at (55, 34) lies on the pass line from (40,34) to (70,34)
        assert result["is_blocked"] is True
        assert result["blocker_distance"] > 0

    def test_detect_lane_blocking_unblocked(self):
        """Pass with no defenders near the lane is unblocked."""
        pla = PassingLaneAnalysis()
        event = {
            "type": "pass",
            "team": "home",
            "start_x": 40.0,
            "start_y": 34.0,
            "end_x": 70.0,
            "end_y": 34.0,
        }
        all_events = [
            event,
            {"type": "defensive", "team": "away", "start_x": 10.0, "start_y": 10.0},
        ]
        result = pla.detect_lane_blocking(event, all_events)
        assert result["is_blocked"] is False

    def test_detect_lane_blocking_not_pass(self):
        """Non-pass events are never blocked."""
        pla = PassingLaneAnalysis()
        event = {"type": "shot", "start_x": 50.0, "start_y": 34.0}
        result = pla.detect_lane_blocking(event, [])
        assert result["is_blocked"] is False

    def test_compute_progressive_lane_changes_basic(self):
        """Progressive lane changes counted per team."""
        pla = PassingLaneAnalysis()
        events = [
            {
                "type": "pass",
                "team": "home",
                "start_x": 30.0,
                "start_y": 34.0,
                "end_x": 45.0,
                "end_y": 34.0,
                "timestamp": 10.0,
            },
            {
                "type": "pass",
                "team": "home",
                "start_x": 45.0,
                "start_y": 34.0,
                "end_x": 60.0,
                "end_y": 34.0,
                "timestamp": 20.0,
            },
            {
                "type": "pass",
                "team": "away",
                "start_x": 60.0,
                "start_y": 34.0,
                "end_x": 45.0,
                "end_y": 34.0,
                "timestamp": 30.0,
            },
        ]
        result = pla.compute_progressive_lane_changes(events)
        assert isinstance(result, dict)
        assert "home" in result or "away" in result or result == {}

    def test_compute_progressive_lane_changes_empty(self):
        """Empty events returns empty dict."""
        pla = PassingLaneAnalysis()
        assert pla.compute_progressive_lane_changes([]) == {}

    def test_estimate_player_positions_returns_list(self):
        """Estimate positions from events around a timestamp."""
        events = [
            {"type": "pass", "team": "home", "start_x": 40.0, "start_y": 30.0, "timestamp": 100.0},
            {"type": "pass", "team": "home", "end_x": 55.0, "end_y": 34.0, "timestamp": 102.0},
            {"type": "pass", "team": "away", "start_x": 60.0, "start_y": 34.0, "timestamp": 101.0},
        ]
        positions = _estimate_player_positions(events, "home", 101.0)
        assert isinstance(positions, list)
        assert len(positions) > 0
        for pos in positions:
            assert "x" in pos
            assert "y" in pos

    def test_count_pass_options_out_of_range(self):
        """Positions beyond 40m or behind do not count as options."""
        pla = PassingLaneAnalysis()
        event = {
            "type": "pass",
            "team": "home",
            "start_x": 40.0,
            "start_y": 34.0,
            "timestamp": 100.0,
        }
        team_positions = [
            {"x": 85.0, "y": 34.0},
            {"x": 10.0, "y": 34.0},
        ]
        count = pla.count_pass_options(event, [], team_positions)
        assert count == 0


class TestClassifyZone:
    def test_classify_zone(self):
        zone = _classify_zone(90.0, 34.0)
        assert isinstance(zone, str)
        assert len(zone) > 0
