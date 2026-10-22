"""Tests for ball recovery analysis module."""

from kawkab.core.ball_recovery import (
    RECOVERY_EVENT_TYPES,
    BallRecoveryAnalyzer,
    _to_zone,
    _zone_key,
)


class TestBallRecovery:
    def setup_method(self):
        self.analyzer = BallRecoveryAnalyzer()

    def test_classify_recovery_interception(self):
        ev = {"type": "interception", "x": 50, "y": 34, "team": "home"}
        rtype, rx, ry = self.analyzer.classify_recovery(ev, [])
        assert rtype == "interception"
        assert rx == 50
        assert ry == 34

    def test_classify_recovery_tackle(self):
        ev = {"type": "tackle", "x": 60, "y": 30, "team": "away"}
        rtype, rx, ry = self.analyzer.classify_recovery(ev, [])
        assert rtype == "tackle"
        assert rx == 60
        assert ry == 30

    def test_classify_recovery_loose_ball(self):
        ev = {"type": "ball_recovery", "x": 40, "y": 20, "team": "home"}
        rtype, rx, ry = self.analyzer.classify_recovery(ev, [])
        assert rtype == "loose_ball"

    def test_classify_recovery_goal_kick(self):
        ev = {"type": "goal_kick", "x": 10, "y": 34, "team": "home"}
        rtype, rx, ry = self.analyzer.classify_recovery(ev, [])
        assert rtype == "goal_kick"

    def test_classify_recovery_clearance(self):
        ev = {"type": "clearance", "x": 20, "y": 40, "team": "away"}
        rtype, rx, ry = self.analyzer.classify_recovery(ev, [])
        assert rtype == "clearance"

    def test_compute_recovery_locations(self):
        events = [
            {"type": "interception", "x": 10, "y": 10, "team": "home"},
            {"type": "tackle", "x": 30, "y": 20, "team": "home"},
            {"type": "clearance", "x": 80, "y": 50, "team": "away"},
            {"type": "interception", "x": 50, "y": 34, "team": "home"},
        ]
        locations = self.analyzer.compute_recovery_locations(events, "home")
        assert len(locations) >= 2
        total = sum(locations.values())
        assert total == 3

    def test_analyze_recoveries_basic(self):
        events = [
            {"type": "interception", "x": 50, "y": 34, "team": "home", "timestamp": 10},
            {"type": "pass", "x": 55, "y": 34, "team": "home", "timestamp": 15},
            {"type": "shot", "x": 90, "y": 34, "team": "home", "timestamp": 20, "is_goal": False},
            {"type": "tackle", "x": 40, "y": 20, "team": "away", "timestamp": 25},
        ]
        report = self.analyzer.analyze_recoveries(events, "home")
        assert report["total_recoveries"] == 1
        assert "recoveries_by_type" in report
        assert "recoveries_by_zone" in report

    def test_analyze_recoveries_empty(self):
        report = self.analyzer.analyze_recoveries([], "home")
        assert report["total_recoveries"] == 0
        assert report["recoveries_leading_to_shot"] == 0
        assert report["recoveries_leading_to_goal"] == 0

    def test_detect_counter_press_true(self):
        event = {"type": "interception", "x": 80, "y": 34, "team": "home", "timestamp": 100}
        events = [
            event,
            {"type": "tackle", "x": 82, "y": 34, "team": "away", "timestamp": 101},
        ]
        is_cp, duration, result = self.analyzer.detect_counter_press(event, events)
        assert is_cp is True
        assert duration > 0
        assert result == "tackle"

    def test_detect_counter_press_false(self):
        event = {"type": "interception", "x": 30, "y": 34, "team": "home", "timestamp": 100}
        events = [event]
        is_cp, duration, result = self.analyzer.detect_counter_press(event, events)
        assert is_cp is False
        assert duration == 0

    def test_compute_recovery_efficiency(self):
        events = [
            {"type": "interception", "x": 10, "y": 10, "team": "home", "timestamp": 0},
            {"type": "pass", "x": 20, "y": 20, "team": "home", "timestamp": 5},
            {"type": "tackle", "x": 15, "y": 15, "team": "away", "timestamp": 10},
            {"type": "shot", "x": 90, "y": 34, "team": "home", "timestamp": 15, "is_goal": True},
        ]
        eff = self.analyzer.compute_recovery_efficiency(events)
        assert "home" in eff
        assert eff["home"]["recoveries"] >= 1

    def test_recovery_to_shot_conversion(self):
        events = [
            {"type": "interception", "x": 50, "y": 34, "team": "home", "timestamp": 0},
            {"type": "pass", "x": 60, "y": 34, "team": "home", "timestamp": 5},
            {"type": "shot", "x": 90, "y": 34, "team": "home", "timestamp": 10, "is_goal": True},
            {"type": "tackle", "x": 40, "y": 20, "team": "away", "timestamp": 15},
        ]
        report = self.analyzer.analyze_recoveries(events, "home")
        assert report["recoveries_leading_to_shot"] == 1
        assert report["recoveries_leading_to_goal"] == 1

    def test_empty_events(self):
        assert self.analyzer.compute_recovery_locations([], "home") == {}
        eff = self.analyzer.compute_recovery_efficiency([])
        assert eff == {}

    def test_to_zone_mapping(self):
        zx, zy = _to_zone(0, 0)
        assert zx == 0
        assert zy == 0
        zx, zy = _to_zone(105, 68)
        assert zx == 4
        assert zy == 4

    def test_zone_key(self):
        assert _zone_key(2, 3) == "2_3"

    def test_recovery_event_types_set(self):
        assert "interception" in RECOVERY_EVENT_TYPES
        assert "tackle" in RECOVERY_EVENT_TYPES
        assert "loose_ball" in RECOVERY_EVENT_TYPES
        assert "pass" not in RECOVERY_EVENT_TYPES

    def test_classify_loose_ball_from_incomplete_pass(self):
        ev = {"type": "pass", "x": 50, "y": 34, "team": "home", "completed": False}
        rtype, _, _ = self.analyzer.classify_recovery(ev, [{"type": "pass"}])
        assert rtype == "loose_ball"
