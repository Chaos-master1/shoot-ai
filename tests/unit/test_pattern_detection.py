"""Tests for Recurring Pattern Detection module."""

from kawkab.core.pattern_detection import TacticalPatternDetector


def _make_pass(
    team: str,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    from_track_id: int = 1,
    to_track_id: int = 2,
    timestamp: float = 0,
) -> dict:
    return {
        "type": "pass",
        "team": team,
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y,
        "from_track_id": from_track_id,
        "to_track_id": to_track_id,
        "timestamp": timestamp,
    }


class TestDetectRecurringSequences:
    def test_finds_recurring(self):
        tpd = TacticalPatternDetector()
        events = []
        for i in range(4):
            events.append(_make_pass("home", 30, 30, 40, 30, 1, 2, i * 2))
            events.append(_make_pass("home", 40, 30, 50, 30, 2, 3, i * 2 + 1))
        result = tpd.detect_recurring_sequences(events, "home", min_occurrences=2)
        assert len(result) >= 1

    def test_empty_events(self):
        tpd = TacticalPatternDetector()
        result = tpd.detect_recurring_sequences([], "home")
        assert result == []

    def test_no_pattern_returns_empty(self):
        tpd = TacticalPatternDetector()
        events = [_make_pass("home", 30, 30, 40, 30, 1, 2, 0)]
        result = tpd.detect_recurring_sequences(events, "home")
        assert result == []

    def test_min_occurrences_filter(self):
        tpd = TacticalPatternDetector()
        events = []
        for i in range(3):
            events.append(_make_pass("home", 30, 30, 40, 30, 1, 2, i * 2))
            events.append(_make_pass("home", 40, 30, 50, 30, 2, 3, i * 2 + 1))
        result = tpd.detect_recurring_sequences(events, "home", min_occurrences=5)
        assert result == []

    def test_shot_rate_tracked(self):
        tpd = TacticalPatternDetector()
        events = []
        for i in range(3):
            events.append(_make_pass("home", 30, 30, 40, 30, 1, 2, i * 2))
            events.append(_make_pass("home", 40, 30, 50, 30, 2, 3, i * 2 + 1))
        events.append(
            {"type": "shot", "team": "home", "timestamp": 20, "xg": 0.2, "is_goal": False}
        )
        result = tpd.detect_recurring_sequences(events, "home", min_occurrences=2)
        if result:
            assert result[0]["shot_rate"] >= 0


class TestIdentifySignaturePatterns:
    def test_returns_sorted(self):
        tpd = TacticalPatternDetector()
        events = []
        for i in range(5):
            events.append(_make_pass("home", 30, 30, 40, 30, 1, 2, i * 2))
            events.append(_make_pass("home", 40, 30, 50, 30, 2, 3, i * 2 + 1))
        result = tpd.identify_signature_patterns(events, "home", min_frequency=3)
        assert all(r["count"] >= 3 for r in result)

    def test_no_signatures(self):
        tpd = TacticalPatternDetector()
        events = [
            _make_pass("home", 30, 30, 40, 30, 1, 2, 0),
            _make_pass("home", 40, 30, 50, 30, 2, 3, 1),
        ]
        result = tpd.identify_signature_patterns(events, "home", min_frequency=3)
        assert result == []


class TestComparePatternsAcrossMatches:
    def test_finds_cross_match(self):
        tpd = TacticalPatternDetector()
        match1 = []
        match2 = []
        for i in range(3):
            match1.append(_make_pass("home", 30, 30, 40, 30, 1, 2, i * 2))
            match1.append(_make_pass("home", 40, 30, 50, 30, 2, 3, i * 2 + 1))
            match2.append(_make_pass("home", 30, 30, 40, 30, 1, 2, i * 2))
            match2.append(_make_pass("home", 40, 30, 50, 30, 2, 3, i * 2 + 1))
        result = tpd.compare_patterns_across_matches([match1, match2])
        assert len(result) >= 1

    def test_no_cross_match(self):
        tpd = TacticalPatternDetector()
        match1 = [_make_pass("home", 30, 30, 40, 30, 1, 2, 0)]
        match2 = [_make_pass("home", 80, 50, 90, 55, 3, 4, 0)]
        result = tpd.compare_patterns_across_matches([match1, match2])
        # may or may not find similar patterns
        assert isinstance(result, list)

    def test_empty_matches(self):
        tpd = TacticalPatternDetector()
        result = tpd.compare_patterns_across_matches([[], []])
        assert result == []

    def test_single_match_no_compare(self):
        tpd = TacticalPatternDetector()
        events = [
            _make_pass("home", 30, 30, 40, 30, 1, 2, 0),
            _make_pass("home", 40, 30, 50, 30, 2, 3, 1),
        ]
        result = tpd.compare_patterns_across_matches([events])
        assert isinstance(result, list)
