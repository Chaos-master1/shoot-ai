"""Tests for Pass Pattern Clustering module."""

from kawkab.core.pass_patterns import PassPatternAnalyzer


def _make_pass_event(
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


class TestExtractPassSequences:
    def test_basic_extraction(self):
        events = [
            _make_pass_event("home", 30, 30, 40, 30, 1, 2, 0),
            _make_pass_event("home", 40, 30, 50, 30, 2, 3, 1),
            _make_pass_event("home", 50, 30, 60, 30, 3, 4, 2),
        ]
        ppa = PassPatternAnalyzer()
        seqs = ppa.extract_pass_sequences(events)
        assert len(seqs) == 1
        assert len(seqs[0]) == 3

    def test_team_change_breaks_sequence(self):
        events = [
            _make_pass_event("home", 30, 30, 40, 30, 1, 2, 0),
            _make_pass_event("away", 40, 30, 50, 30, 5, 6, 1),
            _make_pass_event("home", 50, 30, 60, 30, 3, 4, 2),
        ]
        ppa = PassPatternAnalyzer()
        seqs = ppa.extract_pass_sequences(events)
        assert len(seqs) == 0

    def test_empty_events(self):
        ppa = PassPatternAnalyzer()
        seqs = ppa.extract_pass_sequences([])
        assert seqs == []

    def test_single_pass_no_sequence(self):
        events = [_make_pass_event("home", 30, 30, 40, 30, 1, 2, 0)]
        ppa = PassPatternAnalyzer()
        seqs = ppa.extract_pass_sequences(events)
        assert seqs == []

    def test_max_length_respected(self):
        events = [
            _make_pass_event("home", 10, 30, 20, 30, 1, 2, 0),
            _make_pass_event("home", 20, 30, 30, 30, 2, 3, 1),
            _make_pass_event("home", 30, 30, 40, 30, 3, 4, 2),
            _make_pass_event("home", 40, 30, 50, 30, 4, 5, 3),
            _make_pass_event("home", 50, 30, 60, 30, 5, 6, 4),
        ]
        ppa = PassPatternAnalyzer()
        seqs = ppa.extract_pass_sequences(events, max_length=3)
        assert len(seqs) == 2
        assert all(len(s) <= 3 for s in seqs)


class TestClassifySequencePattern:
    def test_build_up_center(self):
        seq = [
            _make_pass_event("home", 20, 34, 30, 34, 1, 2, 0),
            _make_pass_event("home", 30, 34, 35, 34, 2, 3, 1),
        ]
        ppa = PassPatternAnalyzer()
        pattern = ppa.classify_sequence_pattern(seq)
        assert pattern == "build_up_center"

    def test_switch_of_play(self):
        seq = [
            _make_pass_event("home", 30, 10, 40, 10, 1, 2, 0),
            _make_pass_event("home", 40, 10, 50, 55, 2, 3, 1),
        ]
        ppa = PassPatternAnalyzer()
        pattern = ppa.classify_sequence_pattern(seq)
        assert pattern == "switch_of_play"

    def test_cross_sequence(self):
        seq = [
            _make_pass_event("home", 70, 40, 85, 42, 1, 2, 0),
            _make_pass_event("home", 85, 42, 95, 5, 2, 3, 1),
        ]
        ppa = PassPatternAnalyzer()
        pattern = ppa.classify_sequence_pattern(seq)
        assert pattern == "cross_sequence"

    def test_progressive_left(self):
        seq = [
            _make_pass_event("home", 30, 5, 50, 20, 1, 2, 0),
            _make_pass_event("home", 50, 20, 72, 22, 2, 3, 1),
        ]
        ppa = PassPatternAnalyzer()
        pattern = ppa.classify_sequence_pattern(seq)
        assert pattern == "progressive_left"

    def test_combination_play(self):
        seq = [
            _make_pass_event("home", 50, 34, 55, 34, 1, 2, 0),
            _make_pass_event("home", 55, 34, 52, 34, 2, 1, 1),
        ]
        ppa = PassPatternAnalyzer()
        pattern = ppa.classify_sequence_pattern(seq)
        assert pattern == "combination_play"


class TestClusterSequencesByPattern:
    def test_returns_counts(self):
        events = [
            _make_pass_event("home", 20, 34, 30, 34, 1, 2, 0),
            _make_pass_event("home", 30, 34, 35, 34, 2, 3, 1),
            _make_pass_event("home", 80, 10, 90, 55, 4, 5, 2),
        ]
        ppa = PassPatternAnalyzer()
        result = ppa.cluster_sequences_by_pattern(events)
        assert "home" in result
        total_count = sum(v["count"] for v in result["home"].values())
        assert total_count > 0


class TestDetectBuildUpPattern:
    def test_major_side(self):
        events = [
            _make_pass_event("home", 15, 15, 22, 18, 1, 2, 0),
            _make_pass_event("home", 22, 18, 30, 20, 2, 3, 1),
            _make_pass_event("home", 30, 20, 35, 22, 3, 4, 2),
            _make_pass_event("home", 10, 50, 18, 48, 5, 6, 3),
            _make_pass_event("home", 18, 48, 25, 46, 6, 7, 4),
        ]
        ppa = PassPatternAnalyzer()
        result = ppa.detect_build_up_pattern(events, "home")
        assert result["primary_side"] in ("left", "center", "right")
        assert result["avg_passes_per_build_up"] > 0

    def test_empty_events(self):
        ppa = PassPatternAnalyzer()
        result = ppa.detect_build_up_pattern([], "home")
        assert result["primary_side"] == "center"
        assert result["avg_passes_per_build_up"] == 0


class TestComputeCombinationPlayFrequency:
    def test_detects_one_two(self):
        events = [
            _make_pass_event("home", 50, 34, 55, 34, 1, 2, 0),
            _make_pass_event("home", 55, 34, 52, 34, 2, 1, 1),
        ]
        ppa = PassPatternAnalyzer()
        result = ppa.compute_combination_play_frequency(events, "home")
        assert result["total_combinations"] >= 1

    def test_no_combos(self):
        events = [
            _make_pass_event("home", 50, 34, 55, 34, 1, 2, 0),
            _make_pass_event("home", 55, 34, 60, 34, 2, 3, 1),
        ]
        ppa = PassPatternAnalyzer()
        result = ppa.compute_combination_play_frequency(events, "home")
        assert result["total_combinations"] == 0


class TestEdgeCases:
    def test_empty_events_safe_defaults(self):
        ppa = PassPatternAnalyzer()
        assert ppa.extract_pass_sequences([]) == []
        assert ppa.cluster_sequences_by_pattern([]) == {}
        result = ppa.detect_build_up_pattern([], "home")
        assert result["primary_side"] == "center"

    def test_single_pass_not_classified_as_sequence(self):
        ppa = PassPatternAnalyzer()
        seqs = ppa.extract_pass_sequences([_make_pass_event("home", 30, 30, 40, 30, 1, 2, 0)])
        assert seqs == []
