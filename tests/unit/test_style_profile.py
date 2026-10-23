"""Tests for the style profile (core/style_profile.py).

The definitions are the product: a PPDA that counts pressure events
reads ~2 for every team, and a progressive pass counted from the own
box reads ~100 per team. Both were measured on the real cohort before
this module shipped, so these tests pin the definitions that produced
publishable ranges — not merely that the code runs.

Hand-computed fixtures: every expected number below comes from the
event list in the test, not from running the function.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kawkab.core.style_profile import (  # noqa: E402
    HIGH_TURNOVER_X,
    MIN_DEFENSIVE_ACTIONS,
    MIN_FINAL_THIRD_PASSES,
    MIN_LOCATED_DEFENSIVE,
    MIN_LOCATED_ENTRIES,
    MIN_LOCATED_PASSES,
    MIN_OPPONENT_BUILD_PASSES,
    build_profile,
    unavailable_reason,
)
from kawkab.core.validation.statsbomb_loader import (  # noqa: E402
    load_statsbomb_match,
    normalize_statsbomb_events,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS = PROJECT_ROOT / "data" / "statsbomb_corpus"


def ev(event_type="pass", x=60.0, y=34.0, end_x=72.0, end_y=34.0, completed=True, won=None):
    return {
        "type": event_type,
        "x": x,
        "y": y,
        "end_x": end_x,
        "end_y": end_y,
        "completed": completed,
        "won": won,
    }


def passes(n, x=60.0, end_x=72.0, completed=True):
    return [ev("pass", x=x, end_x=end_x, completed=completed) for _ in range(n)]


class TestPPDA:
    def test_opponent_build_passes_per_defensive_action(self):
        own = [ev("interception", x=50.0) for _ in range(5)]
        opponent = passes(MIN_OPPONENT_BUILD_PASSES, x=40.0, end_x=48.0)
        profile = build_profile(own, opponent, 90.0)
        assert profile["metrics"]["ppda"] == pytest.approx(
            MIN_OPPONENT_BUILD_PASSES / MIN_DEFENSIVE_ACTIONS
        )

    def test_pressure_events_are_not_defensive_actions(self):
        """Counting pressures drives PPDA to ~2 regardless of pressing."""
        own = [ev("interception", x=50.0) for _ in range(5)]
        own += [ev("pressure", x=50.0) for _ in range(200)]
        opponent = passes(MIN_OPPONENT_BUILD_PASSES, x=40.0, end_x=48.0)
        profile = build_profile(own, opponent, 90.0)
        assert profile["metrics"]["ppda"] == pytest.approx(
            MIN_OPPONENT_BUILD_PASSES / MIN_DEFENSIVE_ACTIONS
        )

    def test_too_few_actions_is_unavailable_with_the_count(self):
        own = [ev("interception", x=50.0) for _ in range(2)]
        opponent = passes(MIN_OPPONENT_BUILD_PASSES, x=40.0, end_x=48.0)
        profile = build_profile(own, opponent, 90.0)
        assert profile["metrics"]["ppda"] is None
        assert profile["coverage"]["ppda"]["n"] == 2
        assert "needs at least" in unavailable_reason("ppda", profile)

    def test_too_few_opponent_build_passes_is_unavailable(self):
        own = [ev("interception", x=50.0) for _ in range(5)]
        opponent = passes(3, x=40.0, end_x=48.0)
        profile = build_profile(own, opponent, 90.0)
        assert profile["metrics"]["ppda"] is None
        assert profile["coverage"]["ppda"]["n"] == 3


class TestFieldTilt:
    def test_share_of_final_third_completions(self):
        own = passes(15, x=75.0, end_x=78.0)
        opponent = passes(5, x=75.0, end_x=78.0)
        profile = build_profile(own, opponent, 90.0)
        assert profile["metrics"]["field_tilt"] == pytest.approx(0.75)

    def test_below_the_floor_is_unavailable(self):
        own = passes(3, x=75.0, end_x=78.0)
        opponent = passes(3, x=75.0, end_x=78.0)
        profile = build_profile(own, opponent, 90.0)
        assert profile["metrics"]["field_tilt"] is None
        assert profile["coverage"]["field_tilt"]["n"] == 6
        assert profile["coverage"]["field_tilt"]["minimum"] == MIN_FINAL_THIRD_PASSES


class TestProgressivePasses:
    def test_halfway_rule(self):
        own = [
            ev("pass", x=60.0, end_x=72.0),  # +12 from beyond halfway: counts
            ev("pass", x=40.0, end_x=50.0),  # +10 but from the own half: does not
        ] + passes(MIN_LOCATED_PASSES - 2, x=60.0, end_x=62.0)
        profile = build_profile(own, [], 90.0)
        # one progressive pass over 90 minutes
        assert profile["metrics"]["progressive_passes_per_90"] == pytest.approx(1.0)

    def test_pass_into_the_box_counts_from_anywhere(self):
        own = [ev("pass", x=55.0, end_x=90.0, end_y=34.0)]
        own += passes(MIN_LOCATED_PASSES - 1, x=60.0, end_x=62.0)
        profile = build_profile(own, [], 90.0)
        assert profile["metrics"]["progressive_passes_per_90"] == pytest.approx(1.0)

    def test_incomplete_pass_is_not_progressive(self):
        own = [ev("pass", x=60.0, end_x=72.0, completed=False)]
        own += passes(MIN_LOCATED_PASSES - 1, x=60.0, end_x=62.0)
        profile = build_profile(own, [], 90.0)
        assert profile["metrics"]["progressive_passes_per_90"] == pytest.approx(0.0)

    def test_unknown_outcomes_are_refused(self):
        own = [ev("pass", completed=None) for _ in range(MIN_LOCATED_PASSES)]
        profile = build_profile(own, [], 90.0)
        assert profile["metrics"]["progressive_passes_per_90"] is None
        assert "recorded outcome" in profile["coverage"]["progressive_passes_per_90"]["unit"]

    def test_half_a_match_doubles_the_rate(self):
        own = [ev("pass", x=60.0, end_x=72.0)]
        own += passes(MIN_LOCATED_PASSES - 1, x=60.0, end_x=62.0)
        profile = build_profile(own, [], 45.0)
        assert profile["metrics"]["progressive_passes_per_90"] == pytest.approx(2.0)


class TestFinalThirdEntries:
    def test_completed_passes_and_carries_count(self):
        own = [
            ev("pass", x=60.0, end_x=72.0),
            ev("carry", x=68.0, end_x=71.0),
            ev("pass", x=60.0, end_x=72.0, completed=False),
        ]
        own += passes(MIN_LOCATED_ENTRIES - 3, x=60.0, end_x=62.0)
        profile = build_profile(own, [], 90.0)
        assert profile["metrics"]["final_third_entries_per_90"] == pytest.approx(2.0)

    def test_below_the_floor_is_unavailable(self):
        profile = build_profile(passes(10, x=60.0, end_x=72.0), [], 90.0)
        assert profile["metrics"]["final_third_entries_per_90"] is None
        assert profile["coverage"]["final_third_entries_per_90"]["minimum"] == MIN_LOCATED_ENTRIES


class TestHighTurnovers:
    def test_only_wins_inside_the_final_forty_metres(self):
        own = [
            ev("interception", x=HIGH_TURNOVER_X + 1),
            ev("interception", x=40.0),
            ev("recovery", x=HIGH_TURNOVER_X + 5),
            ev("duel", x=HIGH_TURNOVER_X + 2, won=True),
            ev("duel", x=HIGH_TURNOVER_X + 2, won=False),
        ]
        own += [ev("interception", x=20.0) for _ in range(MIN_LOCATED_DEFENSIVE)]
        profile = build_profile(own, [], 90.0)
        # interception + recovery + won duel = 3
        assert profile["metrics"]["high_turnovers_per_90"] == pytest.approx(3.0)

    def test_below_the_floor_is_unavailable(self):
        profile = build_profile([ev("interception", x=70.0)] * 3, [], 90.0)
        assert profile["metrics"]["high_turnovers_per_90"] is None


class TestPassCompletion:
    def test_completed_over_attempted(self):
        own = passes(3, x=60.0, end_x=62.0) + [ev("pass", x=60.0, end_x=62.0, completed=False)]
        own += passes(MIN_LOCATED_PASSES - 4, x=60.0, end_x=62.0)
        profile = build_profile(own, [], 90.0)
        assert profile["metrics"]["pass_completion_pct"] == pytest.approx(
            100.0 * (MIN_LOCATED_PASSES - 1) / MIN_LOCATED_PASSES
        )

    def test_partial_outcomes_are_refused_not_averaged(self):
        own = passes(MIN_LOCATED_PASSES, x=60.0, end_x=62.0)
        own += [ev("pass", completed=None)]
        profile = build_profile(own, [], 90.0)
        assert profile["metrics"]["pass_completion_pct"] is None
        assert profile["coverage"]["pass_completion_pct"]["minimum"] == len(own)


class TestLoaderStream:
    def test_corpus_events_normalize_to_the_shared_shape(self):
        files = sorted(CORPUS.glob("*.json"))
        if not files:
            pytest.skip("statsbomb corpus not on this machine")
        match = load_statsbomb_match(files[0])
        assert len(match.team_events) == 2
        for events in match.team_events.values():
            assert events
            for event in events[:50]:
                assert event["type"] in {
                    "pass", "carry", "pressure", "interception", "duel",
                    "recovery", "clearance", "shot", "foul",
                }
                if event["x"] is not None:
                    assert 0.0 <= event["x"] <= 105.5
                    assert 0.0 <= event["y"] <= 68.5
        passes_seen = [
            event for events in match.team_events.values() for event in events
            if event["type"] == "pass"
        ]
        assert passes_seen
        assert all(isinstance(event["completed"], bool) for event in passes_seen)

    def test_normalizer_drops_unmapped_types(self):
        stream = normalize_statsbomb_events(
            [
                {"type": {"name": "Starting XI"}, "team": {"name": "A"}},
                {"type": {"name": "Pass"}, "team": {"name": "A"},
                 "location": [60.0, 40.0], "pass": {"end_location": [80.0, 40.0]}},
            ]
        )
        assert list(stream) == ["A"]
        assert len(stream["A"]) == 1
        assert stream["A"][0]["completed"] is True
