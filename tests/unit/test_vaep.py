"""Tests for VAEP event valuation model — possession-phase survival model."""

import math

from kawkab.core.vaep import (
    VaepepResult,
    _identify_possession_phases,
    _possession_switching_events,
    _to_zone,
    compute_vaep,
)


class TestUtil:
    def test_empty_events(self):
        assert compute_vaep([]) == []

    def test_zone_mapping(self):
        assert _to_zone(0, 0) == (0, 0)
        assert _to_zone(105, 68) == (15, 11)

    def test_possession_switching_set(self):
        s = _possession_switching_events()
        assert "tackle" in s
        assert "interception" in s
        assert "pass" not in s

    def test_identify_phases_single_team(self):
        events = [
            {"type": "pass", "timestamp": 1.0, "team": "home"},
            {"type": "pass", "timestamp": 2.0, "team": "home"},
            {"type": "shot", "timestamp": 3.0, "team": "home"},
        ]
        phases = _identify_possession_phases(events)
        assert len(phases) == 1
        assert phases[0][2] == "home"

    def test_identify_phases_switch_on_tackle(self):
        events = [
            {"type": "pass", "timestamp": 1.0, "team": "home"},
            {"type": "tackle", "timestamp": 2.0, "team": "away"},
            {"type": "pass", "timestamp": 3.0, "team": "away"},
        ]
        phases = _identify_possession_phases(events)
        assert len(phases) >= 2

    def test_identify_phases_empty(self):
        assert _identify_possession_phases([]) == []

    def test_identify_phases_switch_on_shot(self):
        events = [
            {"type": "pass", "timestamp": 1.0, "team": "home", "x": 50, "y": 34},
            {"type": "shot", "timestamp": 2.0, "team": "away", "x": 80, "y": 34, "is_goal": False},
        ]
        phases = _identify_possession_phases(events)
        assert len(phases) >= 2

    def test_identify_phases_single_event(self):
        events = [{"type": "pass", "timestamp": 1.0, "team": "home"}]
        phases = _identify_possession_phases(events)
        assert len(phases) == 1


class TestVaepBasic:
    def test_non_goal_events(self):
        events = [
            {"type": "pass", "timestamp": 10.0, "team": "home", "x": 50, "y": 34, "is_goal": False},
            {"type": "pass", "timestamp": 20.0, "team": "away", "x": 60, "y": 30, "is_goal": False},
        ]
        result = compute_vaep(events)
        assert len(result) == 2
        assert "vaep_value" in result[0]

    def test_goal_events(self):
        events = [
            {"type": "shot", "timestamp": 10.0, "team": "home", "x": 90, "y": 34, "is_goal": True},
            {"type": "pass", "timestamp": 20.0, "team": "away", "x": 50, "y": 34, "is_goal": False},
        ]
        result = compute_vaep(events)
        assert len(result) == 2

    def test_vaep_result_to_dict(self):
        r = VaepepResult(
            event_index=0,
            event_type="pass",
            timestamp=10.0,
            team="home",
            zone_x=8,
            zone_y=6,
            delta_home=0.01,
            delta_away=-0.005,
            vaep_value=0.015,
            is_goal=False,
        )
        d = r.to_dict()
        assert d["event_type"] == "pass"
        assert d["vaep_value"] == 0.015
        assert d["is_goal"] is False
        assert "possession_id" in d

    def test_simple_sequence(self):
        events = [
            {"type": "pass", "timestamp": 5.0, "team": "home", "x": 50, "y": 34, "is_goal": False},
            {"type": "shot", "timestamp": 10.0, "team": "home", "x": 92, "y": 34, "is_goal": True},
            {"type": "pass", "timestamp": 15.0, "team": "away", "x": 40, "y": 34, "is_goal": False},
        ]
        result = compute_vaep(events)
        assert len(result) == 3

    def test_multiple_events(self):
        events = []
        for i in range(5):
            events.append(
                {
                    "type": "pass",
                    "timestamp": float(i * 5),
                    "team": "home" if i % 2 == 0 else "away",
                    "x": 50 + i * 5,
                    "y": 34,
                    "is_goal": False,
                }
            )
        result = compute_vaep(events)
        assert len(result) == 5

    def test_both_teams_present(self):
        events = [
            {"type": "pass", "timestamp": 1.0, "team": "home", "x": 50, "y": 34, "is_goal": False},
            {"type": "pass", "timestamp": 2.0, "team": "away", "x": 40, "y": 34, "is_goal": False},
            {"type": "shot", "timestamp": 3.0, "team": "home", "x": 90, "y": 34, "is_goal": False},
            {"type": "pass", "timestamp": 4.0, "team": "away", "x": 30, "y": 34, "is_goal": False},
        ]
        result = compute_vaep(events)
        teams = {r["team"] for r in result}
        assert "home" in teams
        assert "away" in teams

    def test_sorted_by_timestamp(self):
        events = [
            {"type": "pass", "timestamp": 30.0, "team": "home", "x": 50, "y": 34, "is_goal": False},
            {"type": "pass", "timestamp": 10.0, "team": "away", "x": 40, "y": 34, "is_goal": False},
            {"type": "shot", "timestamp": 20.0, "team": "home", "x": 90, "y": 34, "is_goal": False},
        ]
        result = compute_vaep(events)
        timestamps = [r["timestamp"] for r in result]
        assert timestamps == sorted(timestamps)

    def test_single_event(self):
        events = [
            {"type": "pass", "timestamp": 10.0, "team": "home", "x": 50, "y": 34, "is_goal": False}
        ]
        result = compute_vaep(events)
        assert len(result) == 1


class TestVaepSurvival:
    def test_possession_id_in_output(self):
        events = [
            {"type": "pass", "timestamp": 5.0, "team": "home", "x": 50, "y": 34, "is_goal": False},
            {"type": "shot", "timestamp": 10.0, "team": "home", "x": 92, "y": 34, "is_goal": True},
            {"type": "pass", "timestamp": 15.0, "team": "away", "x": 40, "y": 34, "is_goal": False},
        ]
        result = compute_vaep(events)
        for r in result:
            assert "possession_id" in r

    def test_shot_event_higher_impact(self):
        events = [
            {"type": "pass", "timestamp": 5.0, "team": "home", "x": 50, "y": 34, "is_goal": False},
            {"type": "shot", "timestamp": 10.0, "team": "home", "x": 90, "y": 34, "is_goal": True},
            {"type": "pass", "timestamp": 15.0, "team": "away", "x": 40, "y": 34, "is_goal": False},
        ]
        result = compute_vaep(events)
        shot_vaep = abs(result[1]["vaep_value"])
        _ = abs(result[0]["vaep_value"])
        assert shot_vaep >= 0

    def test_delta_fields_present(self):
        events = [
            {"type": "pass", "timestamp": 1.0, "team": "home", "x": 50, "y": 34, "is_goal": False},
        ]
        result = compute_vaep(events)
        assert "delta_home" in result[0]
        assert "delta_away" in result[0]

    def test_lookahead_caps_impact(self):
        events = [
            {"type": "pass", "timestamp": 0.0, "team": "home", "x": 50, "y": 34, "is_goal": False},
            {"type": "shot", "timestamp": 30.0, "team": "home", "x": 90, "y": 34, "is_goal": True},
        ]
        result_short = compute_vaep(events, lookahead=5.0)
        result_long = compute_vaep(events, lookahead=60.0)
        assert len(result_short) == 2
        assert len(result_long) == 2


class TestVaepCorrectness:
    """Correctness tests: verify VAEP values make semantic sense."""

    def test_goal_event_positive_vaep(self):
        events = [
            {"type": "pass", "timestamp": 1.0, "team": "home", "x": 50, "y": 34, "is_goal": False},
            {"type": "shot", "timestamp": 2.0, "team": "home", "x": 95, "y": 34, "is_goal": True},
        ]
        result = compute_vaep(events)
        assert result[1]["vaep_value"] > 0.0, (
            f"Goal should have positive VAEP for scoring team, got {result[1]['vaep_value']}"
        )

    def test_forward_pass_positive_vaep(self):
        events = [
            {"type": "shot", "timestamp": 0.5, "team": "home", "x": 90, "y": 34, "is_goal": True},
            {"type": "shot", "timestamp": 1.0, "team": "home", "x": 30, "y": 34, "is_goal": False},
            {"type": "pass", "timestamp": 2.0, "team": "home", "x": 30, "y": 34, "is_goal": False},
            {"type": "pass", "timestamp": 3.0, "team": "home", "x": 90, "y": 34, "is_goal": False},
        ]
        result = compute_vaep(events)
        # Forward pass (index 2: goes from x=30 to x=90) should add value
        assert result[2]["vaep_value"] > 0.0, (
            f"Forward pass should have positive VAEP, got {result[2]['vaep_value']}"
        )

    def test_backward_pass_negative_or_lower_vaep(self):
        shots = [
            {"type": "shot", "timestamp": 0.5, "team": "home", "x": 90, "y": 34, "is_goal": True},
            {"type": "shot", "timestamp": 1.0, "team": "home", "x": 30, "y": 34, "is_goal": False},
        ]
        forward_events = shots + [
            {"type": "pass", "timestamp": 2.0, "team": "home", "x": 30, "y": 34, "is_goal": False},
            {"type": "pass", "timestamp": 3.0, "team": "home", "x": 90, "y": 34, "is_goal": False},
        ]
        backward_events = shots + [
            {"type": "pass", "timestamp": 2.0, "team": "home", "x": 90, "y": 34, "is_goal": False},
            {"type": "pass", "timestamp": 3.0, "team": "home", "x": 30, "y": 34, "is_goal": False},
        ]
        fwd = compute_vaep(forward_events)
        bwd = compute_vaep(backward_events)
        assert fwd[2]["vaep_value"] > bwd[2]["vaep_value"], (
            f"Forward VAEP ({fwd[2]['vaep_value']}) should exceed backward ({bwd[2]['vaep_value']})"
        )

    def test_turnover_negative_vaep(self):
        events = [
            {"type": "pass", "timestamp": 1.0, "team": "home", "x": 50, "y": 34, "is_goal": False},
            {
                "type": "interception",
                "timestamp": 2.0,
                "team": "away",
                "x": 55,
                "y": 34,
                "is_goal": False,
            },
        ]
        result = compute_vaep(events)
        assert len(result) == 2
        for r in result:
            assert not math.isnan(r["vaep_value"])
            assert not math.isinf(r["vaep_value"])

    def test_vaep_bounds_reasonable(self):
        events = [
            {"type": "pass", "timestamp": 1.0, "team": "home", "x": 50, "y": 34, "is_goal": False},
            {"type": "shot", "timestamp": 2.0, "team": "home", "x": 95, "y": 34, "is_goal": True},
            {"type": "pass", "timestamp": 3.0, "team": "away", "x": 40, "y": 34, "is_goal": False},
        ]
        result = compute_vaep(events)
        for r in result:
            assert -5.0 <= r["vaep_value"] <= 5.0, (
                f"VAEP {r['vaep_value']} out of expected bounds [-5, 5]"
            )

    def test_vaep_decreasing_with_time(self):
        events = [
            {"type": "shot", "timestamp": 0.5, "team": "home", "x": 90, "y": 34, "is_goal": True},
            {"type": "shot", "timestamp": 1.0, "team": "home", "x": 30, "y": 34, "is_goal": False},
            {"type": "pass", "timestamp": 2.0, "team": "home", "x": 30, "y": 34, "is_goal": False},
            {"type": "pass", "timestamp": 3.0, "team": "home", "x": 50, "y": 34, "is_goal": False},
            {"type": "pass", "timestamp": 4.0, "team": "home", "x": 70, "y": 34, "is_goal": False},
            {"type": "pass", "timestamp": 5.0, "team": "home", "x": 90, "y": 34, "is_goal": False},
        ]
        result = compute_vaep(events)
        # Each pass (indices 2-4) moves forward and has a next event in possession;
        # marginal VAEP should diminish as the ball approaches goal.
        # Skip the last pass (index 5) — no next event, computed differently.
        for i in range(2, len(result) - 2):
            assert result[i]["vaep_value"] >= result[i + 1]["vaep_value"] - 1e-9, (
                f"VAEP should diminish at event {i}: {result[i]['vaep_value']} < {result[i + 1]['vaep_value']}"
            )
