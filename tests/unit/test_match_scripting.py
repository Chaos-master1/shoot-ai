"""Tests for match scripting."""

import random

import pytest

from kawkab.core.match_scripting import (
    generate_counter_attack_phase,
    generate_match_script,
    generate_possession_phase,
    generate_pressing_phase,
)


class TestMatchScripting:
    @pytest.fixture(autouse=True)
    def _seeded_random(self):
        """match_scripting.py generates event positions/outcomes via the
        unseeded global `random` module. Without a fixed seed, statistical
        assertions below (e.g. "home_events > away_events" for a 5:1
        possession-weighted template) pass the vast majority of runs but
        fail intermittently by chance -- flaky in the full suite while
        always passing alone, since collection order shifts which prior
        tests already consumed random draws from the same global state.
        Seeding makes every test in this file deterministic and immune to
        run order."""
        state = random.getstate()
        random.seed(20260729)
        yield
        random.setstate(state)

    def test_possession_phase(self):
        phase = generate_possession_phase("home", 0, 3, 0.5)
        assert phase.name == "Home Possession"
        assert phase.start_minute == 0
        assert phase.end_minute == 3
        assert len(phase.events) >= 5
        assert all(e.team == "home" for e in phase.events)

    def test_pressing_phase(self):
        phase = generate_pressing_phase("away", 30, 2, 0.9)
        assert phase.name == "Away High Press"
        assert phase.start_minute == 30
        assert phase.end_minute == 32
        assert len(phase.events) >= 3

    def test_counter_attack_phase(self):
        phase = generate_counter_attack_phase("home", 45)
        assert phase.name == "Counter Attack"
        assert len(phase.events) >= 4
        event_types = [e.event_type for e in phase.events]
        assert "shot" in event_types
        assert "tackle" in event_types

    def test_generate_match_script_balanced(self):
        script = generate_match_script("balanced", "Team A", "Team B")
        assert script.home_team == "Team A"
        assert script.away_team == "Team B"
        assert len(script.phases) >= 1
        assert script.total_events >= 5

    def test_generate_match_script_home_dominant(self):
        script = generate_match_script("home_dominant")
        home_events = sum(1 for p in script.phases for e in p.events if e.team == "home")
        away_events = sum(1 for p in script.phases for e in p.events if e.team == "away")
        assert home_events > away_events

    def test_generate_match_script_invalid_template(self):
        script = generate_match_script("nonexistent")
        assert script.title is not None
        assert len(script.phases) >= 1

    def test_all_events_have_valid_coordinates(self):
        script = generate_match_script("high_pressing")
        for phase in script.phases:
            for event in phase.events:
                assert 0 <= event.start_x <= 105
                assert 0 <= event.start_y <= 68
                assert 0 <= event.end_x <= 105
                assert 0 <= event.end_y <= 68

    def test_serialization(self):
        script = generate_match_script("balanced", "Test H", "Test A")
        d = script.to_dict()
        assert d["home_team"] == "Test H"
        assert d["away_team"] == "Test A"
        assert isinstance(d["phases"], list)
        assert d["total_events"] > 0
