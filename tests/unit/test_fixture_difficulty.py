"""Tests for Fixture Difficulty — opponent strength, stretches, density."""

from kawkab.core.fixture_difficulty import (
    _difficulty_color,
    analyze_fixture_difficulty,
)


class TestDifficultyColor:
    def test_green_below_40(self):
        assert _difficulty_color(30) == "green"

    def test_yellow_40_to_65(self):
        assert _difficulty_color(50) == "yellow"

    def test_yellow_at_65(self):
        assert _difficulty_color(65) == "yellow"

    def test_red_above_65(self):
        assert _difficulty_color(80) == "red"

    def test_red_at_100(self):
        assert _difficulty_color(100) == "red"


class TestAnalyzeFixtureDifficulty:
    def test_empty_fixtures(self):
        r = analyze_fixture_difficulty("team_a", [], {})
        assert r.team_id == "team_a"
        assert r.fixtures == []
        assert r.avg_difficulty == 0.0
        assert r.home_away_balance == 0.0

    def test_single_home_fixture(self):
        fixtures = [{"opponent_id": "opp_b", "venue": "home", "date": "2026-01-15"}]
        strength = {"opp_b": 80.0}
        r = analyze_fixture_difficulty("team_a", fixtures, strength)
        assert len(r.fixtures) == 1
        assert r.fixtures[0]["venue"] == "home"
        assert r.fixtures[0]["difficulty_score"] < 80.0

    def test_single_away_fixture(self):
        fixtures = [{"opponent_id": "opp_b", "venue": "away", "date": "2026-01-15"}]
        strength = {"opp_b": 80.0}
        r = analyze_fixture_difficulty("team_a", fixtures, strength)
        assert r.fixtures[0]["venue"] == "away"
        assert r.fixtures[0]["difficulty_score"] > 80.0

    def test_multiple_fixtures_avg(self):
        fixtures = [
            {"opponent_id": "opp_b", "venue": "home", "date": "2026-01-15"},
            {"opponent_id": "opp_c", "venue": "away", "date": "2026-01-22"},
            {"opponent_id": "opp_d", "venue": "home", "date": "2026-01-29"},
        ]
        strength = {"opp_b": 50.0, "opp_c": 70.0, "opp_d": 30.0}
        r = analyze_fixture_difficulty("team_a", fixtures, strength)
        assert len(r.fixtures) == 3
        assert 1.0 <= r.avg_difficulty <= 100.0

    def test_home_away_balance(self):
        fixtures = [
            {"opponent_id": "opp_b", "venue": "home", "date": "2026-01-15"},
            {"opponent_id": "opp_c", "venue": "away", "date": "2026-01-22"},
            {"opponent_id": "opp_d", "venue": "home", "date": "2026-01-29"},
        ]
        strength = {"opp_b": 50.0, "opp_c": 50.0, "opp_d": 50.0}
        r = analyze_fixture_difficulty("team_a", fixtures, strength)
        assert r.home_away_balance == round(2 / 3, 2)

    def test_all_away(self):
        fixtures = [
            {"opponent_id": "opp_b", "venue": "away", "date": "2026-01-15"},
            {"opponent_id": "opp_c", "venue": "away", "date": "2026-01-22"},
        ]
        strength = {"opp_b": 50.0, "opp_c": 50.0}
        r = analyze_fixture_difficulty("team_a", fixtures, strength)
        assert r.home_away_balance == 0.0

    def test_color_green_for_easy(self):
        fixtures = [{"opponent_id": "opp_b", "venue": "home", "date": "2026-01-15"}]
        strength = {"opp_b": 10.0}
        r = analyze_fixture_difficulty("team_a", fixtures, strength)
        assert r.fixtures[0]["color"] == "green"

    def test_color_red_for_hard(self):
        fixtures = [{"opponent_id": "opp_b", "venue": "away", "date": "2026-01-15"}]
        strength = {"opp_b": 90.0}
        r = analyze_fixture_difficulty("team_a", fixtures, strength)
        assert r.fixtures[0]["color"] == "red"

    def test_hardest_stretch_three_fixtures(self):
        fixtures = [
            {"opponent_id": "opp_b", "venue": "home", "date": "2026-01-15"},
            {"opponent_id": "opp_c", "venue": "away", "date": "2026-01-22"},
            {"opponent_id": "opp_d", "venue": "away", "date": "2026-01-29"},
        ]
        strength = {"opp_b": 90.0, "opp_c": 80.0, "opp_d": 70.0}
        r = analyze_fixture_difficulty("team_a", fixtures, strength)
        assert "Hardest" in r.hardest_stretch

    def test_schedule_density_with_dates(self):
        fixtures = [
            {"opponent_id": "opp_b", "venue": "home", "date": "2026-01-15"},
            {"opponent_id": "opp_c", "venue": "away", "date": "2026-01-22"},
            {"opponent_id": "opp_d", "venue": "away", "date": "2026-01-29"},
        ]
        strength = {"opp_b": 50.0, "opp_c": 50.0, "opp_d": 50.0}
        r = analyze_fixture_difficulty("team_a", fixtures, strength)
        assert r.schedule_density == 7.0

    def test_schedule_density_no_dates(self):
        fixtures = [
            {"opponent_id": "opp_b", "venue": "home"},
            {"opponent_id": "opp_c", "venue": "away"},
        ]
        strength = {"opp_b": 50.0, "opp_c": 50.0}
        r = analyze_fixture_difficulty("team_a", fixtures, strength)
        assert r.schedule_density == 0.0

    def test_unknown_opponent_default_strength(self):
        fixtures = [{"opponent_id": "unknown", "venue": "home", "date": "2026-01-15"}]
        r = analyze_fixture_difficulty("team_a", fixtures, {})
        assert r.fixtures[0]["difficulty_score"] == 42.5

    def test_custom_home_advantage(self):
        fixtures = [{"opponent_id": "opp_b", "venue": "home", "date": "2026-01-15"}]
        strength = {"opp_b": 50.0}
        r = analyze_fixture_difficulty("team_a", fixtures, strength, home_advantage=0.25)
        assert r.fixtures[0]["difficulty_score"] == 37.5

    def test_weight_in_fixtures(self):
        fixtures = [{"opponent_id": "opp_b", "venue": "home", "date": "2026-01-15"}]
        strength = {"opp_b": 50.0}
        r = analyze_fixture_difficulty("team_a", fixtures, strength)
        assert "weight" in r.fixtures[0]
        assert 0.0 < r.fixtures[0]["weight"] <= 1.0
