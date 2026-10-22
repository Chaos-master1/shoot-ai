"""Tests for form_by_competition and form_by_opponent_strength."""

from __future__ import annotations

import pytest

from kawkab.core.form_analysis import form_by_competition, form_by_opponent_strength


def make_match(
    home: str,
    away: str,
    hg: int,
    ag: int,
    competition_type: str = "league",
    opponent_strength: float | None = None,
) -> dict:
    m = {
        "home_team": home,
        "away_team": away,
        "home_goals": hg,
        "away_goals": ag,
        "competition_type": competition_type,
    }
    if opponent_strength is not None:
        m["opponent_strength"] = opponent_strength
    return m


class TestFormByCompetition:
    def test_league_vs_cup_split(self):
        matches = [
            make_match("Team A", "Team B", 2, 0, "league"),
            make_match("Team C", "Team A", 1, 0, "league"),
            make_match("Team A", "Team D", 3, 1, "cup"),
        ]
        result = form_by_competition("Team A", matches)
        assert result["league"]["played"] == 2
        assert result["league"]["won"] == 1
        assert result["cup"]["played"] == 1
        assert result["cup"]["won"] == 1

    def test_friendly_results_excluded_from_league(self):
        matches = [
            make_match("Team A", "Team B", 1, 0, "league"),
            make_match("Team A", "Team B", 0, 0, "friendly"),
        ]
        result = form_by_competition("Team A", matches)
        assert result["league"]["played"] == 1
        assert result["friendly"]["played"] == 1

    def test_recent_form_sequences_correct(self):
        matches = [
            make_match("Team A", "Team B", 2, 0, "league"),
            make_match("Team A", "Team C", 1, 1, "league"),
            make_match("Team D", "Team A", 0, 0, "league"),
        ]
        result = form_by_competition("Team A", matches)
        assert result["recent_form"]["league"] == ["W", "D", "D"]

    def test_empty_matches_returns_empty_dicts(self):
        result = form_by_competition("Team A", [])
        for ct in ("league", "cup", "friendly", "continental"):
            assert result[ct]["played"] == 0
            assert result[ct]["won"] == 0
            assert result[ct]["points_per_game"] == 0.0

    def test_single_match_data(self):
        matches = [make_match("Team A", "Team B", 2, 0, "league")]
        result = form_by_competition("Team A", matches)
        assert result["league"]["played"] == 1
        assert result["league"]["won"] == 1
        assert result["league"]["goals_for"] == 2
        assert result["league"]["goals_against"] == 0
        assert result["league"]["win_pct"] == 100.0

    def test_points_per_game_correct(self):
        matches = [
            make_match("Team A", "Team B", 2, 0, "league"),
            make_match("Team A", "Team C", 1, 1, "league"),
            make_match("Team D", "Team A", 0, 0, "league"),
        ]
        result = form_by_competition("Team A", matches)
        assert result["league"]["points_per_game"] == pytest.approx(1.667, abs=0.001)

    def test_goals_for_against_across_competitions(self):
        matches = [
            make_match("Team A", "Team B", 2, 0, "league"),
            make_match("Team C", "Team A", 1, 3, "league"),
        ]
        result = form_by_competition("Team A", matches)
        assert result["league"]["goals_for"] == 5
        assert result["league"]["goals_against"] == 1

    def test_continental_matches_separate(self):
        matches = [
            make_match("Team A", "Team B", 2, 1, "continental"),
            make_match("Team C", "Team A", 0, 0, "continental"),
            make_match("Team A", "Team D", 1, 0, "league"),
        ]
        result = form_by_competition("Team A", matches)
        assert result["continental"]["played"] == 2
        assert result["league"]["played"] == 1
        assert result["continental"]["won"] == 1

    def test_win_pct_calculation(self):
        matches = [
            make_match("Team A", "Team B", 2, 0, "league"),
            make_match("Team A", "Team C", 1, 0, "league"),
            make_match("Team D", "Team A", 0, 0, "league"),
            make_match("Team A", "Team E", 0, 1, "league"),
        ]
        result = form_by_competition("Team A", matches)
        assert result["league"]["win_pct"] == 50.0


class TestFormByOpponentStrength:
    def test_opponent_strength_tiers(self):
        matches = [
            make_match("Team A", "Team B", 2, 0, opponent_strength=1.8),
            make_match("Team A", "Team C", 1, 0, opponent_strength=0.6),
            make_match("Team D", "Team A", 0, 0, opponent_strength=0.1),
        ]
        result = form_by_opponent_strength("Team A", matches)
        assert "top" in result
        assert "mid" in result
        assert "bottom" in result

    def test_top_tier_opponents_lower_win_pct(self):
        matches = [
            make_match("Team A", "Team B", 0, 0, opponent_strength=1.9),
            make_match("Team A", "Team C", 1, 2, opponent_strength=1.8),
        ]
        result = form_by_opponent_strength("Team A", matches)
        assert result["top"]["played"] == 2
        assert result["top"]["won"] == 0

    def test_empty_matches_returns_empty_dicts(self):
        result = form_by_opponent_strength("Team A", [])
        for tier in ("top", "mid", "bottom"):
            assert result[tier]["played"] == 0

    def test_single_match_correct(self):
        matches = [make_match("Team A", "Team B", 2, 0, opponent_strength=1.5)]
        result = form_by_opponent_strength("Team A", matches)
        assert result["top"]["played"] == 1
        assert result["top"]["won"] == 1
        assert result["top"]["goals_for"] == 2

    def test_mid_tier_stats(self):
        matches = [
            make_match("Team A", "Team B", 2, 1, opponent_strength=0.5),
            make_match("Team A", "Team C", 1, 1, opponent_strength=0.45),
        ]
        result = form_by_opponent_strength("Team A", matches)
        assert result["mid"]["played"] == 2
        assert result["mid"]["won"] == 1
        assert result["mid"]["points_per_game"] == pytest.approx(4.0 / 2.0)

    def test_bottom_tier_dominance(self):
        matches = [
            make_match("Team A", "Team B", 4, 0, opponent_strength=0.1),
            make_match("Team A", "Team C", 3, 0, opponent_strength=0.05),
        ]
        result = form_by_opponent_strength("Team A", matches)
        assert result["bottom"]["won"] == 2
        assert result["bottom"]["goals_for"] == 7
        assert result["bottom"]["goals_against"] == 0

    def test_custom_strength_tiers(self):
        matches = [
            make_match("Team A", "Team B", 2, 0, opponent_strength=2.0),
            make_match("Team A", "Team C", 1, 0, opponent_strength=1.0),
        ]
        tiers = [("strong", 0.5, float("inf")), ("weak", 0.0, 0.5)]
        result = form_by_opponent_strength("Team A", matches, strength_tiers=tiers)
        assert "strong" in result
        assert "weak" in result
        assert result["strong"]["played"] == 2
