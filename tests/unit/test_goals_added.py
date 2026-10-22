"""Tests for Goals Added (g+) — on-ball contribution in goals."""

from kawkab.core.goals_added import (
    _compute_g_plus_from_match,
    _percentile_value,
    compute_goals_added,
)


class TestComputeGPlusFromMatch:
    def test_zero_stats(self):
        m = {"match_id": "m1", "xg": 0, "xa": 0, "xt": 0, "defensive_actions": 0, "obv": 0}
        assert _compute_g_plus_from_match(m) == 0.0

    def test_positive_stats(self):
        m = {"match_id": "m1", "xg": 1.0, "xa": 0.5, "xt": 0.3, "defensive_actions": 5, "obv": 0.2}
        val = _compute_g_plus_from_match(m)
        expected = 1.0 * 1.0 + 0.5 * 0.8 + 0.3 * 0.3 + 5 * 0.02 + 0.2 * 1.0
        assert abs(val - expected) < 0.001

    def test_missing_keys_default_zero(self):
        m = {"match_id": "m1"}
        assert _compute_g_plus_from_match(m) == 0.0


class TestPercentileValue:
    def test_perfect_percentile(self):
        assert _percentile_value(10.0, [1, 2, 3, 4, 5, 6, 7, 8, 9]) == 100.0

    def test_bottom_percentile(self):
        assert _percentile_value(1.0, [2, 3, 4, 5]) == 0.0

    def test_mid_percentile(self):
        p = _percentile_value(3.0, [1, 2, 3, 4, 5])
        assert 40.0 <= p <= 60.0

    def test_empty_distribution(self):
        assert _percentile_value(5.0, []) == 50.0


class TestComputeGoalsAdded:
    def test_empty_match_stats(self):
        r = compute_goals_added("p1", [], "mid")
        assert r.total_g_plus == 0.0
        assert r.g_plus_per_90 == 0.0
        assert r.per_game == []
        assert r.components["xg_contribution"] == 0.0

    def test_single_match(self):
        stats = [
            {
                "match_id": "m1",
                "xg": 1.0,
                "xa": 0.5,
                "xt": 0.3,
                "defensive_actions": 4,
                "obv": 0.2,
                "minutes": 90,
            }
        ]
        r = compute_goals_added("p1", stats, "fwd")
        assert r.total_g_plus > 0
        assert r.g_plus_per_90 > 0
        assert len(r.per_game) == 1
        assert r.per_game[0]["match_id"] == "m1"

    def test_multiple_matches(self):
        stats = [
            {
                "match_id": "m1",
                "xg": 0.5,
                "xa": 0.3,
                "xt": 0.1,
                "defensive_actions": 2,
                "obv": 0.1,
                "minutes": 90,
            },
            {
                "match_id": "m2",
                "xg": 0.0,
                "xa": 0.0,
                "xt": 0.0,
                "defensive_actions": 0,
                "obv": 0.0,
                "minutes": 90,
            },
            {
                "match_id": "m3",
                "xg": 1.2,
                "xa": 0.4,
                "xt": 0.2,
                "defensive_actions": 3,
                "obv": 0.3,
                "minutes": 90,
            },
        ]
        r = compute_goals_added("p1", stats, "mid")
        assert len(r.per_game) == 3
        assert r.total_g_plus > 0
        assert r.g_plus_per_90 > 0

    def test_per_game_entries(self):
        stats = [
            {
                "match_id": "m1",
                "xg": 0.0,
                "xa": 0.0,
                "xt": 0.0,
                "defensive_actions": 0,
                "obv": 0.0,
                "minutes": 90,
            },
            {
                "match_id": "m2",
                "xg": 0.0,
                "xa": 0.0,
                "xt": 0.0,
                "defensive_actions": 0,
                "obv": 0.0,
                "minutes": 45,
            },
        ]
        r = compute_goals_added("p1", stats, "def")
        assert len(r.per_game) == 2
        assert r.per_game[0]["minutes"] == 90
        assert r.per_game[1]["minutes"] == 45

    def test_all_components_present(self):
        stats = [
            {
                "match_id": "m1",
                "xg": 0.4,
                "xa": 0.2,
                "xt": 0.1,
                "defensive_actions": 5,
                "obv": 0.05,
                "minutes": 90,
            }
        ]
        r = compute_goals_added("p1", stats, "fwd")
        assert "xg_contribution" in r.components
        assert "xa_contribution" in r.components
        assert "xt_contribution" in r.components
        assert "defensive_contribution" in r.components
        assert "obv_contribution" in r.components

    def test_percentile_vs_position_with_league_stats(self):
        stats = [
            {
                "match_id": "m1",
                "xg": 0.5,
                "xa": 0.3,
                "xt": 0.2,
                "defensive_actions": 2,
                "obv": 0.1,
                "minutes": 90,
            }
        ]
        league = {
            "position_averages": {
                "mid": {"g_plus_per_90": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
            }
        }
        r = compute_goals_added("p1", stats, "mid", league_stats=league)
        assert 0.0 <= r.percentile_vs_position <= 100.0

    def test_percentile_default_when_no_league_stats(self):
        stats = [
            {
                "match_id": "m1",
                "xg": 0.5,
                "xa": 0.3,
                "xt": 0.2,
                "defensive_actions": 2,
                "obv": 0.1,
                "minutes": 90,
            }
        ]
        r = compute_goals_added("p1", stats, "mid")
        assert r.percentile_vs_position == 50.0

    def test_zero_minutes(self):
        stats = [
            {
                "match_id": "m1",
                "xg": 0.0,
                "xa": 0.0,
                "xt": 0.0,
                "defensive_actions": 0,
                "obv": 0.0,
                "minutes": 0,
            }
        ]
        r = compute_goals_added("p1", stats, "fwd")
        assert r.g_plus_per_90 == 0.0

    def test_partial_minutes(self):
        stats = [
            {
                "match_id": "m1",
                "xg": 0.5,
                "xa": 0.2,
                "xt": 0.1,
                "defensive_actions": 3,
                "obv": 0.05,
                "minutes": 45,
            }
        ]
        r = compute_goals_added("p1", stats, "mid")
        assert r.total_g_plus > 0
        assert r.g_plus_per_90 > r.total_g_plus

    def test_defensive_actions_contribute(self):
        stats_no_def = [
            {
                "match_id": "m1",
                "xg": 0.0,
                "xa": 0.0,
                "xt": 0.0,
                "defensive_actions": 0,
                "obv": 0.0,
                "minutes": 90,
            }
        ]
        stats_def = [
            {
                "match_id": "m1",
                "xg": 0.0,
                "xa": 0.0,
                "xt": 0.0,
                "defensive_actions": 10,
                "obv": 0.0,
                "minutes": 90,
            }
        ]
        r_no = compute_goals_added("p1", stats_no_def, "def")
        r_def = compute_goals_added("p1", stats_def, "def")
        assert r_def.total_g_plus > r_no.total_g_plus

    def test_scalar_league_stat(self):
        stats = [
            {
                "match_id": "m1",
                "xg": 0.5,
                "xa": 0.3,
                "xt": 0.2,
                "defensive_actions": 2,
                "obv": 0.1,
                "minutes": 90,
            }
        ]
        league = {"position_averages": {"fwd": {"g_plus_per_90": 0.15}}}
        r = compute_goals_added("p1", stats, "fwd", league_stats=league)
        assert 0.0 <= r.percentile_vs_position <= 100.0
