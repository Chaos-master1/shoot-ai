"""Tests for Squad Value Estimation — heuristic market valuation."""

from kawkab.core.squad_valuation import (
    LEAGUE_MULTIPLIERS,
    _age_multiplier,
    _confidence_label,
    _contract_multiplier,
    _league_multiplier,
    _performance_score,
    estimate_player_value,
    estimate_squad_value,
)


class TestAgeMultiplier:
    def test_u21(self):
        assert _age_multiplier(18) == 1.2

    def test_age_21(self):
        assert _age_multiplier(21) == 1.2

    def test_peak_age(self):
        assert _age_multiplier(25) == 1.1

    def test_age_29(self):
        assert _age_multiplier(29) == 0.7

    def test_age_32(self):
        assert _age_multiplier(32) == 0.7

    def test_veteran_33(self):
        assert _age_multiplier(33) == 0.4

    def test_veteran_older(self):
        assert _age_multiplier(40) == 0.4


class TestPerformanceScore:
    def test_zero_stats(self):
        assert _performance_score({}) == 0.0

    def test_below_90_minutes(self):
        stats = {"minutes_played": 45, "xg_per_90": 1.0, "xa_per_90": 0.5}
        score = _performance_score(stats)
        assert score == 5.0

    def test_good_stats(self):
        stats = {
            "minutes_played": 1800,
            "xg_per_90": 0.5,
            "xa_per_90": 0.3,
            "goals_per_90": 0.4,
            "assists_per_90": 0.2,
            "rating_per_90": 7.5,
        }
        score = _performance_score(stats)
        assert 10.0 <= score <= 100.0

    def test_elite_stats(self):
        stats = {
            "minutes_played": 2000,
            "xg_per_90": 1.0,
            "xa_per_90": 0.8,
            "goals_per_90": 0.9,
            "assists_per_90": 0.5,
            "rating_per_90": 9.0,
        }
        score = _performance_score(stats)
        assert score <= 100.0

    def test_with_rating(self):
        stats = {
            "minutes_played": 1500,
            "xg_per_90": 0.0,
            "xa_per_90": 0.0,
            "goals_per_90": 0.0,
            "assists_per_90": 0.0,
            "rating_per_90": 10.0,
        }
        score = _performance_score(stats)
        assert score > 5.0


class TestContractMultiplier:
    def test_four_plus_years(self):
        assert _contract_multiplier(5) == 1.2
        assert _contract_multiplier(4) == 1.2

    def test_two_to_three_years(self):
        assert _contract_multiplier(2) == 1.0
        assert _contract_multiplier(3) == 1.0

    def test_one_year(self):
        assert _contract_multiplier(1) == 0.8

    def test_less_than_one(self):
        assert _contract_multiplier(0) == 0.6


class TestLeagueMultiplier:
    def test_premier_league(self):
        assert _league_multiplier("premier_league") == 1.0

    def test_la_liga(self):
        assert _league_multiplier("la_liga") == 0.9

    def test_bundesliga(self):
        assert _league_multiplier("bundesliga") == 0.85

    def test_unknown_league(self):
        assert _league_multiplier("unknown") == 0.3

    def test_league_multipliers_dict(self):
        assert "premier_league" in LEAGUE_MULTIPLIERS
        assert "championship" in LEAGUE_MULTIPLIERS


class TestConfidenceLabel:
    def test_high_confidence(self):
        assert _confidence_label(60.0, 1500, 2) == "high"

    def test_medium_confidence_low_minutes(self):
        assert _confidence_label(60.0, 500, 2) == "medium"

    def test_low_confidence_very_low_minutes(self):
        assert _confidence_label(20.0, 100, 1) == "low"

    def test_high_needs_1500_minutes(self):
        assert _confidence_label(50.0, 1499, 2) == "medium"


class TestEstimatePlayerValue:
    def test_basic_forward(self):
        val = estimate_player_value(
            "p1",
            age=25,
            position="fwd",
            performance_stats={
                "minutes_played": 1800,
                "xg_per_90": 0.4,
                "xa_per_90": 0.2,
                "goals_per_90": 0.35,
                "assists_per_90": 0.15,
                "rating_per_90": 7.0,
            },
            contract_years_remaining=3,
            league_tier="premier_league",
        )
        assert val.player_id == "p1"
        assert val.estimated_value > 0

    def test_young_defender_low_value(self):
        val = estimate_player_value(
            "p2",
            age=19,
            position="def",
            performance_stats={
                "minutes_played": 200,
                "xg_per_90": 0.01,
                "xa_per_90": 0.01,
                "goals_per_90": 0.0,
                "assists_per_90": 0.0,
                "rating_per_90": 6.0,
            },
            contract_years_remaining=5,
            league_tier="championship",
        )
        assert val.estimated_value > 0
        assert val.age_multiplier == 1.2

    def test_old_player_depreciation(self):
        val = estimate_player_value(
            "p3",
            age=35,
            position="mid",
            performance_stats={
                "minutes_played": 1500,
                "xg_per_90": 0.2,
                "xa_per_90": 0.15,
                "goals_per_90": 0.1,
                "assists_per_90": 0.12,
                "rating_per_90": 6.8,
            },
            contract_years_remaining=1,
            league_tier="premier_league",
        )
        assert val.age_multiplier == 0.4
        assert val.contract_multiplier == 0.8

    def test_short_contract_depression(self):
        val1 = estimate_player_value(
            "p4",
            age=25,
            position="mid",
            performance_stats={
                "minutes_played": 1500,
                "xg_per_90": 0.3,
                "xa_per_90": 0.2,
                "goals_per_90": 0.2,
                "assists_per_90": 0.1,
                "rating_per_90": 7.0,
            },
            contract_years_remaining=4,
        )
        val2 = estimate_player_value(
            "p4",
            age=25,
            position="mid",
            performance_stats={
                "minutes_played": 1500,
                "xg_per_90": 0.3,
                "xa_per_90": 0.2,
                "goals_per_90": 0.2,
                "assists_per_90": 0.1,
                "rating_per_90": 7.0,
            },
            contract_years_remaining=0,
        )
        assert val1.estimated_value > val2.estimated_value

    def test_league_multiplier_effect(self):
        val_pl = estimate_player_value(
            "p5",
            age=25,
            position="fwd",
            performance_stats={
                "minutes_played": 1500,
                "xg_per_90": 0.5,
                "xa_per_90": 0.3,
                "goals_per_90": 0.4,
                "assists_per_90": 0.2,
                "rating_per_90": 7.5,
            },
            league_tier="premier_league",
        )
        val_other = estimate_player_value(
            "p5",
            age=25,
            position="fwd",
            performance_stats={
                "minutes_played": 1500,
                "xg_per_90": 0.5,
                "xa_per_90": 0.3,
                "goals_per_90": 0.4,
                "assists_per_90": 0.2,
                "rating_per_90": 7.5,
            },
            league_tier="other",
        )
        assert val_pl.estimated_value > val_other.estimated_value

    def test_position_baselines(self):
        gk = estimate_player_value(
            "g",
            age=25,
            position="gk",
            performance_stats={
                "minutes_played": 1500,
                "xg_per_90": 0,
                "xa_per_90": 0,
                "goals_per_90": 0,
                "assists_per_90": 0,
                "rating_per_90": 6.5,
            },
        )
        fwd = estimate_player_value(
            "f",
            age=25,
            position="fwd",
            performance_stats={
                "minutes_played": 1500,
                "xg_per_90": 0,
                "xa_per_90": 0,
                "goals_per_90": 0,
                "assists_per_90": 0,
                "rating_per_90": 6.5,
            },
        )
        assert fwd.estimated_value > gk.estimated_value

    def test_confidence_in_report(self):
        val = estimate_player_value(
            "p6",
            age=22,
            position="mid",
            performance_stats={
                "minutes_played": 1600,
                "xg_per_90": 0.2,
                "xa_per_90": 0.1,
                "goals_per_90": 0.15,
                "assists_per_90": 0.1,
                "rating_per_90": 6.8,
            },
            contract_years_remaining=3,
        )
        assert val.confidence in ("low", "medium", "high")


class TestEstimateSquadValue:
    def test_empty_squad(self):
        r = estimate_squad_value("team_a", [])
        assert r.total_squad_value == 0.0
        assert r.players == []
        assert r.most_valuable == ""

    def test_single_player_squad(self):
        players = [
            {
                "player_id": "p1",
                "age": 25,
                "position": "fwd",
                "performance_stats": {
                    "minutes_played": 1800,
                    "xg_per_90": 0.5,
                    "xa_per_90": 0.3,
                    "goals_per_90": 0.4,
                    "assists_per_90": 0.2,
                    "rating_per_90": 7.5,
                },
                "contract_years_remaining": 3,
            }
        ]
        r = estimate_squad_value("team_a", players)
        assert r.total_squad_value > 0
        assert r.most_valuable == "p1"

    def test_multiple_players(self):
        players = [
            {
                "player_id": "p1",
                "age": 25,
                "position": "fwd",
                "performance_stats": {
                    "minutes_played": 1800,
                    "xg_per_90": 0.5,
                    "xa_per_90": 0.3,
                    "goals_per_90": 0.4,
                    "assists_per_90": 0.2,
                    "rating_per_90": 7.5,
                },
            },
            {
                "player_id": "p2",
                "age": 32,
                "position": "def",
                "performance_stats": {
                    "minutes_played": 1500,
                    "xg_per_90": 0.05,
                    "xa_per_90": 0.02,
                    "goals_per_90": 0.02,
                    "assists_per_90": 0.01,
                    "rating_per_90": 6.5,
                },
                "contract_years_remaining": 1,
            },
            {
                "player_id": "p3",
                "age": 19,
                "position": "mid",
                "performance_stats": {
                    "minutes_played": 300,
                    "xg_per_90": 0.1,
                    "xa_per_90": 0.05,
                    "goals_per_90": 0.05,
                    "assists_per_90": 0.02,
                    "rating_per_90": 6.2,
                },
                "contract_years_remaining": 4,
            },
        ]
        r = estimate_squad_value("team_a", players)
        assert r.total_squad_value > 0
        assert r.avg_value > 0
        assert r.most_valuable in ("p1", "p2", "p3")

    def test_age_distribution(self):
        players = [
            {"player_id": "p1", "age": 19, "position": "mid", "performance_stats": {}},
            {"player_id": "p2", "age": 25, "position": "fwd", "performance_stats": {}},
            {"player_id": "p3", "age": 30, "position": "def", "performance_stats": {}},
            {"player_id": "p4", "age": 35, "position": "gk", "performance_stats": {}},
        ]
        r = estimate_squad_value("team_a", players)
        assert r.age_distribution["u21"] == 1
        assert r.age_distribution["prime"] == 1
        assert r.age_distribution["veteran"] == 2

    def test_value_rating(self):
        players = [
            {
                "player_id": "p1",
                "age": 25,
                "position": "fwd",
                "performance_stats": {
                    "minutes_played": 1800,
                    "xg_per_90": 0.5,
                    "xa_per_90": 0.3,
                    "goals_per_90": 0.4,
                    "assists_per_90": 0.2,
                    "rating_per_90": 7.5,
                },
            }
        ]
        r = estimate_squad_value("team_a", players)
        assert r.value_rating in ("underpriced", "fair", "overpriced")

    def test_custom_league_tier(self):
        players = [
            {
                "player_id": "p1",
                "age": 25,
                "position": "fwd",
                "performance_stats": {
                    "minutes_played": 1800,
                    "xg_per_90": 0.5,
                    "xa_per_90": 0.3,
                    "goals_per_90": 0.4,
                    "assists_per_90": 0.2,
                    "rating_per_90": 7.5,
                },
            }
        ]
        r_pl = estimate_squad_value("team_a", players, league_tier="premier_league")
        r_other = estimate_squad_value("team_a", players, league_tier="other")
        assert r_pl.total_squad_value > r_other.total_squad_value
