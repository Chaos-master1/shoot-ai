"""Tests for Transfer Fee Extension - estimate_player_transfer_fee."""

from kawkab.core.squad_valuation import (
    _transfer_age_factor,
    _transfer_contract_factor,
    _transfer_performance_score,
    estimate_player_transfer_fee,
)


class TestTransferAgeFactor:
    def test_youngest_has_highest_factor(self):
        assert _transfer_age_factor(15) == 1.3
        assert _transfer_age_factor(16) == 1.3

    def test_teen_to_early_twenties(self):
        assert _transfer_age_factor(18) == 1.2
        assert _transfer_age_factor(20) == 1.2

    def test_prime_years(self):
        assert _transfer_age_factor(22) == 1.1
        assert _transfer_age_factor(25) == 1.0

    def test_starts_decline_mid_20s(self):
        assert _transfer_age_factor(26) == 0.9
        assert _transfer_age_factor(28) == 0.9
        assert _transfer_age_factor(30) == 0.7

    def test_older_players_declining(self):
        assert _transfer_age_factor(32) == 0.5
        assert _transfer_age_factor(34) == 0.35
        assert _transfer_age_factor(40) == 0.2


class TestTransferPerformanceScore:
    def test_empty_stats(self):
        assert _transfer_performance_score({}) == 0.0

    def test_good_stats_produces_score(self):
        stats = {"minutes_played": 1800, "xg_per_90": 0.5, "rating_per_90": 7.5}
        score = _transfer_performance_score(stats)
        assert 15.0 <= score <= 100.0

    def test_zero_stats(self):
        stats = {"minutes_played": 100, "xg_per_90": 0, "rating_per_90": 0}
        score = _transfer_performance_score(stats)
        assert score >= 0.0


class TestTransferContractFactor:
    def test_four_plus_years(self):
        assert _transfer_contract_factor(4) == 1.2
        assert _transfer_contract_factor(5) == 1.2

    def test_two_to_three_years(self):
        assert _transfer_contract_factor(2) == 1.0
        assert _transfer_contract_factor(3) == 1.0

    def test_one_year(self):
        assert _transfer_contract_factor(1) == 0.7

    def test_less_than_one(self):
        assert _transfer_contract_factor(0) == 0.5


class TestEstimatePlayerTransferFee:
    def test_young_high_performer_higher_than_old_low(self):
        young = estimate_player_transfer_fee(
            age=22,
            position="FWD",
            performance_stats={
                "minutes_played": 2000,
                "xg_per_90": 0.6,
                "xa_per_90": 0.4,
                "goals_per_90": 0.5,
                "assists_per_90": 0.3,
                "rating_per_90": 7.8,
            },
            contract_years_remaining=4,
        )
        old = estimate_player_transfer_fee(
            age=34,
            position="FWD",
            performance_stats={
                "minutes_played": 500,
                "xg_per_90": 0.05,
                "xa_per_90": 0.02,
                "goals_per_90": 0.03,
                "assists_per_90": 0.01,
                "rating_per_90": 6.0,
            },
            contract_years_remaining=1,
        )
        assert young["estimated_fee_millions"] > old["estimated_fee_millions"]

    def test_contract_length_increases_fee(self):
        long = estimate_player_transfer_fee(
            age=25,
            position="FWD",
            performance_stats={"minutes_played": 1500},
            contract_years_remaining=4,
        )
        short = estimate_player_transfer_fee(
            age=25,
            position="FWD",
            performance_stats={"minutes_played": 1500},
            contract_years_remaining=0,
        )
        assert long["estimated_fee_millions"] >= short["estimated_fee_millions"]
        assert long["contract_factor"] > short["contract_factor"]

    def test_international_premium_applied(self):
        yes = estimate_player_transfer_fee(
            age=25,
            position="FWD",
            performance_stats={"minutes_played": 1500},
            is_international=True,
        )
        no = estimate_player_transfer_fee(
            age=25,
            position="FWD",
            performance_stats={"minutes_played": 1500},
            is_international=False,
        )
        assert yes["international_premium"] == 1.2
        assert no["international_premium"] == 1.0
        assert yes["estimated_fee_millions"] > no["estimated_fee_millions"]

    def test_injury_discount_applied(self):
        healthy = estimate_player_transfer_fee(
            age=25, position="DEF", performance_stats={"minutes_played": 1500}, injury_history="low"
        )
        injured = estimate_player_transfer_fee(
            age=25,
            position="DEF",
            performance_stats={"minutes_played": 1500},
            injury_history="high",
        )
        assert healthy["injury_discount"] == 1.0
        assert injured["injury_discount"] == 0.5
        assert healthy["estimated_fee_millions"] > injured["estimated_fee_millions"]

    def test_market_trend_rising_increases_fee(self):
        rising = estimate_player_transfer_fee(
            age=25,
            position="FWD",
            performance_stats={"minutes_played": 1500},
            market_trend="rising",
        )
        decl = estimate_player_transfer_fee(
            age=25,
            position="FWD",
            performance_stats={"minutes_played": 1500},
            market_trend="declining",
        )
        assert rising["market_trend_adjustment"] == 1.15
        assert decl["market_trend_adjustment"] == 0.85
        assert rising["estimated_fee_millions"] > decl["estimated_fee_millions"]

    def test_fee_range_low_mid_high(self):
        r = estimate_player_transfer_fee(
            age=22,
            position="MID",
            performance_stats={"minutes_played": 1500, "xg_per_90": 0.4, "rating_per_90": 7.5},
        )
        assert r["fee_range"]["low"] < r["fee_range"]["mid"] < r["fee_range"]["high"]
        assert r["fee_range"]["mid"] <= r["estimated_fee_millions"]

    def test_age_15_edge(self):
        r = estimate_player_transfer_fee(
            age=15, position="FWD", performance_stats={"minutes_played": 0}
        )
        assert r["estimated_fee_millions"] > 0
        assert r["age_factor"] == 1.3

    def test_age_40_edge(self):
        r = estimate_player_transfer_fee(age=40, position="FWD", performance_stats={})
        assert r["age_factor"] == 0.2

    def test_all_edges_zero_stats(self):
        r = estimate_player_transfer_fee(
            age=25, position="FWD", performance_stats={"minutes_played": 0}
        )
        assert r["estimated_fee_millions"] > 0
        assert r["confidence"] == "low"

    def test_all_factors_in_output(self):
        r = estimate_player_transfer_fee(
            age=25,
            position="MID",
            performance_stats={"minutes_played": 1500},
            league_tier="premier_league",
        )
        for key in (
            "estimated_fee_millions",
            "age_factor",
            "performance_score",
            "position_baseline",
            "contract_factor",
            "league_factor",
            "market_trend_adjustment",
            "international_premium",
            "injury_discount",
            "confidence",
            "fee_range",
        ):
            assert key in r

    def test_league_factor_effect(self):
        pl = estimate_player_transfer_fee(
            age=25,
            position="FWD",
            performance_stats={"minutes_played": 1500},
            league_tier="premier_league",
        )
        mls = estimate_player_transfer_fee(
            age=25, position="FWD", performance_stats={"minutes_played": 1500}, league_tier="mls"
        )
        assert pl["league_factor"] > mls["league_factor"]
        assert pl["estimated_fee_millions"] > mls["estimated_fee_millions"]
