"""Tests for League Simulation — Monte Carlo simulation from xG."""

from kawkab.core.league_simulation import (
    _poisson_goals,
    _poisson_knuth,
    simulate_league,
    simulate_match,
)


class TestPoisson:
    def test_knuth_zero_lambda(self):
        assert _poisson_knuth(0.0) == 0

    def test_knuth_negative_lambda(self):
        assert _poisson_knuth(-1.0) == 0

    def test_knuth_non_deterministic(self):
        results = set()
        for _ in range(100):
            results.add(_poisson_knuth(2.5))
        assert len(results) > 1

    def test_poisson_goals_non_negative(self):
        for _ in range(100):
            g = _poisson_goals(1.5)
            assert g >= 0
            assert g <= 30

    def test_poisson_goals_zero_lambda(self):
        assert _poisson_goals(0.0) == 0

    def test_poisson_goals_small_lambda(self):
        total = 0
        for _ in range(500):
            total += _poisson_goals(0.1)
        avg = total / 500
        assert 0.0 <= avg <= 0.5


class TestSimulateMatch:
    def test_non_negative_goals(self):
        for _ in range(100):
            h, a = simulate_match(1.5, 1.2)
            assert h >= 0
            assert a >= 0

    def test_high_xg_more_goals_tendency(self):
        home_goals = 0
        away_goals = 0
        for _ in range(500):
            h, a = simulate_match(3.0, 0.5)
            home_goals += h
            away_goals += a
        assert home_goals > away_goals

    def test_equal_xg_symmetric(self):
        home_wins = 0
        away_wins = 0
        for _ in range(1000):
            h, a = simulate_match(1.0, 1.0)
            if h > a:
                home_wins += 1
            elif a > h:
                away_wins += 1
        assert 0.2 <= home_wins / max(away_wins, 1) <= 5.0

    def test_zero_xg(self):
        h, a = simulate_match(0.0, 0.0)
        assert h == 0
        assert a == 0


class TestSimulateLeague:
    def test_basic_simulation(self):
        fixtures = [
            {"home_team": "team_a", "away_team": "team_b", "home_xg": 1.5, "away_xg": 1.0},
            {"home_team": "team_c", "away_team": "team_d", "home_xg": 0.8, "away_xg": 1.2},
        ]
        table = [
            {"team_id": "team_a", "points": 10, "played": 5},
            {"team_id": "team_b", "points": 8, "played": 5},
            {"team_id": "team_c", "points": 6, "played": 5},
            {"team_id": "team_d", "points": 4, "played": 5},
        ]
        r = simulate_league(fixtures, table, n_simulations=100)
        assert r.n_simulations == 100
        assert len(r.standings) == 4
        assert r.most_likely_table is not None

    def test_standings_have_all_fields(self):
        fixtures = [{"home_team": "team_a", "away_team": "team_b", "home_xg": 1.0, "away_xg": 1.0}]
        table = [
            {"team_id": "team_a", "points": 0, "played": 0},
            {"team_id": "team_b", "points": 0, "played": 0},
        ]
        r = simulate_league(fixtures, table, n_simulations=50)
        s = r.standings[0]
        assert "team_id" in s
        assert "avg_points" in s
        assert "title_pct" in s
        assert "top4_pct" in s
        assert "relegation_pct" in s

    def test_title_and_relegation_percentages(self):
        fixtures = [{"home_team": "team_a", "away_team": "team_b", "home_xg": 3.0, "away_xg": 0.3}]
        table = [
            {"team_id": "team_a", "points": 0, "played": 0},
            {"team_id": "team_b", "points": 0, "played": 0},
        ]
        r = simulate_league(fixtures, table, n_simulations=100)
        a_row = next(s for s in r.standings if s["team_id"] == "team_a")
        b_row = next(s for s in r.standings if s["team_id"] == "team_b")
        assert a_row["title_pct"] >= b_row["title_pct"]

    def test_point_distributions(self):
        fixtures = [{"home_team": "team_a", "away_team": "team_b", "home_xg": 1.0, "away_xg": 1.0}]
        table = [
            {"team_id": "team_a", "points": 0, "played": 0},
            {"team_id": "team_b", "points": 0, "played": 0},
        ]
        r = simulate_league(fixtures, table, n_simulations=50)
        assert len(r.point_distributions) == 2
        assert len(r.point_distributions["team_a"]) == 50

    def test_teams_from_fixtures_added_automatically(self):
        fixtures = [
            {"home_team": "new_team", "away_team": "other_team", "home_xg": 1.0, "away_xg": 1.0}
        ]
        table = []
        r = simulate_league(fixtures, table, n_simulations=50)
        team_ids = {s["team_id"] for s in r.standings}
        assert "new_team" in team_ids
        assert "other_team" in team_ids

    def test_most_likely_table_sorted(self):
        fixtures = [
            {"home_team": "team_a", "away_team": "team_b", "home_xg": 1.5, "away_xg": 0.5},
            {"home_team": "team_c", "away_team": "team_d", "home_xg": 1.0, "away_xg": 1.2},
        ]
        table = [
            {"team_id": "team_a", "points": 20, "played": 10},
            {"team_id": "team_b", "points": 15, "played": 10},
            {"team_id": "team_c", "points": 10, "played": 10},
            {"team_id": "team_d", "points": 5, "played": 10},
        ]
        r = simulate_league(fixtures, table, n_simulations=100)
        positions = [t["position"] for t in r.most_likely_table]
        assert positions == sorted(positions)

    def test_relegation_spots(self):
        fixtures = [{"home_team": "team_a", "away_team": "team_b", "home_xg": 1.0, "away_xg": 1.0}]
        table = [
            {"team_id": "team_a", "points": 0, "played": 0},
            {"team_id": "team_b", "points": 0, "played": 0},
        ]
        r = simulate_league(fixtures, table, n_simulations=50, relegation_spots=1)
        rel_pcts = [s["relegation_pct"] for s in r.standings]
        assert all(0.0 <= p <= 100.0 for p in rel_pcts)

    def test_top4_spots(self):
        fixtures = [{"home_team": "team_a", "away_team": "team_b", "home_xg": 1.0, "away_xg": 1.0}]
        table = [
            {"team_id": "team_a", "points": 0, "played": 0},
            {"team_id": "team_b", "points": 0, "played": 0},
        ]
        r = simulate_league(fixtures, table, n_simulations=50, top4_spots=2)
        top4_pcts = [s["top4_pct"] for s in r.standings]
        assert all(0.0 <= p <= 100.0 for p in top4_pcts)

    def test_single_team(self):
        fixtures = [{"home_team": "team_a", "away_team": "team_a", "home_xg": 0.5, "away_xg": 0.5}]
        table = [{"team_id": "team_a", "points": 5, "played": 3}]
        r = simulate_league(fixtures, table, n_simulations=10)
        assert len(r.standings) == 1

    def test_no_fixtures(self):
        r = simulate_league(
            [], [{"team_id": "team_a", "points": 10, "played": 10}], n_simulations=10
        )
        assert len(r.standings) == 1
        assert r.standings[0]["avg_points"] == 10.0
