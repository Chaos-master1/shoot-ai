"""Tests for season aggregation."""

from kawkab.core.season_aggregator import (
    HeadToHeadComparison,
    PlayerSeasonStats,
    SeasonAggregator,
    SeasonReport,
)


class TestPlayerSeasonStats:
    def test_per90(self):
        p = PlayerSeasonStats(total_minutes=180)
        assert p.per90(6) == 3.0

    def test_pass_accuracy(self):
        p = PlayerSeasonStats(passes_attempted=50, passes_completed=40)
        assert p.pass_accuracy == 0.8

    def test_pass_accuracy_zero(self):
        p = PlayerSeasonStats()
        assert p.pass_accuracy == 0.0

    def test_to_dict(self):
        p = PlayerSeasonStats(name="Test", matches_played=5, goals=2.0, total_minutes=450)
        d = p.to_dict()
        assert d["name"] == "Test"
        assert d["matches"] == 5
        assert d["goals"] == 2.0
        assert d["goals_per90"] == 0.4


class TestSeasonAggregator:
    def test_empty_data(self):
        sa = SeasonAggregator()
        report = sa.aggregate_team_season([], team_name="Team A")
        assert isinstance(report, SeasonReport)
        assert report.matches == 0

    def test_single_match_aggregation(self):
        sa = SeasonAggregator()
        data = [
            {
                "home_team": {
                    "team_name": "Team A",
                    "possession": 55.0,
                    "pass_accuracy": 0.82,
                    "shots": 12,
                },
                "away_team": {
                    "team_name": "Team B",
                    "possession": 45.0,
                    "pass_accuracy": 0.78,
                    "shots": 8,
                },
                "events": [],
                "duration": 5400,
                "players": {},
            }
        ]
        report = sa.aggregate_team_season(data, team_name="Team A")
        assert report.matches == 1
        assert report.avg_possession == 55.0
        assert report.total_shots == 12

    def test_multiple_matches(self):
        sa = SeasonAggregator()
        data = [
            {
                "home_team": {
                    "team_name": "Team A",
                    "possession": 55.0,
                    "pass_accuracy": 0.82,
                    "shots": 12,
                },
                "away_team": {"team_name": "Team B", "possession": 45.0},
                "events": [],
                "duration": 5400,
                "players": {},
            },
            {
                "home_team": {
                    "team_name": "Team A",
                    "possession": 60.0,
                    "pass_accuracy": 0.85,
                    "shots": 15,
                },
                "away_team": {"team_name": "Team C", "possession": 40.0},
                "events": [],
                "duration": 5400,
                "players": {},
            },
        ]
        report = sa.aggregate_team_season(data, team_name="Team A")
        assert report.matches == 2
        assert report.avg_possession == 57.5
        assert report.total_shots == 27

    def test_player_aggregation(self):
        sa = SeasonAggregator()
        data = [
            {
                "home_team": {"team_name": "Team A", "possession": 50.0},
                "away_team": {"team_name": "Team B", "possession": 50.0},
                "events": [],
                "duration": 5400,
                "players": {
                    "1": {
                        "name": "Player X",
                        "shots": 3,
                        "passes_attempted": 30,
                        "passes_completed": 25,
                        "tackles": 2,
                        "distance_covered_m": 9000,
                    },
                },
            }
        ]
        report = sa.aggregate_team_season(data, team_name="Team A")
        assert 1 in report.players
        p = report.players[1]
        assert p.shots == 3
        assert p.passes_attempted == 30
        assert p.passes_completed == 25


class TestHeadToHead:
    def test_empty_comparison(self):
        sa = SeasonAggregator()
        comp = sa.compare_teams([], "Team A", "Team B")
        assert isinstance(comp, HeadToHeadComparison)
        assert comp.team_a_name == "Team A"
        assert comp.team_b_name == "Team B"

    def test_comparison_with_data(self):
        sa = SeasonAggregator()
        data = [
            {
                "home_team": {
                    "team_name": "Team A",
                    "possession": 55.0,
                    "pass_accuracy": 0.82,
                    "shots": 12,
                },
                "away_team": {
                    "team_name": "Team B",
                    "possession": 45.0,
                    "pass_accuracy": 0.78,
                    "shots": 8,
                },
                "events": [],
                "duration": 5400,
                "players": {},
            },
        ]
        comp = sa.compare_teams(data, "Team A", "Team B")
        assert comp.team_a_name == "Team A"
        assert comp.team_b_name == "Team B"

    def test_comparison_to_dict(self):
        comp = HeadToHeadComparison(
            team_a_name="A", team_b_name="B", possession_a=55.0, possession_b=45.0
        )
        d = comp.to_dict()
        assert d["possession"]["a"] == 55.0
        assert d["possession"]["b"] == 45.0
