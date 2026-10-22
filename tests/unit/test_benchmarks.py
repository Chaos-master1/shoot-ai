"""Tests for player stat benchmarking."""

from kawkab.core.benchmarks import (
    compute_player_benchmarks,
    get_position_groups,
)


class TestBenchmarks:
    def test_empty_ratings(self):
        result = compute_player_benchmarks({})
        assert result == []

    def test_single_player(self):
        ratings = {
            1: {"name": "Player 1", "pass_accuracy": 0.85, "shots": 3, "position": "FW"},
        }
        result = compute_player_benchmarks(ratings)
        assert len(result) == 1
        assert result[0].track_id == 1

    def test_multiple_players_percentile(self):
        ratings = {
            1: {"name": "A", "pass_accuracy": 0.95, "shots": 5, "position": "MF"},
            2: {"name": "B", "pass_accuracy": 0.80, "shots": 2, "position": "DF"},
            3: {"name": "C", "pass_accuracy": 0.65, "shots": 1, "position": "GK"},
        }
        result = compute_player_benchmarks(ratings)
        # Player A with 0.95 accuracy should be ~100th percentile
        a_bench = [r for r in result if r.track_id == 1][0]
        acc_result = [r for r in a_bench.results if r.stat_name == "Pass Accuracy"][0]
        assert acc_result.percentile == 100.0

    def test_z_score_computed(self):
        ratings = {
            1: {"name": "A", "pass_accuracy": 0.90, "shots": 5, "position": "FW"},
            2: {"name": "B", "pass_accuracy": 0.75, "shots": 3, "position": "FW"},
        }
        result = compute_player_benchmarks(ratings)
        assert len(result[0].results) > 0
        assert result[0].results[0].z_score != 0.0

    def test_non_dict_rating_skipped(self):
        ratings = {1: None}
        result = compute_player_benchmarks(ratings)
        assert result == []

    def test_get_position_groups_returns_expected(self):
        groups = get_position_groups()
        assert set(groups) == {"CB", "FB", "CM", "Winger", "ST"}

    def test_position_group_filters_correctly(self):
        ratings = {
            1: {"name": "A", "pass_accuracy": 0.95, "shots": 5, "position": "CB"},
            2: {"name": "B", "pass_accuracy": 0.80, "shots": 2, "position": "ST"},
            3: {"name": "C", "pass_accuracy": 0.90, "shots": 4, "position": "RCB"},
            4: {"name": "D", "pass_accuracy": 0.70, "shots": 1, "position": "CF"},
        }
        # Filter by CB — only players with CB/LCB/RCB
        result = compute_player_benchmarks(ratings, position_group="CB")
        assert len(result) == 2
        tids = {r.track_id for r in result}
        assert tids == {1, 3}

    def test_position_group_st_percentile_within_group(self):
        ratings = {
            1: {"name": "ST1", "pass_accuracy": 0.90, "shots": 5, "position": "ST"},
            2: {"name": "ST2", "pass_accuracy": 0.70, "shots": 3, "position": "CF"},
            3: {"name": "MF", "pass_accuracy": 0.95, "shots": 2, "position": "CM"},
        }
        # ST group should only include ST1 and ST2, not MF
        result = compute_player_benchmarks(ratings, position_group="ST")
        assert len(result) == 2
        # ST1 should be 100th percentile within ST group
        st1 = [r for r in result if r.track_id == 1][0]
        acc = [r for r in st1.results if r.stat_name == "Pass Accuracy"][0]
        assert acc.percentile == 100.0

    def test_unknown_position_group_returns_all(self):
        ratings = {
            1: {"name": "A", "pass_accuracy": 0.85, "shots": 3, "position": "CB"},
            2: {"name": "B", "pass_accuracy": 0.75, "shots": 2, "position": "ST"},
        }
        result = compute_player_benchmarks(ratings, position_group="UNKNOWN")
        assert len(result) == 2

    def test_no_players_in_group_returns_empty(self):
        ratings = {
            1: {"name": "A", "pass_accuracy": 0.85, "shots": 3, "position": "GK"},
        }
        result = compute_player_benchmarks(ratings, position_group="ST")
        assert result == []

    def test_position_group_empty_if_no_matching_players(self):
        ratings = {
            1: {"name": "A", "pass_accuracy": 0.85, "shots": 3, "position": "CB"},
            2: {"name": "B", "pass_accuracy": 0.75, "shots": 2, "position": "CB"},
        }
        result = compute_player_benchmarks(ratings, position_group="Winger")
        assert result == []
