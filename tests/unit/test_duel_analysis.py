"""Tests for duel analysis module."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import install_kawkab_stubs

install_kawkab_stubs()


from kawkab.core.duel_analysis import analyze_duels, classify_duel_type


class TestClassifyDuelType:
    """Tests for duel type classification."""

    def test_aerial_by_height(self):
        assert classify_duel_type({"ball_height": 2.0}) == "aerial"

    def test_ground_by_low_height(self):
        assert classify_duel_type({"ball_height": 0.3}) == "ground"

    def test_ground_by_zero_height(self):
        assert classify_duel_type({"ball_height": 0}) == "ground"

    def test_explicit_aerial_type(self):
        assert classify_duel_type({"duel_type": "aerial", "ball_height": 0.5}) == "aerial"

    def test_explicit_ground_type(self):
        assert classify_duel_type({"duel_type": "ground"}) == "ground"

    def test_explicit_50_50(self):
        assert classify_duel_type({"duel_type": "50_50"}) == "50_50"

    def test_unknown_type_defaults_ground(self):
        assert classify_duel_type({"duel_type": "shoulder_charge"}) == "ground"

    def test_empty_metadata(self):
        assert classify_duel_type({}) == "ground"

    def test_boundary_height_aerial(self):
        assert classify_duel_type({"ball_height": 1.51}) == "aerial"

    def test_boundary_height_ground(self):
        assert classify_duel_type({"ball_height": 1.5}) == "ground"


class TestAnalyzeDuels:
    """Tests for full duel analysis."""

    def test_empty_events(self):
        result = analyze_duels([])
        assert result["home"]["total_duels"] == 0
        assert result["away"]["total_duels"] == 0

    def test_single_duel_home(self):
        events = [{"type": "duel", "team": "home", "won": True, "metadata": {"ball_height": 0.5}}]
        result = analyze_duels(events)
        assert result["home"]["total_duels"] == 1
        assert result["home"]["ground_duels"] == 1
        assert result["home"]["duels_won"] == 1

    def test_single_duel_away(self):
        events = [{"type": "duel", "team": "away", "won": False, "metadata": {"ball_height": 2.0}}]
        result = analyze_duels(events)
        assert result["away"]["total_duels"] == 1
        assert result["away"]["aerial_duels"] == 1
        assert result["away"]["duels_won"] == 0

    def test_non_duel_events_ignored(self):
        events = [{"type": "pass", "team": "home"}, {"type": "shot", "team": "away"}]
        result = analyze_duels(events)
        assert result["home"]["total_duels"] == 0
        assert result["away"]["total_duels"] == 0

    def test_mixed_duel_types(self):
        events = [
            {"type": "duel", "team": "home", "won": True, "metadata": {"ball_height": 2.0}},
            {"type": "duel", "team": "home", "won": False, "metadata": {"ball_height": 0.3}},
            {"type": "duel", "team": "home", "won": True, "metadata": {"duel_type": "50_50"}},
        ]
        result = analyze_duels(events)
        assert result["home"]["total_duels"] == 3
        assert result["home"]["aerial_duels"] == 1
        assert result["home"]["ground_duels"] == 1  # only ball_height=0.3
        assert result["home"]["50_50"] == 1

    def test_fifty_fifty_count(self):
        events = [
            {"type": "duel", "team": "home", "won": True, "metadata": {"duel_type": "50_50"}},
        ]
        result = analyze_duels(events)
        assert result["home"]["50_50"] == 1

    def test_win_rate(self):
        events = [
            {"type": "duel", "team": "home", "won": True, "metadata": {}},
            {"type": "duel", "team": "home", "won": True, "metadata": {}},
            {"type": "duel", "team": "home", "won": False, "metadata": {}},
        ]
        result = analyze_duels(events)
        assert result["home"]["win_rate"] == round(2 / 3, 2)

    def test_player_breakdown(self):
        events = [
            {"type": "duel", "team": "home", "won": True, "track_id": 10, "metadata": {}},
            {"type": "duel", "team": "home", "won": False, "track_id": 10, "metadata": {}},
            {"type": "duel", "team": "home", "won": True, "track_id": 11, "metadata": {}},
        ]
        result = analyze_duels(events)
        assert len(result["home"]["players"]) == 2
        p10 = [p for p in result["home"]["players"] if p["player_id"] == "10"][0]
        assert p10["total_duels"] == 2
        assert p10["duels_won"] == 1
        assert p10["win_rate"] == 0.5

    def test_both_teams(self):
        events = [
            {"type": "duel", "team": "home", "won": True, "metadata": {}},
            {"type": "duel", "team": "away", "won": True, "metadata": {}},
        ]
        result = analyze_duels(events)
        assert result["home"]["total_duels"] == 1
        assert result["away"]["total_duels"] == 1

    def test_unknown_team_ignored(self):
        events = [{"type": "duel", "team": "unknown", "won": False, "metadata": {}}]
        result = analyze_duels(events)
        assert result["home"]["total_duels"] == 0
        assert result["away"]["total_duels"] == 0
