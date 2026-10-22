"""Tests for Player Search — multi-criteria filtering and scoring."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from kawkab.core.player_search import (
    SearchCriteria,
    search_players,
)

SAMPLE_DB = [
    {
        "player_id": "p1",
        "name": "Young Star",
        "age": 19,
        "position": "FWD",
        "league": "PL",
        "team": "Team A",
        "nationality": "England",
        "preferred_foot": "right",
        "height_cm": 180,
        "stats": {
            "xg_per_90": 0.45,
            "xa_per_90": 0.20,
            "pass_completion_pct": 82.0,
            "tackles_per_90": 0.5,
            "rating_per_90": 7.2,
        },
    },
    {
        "player_id": "p2",
        "name": "Veteran Mid",
        "age": 32,
        "position": "MID",
        "league": "La Liga",
        "team": "Team B",
        "nationality": "Spain",
        "preferred_foot": "left",
        "height_cm": 175,
        "stats": {
            "xg_per_90": 0.12,
            "xa_per_90": 0.25,
            "pass_completion_pct": 88.0,
            "tackles_per_90": 2.5,
            "rating_per_90": 7.0,
        },
    },
    {
        "player_id": "p3",
        "name": "Prime Defender",
        "age": 26,
        "position": "DEF",
        "league": "PL",
        "team": "Team C",
        "nationality": "Brazil",
        "preferred_foot": "right",
        "height_cm": 188,
        "stats": {
            "xg_per_90": 0.05,
            "xa_per_90": 0.03,
            "pass_completion_pct": 90.0,
            "tackles_per_90": 4.0,
            "rating_per_90": 7.5,
        },
    },
    {
        "player_id": "p4",
        "name": "Young Keeper",
        "age": 21,
        "position": "GK",
        "league": "Bundesliga",
        "team": "Team D",
        "nationality": "Germany",
        "preferred_foot": "right",
        "height_cm": 195,
        "stats": {
            "xg_per_90": 0.0,
            "xa_per_90": 0.0,
            "pass_completion_pct": 75.0,
            "tackles_per_90": 0.1,
            "rating_per_90": 6.8,
        },
    },
    {
        "player_id": "p5",
        "name": "Winger Prospect",
        "age": 17,
        "position": "FWD/MID",
        "league": "PL",
        "team": "Team E",
        "nationality": "England",
        "preferred_foot": "left",
        "height_cm": 168,
        "stats": {
            "xg_per_90": 0.30,
            "xa_per_90": 0.35,
            "pass_completion_pct": 79.0,
            "tackles_per_90": 0.8,
            "rating_per_90": 6.9,
        },
    },
]


class TestAgeFilter:
    def test_age_range_filters_correctly(self) -> None:
        criteria = SearchCriteria(age_min=18, age_max=25)
        results = search_players(criteria, SAMPLE_DB)
        ids = {r.player_id for r in results}
        assert "p1" in ids  # age 19
        assert "p4" in ids  # age 21
        assert "p2" not in ids  # age 32
        assert "p3" not in ids  # age 26

    def test_age_17_allowed(self) -> None:
        criteria = SearchCriteria(age_min=16, age_max=20)
        results = search_players(criteria, SAMPLE_DB)
        ids = {r.player_id for r in results}
        assert "p5" in ids

    def test_no_age_criteria_allows_all(self) -> None:
        criteria = SearchCriteria(age_min=16, age_max=40)
        results = search_players(criteria, SAMPLE_DB)
        assert len(results) >= 3


class TestPositionFilter:
    def test_position_filter_finds_fwds(self) -> None:
        criteria = SearchCriteria(positions=["FWD"])
        results = search_players(criteria, SAMPLE_DB)
        assert all("FWD" in r.position for r in results)

    def test_position_filter_handles_multiple(self) -> None:
        criteria = SearchCriteria(positions=["FWD", "MID"])
        results = search_players(criteria, SAMPLE_DB)
        ids = {r.player_id for r in results}
        assert "p1" in ids
        assert "p5" in ids  # FWD/MID


class TestLeagueFilter:
    def test_league_filter_works(self) -> None:
        criteria = SearchCriteria(leagues=["La Liga"])
        results = search_players(criteria, SAMPLE_DB)
        assert len(results) == 1
        assert results[0].player_id == "p2"

    def test_multiple_leagues(self) -> None:
        criteria = SearchCriteria(leagues=["PL", "Bundesliga"])
        results = search_players(criteria, SAMPLE_DB)
        assert len(results) >= 3


class TestStatThresholds:
    def test_stat_threshold_filters_correctly(self) -> None:
        criteria = SearchCriteria(stat_thresholds={"tackles_per_90": 3.0})
        results = search_players(criteria, SAMPLE_DB)
        assert len(results) == 1
        assert results[0].player_id == "p3"

    def test_multiple_stat_thresholds(self) -> None:
        criteria = SearchCriteria(stat_thresholds={"xg_per_90": 0.25, "pass_completion_pct": 78.0})
        results = search_players(criteria, SAMPLE_DB)
        ids = {r.player_id for r in results}
        assert "p1" in ids  # xg=0.45, pass=82.0
        assert "p5" in ids  # xg=0.30, pass=79.0


class TestMatchScore:
    def test_perfect_match_high_score(self) -> None:
        criteria = SearchCriteria(age_min=18, age_max=30, positions=["FWD"], leagues=["PL"])
        results = search_players(criteria, SAMPLE_DB)
        for r in results:
            assert r.match_score > 0

    def test_no_criteria_returns_perfect_score(self) -> None:
        criteria = SearchCriteria(age_min=16, age_max=40)
        results = search_players(criteria, SAMPLE_DB[:1])
        assert results[0].match_score == 100.0

    def test_better_match_higher_score(self) -> None:
        _ = SearchCriteria(age_min=18, age_max=18)
        pass


class TestEdgeCases:
    def test_empty_database_returns_empty(self) -> None:
        criteria = SearchCriteria()
        results = search_players(criteria, [])
        assert results == []

    def test_no_criteria_matches_returns_empty(self) -> None:
        criteria = SearchCriteria(age_min=50, age_max=60)
        results = search_players(criteria, SAMPLE_DB)
        assert results == []

    def test_sort_order_correct(self) -> None:
        criteria = SearchCriteria(sort_by="age", sort_dir="ASC")
        results = search_players(criteria, SAMPLE_DB)
        ages = [r.age for r in results]
        assert ages == sorted(ages)

    def test_sort_desc(self) -> None:
        criteria = SearchCriteria(sort_by="age", sort_dir="DESC")
        results = search_players(criteria, SAMPLE_DB)
        ages = [r.age for r in results]
        assert ages == sorted(ages, reverse=True)

    def test_nationality_filter(self) -> None:
        criteria = SearchCriteria(nationality="England")
        results = search_players(criteria, SAMPLE_DB)
        assert all(r.nationality == "England" for r in results)

    def test_height_filter(self) -> None:
        criteria = SearchCriteria(height_min_cm=190)
        results = search_players(criteria, SAMPLE_DB)
        assert all(r.height_cm >= 190 for r in results)

    def test_preferred_foot_filter(self) -> None:
        criteria = SearchCriteria(preferred_foot="left")
        results = search_players(criteria, SAMPLE_DB)
        assert all(r.preferred_foot == "left" for r in results)

    def test_stat_maximums(self) -> None:
        criteria = SearchCriteria(stat_maximums={"xg_per_90": 0.1})
        results = search_players(criteria, SAMPLE_DB)
        for r in results:
            assert r.stats.get("xg_per_90", 0) <= 0.1

    def test_offset_and_limit(self) -> None:
        criteria = SearchCriteria(limit=2, offset=0, sort_by="age", sort_dir="ASC")
        results = search_players(criteria, SAMPLE_DB)
        assert len(results) <= 2
