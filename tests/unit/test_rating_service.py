"""Tests for RatingService."""

from __future__ import annotations

import pytest

from kawkab.services.rating_service import RatingService


class TestRatingService:
    def test_no_players_returns_empty(self):
        svc = RatingService()
        ratings = svc.compute_ratings([], [])
        assert ratings == []

    def test_single_player_pass_accuracy(self):
        svc = RatingService()
        events = [
            {"type": "pass", "from_track_id": 1, "completed": True},
            {"type": "pass", "from_track_id": 1, "completed": True},
            {"type": "pass", "from_track_id": 1, "completed": False},
        ]
        players = [{"track_id": 1, "name": "Player 1", "team": "home"}]
        ratings = svc.compute_ratings(events, players)
        assert len(ratings) == 1
        assert ratings[0]["pass_accuracy"] == pytest.approx(2.0 / 3.0, abs=0.001)

    def test_player_with_goal_gets_higher_rating(self):
        svc = RatingService()
        events = [
            {"type": "shot", "from_track_id": 1, "is_goal": True},
            {"type": "pass", "from_track_id": 1, "completed": True},
        ]
        players = [{"track_id": 1, "name": "Scorer", "team": "home"}]
        ratings = svc.compute_ratings(events, players)
        scorer_rating = ratings[0]["rating"]

        events2 = [{"type": "pass", "from_track_id": 2, "completed": True}]
        players2 = [{"track_id": 2, "name": "Non-scorer", "team": "home"}]
        ratings2 = svc.compute_ratings(events2, players2)
        assert scorer_rating > ratings2[0]["rating"]

    def test_tackles_contribute_to_rating(self):
        svc = RatingService()
        events = [
            {"type": "tackle", "from_track_id": 1},
            {"type": "tackle", "from_track_id": 1},
            {"type": "tackle", "from_track_id": 1},
        ]
        players = [{"track_id": 1, "name": "Defender", "team": "home"}]
        ratings = svc.compute_ratings(events, players)
        assert ratings[0]["tackles"] == 3

    def test_rating_between_0_and_100(self):
        svc = RatingService()
        events = [
            {"type": "pass", "from_track_id": 1, "completed": True},
            {"type": "pass", "from_track_id": 1, "completed": True},
            {"type": "shot", "from_track_id": 1, "is_goal": True},
            {"type": "tackle", "from_track_id": 1},
            {"type": "carry", "from_track_id": 1},
            {"type": "dribble", "from_track_id": 1},
        ]
        players = [{"track_id": 1, "name": "All-rounder", "team": "home"}]
        ratings = svc.compute_ratings(events, players)
        assert 0 <= ratings[0]["rating"] <= 100

    def test_multiple_players(self):
        svc = RatingService()
        events = [
            {"type": "pass", "from_track_id": 1, "completed": True},
            {"type": "pass", "from_track_id": 2, "completed": False},
            {"type": "shot", "from_track_id": 2, "is_goal": True},
        ]
        players = [
            {"track_id": 1, "name": "A", "team": "home"},
            {"track_id": 2, "name": "B", "team": "home"},
        ]
        ratings = svc.compute_ratings(events, players)
        assert len(ratings) == 2

    def test_carries_and_dribbles_counted(self):
        svc = RatingService()
        events = [
            {"type": "carry", "from_track_id": 1},
            {"type": "dribble", "from_track_id": 1},
            {"type": "dribble", "from_track_id": 1},
        ]
        players = [{"track_id": 1, "name": "Dribbler", "team": "home"}]
        ratings = svc.compute_ratings(events, players)
        # Rating should be > 0 due to carries + dribbles contribution
        assert ratings[0]["rating"] > 0
