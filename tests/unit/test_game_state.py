"""Tests for game state analysis."""

from kawkab.core.game_state import GameStateReport, analyze_game_state


class TestGameState:
    def test_empty_data(self):
        result = analyze_game_state([], [])
        assert isinstance(result, GameStateReport)

    def test_no_goals_all_drawing(self):
        events = [
            {"type": "pass", "timestamp": 1.0, "team": "home", "completed": True},
            {"type": "shot", "timestamp": 2.0, "team": "home", "is_goal": False},
        ]
        frames = [
            {
                "timestamp": 1.0,
                "possession": True,
                "home_positions": [(50, 34)],
                "away_positions": [(70, 34)],
            },
            {
                "timestamp": 2.0,
                "possession": True,
                "home_positions": [(50, 34)],
                "away_positions": [(70, 34)],
            },
        ]
        result = analyze_game_state(events, frames)
        assert result.drawing.duration_s > 0
        assert result.home_winning.duration_s == 0

    def test_goal_changes_state(self):
        events = [
            {"type": "pass", "timestamp": 5.0, "team": "home", "completed": True},
            {"type": "shot", "timestamp": 10.0, "team": "home", "is_goal": True},
            {"type": "pass", "timestamp": 15.0, "team": "home", "completed": True},
        ]
        frames = [
            {
                "timestamp": t,
                "possession": True,
                "home_positions": [(50, 34)],
                "away_positions": [(70, 34)],
            }
            for t in range(0, 20)
        ]
        result = analyze_game_state(events, frames)
        assert result.home_winning.duration_s > 0
        assert result.drawing.duration_s > 0

    def test_to_dict(self):
        result = analyze_game_state([], [])
        d = result.to_dict()
        assert "home_winning" in d
        assert "drawing" in d
        assert "home_losing" in d
