"""Tests for game plan generator."""

from __future__ import annotations

from kawkab.core.game_plan import GamePlanGenerator, generate_game_plan


class TestGamePlanGenerator:
    def test_generates_valid_plan(self):
        gen = GamePlanGenerator()
        events = [{"type": "shot", "is_goal": True}, {"type": "pass"}]
        plan = gen.generate(events, opponent="FC Barcelona")
        assert plan["opponent"] == "FC Barcelona"
        assert plan["formation_recommendation"] != ""
        assert len(plan["key_players_to_neutralize"]) > 0
        assert "set" in plan["set_piece_plan"].lower() or "defend" in plan["set_piece_plan"].lower()

    def test_empty_events(self):
        gen = GamePlanGenerator()
        plan = gen.generate([], opponent="Test FC")
        assert plan["opponent"] == "Test FC"

    def test_default_opponent(self):
        gen = GamePlanGenerator()
        plan = gen.generate([])
        assert plan["opponent"] == "Unknown"

    def test_generate_game_plan_function(self):
        events = [{"type": "shot", "is_goal": False}]
        plan = generate_game_plan(events, "Real Madrid")
        assert isinstance(plan, dict)

    def test_many_shots_attacking_hint(self):
        gen = GamePlanGenerator()
        events = [{"type": "shot"} for _ in range(20)]
        plan = gen.generate(events, "Attacking Team")
        assert "attack" in plan["formation_recommendation"].lower()

    def test_scoreline_prediction_present(self):
        gen = GamePlanGenerator()
        plan = gen.generate([{"type": "shot"}])
        assert "-" in plan["scoreline_prediction"]
