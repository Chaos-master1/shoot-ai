"""Tests for FootballRulesService (IFAB Laws of the Game)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()
_svc = load_service_module("frs_test", "football_rules_service.py")
FootballRulesService = _svc.FootballRulesService
Law = _svc.Law
RestartType = _svc.RestartType

import pytest


@pytest.fixture
def rules() -> FootballRulesService:
    return FootballRulesService()


class TestLawsLoaded:
    def test_all_17_laws_loaded(self, rules: FootballRulesService) -> None:
        assert rules.available
        assert len(rules._laws) >= 17

    def test_law_1_field_of_play(self, rules: FootballRulesService) -> None:
        law = rules.get_law_summary(1)
        assert law is not None
        assert "Field" in law.get("name", "")

    def test_law_11_offside(self, rules: FootballRulesService) -> None:
        law = rules.get_law_summary(11)
        assert law is not None
        assert "Offside" in law.get("name", "")

    def test_law_14_penalty(self, rules: FootballRulesService) -> None:
        law = rules.get_law_summary(14)
        assert law is not None
        assert "Penalty" in law.get("name", "")

    def test_law_17_corner(self, rules: FootballRulesService) -> None:
        law = rules.get_law_summary(17)
        assert law is not None
        assert "Corner" in law.get("name", "")

    def test_get_all_laws_sorted(self, rules: FootballRulesService) -> None:
        laws = rules.get_all_laws()
        numbers = [law["number"] for law in laws]
        assert numbers == sorted(numbers)

    def test_get_unknown_law_returns_empty(self, rules: FootballRulesService) -> None:
        assert rules.get_law_summary(99) == {}


class TestClassifyFoul:
    def test_foul_in_penalty_area_home(self, rules: FootballRulesService) -> None:
        ref = rules.classify_event("foul", 92, 34, "home")
        assert ref.law == 14
        assert ref.restart == RestartType.PENALTY_KICK

    def test_foul_in_penalty_area_away(self, rules: FootballRulesService) -> None:
        ref = rules.classify_event("foul", 13, 34, "away")
        assert ref.law == 14
        assert ref.restart == RestartType.PENALTY_KICK

    def test_foul_outside_penalty_area(self, rules: FootballRulesService) -> None:
        ref = rules.classify_event("foul", 60, 34, "home")
        assert ref.law == 12
        assert ref.restart == RestartType.DIRECT_FREE_KICK

    def test_foul_card_likely_in_box(self, rules: FootballRulesService) -> None:
        ref = rules.classify_event("foul", 92, 34, "home")
        assert ref.card_likely in {"yellow", "yellow_or_red"}


class TestClassifyBallOut:
    def test_ball_out_on_touchline(self, rules: FootballRulesService) -> None:
        ref = rules.classify_event("ball_out", 50, 0, "home")
        assert ref.restart == RestartType.THROW_IN
        assert ref.law == 15

    def test_ball_out_over_home_goal_attack_touch(self, rules: FootballRulesService) -> None:
        ref = rules.classify_event("ball_out", 0, 34, "home")
        assert ref.restart == RestartType.CORNER_KICK
        assert ref.law == 17

    def test_ball_out_over_home_goal_defense_touch(self, rules: FootballRulesService) -> None:
        ref = rules.classify_event("ball_out", 0, 34, "away")
        assert ref.restart == RestartType.GOAL_KICK
        assert ref.law == 16


class TestClassifyHandball:
    def test_handball_in_penalty_area(self, rules: FootballRulesService) -> None:
        ref = rules.classify_event("handball", 95, 34, "home")
        assert ref.law == 14
        assert ref.restart == RestartType.PENALTY_KICK

    def test_handball_outside_box(self, rules: FootballRulesService) -> None:
        ref = rules.classify_event("handball", 50, 34, "home")
        assert ref.law == 12
        assert ref.restart == RestartType.DIRECT_FREE_KICK


class TestClassifyOther:
    def test_offside(self, rules: FootballRulesService) -> None:
        ref = rules.classify_event("offside", 80, 34, "home")
        assert ref.law == 11
        assert ref.restart == RestartType.INDIRECT_FREE_KICK

    def test_goal(self, rules: FootballRulesService) -> None:
        ref = rules.classify_event("goal", 50, 34, "home")
        assert ref.law == 10
        assert ref.restart is None

    def test_unknown_event(self, rules: FootballRulesService) -> None:
        ref = rules.classify_event("dance_party", 50, 34, "home")
        assert ref.law == 0


class TestOffside:
    def test_clearly_offside(self, rules: FootballRulesService) -> None:
        result = rules.is_offside(
            attacker_x=85, second_last_defender_x=75, ball_x=78, attacking_direction="right"
        )
        assert result.is_offside

    def test_clearly_onside(self, rules: FootballRulesService) -> None:
        result = rules.is_offside(
            attacker_x=70, second_last_defender_x=75, ball_x=78, attacking_direction="right"
        )
        assert not result.is_offside

    def test_level_with_ball(self, rules: FootballRulesService) -> None:
        result = rules.is_offside(
            attacker_x=78, second_last_defender_x=75, ball_x=78, attacking_direction="right"
        )
        assert not result.is_offside

    def test_left_direction(self, rules: FootballRulesService) -> None:
        result = rules.is_offside(
            attacker_x=20, second_last_defender_x=30, ball_x=27, attacking_direction="left"
        )
        assert result.is_offside

    def test_offside_explanation_mentions_law(self, rules: FootballRulesService) -> None:
        result = rules.is_offside(85, 75, 78, "right")
        assert "Law 11" in result.explanation


class TestRestartLookup:
    def test_get_restart_foul(self, rules: FootballRulesService) -> None:
        r = rules.get_restart_for_event("foul", 50, 34, "home")
        assert r == RestartType.DIRECT_FREE_KICK

    def test_get_restart_ball_out(self, rules: FootballRulesService) -> None:
        r = rules.get_restart_for_event("ball_out", 50, 0, "home")
        assert r == RestartType.THROW_IN
