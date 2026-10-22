"""Tests for formation-based lineup optimizer."""

from kawkab.core.lineup_optimizer import (
    FORMATION_TEMPLATES,
    POSITION_ROLES,
    LineupOptimizer,
    LineupSuggestion,
    _canonical_role,
)


class TestConstants:
    def test_all_7_formations_exist(self):
        expected = {"4-4-2", "4-3-3", "3-5-2", "4-2-3-1", "3-4-3", "5-3-2", "4-1-4-1"}
        assert set(FORMATION_TEMPLATES.keys()) == expected

    def test_each_formation_has_11_slots(self):
        for name, template in FORMATION_TEMPLATES.items():
            assert len(template) == 11, f"{name} has {len(template)} slots, expected 11"

    def test_gk_in_every_formation(self):
        for name, template in FORMATION_TEMPLATES.items():
            positions = [p[0] for p in template]
            assert "GK" in positions, f"{name} missing GK"

    def test_position_roles_mapping(self):
        assert "GK" in POSITION_ROLES
        assert "CB" in POSITION_ROLES
        assert "ST" in POSITION_ROLES

    def test_canonical_role(self):
        assert _canonical_role("striker") == "ST"
        assert _canonical_role("cb") == "CB"
        assert _canonical_role("unknown_role") is None


class TestSuggestLineup:
    def test_basic_442(self):
        opt = LineupOptimizer()
        suggestion = opt.suggest_lineup(formation="4-4-2")
        assert isinstance(suggestion, LineupSuggestion)
        assert suggestion.formation == "4-4-2"
        assert len(suggestion.slots) == 11

    def test_all_formations_return_11_slots(self):
        opt = LineupOptimizer()
        for fm in FORMATION_TEMPLATES:
            s = opt.suggest_lineup(formation=fm)
            assert len(s.slots) == 11, f"{fm} returned {len(s.slots)} slots"

    def test_unsupported_formation_returns_error_desc(self):
        opt = LineupOptimizer()
        s = opt.suggest_lineup(formation="1-2-7")
        assert s.formation == "1-2-7"
        assert "Unsupported" in s.description

    def test_home_away_mirror(self):
        opt = LineupOptimizer()
        home = opt.suggest_lineup(formation="4-4-2", home_away="home")
        away = opt.suggest_lineup(formation="4-4-2", home_away="away")
        home_xs = [sl.x for sl in home.slots]
        away_xs = [sl.x for sl in away.slots]
        for i in range(11):
            assert abs(home_xs[i] + away_xs[i] - 105.0) < 1.0

    def test_missing_players_handled_gracefully(self):
        opt = LineupOptimizer()
        s = opt.suggest_lineup(formation="4-4-2", players=None)
        assert len(s.slots) == 11

    def test_players_assigned_to_slots(self):
        opt = LineupOptimizer()
        players = [
            {"track_id": 1, "role": "GK", "name": "Keeper"},
            {"track_id": 2, "role": "CB", "name": "Def1"},
            {"track_id": 3, "role": "CB", "name": "Def2"},
            {"track_id": 4, "role": "LB", "name": "LeftBack"},
            {"track_id": 5, "role": "RB", "name": "RightBack"},
            {"track_id": 6, "role": "CM", "name": "Mid1"},
            {"track_id": 7, "role": "CM", "name": "Mid2"},
            {"track_id": 8, "role": "LM", "name": "LeftMid"},
            {"track_id": 9, "role": "RM", "name": "RightMid"},
            {"track_id": 10, "role": "ST", "name": "Striker1"},
            {"track_id": 11, "role": "ST", "name": "Striker2"},
        ]
        s = opt.suggest_lineup(formation="4-4-2", players=players)
        assert len(s.slots) >= 11
        assigned_roles = [sl.role for sl in s.slots]
        assert "Keeper" in assigned_roles

    def test_extra_players_become_subs(self):
        opt = LineupOptimizer()
        players = [
            {"track_id": 1, "role": "GK", "name": "Keeper"},
            {"track_id": 2, "role": "ST", "name": "Striker1"},
            {"track_id": 3, "role": "ST", "name": "Striker2"},
            {"track_id": 4, "role": "CB", "name": "CB1"},
            {"track_id": 5, "role": "CB", "name": "CB2"},
            {"track_id": 6, "role": "LB", "name": "LB1"},
            {"track_id": 7, "role": "RB", "name": "RB1"},
            {"track_id": 8, "role": "CM", "name": "CM1"},
            {"track_id": 9, "role": "CM", "name": "CM2"},
            {"track_id": 10, "role": "LM", "name": "LM1"},
            {"track_id": 11, "role": "RM", "name": "RM1"},
            {"track_id": 12, "role": "ST", "name": "Extra"},
        ]
        s = opt.suggest_lineup(formation="4-4-2", players=players)
        sub_roles = [sl.role for sl in s.slots if sl.position_name == "SUB"]
        assert len(sub_roles) >= 1
        assert "Extra" in sub_roles

    def test_confidence_in_range(self):
        opt = LineupOptimizer()
        s = opt.suggest_lineup(formation="4-4-2", opponent_formation="4-3-3")
        assert 0.0 <= s.confidence <= 1.0

    def test_description_contains_formation(self):
        opt = LineupOptimizer()
        s = opt.suggest_lineup(formation="4-4-2", opponent_formation="3-5-2")
        assert "4-4-2" in s.description
        assert "3-5-2" in s.description

    def test_away_mirror_x(self):
        opt = LineupOptimizer()
        s = opt.suggest_lineup(formation="4-4-2", home_away="away")
        for sl in s.slots:
            assert sl.x <= 105.0 + 1.0

    def test_slot_dicts_have_keys(self):
        opt = LineupOptimizer()
        s = opt.suggest_lineup(formation="4-4-2")
        for sl in s.slots:
            d = sl.to_dict()
            for key in ("position", "x", "y", "role"):
                assert key in d

    def test_suggestion_to_dict(self):
        opt = LineupOptimizer()
        s = opt.suggest_lineup(formation="4-4-2")
        d = s.to_dict()
        assert "formation" in d
        assert "slots" in d
        assert "confidence" in d


class TestCompareFormations:
    def test_empty_list_returns_empty_result(self):
        opt = LineupOptimizer()
        result = opt.compare_formations([])
        assert result.best_formation == ""

    def test_single_formation(self):
        opt = LineupOptimizer()
        result = opt.compare_formations(["4-4-2"])
        assert result.best_formation == "4-4-2"
        assert len(result.suggestions) == 1

    def test_multiple_formations_ranked(self):
        opt = LineupOptimizer()
        result = opt.compare_formations(["4-4-2", "4-3-3", "3-5-2"], opponent_formation="4-4-2")
        assert len(result.suggestions) == 3
        assert result.suggestions[0].confidence >= result.suggestions[1].confidence
        assert result.suggestions[1].confidence >= result.suggestions[2].confidence

    def test_result_to_dict_keys(self):
        opt = LineupOptimizer()
        result = opt.compare_formations(["4-4-2", "4-3-3"])
        d = result.to_dict()
        assert "suggestions" in d
        assert "best_formation" in d
        assert "best_confidence" in d

    def test_best_confidence_non_negative(self):
        opt = LineupOptimizer()
        result = opt.compare_formations(["4-4-2", "4-3-3"], opponent_formation="5-3-2")
        assert result.best_confidence >= 0.0

    def test_empty_formations_to_dict(self):
        opt = LineupOptimizer()
        result = opt.compare_formations([])
        d = result.to_dict()
        assert d["best_formation"] == ""
        assert d["best_confidence"] == 0.0


class TestRoleMatching:
    def test_direct_match(self):
        score = LineupOptimizer._role_match_score("GK", "gk")
        assert score == 1.0

    def test_alias_match(self):
        score = LineupOptimizer._role_match_score("GK", "goalkeeper")
        assert score == 0.9

    def test_empty_role_returns_zero(self):
        score = LineupOptimizer._role_match_score("ST", "")
        assert score == 0.0

    def test_unrecognized_role_returns_zero(self):
        score = LineupOptimizer._role_match_score("GK", "totally_bogus_role_999")
        assert score == 0.0

    def test_fuzzy_partial_match(self):
        score = LineupOptimizer._role_match_score("CM", "central_midfielder")
        assert score >= 0.4
