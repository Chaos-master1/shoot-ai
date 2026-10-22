"""Tests for Formation Effectiveness vs Opponents module."""

from kawkab.core.formation_effectiveness import FormationEffectivenessAnalyzer


class TestAnalyzeVsFormation:
    def test_returns_stats(self):
        fea = FormationEffectivenessAnalyzer()
        data = {
            "matches": [
                {
                    "opponent_formation": "4-3-3",
                    "goals_scored": 2,
                    "goals_conceded": 1,
                    "xg_for": 1.5,
                    "xg_against": 0.8,
                    "possession": 55,
                    "pass_completion": 82,
                    "chances_created": 12,
                    "pressing_intensity": 15,
                },
                {
                    "opponent_formation": "4-3-3",
                    "goals_scored": 0,
                    "goals_conceded": 3,
                    "xg_for": 0.5,
                    "xg_against": 2.0,
                    "possession": 45,
                    "pass_completion": 78,
                    "chances_created": 5,
                    "pressing_intensity": 10,
                },
            ]
        }
        result = fea.analyze_vs_formation(data, "4-3-3")
        assert result["matches_analyzed"] == 2
        assert result["avg_goals_scored"] == 1.0
        assert result["win_rate"] == 50.0

    def test_no_matches(self):
        fea = FormationEffectivenessAnalyzer()
        result = fea.analyze_vs_formation({"matches": []}, "4-4-2")
        assert result["matches_analyzed"] == 0
        assert result["win_rate"] == 0.0

    def test_no_matching_formation(self):
        fea = FormationEffectivenessAnalyzer()
        data = {
            "matches": [
                {
                    "opponent_formation": "4-3-3",
                    "goals_scored": 2,
                    "goals_conceded": 1,
                    "xg_for": 1.5,
                    "xg_against": 0.8,
                    "possession": 55,
                    "pass_completion": 82,
                    "chances_created": 12,
                    "pressing_intensity": 15,
                }
            ]
        }
        result = fea.analyze_vs_formation(data, "3-5-2")
        assert result["matches_analyzed"] == 0


class TestCompareFormationPerformances:
    def test_identifies_best(self):
        fea = FormationEffectivenessAnalyzer()
        history = [
            {
                "formation": "4-3-3",
                "opponent_formation": "4-4-2",
                "goals_scored": 3,
                "goals_conceded": 0,
                "xg_for": 2.0,
                "xg_against": 0.5,
                "possession": 60,
                "pass_completion": 85,
            },
            {
                "formation": "4-3-3",
                "opponent_formation": "4-3-3",
                "goals_scored": 1,
                "goals_conceded": 2,
                "xg_for": 1.0,
                "xg_against": 1.5,
                "possession": 50,
                "pass_completion": 80,
            },
            {
                "formation": "3-5-2",
                "opponent_formation": "4-4-2",
                "goals_scored": 0,
                "goals_conceded": 0,
                "xg_for": 0.5,
                "xg_against": 0.5,
                "possession": 48,
                "pass_completion": 75,
            },
        ]
        result = fea.compare_formation_performances(history)
        assert result["best_formation"] == "4-3-3"
        assert "formation_stats" in result

    def test_empty_history(self):
        fea = FormationEffectivenessAnalyzer()
        result = fea.compare_formation_performances([])
        assert result["best_formation"] == ""

    def test_single_formation(self):
        fea = FormationEffectivenessAnalyzer()
        history = [
            {
                "formation": "4-3-3",
                "opponent_formation": "4-4-2",
                "goals_scored": 2,
                "goals_conceded": 1,
                "xg_for": 1.5,
                "xg_against": 0.8,
                "possession": 55,
                "pass_completion": 82,
            },
        ]
        result = fea.compare_formation_performances(history)
        assert result["best_formation"] == "4-3-3"
        assert result["worst_formation"] == "4-3-3"


class TestComputeFormationFlexibilityScore:
    def test_high_flexibility(self):
        fea = FormationEffectivenessAnalyzer()
        history = [
            {
                "formation": "4-3-3",
                "goals_scored": 2,
                "goals_conceded": 1,
                "xg_for": 1.5,
                "xg_against": 0.8,
                "possession": 55,
                "pass_completion": 82,
            },
            {
                "formation": "3-5-2",
                "goals_scored": 1,
                "goals_conceded": 0,
                "xg_for": 0.8,
                "xg_against": 0.3,
                "possession": 50,
                "pass_completion": 80,
            },
            {
                "formation": "4-4-2",
                "goals_scored": 3,
                "goals_conceded": 2,
                "xg_for": 2.0,
                "xg_against": 1.5,
                "possession": 52,
                "pass_completion": 78,
            },
        ]
        result = fea.compute_formation_flexibility_score(history)
        assert result["formations_used"] == 3
        assert result["flexibility_score"] > 0

    def test_no_data(self):
        fea = FormationEffectivenessAnalyzer()
        result = fea.compute_formation_flexibility_score([])
        assert result["flexibility_score"] == 0.0
        assert result["verdict"] == "No data"

    def test_no_formations_recorded(self):
        fea = FormationEffectivenessAnalyzer()
        history = [{"goals_scored": 2, "goals_conceded": 1}]  # missing formation
        result = fea.compute_formation_flexibility_score(history)
        assert result["flexibility_score"] == 0.0
