"""Tests for Phase 6 Sprint 1 — Injury Risk Dashboard + Training Auto-Generate."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kawkab.core.injury_risk import InjuryRiskPredictor


class TestInjuryRiskBridge:
    """6 tests for injury risk bridge methods."""

    @pytest.mark.asyncio
    async def test_get_injury_risk_success(self):
        from kawkab.ui.bridge_handlers.bridge_analysis import AnalysisHandler

        bridge = MagicMock()
        services = {
            "storage_service": AsyncMock(),
            "knowledge_service": MagicMock(),
        }
        services["storage_service"].get_match_players.return_value = [
            {
                "track_id": 1,
                "name": "Player 1",
                "team": "home",
                "position": "MID",
                "jersey_number": "10",
            }
        ]
        services["storage_service"].get_match_events.return_value = []
        handler = AnalysisHandler(bridge, services)
        result = json.loads(await handler.get_injury_risk(1, 1))
        assert "risk_score" in result
        assert "acwr" in result
        assert "risk_level" in result
        assert "recovery_recommendation" in result
        assert result["player_name"] == "Player 1"

    @pytest.mark.asyncio
    async def test_get_injury_risk_unknown_player_still_returns(self):
        """Unknown player: honest 'Player not found' error, never a fabricated score.

        (The pre-merge duplicate fabricated ACWR from a synthetic sawtooth for
        ANY track_id, including nonexistent ones — this test now pins the
        honest contract instead.)
        """
        from kawkab.ui.bridge_handlers.bridge_analysis import AnalysisHandler

        bridge = MagicMock()
        services = {
            "storage_service": AsyncMock(),
            "knowledge_service": MagicMock(),
        }
        services["storage_service"].get_match_players.return_value = []
        services["storage_service"].get_match_events.return_value = []
        handler = AnalysisHandler(bridge, services)
        result = json.loads(await handler.get_injury_risk(1, 999))
        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_squad_injury_report_success(self):
        from kawkab.ui.bridge_handlers.bridge_analysis import AnalysisHandler

        bridge = MagicMock()
        services = {
            "storage_service": AsyncMock(),
            "knowledge_service": MagicMock(),
        }
        services["storage_service"].get_match_players.return_value = [
            {"track_id": 1, "name": "P1", "team": "home", "position": "MID", "jersey_number": "10"},
            {"track_id": 2, "name": "P2", "team": "away", "position": "FWD", "jersey_number": "9"},
        ]
        services["storage_service"].get_match_events.return_value = []
        handler = AnalysisHandler(bridge, services)
        result = json.loads(await handler.get_squad_injury_report(1))
        assert len(result["home_players"]) == 1
        assert len(result["away_players"]) == 1
        assert "avg_risk_home" in result
        assert "avg_risk_away" in result
        assert "high_risk_count" in result
        assert "total_players" in result

    @pytest.mark.asyncio
    async def test_get_squad_injury_report_empty(self):
        from kawkab.ui.bridge_handlers.bridge_analysis import AnalysisHandler

        bridge = MagicMock()
        services = {
            "storage_service": AsyncMock(),
            "knowledge_service": MagicMock(),
        }
        services["storage_service"].get_match_players.return_value = []
        services["storage_service"].get_match_events.return_value = []
        handler = AnalysisHandler(bridge, services)
        result = json.loads(await handler.get_squad_injury_report(1))
        assert result["home_players"] == []
        assert result["away_players"] == []
        assert result["total_players"] == 0

    @pytest.mark.asyncio
    async def test_get_squad_injury_report_high_risk(self):
        from kawkab.ui.bridge_handlers.bridge_analysis import AnalysisHandler

        bridge = MagicMock()
        services = {
            "storage_service": AsyncMock(),
            "knowledge_service": MagicMock(),
        }
        services["storage_service"].get_match_players.return_value = [
            {"track_id": 1, "name": "P1", "team": "home", "position": "MID", "jersey_number": "10"},
        ]
        services["storage_service"].get_match_events.return_value = [
            {"event_type": "sprint", "from_track_id": 1, "completed": True},
        ] * 30
        handler = AnalysisHandler(bridge, services)
        result = json.loads(await handler.get_squad_injury_report(1))
        assert result["total_players"] == 1

    @pytest.mark.asyncio
    async def test_get_injury_risk_error_safe(self):
        from kawkab.ui.bridge_handlers.bridge_analysis import AnalysisHandler

        bridge = MagicMock()
        services = {
            "storage_service": AsyncMock(),
            "knowledge_service": MagicMock(),
        }
        services["storage_service"].get_match_events.side_effect = Exception("DB err")
        handler = AnalysisHandler(bridge, services)
        result = json.loads(await handler.get_injury_risk(1, 1))
        assert "error" in result


class TestTrainingPlanBridge:
    """4 tests for training plan bridge methods with patched KnowledgeService."""

    @pytest.mark.asyncio
    @patch("kawkab.services.knowledge_service.KnowledgeService")
    async def test_generate_training_plan_success(self, mock_ks_cls):
        from kawkab.ui.bridge_handlers.bridge_analysis import AnalysisHandler

        mock_kb = MagicMock()
        mock_kb.initialize = AsyncMock()
        mock_kb.get_drill.return_value = None
        mock_kb.get_all_drills.return_value = []
        mock_ks_cls.return_value = mock_kb

        bridge = MagicMock()
        services = {
            "storage_service": AsyncMock(),
            "knowledge_service": MagicMock(),
        }
        services["storage_service"].get_match_events.return_value = [
            {"event_type": "pass", "from_track_id": 1, "completed": 1},
            {"event_type": "shot", "from_track_id": 2},
        ]
        services["storage_service"].get_match_players.return_value = []
        services["storage_service"].get_match.return_value = {
            "home_team": "Home",
            "away_team": "Away",
        }
        handler = AnalysisHandler(bridge, services)
        result = json.loads(await handler.generate_training_plan(1))
        assert result["success"] is True
        assert "plan" in result
        assert result["plan"]["duration_weeks"] == 4
        assert len(result["plan"]["weeks"]) == 4

    @pytest.mark.asyncio
    @patch("kawkab.services.knowledge_service.KnowledgeService")
    async def test_generate_training_plan_structure(self, mock_ks_cls):
        from kawkab.ui.bridge_handlers.bridge_analysis import AnalysisHandler

        mock_kb = MagicMock()
        mock_kb.initialize = AsyncMock()
        mock_kb.get_drill.return_value = None
        mock_kb.get_all_drills.return_value = []
        mock_ks_cls.return_value = mock_kb

        bridge = MagicMock()
        services = {
            "storage_service": AsyncMock(),
            "knowledge_service": MagicMock(),
        }
        services["storage_service"].get_match_events.return_value = []
        services["storage_service"].get_match_players.return_value = []
        services["storage_service"].get_match.return_value = {
            "home_team": "Home",
            "away_team": "Away",
        }
        handler = AnalysisHandler(bridge, services)
        result = json.loads(await handler.generate_training_plan(1))
        plan = result["plan"]
        assert "plan_id" in plan
        assert "weeks" in plan
        assert "weekly_schedule" in plan
        assert "priority_addressed" in plan
        assert "expected_overall_improvement" in plan

    @pytest.mark.asyncio
    @patch("kawkab.services.knowledge_service.KnowledgeService")
    async def test_generate_training_plan_weekly_themes(self, mock_ks_cls):
        from kawkab.ui.bridge_handlers.bridge_analysis import AnalysisHandler

        mock_kb = MagicMock()
        mock_kb.initialize = AsyncMock()
        mock_kb.get_drill.return_value = None
        mock_kb.get_all_drills.return_value = []
        mock_ks_cls.return_value = mock_kb

        bridge = MagicMock()
        services = {
            "storage_service": AsyncMock(),
            "knowledge_service": MagicMock(),
        }
        services["storage_service"].get_match_events.return_value = []
        services["storage_service"].get_match_players.return_value = []
        services["storage_service"].get_match.return_value = {
            "home_team": "Home",
            "away_team": "Away",
        }
        handler = AnalysisHandler(bridge, services)
        result = json.loads(await handler.generate_training_plan(1))
        themes = [w["theme"] for w in result["plan"]["weeks"]]
        assert themes == ["Foundation", "Building", "Application", "Mastery"]

    @pytest.mark.asyncio
    async def test_generate_training_plan_error_handling(self):
        from kawkab.ui.bridge_handlers.bridge_analysis import AnalysisHandler

        bridge = MagicMock()
        services = {
            "storage_service": AsyncMock(),
            "knowledge_service": MagicMock(),
        }
        services["storage_service"].get_match_events.side_effect = Exception("DB error")
        handler = AnalysisHandler(bridge, services)
        result = json.loads(await handler.generate_training_plan(1))
        assert "error" in result


class TestRecoveryRecommendations:
    """4 tests for recovery recommendation logic."""

    def test_recovery_low_risk(self):
        predictor = InjuryRiskPredictor()
        rec = predictor.compute_recovery_recommendation(0.1, "MID")
        assert rec == "full training"

    def test_recovery_moderate_risk(self):
        predictor = InjuryRiskPredictor()
        rec = predictor.compute_recovery_recommendation(0.3, "FWD")
        assert rec == "full training"

    def test_recovery_high_risk(self):
        predictor = InjuryRiskPredictor()
        rec = predictor.compute_recovery_recommendation(0.5, "DEF")
        assert rec == "modified training"

    def test_recovery_critical_risk(self):
        predictor = InjuryRiskPredictor()
        rec = predictor.compute_recovery_recommendation(0.8, "GK")
        assert rec == "rest day"


class TestACWRComputation:
    """4 tests for ACWR computation."""

    def test_acwr_normal(self):
        predictor = InjuryRiskPredictor()
        workload = [
            50,
            52,
            48,
            55,
            60,
            58,
            65,
            50,
            48,
            52,
            55,
            60,
            58,
            62,
            50,
            48,
            52,
            55,
            60,
            58,
            65,
            62,
            50,
            48,
            52,
            55,
            60,
            58,
        ]
        result = predictor.compute_acwr_overload(workload)
        assert result["acwr"] > 0
        assert result["risk_level"] in ("low", "moderate", "high", "critical")

    def test_acwr_insufficient_data(self):
        predictor = InjuryRiskPredictor()
        result = predictor.compute_acwr_overload([50, 52, 48])
        assert result["acwr"] == 0.0
        assert "Insufficient data" in result["recommendation"]

    def test_acwr_high_acute_load(self):
        predictor = InjuryRiskPredictor()
        workload = [10] * 21 + [100] * 7
        result = predictor.compute_acwr_overload(workload)
        assert result["acwr"] > 1.0

    def test_acwr_exact_28_days(self):
        predictor = InjuryRiskPredictor()
        workload = list(range(1, 29))
        result = predictor.compute_acwr_overload(workload)
        assert result["acwr"] > 0
        assert "risk_level" in result
