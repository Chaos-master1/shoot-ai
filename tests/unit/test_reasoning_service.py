"""Tests for ReasoningService — tactical diagnosis engine."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()

# ---------------------------------------------------------------------------
# Stub kawkab.services package so sub-module imports resolve
# ---------------------------------------------------------------------------
if "kawkab.services" not in sys.modules:
    _svc_mod = types.ModuleType("kawkab.services")
    _svc_mod.__path__ = []
    sys.modules["kawkab.services"] = _svc_mod

# ---------------------------------------------------------------------------
# Stub analysis_service — loading the real one triggers a cascade of
# core imports (kawkab.core.events, xg_model, pitch_control, player_rating,
# cv_service). We provide the three classes reasoning_service needs.
# ---------------------------------------------------------------------------


@dataclass
class _TeamStats:
    team_name: str = ""
    possession_pct: float = 0.0
    passes_completed: int = 0
    passes_attempted: int = 0
    shots: int = 0
    shots_on_target: int = 0
    tackles: int = 0
    corners: int = 0
    fouls: int = 0
    distance_covered_km: float = 0.0


@dataclass
class _PlayerStats:
    track_id: int = 0
    jersey_number: int | None = None
    name: str | None = None
    team: str | None = None
    position: str | None = None
    distance_covered_m: float = 0.0
    max_speed_kmh: float = 0.0
    avg_speed_kmh: float = 0.0
    passes_attempted: int = 0
    passes_completed: int = 0
    shots: int = 0
    tackles: int = 0
    interceptions: int = 0


@dataclass
class _MatchAnalysis:
    match_id: int = 0
    duration_seconds: float = 0.0
    home_team: _TeamStats = field(default_factory=_TeamStats)
    away_team: _TeamStats = field(default_factory=_TeamStats)
    players: dict[int, _PlayerStats] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    pass_network: dict = field(default_factory=dict)
    formations: dict = field(default_factory=dict)
    pressing_intensity: float = 0.0


if "kawkab.services.analysis_service" not in sys.modules:
    _ana_mod = types.ModuleType("kawkab.services.analysis_service")
    _ana_mod.MatchAnalysis = _MatchAnalysis
    _ana_mod.TeamStats = _TeamStats
    _ana_mod.PlayerStats = _PlayerStats
    sys.modules["kawkab.services.analysis_service"] = _ana_mod

# ---------------------------------------------------------------------------
# Load knowledge_service and reasoning_service
# ---------------------------------------------------------------------------
if "kawkab.services.knowledge_service" not in sys.modules:
    _kmod = load_service_module("kawkab.services.knowledge_service", "knowledge_service.py")
else:
    _kmod = sys.modules["kawkab.services.knowledge_service"]

_mod = load_service_module("reason_test", "reasoning_service.py")

TacticalRule = _kmod.TacticalRule
Drill = _kmod.Drill
KnowledgeService = _kmod.KnowledgeService
MatchAnalysis = _MatchAnalysis
TeamStats = _TeamStats
PlayerStats = _PlayerStats

ReasoningService = _mod.ReasoningService
Diagnosis = _mod.Diagnosis
DiagnosisReport = _mod.DiagnosisReport


# ===========================================================================
# Helpers
# ===========================================================================


def _make_rule(
    rule_id: str,
    pattern_type: str,
    severity: str = "medium",
    category: str = "defensive",
    hypotheses: list | None = None,
):
    h = hypotheses or [
        {
            "condition": "test",
            "action": "fix",
            "coaching_notes": {"en": "Fix this issue", "ar": "اصلح هذه المشكلة"},
            "recommended_drills": ["D001"],
        }
    ]
    return TacticalRule(
        rule_id=rule_id,
        category=category,
        subcategory="general",
        severity=severity,
        names={"en": rule_id, "ar": f"قاعدة {rule_id}"},
        description={"en": "Test rule"},
        pattern_signature={"type": pattern_type},
        hypotheses=h,
        recommended_drills=["D001"],
        sources=["test"],
    )


def _make_diagnosis(
    rule_id: str,
    name: str,
    confidence: float = 0.8,
    severity: str = "medium",
    drills: list[str] | None = None,
):
    return Diagnosis(
        rule_id=rule_id,
        rule_name=name,
        rule_name_ar=f"اسم {name}",
        category="defensive",
        severity=severity,
        confidence=confidence,
        evidence={"key": "value"},
        explanation=f"Explanation for {name}",
        explanation_ar=f"شرح لـ {name}",
        recommended_drills=drills or ["D001"],
    )


def _make_match_analysis(match_id: int = 1, duration: float = 3600.0):
    home = TeamStats(team_name="Home", possession_pct=55.0)
    away = TeamStats(team_name="Away", possession_pct=45.0)
    return MatchAnalysis(
        match_id=match_id,
        duration_seconds=duration,
        home_team=home,
        away_team=away,
        players={},
        formations={"home": {"line_height": 0.75}},
    )


# ===========================================================================
# Tests
# ===========================================================================


class TestInit:
    def test_init_creates_service(self):
        kb = MagicMock(spec=KnowledgeService)
        svc = ReasoningService(kb)
        assert svc.kb is kb
        assert svc._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_loads_kb(self):
        kb = MagicMock(spec=KnowledgeService)
        kb.initialize = AsyncMock()
        kb.stats = {"rules": 10}
        svc = ReasoningService(kb)
        await svc.initialize()
        assert svc._initialized is True
        kb.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_initialize_skips_if_already_done(self):
        kb = MagicMock(spec=KnowledgeService)
        kb.initialize = AsyncMock()
        svc = ReasoningService(kb)
        svc._initialized = True
        await svc.initialize()
        kb.initialize.assert_not_called()


class TestCheckZoneConcession:
    def test_high_zone_concession_detected(self):
        svc = ReasoningService(MagicMock())
        rule = _make_rule("R001", "zone_based_goal_concession")
        rule.pattern_signature["zone"] = "left_channel"
        analysis = _make_match_analysis()
        events = [
            {"type": "goal", "zone": "left_channel"},
            {"type": "goal", "zone": "left_channel"},
            {"type": "goal", "zone": "center"},
        ]
        conf, ev = svc._check_zone_concession(rule, analysis, events)
        assert conf > 0.5
        assert ev["zone_goals"] == 2

    def test_zone_concession_no_goals(self):
        svc = ReasoningService(MagicMock())
        rule = _make_rule("R001", "zone_based_goal_concession")
        rule.pattern_signature["zone"] = "left_channel"
        analysis = _make_match_analysis()
        conf, ev = svc._check_zone_concession(rule, analysis, [])
        assert conf == 0.0
        assert ev == {}


class TestCheckPossessionLoss:
    def test_possession_loss_detected(self):
        svc = ReasoningService(MagicMock())
        rule = _make_rule("R002", "zone_based_possession_loss")
        analysis = _make_match_analysis()
        events = [
            {"type": "turnover", "zone": "defensive_third"},
            {"type": "turnover", "zone": "defensive_third"},
            {"type": "turnover", "zone": "middle_third"},
            {"type": "turnover", "zone": "final_third"},
        ]
        conf, ev = svc._check_possession_loss(rule, analysis, events)
        assert conf > 0.0

    def test_possession_loss_low_pct(self):
        svc = ReasoningService(MagicMock())
        rule = _make_rule("R002", "zone_based_possession_loss")
        analysis = _make_match_analysis()
        events = [
            {"type": "turnover", "zone": "final_third"},
            {"type": "turnover", "zone": "final_third"},
        ]
        conf, ev = svc._check_possession_loss(rule, analysis, events)
        assert conf == 0.0


class TestCheckHighLine:
    def test_high_line_detected(self):
        svc = ReasoningService(MagicMock())
        rule = _make_rule("R003", "through_balls_behind_defense")
        analysis = _make_match_analysis()
        events = [
            {"zone": "behind_defensive_line"},
            {"zone": "behind_defensive_line"},
        ]
        conf, ev = svc._check_high_line(rule, analysis, events)
        assert conf > 0.0

    def test_high_line_no_formations(self):
        svc = ReasoningService(MagicMock())
        rule = _make_rule("R003", "through_balls_behind_defense")
        analysis = _make_match_analysis()
        analysis.formations = {}
        conf, ev = svc._check_high_line(rule, analysis, [])
        assert conf == 0.0


class TestCheckCounterAttack:
    def test_counter_attack_detected(self):
        svc = ReasoningService(MagicMock())
        rule = _make_rule("R004", "counter_attack_conceded")
        analysis = _make_match_analysis()
        events = [
            {"type": "shot", "situation": "counter_attack", "outcome": "goal"},
            {"type": "shot", "situation": "counter_attack", "outcome": "saved"},
            {"type": "shot", "situation": "counter_attack", "outcome": "blocked"},
        ]
        conf, ev = svc._check_counter_attack(rule, analysis, events)
        assert conf > 0.0

    def test_counter_attack_few_shots(self):
        svc = ReasoningService(MagicMock())
        rule = _make_rule("R004", "counter_attack_conceded")
        analysis = _make_match_analysis()
        events = [
            {"type": "shot", "situation": "counter_attack", "outcome": "goal"},
        ]
        conf, ev = svc._check_counter_attack(rule, analysis, events)
        assert conf == 0.0


class TestCheckSetPiece:
    def test_set_piece_weakness_detected(self):
        svc = ReasoningService(MagicMock())
        rule = _make_rule("R005", "set_piece_goals_conceded")
        analysis = _make_match_analysis()
        events = [
            {"type": "goal", "situation": "corner"},
            {"type": "goal", "situation": "free_kick"},
            {"type": "goal", "situation": "open_play"},
        ]
        conf, ev = svc._check_set_piece(rule, analysis, events)
        assert conf > 0.0

    def test_set_piece_no_goals(self):
        svc = ReasoningService(MagicMock())
        rule = _make_rule("R005", "set_piece_goals_conceded")
        analysis = _make_match_analysis()
        conf, ev = svc._check_set_piece(rule, analysis, [])
        assert conf == 0.0


class TestCheckTurnovers:
    def test_high_turnover_detected(self):
        svc = ReasoningService(MagicMock())
        rule = _make_rule("R006", "high_turnover_rate")
        analysis = _make_match_analysis()
        events = (
            [{"type": "turnover", "zone": "defensive_third"}] * 25
            + [{"type": "turnover", "zone": "middle_third"}] * 20
            + [{"type": "pass", "team": "home", "completed": True}] * 20
        )
        conf, ev = svc._check_turnovers(rule, analysis, events)
        assert conf > 0.0

    def test_high_turnover_no_events(self):
        svc = ReasoningService(MagicMock())
        rule = _make_rule("R006", "high_turnover_rate")
        analysis = _make_match_analysis()
        conf, ev = svc._check_turnovers(rule, analysis, [])
        assert conf == 0.0


class TestCheckLateGame:
    def test_late_game_decline_detected(self):
        svc = ReasoningService(MagicMock())
        rule = _make_rule("R007", "late_game_decline")
        analysis = _make_match_analysis(duration=3600.0)
        events = (
            [{"type": "shot", "timestamp": 500}] * 2
            + [{"type": "goal", "timestamp": 500}]
            + [{"type": "shot", "timestamp": 3000}] * 5
            + [{"type": "goal", "timestamp": 3000}] * 2
        )
        conf, ev = svc._check_late_game(rule, analysis, events)
        assert conf > 0.0

    def test_late_game_short_match(self):
        svc = ReasoningService(MagicMock())
        rule = _make_rule("R007", "late_game_decline")
        analysis = _make_match_analysis(duration=30.0)
        conf, ev = svc._check_late_game(rule, analysis, [])
        assert conf == 0.0


class TestCheckWidePlay:
    def test_poor_wide_play_detected(self):
        svc = ReasoningService(MagicMock())
        rule = _make_rule("R008", "poor_wide_play")
        analysis = _make_match_analysis()
        events = [
            {"type": "cross", "outcome": "blocked"},
            {"type": "cross", "outcome": "missed"},
            {"type": "cross", "outcome": "blocked"},
        ]
        conf, ev = svc._check_wide_play(rule, analysis, events)
        assert conf > 0.0

    def test_poor_wide_play_no_crosses(self):
        svc = ReasoningService(MagicMock())
        rule = _make_rule("R008", "poor_wide_play")
        analysis = _make_match_analysis()
        conf, ev = svc._check_wide_play(rule, analysis, [])
        assert conf == 0.0


class TestDiagnoseMatch:
    @pytest.mark.asyncio
    async def test_diagnose_match_returns_report(self):
        kb = MagicMock(spec=KnowledgeService)
        kb.initialize = AsyncMock()
        kb.stats = {"rules": 3}
        kb.get_all_rules.return_value = [
            _make_rule("R001", "zone_based_goal_concession"),
            _make_rule("R002", "zone_based_possession_loss"),
            _make_rule("R003", "poor_wide_play"),
        ]
        kb.get_drill.return_value = Drill(
            drill_id="D001",
            name="Passing Drill",
            category="technical",
            targets=["accuracy"],
            duration_min=15,
            players_required=6,
            intensity="medium",
            equipment=[],
            space="half_pitch",
            setup="",
            rules=[],
            progressions=[],
            regressions=[],
            coaching_points=[],
            addresses_problems=[],
            source="test",
        )

        svc = ReasoningService(kb)
        analysis = _make_match_analysis()
        events = [
            {"type": "goal", "zone": "left_channel"},
            {"type": "goal", "zone": "left_channel"},
            {"type": "goal", "zone": "center"},
            {"type": "turnover", "zone": "defensive_third"},
            {"type": "cross", "outcome": "blocked"},
            {"type": "cross", "outcome": "missed"},
        ]
        report = await svc.diagnose_match(analysis, events)
        assert isinstance(report, DiagnosisReport)
        assert report.match_id == 1
        assert len(report.diagnoses) > 0
        assert report.confidence > 0.0

    @pytest.mark.asyncio
    async def test_diagnose_match_without_events(self):
        kb = MagicMock(spec=KnowledgeService)
        kb.initialize = AsyncMock()
        kb.stats = {"rules": 1}
        rule = _make_rule("R001", "zone_based_goal_concession")
        rule.pattern_signature["zone"] = "left_channel"
        kb.get_all_rules.return_value = [rule]
        kb.get_drill.return_value = Drill(
            drill_id="D001",
            name="Passing Drill",
            category="technical",
            targets=["accuracy"],
            duration_min=15,
            players_required=6,
            intensity="medium",
            equipment=[],
            space="half_pitch",
            setup="",
            rules=[],
            progressions=[],
            regressions=[],
            coaching_points=[],
            addresses_problems=[],
            source="test",
        )
        svc = ReasoningService(kb)
        analysis = _make_match_analysis()
        analysis.events = [
            {"type": "goal", "zone": "left_channel"},
            {"type": "goal", "zone": "left_channel"},
        ]
        report = await svc.diagnose_match(analysis)
        assert len(report.diagnoses) > 0

    @pytest.mark.asyncio
    async def test_diagnose_match_no_rules_match(self):
        kb = MagicMock(spec=KnowledgeService)
        kb.initialize = AsyncMock()
        kb.stats = {"rules": 1}
        kb.get_all_rules.return_value = [
            _make_rule("R001", "unknown_pattern_type"),
        ]
        svc = ReasoningService(kb)
        analysis = _make_match_analysis()
        report = await svc.diagnose_match(analysis, [])
        assert len(report.diagnoses) == 0
        assert "No critical tactical issues" in report.overall_assessment


class TestBuildPriorityActions:
    def test_builds_actions_from_diagnoses(self):
        kb = MagicMock(spec=KnowledgeService)
        kb.get_drill.return_value = Drill(
            drill_id="D001",
            name="Drill A",
            category="technical",
            targets=["accuracy"],
            duration_min=15,
            players_required=6,
            intensity="medium",
            equipment=[],
            space="half_pitch",
            setup="",
            rules=[],
            progressions=[],
            regressions=[],
            coaching_points=[],
            addresses_problems=[],
            source="test",
        )
        svc = ReasoningService(kb)
        diags = [_make_diagnosis("R001", "Issue A", 0.9, "high")]
        actions = svc._build_priority_actions(diags, "en")
        assert "Priority 1" in actions["en"][0]
        assert "Drill A" in actions["en"][0]

    def test_builds_arabic_actions(self):
        kb = MagicMock(spec=KnowledgeService)
        svc = ReasoningService(kb)
        diags = []
        actions = svc._build_priority_actions(diags, "ar")
        assert "لم يتم اكتشاف" in actions["ar"][0]

    def test_no_diagnoses_returns_default(self):
        kb = MagicMock(spec=KnowledgeService)
        svc = ReasoningService(kb)
        actions = svc._build_priority_actions([], "en")
        assert "No critical issues" in actions["en"][0]


class TestBuildOverallAssessment:
    def test_no_diagnoses(self):
        kb = MagicMock(spec=KnowledgeService)
        svc = ReasoningService(kb)
        analysis = _make_match_analysis()
        assessment = svc._build_overall_assessment([], analysis, "en")
        assert "No critical tactical issues" in assessment["en"]

    def test_with_diagnoses(self):
        kb = MagicMock(spec=KnowledgeService)
        svc = ReasoningService(kb)
        analysis = _make_match_analysis()
        diags = [_make_diagnosis("R001", "Issue A", 0.85)]
        assessment = svc._build_overall_assessment(diags, analysis, "en")
        assert "1 tactical issues" in assessment["en"]
        assert "Issue A" in assessment["en"]


class TestTestRule:
    @pytest.mark.asyncio
    async def test_unknown_pattern_returns_none(self):
        kb = MagicMock(spec=KnowledgeService)
        svc = ReasoningService(kb)
        rule = _make_rule("R999", "nonexistent_pattern")
        analysis = _make_match_analysis()
        result = await svc._test_rule(rule, analysis, [])
        assert result is None

    @pytest.mark.asyncio
    async def test_low_confidence_returns_none(self):
        kb = MagicMock(spec=KnowledgeService)
        svc = ReasoningService(kb)
        rule = _make_rule("R001", "zone_based_goal_concession")
        rule.pattern_signature["zone"] = "left_channel"
        analysis = _make_match_analysis()
        result = await svc._test_rule(rule, analysis, [])
        assert result is None
