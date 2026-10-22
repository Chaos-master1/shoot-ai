"""Tests for TrainingPlanGenerator — 4-week training plan from diagnoses."""

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
# Stub analysis_service — needed because reasoning_service imports it
# ---------------------------------------------------------------------------


@dataclass
class _TeamStats:
    team_name: str = ""
    possession_pct: float = 0.0


@dataclass
class _PlayerStats:
    track_id: int = 0


@dataclass
class _MatchAnalysis:
    match_id: int = 0
    duration_seconds: float = 0.0
    home_team: _TeamStats = field(default_factory=_TeamStats)
    away_team: _TeamStats = field(default_factory=_TeamStats)
    players: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    formations: dict = field(default_factory=dict)


if "kawkab.services.analysis_service" not in sys.modules:
    _ana_mod = types.ModuleType("kawkab.services.analysis_service")
    _ana_mod.MatchAnalysis = _MatchAnalysis
    _ana_mod.TeamStats = _TeamStats
    _ana_mod.PlayerStats = _PlayerStats
    sys.modules["kawkab.services.analysis_service"] = _ana_mod

# ---------------------------------------------------------------------------
# Load knowledge_service and reasoning_service with proper dotted names
# ---------------------------------------------------------------------------
if "kawkab.services.knowledge_service" not in sys.modules:
    _kmod = load_service_module("kawkab.services.knowledge_service", "knowledge_service.py")
else:
    _kmod = sys.modules["kawkab.services.knowledge_service"]

if "kawkab.services.reasoning_service" not in sys.modules:
    _rmod = load_service_module("kawkab.services.reasoning_service", "reasoning_service.py")
else:
    _rmod = sys.modules["kawkab.services.reasoning_service"]

# ---------------------------------------------------------------------------
# Load training_plan_service
# ---------------------------------------------------------------------------
_mod = load_service_module("tp_test", "training_plan_service.py")

KnowledgeService = _kmod.KnowledgeService
TacticalRule = _kmod.TacticalRule
Drill = _kmod.Drill

Diagnosis = _rmod.Diagnosis
DiagnosisReport = _rmod.DiagnosisReport

TrainingPlanGenerator = _mod.TrainingPlanGenerator
TrainingPlan = _mod.TrainingPlan
TrainingWeek = _mod.TrainingWeek
DrillSession = _mod.DrillSession


# ===========================================================================
# Helpers
# ===========================================================================


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


def _make_diagnosis_report(match_id: int = 1, diagnoses: list | None = None):
    if diagnoses is None:
        diagnoses = [
            _make_diagnosis("R001", "High Press Trap", 0.85, "high"),
            _make_diagnosis("R002", "Set Piece Weakness", 0.72, "medium"),
        ]
    return DiagnosisReport(
        match_id=match_id,
        diagnoses=diagnoses,
        overall_assessment="Test assessment",
        overall_assessment_ar="تقييم اختبار",
        priority_actions=["Fix pressing"],
        priority_actions_ar=["إصلاح الضغط"],
        confidence=0.8,
    )


def _make_drill(drill_id: str = "D001", name: str = "Test Drill", duration: int = 15):
    return Drill(
        drill_id=drill_id,
        name=name,
        category="technical",
        targets=["accuracy"],
        duration_min=duration,
        players_required=6,
        intensity="medium",
        equipment=[],
        space="half_pitch",
        setup="Set up cones",
        rules=["Rule 1"],
        progressions=[],
        regressions=[],
        coaching_points=[],
        addresses_problems=[],
        source="test",
    )


def _make_kb_mock():
    kb = MagicMock(spec=KnowledgeService)
    kb.initialize = AsyncMock()
    kb.get_drill.return_value = _make_drill()
    return kb


# ===========================================================================
# Tests
# ===========================================================================


class TestInit:
    def test_init_creates_generator(self):
        kb = _make_kb_mock()
        gen = TrainingPlanGenerator(kb)
        assert gen.kb is kb

    def test_logs_initialized(self):
        kb = _make_kb_mock()
        gen = TrainingPlanGenerator(kb)
        assert "TrainingPlanGenerator" in str(type(gen).__name__)


class TestInitialize:
    @pytest.mark.asyncio
    async def test_initialize_calls_kb(self):
        kb = _make_kb_mock()
        gen = TrainingPlanGenerator(kb)
        await gen.initialize()
        kb.initialize.assert_awaited_once()


class TestGeneratePlan:
    @pytest.mark.asyncio
    async def test_generate_plan_returns_training_plan(self):
        kb = _make_kb_mock()
        gen = TrainingPlanGenerator(kb)
        report = _make_diagnosis_report()
        plan = await gen.generate_plan(report)
        assert isinstance(plan, TrainingPlan)
        assert plan.match_id == 1
        assert plan.duration_weeks == 4
        assert plan.total_drills > 0
        assert len(plan.weeks) == 4

    @pytest.mark.asyncio
    async def test_generate_plan_custom_duration(self):
        kb = _make_kb_mock()
        gen = TrainingPlanGenerator(kb)
        report = _make_diagnosis_report()
        plan = await gen.generate_plan(report, duration_weeks=2)
        assert len(plan.weeks) == 2

    @pytest.mark.asyncio
    async def test_generate_plan_empty_diagnoses(self):
        kb = _make_kb_mock()
        gen = TrainingPlanGenerator(kb)
        report = _make_diagnosis_report(diagnoses=[])
        plan = await gen.generate_plan(report)
        assert plan.total_drills == 0
        assert len(plan.weeks) == 4

    @pytest.mark.asyncio
    async def test_generate_plan_arabic(self):
        kb = _make_kb_mock()
        gen = TrainingPlanGenerator(kb)
        report = _make_diagnosis_report()
        plan = await gen.generate_plan(report, language="ar")
        assert plan.language == "ar"
        assert len(plan.priority_addressed) > 0

    @pytest.mark.asyncio
    async def test_generate_plan_plan_id_format(self):
        kb = _make_kb_mock()
        gen = TrainingPlanGenerator(kb)
        report = _make_diagnosis_report(match_id=42)
        plan = await gen.generate_plan(report)
        assert plan.plan_id.startswith("plan_42_")
        assert plan.match_id == 42


class TestBuildWeek:
    def test_build_week_returns_week_with_diagnoses(self):
        kb = _make_kb_mock()
        gen = TrainingPlanGenerator(kb)
        diags = [_make_diagnosis("R001", "Issue A")]
        week = gen._build_week(1, diags, 3, "en")
        assert isinstance(week, TrainingWeek)
        assert week.week_number == 1
        assert week.theme == "Foundation"
        assert len(week.sessions) == 3

    def test_build_week_themes_progress(self):
        kb = _make_kb_mock()
        gen = TrainingPlanGenerator(kb)
        diags = [_make_diagnosis("R001", "Issue A")]
        themes = ["Foundation", "Building", "Application", "Mastery"]
        for i, expected in enumerate(themes, 1):
            week = gen._build_week(i, diags, 2, "en")
            assert week.theme == expected

    def test_build_week_arabic_themes(self):
        kb = _make_kb_mock()
        gen = TrainingPlanGenerator(kb)
        diags = [_make_diagnosis("R001", "Issue A")]
        week = gen._build_week(1, diags, 2, "ar")
        assert week.theme == "التأسيس"

    def test_build_week_no_diagnoses_generic(self):
        kb = _make_kb_mock()
        gen = TrainingPlanGenerator(kb)
        week = gen._build_week(2, [], 2, "en")
        assert week.primary_focus == "General fitness and possession"


class TestBuildSessions:
    def test_build_sessions_creates_correct_count(self):
        kb = _make_kb_mock()
        gen = TrainingPlanGenerator(kb)
        primary = _make_diagnosis("R001", "Primary Issue")
        sessions = gen._build_sessions(1, primary, None, None, 3, "intro", "en")
        assert len(sessions) == 3
        assert sessions[0].intensity == "low"

    def test_build_sessions_progression_intensity(self):
        kb = _make_kb_mock()
        gen = TrainingPlanGenerator(kb)
        primary = _make_diagnosis("R001", "Issue")
        intensities = ["low", "medium", "high", "very_high"]
        for i, prog in enumerate(["intro", "basic", "intermediate", "advanced"]):
            sessions = gen._build_sessions(i + 1, primary, None, None, 2, prog, "en")
            assert sessions[0].intensity == intensities[i]

    def test_build_sessions_with_secondary(self):
        kb = _make_kb_mock()
        gen = TrainingPlanGenerator(kb)
        primary = _make_diagnosis("R001", "Primary", drills=["D001", "D002"])
        secondary = _make_diagnosis("R002", "Secondary", drills=["D003"])
        sessions = gen._build_sessions(1, primary, secondary, None, 2, "intro", "en")
        assert len(sessions) == 2


class TestExpectedImprovements:
    def test_build_expected_improvements_with_diagnoses(self):
        kb = _make_kb_mock()
        gen = TrainingPlanGenerator(kb)
        primary = _make_diagnosis("R001", "Pressing")
        improvements = gen._build_expected_improvements(primary, None, None, "en")
        assert len(improvements) == 1
        assert "Pressing" in improvements[0]

    def test_build_expected_improvements_arabic(self):
        kb = _make_kb_mock()
        gen = TrainingPlanGenerator(kb)
        primary = _make_diagnosis("R001", "Pressing")
        improvements = gen._build_expected_improvements(primary, None, None, "ar")
        assert len(improvements) == 1
        assert "تحسن" in improvements[0]

    def test_build_expected_improvements_no_diagnoses(self):
        kb = _make_kb_mock()
        gen = TrainingPlanGenerator(kb)
        improvements = gen._build_expected_improvements(None, None, None, "en")
        assert improvements == ["Maintain current form"]


class TestGenericWeek:
    def test_generic_week_returns_correct_structure(self):
        kb = _make_kb_mock()
        gen = TrainingPlanGenerator(kb)
        week = gen._generic_week(1, "en")
        assert week.sessions == []
        assert week.expected_improvements == []
        assert week.re_test_focus == "Re-analyze after week 4"

    def test_generic_week_arabic(self):
        kb = _make_kb_mock()
        gen = TrainingPlanGenerator(kb)
        week = gen._generic_week(3, "ar")
        assert week.theme == "التطبيق"

    def test_generic_week_all_weeks(self):
        kb = _make_kb_mock()
        gen = TrainingPlanGenerator(kb)
        themes_en = ["Foundation", "Building", "Application", "Mastery"]
        for i in range(1, 5):
            week = gen._generic_week(i, "en")
            assert week.theme == themes_en[i - 1]


class TestExportToDict:
    def test_export_to_dict_returns_serializable(self):
        kb = _make_kb_mock()
        gen = TrainingPlanGenerator(kb)
        import json

        report = _make_diagnosis_report()
        import asyncio

        plan = asyncio.run(gen.generate_plan(report))
        d = gen.export_to_dict(plan)
        assert d["plan_id"] == plan.plan_id
        assert d["match_id"] == 1
        assert d["duration_weeks"] == 4
        assert len(d["weeks"]) == 4
        for w in d["weeks"]:
            assert "week_number" in w
            assert "sessions" in w
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_export_empty_sessions(self):
        kb = _make_kb_mock()
        gen = TrainingPlanGenerator(kb)
        plan = TrainingPlan(
            plan_id="test",
            match_id=0,
            created_at="now",
            duration_weeks=1,
            weeks=[],
            total_drills=0,
            weekly_schedule={},
            priority_addressed=[],
            expected_overall_improvement="",
        )
        d = gen.export_to_dict(plan)
        assert d["plan_id"] == "test"
        assert d["weeks"] == []


class TestBuildOverallImprovement:
    def test_overall_improvement_english(self):
        kb = _make_kb_mock()
        gen = TrainingPlanGenerator(kb)
        diags = [_make_diagnosis("R001", "Pressing")]
        result = gen._build_overall_improvement(diags, "en")
        assert "After 4 weeks" in result
        assert "Pressing" in result

    def test_overall_improvement_arabic(self):
        kb = _make_kb_mock()
        gen = TrainingPlanGenerator(kb)
        diags = [_make_diagnosis("R001", "Pressing")]
        result = gen._build_overall_improvement(diags, "ar")
        assert "بعد 4 أسابيع" in result

    def test_overall_improvement_empty(self):
        kb = _make_kb_mock()
        gen = TrainingPlanGenerator(kb)
        result = gen._build_overall_improvement([], "en")
        assert result == ""


class TestBuildWeeklySchedule:
    def test_build_weekly_schedule_english(self):
        kb = _make_kb_mock()
        gen = TrainingPlanGenerator(kb)
        schedule = gen._build_weekly_schedule(3, "en")
        assert len(schedule) == 3
        assert "Monday" in schedule
        assert "Wednesday" in schedule
        assert "Friday" in schedule

    def test_build_weekly_schedule_arabic(self):
        kb = _make_kb_mock()
        gen = TrainingPlanGenerator(kb)
        schedule = gen._build_weekly_schedule(2, "ar")
        assert len(schedule) == 2
        assert "الإثنين" in schedule
        assert "الأربعاء" in schedule
