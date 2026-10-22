"""Training plan generator.

Generates a 4-week training plan based on tactical diagnoses from
the reasoning engine. Selects appropriate drills from the knowledge base
and structures them into a progressive overload program.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from kawkab.core.logging import get_logger
from kawkab.services.knowledge_service import KnowledgeService
from kawkab.services.reasoning_service import Diagnosis, DiagnosisReport

logger = get_logger(__name__)


@dataclass
class DrillSession:
    """A single training session."""

    week: int
    day: str
    focus: str
    drills: list[str]
    total_duration_min: int
    intensity: str
    md_offset: str = ""  # morphocycle position: MD+1, MD-3, MD-1, ... (empty = fixed weekday)


@dataclass
class TrainingWeek:
    """A week of training sessions."""

    week_number: int
    theme: str
    primary_focus: str
    secondary_focus: str
    sessions: list[DrillSession]
    expected_improvements: list[str]
    re_test_focus: str


@dataclass
class TrainingPlan:
    """Complete 4-week training plan."""

    plan_id: str
    match_id: int
    created_at: str
    duration_weeks: int
    weeks: list[TrainingWeek]
    total_drills: int
    weekly_schedule: dict[str, list[str]]
    priority_addressed: list[str]
    expected_overall_improvement: str
    re_test_at_end: bool = True
    language: str = "en"
    # Honesty: drill IDs referenced by this plan that the knowledge base
    # cannot resolve. Empty list = every drill resolves. The handler
    # surfaces this so a plan never silently shows phantom drills.
    unresolved_drills: list[str] = field(default_factory=list)
    # Club game-model principles the plan is built to develop (Phase B).
    # Empty when no active game model exists — the plan is then built
    # from diagnoses alone, and says so.
    game_model_principles: list[str] = field(default_factory=list)
    game_model_loaded: bool = False
    # Diagnosis provenance: which matches the underlying report pooled,
    # and whether it was a multi-match (min_matches-capable) report.
    based_on_match_ids: list[int] = field(default_factory=list)
    multi_match: bool = False


class TrainingPlanGenerator:
    """Generates 4-week training plans from diagnosis reports."""

    WEEKLY_SCHEDULE = {
        "en": ["Monday", "Wednesday", "Friday", "Saturday"],
        "ar": ["الإثنين", "الأربعاء", "الجمعة", "السبت"],
    }

    # Tactical Periodization morphocycle: the weekly cycle is structured
    # around match day (MD), not calendar weekdays. Recovery immediately
    # after the match, tactical/physical build mid-week, activation the
    # day before. (Frade's morphocycle; the MD+1→MD-1 rhythm used across
    # professional one-match weeks.)
    MORPHOCYCLE = [
        {
            "md_offset": "MD+1",
            "label_en": "MD+1 — Recovery",
            "label_ar": "MD+1 — استشفاء",
            "intensity": "low",
        },
        {
            "md_offset": "MD-3",
            "label_en": "MD-3 — Tactical build",
            "label_ar": "MD-3 — بناء تكتيكي",
            "intensity": "high",
        },
        {
            "md_offset": "MD-2",
            "label_en": "MD-2 — Physical + tactics",
            "label_ar": "MD-2 — بدني وتكتيكي",
            "intensity": "medium",
        },
        {
            "md_offset": "MD-1",
            "label_en": "MD-1 — Activation",
            "label_ar": "MD-1 — تفعيل",
            "intensity": "low",
        },
    ]

    def __init__(self, knowledge_service: KnowledgeService) -> None:
        self.kb = knowledge_service
        logger.info("TrainingPlanGenerator initialized")

    async def initialize(self) -> None:
        await self.kb.initialize()

    async def generate_plan(
        self,
        diagnosis: DiagnosisReport,
        duration_weeks: int = 4,
        training_days_per_week: int = 3,
        language: str = "en",
        use_morphocycle: bool = True,
        game_model: dict[str, Any] | None = None,
    ) -> TrainingPlan:
        """Generate a multi-week training plan from a diagnosis report.

        Args:
            diagnosis: DiagnosisReport from the reasoning engine
            duration_weeks: Plan length (default 4)
            training_days_per_week: Sessions per week (fixed-weekday mode)
            language: "en" or "ar"
            use_morphocycle: structure sessions around match day
                (MD+1 recovery → MD-1 activation). Default on: the
                morphocycle is the professional weekly rhythm; pass
                False for fixed weekdays.
            game_model: the club's active game model dict (as persisted
                via save_game_model), whose principles shape the plan.
                None/missing => plan built from diagnoses alone.

        Returns:
            TrainingPlan with weekly structure and drill selections.
            ``plan.unresolved_drills`` lists any drill IDs the knowledge
            base cannot resolve — callers must surface this, never hide it.
        """
        await self.initialize()

        plan_id = f"plan_{diagnosis.match_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        top_diagnoses = diagnosis.diagnoses[:5]

        game_model_principles: list[str] = []
        game_model_loaded = False
        if isinstance(game_model, dict) and game_model:
            # Keys follow the persisted schema (migration 032 game_model
            # columns: transition_attack / transition_defence).
            for key in (
                "in_possession",
                "out_of_possession",
                "transition_attack",
                "transition_defence",
            ):
                value = str(game_model.get(key) or "").strip()
                if value:
                    game_model_principles.append(value)
            non_negotiables = game_model.get("non_negotiables")
            if isinstance(non_negotiables, str):
                try:
                    non_negotiables = json.loads(non_negotiables)
                except (TypeError, json.JSONDecodeError):
                    non_negotiables = []
            if isinstance(non_negotiables, list):
                game_model_principles.extend(
                    str(n).strip() for n in non_negotiables if str(n).strip()
                )
            game_model_loaded = bool(game_model_principles)
            if not game_model_loaded:
                logger.warning("Game model provided but has no principle text; ignoring")

        weeks = []
        all_drill_ids = set()
        priority_addressed = []

        unresolved: set[str] = set()
        for week_num in range(1, duration_weeks + 1):
            week = self._build_week(
                week_num=week_num,
                diagnoses=top_diagnoses,
                training_days=training_days_per_week,
                language=language,
                use_morphocycle=use_morphocycle,
            )
            weeks.append(week)
            for session in week.sessions:
                for drill_id in session.drills:
                    all_drill_ids.add(drill_id)
                    if self.kb.get_drill(drill_id) is None:
                        unresolved.add(drill_id)
        if unresolved:
            logger.warning(
                f"Plan references {len(unresolved)} drills missing from the "
                f"knowledge base: {sorted(unresolved)}"
            )

        for d in top_diagnoses:
            priority_addressed.append(d.rule_name_ar if language == "ar" else d.rule_name)

        if use_morphocycle:
            schedule = self.build_morphocycle_schedule(language)
        else:
            schedule = self._build_weekly_schedule(training_days_per_week, language)

        overall_improvement = self._build_overall_improvement(top_diagnoses, language)

        plan = TrainingPlan(
            plan_id=plan_id,
            match_id=diagnosis.match_id,
            created_at=datetime.now().isoformat(),
            duration_weeks=duration_weeks,
            weeks=weeks,
            total_drills=len(all_drill_ids),
            weekly_schedule=schedule,
            priority_addressed=priority_addressed,
            expected_overall_improvement=overall_improvement,
            re_test_at_end=True,
            language=language,
            unresolved_drills=sorted(unresolved),
            game_model_principles=game_model_principles,
            game_model_loaded=game_model_loaded,
            based_on_match_ids=list(getattr(diagnosis, "match_ids", []) or []),
            multi_match=bool(getattr(diagnosis, "multi_match", False)),
        )

        logger.info(
            f"Generated {duration_weeks}-week plan: "
            f"{len(all_drill_ids)} unique drills, "
            f"addresses {len(top_diagnoses)} diagnoses"
        )
        return plan

    def _build_week(
        self,
        week_num: int,
        diagnoses: list[Diagnosis],
        training_days: int,
        language: str,
        use_morphocycle: bool = False,
    ) -> TrainingWeek:
        """Build a single training week."""
        if not diagnoses:
            return self._generic_week(week_num, language)

        primary = diagnoses[0]
        secondary = diagnoses[1] if len(diagnoses) > 1 else None
        tertiary = diagnoses[2] if len(diagnoses) > 2 else None

        if week_num == 1:
            theme = "Foundation" if language == "en" else "التأسيس"
            progression = "intro"
        elif week_num == 2:
            theme = "Building" if language == "en" else "البناء"
            progression = "basic"
        elif week_num == 3:
            theme = "Application" if language == "en" else "التطبيق"
            progression = "intermediate"
        else:
            theme = "Mastery" if language == "en" else "الإتقان"
            progression = "advanced"

        focus_en = f"Address {primary.rule_name}"
        focus_ar = f"معالجة: {primary.rule_name_ar}"

        if secondary:
            focus_en += f" + {secondary.rule_name}"
            focus_ar += f" + {secondary.rule_name_ar}"

        sessions = self._build_sessions(
            week_num=week_num,
            primary=primary,
            secondary=secondary,
            tertiary=tertiary,
            training_days=training_days,
            progression=progression,
            language=language,
            use_morphocycle=use_morphocycle,
        )

        expected = self._build_expected_improvements(primary, secondary, tertiary, language)

        re_test = (
            f"Re-analyze with Kawkab AI: check if '{primary.rule_name}' "
            f"has improved (target: {primary.confidence * 0.7:.0%} reduction in issue severity)"
            if language == "en"
            else f"أعد التحليل بكوكب AI: تحقق من تحسن '{primary.rule_name_ar}' "
            f"(الهدف: تقليل {primary.confidence * 0.7:.0%} في شدة المشكلة)"
        )

        return TrainingWeek(
            week_number=week_num,
            theme=theme,
            primary_focus=focus_en if language == "en" else focus_ar,
            secondary_focus="",
            sessions=sessions,
            expected_improvements=expected,
            re_test_focus=re_test,
        )

    def _build_sessions(
        self,
        week_num: int,
        primary: Diagnosis,
        secondary: Diagnosis | None,
        tertiary: Diagnosis | None,
        training_days: int,
        progression: str,
        language: str,
        use_morphocycle: bool = False,
    ) -> list[DrillSession]:
        """Build the sessions for a week."""
        if use_morphocycle:
            days = [m[f"label_{language}"] for m in self.MORPHOCYCLE]
            md_offsets = [m["md_offset"] for m in self.MORPHOCYCLE]
            md_intensities = [m["intensity"] for m in self.MORPHOCYCLE]
        else:
            days = self.WEEKLY_SCHEDULE[language][:training_days]
            md_offsets = [""] * len(days)
            md_intensities = [""] * len(days)
        sessions = []

        primary_drills = primary.recommended_drills[:2] if primary.recommended_drills else []
        secondary_drills = secondary.recommended_drills[:1] if secondary else []

        session_intensity = {
            "intro": "low",
            "basic": "medium",
            "intermediate": "high",
            "advanced": "very_high",
        }[progression]

        for i, day in enumerate(days):
            if i == 0:
                drills = primary_drills[:1]
                focus = f"Week {week_num} - Primary"
            elif i == 1:
                drills = primary_drills[1:2] + secondary_drills[:1]
                focus = f"Week {week_num} - Primary + Secondary"
            else:
                drills = primary_drills[:1] + secondary_drills[:1]
                focus = f"Week {week_num} - Review"

            if not drills:
                drills = primary_drills[:1] if primary_drills else []

            total_duration = 0
            for d_id in drills:
                drill = self.kb.get_drill(d_id)
                if drill:
                    total_duration += drill.duration_min

            sessions.append(
                DrillSession(
                    week=week_num,
                    day=day,
                    focus=focus,
                    drills=drills,
                    total_duration_min=total_duration,
                    intensity=md_intensities[i] or session_intensity,
                    md_offset=md_offsets[i],
                )
            )

        return sessions

    def _build_weekly_schedule(self, training_days: int, language: str) -> dict[str, list[str]]:
        """Build the weekly schedule structure."""
        days = self.WEEKLY_SCHEDULE[language][:training_days]
        schedule = {}
        for day in days:
            schedule[day] = ["60-90 min session"]
        return schedule

    def build_morphocycle_schedule(self, language: str = "en") -> dict[str, list[str]]:
        """The MD-anchored weekly structure (for UI display alongside the plan)."""
        return {m[f"label_{language}"]: ["session"] for m in self.MORPHOCYCLE}

    def _build_expected_improvements(
        self,
        primary: Diagnosis,
        secondary: Diagnosis | None,
        tertiary: Diagnosis | None,
        language: str,
    ) -> list[str]:
        """Build the expected improvements for this week."""
        improvements = []
        for d in [primary, secondary, tertiary]:
            if d is None:
                continue
            if language == "ar":
                improvements.append(f"تحسن بنسبة 30-40% في '{d.rule_name_ar}'")
            else:
                improvements.append(f"30-40% improvement in '{d.rule_name}'")
        if not improvements:
            improvements = [
                "Maintain current form" if language == "en" else "الحفاظ على المستوى الحالي"
            ]
        return improvements

    def _build_overall_improvement(self, diagnoses: list[Diagnosis], language: str) -> str:
        """Build the overall expected improvement message."""
        if not diagnoses:
            return ""
        if language == "ar":
            top = diagnoses[0]
            return (
                f"بعد 4 أسابيع من التدريب، نتوقع تحسناً كبيراً في '{top.rule_name_ar}' "
                f"والمشكلات الأخرى المحددة. أعد تحليل المباراة لتقييم التقدم."
            )
        top = diagnoses[0]
        return (
            f"After 4 weeks of this plan, we expect significant improvement in "
            f"'{top.rule_name}' and other identified issues. "
            f"Re-analyze the match to measure progress."
        )

    def _generic_week(self, week_num: int, language: str) -> TrainingWeek:
        """Build a generic week when no diagnoses are available."""
        theme = ["Foundation", "Building", "Application", "Mastery"][week_num - 1]
        theme_ar = ["التأسيس", "البناء", "التطبيق", "الإتقان"][week_num - 1]
        return TrainingWeek(
            week_number=week_num,
            theme=theme if language == "en" else theme_ar,
            primary_focus="General fitness and possession",
            secondary_focus="",
            sessions=[],
            expected_improvements=[],
            re_test_focus="Re-analyze after week 4"
            if language == "en"
            else "أعد التحليل بعد الأسبوع 4",
        )

    def export_to_dict(self, plan: TrainingPlan) -> dict[str, Any]:
        """Export a plan to a JSON-serializable dict."""
        return {
            "plan_id": plan.plan_id,
            "match_id": plan.match_id,
            "created_at": plan.created_at,
            "duration_weeks": plan.duration_weeks,
            "total_drills": plan.total_drills,
            "priority_addressed": plan.priority_addressed,
            "expected_overall_improvement": plan.expected_overall_improvement,
            "weekly_schedule": plan.weekly_schedule,
            "weeks": [
                {
                    "week_number": w.week_number,
                    "theme": w.theme,
                    "primary_focus": w.primary_focus,
                    "secondary_focus": w.secondary_focus,
                    "expected_improvements": w.expected_improvements,
                    "re_test_focus": w.re_test_focus,
                    "sessions": [
                        {
                            "week": s.week,
                            "day": s.day,
                            "focus": s.focus,
                            "drills": s.drills,
                            "total_duration_min": s.total_duration_min,
                            "intensity": s.intensity,
                            "md_offset": s.md_offset,
                        }
                        for s in w.sessions
                    ],
                }
                for w in plan.weeks
            ],
            "unresolved_drills": plan.unresolved_drills,
            "drills_resolved": not plan.unresolved_drills,
            "game_model_principles": plan.game_model_principles,
            "game_model_loaded": plan.game_model_loaded,
            "based_on_match_ids": plan.based_on_match_ids,
            "multi_match": plan.multi_match,
        }
