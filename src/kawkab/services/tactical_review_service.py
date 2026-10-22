"""Tactical review service using the local LLM.

Generates structured tactical analysis reports from match statistics,
formations, and events.  Runs via Ollama (local) and supports both
English and Arabic output.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

_TACTICAL_SYSTEM_PROMPT = """You are an expert football tactical analyst.
Review the match data below and provide concise, actionable tactical feedback.
Focus on:
1. Formation analysis and adjustments
2. Attacking patterns and threats
3. Defensive organization and vulnerabilities
4. Transition moments (attack ↔ defense)
5. Set-piece strengths and weaknesses
6. Key player performance insights
7. Momentum shifts and critical phases

Keep each section to 2-3 sentences. Be specific and reference data."""

_AR_SYSTEM_PROMPT = """أنت محلل تكتيكي لكرة القدم. راجع بيانات المباراة التالية وقدم تحليلاً تكتيكياً موجزاً وقابلاً للتنفيذ.
ركز على:
1. تحليل التشكيلات والتعديلات
2. أنماط الهجوم والتهديدات
3. التنظيم الدفاعي ونقاط الضعف
4. لحظات الانتقال (هجوم ← دفاع)
5. نقاط القوة والضعف في الكرات الثابتة
6. أداء اللاعبين الرئيسيين
7. تحولات الزخم والمراحل الحرجة

اجعل كل قسم في 2-3 جمل. كن محدداً واستخدم البيانات."""


@dataclass
class TacticalReview:
    """Structured tactical analysis output."""

    formation_analysis: str = ""
    attacking_patterns: str = ""
    defensive_organization: str = ""
    transitions: str = ""
    set_pieces: str = ""
    key_players: str = ""
    momentum_shifts: str = ""
    summary: str = ""
    language: str = "en"

    def to_dict(self) -> dict[str, str]:
        return {k: v for k, v in asdict(self).items() if isinstance(v, str)}


class TacticalReviewService:
    """Provides AI-powered tactical analysis using the local LLM."""

    def __init__(self, llm_service: Any | None = None) -> None:
        self._llm = llm_service

    # ── Public API ─────────────────────────────────────────────────

    async def review(
        self,
        match_stats: dict[str, Any],
        language: str = "en",
    ) -> TacticalReview:
        """Generate a full tactical review from match data.

        Args:
            match_stats: Dictionary with keys like
                ``home_team``, ``away_team``, ``events``, ``formations``,
                ``possession``, ``xg``, ``set_pieces``, etc.
            language: ``"en"`` or ``"ar"``

        Returns:
            A :class:`TacticalReview` with per-section analysis.
        """
        if self._llm is None:
            return TacticalReview(summary="LLM service not available", language=language)

        prompt = self._build_prompt(match_stats, language)
        system = _AR_SYSTEM_PROMPT if language == "ar" else _TACTICAL_SYSTEM_PROMPT

        try:
            raw = await self._llm.generate(prompt=prompt, system=system)
            return self._parse_response(raw, language)
        except Exception as e:
            logger.error("Tactical review generation failed: %s", e)
            return TacticalReview(summary=f"Failed to generate review: {e}", language=language)

    async def review_formation(
        self,
        formation_home: str,
        formation_away: str,
        possession_pct_home: float,
        language: str = "en",
    ) -> str:
        """Quick formation-versus-formation tactical note."""
        if self._llm is None:
            return ""
        prompt = (
            f"Analyze the tactical matchup: {formation_home} vs {formation_away}. "
            f"Possession: home {possession_pct_home:.0f}% vs away {100 - possession_pct_home:.0f}%. "
            "What are the key tactical advantages and vulnerabilities for each side? "
            "Be specific in 3-4 sentences."
        )
        try:
            result = await self._llm.generate(prompt=prompt)
            return str(result)
        except Exception as e:
            logger.error("Formation review failed: %s", e)
            return ""

    # ── Internals ──────────────────────────────────────────────────

    def _build_prompt(self, stats: dict[str, Any], language: str) -> str:
        sections: list[str] = []
        lang_tag = " (Arabic)" if language == "ar" else ""

        sections.append(f"## Match Data{lang_tag}")
        sections.append(json.dumps(stats, indent=2, ensure_ascii=False))

        sections.append(
            "\nBased on the data above, provide a structured tactical review "
            "covering formation, attacking patterns, defensive organization, "
            "transitions, set pieces, key players, and momentum shifts."
        )
        return "\n".join(sections)

    def _parse_response(self, raw: str, language: str) -> TacticalReview:
        review = TacticalReview(language=language)

        sections = {
            "formation": "formation_analysis",
            "attack": "attacking_patterns",
            "defensive": "defensive_organization",
            "transition": "transitions",
            "set piece": "set_pieces",
            "key player": "key_players",
            "momentum": "momentum_shifts",
            "summary": "summary",
        }

        current_field: str | None = None
        current_lines: list[str] = []

        for line in raw.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            # Only treat as section header if line ends with colon AND is short
            is_header = stripped.endswith(":") and len(stripped) < 60
            if is_header:
                line_lower = stripped.lower().rstrip(":")
                matched = False
                for keyword, field in sections.items():
                    if keyword in line_lower:
                        if current_field and current_lines:
                            setattr(
                                review,
                                current_field,
                                " ".join(current_lines).strip(),
                            )
                        current_field = field
                        current_lines = []
                        matched = True
                        break
                if matched:
                    continue
            if current_field:
                current_lines.append(stripped)

        if current_field and current_lines:
            setattr(review, current_field, " ".join(current_lines).strip())

        # Capture remaining content after last header
        if current_field and current_lines:
            setattr(review, current_field, " ".join(current_lines).strip())

        if not any(getattr(review, f) for f in sections.values()):
            review.summary = raw.strip()

        return review
