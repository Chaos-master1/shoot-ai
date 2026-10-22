"""Tests for TacticalReviewService."""

from __future__ import annotations

import pytest
from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()
_trs = load_service_module("trs_test", "tactical_review_service.py")

TacticalReview = _trs.TacticalReview
TacticalReviewService = _trs.TacticalReviewService


class DummyLLM:
    """Simulates an LLM that returns structured tactical text."""

    async def generate(self, prompt: str, system: str | None = None) -> str:
        return (
            "Formation Analysis:\n"
            "The 4-3-3 formation created width but left gaps in central midfield.\n"
            "Attacking Patterns:\n"
            "Most attacks came through the left flank (62%).\n"
            "Defensive Organization:\n"
            "The defensive line sat deep (38m) but was caught out by through balls.\n"
            "Transitions:\n"
            "Counter-attacks were slow to develop (< 3 passes before turnover).\n"
            "Set Pieces:\n"
            "All 7 corners were短 deliveries; no aerial threat created.\n"
            "Key Players:\n"
            "Player #10 completed 89% of progressive passes.\n"
            "Momentum Shifts:\n"
            "After the 60th minute, home team dropped intensity.\n"
            "Summary:\n"
            "Home needs to address central midfield cover.\n"
        )


class FailingLLM:
    async def generate(self, prompt: str, system: str | None = None) -> str:
        msg = "LLM unavailable"
        raise ConnectionError(msg)


class EmptyLLM:
    async def generate(self, prompt: str, system: str | None = None) -> str:
        return ""


class TestTacticalReview:
    def test_dataclass_defaults(self):
        r = TacticalReview()
        assert r.formation_analysis == ""
        assert r.language == "en"

    def test_dataclass_custom(self):
        r = TacticalReview(
            formation_analysis="4-3-3 is solid",
            summary="Good match",
            language="ar",
        )
        assert r.formation_analysis == "4-3-3 is solid"
        assert r.language == "ar"

    def test_to_dict(self):
        r = TacticalReview(formation_analysis="test", language="en")
        d = r.to_dict()
        assert d["formation_analysis"] == "test"
        assert d["language"] == "en"


class TestTacticalReviewService:
    @pytest.mark.asyncio
    async def test_review_with_dummy_llm(self):
        service = TacticalReviewService(llm_service=DummyLLM())  # type: ignore[arg-type]
        stats = {
            "home_team": {"possession": 58},
            "away_team": {"possession": 42},
            "formations": {"home": "4-3-3", "away": "4-4-2"},
        }
        review = await service.review(stats, language="en")
        assert review.formation_analysis != ""
        assert review.attacking_patterns != ""
        assert review.defensive_organization != ""
        assert review.transitions != ""
        assert review.set_pieces != ""
        assert review.key_players != ""
        assert review.momentum_shifts != ""
        assert review.summary != ""

    @pytest.mark.asyncio
    async def test_review_without_llm(self):
        service = TacticalReviewService(llm_service=None)
        review = await service.review({})
        assert "not available" in review.summary

    @pytest.mark.asyncio
    async def test_review_failing_llm(self):
        service = TacticalReviewService(llm_service=FailingLLM())  # type: ignore[arg-type]
        review = await service.review({})
        assert "Failed" in review.summary

    @pytest.mark.asyncio
    async def test_review_empty_llm_response(self):
        service = TacticalReviewService(llm_service=EmptyLLM())  # type: ignore[arg-type]
        stats = {"home_team": {"possession": 50}}
        review = await service.review(stats)
        assert isinstance(review.summary, str)

    @pytest.mark.asyncio
    async def test_formation_review(self):
        service = TacticalReviewService(llm_service=DummyLLM())  # type: ignore[arg-type]
        result = await service.review_formation("4-3-3", "4-4-2", 58.0)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_formation_review_no_llm(self):
        service = TacticalReviewService(llm_service=None)
        result = await service.review_formation("4-3-3", "4-4-2", 50.0)
        assert result == ""

    @pytest.mark.asyncio
    async def test_arabic_review(self):
        service = TacticalReviewService(llm_service=DummyLLM())  # type: ignore[arg-type]
        review = await service.review({}, language="ar")
        assert review.language == "ar"
