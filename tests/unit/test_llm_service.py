"""Tests for LLMService — multi-provider LLM report generation."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()

_mod = load_service_module("llm_test", "llm_service.py")

LLMConfig = _mod.LLMConfig
LLMService = _mod.LLMService
OllamaProvider = _mod.OllamaProvider
GroqProvider = _mod.GroqProvider
GoogleProvider = _mod.GoogleProvider


# ===========================================================================
# LLMConfig
# ===========================================================================


class TestLLMConfig:
    def test_defaults(self):
        cfg = LLMConfig()
        assert cfg.provider == "ollama"
        assert cfg.ollama_model == "ministral-3:14b"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 8000

    def test_custom_values(self):
        cfg = LLMConfig(provider="groq", api_key="key123", temperature=0.5)
        assert cfg.provider == "groq"
        assert cfg.api_key == "key123"
        assert cfg.temperature == 0.5


# ===========================================================================
# LLMService
# ===========================================================================


class TestLLMServiceInit:
    def test_ollama_provider_added(self):
        cfg = LLMConfig(provider="ollama")
        svc = LLMService(cfg)
        assert len(svc.providers) >= 1
        assert isinstance(svc.providers[0], OllamaProvider)

    def test_groq_provider_added(self):
        cfg = LLMConfig(provider="groq", api_key="gk-xxx")
        svc = LLMService(cfg)
        assert any(isinstance(p, GroqProvider) for p in svc.providers)

    def test_google_provider_added(self):
        cfg = LLMConfig(provider="google", api_key="ai-xxx")
        svc = LLMService(cfg)
        assert any(isinstance(p, GoogleProvider) for p in svc.providers)

    def test_ollama_always_fallback(self):
        cfg = LLMConfig(provider="groq", api_key="gk-xxx")
        svc = LLMService(cfg)
        assert any(isinstance(p, OllamaProvider) for p in svc.providers)


class TestBuildMatchContext:
    def test_basic_context(self):
        cfg = LLMConfig()
        svc = LLMService(cfg)
        ctx = svc.build_match_context(3600.0)
        assert ctx["duration_seconds"] == 3600.0
        assert ctx["is_clip"] is False
        assert ctx["final_score"] is None

    def test_context_with_clip(self):
        cfg = LLMConfig()
        svc = LLMService(cfg)
        ctx = svc.build_match_context(120.0, is_clip=True, final_score="2-1")
        assert ctx["is_clip"] is True
        assert ctx["final_score"] == "2-1"
        assert ctx["full_match_threshold_sec"] == 1500

    def test_context_with_goals_and_cards(self):
        cfg = LLMConfig()
        svc = LLMService(cfg)
        goals = [{"time": 10, "scorer": "Player A", "team": "home"}]
        cards = [{"time": 45, "player": "Player B", "type": "yellow", "team": "away"}]
        ctx = svc.build_match_context(3600.0, goals=goals, cards=cards)
        assert len(ctx["goals"]) == 1
        assert len(ctx["cards"]) == 1


class TestGenerate:
    @pytest.mark.asyncio
    async def test_generate_success(self):
        cfg = LLMConfig(provider="ollama")
        svc = LLMService(cfg)
        for p in svc.providers:
            p.is_available = AsyncMock(return_value=True)
            p.generate = AsyncMock(return_value="Generated report")
        result = await svc.generate("Test prompt")
        assert result == "Generated report"

    @pytest.mark.asyncio
    async def test_generate_fallback_on_failure(self):
        cfg = LLMConfig(provider="ollama")
        svc = LLMService(cfg)
        for i, p in enumerate(svc.providers):
            p.is_available = AsyncMock(return_value=True)
            if i == 0:
                p.generate = AsyncMock(side_effect=Exception("First failed"))
            else:
                p.generate = AsyncMock(return_value="Fallback report")
        result = await svc.generate("Test prompt")
        assert result == "Fallback report"

    @pytest.mark.asyncio
    async def test_generate_raises_when_all_fail(self):
        cfg = LLMConfig(provider="ollama")
        svc = LLMService(cfg)
        for p in svc.providers:
            p.is_available = AsyncMock(return_value=True)
            p.generate = AsyncMock(side_effect=Exception("Failed"))
        with pytest.raises(RuntimeError, match="All LLM providers failed"):
            await svc.generate("Test prompt")

    @pytest.mark.asyncio
    async def test_generate_skips_unavailable_providers(self):
        cfg = LLMConfig(provider="ollama")
        svc = LLMService(cfg)
        for i, p in enumerate(svc.providers):
            if i == 0:
                p.is_available = AsyncMock(return_value=False)
                p.generate = AsyncMock(return_value="Should not call")
            else:
                p.is_available = AsyncMock(return_value=True)
                p.generate = AsyncMock(return_value="Available report")
        result = await svc.generate("Test")
        assert result == "Available report"


class TestGenerateCoachReport:
    @pytest.mark.asyncio
    async def test_coach_report_english(self):
        cfg = LLMConfig(provider="ollama")
        svc = LLMService(cfg)
        for p in svc.providers:
            p.is_available = AsyncMock(return_value=True)
            p.generate = AsyncMock(return_value="Coach report en")
        result = await svc.generate_coach_report("Match data summary", "en")
        assert result == "Coach report en"

    @pytest.mark.asyncio
    async def test_coach_report_arabic(self):
        cfg = LLMConfig(provider="ollama")
        svc = LLMService(cfg)
        for p in svc.providers:
            p.is_available = AsyncMock(return_value=True)
            p.generate = AsyncMock(return_value="تقرير المدرب")
        result = await svc.generate_coach_report("Match data", "ar")
        assert result == "تقرير المدرب"

    @pytest.mark.asyncio
    async def test_coach_report_with_clip_context(self):
        cfg = LLMConfig(provider="ollama")
        svc = LLMService(cfg)
        for p in svc.providers:
            p.is_available = AsyncMock(return_value=True)
            p.generate = AsyncMock(return_value="Clip report")
        ctx = {"is_clip": True, "duration_seconds": 120}
        _ = await svc.generate_coach_report("Data", match_context=ctx)
        assert "IMPORTANT CONTEXT" in svc.providers[0].generate.call_args[0][0]


class TestOllamaProvider:
    @pytest.mark.asyncio
    async def test_is_available_returns_true(self):
        cfg = LLMConfig()
        prov = OllamaProvider(cfg)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "ministral-3:14b"}]}
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_resp
            result = await prov.is_available()
        assert result is True

    @pytest.mark.asyncio
    async def test_is_available_returns_false_on_error(self):
        cfg = LLMConfig()
        prov = OllamaProvider(cfg)
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = Exception("Conn err")
            result = await prov.is_available()
        assert result is False

    @pytest.mark.asyncio
    async def test_generate_success(self):
        cfg = LLMConfig()
        prov = OllamaProvider(cfg)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "response": "This is the generated response text.",
            "eval_count": 15,
            "done_reason": "stop",
        }
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_resp
            result = await prov.generate("Hi")
        assert result == "This is the generated response text."

    @pytest.mark.asyncio
    async def test_generate_retry_on_empty(self):
        cfg = LLMConfig()
        prov = OllamaProvider(cfg)
        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.json.return_value = {
            "response": "Long enough response text.",
            "eval_count": 10,
            "done_reason": "stop",
        }
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post.return_value = resp_ok
            result = await prov.generate("Hi")
        assert result == "Long enough response text."


class TestGroqProvider:
    @pytest.mark.asyncio
    async def test_is_available_with_key(self):
        cfg = LLMConfig(provider="groq", api_key="gk-test")
        prov = GroqProvider(cfg)
        assert await prov.is_available() is True

    @pytest.mark.asyncio
    async def test_is_available_without_key(self):
        cfg = LLMConfig(provider="groq")
        prov = GroqProvider(cfg)
        assert await prov.is_available() is False

    @pytest.mark.asyncio
    async def test_generate_success(self):
        cfg = LLMConfig(provider="groq", api_key="gk-test")
        prov = GroqProvider(cfg)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": [{"message": {"content": "Groq answer"}}]}
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_resp
            result = await prov.generate("Test")
        assert result == "Groq answer"


class TestGoogleProvider:
    @pytest.mark.asyncio
    async def test_is_available_with_key(self):
        cfg = LLMConfig(provider="google", api_key="ai-test")
        prov = GoogleProvider(cfg)
        assert await prov.is_available() is True

    @pytest.mark.asyncio
    async def test_is_available_without_key(self):
        cfg = LLMConfig(provider="google")
        prov = GoogleProvider(cfg)
        assert await prov.is_available() is False

    @pytest.mark.asyncio
    async def test_generate_success(self):
        cfg = LLMConfig(provider="google", api_key="ai-test")
        prov = GoogleProvider(cfg)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Google answer"}]}}]
        }
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_resp
            result = await prov.generate("Test")
        assert result == "Google answer"
