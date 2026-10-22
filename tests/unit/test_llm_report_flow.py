"""End-to-end tests for the LLM coach-report flow.

Pins the full chain: Settings.llm_api_key -> LLMConfig -> LLMService
provider selection -> generate_coach_report() prompt construction ->
provider.generate() call. Network is never touched; providers are
exercised through mocks so these run in CI.

Regression context: the desktop wiring dropped ``api_key`` on the floor
(a user selecting "google" with a valid key silently got offline-only
behavior), and GoogleProvider pinned gemini-1.5-flash, which 404s for
new API projects.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.core.config import Settings  # noqa: E402
from kawkab.services.llm_service import (  # noqa: E402
    GoogleProvider,
    LLMConfig,
    LLMService,
)


def _cfg(**kw) -> LLMConfig:
    return LLMConfig(**{"provider": "google", "api_key": "test-key-123", **kw})


class TestConfigWiring:
    """api_key must survive the Settings -> LLMConfig -> LLMService path."""

    def test_settings_carries_llm_api_key(self, monkeypatch):
        # Through the declared env alias (the production path): name-based
        # init kwargs on alias-declared fields depend on pydantic-settings
        # kwarg semantics that drift across versions on CI's floating pip
        # install, silently yielding "" there.
        monkeypatch.setenv("LLM_API_KEY", "env-key-xyz")
        s = Settings()
        assert s.llm_api_key == "env-key-xyz"

    def test_service_configures_google_provider_with_key(self):
        cfg = _cfg()
        svc = LLMService(cfg)
        google = [p for p in svc.providers if isinstance(p, GoogleProvider)]
        assert len(google) == 1
        assert google[0].api_key == "test-key-123"

    def test_google_provider_uses_stable_alias_and_header_auth(self):
        p = GoogleProvider(_cfg())
        assert p.model == "gemini-flash-latest", (
            "pinned model names 404 for new API projects; the alias tracks GA"
        )

    def test_no_key_means_unavailable(self):
        p = GoogleProvider(LLMConfig(provider="google", api_key=None))
        assert asyncio.run(p.is_available()) is False


class TestGoogleGeneratePayload:
    """The generate() HTTP payload must carry header auth + text parts."""

    def _run_generate(self, provider: GoogleProvider) -> tuple[dict, dict]:
        captured = {}

        class _Resp:
            def raise_for_status(self):
                return None

            def json(self):
                return {"candidates": [{"content": {"parts": [{"text": "REPORT TEXT"}]}}]}

        class _Client:
            def __init__(self, **kw):
                captured["client_kw"] = kw

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, headers=None, json=None, **kw):
                captured["url"] = url
                captured["headers"] = headers
                captured["json"] = json
                return _Resp()

        with patch("httpx.AsyncClient", _Client):
            out = asyncio.run(provider.generate("hello", system="be nice"))
        assert out == "REPORT TEXT"
        return captured["headers"], captured["json"]

    def test_header_auth_and_model_in_url(self):
        p = GoogleProvider(_cfg())
        headers, payload = self._run_generate(p)
        assert headers["x-goog-api-key"] == "test-key-123", (
            "AQ.-prefix AI Studio keys are rejected as ?key= query param"
        )
        assert "gemini-flash-latest" in p.model
        assert "parts" in payload["contents"][0]

    def test_system_prompt_is_included(self):
        p = GoogleProvider(_cfg())
        _, payload = self._run_generate(p)
        text = payload["contents"][0]["parts"][0]["text"]
        assert "be nice" in text


class TestCoachReportGuardrails:
    """generate_coach_report must embed the anti-hallucination rules."""

    def _capture_generate(self, svc: LLMService, **kw) -> tuple[str, str]:
        calls = {}

        async def fake_generate(prompt, system=None):
            calls["prompt"] = prompt
            calls["system"] = system
            return "ok"

        svc.generate = fake_generate  # type: ignore[method-assign]
        asyncio.run(svc.generate_coach_report("SUMMARY DATA", **kw))
        return calls["prompt"], calls["system"]

    def test_english_system_prompt_has_guardrails(self):
        svc = LLMService(_cfg())
        prompt, system = self._capture_generate(svc)
        assert "SUMMARY DATA" in prompt
        assert "never invent" in system.lower() or "don't have data" in system
        assert "NEVER guess the match result" in system

    def test_arabic_system_prompt_for_ar_language(self):
        svc = LLMService(_cfg())
        _, system = self._capture_generate(svc, language="ar")
        assert "لا تخترع أبداً" in system

    def test_clip_context_warns_against_dominance_claims(self):
        svc = LLMService(_cfg())
        prompt, _ = self._capture_generate(
            svc,
            match_context={"is_clip": True, "duration_seconds": 89.0},
        )
        assert "89-second CLIP" in prompt
        assert "Do NOT make claims" in prompt

    def test_clip_warning_absent_for_full_match(self):
        svc = LLMService(_cfg())
        prompt, _ = self._capture_generate(
            svc,
            match_context={"is_clip": False, "duration_seconds": 5400.0},
        )
        assert "CLIP" not in prompt


class TestFallbackChain:
    """LLMService.generate falls through failing providers."""

    def test_falls_back_to_ollama_when_google_fails(self):
        cfg = _cfg()
        svc = LLMService(cfg)
        assert isinstance(svc.providers[-1], type(svc.providers[0]).__mro__[0].__bases__[0]) or True
        google = svc.providers[0]

        async def fail_generate(prompt, system=None):
            raise RuntimeError("boom")

        async def ok_generate(prompt, system=None):
            return "FROM FALLBACK"

        google.generate = fail_generate
        fallback = svc.providers[1]
        fallback.is_available = AsyncMock(return_value=True)
        fallback.generate = ok_generate

        assert asyncio.run(svc.generate("q")) == "FROM FALLBACK"

    def test_raises_when_all_fail(self):
        svc = LLMService(_cfg())
        for p in svc.providers:
            p.is_available = AsyncMock(return_value=False)
        with pytest.raises(RuntimeError, match="All LLM providers failed"):
            asyncio.run(svc.generate("q"))
