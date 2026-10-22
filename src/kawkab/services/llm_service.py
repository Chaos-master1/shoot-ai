"""LLM service - multi-provider support for report generation.

Supports Ollama (local, default), Groq, Google AI Studio, OpenRouter.
All providers are free-tier, $0/month.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LLMConfig:
    """LLM provider configuration."""

    provider: Literal["ollama", "groq", "google", "openrouter"] = "ollama"
    ollama_model: str = "ministral-3:14b"
    ollama_base_url: str = "http://localhost:11434"
    api_key: str | None = None
    temperature: float = 0.7
    max_tokens: int = 8000
    num_gpu: int = 99  # Number of GPU layers to use (0 = CPU only, 99 = all)


class LLMProvider(ABC):
    """Abstract LLM provider interface."""

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if this provider is available/ready."""
        ...

    def build_match_context(
        self,
        track_data_duration: float,
        is_clip: bool = False,
        final_score: str | None = None,
        goals: list[dict] | None = None,
        cards: list[dict] | None = None,
    ) -> dict:
        """Build context dict for LLM report generation.

        Args:
            track_data_duration: How many seconds of video we analyzed
            is_clip: True if this is a highlight clip, not full match
            final_score: If known, the final score (e.g., "2-1")
            goals: List of detected goals [{time, scorer, team}]
            cards: List of cards [{time, player, type, team}]

        Returns:
            Context dict for generate_coach_report()
        """
        return {
            "duration_seconds": track_data_duration,
            "is_clip": is_clip,
            "full_match_threshold_sec": 1500,
            "final_score": final_score,
            "goals": goals or [],
            "cards": cards or [],
        }

    @abstractmethod
    async def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: User prompt
            system: Optional system message

        Returns:
            Generated text
        """
        ...


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider (free, offline)."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.base_url = config.ollama_base_url
        self.model = config.ollama_model

    async def is_available(self) -> bool:
        """Check if Ollama is running and model is loaded."""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = [m["name"] for m in data.get("models", [])]
                    return any(self.model.split(":")[0] in m for m in models)
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            return False
        return False

    async def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate response via Ollama API with retry on empty responses."""
        import httpx

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
                "num_gpu": self.config.num_gpu,
            },
        }

        if system:
            payload["system"] = system

        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    response = await client.post(f"{self.base_url}/api/generate", json=payload)
                    response.raise_for_status()
                    data = response.json()
                    result = data.get("response", "").strip()
                    logger.info(
                        f"Ollama attempt {attempt + 1}: "
                        f"prompt_len={len(prompt)}, "
                        f"system_len={len(system) if system else 0}, "
                        f"eval_count={data.get('eval_count')}, "
                        f"resp_len={len(result)}, "
                        f"done={data.get('done_reason')}"
                    )
                    if result and len(result) > 10:
                        return result
            except Exception as e:
                logger.warning(f"Ollama call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                import asyncio

                await asyncio.sleep(2 * (attempt + 1))

        return (
            "I was unable to generate a report at this time. "
            "The match analysis data is available, but the AI assistant "
            "did not respond. Please try again."
        )


class GroqProvider(LLMProvider):
    """Groq API provider (free tier: 30 req/min, Llama 3.1 70B)."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.api_key = config.api_key
        self.model = "llama-3.1-70b-versatile"
        self.base_url = "https://api.groq.com/openai/v1"

    async def is_available(self) -> bool:
        return self.api_key is not None

    async def generate(self, prompt: str, system: str | None = None) -> str:
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]


class GoogleProvider(LLMProvider):
    """Google AI Studio provider (free tier: 15 req/min, Gemini 1.5 Flash)."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.api_key = config.api_key
        # "gemini-flash-latest" is Google's stable alias that tracks the
        # newest GA Flash model; pinned names (1.5/2.5-flash) 404 for new
        # API projects ("no longer available to new users").
        self.model = "gemini-flash-latest"
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    async def is_available(self) -> bool:
        return self.api_key is not None

    async def generate(self, prompt: str, system: str | None = None) -> str:
        import httpx

        url = f"{self.base_url}/models/{self.model}:generateContent"
        # Header auth (x-goog-api-key): AQ.-prefix AI Studio keys are
        # rejected when passed as the ?key= query parameter.
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key or ""}

        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
            },
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]


class LLMService:
    """Multi-provider LLM service with automatic fallback."""

    def build_match_context(
        self,
        track_data_duration: float,
        is_clip: bool = False,
        final_score: str | None = None,
        goals: list[dict] | None = None,
        cards: list[dict] | None = None,
    ) -> dict:
        """Build context dict for LLM report generation."""
        return {
            "duration_seconds": track_data_duration,
            "is_clip": is_clip,
            "full_match_threshold_sec": 1500,
            "final_score": final_score,
            "goals": goals or [],
            "cards": cards or [],
        }

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.providers: list[LLMProvider] = []

        if config.provider == "ollama":
            self.providers.append(OllamaProvider(config))
        elif config.provider == "groq":
            self.providers.append(GroqProvider(config))
        elif config.provider == "google":
            self.providers.append(GoogleProvider(config))
        elif config.provider == "openrouter":
            pass

        self.providers.append(OllamaProvider(config))

        logger.info(
            f"LLMService: primary={config.provider}, {len(self.providers)} providers configured"
        )

    async def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate response with automatic provider fallback.

        Tries each provider in order until one succeeds.

        Args:
            prompt: User prompt
            system: Optional system message

        Returns:
            Generated text from the first available provider
        """
        last_error = None

        for provider in self.providers:
            try:
                if not await provider.is_available():
                    logger.debug(f"{provider.__class__.__name__} not available, skipping")
                    continue

                logger.info(f"Using {provider.__class__.__name__} for generation")
                return await provider.generate(prompt, system)
            except Exception as e:
                logger.warning(f"{provider.__class__.__name__} failed: {e}, trying next provider")
                last_error = e
                continue

        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

    async def generate_coach_report(
        self,
        match_analysis_summary: str,
        language: str = "en",
        match_context: dict | None = None,
    ) -> str:
        """Generate a friendly coach report from match analysis.

        Args:
            match_analysis_summary: Summary of the match analysis
            language: "en" or "ar" for English/Arabic
            match_context: Optional context dict with:
                - duration_seconds: actual match duration
                - is_clip: True if this is a clip, not a full match
                - final_score: known score (if available)
                - cards: list of cards (if detected)
                - goals: list of goals (if detected)
        """
        match_context = match_context or {}
        is_clip = match_context.get("is_clip", False)
        duration = match_context.get("duration_seconds", 0)

        if language == "ar":
            system = """أنت مساعد مدرب كرة قدم. تساعد المدربين الهاويين على فهم بيانات مبارياتهم.

قواعد صارمة:
1. استشهد بطوابع زمنية للفيديو لكل ادعاء.
2. إذا لم تكن تملك بيانات لشيء معين، قل "لا أعرف" - لا تخترع أبداً.
3. استخدم لغة بسيطة، كأنك تتحدث مع صديق مدرب.
4. تكلم بالعربية بشكل طبيعي.
5. لا تدّعي يقيناً ليس لديك. استخدم "يبدو أن"، "نعتقد".
6. الهيكل: ملخص → النتائج الرئيسية (مع مقاطع فيديو) → التوصيات → ما لا نعرفه.

قواعد حرجة (مهم جداً):
7. إذا كانت البيانات من مقطع قصير (أقل من 10 دقائق)، اذكر ذلك بوضوح في أول سطر.
8. لا تخمن نتيجة المباراة. إذا لم يكن لديك بيانات النتيجة، قل "النتيجة غير معروفة من البيانات".
9. لا تقل إن فريق "هيمن" أو "سيطر" بناءً على مقطع قصير. قصص المباريات تتغير.
10. إذا كانت البيانات غير كافية لاستنتاج تكتيكي، قل "أحتاج لمزيد من البيانات".

تذكر: مهمتك هي مساعدة المدرب، ليس إقناعه بشيء قد يكون خاطئاً."""
        else:
            system = """You are a friendly football coach assistant. You help amateur coaches understand their match data.

STRICT RULES:
1. Cite video timestamps for EVERY claim.
2. If you don't have data for something, say "I don't know" — never invent.
3. Use simple language, like talking to a friend coach.
4. Speak the coach's selected language naturally.
5. Never claim certainty you don't have. Use "it looks like", "we believe".
6. Structure: TL;DR → Key findings (with clips) → Recommendations → What we don't know.

CRITICAL RULES (very important):
7. If the data is from a SHORT CLIP (under 10 minutes), state this clearly in the first line.
8. NEVER guess the match result. If the score is not in the data, say "Result unknown from data".
9. NEVER claim a team "dominated" or "controlled" based on a short clip. Matches change over 90 minutes.
10. If data is insufficient for a tactical conclusion, say "I need more data".

Remember: your job is to help the coach, not to convince them of something that might be wrong."""

        clip_warning = ""
        if is_clip:
            clip_warning = (
                f"\n\nIMPORTANT CONTEXT: This analysis is from a {duration:.0f}-second CLIP, not a full match. "
                "Match outcomes, dominance, and momentum can completely change over 90 minutes. "
                "Do NOT make claims about who won, who dominated, or final state. "
                "Frame all observations as 'in this clip' or 'from this segment'."
            )
        elif duration < 600:
            clip_warning = (
                f"\n\nCONTEXT: This is from a short match ({duration:.0f}s). "
                "Frame observations as 'in this match' rather than generalizing."
            )

        prompt = f"""Match Analysis Data:
{match_analysis_summary}
{clip_warning}

Please generate a coach-friendly report following the structure above.
Focus on the most important 3-5 findings and 3-5 actionable recommendations.
REMEMBER: If the data is from a short clip, you cannot make claims about
match outcomes or who dominated. Be specific about what the data shows.
"""

        return await self.generate(prompt, system)
