"""Smoke test for LLM service - verifies Ollama integration."""

import asyncio
import sys

from kawkab.core.logging import setup_logging
from kawkab.services.llm_service import LLMConfig, LLMService

setup_logging(debug=False)


async def main() -> int:
    print("=" * 60)
    print("Kawkab AI - LLM Service Smoke Test")
    print("=" * 60)
    print()
    config = LLMConfig(
        provider="ollama",
        ollama_model="qwen3:14b",
        ollama_base_url="http://localhost:11434",
    )
    llm = LLMService(config)
    print(f"Configured {len(llm.providers)} provider(s)")
    for p in llm.providers:
        available = await p.is_available()
        print(f"  - {p.__class__.__name__}: {'available' if available else 'unavailable'}")
    print()
    print("Testing generation (in English)...")
    system = "You are a friendly football coach. Keep answers under 30 words."
    prompt = "What's the most important thing in defending a corner kick?"
    try:
        result = await llm.generate(prompt, system)
        print(f"[OK] English response: {result.strip()[:120]}")
    except Exception as e:
        print(f"[FAIL] English test failed: {e}")
        return 1
    print()
    print("Testing generation (in Arabic)...")
    system = "أنت مساعد مدرب كرة قدم. أبقِ الإجابات أقل من 30 كلمة."
    prompt = "ما أهم شيء في الدفاع عن ركلة ركنية؟"
    try:
        result = await llm.generate(prompt, system)
        print(f"[OK] Arabic response: {result.strip()[:120]}")
    except Exception as e:
        print(f"[FAIL] Arabic test failed: {e}")
        return 1
    print()
    print("Testing coach report generation...")
    summary = """Match: Team A vs Team B
Possession: Team A 58%, Team B 42%
Shots: Team A 12 (5 on target), Team B 8 (3 on target)
Goals: Team A 2, Team B 1
Key events: 5 goals conceded from left channel. Player 7 (LB) takes 2.4s to recover (team avg 1.2s)."""
    try:
        report = await llm.generate_coach_report(summary, language="en")
        print(f"[OK] Coach report ({len(report)} chars):")
        print("---")
        print(report.strip()[:400])
        print("---")
    except Exception as e:
        print(f"[FAIL] Coach report failed: {e}")
        return 1
    print()
    print("[OK] LLM service fully working!")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
