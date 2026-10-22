"""Quick LLM test - just one English + one Arabic."""

import asyncio
import os
import sys

os.environ["PYTHONIOENCODING"] = "utf-8"

from kawkab.services.llm_service import LLMConfig, LLMService


async def main() -> int:
    config = LLMConfig(
        provider="ollama",
        ollama_model="qwen3:14b",
        ollama_base_url="http://localhost:11434",
        max_tokens=150,
    )
    llm = LLMService(config)
    print("Test 1: English")
    en = await llm.generate(
        "Reply in one sentence: what is xG?",
        "You are a football coach. Keep it under 20 words.",
    )
    print(f"  EN: {en.strip()[:100]}")
    print()
    print("Test 2: Arabic")
    ar = await llm.generate(
        "Reply in one sentence in Arabic: what is xG?",
        "انت مدرب كرة قدم. اجب بالعربية في اقل من 20 كلمة.",
    )
    print(f"  AR: {ar.strip()[:100]}")
    print()
    print("[OK] All tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
