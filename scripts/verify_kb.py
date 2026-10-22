"""Verify knowledge base content."""

import asyncio
import os

os.environ["PYTHONIOENCODING"] = "utf-8"

from kawkab.services.knowledge_service import KnowledgeService


async def main() -> int:
    kb = KnowledgeService()
    await kb.initialize()
    print(f"Rules: {kb.stats['rules']}")
    print(f"Drills: {kb.stats['drills']}")
    print()
    print("Tactical rules:")
    for rule in kb.get_all_rules():
        print(f"  - [{rule.category:12s}] {rule.rule_id}: {rule.names.get('en', '')[:60]}")
    print()
    print("Drills:")
    for drill in kb.get_all_drills():
        print(f"  - [{drill.category:18s}] {drill.drill_id}: {drill.name[:50]}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
