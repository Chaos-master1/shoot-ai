"""Shared base class for bridge handler classes.

Every handler class historically copy-pasted the same __init__ and the
same _check_rate_limit helper. Two problems with that:

    1. The copies drifted (three of them called a nonexistent
       RateLimiter.check() -- a bug that could not exist with one copy).
    2. ``rate_limiter=None`` meant "no limiting at all", which is what
       let tests (and every direct construction) silently run without
       the production limiter. The base class makes a real RateLimiter
       the default: None now means "construct one", never "disable".

Handlers needing custom service accessors keep defining them; the base
only owns the wiring that used to be duplicated.
"""

from __future__ import annotations

from typing import Any

from kawkab.core.security import RateLimiter


class BridgeHandlerBase:
    """Common wiring for bridge handler classes."""

    def __init__(self, bridge, services, rate_limiter=None):
        self._bridge = bridge
        self._services = services if services is not None else {}
        # None means "make one", not "no limiting": a handler constructed
        # without an explicit limiter must still be rate limited, so tests
        # cannot accidentally exercise a limiter-less surface.
        self._rate_limiter = rate_limiter if rate_limiter is not None else RateLimiter()

    def _check_rate_limit(self, category: str = "analysis") -> None:
        if self._rate_limiter.acquire(category):
            return
        raise RuntimeError(f"Rate limit exceeded for {category}")

    async def _knowledge(self) -> Any:
        """A KnowledgeService whose rules are actually loaded.

        The Bridge is injected with a bare ``KnowledgeService()`` that
        nothing has ``initialize()``d (app.py constructs it but never
        awaits the loader), so handlers that trusted it silently saw an
        empty knowledge base — rules ``not found`` that exist on disk.
        This helper initializes the injected instance once per process;
        ``initialize()`` is idempotent, and a later engine fix that
        initializes at startup makes this a no-op.
        """
        kb = self._services.get("knowledge_service")
        if kb is None:
            from kawkab.services.knowledge_service import KnowledgeService

            kb = KnowledgeService()
            self._services["knowledge_service"] = kb
        if not getattr(kb, "_rules", None):
            await kb.initialize()
        return kb
