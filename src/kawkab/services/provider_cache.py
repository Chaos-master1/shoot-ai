"""TTL cache for external provider responses, with rate-limit awareness."""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Any


class ProviderCache:
    """TTL-based cache for external provider API responses.

    - In-memory LRU with configurable max size and per-entry TTL.
    - Rate-limit awareness: can back off when remaining quota is low.
    """

    def __init__(self, max_size: int = 500, default_ttl_s: int = 300):
        self._max_size = max_size
        self._default_ttl = default_ttl_s
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._rate_limit_state: dict[str, _RateLimitState] = {}
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None
        if time.time() > entry.expires_at:
            self._cache.pop(key, None)
            self._misses += 1
            return None
        self._cache.move_to_end(key)
        self._hits += 1
        return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl_s: int | None = None,
        provider: str | None = None,
    ) -> None:
        effective_ttl = self._effective_ttl(ttl_s, provider)
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        self._cache[key] = _CacheEntry(value=value, expires_at=time.time() + effective_ttl)

    def invalidate(self, key: str) -> None:
        self._cache.pop(key, None)

    def invalidate_prefix(self, prefix: str) -> int:
        removed = 0
        for k in list(self._cache.keys()):
            if k.startswith(prefix):
                self._cache.pop(k, None)
                removed += 1
        return removed

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict[str, Any]:
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_ratio": round(self._hits / max(1, self._hits + self._misses), 3),
        }

    def _effective_ttl(self, ttl_s: int | None, provider: str | None) -> int:
        base = ttl_s if ttl_s is not None else self._default_ttl
        if provider and provider in self._rate_limit_state:
            rl = self._rate_limit_state[provider]
            if rl.remaining < rl.soft_limit:
                base = min(int(base * 1.5), base + 300)
        return base

    def update_rate_limit(
        self,
        provider: str,
        remaining: int,
        limit: int = 100,
        soft_limit_pct: float = 0.2,
    ) -> None:
        self._rate_limit_state[provider] = _RateLimitState(
            remaining=remaining,
            limit=limit,
            soft_limit=int(limit * soft_limit_pct),
        )

    def build_key(self, *parts: str) -> str:
        raw = ":".join(parts)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def memoize(self, ttl_s: int | None = None, provider: str | None = None):
        """Decorator that caches async function results by args/kwargs."""

        def decorator(fn):
            import functools

            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                key_parts = [fn.__name__]
                key_parts.extend(str(a) for a in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                key = self.build_key(*key_parts)
                cached = self.get(key)
                if cached is not None:
                    return cached
                result = await fn(*args, **kwargs)
                self.set(key, result, ttl_s=ttl_s, provider=provider)
                return result

            return wrapper

        return decorator


class _CacheEntry:
    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, expires_at: float):
        self.value = value
        self.expires_at = expires_at


class _RateLimitState:
    __slots__ = ("remaining", "limit", "soft_limit")

    def __init__(self, remaining: int, limit: int, soft_limit: int):
        self.remaining = remaining
        self.limit = limit
        self.soft_limit = soft_limit
