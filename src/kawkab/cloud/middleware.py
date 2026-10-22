"""FastAPI middleware for rate limiting, pagination, and security."""

from __future__ import annotations

import os
import time
from collections import defaultdict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Maximum distinct bucket keys to retain before opportunistically pruning
# idle ones. _buckets was previously fully unbounded -- attacker-chosen
# source IPs (or, before the fix below, X-Forwarded-For values once that
# was trusted) could grow it indefinitely as a memory-exhaustion vector.
_MAX_BUCKETS = 10_000
_PRUNE_IDLE_AFTER_S = 3600  # an hour with no requests -> safe to forget


class RateLimitMiddleware:
    """Token-bucket rate limiter per IP + endpoint category.

    Limits: analysis=5/min, export=10/min, search=30/min, general=60/min.
    Set KAWKAB_RATE_LIMIT_DISABLE=1 to disable for tests.
    """

    def __init__(self, app: FastAPI):
        self.app = app
        self.disabled = os.environ.get("KAWKAB_RATE_LIMIT_DISABLE") == "1"
        # Off by default: trusting X-Forwarded-For/X-Real-IP when the app
        # is NOT actually behind a trusted reverse proxy lets any client
        # spoof its rate-limit identity via a plain HTTP header. Set this
        # only when deploying behind the shipped nginx (infrastructure/
        # nginx.conf sets both headers) or an equivalent trusted proxy.
        self.trust_proxy_headers = os.environ.get("KAWKAB_TRUST_PROXY_HEADERS") == "1"
        self._buckets: dict[str, dict] = defaultdict(
            lambda: {"tokens": 60, "last_refill": time.time()}
        )
        self._limits = {
            "/api/v1/matches/": 30,
            "/api/v1/analysis": 5,
            "/api/v1/recruitment": 30,
            "/api/v1/webhooks": 10,
            "/api/v1/streaming": 10,
            "/api/v1/monitoring": 30,
            "/auth/": 10,
            "/sync/": 20,
            "/teams": 20,
        }
        self._default_limit = 60

    def _get_limit_and_category(self, path: str) -> tuple[int, str]:
        for prefix, limit in self._limits.items():
            if path.startswith(prefix):
                return limit, prefix
        return self._default_limit, "__default__"

    def _client_ip(self, request: Request) -> str:
        if self.trust_proxy_headers:
            real_ip = request.headers.get("x-real-ip")
            if real_ip:
                return real_ip.strip()
            forwarded_for = request.headers.get("x-forwarded-for")
            if forwarded_for:
                # First entry is the original client; later ones are
                # intermediate proxies (see nginx's $proxy_add_x_forwarded_for).
                return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _prune_idle_buckets(self, now: float) -> None:
        if len(self._buckets) <= _MAX_BUCKETS:
            return
        stale = [
            key
            for key, bucket in self._buckets.items()
            if now - bucket["last_refill"] > _PRUNE_IDLE_AFTER_S
        ]
        for key in stale:
            del self._buckets[key]

    async def __call__(self, scope, receive, send):
        if self.disabled or scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        client_ip = self._client_ip(request)
        path = request.url.path

        # The bucket key and the limit applied to it must always come from
        # the SAME category lookup. Previously the key was derived from
        # path.split('/')[3] (an unrelated path-segment heuristic) while
        # the limit was derived from a prefix match -- for any path with
        # 3 or fewer segments (/auth/login, /auth/register, /health,
        # /sync/push, /sync/pull, ...) that heuristic fell through to the
        # literal string "general", silently merging traffic to
        # completely different endpoints -- with completely different
        # intended limits -- into one shared bucket. E.g. /auth/login
        # (limit 10/min) and /health (no configured limit -> default
        # 60/min) shared a bucket refilled at whichever limit the
        # *current* request happened to carry, so interleaving /health
        # requests refilled the shared bucket at 6x the rate an attacker
        # brute-forcing /auth/login should have been allowed.
        limit, category = self._get_limit_and_category(path)
        key = f"{client_ip}:{category}"
        now = time.time()
        bucket = self._buckets[key]

        elapsed = now - bucket["last_refill"]
        bucket["tokens"] = min(limit, bucket["tokens"] + elapsed * (limit / 60.0))
        bucket["last_refill"] = now

        self._prune_idle_buckets(now)

        if bucket["tokens"] < 1:
            response = JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded", "retry_after_s": 60},
            )
            response.headers["X-RateLimit-Limit"] = str(limit)
            response.headers["X-RateLimit-Remaining"] = "0"
            response.headers["Retry-After"] = "60"
            await response(scope, receive, send)
            return

        bucket["tokens"] -= 1

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = message.get("headers", [])
                headers.append((b"X-RateLimit-Limit", str(limit).encode()))
                headers.append((b"X-RateLimit-Remaining", str(int(bucket["tokens"])).encode()))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)
