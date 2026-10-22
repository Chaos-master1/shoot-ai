"""Provider Health Service — monitors external provider status, rate limits, and errors."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass
class ProviderStatus:
    name: str = ""
    available: bool = False
    last_check: str = ""
    response_time_ms: float = 0.0
    rate_limit_remaining: int = 0
    rate_limit_reset: str = ""
    daily_calls: int = 0
    error_count: int = 0
    consecutive_failures: int = 0
    health_score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "last_check": self.last_check,
            "response_time_ms": round(self.response_time_ms, 1),
            "rate_limit_remaining": self.rate_limit_remaining,
            "rate_limit_reset": self.rate_limit_reset,
            "daily_calls": self.daily_calls,
            "error_count": self.error_count,
            "consecutive_failures": self.consecutive_failures,
            "health_score": round(self.health_score, 2),
        }


@dataclass
class ProviderCallRecord:
    provider: str = ""
    method: str = ""
    duration_ms: float = 0.0
    success: bool = True
    status_code: int = 200
    timestamp: str = ""


class ProviderHealthService:
    def __init__(self, check_interval_s: int = 300):
        self._check_interval = check_interval_s
        self._statuses: dict[str, ProviderStatus] = {}
        self._call_log: deque[ProviderCallRecord] = deque(maxlen=1000)
        self._daily_counts: dict[str, int] = defaultdict(int)
        self._last_check_time: dict[str, float] = {}
        self._check_fns: dict[str, Callable] = {}
        self._last_daily_reset: str = ""

    def register(self, name: str, check_fn: Callable) -> None:
        self._check_fns[name] = check_fn
        if name not in self._statuses:
            self._statuses[name] = ProviderStatus(name=name)

    def unregister(self, name: str) -> None:
        self._check_fns.pop(name, None)
        self._statuses.pop(name, None)

    def record_call(
        self,
        provider: str,
        method: str,
        duration_ms: float,
        success: bool,
        status_code: int = 200,
    ) -> None:
        self._reset_daily_if_needed()
        self._call_log.append(
            ProviderCallRecord(
                provider=provider,
                method=method,
                duration_ms=duration_ms,
                success=success,
                status_code=status_code,
                timestamp=datetime.now(UTC).isoformat(),
            )
        )
        self._daily_counts[provider] += 1

        if provider in self._statuses:
            s = self._statuses[provider]
            if not success:
                s.error_count += 1
                s.consecutive_failures += 1
            else:
                s.consecutive_failures = 0
            s.health_score = self._compute_health(s)

    def _reset_daily_if_needed(self) -> None:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if today != self._last_daily_reset:
            self._daily_counts.clear()
            self._last_daily_reset = today

    def _compute_health(self, s: ProviderStatus) -> float:
        score = 1.0
        if s.consecutive_failures > 0:
            score -= min(0.5, s.consecutive_failures * 0.1)
        if s.error_count > 10:
            score -= min(0.3, (s.error_count - 10) * 0.02)
        if s.response_time_ms > 5000:
            score -= 0.2
        if not s.available:
            score = max(0.0, score - 0.3)
        return max(0.0, score)

    async def check_provider(self, name: str) -> ProviderStatus:
        now = time.time()
        last = self._last_check_time.get(name, 0)
        if now - last < self._check_interval and name in self._statuses:
            return self._statuses[name]

        check_fn = self._check_fns.get(name)
        if not check_fn:
            s = self._statuses.get(name, ProviderStatus(name=name))
            s.available = False
            return s

        start = time.perf_counter()
        try:
            result = await check_fn()
            duration = (time.perf_counter() - start) * 1000
            s = self._statuses.setdefault(name, ProviderStatus(name=name))
            s.available = result.get("available", True)
            s.last_check = datetime.now(UTC).isoformat()
            s.response_time_ms = duration
            s.rate_limit_remaining = result.get("rate_limit_remaining", s.rate_limit_remaining)
            s.rate_limit_reset = result.get("rate_limit_reset", s.rate_limit_reset)
            s.daily_calls = self._daily_counts.get(name, 0)
            if not s.available:
                s.consecutive_failures += 1
            s.health_score = self._compute_health(s)
        except Exception:
            duration = (time.perf_counter() - start) * 1000
            s = self._statuses.setdefault(name, ProviderStatus(name=name))
            s.available = False
            s.last_check = datetime.now(UTC).isoformat()
            s.response_time_ms = duration
            s.consecutive_failures += 1
            s.error_count += 1
            s.health_score = self._compute_health(s)

        self._last_check_time[name] = now
        return self._statuses[name]

    async def check_all(self) -> list[ProviderStatus]:
        results = []
        for name in list(self._check_fns.keys()):
            s = await self.check_provider(name)
            results.append(s)
        return results

    def get_status(self, name: str) -> ProviderStatus | None:
        return self._statuses.get(name)

    def get_all_statuses(self) -> list[ProviderStatus]:
        return list(self._statuses.values())

    def get_recent_calls(self, n: int = 20) -> list[dict]:
        return [
            {
                "provider": r.provider,
                "method": r.method,
                "duration_ms": round(r.duration_ms, 1),
                "success": r.success,
                "status_code": r.status_code,
                "timestamp": r.timestamp,
            }
            for r in list(self._call_log)[-n:]
        ]

    def get_summary(self) -> dict[str, Any]:
        statuses = self.get_all_statuses()
        available = sum(1 for s in statuses if s.available)
        total = len(statuses)
        avg_health = sum(s.health_score for s in statuses) / max(1, total)
        return {
            "total_providers": total,
            "available": available,
            "unavailable": total - available,
            "avg_health_score": round(avg_health, 2),
            "total_calls_last_1000": len(self._call_log),
            "daily_calls": dict(self._daily_counts),
        }
