"""Handler for provider health/cache bridge methods."""

from __future__ import annotations

import json

from kawkab.core.logging import get_logger
from kawkab.services.provider_cache import ProviderCache
from kawkab.services.provider_health_service import ProviderHealthService
from kawkab.ui.bridge_handlers.base import BridgeHandlerBase

logger = get_logger(__name__)


class ProviderHandler(BridgeHandlerBase):
    def __init__(self, bridge, services, rate_limiter=None):
        super().__init__(bridge, services, rate_limiter)
        self._health = ProviderHealthService(check_interval_s=120)
        self._cache = ProviderCache(max_size=500, default_ttl_s=300)
        self._register_providers()

    def _register_providers(self):
        names = [
            "football_data_service",
            "bzzoiro_service",
            "easy_soccer_service",
            "api_football_service",
            "thesportsdb_service",
            "statsbomb_service",
            "openfootball_service",
        ]
        for name in names:
            svc = self._services.get(name)
            if svc is not None:

                async def _check(svc_ref=svc, svc_name=name):
                    try:
                        if hasattr(svc_ref, "check_status"):
                            result = await svc_ref.check_status()
                        elif hasattr(svc_ref, "check_available"):
                            result = {"available": svc_ref.check_available()}
                        else:
                            result = self._fallback_check(svc_ref, svc_name)
                        return result
                    except Exception:
                        return {"available": False}
                    finally:
                        pass

                self._health.register(name, _check)

    def _fallback_check(self, svc, name: str) -> dict:
        if hasattr(svc, "get_competitions"):
            try:
                import asyncio

                _ = asyncio.run_coroutine_threadsafe(svc.get_competitions(), None)  # type: ignore[arg-type]
                return {"available": True}
            except Exception:
                pass
        return {"available": True}

    # ── Provider Health ──────────────────────────────────────

    async def get_provider_status(self, provider_name: str) -> str:
        name = self._sanitize_name(provider_name)
        try:
            s = await self._health.check_provider(name)
            if s is None:
                return json.dumps({"error": f"Provider '{name}' not registered"})
            return json.dumps(s.to_dict())
        except Exception as e:
            logger.error(f"get_provider_status failed: {e}")
            return json.dumps({"error": str(e)})

    async def get_all_provider_statuses(self) -> str:
        try:
            statuses = await self._health.check_all()
            return json.dumps([s.to_dict() for s in statuses])
        except Exception as e:
            logger.error(f"get_all_provider_statuses failed: {e}")
            return json.dumps({"error": str(e)})

    async def get_provider_summary(self) -> str:
        try:
            summary = self._health.get_summary()
            return json.dumps(summary)
        except Exception as e:
            logger.error(f"get_provider_summary failed: {e}")
            return json.dumps({"error": str(e)})

    async def get_recent_provider_calls(self, n: int = 20) -> str:
        try:
            calls = self._health.get_recent_calls(n)
            return json.dumps(calls)
        except Exception as e:
            logger.error(f"get_recent_provider_calls failed: {e}")
            return json.dumps([])

    async def record_provider_call(
        self,
        provider: str,
        method: str,
        duration_ms: float,
        success: bool,
        status_code: int = 200,
    ) -> str:
        try:
            self._health.record_call(provider, method, duration_ms, success, status_code)
            return json.dumps({"ok": True})
        except Exception as e:
            logger.error(f"record_provider_call failed: {e}")
            return json.dumps({"error": str(e)})

    # ── Provider Cache ───────────────────────────────────────

    async def cache_get(self, key: str) -> str:
        try:
            val = self._cache.get(key)
            if val is None:
                return json.dumps(None)
            return json.dumps(val)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def cache_set(self, key: str, value_json: str, ttl_s: int = -1) -> str:
        try:
            value = json.loads(value_json)
            if ttl_s > 0:
                self._cache.set(key, value, ttl_s=ttl_s)
            else:
                self._cache.set(key, value)
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def cache_invalidate(self, key: str) -> str:
        try:
            self._cache.invalidate(key)
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def cache_invalidate_prefix(self, prefix: str) -> str:
        try:
            removed = self._cache.invalidate_prefix(prefix)
            return json.dumps({"removed": removed})
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def cache_clear(self) -> str:
        try:
            self._cache.clear()
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def cache_stats(self) -> str:
        try:
            return json.dumps(self._cache.stats())
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ── helpers ──────────────────────────────────────────────

    def _sanitize_name(self, name: str) -> str:
        allowed = {
            "football_data_service",
            "bzzoiro_service",
            "easy_soccer_service",
            "api_football_service",
            "thesportsdb_service",
            "statsbomb_service",
            "openfootball_service",
        }
        if name in allowed:
            return name
        return ""
