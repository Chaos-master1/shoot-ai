"""Tests for ProviderCache."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from kawkab.services.provider_cache import ProviderCache


@pytest.fixture
def cache():
    return ProviderCache(max_size=10, default_ttl_s=60)


def test_set_and_get(cache):
    cache.set("key1", {"data": 42})
    assert cache.get("key1") == {"data": 42}


def test_get_miss(cache):
    assert cache.get("nonexistent") is None


def test_get_expired(cache):
    cache.set("key", "value", ttl_s=0)
    import time

    time.sleep(0.001)
    assert cache.get("key") is None


def test_invalidate(cache):
    cache.set("key", "value")
    cache.invalidate("key")
    assert cache.get("key") is None


def test_invalidate_prefix(cache):
    cache.set("match:1", "a")
    cache.set("match:2", "b")
    cache.set("player:1", "c")
    assert cache.invalidate_prefix("match:") == 2
    assert cache.get("match:1") is None
    assert cache.get("player:1") == "c"


def test_clear(cache):
    cache.set("a", 1)
    cache.set("b", 2)
    cache.clear()
    assert cache.get("a") is None
    assert cache.stats()["size"] == 0


def test_stats(cache):
    cache.set("a", 1)
    cache.get("a")
    cache.get("b")
    stats = cache.stats()
    assert stats["size"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert 0 < stats["hit_ratio"] < 1


def test_max_size_eviction(cache):
    for i in range(20):
        cache.set(f"key{i}", i)
    assert cache.stats()["size"] == 10
    assert cache.get("key0") is None
    assert cache.get("key19") is not None


def test_build_key(cache):
    k1 = cache.build_key("match", "12345")
    k2 = cache.build_key("match", "12345")
    k3 = cache.build_key("match", "99999")
    assert k1 == k2
    assert k1 != k3
    assert len(k1) == 32


def test_update_rate_limit_and_ttl_bloat(cache):
    cache.update_rate_limit("statsbomb", remaining=5, limit=100, soft_limit_pct=0.2)
    cache.set("key", "val", provider="statsbomb")
    assert cache.get("key") == "val"


@pytest.mark.asyncio
async def test_memoize_decorator():
    cache = ProviderCache()
    mock_fn = AsyncMock(return_value="result")

    decorated = cache.memoize(ttl_s=60)(mock_fn)
    r1 = await decorated("arg1", kw="val1")
    r2 = await decorated("arg1", kw="val1")
    assert r1 == r2 == "result"
    assert mock_fn.await_count == 1


@pytest.mark.asyncio
async def test_memoize_different_args_different_cache():
    cache = ProviderCache()
    mock_fn = AsyncMock(side_effect=lambda *a, **kw: f"result_{a}_{kw}")

    decorated = cache.memoize()(mock_fn)
    r1 = await decorated("a")
    r2 = await decorated("b")
    assert r1 != r2
    assert mock_fn.await_count == 2


def test_invalidate_prefix_no_match(cache):
    cache.set("a", 1)
    assert cache.invalidate_prefix("z:") == 0
    assert cache.get("a") == 1


def test_lru_order(cache):
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    cache.get("a")
    cache.set("d", 4)
    cache.set("e", 5)
    cache.set("f", 6)
    cache.set("g", 7)
    cache.set("h", 8)
    cache.set("i", 9)
    cache.set("j", 10)
    cache.set("k", 11)
    assert cache.get("a") is not None
    assert cache.get("b") is None


def test_graceful_on_empty_cache(cache):
    assert cache.stats()["size"] == 0
    assert cache.get("anything") is None
    cache.invalidate("nothing")
    assert cache.invalidate_prefix("x:") == 0


def test_custom_ttl_overrides_default(cache):
    cache.set("short", "val", ttl_s=1)
    cache.set("long", "val", ttl_s=3600)
    time.sleep(1.001)
    assert cache.get("short") is None
    assert cache.get("long") == "val"
