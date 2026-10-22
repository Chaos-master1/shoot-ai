"""Tests for ProviderHandler bridge module."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.ui.bridge_handlers.bridge_provider import ProviderHandler


@pytest.fixture
def mock_bridge():
    return MagicMock()


@pytest.fixture
def handler(mock_bridge):
    return ProviderHandler(mock_bridge, {})


def make_mock_service(name: str, available: bool = True):
    svc = MagicMock()
    svc.check_status = AsyncMock(return_value={"available": available})
    svc.__class__.__name__ = f"Mock{name.title()}Service"
    return svc


@pytest.mark.asyncio
async def test_get_provider_status_unregistered(handler):
    result = await handler.get_provider_status("nonexistent")
    data = json.loads(result)
    assert data.get("available") is False or "error" in data


@pytest.mark.asyncio
async def test_get_provider_status_registered(handler):
    svc = make_mock_service("football_data_service")
    handler._services = {"football_data_service": svc}
    handler._register_providers()
    result = await handler.get_provider_status("football_data_service")
    data = json.loads(result)
    assert data["name"] == "football_data_service"


@pytest.mark.asyncio
async def test_get_all_provider_statuses(handler):
    svc = make_mock_service("statsbomb_service")
    handler._services = {"statsbomb_service": svc}
    handler._register_providers()
    result = await handler.get_all_provider_statuses()
    data = json.loads(result)
    assert isinstance(data, list)
    names = [s["name"] for s in data]
    assert "statsbomb_service" in names


@pytest.mark.asyncio
async def test_get_provider_summary(handler):
    result = await handler.get_provider_summary()
    data = json.loads(result)
    assert "total_providers" in data
    assert "avg_health_score" in data


@pytest.mark.asyncio
async def test_get_recent_provider_calls(handler):
    handler._health.record_call("test", "method", 10.0, True)
    result = await handler.get_recent_provider_calls(5)
    data = json.loads(result)
    assert len(data) >= 1
    assert data[0]["provider"] == "test"


@pytest.mark.asyncio
async def test_record_provider_call(handler):
    result = await handler.record_provider_call("test", "get_match", 50.0, True, 200)
    data = json.loads(result)
    assert data["ok"] is True


@pytest.mark.asyncio
async def test_cache_get_miss(handler):
    result = await handler.cache_get("missing")
    data = json.loads(result)
    assert data is None


@pytest.mark.asyncio
async def test_cache_set_and_get(handler):
    await handler.cache_set("key1", json.dumps({"data": 42}))
    result = await handler.cache_get("key1")
    data = json.loads(result)
    assert data == {"data": 42}


@pytest.mark.asyncio
async def test_cache_set_with_ttl(handler):
    await handler.cache_set("short", json.dumps("val"), 1)
    result = await handler.cache_get("short")
    assert json.loads(result) == "val"


@pytest.mark.asyncio
async def test_cache_invalidate(handler):
    await handler.cache_set("key", json.dumps("val"))
    await handler.cache_invalidate("key")
    result = await handler.cache_get("key")
    assert json.loads(result) is None


@pytest.mark.asyncio
async def test_cache_invalidate_prefix(handler):
    await handler.cache_set("a:1", json.dumps(1))
    await handler.cache_set("a:2", json.dumps(2))
    await handler.cache_set("b:1", json.dumps(3))
    result = await handler.cache_invalidate_prefix("a:")
    data = json.loads(result)
    assert data["removed"] == 2


@pytest.mark.asyncio
async def test_cache_clear(handler):
    await handler.cache_set("a", json.dumps(1))
    await handler.cache_clear()
    result = await handler.cache_get("a")
    assert json.loads(result) is None


@pytest.mark.asyncio
async def test_cache_stats(handler):
    await handler.cache_set("a", json.dumps(1))
    result = await handler.cache_stats()
    data = json.loads(result)
    assert data["size"] >= 1
    assert "hit_ratio" in data


@pytest.mark.asyncio
async def test_record_provider_call_updates_health(handler):
    await handler.record_provider_call("statsbomb", "x", 100.0, False, 500)
    result = await handler.get_provider_status("statsbomb")
    data = json.loads(result)
    assert data.get("error") is None or data["health_score"] < 1.0


@pytest.mark.asyncio
async def test_fallback_check_no_check_method(handler):
    svc = MagicMock()
    del svc.check_status
    result = handler._fallback_check(svc, "test")
    assert result is not None
