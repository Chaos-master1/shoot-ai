"""Tests for ProviderHealthService."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from kawkab.services.provider_health_service import (
    ProviderCallRecord,
    ProviderHealthService,
    ProviderStatus,
)


@pytest.fixture
def health_service():
    return ProviderHealthService(check_interval_s=0)


@pytest.mark.asyncio
async def test_register_and_check(health_service):
    mock_check = AsyncMock(
        return_value={
            "available": True,
            "rate_limit_remaining": 50,
            "rate_limit_reset": "2026-07-14T00:00:00",
        }
    )
    health_service.register("statsbomb", mock_check)
    status = await health_service.check_provider("statsbomb")
    assert status.available is True
    assert status.rate_limit_remaining == 50
    assert status.name == "statsbomb"
    assert status.health_score == 1.0
    mock_check.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_returns_unavailable_when_not_registered(health_service):
    status = await health_service.check_provider("nonexistent")
    assert status.available is False


@pytest.mark.asyncio
async def test_check_handles_exception(health_service):
    mock_check = AsyncMock(side_effect=RuntimeError("API down"))
    health_service.register("apifootball", mock_check)
    status = await health_service.check_provider("apifootball")
    assert status.available is False
    assert status.consecutive_failures == 1
    assert status.error_count == 1
    assert status.health_score < 1.0


@pytest.mark.asyncio
async def test_check_all(health_service):
    health_service.register("a", AsyncMock(return_value={"available": True}))
    health_service.register("b", AsyncMock(return_value={"available": False}))
    results = await health_service.check_all()
    assert len(results) == 2
    names = {r.name for r in results}
    assert names == {"a", "b"}


def test_record_call_success(health_service):
    health_service.register("statsbomb", lambda: None)
    health_service.record_call("statsbomb", "get_match_events", 120.0, True)
    s = health_service.get_status("statsbomb")
    assert s is not None
    assert s.consecutive_failures == 0


def test_record_call_failure(health_service):
    health_service.register("statsbomb", lambda: None)
    health_service.record_call("statsbomb", "get_match", 200.0, False, 500)
    s = health_service.get_status("statsbomb")
    assert s.consecutive_failures == 1
    assert s.error_count == 1
    assert s.health_score < 1.0


def test_record_call_multiple_failures_drops_health(health_service):
    health_service.register("statsbomb", lambda: None)
    for _ in range(5):
        health_service.record_call("statsbomb", "get_match", 100.0, False, 500)
    s = health_service.get_status("statsbomb")
    assert s.consecutive_failures == 5
    assert s.health_score < 1.0


def test_get_all_statuses(health_service):
    health_service.register("a", lambda: None)
    health_service.register("b", lambda: None)
    assert len(health_service.get_all_statuses()) == 2


def test_get_recent_calls(health_service):
    health_service.register("statsbomb", lambda: None)
    for i in range(5):
        health_service.record_call("statsbomb", f"method_{i}", float(i * 10), i % 2 == 0)
    calls = health_service.get_recent_calls(n=3)
    assert len(calls) == 3
    assert calls[-1]["method"] == "method_4"


def test_unregister(health_service):
    health_service.register("statsbomb", lambda: None)
    health_service.unregister("statsbomb")
    assert health_service.get_status("statsbomb") is None
    assert len(health_service.get_all_statuses()) == 0


def test_get_summary(health_service):
    health_service.register("a", lambda: None)
    health_service.register("b", lambda: None)
    health_service.record_call("a", "x", 10.0, True)
    summary = health_service.get_summary()
    assert summary["total_providers"] == 2
    assert summary["total_calls_last_1000"] == 1
    assert "a" in summary["daily_calls"]


def test_health_score_penalty_for_slow_response(health_service):
    mock_check = AsyncMock(return_value={"available": True})
    health_service.register("slow", mock_check)

    s = health_service._statuses["slow"]
    s.response_time_ms = 6000
    s.health_score = health_service._compute_health(s)
    assert s.health_score < 1.0
    assert s.health_score >= 0.5


def test_health_score_minimum_zero(health_service):
    health_service.register("bad", lambda: None)
    s = health_service._statuses["bad"]
    s.consecutive_failures = 20
    s.error_count = 50
    s.response_time_ms = 10000
    s.available = False
    s.health_score = health_service._compute_health(s)
    assert s.health_score == 0.0


@pytest.mark.asyncio
async def test_check_respects_interval(health_service):
    health_service._check_interval = 3600
    mock_check = AsyncMock(return_value={"available": True})
    health_service.register("cached", mock_check)
    await health_service.check_provider("cached")
    await health_service.check_provider("cached")
    assert mock_check.await_count == 1


def test_daily_counts_reset():
    hs = ProviderHealthService()
    hs.register("statsbomb", lambda: None)
    hs.record_call("statsbomb", "x", 10.0, True)
    assert hs._daily_counts["statsbomb"] == 1
    hs._last_daily_reset = "2026-01-01"
    hs._reset_daily_if_needed()
    assert hs._daily_counts.get("statsbomb", 0) == 0


def test_provider_status_to_dict():
    s = ProviderStatus(
        name="test",
        available=True,
        last_check="2026-07-13T12:00:00",
        response_time_ms=150.0,
        rate_limit_remaining=42,
        rate_limit_reset="2026-07-14",
        daily_calls=10,
        error_count=2,
        consecutive_failures=1,
        health_score=0.85,
    )
    d = s.to_dict()
    assert d["name"] == "test"
    assert d["available"] is True
    assert d["response_time_ms"] == 150.0
    assert d["health_score"] == 0.85


def test_call_record_dataclass():
    r = ProviderCallRecord(
        provider="statsbomb",
        method="get_match",
        duration_ms=100.0,
        success=True,
        status_code=200,
        timestamp=datetime.now(UTC).isoformat(),
    )
    assert r.provider == "statsbomb"
    assert r.success is True
    assert r.status_code == 200
