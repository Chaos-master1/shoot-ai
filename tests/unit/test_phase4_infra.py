"""Sprint 4 Infrastructure tests — rate limiter, coordinate validation, timing, secrets."""

import os
import time
import warnings
from unittest.mock import patch

import pytest

from kawkab.core import secrets as secrets_mod
from kawkab.core.coordinate_validator import CoordinateValidator
from kawkab.core.perf_timing import timed
from kawkab.core.security import RateLimiter

# =============================================================================
# 1. Rate limiter integration tests
# =============================================================================


class TestRateLimiterIntegration:
    """Rate limiter allowed/blocked/different categories/reset."""

    def test_allowed_within_limit(self):
        rl = RateLimiter(tokens_per_minute=60)
        # Burst of 5 should all be allowed
        for _ in range(5):
            assert rl.acquire("test_key", cost=1) is True

    def test_blocked_when_exceeded(self):
        rl = RateLimiter(tokens_per_minute=1)
        assert rl.acquire("exceed_key", cost=1) is True
        assert rl.acquire("exceed_key", cost=1) is False

    def test_different_categories_independent(self):
        rl = RateLimiter(tokens_per_minute=1)
        assert rl.acquire("cat_a", cost=1) is True
        assert rl.acquire("cat_a", cost=1) is False
        assert rl.acquire("cat_b", cost=1) is True
        assert rl.acquire("cat_b", cost=1) is False

    def test_cost_accounting(self):
        rl = RateLimiter(tokens_per_minute=5)
        assert rl.acquire("cost_key", cost=3) is True
        assert rl.acquire("cost_key", cost=3) is False
        assert rl.acquire("cost_key", cost=2) is True
        assert rl.acquire("cost_key", cost=1) is False

    def test_configure_override(self):
        rl = RateLimiter(tokens_per_minute=10)
        rl.configure("slow", 2)
        assert rl.acquire("slow:x", cost=1) is True
        assert rl.acquire("slow:x", cost=1) is True
        assert rl.acquire("slow:x", cost=1) is False

    def test_reset_via_separate_key(self):
        rl = RateLimiter(tokens_per_minute=1)
        assert rl.acquire("reset_me", cost=1) is True
        assert rl.acquire("reset_me", cost=1) is False
        # New key = fresh bucket
        assert rl.acquire("fresh_key", cost=1) is True

    def test_default_rate_fallback(self):
        rl = RateLimiter(tokens_per_minute=3)
        rl.configure("analysis", 1)
        # "unknown" prefix falls back to default (3)
        assert rl.acquire("unknown:1", cost=1) is True
        assert rl.acquire("unknown:1", cost=1) is True
        assert rl.acquire("unknown:1", cost=1) is True
        assert rl.acquire("unknown:1", cost=1) is False

    def test_large_cost_exceeds_bucket(self):
        rl = RateLimiter(tokens_per_minute=5)
        assert rl.acquire("large", cost=10) is False


# =============================================================================
# 2. CoordinateValidator integration tests
# =============================================================================


class TestCoordinateValidatorIntegration:
    """Valid coords pass through, invalid are clamped, edge cases."""

    def test_valid_point(self):
        r = CoordinateValidator.validate_point(52.5, 34.0)
        assert r.valid is True
        assert r.clamped is False

    def test_x_negative_clamped(self):
        r = CoordinateValidator.validate_x(-5.0)
        assert r.valid is True
        assert r.clamped is True
        assert "clamped to 0" in " ".join(r.warnings)

    def test_x_overshoot_clamped(self):
        r = CoordinateValidator.validate_x(200.0)
        assert r.valid is True
        assert r.clamped is True

    def test_y_negative_clamped(self):
        r = CoordinateValidator.validate_y(-10.0)
        assert r.valid is True
        assert r.clamped is True

    def test_y_overshoot_clamped(self):
        r = CoordinateValidator.validate_y(100.0)
        assert r.valid is True
        assert r.clamped is True

    def test_non_numeric_x(self):
        r = CoordinateValidator.validate_x("abc")
        assert r.valid is False
        assert len(r.errors) > 0

    def test_non_numeric_y(self):
        r = CoordinateValidator.validate_y(None)
        assert r.valid is False
        assert len(r.errors) > 0

    def test_clamp_x(self):
        assert CoordinateValidator.clamp_x(-1) == 0.0
        assert CoordinateValidator.clamp_x(50) == 50.0
        assert CoordinateValidator.clamp_x(200) == 105.0

    def test_clamp_y(self):
        assert CoordinateValidator.clamp_y(-1) == 0.0
        assert CoordinateValidator.clamp_y(34) == 34.0
        assert CoordinateValidator.clamp_y(100) == 68.0

    def test_validate_event_spatial_clamps_coords(self):
        event = {"x": -10.0, "y": 200.0}
        r = CoordinateValidator.validate_event_spatial(event)
        assert r.clamped is True
        assert event["x"] == 0.0
        assert event["y"] == 68.0

    def test_validate_event_spatial_missing_fields(self):
        event = {"type": "shot"}
        r = CoordinateValidator.validate_event_spatial(event)
        assert r.valid is True

    def test_validate_event_spatial_end_coords(self):
        event = {"end_x": 150.0, "end_y": -5.0}
        r = CoordinateValidator.validate_event_spatial(event)
        assert r.clamped is True
        assert event["end_x"] == 105.0
        assert event["end_y"] == 0.0

    def test_validate_point_both_axes(self):
        r = CoordinateValidator.validate_point(-1.0, -2.0)
        assert r.valid is True
        assert r.clamped is True


# =============================================================================
# 3. Timing decorator tests
# =============================================================================


class TestTimingDecorator:
    """Basic timing, slow function warning, nested, stackable."""

    def test_basic_timing_no_warning(self):
        @timed()
        def fast_func():
            return 42

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = fast_func()
        assert result == 42
        # Fast functions <100ms should not trigger warnings
        slow_warnings = [x for x in w if "took" in str(x.message)]
        assert len(slow_warnings) == 0

    def test_timing_returns_value(self):
        @timed()
        def add(a, b):
            return a + b

        assert add(3, 4) == 7

    def test_slow_function_warning(self):
        @timed()
        def slow_func():
            time.sleep(0.15)
            return "done"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = slow_func()
        assert result == "done"
        slow_warnings = [x for x in w if "took" in str(x.message)]
        assert len(slow_warnings) > 0
        assert any("slow_func" in str(m.message) for m in slow_warnings)

    def test_nested_timing(self):
        @timed()
        def inner():
            return "inner"

        @timed()
        def outer():
            return inner() + "_outer"

        with warnings.catch_warnings(record=True) as _:
            warnings.simplefilter("always")
            result = outer()
        assert result == "inner_outer"

    def test_stackable(self):
        call_order = []

        @timed()
        def level1():
            call_order.append(1)
            return level2()

        @timed()
        def level2():
            call_order.append(2)
            return "ok"

        result = level1()
        assert result == "ok"
        assert call_order == [1, 2]

    def test_exception_preserved(self):
        @timed()
        def crash():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            crash()


# =============================================================================
# 4. Secrets management tests
# =============================================================================


class TestSecretsManagement:
    """get/set, persistence, missing key, env var override."""

    def test_get_missing_key_returns_none(self):
        assert secrets_mod.get_api_key("nonexistent_service_xyz") is None

    def test_set_and_get(self, tmp_path):
        with patch.object(secrets_mod, "_secrets_path", return_value=tmp_path / "secrets.json"):
            secrets_mod.set_api_key("test_svc", "key123")
            assert secrets_mod.get_api_key("test_svc") == "key123"

    def test_persistence(self, tmp_path):
        with patch.object(secrets_mod, "_secrets_path", return_value=tmp_path / "secrets.json"):
            secrets_mod.set_api_key("persist_svc", "persist_key")
            # Re-load from disk
            val = secrets_mod.get_api_key("persist_svc")
            assert val == "persist_key"

    def test_list_services(self, tmp_path):
        with patch.object(secrets_mod, "_secrets_path", return_value=tmp_path / "secrets.json"):
            secrets_mod.set_api_key("svc_a", "a")
            secrets_mod.set_api_key("svc_b", "b")
            services = secrets_mod.list_services()
            assert "svc_a" in services
            assert "svc_b" in services

    def test_env_var_override(self, tmp_path):
        with patch.object(secrets_mod, "_secrets_path", return_value=tmp_path / "secrets.json"):
            secrets_mod.set_api_key("env_test", "file_value")
            with patch.dict(os.environ, {"ENV_TEST_API_KEY": "env_value"}, clear=False):
                assert secrets_mod.get_api_key("env_test") == "env_value"

    def test_env_var_no_file_fallback(self):
        with patch.dict(os.environ, {"MY_API_KEY": "direct_env"}, clear=False):
            assert secrets_mod.get_api_key("my") == "direct_env"

    def test_delete_key(self, tmp_path):
        with patch.object(secrets_mod, "_secrets_path", return_value=tmp_path / "secrets.json"):
            secrets_mod.set_api_key("del_svc", "to_delete")
            assert secrets_mod.get_api_key("del_svc") == "to_delete"
            assert secrets_mod.delete_api_key("del_svc") is True
            assert secrets_mod.get_api_key("del_svc") is None

    def test_delete_missing_key(self):
        assert secrets_mod.delete_api_key("never_existed") is False
