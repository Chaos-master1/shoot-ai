"""Tests for security validation utilities."""

from unittest.mock import patch

import pytest

from kawkab.core.security import (
    ErrorSanitizer,
    RateLimiter,
    SecurityValidator,
    _global_rate_limiter,
)


class TestSecurityValidator:
    """Test input validation utilities."""

    def test_validate_match_id_valid(self):
        assert SecurityValidator.validate_match_id(42) == 42
        assert SecurityValidator.validate_match_id("123") == 123
        assert SecurityValidator.validate_match_id(0) == 0

    def test_validate_match_id_invalid(self):
        with pytest.raises(ValueError, match="Invalid match_id"):
            SecurityValidator.validate_match_id("abc")
        with pytest.raises(ValueError, match="Invalid match_id"):
            SecurityValidator.validate_match_id(None)
        with pytest.raises(ValueError, match="match_id must be non-negative"):
            SecurityValidator.validate_match_id(-1)
        with pytest.raises(ValueError, match="match_id too large"):
            SecurityValidator.validate_match_id(1_000_000_000)

    def test_validate_jersey_number(self):
        assert SecurityValidator.validate_jersey_number(7) == 7
        assert SecurityValidator.validate_jersey_number(99) == 99
        assert SecurityValidator.validate_jersey_number(0) == 0
        with pytest.raises(ValueError, match="must be 0-99"):
            SecurityValidator.validate_jersey_number(-1)
        with pytest.raises(ValueError, match="must be 0-99"):
            SecurityValidator.validate_jersey_number(100)
        with pytest.raises(ValueError, match="Invalid jersey number"):
            SecurityValidator.validate_jersey_number("abc")

    def test_sanitize_string(self):
        assert SecurityValidator.sanitize_string("hello") == "hello"
        assert SecurityValidator.sanitize_string("  hello  ") == "hello"
        assert SecurityValidator.sanitize_string("a" * 300, max_length=10) == "a" * 10
        assert SecurityValidator.sanitize_string("hello\x00world") == "helloworld"
        assert SecurityValidator.sanitize_string("hello<script>world") == "helloscriptworld"
        assert SecurityValidator.sanitize_string("test@#$%world") == "testworld"

    def test_validate_team_name(self):
        assert SecurityValidator.validate_team_name("Team A") == "Team A"
        with pytest.raises(ValueError, match="cannot be empty"):
            SecurityValidator.validate_team_name("")
        with pytest.raises(ValueError, match="cannot be empty"):
            SecurityValidator.validate_team_name("   ")

    def test_validate_video_path_outside_docs_rejected(self, tmp_path):
        video = tmp_path / "test.mp4"
        video.write_text("dummy")
        with pytest.raises(ValueError, match="Path traversal denied"):
            SecurityValidator.validate_video_path(str(video))

    def test_validate_video_path_accepted_in_docs(self, tmp_path, monkeypatch):
        from kawkab.core.paths import get_paths

        real_docs = get_paths().documents
        in_docs = real_docs / "test_video_accepted.mp4"
        in_docs.parent.mkdir(parents=True, exist_ok=True)
        in_docs.write_text("dummy")
        result = SecurityValidator.validate_video_path(str(in_docs))
        assert result.resolve() == in_docs.resolve()
        in_docs.unlink()

    def test_validate_video_path_unsupported_extension(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported file type"):
            SecurityValidator.validate_video_path(str(txt))


class TestErrorSanitizer:
    """Test error message sanitization."""

    def test_sanitize_error_removes_paths(self):
        error = "Failed to load C:\\Users\\Test\\video.mp4"
        sanitized = ErrorSanitizer.sanitize_error(error)
        assert "[path]" in sanitized
        assert "C:\\Users" not in sanitized

    def test_sanitize_error_removes_ips(self):
        error = "Connection failed to 192.168.1.1:8080"
        sanitized = ErrorSanitizer.sanitize_error(error)
        assert "[ip]" in sanitized
        assert "192.168.1.1" not in sanitized

    def test_sanitize_error_removes_emails(self):
        error = "Contact admin@example.com for help"
        sanitized = ErrorSanitizer.sanitize_error(error)
        assert "[email]" in sanitized
        assert "admin@example.com" not in sanitized

    def test_sanitize_error_truncates(self):
        error = "x" * 1000
        sanitized = ErrorSanitizer.sanitize_error(error)
        assert len(sanitized) <= 503  # 500 + "..."


class TestTokenBucketRateLimiter:
    """Token bucket RateLimiter tests."""

    def test_token_bucket_starts_full(self):
        limiter = RateLimiter(tokens_per_minute=60)
        assert limiter.acquire("test:1") is True

    def test_requests_within_limit_succeed(self):
        limiter = RateLimiter(tokens_per_minute=60)
        for _ in range(5):
            assert limiter.acquire("test:1") is True

    def test_requests_exceeding_limit_denied(self):
        limiter = RateLimiter(tokens_per_minute=3)
        assert limiter.acquire("heavy:1") is True
        assert limiter.acquire("heavy:1") is True
        assert limiter.acquire("heavy:1") is True
        assert limiter.acquire("heavy:1") is False

    def test_tokens_replenish_over_time(self):
        fake_now = [1000.0]
        with patch("time.time", side_effect=lambda: fake_now[0]):
            limiter = RateLimiter(tokens_per_minute=60)
            assert limiter.acquire("refill:1") is True
            assert limiter.acquire("refill:1") is True
            assert limiter.acquire("refill:1") is True
            fake_now[0] += 5.0
            assert limiter.acquire("refill:1") is True
            fake_now[0] += 100.0
            for _ in range(60):
                assert limiter.acquire("refill:1") is True
            assert limiter.acquire("refill:1") is False

    def test_different_keys_have_independent_buckets(self):
        limiter = RateLimiter(tokens_per_minute=2)
        assert limiter.acquire("key_a:1") is True
        assert limiter.acquire("key_a:1") is True
        assert limiter.acquire("key_a:1") is False
        assert limiter.acquire("key_b:1") is True
        assert limiter.acquire("key_b:1") is True
        assert limiter.acquire("key_b:1") is False

    def test_configure_overrides_default_rate(self):
        limiter = RateLimiter(tokens_per_minute=60)
        limiter.configure("slow", 2)
        assert limiter.acquire("slow:1") is True
        assert limiter.acquire("slow:1") is True
        assert limiter.acquire("slow:1") is False

    def test_configure_other_prefix_still_uses_default(self):
        limiter = RateLimiter(tokens_per_minute=60)
        limiter.configure("slow", 2)
        assert limiter.acquire("fast:1") is True
        for _ in range(59):
            assert limiter.acquire("fast:1") is True

    def test_security_validator_check_rate_limit(self):
        assert SecurityValidator.check_rate_limit("analysis", "test") is True

    def test_security_validator_check_rate_limit_exhausted(self):
        import time

        now = time.time()
        _global_rate_limiter._buckets.clear()
        key = "analysis:test_exhaust"
        _global_rate_limiter._buckets[key] = (now, 0.0)
        assert SecurityValidator.check_rate_limit("analysis", "test_exhaust") is False
