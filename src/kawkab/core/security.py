"""Security utilities - input validation, sanitization, and path safety.

Production-grade security hardening for Kawkab AI:
1. Input validation (match IDs, file paths, user inputs)
2. Path traversal prevention
3. SQL injection prevention (parameterized queries - already done in storage)
4. File type validation
5. Rate limiting for expensive operations
6. Error message sanitization (don't leak internal paths to UI)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


class SecurityValidator:
    """Validates and sanitizes user inputs for security."""

    # Allowed video file extensions
    ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

    # Allowed data-file extensions for vendor imports (elite interop path):
    # event/tracking feeds arrive as JSON, XML (Opta/EPTS), or CSV (Metrica)
    ALLOWED_DATA_EXTENSIONS = {".json", ".xml", ".csv"}
    # Wearable imports accept formats the generic data allowlist must not
    # cover (GPX/FIT/TCX are device exports) -- dedicated allowlist so
    # neither set is silently widened.
    ALLOWED_WEARABLE_EXTENSIONS = {".gpx", ".fit", ".tcx", ".xml", ".csv", ".json"}
    # Face-photo uploads are images cv2.imread can decode -- the data
    # allowlist (.json/.xml/.csv) would reject every legitimate photo.
    ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    # Max file size (2 GB)
    MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024

    @staticmethod
    def validate_match_id(match_id: Any) -> int:
        """Validate and convert match_id to a safe integer.

        Args:
            match_id: Any input claiming to be a match ID

        Returns:
            Validated integer match_id

        Raises:
            ValueError: If input is not a valid positive integer
        """
        try:
            mid = int(match_id)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid match_id: {match_id!r}") from e

        if mid < 0:
            raise ValueError(f"match_id must be non-negative, got {mid}")
        if mid > 999_999_999:
            raise ValueError(f"match_id too large: {mid}")

        return mid

    @staticmethod
    def validate_video_path(video_path: str | Path) -> Path:
        """Validate a video file path for safety.

        Checks:
        - Path is within allowed directories (Documents/KawkabAI/videos/)
        - File extension is allowed
        - File exists (optional, can be checked separately)
        - No path traversal sequences

        Args:
            video_path: Path to validate

        Returns:
            Validated Path object

        Raises:
            ValueError: If path is invalid or unsafe
        """
        path = Path(video_path)

        # Check for path traversal
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Invalid path: {video_path}") from e

        # Check extension
        if resolved.suffix.lower() not in SecurityValidator.ALLOWED_VIDEO_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {resolved.suffix}. "
                f"Allowed: {', '.join(SecurityValidator.ALLOWED_VIDEO_EXTENSIONS)}"
            )

        # Check for path traversal (resolved path must be under user docs)
        from kawkab.core.paths import get_paths

        docs = get_paths().documents.resolve()
        try:
            resolved.relative_to(docs)
        except ValueError:
            logger.error(f"Video path outside KawkabAI directory: {resolved}")
            raise ValueError(
                f"Path traversal denied: {resolved} is not within {docs}. "
                f"Only files in the KawkabAI videos directory are allowed."
            ) from None

        return resolved

    @staticmethod
    def validate_data_file_path(file_path: str | Path) -> Path:
        """Validate a vendor data-file path (JSON/XML/CSV) for safety.

        Same protections as validate_video_path (allowlist directory,
        extension allowlist, traversal denial) but for the data feeds the
        vendor-import endpoints consume. Kept separate from
        validate_video_path so neither allowlist is silently widened.

        Note: the StatsBomb import endpoint previously called
        validate_video_path on .json feeds — every local-file StatsBomb
        import was rejected with "Unsupported file type" before this
        validator existed.
        """
        path = Path(file_path)
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Invalid path: {file_path}") from e

        if resolved.suffix.lower() not in SecurityValidator.ALLOWED_DATA_EXTENSIONS:
            raise ValueError(
                f"Unsupported data file type: {resolved.suffix}. "
                f"Allowed: {', '.join(sorted(SecurityValidator.ALLOWED_DATA_EXTENSIONS))}"
            )

        from kawkab.core.paths import get_paths

        docs = get_paths().documents.resolve()
        try:
            resolved.relative_to(docs)
        except ValueError:
            logger.error(f"Data file outside KawkabAI directory: {resolved}")
            raise ValueError(
                f"Path traversal denied: {resolved} is not within {docs}. "
                f"Only files in the KawkabAI directory are allowed."
            ) from None

        return resolved

    @staticmethod
    def validate_directory_path(dir_path: str | Path) -> Path:
        """Validate a vendor data *directory* for season-scale imports.

        Same protections as validate_data_file_path (documents-dir
        allowlist, traversal denial) but for directories: no extension
        check (a directory has none) plus an explicit must-be-a-dir
        check. Kept separate so the file-validator semantics are not
        silently widened to accept directories.
        """
        path = Path(dir_path)
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Invalid path: {dir_path}") from e

        from kawkab.core.paths import get_paths

        docs = get_paths().documents.resolve()
        try:
            resolved.relative_to(docs)
        except ValueError:
            logger.error(f"Directory outside KawkabAI directory: {resolved}")
            raise ValueError(
                f"Path traversal denied: {resolved} is not within {docs}. "
                f"Only directories in the KawkabAI directory are allowed."
            ) from None

        if not resolved.is_dir():
            raise ValueError(f"not a directory: {resolved}")

        return resolved

    @staticmethod
    def validate_image_path(file_path: str | Path) -> Path:
        """Validate a user-supplied image path (face photos) for safety.

        Same structure as validate_data_file_path (documents-directory
        allowlist, extension allowlist, traversal denial) but with the
        image extension set cv2.imread actually decodes.
        """
        path = Path(file_path)
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Invalid path: {file_path}") from e

        if resolved.suffix.lower() not in SecurityValidator.ALLOWED_IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image file type: {resolved.suffix}. "
                f"Allowed: {', '.join(sorted(SecurityValidator.ALLOWED_IMAGE_EXTENSIONS))}"
            )

        from kawkab.core.paths import get_paths

        docs = get_paths().documents.resolve()
        try:
            resolved.relative_to(docs)
        except ValueError:
            logger.error(f"Image outside KawkabAI directory: {resolved}")
            raise ValueError(
                f"Path traversal denied: {resolved} is not within {docs}. "
                f"Only files in the KawkabAI directory are allowed."
            ) from None

        return resolved

    @staticmethod
    def validate_wearable_path(file_path: str | Path) -> Path:
        """Validate a wearable device-export path (GPX/FIT/TCX/...) for safety.

        Same structure as validate_data_file_path (documents-directory
        allowlist, extension allowlist, traversal denial) but with the
        wearable extension set -- import_wearable used to be unvalidated,
        and validating it against ALLOWED_DATA_EXTENSIONS would have
        rejected every legitimate .gpx/.fit/.tcx import.
        """
        path = Path(file_path)
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Invalid path: {file_path}") from e

        if resolved.suffix.lower() not in SecurityValidator.ALLOWED_WEARABLE_EXTENSIONS:
            raise ValueError(
                f"Unsupported wearable file type: {resolved.suffix}. "
                f"Allowed: {', '.join(sorted(SecurityValidator.ALLOWED_WEARABLE_EXTENSIONS))}"
            )

        from kawkab.core.paths import get_paths

        docs = get_paths().documents.resolve()
        try:
            resolved.relative_to(docs)
        except ValueError:
            logger.error(f"Wearable file outside KawkabAI directory: {resolved}")
            raise ValueError(
                f"Path traversal denied: {resolved} is not within {docs}. "
                f"Only files in the KawkabAI directory are allowed."
            ) from None

        return resolved

    @staticmethod
    def sanitize_string(value: str, max_length: int = 255, allowed_chars: str | None = None) -> str:
        """Sanitize a string input.

        Args:
            value: Input string
            max_length: Maximum allowed length
            allowed_chars: Regex pattern of allowed characters (default: alphanumeric + spaces + basic punctuation)

        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            value = str(value)

        if len(value) > max_length:
            value = value[:max_length]

        if allowed_chars is None:
            # Allow alphanumeric, spaces, hyphens, underscores, dots, commas, apostrophes
            allowed_chars = r"[^a-zA-Z0-9\s\-_.,'()/:]"

        # Remove control characters and null bytes
        value = value.replace("\x00", "")
        value = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", value)

        # Apply character filter if provided
        if allowed_chars:
            value = re.sub(allowed_chars, "", value)

        return value.strip()

    @staticmethod
    def validate_jersey_number(number: Any) -> int:
        """Validate a jersey number.

        Args:
            number: Input claiming to be a jersey number

        Returns:
            Validated integer

        Raises:
            ValueError: If invalid
        """
        try:
            n = int(number)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid jersey number: {number!r}") from e

        if n < 0 or n > 99:
            raise ValueError(f"Jersey number must be 0-99, got {n}")

        return n

    @staticmethod
    def validate_team_name(name: str) -> str:
        """Validate and sanitize a team name.

        Args:
            name: Team name input

        Returns:
            Sanitized name
        """
        sanitized = SecurityValidator.sanitize_string(name, max_length=100)
        if not sanitized:
            raise ValueError("Team name cannot be empty")
        return sanitized

    @staticmethod
    def validate_season_name(name: str) -> str:
        """Validate and sanitize a season name.

        Args:
            name: Season name input

        Returns:
            Sanitized name
        """
        sanitized = SecurityValidator.sanitize_string(name, max_length=100)
        if not sanitized:
            raise ValueError("Season name cannot be empty")
        return sanitized

    @staticmethod
    def validate_int(value: Any) -> int:
        """Validate and convert a generic ID-like input to a safe integer.

        For inputs that are IDs but not specifically a match (player_id,
        session_id, ...) -- same bounds as validate_match_id.

        Args:
            value: Any input claiming to be a non-negative integer ID

        Returns:
            Validated integer

        Raises:
            ValueError: If input is not a valid non-negative integer
        """
        try:
            v = int(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid integer: {value!r}") from e

        if v < 0:
            raise ValueError(f"Value must be non-negative, got {v}")
        if v > 999_999_999:
            raise ValueError(f"Value too large: {v}")

        return v

    @staticmethod
    def validate_track_id(track_id: Any) -> int:
        """Validate and convert a player tracking ID to a safe integer.

        Args:
            track_id: Any input claiming to be a player track ID

        Returns:
            Validated integer track_id

        Raises:
            ValueError: If input is not a valid non-negative integer
        """
        try:
            tid = int(track_id)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid track_id: {track_id!r}") from e

        if tid < 0:
            raise ValueError(f"track_id must be non-negative, got {tid}")

        return tid

    @staticmethod
    def validate_positive_float(value: Any, field_name: str = "value") -> float:
        """Validate a numeric input and clamp it to a non-negative float.

        Args:
            value: Any input claiming to be a non-negative measurement
                (distance, speed, count, ...)
            field_name: Name used in the error message on invalid input

        Returns:
            Validated float, clamped to >= 0.0

        Raises:
            ValueError: If input cannot be converted to a float
        """
        try:
            v = float(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid {field_name}: {value!r}") from e

        return max(0.0, v)

    @staticmethod
    def validate_float_range(value: Any, lo: float, hi: float, field_name: str = "value") -> float:
        """Validate a numeric input and clamp it to an inclusive [lo, hi] range.

        Args:
            value: Any input claiming to be a number in [lo, hi] (e.g. a
                1-5 coach rating)
            lo: Lower bound (inclusive)
            hi: Upper bound (inclusive)
            field_name: Name used in the error message on invalid input

        Returns:
            Validated float, clamped to [lo, hi]

        Raises:
            ValueError: If input cannot be converted to a float
        """
        try:
            v = float(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid {field_name}: {value!r}") from e

        return max(float(lo), min(float(hi), v))

    @staticmethod
    def validate_event_type(event_type: Any) -> str:
        """Validate and sanitize a match event type string.

        Args:
            event_type: Any input claiming to be an event type (e.g. "pass")

        Returns:
            Sanitized event type string

        Raises:
            ValueError: If input is empty after sanitization
        """
        sanitized = SecurityValidator.sanitize_string(str(event_type), max_length=50)
        if not sanitized:
            raise ValueError(f"Invalid event_type: {event_type!r}")
        return sanitized

    @staticmethod
    def validate_event_dict(event: dict) -> dict:
        """Validate that an event dict has the fields save_event requires.

        Args:
            event: Event data about to be persisted

        Returns:
            The same event dict, unchanged

        Raises:
            ValueError: If event is missing the required "type" or
                "timestamp" keys
        """
        if not isinstance(event, dict):
            raise ValueError(f"Event must be a dict, got {type(event).__name__}")
        missing = [k for k in ("type", "timestamp") if k not in event]
        if missing:
            raise ValueError(f"Event dict missing required field(s): {', '.join(missing)}")
        return event

    @staticmethod
    def check_rate_limit(operation: str, key: str = "global") -> bool:
        """Check whether an operation is allowed under its rate limit.

        Args:
            operation: One of 'analysis', 'export', 'search', or a custom key.
            key: Sub-key for independent rate tracking (e.g. match ID).

        Returns:
            True if the operation is allowed, False if rate limited.
        """
        bucket_key = f"{operation}:{key}"
        return _global_rate_limiter.acquire(bucket_key, cost=1)


class RateLimiter:
    """Token bucket rate limiter for expensive operations.

    Default per-operation limits:
        - analysis: 5 tokens/min
        - export:   10 tokens/min
        - search:   30 tokens/min
        - default:  60 tokens/min
    """

    _DEFAULT_RATES: dict[str, int] = {
        "analysis": 5,
        "export": 10,
        "search": 30,
    }

    def __init__(self, tokens_per_minute: int = 60) -> None:
        self._default_rate = tokens_per_minute
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (fill_time, tokens)
        self._overrides: dict[str, int] = {}

    def configure(self, key_prefix: str, tokens_per_minute: int) -> None:
        """Set a custom rate limit for a given operation prefix."""
        self._overrides[key_prefix] = tokens_per_minute

    def _rate_for_key(self, key: str) -> int:
        prefix = key.split(":")[0]
        if prefix in self._overrides:
            return self._overrides[prefix]
        return self._DEFAULT_RATES.get(prefix, self._default_rate)

    def acquire(self, key: str, cost: int = 1) -> bool:
        """Try to consume *cost* tokens. Returns True if allowed, False if rate limited."""
        import time

        now = time.time()
        rate = self._rate_for_key(key)
        max_tokens = float(rate)

        fill_time, tokens = self._buckets.get(key, (now, max_tokens))
        elapsed = now - fill_time
        tokens = min(max_tokens, tokens + elapsed * (rate / 60.0))
        fill_time = now

        if tokens < cost:
            self._buckets[key] = (fill_time, tokens)
            return False

        tokens -= cost
        self._buckets[key] = (fill_time, tokens)
        return True


_global_rate_limiter = RateLimiter()
_global_rate_limiter.configure("analysis", 5)
_global_rate_limiter.configure("export", 10)
_global_rate_limiter.configure("search", 30)


class ErrorSanitizer:
    """Sanitizes error messages for user display."""

    @staticmethod
    def sanitize_error(error: str | Exception) -> str:
        """Sanitize an error message for safe display.

        Removes:
        - File paths
        - Internal function names
        - Stack traces
        - Sensitive information
        """
        message = str(error) if isinstance(error, Exception) else str(error)

        # Remove Windows paths
        message = re.sub(r"[A-Za-z]:\\[^\s]+", "[path]", message)
        # Remove Unix paths
        message = re.sub(r"/[^\s]+", "[path]", message)
        # Remove email addresses
        message = re.sub(r"[\w.-]+@[\w.-]+\.[a-zA-Z]{2,}", "[email]", message)
        # Remove IP addresses
        message = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[ip]", message)
        # Remove long hex strings (could be tokens/keys)
        message = re.sub(r"\b[a-f0-9]{32,}\b", "[token]", message)

        # Truncate if too long
        if len(message) > 500:
            message = message[:500] + "..."

        return message
