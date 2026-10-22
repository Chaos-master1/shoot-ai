"""Performance timing decorator for expensive analytical paths.

Provides a @timed decorator that wraps any function with time.perf_counter()
and logs warnings for calls exceeding 100ms. Thread-safe via contextvars.
"""

from __future__ import annotations

import functools
import time
import warnings
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

_depth: ContextVar[int] = ContextVar("timing_depth", default=0)
_SLOW_THRESHOLD_MS = 100.0


def timed(log_level: str = "DEBUG") -> Callable[[F], F]:
    """Decorator that times function execution and warns on slow calls.

    Args:
        log_level: Ignored (always uses warnings.warn for slow calls,
                   but kept for API compatibility).

    Returns:
        Decorated function with timing instrumentation.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            token = _depth.set(_depth.get() + 1)
            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                depth = _depth.get()
                indent = "  " * (depth - 1)
                if elapsed_ms > _SLOW_THRESHOLD_MS:
                    warnings.warn(
                        f"{indent}{func.__name__} took {elapsed_ms:.1f}ms "
                        f"(threshold {_SLOW_THRESHOLD_MS:.0f}ms)",
                        stacklevel=2,
                    )
                _depth.reset(token)

        return wrapper  # type: ignore[return-value]

    return decorator
