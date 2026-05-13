"""Exponential backoff retry decorator with full jitter."""
from __future__ import annotations

import functools
import logging
import random
import time
from typing import Callable, Optional, Tuple, Type

from src.core.exceptions import QuotaExceededError, RateLimitError, APIError

logger = logging.getLogger(__name__)

_DEFAULT_EXCEPTIONS: Tuple[Type[Exception], ...] = (RateLimitError, APIError)


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: Tuple[Type[Exception], ...] = _DEFAULT_EXCEPTIONS,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable:
    """Decorator that retries a function on specified exceptions with full-jitter backoff.

    Full jitter formula: sleep = random(0, min(cap, base * 2^attempt))
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc: Optional[Exception] = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if isinstance(exc, QuotaExceededError):
                        raise  # credits depleted — retrying won't help
                    if attempt == max_attempts - 1:
                        raise
                    # Honor retry_after from RateLimitError if available
                    if isinstance(exc, RateLimitError) and exc.retry_after > 0:
                        sleep_time = exc.retry_after
                    else:
                        cap = min(max_delay, base_delay * (2 ** attempt))
                        sleep_time = random.uniform(0, cap)
                    logger.warning(
                        "Retry %d/%d for %s after %.2fs — %s: %s",
                        attempt + 1,
                        max_attempts,
                        func.__name__,
                        sleep_time,
                        type(exc).__name__,
                        exc,
                    )
                    if on_retry:
                        on_retry(exc, attempt)
                    time.sleep(sleep_time)
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator
