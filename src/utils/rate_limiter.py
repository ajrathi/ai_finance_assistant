"""Token bucket rate limiter for Gemini API (60 req/min)."""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Optional


class TokenBucketRateLimiter:
    """Thread-safe and asyncio-compatible token bucket rate limiter.

    By default, configured for Gemini's free tier: 60 requests/minute.
    """

    def __init__(self, requests_per_minute: int = 60):
        self._capacity = requests_per_minute
        self._tokens = float(requests_per_minute)
        self._refill_rate = requests_per_minute / 60.0  # tokens per second
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now

    @property
    def wait_time_seconds(self) -> float:
        """Estimated seconds until the next token is available."""
        with self._lock:
            self._refill()
            if self._tokens >= 1:
                return 0.0
            return (1 - self._tokens) / self._refill_rate

    def acquire(self, n_tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """Block until n_tokens are available. Returns True on success, False on timeout."""
        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= n_tokens:
                    self._tokens -= n_tokens
                    return True
                wait = (n_tokens - self._tokens) / self._refill_rate
            if deadline is not None and time.monotonic() + wait > deadline:
                return False
            time.sleep(min(wait, 0.1))

    async def async_acquire(self, n_tokens: int = 1) -> None:
        """Async-compatible acquire. Yields control while waiting."""
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= n_tokens:
                    self._tokens -= n_tokens
                    return
                wait = (n_tokens - self._tokens) / self._refill_rate
            await asyncio.sleep(min(wait, 0.1))

    def try_acquire(self, n_tokens: int = 1) -> bool:
        """Non-blocking attempt. Returns True if acquired, False if not enough tokens."""
        with self._lock:
            self._refill()
            if self._tokens >= n_tokens:
                self._tokens -= n_tokens
                return True
            return False


# Module-level singleton, configured from settings at first import.
# Replaced in tests by patching.
_limiter: Optional[TokenBucketRateLimiter] = None


def get_rate_limiter() -> TokenBucketRateLimiter:
    """Return the module-level rate limiter, creating it if needed."""
    global _limiter
    if _limiter is None:
        from src.core.config import settings
        _limiter = TokenBucketRateLimiter(
            requests_per_minute=settings.llm.rate_limit.requests_per_minute
        )
    return _limiter
