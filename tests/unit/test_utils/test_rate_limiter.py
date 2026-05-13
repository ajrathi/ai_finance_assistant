"""Unit tests for TokenBucketRateLimiter."""
import time
from unittest.mock import patch

import pytest

from src.utils.rate_limiter import TokenBucketRateLimiter


class TestTokenBucketRateLimiter:
    def test_acquire_within_capacity(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        # Should acquire immediately when tokens are full
        assert limiter.acquire(n_tokens=1) is True

    def test_try_acquire_success(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        assert limiter.try_acquire(1) is True

    def test_try_acquire_fails_when_empty(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        # Drain all tokens
        limiter._tokens = 0.0
        assert limiter.try_acquire(1) is False

    def test_wait_time_zero_when_tokens_available(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        assert limiter.wait_time_seconds == 0.0

    def test_wait_time_positive_when_empty(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        limiter._tokens = 0.0
        assert limiter.wait_time_seconds > 0.0

    def test_refill_over_time(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        limiter._tokens = 0.0
        # Simulate 1 second passing
        limiter._last_refill = time.monotonic() - 1.0
        limiter._refill()
        assert limiter._tokens == pytest.approx(1.0, abs=0.1)

    def test_cannot_exceed_capacity(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        limiter._tokens = 59.0
        # Simulate 10 seconds
        limiter._last_refill = time.monotonic() - 10.0
        limiter._refill()
        assert limiter._tokens <= limiter._capacity

    def test_acquire_timeout(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        limiter._tokens = 0.0
        # Timeout immediately
        result = limiter.acquire(n_tokens=1, timeout=0.001)
        assert result is False
