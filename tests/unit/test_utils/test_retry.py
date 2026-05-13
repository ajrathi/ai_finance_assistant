"""Unit tests for the retry decorator."""
from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import APIError, RateLimitError
from src.utils.retry import with_retry


class TestWithRetry:
    def test_no_retry_on_success(self):
        call_count = 0

        @with_retry(max_attempts=3)
        def success_func():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = success_func()
        assert result == "ok"
        assert call_count == 1

    def test_retries_on_api_error(self):
        call_count = 0

        @with_retry(max_attempts=3, base_delay=0.001, max_delay=0.01)
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise APIError("temporary failure")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert call_count == 3

    def test_raises_after_max_attempts(self):
        @with_retry(max_attempts=2, base_delay=0.001)
        def always_fails():
            raise APIError("permanent failure")

        with pytest.raises(APIError):
            always_fails()

    def test_does_not_retry_on_unspecified_exception(self):
        call_count = 0

        @with_retry(max_attempts=3, exceptions=(APIError,))
        def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("not retried")

        with pytest.raises(ValueError):
            raises_value_error()
        assert call_count == 1

    def test_rate_limit_error_retried(self):
        call_count = 0

        @with_retry(max_attempts=3, base_delay=0.001)
        def rate_limited():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RateLimitError("rate limited")
            return "ok"

        result = rate_limited()
        assert result == "ok"
        assert call_count == 2

    def test_on_retry_callback(self):
        retries = []

        @with_retry(max_attempts=3, base_delay=0.001, on_retry=lambda exc, n: retries.append(n))
        def fails_once():
            if not retries:
                raise APIError("first failure")
            return "ok"

        result = fails_once()
        assert result == "ok"
        assert len(retries) == 1
