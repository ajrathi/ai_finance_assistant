"""Unit tests for GeminiClient — caching, error mapping, and response handling.

All google-genai SDK calls are mocked; no real API key is required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import APIError, QuotaExceededError, RateLimitError
from src.core.llm_client import GeminiClient, _map_genai_error


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_client(enable_cache: bool = True) -> GeminiClient:
    """Return a GeminiClient whose google-genai SDK calls are fully mocked."""
    with patch("src.core.llm_client.genai.Client"):
        client = GeminiClient(api_key="fake-key", enable_cache=enable_cache)
    return client


def _mock_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.text = text
    return resp


# ─── Cache behaviour ──────────────────────────────────────────────────────────

class TestGenerateCaching:

    def test_cache_hit_returns_cached_value_without_api_call(self):
        """Pre-populating the cache must prevent the API from being called."""
        client = _make_client()
        cache_key = client._make_key("You are helpful.", "Say hello")
        client._cache.set(cache_key, "Cached response")

        result = client.generate("Say hello", system_prompt="You are helpful.")

        client._client.models.generate_content.assert_not_called()
        assert result == "Cached response"

    def test_api_response_is_stored_in_cache(self):
        """A successful API call must populate the cache for future hits."""
        client = _make_client()
        client._client.models.generate_content.return_value = _mock_response("Fresh response")

        client.generate("What is a bond?")

        cache_key = client._make_key("", "What is a bond?")
        assert client._cache.get(cache_key) == "Fresh response"

    def test_use_cache_false_bypasses_cache_read(self):
        """use_cache=False must call the API even if a cached value exists."""
        client = _make_client()
        cache_key = client._make_key("", "Explain bonds")
        client._cache.set(cache_key, "Old cached value")
        client._client.models.generate_content.return_value = _mock_response("Fresh")

        result = client.generate("Explain bonds", use_cache=False)

        client._client.models.generate_content.assert_called_once()
        assert result == "Fresh"

    def test_use_cache_false_does_not_write_to_cache(self):
        """use_cache=False must not cache the response."""
        client = _make_client()
        client._client.models.generate_content.return_value = _mock_response("Uncached")

        client.generate("Some query", use_cache=False)

        cache_key = client._make_key("", "Some query")
        assert client._cache.get(cache_key) is None

    def test_generate_returns_text_string(self):
        client = _make_client()
        client._client.models.generate_content.return_value = _mock_response("Educational content.")
        result = client.generate("What is an ETF?")
        assert result == "Educational content."

    def test_cache_disabled_client_never_caches(self):
        """enable_cache=False must not store or read from any cache."""
        client = _make_client(enable_cache=False)
        assert client._cache is None
        client._client.models.generate_content.return_value = _mock_response("No cache")
        result = client.generate("Same question")
        assert result == "No cache"


# ─── None / empty response ────────────────────────────────────────────────────

class TestNoneResponse:

    def test_none_text_raises_api_error(self):
        client = _make_client()
        resp = MagicMock()
        resp.text = None
        client._client.models.generate_content.return_value = resp
        with pytest.raises(APIError, match="empty response"):
            client.generate("What is inflation?", use_cache=False)


# ─── Error mapping (_map_genai_error) ─────────────────────────────────────────

class TestErrorMapping:
    """_map_genai_error must translate SDK errors to internal exception types."""

    def test_client_error_429_becomes_rate_limit_error(self):
        from google.genai import errors as genai_errors
        exc = genai_errors.ClientError(
            429, {"error": {"message": "Rate limit exceeded"}}, MagicMock()
        )
        result = _map_genai_error(exc)
        assert isinstance(result, RateLimitError)

    def test_client_error_with_quota_in_message_becomes_rate_limit_error(self):
        from google.genai import errors as genai_errors
        exc = genai_errors.ClientError(
            429, {"error": {"message": "Quota exceeded for the day"}}, MagicMock()
        )
        result = _map_genai_error(exc)
        assert isinstance(result, RateLimitError)

    def test_client_error_with_billing_becomes_quota_exceeded_error(self):
        from google.genai import errors as genai_errors
        exc = genai_errors.ClientError(
            402, {"error": {"message": "Billing account required"}}, MagicMock()
        )
        result = _map_genai_error(exc)
        assert isinstance(result, QuotaExceededError)

    def test_client_error_404_becomes_quota_exceeded_error(self):
        # 404 model-not-found is non-retryable → mapped to QuotaExceededError
        from google.genai import errors as genai_errors
        exc = genai_errors.ClientError(
            404, {"error": {"message": "Model not found"}}, MagicMock()
        )
        result = _map_genai_error(exc)
        assert isinstance(result, QuotaExceededError)

    def test_server_error_becomes_api_error(self):
        from google.genai import errors as genai_errors
        exc = genai_errors.ServerError(
            500, {"error": {"message": "Internal server error"}}, MagicMock()
        )
        result = _map_genai_error(exc)
        assert isinstance(result, APIError)

    def test_unexpected_exception_becomes_api_error(self):
        result = _map_genai_error(RuntimeError("unexpected failure"))
        assert isinstance(result, APIError)
        assert "unexpected failure" in str(result)


# ─── Missing API key ──────────────────────────────────────────────────────────

class TestMissingApiKey:

    def test_no_api_key_raises_api_error(self):
        with patch("src.core.llm_client.genai.Client"):
            with patch("src.core.config.settings") as mock_settings:
                mock_settings.google_api_key = ""
                with pytest.raises(APIError, match="GOOGLE_API_KEY"):
                    GeminiClient(api_key="")


# ─── generate_structured ──────────────────────────────────────────────────────

class TestGenerateStructured:

    def test_structured_returns_string(self):
        client = _make_client()
        client._client.models.generate_content.return_value = _mock_response(
            '{"intent": "general_finance", "confidence": 0.9}'
        )
        result = client.generate_structured("Classify: what is a bond?")
        assert isinstance(result, str)
        assert "general_finance" in result
