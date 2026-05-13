"""Gemini 2.0 Flash LLM client with rate limiting, retry, and caching."""
from __future__ import annotations

import logging
from typing import Generator, Optional

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from src.core.exceptions import APIError, QuotaExceededError, RateLimitError
from src.utils.cache import get_llm_cache, make_cache_key
from src.utils.rate_limiter import get_rate_limiter
from src.utils.retry import with_retry

logger = logging.getLogger(__name__)

_RETRYABLE = (RateLimitError, APIError)


def _map_genai_error(exc: Exception) -> Exception:
    """Convert google-genai SDK errors to internal exception types."""
    msg = str(exc)
    if isinstance(exc, genai_errors.ClientError):
        status = getattr(exc, "status_code", None)
        low = msg.lower()
        if "credits" in low or "billing" in low:
            return QuotaExceededError(msg, service="gemini", status_code=429)
        if status == 429 or "quota" in low or "rate" in low:
            return RateLimitError(msg)
        # 4xx errors other than 429 are not transient — don't retry
        return QuotaExceededError(msg, service="gemini", status_code=status or 400)
    if isinstance(exc, genai_errors.ServerError):
        return APIError(msg, service="gemini")
    return APIError(f"Gemini error: {exc}", service="gemini")


class GeminiClient:
    """Wrapper around Google Gemini 2.0 Flash with rate limiting and caching."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        enable_cache: bool = True,
    ):
        from src.core.config import settings
        key = api_key or settings.google_api_key
        if not key:
            raise APIError(
                "GOOGLE_API_KEY is not set. Add it to your .env file.",
                service="gemini",
            )
        self._client = genai.Client(api_key=key)
        self._model_name = model or settings.llm.model
        self._max_tokens = settings.llm.max_tokens
        self._rate_limiter = get_rate_limiter()
        self._cache = get_llm_cache() if enable_cache else None
        logger.info("GeminiClient initialized with model: %s", self._model_name)

    def _make_key(self, system_prompt: str, user_prompt: str) -> str:
        return make_cache_key(self._model_name, system_prompt, user_prompt)

    def _build_config(self, temperature: float) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=self._max_tokens,
        )

    @with_retry(max_attempts=3, base_delay=1.0, max_delay=30.0, exceptions=_RETRYABLE)
    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
        use_cache: bool = True,
    ) -> str:
        """Generate a response. Blocks on rate limit, caches identical prompts."""
        if use_cache and self._cache:
            cache_key = self._make_key(system_prompt, user_prompt)
            cached = self._cache.get(cache_key)
            if cached:
                logger.debug("LLM cache hit")
                return cached

        self._rate_limiter.acquire()
        try:
            full_prompt = f"{system_prompt}\n\n{user_prompt}".strip() if system_prompt else user_prompt
            config = self._build_config(temperature)
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=full_prompt,
                config=config,
            )
            text = response.text
            if text is None:
                raise APIError("Gemini returned empty response (possibly blocked)", service="gemini")
        except (QuotaExceededError, RateLimitError, APIError):
            raise
        except Exception as exc:
            raise _map_genai_error(exc) from exc

        if use_cache and self._cache:
            self._cache.set(cache_key, text)

        return text

    def generate_structured(
        self,
        user_prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
    ) -> str:
        """For classification tasks — always temperature 0, no caching of results."""
        return self.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            use_cache=True,
        )

    def stream(
        self,
        user_prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> Generator[str, None, None]:
        """Stream response chunks. Not cached."""
        self._rate_limiter.acquire()
        try:
            full_prompt = f"{system_prompt}\n\n{user_prompt}".strip() if system_prompt else user_prompt
            config = self._build_config(temperature)
            for chunk in self._client.models.generate_content_stream(
                model=self._model_name,
                contents=full_prompt,
                config=config,
            ):
                if chunk.text:
                    yield chunk.text
        except (QuotaExceededError, RateLimitError, APIError):
            raise
        except Exception as exc:
            raise _map_genai_error(exc) from exc


# Module-level singleton
_client: Optional[GeminiClient] = None


def get_llm_client() -> GeminiClient:
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client
