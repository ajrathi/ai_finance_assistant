"""In-memory TTL + LRU cache for market data, news, LLM responses, and FAISS results."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional


class TTLCache:
    """Thread-safe in-memory cache with per-entry TTL and LRU eviction."""

    def __init__(self, default_ttl: int = 1800, max_size: int = 500):
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Return cached value if present and not expired, else None."""
        with self._lock:
            if key not in self._store:
                return None
            value, expires_at = self._store[key]
            if time.monotonic() > expires_at:
                del self._store[key]
                return None
            # LRU: move to end
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store value under key with given TTL (seconds)."""
        ttl = ttl if ttl is not None else self._default_ttl
        expires_at = time.monotonic() + ttl
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, expires_at)
            # LRU eviction when over capacity
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def stats(self) -> Dict[str, int]:
        with self._lock:
            now = time.monotonic()
            alive = sum(1 for _, (_, exp) in self._store.items() if now <= exp)
            return {"total": len(self._store), "alive": alive, "expired": len(self._store) - alive}


class CacheManager:
    """Named cache instances with their own TTLs, backed by shared max_size."""

    def __init__(self, max_size: int = 500):
        self._max_size = max_size
        self._caches: Dict[str, TTLCache] = {}
        self._lock = threading.Lock()

    def get_cache(self, name: str, default_ttl: int = 1800) -> TTLCache:
        with self._lock:
            if name not in self._caches:
                self._caches[name] = TTLCache(default_ttl=default_ttl, max_size=self._max_size)
            return self._caches[name]


def make_cache_key(*args: Any, **kwargs: Any) -> str:
    """Generate a deterministic SHA-256 cache key from arbitrary arguments."""
    payload = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


# Module-level cache manager singleton
_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    global _manager
    if _manager is None:
        from src.core.config import settings
        _manager = CacheManager(max_size=settings.cache.max_size)
    return _manager


def get_market_cache() -> TTLCache:
    from src.core.config import settings
    return get_cache_manager().get_cache("market", default_ttl=settings.cache.market_data_ttl)


def get_news_cache() -> TTLCache:
    from src.core.config import settings
    return get_cache_manager().get_cache("news", default_ttl=settings.cache.news_ttl)


def get_llm_cache() -> TTLCache:
    from src.core.config import settings
    return get_cache_manager().get_cache("llm", default_ttl=settings.cache.llm_response_ttl)


def get_rag_cache() -> TTLCache:
    from src.core.config import settings
    return get_cache_manager().get_cache("rag", default_ttl=settings.cache.faiss_retrieval_ttl)
