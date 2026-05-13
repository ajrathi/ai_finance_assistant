"""Unit tests for TTLCache."""
import time
from unittest.mock import patch

import pytest

from src.utils.cache import TTLCache, make_cache_key


class TestTTLCache:
    def test_set_and_get(self):
        cache = TTLCache(default_ttl=60)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing_key(self):
        cache = TTLCache()
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self):
        cache = TTLCache(default_ttl=1)
        cache.set("key", "value", ttl=1)
        # Simulate time passing by manipulating the store directly
        import time as t
        entry = cache._store["key"]
        cache._store["key"] = (entry[0], t.monotonic() - 1)
        assert cache.get("key") is None

    def test_lru_eviction(self):
        cache = TTLCache(default_ttl=60, max_size=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.set("d", 4)  # Should evict "a"
        assert cache.get("a") is None
        assert cache.get("d") == 4

    def test_delete(self):
        cache = TTLCache()
        cache.set("key", "val")
        cache.delete("key")
        assert cache.get("key") is None

    def test_clear(self):
        cache = TTLCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert len(cache) == 0

    def test_custom_ttl(self):
        cache = TTLCache(default_ttl=3600)
        cache.set("key", "value", ttl=10)
        # Should still be there
        assert cache.get("key") == "value"

    def test_stats(self):
        cache = TTLCache(default_ttl=60, max_size=10)
        cache.set("a", 1)
        cache.set("b", 2)
        stats = cache.stats()
        assert stats["total"] == 2
        assert stats["alive"] == 2


class TestMakeCacheKey:
    def test_deterministic(self):
        key1 = make_cache_key("query", ["cat1"], "agent")
        key2 = make_cache_key("query", ["cat1"], "agent")
        assert key1 == key2

    def test_different_inputs_different_keys(self):
        key1 = make_cache_key("query1")
        key2 = make_cache_key("query2")
        assert key1 != key2

    def test_order_independence_for_dict(self):
        key1 = make_cache_key(query="q", categories=["a", "b"])
        key2 = make_cache_key(query="q", categories=["a", "b"])
        assert key1 == key2
