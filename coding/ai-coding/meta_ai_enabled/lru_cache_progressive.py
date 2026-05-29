"""
LRU Cache (Progressive) — Meta AI-Enabled Round (Confirmed Pool #11)
=====================================================================

Problem:
    Implement an LRU (Least Recently Used) cache that evolves through interview
    checkpoints. The cache starts simple and gains features at each checkpoint.

    This is a "progressive" problem: each checkpoint adds a new requirement.

Checkpoint structure (mirrors actual Meta AI-enabled interview):
    Checkpoint 1: Basic LRU — get(key) and put(key, value) in O(1)
    Checkpoint 2: Eviction policy — evict LRU when at capacity (already in CP1)
    Checkpoint 3: TTL (time-to-live) — entries expire after ttl_seconds

Key insight (TTL extension):
    Lazy expiry: check expiry on access (get/put), not in background.
    Store (value, expiry_timestamp) instead of just value.
    Use time.monotonic() for testable, drift-free timestamps.
    On eviction, always prefer expired entries before evicting the true LRU.

Complexity:
    All operations: O(1) amortized (with lazy expiry)
    Space: O(capacity)
"""

import time
from collections import OrderedDict
from typing import Optional


_MISSING = object()  # sentinel for cache miss


# ---------------------------------------------------------------------------
# Checkpoint 1 + 2: Basic LRU Cache (no TTL)
# ---------------------------------------------------------------------------

class LRUCache:
    """LRU cache with O(1) get and put, evicting least recently used entry.

    Args:
        capacity: Maximum number of key-value pairs to store.

    Raises:
        ValueError: If capacity < 1.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._cache: OrderedDict[int, int] = OrderedDict()

    def get(self, key: int) -> int:
        """Return value for key, or -1 if missing.

        Args:
            key: Cache key.

        Returns:
            Cached value or -1.

        Complexity:
            Time:  O(1)
            Space: O(1)
        """
        if key not in self._cache:
            return -1
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: int, value: int) -> None:
        """Insert or update key-value pair. Evicts LRU if at capacity.

        Args:
            key:   Cache key.
            value: Value to store.

        Complexity:
            Time:  O(1)
            Space: O(1)
        """
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
            return
        if len(self._cache) >= self._capacity:
            self._cache.popitem(last=False)  # evict LRU (oldest)
        self._cache[key] = value

    def __len__(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Checkpoint 3: LRU Cache with TTL
# ---------------------------------------------------------------------------

class LRUCacheWithTTL:
    """LRU cache where each entry expires after ttl_seconds.

    Expiry is checked lazily: expired entries are removed on access or put.
    Expired entries are treated as missing (get returns -1).

    Args:
        capacity:    Maximum number of live (non-expired) entries.
        ttl_seconds: Seconds before an entry expires. None = no expiry.

    Raises:
        ValueError: If capacity < 1 or ttl_seconds <= 0.
    """

    def __init__(self, capacity: int, ttl_seconds: Optional[float] = None) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be > 0, got {ttl_seconds}")
        self._capacity = capacity
        self._ttl = ttl_seconds
        # Maps key -> (value, expiry_time_or_None)
        self._cache: OrderedDict[int, tuple[int, Optional[float]]] = OrderedDict()

    def _is_expired(self, expiry: Optional[float]) -> bool:
        """Check if an entry with given expiry timestamp is expired.

        Args:
            expiry: Monotonic expiry time, or None if no expiry.

        Returns:
            True if the entry should be treated as missing.
        """
        if expiry is None:
            return False
        return time.monotonic() >= expiry

    def get(self, key: int) -> int:
        """Return value for key, -1 if missing or expired.

        Complexity:
            Time:  O(1)
            Space: O(1)
        """
        if key not in self._cache:
            return -1
        value, expiry = self._cache[key]
        if self._is_expired(expiry):
            del self._cache[key]
            return -1
        self._cache.move_to_end(key)
        return value

    def put(self, key: int, value: int) -> None:
        """Insert or update key with TTL. Evicts LRU when at capacity.

        Expired entries are removed first before evicting the true LRU.

        Args:
            key:   Cache key.
            value: Value to store.

        Complexity:
            Time:  O(1) amortized
            Space: O(1)
        """
        expiry = time.monotonic() + self._ttl if self._ttl is not None else None

        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = (value, expiry)
            return

        # Evict expired entries first (lazy cleanup)
        expired_keys = [k for k, (_, exp) in self._cache.items() if self._is_expired(exp)]
        for k in expired_keys:
            del self._cache[k]

        # Evict LRU if still at capacity after expiry cleanup
        if len(self._cache) >= self._capacity:
            self._cache.popitem(last=False)

        self._cache[key] = (value, expiry)

    def __len__(self) -> int:
        """Count of entries including potentially expired ones (lazy eviction)."""
        return len(self._cache)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    # Checkpoint 1+2: basic LRU
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1       # key 1 accessed -> MRU
    cache.put(3, 3)                # evicts key 2 (LRU)
    assert cache.get(2) == -1      # evicted
    assert cache.get(3) == 3

    # Update existing key: no eviction
    cache2 = LRUCache(1)
    cache2.put(1, 10)
    cache2.put(1, 20)              # update, not eviction
    assert cache2.get(1) == 20

    # Capacity 1: every new key evicts previous
    cache3 = LRUCache(1)
    cache3.put(1, 1)
    cache3.put(2, 2)               # evicts 1
    assert cache3.get(1) == -1
    assert cache3.get(2) == 2

    # Checkpoint 3: TTL
    ttl_cache = LRUCacheWithTTL(capacity=3, ttl_seconds=0.1)
    ttl_cache.put(1, 100)
    ttl_cache.put(2, 200)
    assert ttl_cache.get(1) == 100   # alive
    assert ttl_cache.get(2) == 200

    time.sleep(0.15)  # wait for TTL expiry
    assert ttl_cache.get(1) == -1    # expired
    assert ttl_cache.get(2) == -1    # expired

    # After expiry, new entries can be added up to capacity
    ttl_cache.put(3, 300)
    assert ttl_cache.get(3) == 300

    # No TTL (None) — entries persist
    no_ttl = LRUCacheWithTTL(capacity=2, ttl_seconds=None)
    no_ttl.put(1, 1)
    time.sleep(0.05)
    assert no_ttl.get(1) == 1       # should not expire

    # LRU eviction still works with TTL
    ttl_cache2 = LRUCacheWithTTL(capacity=2, ttl_seconds=60)
    ttl_cache2.put(1, 1)
    ttl_cache2.put(2, 2)
    ttl_cache2.get(1)               # key 1 becomes MRU
    ttl_cache2.put(3, 3)            # evicts key 2 (LRU)
    assert ttl_cache2.get(2) == -1
    assert ttl_cache2.get(1) == 1

    print("  LRUCache + LRUCacheWithTTL: all tests passed")


if __name__ == "__main__":
    _test()
