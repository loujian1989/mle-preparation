"""
LFU Cache — Least Frequently Used (LeetCode 460) — Hard
========================================================

Problem:
    Design LFU cache: get(key) and put(key, value) in O(1).
    Evict the LEAST FREQUENTLY used item. Break ties by LRU (least recently used).

    Unlike LRU which only tracks recency, LFU tracks usage frequency.
    Item with lowest access count is evicted; ties broken by recency.

Edge cases:
    - Capacity = 1: every put evicts previous (unless same key)
    - put on existing key: update value + increment frequency
    - get on missing key: return -1
    - Frequency tie: evict LRU among tied items

Design:
    Three data structures:
    1. key_map:   key → (value, frequency)
    2. freq_map:  frequency → OrderedDict[key → None]  (ordered by LRU within freq)
    3. min_freq:  current minimum frequency

    On access (get or put on existing):
        - Move key from freq_map[old_freq] to freq_map[old_freq + 1]
        - Update min_freq if old bucket is now empty

    On evict:
        - Pop from front of freq_map[min_freq] (LRU among lowest freq)

Complexity:
    Time:  O(1) get, O(1) put
    Space: O(capacity)
"""

from collections import OrderedDict, defaultdict
from typing import Optional


class LFUCache:
    """LFU cache with O(1) get/put.

    Args:
        capacity: Maximum number of entries. Must be >= 1.

    Raises:
        ValueError: If capacity < 1.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity

        # key → [value, frequency]
        self._key_map: dict = {}

        # frequency → OrderedDict {key: None} (oldest first = LRU among this freq)
        self._freq_map: dict = defaultdict(OrderedDict)

        # Track minimum frequency for O(1) eviction
        self._min_freq = 0

    def get(self, key: int) -> int:
        """Return value for key, or -1 if not present.

        Args:
            key: Cache key.

        Returns:
            Cached value or -1.

        Complexity:
            Time:  O(1)
            Space: O(1)
        """
        if key not in self._key_map:
            return -1
        self._increment_freq(key)
        return self._key_map[key][0]

    def put(self, key: int, value: int) -> None:
        """Insert or update key. Evicts LFU (LRU tie-break) if at capacity.

        Args:
            key:   Cache key.
            value: Value to store.

        Complexity:
            Time:  O(1)
            Space: O(1)
        """
        if key in self._key_map:
            self._key_map[key][0] = value
            self._increment_freq(key)
            return

        # New key: evict if needed
        if len(self._key_map) >= self._capacity:
            self._evict()

        self._key_map[key] = [value, 1]
        self._freq_map[1][key] = None
        self._min_freq = 1

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _increment_freq(self, key: int) -> None:
        """Move key to the next frequency bucket."""
        val, freq = self._key_map[key]

        # Remove from current frequency bucket
        del self._freq_map[freq][key]
        if not self._freq_map[freq]:
            del self._freq_map[freq]
            if self._min_freq == freq:
                self._min_freq += 1

        # Add to next frequency bucket
        new_freq = freq + 1
        self._key_map[key][1] = new_freq
        self._freq_map[new_freq][key] = None

    def _evict(self) -> None:
        """Evict the LRU item among items with minimum frequency."""
        # OrderedDict.popitem(last=False) removes the LRU (oldest) item
        evicted_key, _ = self._freq_map[self._min_freq].popitem(last=False)
        if not self._freq_map[self._min_freq]:
            del self._freq_map[self._min_freq]
        del self._key_map[evicted_key]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    # LeetCode example
    cache = LFUCache(2)
    cache.put(1, 1)                 # freq: {1:1, }
    cache.put(2, 2)                 # freq: {1:1, 2:1}
    assert cache.get(1) == 1       # key 1 freq → 2
    cache.put(3, 3)                 # evict key 2 (LFU, LRU tie)
    assert cache.get(2) == -1      # evicted
    assert cache.get(3) == 3       # key 3 freq → 2
    cache.put(4, 4)                 # evict key 1 (freq 2) vs key 3 (freq 2) → evict 1 (LRU)
    assert cache.get(1) == -1      # evicted
    assert cache.get(3) == 3       # still there (accessed more recently)
    assert cache.get(4) == 4

    # Capacity 1
    cache2 = LFUCache(1)
    cache2.put(1, 10)
    cache2.put(2, 20)               # evicts 1
    assert cache2.get(1) == -1
    assert cache2.get(2) == 20

    # Update existing key
    cache3 = LFUCache(2)
    cache3.put(1, 1)
    cache3.put(1, 2)               # update, not eviction
    assert cache3.get(1) == 2

    print("  LFUCache: all tests passed")


if __name__ == "__main__":
    _test()
