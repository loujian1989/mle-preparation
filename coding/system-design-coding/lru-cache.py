"""
LRU Cache (Least Recently Used)
================================
Problem:
    Design a data structure that supports get(key) and put(key, value) in O(1).
    Evict the least recently used item when capacity is exceeded.

Constraints:
    - 1 <= capacity <= 3000
    - 0 <= key, value <= 1e4
    - get / put called up to 2×10^5 times

Edge cases:
    - put on existing key: update value + move to front (no eviction)
    - get on missing key: return -1
    - capacity = 1: every put evicts the prior entry unless same key

Approach A — OrderedDict (clean, interview-fast):
    OrderedDict maintains insertion order; move_to_end() on access.
    Evict the first item (LRU) when over capacity.

Approach B — Doubly Linked List + HashMap (explicit, shows internals):
    DLL: O(1) removal and insertion anywhere given a node pointer.
    HashMap: key → node, O(1) lookup.
    Head = MRU, Tail = LRU. Evict from tail.

Complexity (both approaches):
    Time:  O(1) get, O(1) put
    Space: O(capacity)
"""

from collections import OrderedDict
from typing import Optional


# ---------------------------------------------------------------------------
# Approach A: OrderedDict
# ---------------------------------------------------------------------------

class LRUCache:
    """LRU cache using OrderedDict.

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
        """Return value for key, or -1 if not present.

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
        self._cache.move_to_end(key)  # mark as most recently used
        return self._cache[key]

    def put(self, key: int, value: int) -> None:
        """Insert or update key. Evicts LRU entry if at capacity.

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
            self._cache.popitem(last=False)  # evict LRU (first item)
        self._cache[key] = value


# ---------------------------------------------------------------------------
# Approach B: Doubly Linked List + HashMap  (explicit internals — shows depth)
# ---------------------------------------------------------------------------

class _Node:
    """DLL node for LRUCacheManual."""

    __slots__ = ("key", "val", "prev", "next")

    def __init__(self, key: int = 0, val: int = 0) -> None:
        self.key = key
        self.val = val
        self.prev: Optional["_Node"] = None
        self.next: Optional["_Node"] = None


class LRUCacheManual:
    """LRU cache using explicit doubly linked list + hash map.

    Layout: dummy_head <-> [MRU ... LRU] <-> dummy_tail

    Args:
        capacity: Maximum number of entries.

    Raises:
        ValueError: If capacity < 1.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._map: dict[int, _Node] = {}

        # Sentinel nodes eliminate edge-case checks for empty list
        self._head = _Node()  # dummy MRU sentinel
        self._tail = _Node()  # dummy LRU sentinel
        self._head.next = self._tail
        self._tail.prev = self._head

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(self, key: int) -> int:
        """Return value for key, or -1 if not present.

        Complexity:
            Time:  O(1)
            Space: O(1)
        """
        if key not in self._map:
            return -1
        node = self._map[key]
        self._move_to_front(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        """Insert or update key. Evicts LRU when at capacity.

        Complexity:
            Time:  O(1)
            Space: O(1)
        """
        if key in self._map:
            node = self._map[key]
            node.val = value
            self._move_to_front(node)
        else:
            node = _Node(key, value)
            self._map[key] = node
            self._insert_at_front(node)
            if len(self._map) > self._capacity:
                self._evict_lru()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _remove(self, node: _Node) -> None:
        """Unlink node from DLL."""
        node.prev.next = node.next  # type: ignore[union-attr]
        node.next.prev = node.prev  # type: ignore[union-attr]

    def _insert_at_front(self, node: _Node) -> None:
        """Insert node right after head sentinel (MRU position)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node  # type: ignore[union-attr]
        self._head.next = node

    def _move_to_front(self, node: _Node) -> None:
        """Remove node from current position and re-insert at MRU position."""
        self._remove(node)
        self._insert_at_front(node)

    def _evict_lru(self) -> None:
        """Remove the node just before tail sentinel (LRU position)."""
        lru = self._tail.prev
        if lru is self._head:
            return  # empty cache, nothing to evict
        self._remove(lru)       # type: ignore[arg-type]
        del self._map[lru.key]  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test_lru(cls: type) -> None:
    # Basic get/put
    cache = cls(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1        # key 1 accessed → MRU
    cache.put(3, 3)                 # evicts key 2 (LRU)
    assert cache.get(2) == -1       # evicted
    assert cache.get(3) == 3

    # Update existing key (no eviction)
    cache2 = cls(1)
    cache2.put(1, 10)
    cache2.put(1, 20)               # update, not eviction
    assert cache2.get(1) == 20

    # Capacity 1
    cache3 = cls(1)
    cache3.put(1, 1)
    cache3.put(2, 2)                # evicts 1
    assert cache3.get(1) == -1
    assert cache3.get(2) == 2

    print(f"  {cls.__name__}: all tests passed")


if __name__ == "__main__":
    print("LRU Cache tests")
    _test_lru(LRUCache)
    _test_lru(LRUCacheManual)
