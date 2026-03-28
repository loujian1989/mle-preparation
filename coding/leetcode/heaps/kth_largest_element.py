"""
Kth Largest Element in Array (LeetCode 215) — Medium
=====================================================

Problem:
    Find the kth largest element in an unsorted array.
    Note: "kth largest" means kth position when sorted in descending order.

Variants covered:
    A. kth largest in array — O(N log k) with min-heap
    B. k largest elements — O(N log k)
    C. kth largest in a stream (LC 703) — online algorithm with fixed-size heap
    D. Top-K frequent elements (LC 347) — bucket sort or heap

Edge cases:
    - k == 1: return max
    - k == len(nums): return min
    - Duplicate elements: ok (position-based, not value-based)

Approach A — Min-heap of size k:
    Maintain a min-heap of the k largest elements seen so far.
    Push each element; if heap size > k, pop the smallest.
    Result: heap[0] = kth largest.

    Alternative: QuickSelect — O(N) average, O(N^2) worst case.
    Min-heap chosen for stream compatibility.

Complexity:
    kth largest (array):  Time O(N log k),  Space O(k)
    kth largest (stream): Time O(log k)/query, Space O(k)
    Top-k frequent:       Time O(N log k),  Space O(N)
"""

import heapq
from collections import Counter
from typing import List


# ---------------------------------------------------------------------------
# Variant A: kth largest in array
# ---------------------------------------------------------------------------

def find_kth_largest(nums: List[int], k: int) -> int:
    """Find kth largest element using a min-heap of size k.

    Args:
        nums: Unsorted list of integers.
        k:    Rank (1 = largest).

    Returns:
        kth largest element.

    Raises:
        ValueError: If k < 1 or k > len(nums).

    Complexity:
        Time:  O(N log k)
        Space: O(k)
    """
    if k < 1 or k > len(nums):
        raise ValueError(f"k must be in [1, {len(nums)}], got {k}")

    heap: List[int] = []
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)  # remove smallest → heap keeps k largest

    return heap[0]  # smallest of k largest = kth largest


# ---------------------------------------------------------------------------
# Variant B: k largest elements (return all k)
# ---------------------------------------------------------------------------

def k_largest(nums: List[int], k: int) -> List[int]:
    """Return the k largest elements in descending order.

    Args:
        nums: Input list.
        k:    Number of elements.

    Returns:
        k largest elements, largest first.

    Complexity:
        Time:  O(N log k)
        Space: O(k)
    """
    if k >= len(nums):
        return sorted(nums, reverse=True)
    # heapq.nlargest uses a heap of size k
    return heapq.nlargest(k, nums)


# ---------------------------------------------------------------------------
# Variant C: kth largest in a stream (LeetCode 703)
# ---------------------------------------------------------------------------

class KthLargest:
    """Data structure for kth largest element in a stream.

    Args:
        k:    Rank to track.
        nums: Initial values.
    """

    def __init__(self, k: int, nums: List[int]) -> None:
        self.k = k
        self.heap: List[int] = []
        for num in nums:
            self.add(num)

    def add(self, val: int) -> int:
        """Add val to stream; return current kth largest.

        Args:
            val: New stream element.

        Returns:
            Current kth largest element.

        Complexity:
            Time:  O(log k)
            Space: O(1) amortized
        """
        heapq.heappush(self.heap, val)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
        return self.heap[0]


# ---------------------------------------------------------------------------
# Variant D: Top K Frequent Elements (LeetCode 347)
# ---------------------------------------------------------------------------

def top_k_frequent(nums: List[int], k: int) -> List[int]:
    """Return k most frequent elements (any order).

    Args:
        nums: Input list.
        k:    Number of top frequent elements.

    Returns:
        k most frequent elements.

    Complexity:
        Time:  O(N log k)
        Space: O(N)
    """
    freq = Counter(nums)
    # heapq.nlargest by frequency
    return heapq.nlargest(k, freq.keys(), key=freq.get)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    # kth largest
    assert find_kth_largest([3, 2, 1, 5, 6, 4], 2) == 5
    assert find_kth_largest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4) == 4
    assert find_kth_largest([1], 1) == 1

    # k largest
    assert k_largest([3, 1, 4, 1, 5, 9, 2, 6], 3) == [9, 6, 5]

    # Stream
    stream = KthLargest(3, [4, 5, 8, 2])
    assert stream.add(3) == 4
    assert stream.add(5) == 5
    assert stream.add(10) == 5
    assert stream.add(9) == 8
    assert stream.add(4) == 8

    # Top K frequent
    result = set(top_k_frequent([1, 1, 1, 2, 2, 3], 2))
    assert result == {1, 2}

    print("  kth_largest / top_k: all tests passed")


if __name__ == "__main__":
    _test()
