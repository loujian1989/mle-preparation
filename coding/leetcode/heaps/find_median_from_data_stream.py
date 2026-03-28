"""
Find Median from Data Stream (LeetCode 295) — Hard
===================================================

Problem:
    Design a data structure that supports:
    - addNum(num): add a number to the stream
    - findMedian(): return the median of all added numbers

Edge cases:
    - Single element → return that element
    - Even number of elements → median = average of two middles
    - Duplicate values

Approach — Two Heaps:
    - max_heap (left half): max-heap of smaller half  [Python: negate values]
    - min_heap (right half): min-heap of larger half

    Invariant:
        1. len(max_heap) == len(min_heap) or len(max_heap) == len(min_heap) + 1
        2. max(max_heap) <= min(min_heap)

    addNum:
        Push to max_heap, move max to min_heap, rebalance if needed.

    findMedian:
        If equal sizes: avg of both tops.
        If max_heap larger: top of max_heap.

    ML use case: rolling median of latency, batch loss, score distributions.

Complexity:
    addNum:    Time O(log N), Space O(1) amortized
    findMedian: Time O(1)
    Space: O(N)
"""

import heapq


class MedianFinder:
    """Online median tracking using two heaps.

    Examples:
        >>> mf = MedianFinder()
        >>> mf.addNum(1); mf.addNum(2); mf.findMedian()
        1.5
        >>> mf.addNum(3); mf.findMedian()
        2.0
    """

    def __init__(self) -> None:
        # max_heap: left (smaller) half — Python uses negated values
        self._max_heap: list = []
        # min_heap: right (larger) half
        self._min_heap: list = []

    def addNum(self, num: int) -> None:
        """Add a number to the stream.

        Args:
            num: Integer to add.

        Complexity:
            Time:  O(log N)
            Space: O(1)
        """
        # Step 1: Push to max_heap (left half)
        heapq.heappush(self._max_heap, -num)

        # Step 2: Ensure all in left <= all in right
        if (self._min_heap and
                -self._max_heap[0] > self._min_heap[0]):
            val = -heapq.heappop(self._max_heap)
            heapq.heappush(self._min_heap, val)

        # Step 3: Rebalance sizes (left can be at most 1 larger than right)
        if len(self._max_heap) > len(self._min_heap) + 1:
            val = -heapq.heappop(self._max_heap)
            heapq.heappush(self._min_heap, val)
        elif len(self._min_heap) > len(self._max_heap):
            val = heapq.heappop(self._min_heap)
            heapq.heappush(self._max_heap, -val)

    def findMedian(self) -> float:
        """Return the current median.

        Returns:
            Median as float.

        Raises:
            RuntimeError: If no numbers have been added.

        Complexity:
            Time:  O(1)
            Space: O(1)
        """
        if not self._max_heap:
            raise RuntimeError("No numbers in stream")

        if len(self._max_heap) == len(self._min_heap):
            return (-self._max_heap[0] + self._min_heap[0]) / 2.0
        return float(-self._max_heap[0])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    mf = MedianFinder()
    mf.addNum(1)
    assert mf.findMedian() == 1.0
    mf.addNum(2)
    assert mf.findMedian() == 1.5
    mf.addNum(3)
    assert mf.findMedian() == 2.0

    # All same values
    mf2 = MedianFinder()
    for _ in range(5):
        mf2.addNum(4)
    assert mf2.findMedian() == 4.0

    # Large stream
    mf3 = MedianFinder()
    for i in range(1, 101):
        mf3.addNum(i)
    assert mf3.findMedian() == 50.5   # median of 1..100

    # Error on empty
    mf_empty = MedianFinder()
    try:
        mf_empty.findMedian()
        assert False, "Should raise RuntimeError"
    except RuntimeError:
        pass

    print("  MedianFinder: all tests passed")


if __name__ == "__main__":
    _test()
