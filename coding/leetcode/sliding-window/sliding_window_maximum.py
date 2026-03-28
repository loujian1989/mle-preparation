"""
Sliding Window Maximum (LeetCode 239) — Hard
=============================================

Problem:
    Given an array nums and window size k, return the max of each window
    as it slides from left to right.

    Output length: len(nums) - k + 1

Edge cases:
    - k == 1: output is nums itself
    - k == len(nums): output is [max(nums)]
    - All equal: each window max equals that value

Approach — Monotonic Deque:
    Maintain a deque of indices in decreasing order of their values.
    - On each step: pop from back while nums[back] <= nums[right] (smaller → useless)
    - Pop from front if index is outside the window (< right - k + 1)
    - Front of deque = max of current window

    Intuition: deque is always non-increasing; front = max.

    Common ML use: rolling maximum of a metric over a sliding time window.

Complexity:
    Time:  O(N) — each element added and removed from deque at most once
    Space: O(k) — deque holds at most k indices
"""

from collections import deque
from typing import List


def max_sliding_window(nums: List[int], k: int) -> List[int]:
    """Return the maximum of each size-k sliding window.

    Args:
        nums: Input array.
        k:    Window size (1 <= k <= len(nums)).

    Returns:
        List of window maximums, length len(nums) - k + 1.

    Raises:
        ValueError: If k < 1 or k > len(nums).

    Complexity:
        Time:  O(N)
        Space: O(k)
    """
    if not nums:
        return []
    if k < 1 or k > len(nums):
        raise ValueError(f"k must be in [1, {len(nums)}], got {k}")

    dq: deque = deque()  # stores indices; front = index of current window max
    result: List[int] = []

    for i, val in enumerate(nums):
        # Remove indices outside the current window
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # Remove indices whose values are <= current (they can never be max)
        while dq and nums[dq[-1]] <= val:
            dq.pop()

        dq.append(i)

        # Window is full: record max (front of deque)
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result


# ---------------------------------------------------------------------------
# Rolling metrics variant (ML use case: rolling max latency, rolling max error)
# ---------------------------------------------------------------------------

def rolling_max(values: List[float], window: int) -> List[float]:
    """Compute rolling maximum over a list of floats.

    Args:
        values: Time-series values (e.g., latency measurements).
        window: Rolling window size.

    Returns:
        Rolling max values (same length as input, NaN for first window-1 entries).

    Complexity:
        Time:  O(N)
        Space: O(window)
    """
    if window <= 0:
        raise ValueError(f"window must be >= 1, got {window}")

    result = [float("nan")] * len(values)
    dq: deque = deque()

    for i, val in enumerate(values):
        while dq and dq[0] < i - window + 1:
            dq.popleft()
        while dq and values[dq[-1]] <= val:
            dq.pop()
        dq.append(i)
        if i >= window - 1:
            result[i] = values[dq[0]]

    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    assert max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
    assert max_sliding_window([1], 1) == [1]
    assert max_sliding_window([1, -1], 1) == [1, -1]
    assert max_sliding_window([4, 3, 2, 1], 2) == [4, 3, 2]   # decreasing
    assert max_sliding_window([1, 2, 3, 4], 4) == [4]

    # Rolling max
    rm = rolling_max([1.0, 3.0, 2.0, 5.0, 4.0], 2)
    assert rm[1] == 3.0
    assert rm[3] == 5.0

    print("  sliding_window_maximum: all tests passed")


if __name__ == "__main__":
    _test()
