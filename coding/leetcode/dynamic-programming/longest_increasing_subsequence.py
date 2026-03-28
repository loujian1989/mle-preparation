"""
Longest Increasing Subsequence (LeetCode 300) — Medium
=======================================================

Problem:
    Given an integer array nums, return the length of the longest strictly
    increasing subsequence.

Follow-ups:
    A. Length only — O(N log N) with patience sorting
    B. Reconstruct the actual subsequence

Edge cases:
    - Single element → 1
    - All equal → 1 (strictly increasing, not non-decreasing)
    - Already sorted → N
    - Reverse sorted → 1

Approach A (O(N^2) DP):
    dp[i] = LIS ending at index i.
    Transition: dp[i] = max(dp[j] + 1) for all j < i where nums[j] < nums[i].

Approach B (O(N log N) patience sorting):
    Maintain a "tails" array where tails[k] = smallest tail of all increasing
    subsequences of length k+1. Use binary search to update.
    Note: tails array is NOT the actual LIS — just used to track length.

Complexity:
    DP:              Time O(N^2), Space O(N)
    Patience sorting: Time O(N log N), Space O(N)
"""

import bisect
from typing import List


# ---------------------------------------------------------------------------
# Approach A: O(N^2) DP — easier to explain in interview
# ---------------------------------------------------------------------------

def length_of_lis_dp(nums: List[int]) -> int:
    """LIS length via O(N^2) DP.

    Args:
        nums: List of integers.

    Returns:
        Length of longest strictly increasing subsequence.

    Complexity:
        Time:  O(N^2)
        Space: O(N)
    """
    if not nums:
        return 0

    n = len(nums)
    dp = [1] * n

    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)


# ---------------------------------------------------------------------------
# Approach B: O(N log N) patience sorting
# ---------------------------------------------------------------------------

def length_of_lis(nums: List[int]) -> int:
    """LIS length via patience sorting / binary search.

    Maintains tails[k] = smallest ending value of any IS of length k+1.
    Binary search to find where current num fits.

    Args:
        nums: List of integers.

    Returns:
        Length of longest strictly increasing subsequence.

    Complexity:
        Time:  O(N log N)
        Space: O(N)
    """
    tails: List[int] = []

    for num in nums:
        # Find first tail >= num (we want strictly increasing → use bisect_left)
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)   # extend LIS
        else:
            tails[pos] = num    # replace with smaller tail (optimistic)

    return len(tails)


# ---------------------------------------------------------------------------
# Reconstruct actual LIS (O(N^2) DP with backtracking)
# ---------------------------------------------------------------------------

def reconstruct_lis(nums: List[int]) -> List[int]:
    """Return one valid LIS (not just length).

    Args:
        nums: List of integers.

    Returns:
        One valid longest increasing subsequence.

    Complexity:
        Time:  O(N^2)
        Space: O(N)
    """
    if not nums:
        return []

    n = len(nums)
    dp = [1] * n
    parent = [-1] * n

    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j

    # Find end of LIS
    max_len = max(dp)
    idx = dp.index(max_len)

    # Backtrack
    path = []
    while idx != -1:
        path.append(nums[idx])
        idx = parent[idx]

    return path[::-1]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    assert length_of_lis([10, 9, 2, 5, 3, 7, 101, 18]) == 4  # [2,3,7,101]
    assert length_of_lis([0, 1, 0, 3, 2, 3]) == 4
    assert length_of_lis([7, 7, 7, 7]) == 1                   # strictly increasing
    assert length_of_lis([1]) == 1
    assert length_of_lis([]) == 0

    # DP variant should match
    assert length_of_lis_dp([10, 9, 2, 5, 3, 7, 101, 18]) == 4

    # Reconstruction
    lis = reconstruct_lis([10, 9, 2, 5, 3, 7, 101, 18])
    assert len(lis) == 4
    assert all(lis[i] < lis[i + 1] for i in range(len(lis) - 1)), "LIS must be strictly increasing"

    print("  LIS: all tests passed")


if __name__ == "__main__":
    _test()
