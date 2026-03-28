"""
0/1 Knapsack & Variants
========================

Three canonical knapsack problems — foundation for interval DP and subset problems.

Problem A — 0/1 Knapsack:
    Given weights[] and values[], pick items to maximize value without exceeding capacity W.
    Each item can be chosen at most once.

Problem B — Partition Equal Subset Sum (LeetCode 416):
    Can the array be partitioned into two subsets with equal sum?
    (Special case of 0/1 knapsack where value = weight = nums[i], target = total/2)

Problem C — Target Sum (LeetCode 494):
    Assign + or - to each number; count ways to reach target.
    (Transformation to subset sum: count subsets summing to (target + total) / 2)

Complexity:
    0/1 Knapsack:    Time O(N * W),        Space O(W)
    Partition Sum:   Time O(N * total/2),  Space O(total/2)
    Target Sum:      Time O(N * S),        Space O(S) where S = (target+total)/2
"""

from typing import List


# ---------------------------------------------------------------------------
# Problem A: 0/1 Knapsack
# ---------------------------------------------------------------------------

def knapsack_01(weights: List[int], values: List[int], capacity: int) -> int:
    """0/1 Knapsack: maximize value without exceeding capacity.

    Each item chosen at most once.

    Args:
        weights:  Item weights (positive integers).
        values:   Item values (positive integers).
        capacity: Maximum weight capacity W.

    Returns:
        Maximum achievable value.

    Raises:
        ValueError: If weights and values have different lengths.

    Complexity:
        Time:  O(N * W)
        Space: O(W) — space-optimized rolling array
    """
    if len(weights) != len(values):
        raise ValueError("weights and values must have equal length")

    dp = [0] * (capacity + 1)

    for weight, value in zip(weights, values):
        # Iterate RIGHT to LEFT to ensure each item is used at most once
        for cap in range(capacity, weight - 1, -1):
            dp[cap] = max(dp[cap], dp[cap - weight] + value)

    return dp[capacity]


# ---------------------------------------------------------------------------
# Problem B: Partition Equal Subset Sum (LeetCode 416)
# ---------------------------------------------------------------------------

def can_partition(nums: List[int]) -> bool:
    """Determine if nums can be partitioned into two equal-sum subsets.

    Args:
        nums: List of positive integers.

    Returns:
        True if partition exists.

    Complexity:
        Time:  O(N * total/2)
        Space: O(total/2)
    """
    total = sum(nums)
    if total % 2 != 0:
        return False

    target = total // 2
    # dp[j] = True if subset summing to j is achievable
    dp = [False] * (target + 1)
    dp[0] = True

    for num in nums:
        for j in range(target, num - 1, -1):  # right to left: 0/1 knapsack
            dp[j] = dp[j] or dp[j - num]

    return dp[target]


# ---------------------------------------------------------------------------
# Problem C: Target Sum (LeetCode 494)
# ---------------------------------------------------------------------------

def find_target_sum_ways(nums: List[int], target: int) -> int:
    """Count ways to assign +/- to reach target sum.

    Key insight / transformation:
        Let P = set of positive items, N = set of negative items.
        sum(P) - sum(N) = target
        sum(P) + sum(N) = total
        → sum(P) = (target + total) / 2
        So: count subsets summing to (target + total) / 2.

    Args:
        nums:   List of non-negative integers.
        target: Target sum (may be negative).

    Returns:
        Number of valid +/- assignments.

    Complexity:
        Time:  O(N * S) where S = (target + total) / 2
        Space: O(S)
    """
    total = sum(nums)
    # Check feasibility
    if (total + target) % 2 != 0 or abs(target) > total:
        return 0

    subset_target = (total + target) // 2
    # dp[j] = number of subsets summing to j
    dp = [0] * (subset_target + 1)
    dp[0] = 1

    for num in nums:
        for j in range(subset_target, num - 1, -1):
            dp[j] += dp[j - num]

    return dp[subset_target]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    # 0/1 Knapsack
    assert knapsack_01([1, 3, 4, 5], [1, 4, 5, 7], 7) == 9
    assert knapsack_01([2, 3, 4, 5], [3, 4, 5, 6], 5) == 7
    assert knapsack_01([], [], 10) == 0

    # Partition Equal Subset Sum
    assert can_partition([1, 5, 11, 5]) is True
    assert can_partition([1, 2, 3, 5]) is False
    assert can_partition([1, 1]) is True
    assert can_partition([1]) is False   # odd total

    # Target Sum
    assert find_target_sum_ways([1, 1, 1, 1, 1], 3) == 5
    assert find_target_sum_ways([1], 1) == 1
    assert find_target_sum_ways([1], 2) == 0
    assert find_target_sum_ways([0, 0, 0], 0) == 8  # 2^3 ways (each 0 can be +/-)

    print("  Knapsack variants: all tests passed")


if __name__ == "__main__":
    _test()
