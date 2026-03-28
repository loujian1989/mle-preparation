"""
Two Sum & Variants (LeetCode 1, 15, 18) — Easy / Medium
=========================================================

Problem A (LC 1): Two Sum
    Given nums and target, return indices of two numbers that sum to target.
    Assume exactly one solution.

Problem B (LC 15): 3Sum
    Find all unique triplets summing to zero.
    No duplicates in output.

Problem C (LC 18): 4Sum
    Find all unique quadruplets summing to target.

Edge cases:
    - Two Sum: same index used twice (not allowed), duplicates in array
    - 3Sum: skip duplicates to avoid repeated triplets
    - 4Sum: generalizes to kSum recursively

Approach:
    Two Sum: HashMap for O(N) lookup.
    3Sum/4Sum: Sort + two-pointer to reduce to O(N^2) / O(N^3).

Complexity:
    Two Sum:   Time O(N),    Space O(N)
    3Sum:      Time O(N^2),  Space O(1) (ignoring output)
    4Sum:      Time O(N^3),  Space O(1)
"""

from typing import List


# ---------------------------------------------------------------------------
# Problem A: Two Sum
# ---------------------------------------------------------------------------

def two_sum(nums: List[int], target: int) -> List[int]:
    """Find indices of two numbers summing to target.

    Args:
        nums:   List of integers.
        target: Target sum.

    Returns:
        [i, j] where nums[i] + nums[j] == target and i != j.

    Raises:
        ValueError: If no valid pair exists.

    Complexity:
        Time:  O(N)
        Space: O(N)
    """
    seen: dict = {}  # value → index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    raise ValueError(f"No two sum solution for target={target}")


# ---------------------------------------------------------------------------
# Problem B: 3Sum
# ---------------------------------------------------------------------------

def three_sum(nums: List[int]) -> List[List[int]]:
    """Find all unique triplets summing to zero.

    Args:
        nums: List of integers.

    Returns:
        List of unique triplets [a, b, c] where a + b + c == 0.

    Complexity:
        Time:  O(N^2)
        Space: O(1) ignoring output
    """
    nums.sort()
    result: List[List[int]] = []
    n = len(nums)

    for i in range(n - 2):
        # Skip duplicates for the first element
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        # Early termination: smallest element > 0 → no zero-sum possible
        if nums[i] > 0:
            break

        left, right = i + 1, n - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                # Skip duplicates for left and right
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1

    return result


# ---------------------------------------------------------------------------
# Problem C: 4Sum (generalizes to kSum)
# ---------------------------------------------------------------------------

def four_sum(nums: List[int], target: int) -> List[List[int]]:
    """Find all unique quadruplets summing to target.

    Args:
        nums:   List of integers.
        target: Target sum.

    Returns:
        List of unique quadruplets.

    Complexity:
        Time:  O(N^3)
        Space: O(1) ignoring output
    """
    nums.sort()
    return _k_sum(nums, target, k=4, start=0)


def _k_sum(nums: List[int], target: int, k: int, start: int) -> List[List[int]]:
    """Recursive kSum helper.

    Args:
        nums:   Sorted list.
        target: Remaining target.
        k:      Number of elements to find.
        start:  Starting index for this recursion level.

    Returns:
        All unique k-tuples summing to target.
    """
    result: List[List[int]] = []
    if start >= len(nums):
        return result

    # Base case: two-pointer
    if k == 2:
        left, right = start, len(nums) - 1
        while left < right:
            s = nums[left] + nums[right]
            if s == target:
                result.append([nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif s < target:
                left += 1
            else:
                right -= 1
        return result

    # Recursive case: fix one element, reduce to (k-1)Sum
    for i in range(start, len(nums) - k + 1):
        if i > start and nums[i] == nums[i - 1]:
            continue
        for subset in _k_sum(nums, target - nums[i], k - 1, i + 1):
            result.append([nums[i]] + subset)

    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    # Two Sum
    assert sorted(two_sum([2, 7, 11, 15], 9)) == [0, 1]
    assert sorted(two_sum([3, 2, 4], 6)) == [1, 2]

    # 3Sum
    result_3 = three_sum([-1, 0, 1, 2, -1, -4])
    assert sorted([sorted(t) for t in result_3]) == [[-1, -1, 2], [-1, 0, 1]]
    assert three_sum([0, 0, 0]) == [[0, 0, 0]]
    assert three_sum([1, 2, 3]) == []   # no zero-sum triplet

    # 4Sum
    result_4 = four_sum([1, 0, -1, 0, -2, 2], 0)
    assert sorted([sorted(q) for q in result_4]) == [[-2, -1, 1, 2], [-2, 0, 0, 2], [-1, 0, 0, 1]]

    print("  two_sum / 3sum / 4sum: all tests passed")


if __name__ == "__main__":
    _test()
