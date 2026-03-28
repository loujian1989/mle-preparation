"""
Find Minimum in Rotated Sorted Array (LeetCode 153 + 154) — Medium / Hard
==========================================================================

Problem A (LC 153): No duplicates. Find the minimum element.
Problem B (LC 154): May have duplicates. Find the minimum element.
Bonus:   Find the rotation pivot index.

Key insight:
    The minimum element is at the pivot (rotation point).
    In a rotated sorted array, the minimum is in the unsorted half.
    If nums[mid] > nums[right]: minimum is in right half.
    Else: minimum is in left half (including mid).

Edge cases:
    - Not rotated: minimum is nums[0]
    - Single element
    - Two elements

Complexity:
    LC 153: Time O(log N), Space O(1)
    LC 154: Time O(log N) average, O(N) worst (all duplicates), Space O(1)
"""

from typing import List


def find_min(nums: List[int]) -> int:
    """Find minimum in rotated sorted array (no duplicates).

    Args:
        nums: Non-empty rotated sorted array without duplicates.

    Returns:
        Minimum element.

    Complexity:
        Time:  O(log N)
        Space: O(1)
    """
    left, right = 0, len(nums) - 1

    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[right]:
            left = mid + 1   # min is in right half
        else:
            right = mid      # min is in left half (mid could be min)

    return nums[left]


def find_min_with_duplicates(nums: List[int]) -> int:
    """Find minimum in rotated sorted array (may have duplicates).

    Args:
        nums: Non-empty rotated sorted array, may contain duplicates.

    Returns:
        Minimum element.

    Complexity:
        Time:  O(log N) average, O(N) worst case
        Space: O(1)
    """
    left, right = 0, len(nums) - 1

    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[right]:
            left = mid + 1
        elif nums[mid] < nums[right]:
            right = mid
        else:
            # nums[mid] == nums[right]: can't determine → shrink right
            right -= 1

    return nums[left]


def find_rotation_index(nums: List[int]) -> int:
    """Find the index of the minimum element (rotation pivot).

    Useful for ML pipeline: e.g., finding the transition point in a
    time-series where metric rolls over.

    Args:
        nums: Non-empty rotated sorted array without duplicates.

    Returns:
        Index of the minimum element.

    Complexity:
        Time:  O(log N)
        Space: O(1)
    """
    left, right = 0, len(nums) - 1

    if nums[left] <= nums[right]:
        return 0   # not rotated

    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid

    return left


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    # LC 153
    assert find_min([3, 4, 5, 1, 2]) == 1
    assert find_min([4, 5, 6, 7, 0, 1, 2]) == 0
    assert find_min([11, 13, 15, 17]) == 11    # not rotated
    assert find_min([2, 1]) == 1
    assert find_min([1]) == 1

    # LC 154
    assert find_min_with_duplicates([2, 2, 2, 0, 1]) == 0
    assert find_min_with_duplicates([1, 1, 1, 1]) == 1
    assert find_min_with_duplicates([3, 1, 3]) == 1

    # Rotation index
    assert find_rotation_index([4, 5, 6, 7, 0, 1, 2]) == 4
    assert find_rotation_index([1, 2, 3, 4]) == 0   # not rotated

    print("  find_min_rotated: all tests passed")


if __name__ == "__main__":
    _test()
