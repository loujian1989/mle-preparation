"""
Search in Rotated Sorted Array (LeetCode 33 + 81) — Medium / Medium
====================================================================

Problem A (LC 33): No duplicates. Find target index or -1.
Problem B (LC 81): May have duplicates. Return True/False.

Key insight for binary search on rotated arrays:
    After rotation, at least one half is always sorted.
    Check which half is sorted, then determine if target falls in that half.

Edge cases:
    - Not rotated (normal sorted array): works as regular binary search
    - All same elements (LC 81): worst case O(N) due to ambiguity
    - Target at pivot
    - Single element

Approach:
    Left half is sorted if nums[left] <= nums[mid].
    Right half is sorted if nums[mid] <= nums[right].
    Only one half is sorted after rotation; search within it.

Complexity:
    LC 33: Time O(log N), Space O(1)
    LC 81: Time O(log N) average, O(N) worst (duplicates), Space O(1)
"""

from typing import List


def search(nums: List[int], target: int) -> int:
    """Search in rotated sorted array (no duplicates).

    Args:
        nums:   Integer array, rotated at some pivot, no duplicates.
        target: Value to search for.

    Returns:
        Index of target, or -1 if not found.

    Complexity:
        Time:  O(log N)
        Space: O(1)
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid

        # Left half is sorted
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1     # target in sorted left half
            else:
                left = mid + 1      # target in right half
        # Right half is sorted
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1      # target in sorted right half
            else:
                right = mid - 1     # target in left half

    return -1


def search_with_duplicates(nums: List[int], target: int) -> bool:
    """Search in rotated sorted array (may have duplicates).

    Args:
        nums:   Integer array, rotated, may contain duplicates.
        target: Value to search for.

    Returns:
        True if target is in nums.

    Complexity:
        Time:  O(log N) average, O(N) worst case (all duplicates)
        Space: O(1)
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return True

        # Ambiguous: nums[left] == nums[mid] == nums[right]
        # Can't determine which half is sorted → shrink both sides
        if nums[left] == nums[mid] == nums[right]:
            left += 1
            right -= 1
        elif nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    # LC 33
    assert search([4, 5, 6, 7, 0, 1, 2], 0) == 4
    assert search([4, 5, 6, 7, 0, 1, 2], 3) == -1
    assert search([1], 0) == -1
    assert search([1], 1) == 0
    assert search([1, 3], 3) == 1          # not rotated effectively

    # LC 81
    assert search_with_duplicates([2, 5, 6, 0, 0, 1, 2], 0) is True
    assert search_with_duplicates([2, 5, 6, 0, 0, 1, 2], 3) is False
    assert search_with_duplicates([1, 1, 1, 1, 1], 1) is True
    assert search_with_duplicates([1, 1, 1, 1, 1], 2) is False

    print("  search_rotated_array: all tests passed")


if __name__ == "__main__":
    _test()
