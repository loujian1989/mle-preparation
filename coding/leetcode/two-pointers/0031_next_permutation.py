# 31. Next Permutation (Medium)
#
# Rearrange nums into the next lexicographically greater permutation.
# If already the largest (fully descending), rearrange to smallest (ascending).
# Must be done in-place with O(1) extra space.
#
# Approach: Three-pass scan   O(n) T / O(1) S
#
# --------------------------------------------------------------------------
# Intuition
#
#   Think of the array as a number. To get the next larger number,
#   we want to increase it as little as possible.
#
#   Strategy:
#     1. Find the rightmost "dip" — index i where nums[i] < nums[i+1].
#        This is the leftmost place we can make the number bigger.
#     2. Swap nums[i] with the smallest element to its RIGHT that's still
#        larger than nums[i]. This makes the number just barely bigger.
#     3. Reverse everything after index i. Because the suffix was descending
#        before the swap (and remains descending after swapping a smaller
#        element in), reversing it gives the smallest possible suffix.
#
#   If no dip exists (whole array is descending), the array is the LARGEST
#   permutation → reverse the whole thing to get the smallest.
#
# --------------------------------------------------------------------------
# Step-by-step example: nums = [1, 2, 3, 6, 5, 4]
#
#   Pass 1 — find dip (right-to-left):
#     i=4: 5 > 4 (not dip)
#     i=3: 6 > 5 (not dip)
#     i=2: 3 < 6 → dip at i=2    ← nums[2]=3 is what we'll increment
#
#   Suffix [6,5,4] is descending (guaranteed by the scan stopping at i=2).
#
#   Pass 2 — find swap target (right-to-left):
#     j=5: nums[5]=4 > nums[2]=3 → swap
#     Swap nums[2] and nums[5]: [1, 2, 4, 6, 5, 3]
#     Suffix after i is still descending: [6, 5, 3]
#
#   Pass 3 — reverse suffix after i:
#     Reverse [6, 5, 3] → [3, 5, 6]
#     Result: [1, 2, 4, 3, 5, 6] ✓
#
# --------------------------------------------------------------------------
# Why the suffix is still descending after the swap
#
#   Before swap: suffix = [..., nums[j], ..., nums[i+1], nums[j], ...]
#   We found j from the RIGHT, so nums[j] is the SMALLEST value > nums[i].
#   Everything to the right of j is ≤ nums[j].
#   After swapping nums[i] ↔ nums[j]:
#     nums[i] is now nums[j] (larger) — this leaves the suffix position.
#     The old nums[i] (smaller) goes into position j.
#     Since old nums[i] < nums[j] and the suffix was descending, putting
#     a smaller value at j keeps the suffix descending.
#   → Reversing the suffix gives the smallest possible ordering.
#
# --------------------------------------------------------------------------
# Edge cases
#
#   Fully descending (e.g., [3,2,1]):
#     No dip found → reverse entire array → [1,2,3] (smallest permutation).
#
#   Single element:
#     No dip, reverse of [x] = [x] → correct.
#
#   All equal (e.g., [1,1,1]):
#     No dip (never strictly less) → reverse → [1,1,1] → correct.
#
# --------------------------------------------------------------------------
# Complexity:
#   Time:  O(n)  — three linear passes (find dip, find swap, reverse)
#   Space: O(1)  — in-place swaps
# --------------------------------------------------------------------------


from typing import List


def next_permutation(nums: List[int]) -> None:
    """
    Modify nums in-place to the next lexicographic permutation.

    Trace: nums=[1,2,3,6,5,4]
      Pass 1: dip at i=2 (nums[2]=3 < nums[3]=6)
      Pass 2: j=5 (nums[5]=4 > 3) → swap → [1,2,4,6,5,3]
      Pass 3: reverse suffix [i+1:] = [6,5,3] → [3,5,6]
      Result: [1,2,4,3,5,6] ✓

    Trace (fully descending): nums=[3,2,1]
      Pass 1: no dip (i ends at -1)
      Pass 3: reverse entire array → [1,2,3] ✓

    Complexity:
        Time:  O(n)
        Space: O(1)
    """
    n = len(nums)

    # Pass 1: find rightmost dip — first i (right-to-left) where nums[i] < nums[i+1]
    i = n - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1

    if i >= 0:
        # Pass 2: find rightmost j where nums[j] > nums[i]
        j = n - 1
        while nums[j] <= nums[i]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]

    # Pass 3: reverse suffix after i (if i == -1, reverses whole array)
    left, right = i + 1, n - 1
    while left < right:
        nums[left], nums[right] = nums[right], nums[left]
        left += 1
        right -= 1


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all() -> None:
    cases = [
        ([1, 2, 3],          [1, 3, 2]),
        ([3, 2, 1],          [1, 2, 3]),   # fully descending → smallest
        ([1, 1, 5],          [1, 5, 1]),
        ([1],                [1]),          # single element
        ([1, 2, 3, 6, 5, 4], [1, 2, 4, 3, 5, 6]),
        ([1, 3, 2],          [2, 1, 3]),
        ([2, 3, 1],          [3, 1, 2]),
        ([1, 1, 1],          [1, 1, 1]),   # all equal → no change
        ([1, 5, 1],          [5, 1, 1]),
    ]
    for nums, expected in cases:
        nums_copy = nums[:]
        next_permutation(nums_copy)
        assert nums_copy == expected, \
            f"next_permutation({nums}) = {nums_copy}, expected {expected}"


if __name__ == "__main__":
    test_all()
    print("All tests passed")
