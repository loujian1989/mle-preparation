# 33. Search in Rotated Sorted Array (Medium)
#
# Given a rotated sorted array of distinct integers, return the index of
# target in O(log n), or -1 if not found.
#
# Approach: Modified binary search — determine which half is sorted,
#           then decide which half target must be in.
#
# --------------------------------------------------------------------------
# Key insight
#
#   A rotated sorted array always has one sorted half and one unsorted half
#   at any mid point. Example: [4,5,6,7,0,1,2]
#
#     mid=3 (value 7):
#       left half  [4,5,6,7] — sorted  ✓
#       right half [7,0,1,2] — unsorted ✗
#
#   Strategy:
#     1. Find which half is sorted (one always is).
#     2. Check if target falls in the sorted half's range.
#     3. If yes → search that half. If no → search the other half.
#
# --------------------------------------------------------------------------
# Redundancies in the naive version
#
#   Naive condition: nums[mid] >= nums[0] AND nums[mid] >= nums[-1]
#
#   Redundancy 1 — `and nums[mid] >= nums[-1]`:
#     In a rotated array nums[-1] < nums[0] always.
#     So nums[mid] >= nums[0] already implies nums[mid] >= nums[-1].
#     The second condition adds nothing → remove it.
#
#     In the non-rotated case (nums[-1] >= nums[0]), the stricter combined
#     condition changes which branch is taken for some mid values, but the
#     algorithm still produces correct results either way.
#
#   Redundancy 2 — `if not nums: return -1`:
#     Constraint guarantees 1 <= nums.length. Guard is never triggered.
#
# --------------------------------------------------------------------------
# Two correct formulations
#
#   Formulation A — fixed endpoints (nums[0], nums[-1]):
#     if nums[mid] >= nums[0]:   # mid in left (upper) sorted portion
#
#   Formulation B — shrinking window (nums[l], nums[r]):  ← standard
#     if nums[l] <= nums[mid]:   # left half is sorted
#
#   Formulation B is preferred in interviews: l and r shrink with the
#   window, no dependency on fixed array endpoints.
#
# --------------------------------------------------------------------------
# Complexity:
#   Time:  O(log n)
#   Space: O(1)
# --------------------------------------------------------------------------


from typing import List


# ---------------------------------------------------------------------------
# Formulation A: fixed endpoints (cleaned up)
# ---------------------------------------------------------------------------

def search_fixed(nums: List[int], target: int) -> int:
    """
    Use nums[0] to determine if mid is in the left or right sorted portion.

    nums[mid] >= nums[0] → mid is in the left (upper) portion.
    nums[mid] <  nums[0] → mid is in the right (lower) portion.

    Trace: nums=[4,5,6,7,0,1,2], target=0
      l=0,r=6, mid=3, nums[3]=7 >= nums[0]=4 → left portion
        target=0 in [4..7)? NO → l=4
      l=4,r=6, mid=5, nums[5]=1 < nums[0]=4 → right portion
        target=0 in (1..2]? NO → r=4
      l=4,r=4, mid=4, nums[4]=0 == target → return 4 ✓

    Complexity:
        Time:  O(log n)
        Space: O(1)
    """
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = l + (r - l) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] >= nums[0]:          # mid in left (upper) portion
            if nums[0] <= target < nums[mid]:
                r = mid - 1
            else:
                l = mid + 1
        else:                             # mid in right (lower) portion
            if nums[mid] < target <= nums[-1]:
                l = mid + 1
            else:
                r = mid - 1
    return -1


# ---------------------------------------------------------------------------
# Formulation B: shrinking window — standard interview answer
# ---------------------------------------------------------------------------

def search(nums: List[int], target: int) -> int:
    """
    Check if the LEFT half [l..mid] is sorted, then decide which half
    the target must be in.

    Why nums[l] <= nums[mid] detects a sorted left half:
      If no rotation point falls in [l..mid], nums[l] <= nums[mid].
      If a rotation point falls in [l..mid], nums[l] > nums[mid].

    Trace: nums=[4,5,6,7,0,1,2], target=0
      l=0,r=6, mid=3: nums[0]=4 <= nums[3]=7 → left sorted [4..7]
        0 in [4,7)? NO → l=4
      l=4,r=6, mid=5: nums[4]=0 > nums[5]=1? NO → left sorted [0..1]
        Wait: nums[l=4]=0 <= nums[mid=5]=1 → left sorted
        0 in [0,1)? YES (0 >= 0 and 0 < 1) → r=4
      l=4,r=4, mid=4: nums[4]=0 == target → return 4 ✓

    Complexity:
        Time:  O(log n)
        Space: O(1)
    """
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = l + (r - l) // 2
        if nums[mid] == target:
            return mid
        if nums[l] <= nums[mid]:              # left half [l..mid] is sorted
            if nums[l] <= target < nums[mid]:
                r = mid - 1
            else:
                l = mid + 1
        else:                                 # right half [mid..r] is sorted
            if nums[mid] < target <= nums[r]:
                l = mid + 1
            else:
                r = mid - 1
    return -1


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all() -> None:
    cases = [
        ([4, 5, 6, 7, 0, 1, 2], 0,  4),
        ([4, 5, 6, 7, 0, 1, 2], 3, -1),
        ([1],                   0, -1),
        ([1],                   1,  0),
        ([1, 3],                3,  1),
        ([1, 3],                0, -1),
        ([3, 1],                1,  1),
        ([3, 1],                3,  0),
        ([5, 1, 3],             3,  2),
        ([0, 1, 2, 3, 4, 5],   4,  4),   # not rotated
        ([2, 3, 4, 5, 0, 1],   0,  4),
    ]
    for nums, target, expected in cases:
        for fn in (search_fixed, search):
            result = fn(nums, target)
            assert result == expected, \
                f"{fn.__name__}({nums}, {target}) = {result}, expected {expected}"


if __name__ == "__main__":
    test_all()
    print("All tests passed")
