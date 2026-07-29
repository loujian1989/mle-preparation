# 153. Find Minimum in Rotated Sorted Array (Medium)
#
# Given a sorted array rotated 1..n times, find the minimum in O(log n).
# All elements are unique.
#
# Approaches:
#   Step 1: Standard  l<r, r=mid        — idiomatic, single return
#   Step 2: User's    l<r-1, two-candidate exit
#
# --------------------------------------------------------------------------
# Core insight: find-target vs find-min — two different paradigms
#
#   Find target  (LC 33, 34, 74):
#     You know WHAT you're looking for.
#     When nums[mid] != target, mid is provably NOT the answer → exclude it.
#     Loop: l <= r  |  moves: l=mid+1 or r=mid-1  |  exit: found or empty
#
#   Find min/max (LC 153, 162):
#     You're locating a PROPERTY (the inflection point), not a value.
#     mid could be the answer — you cannot exclude it without proof.
#     Loop: l < r   |  moves: l=mid+1 OR r=mid  |  exit: l==r → answer
#
#   The key difference is in the right-pointer move:
#     find-target:  r = mid-1  (mid is eliminated)
#     find-min:     r = mid    (mid stays as a candidate)
#
# --------------------------------------------------------------------------
# How to decide which half the minimum is in
#
#   Compare nums[mid] to nums[r] (shrinking right endpoint):
#
#     nums[mid] > nums[r]:
#       mid is in the LEFT (high-value) portion.
#       The minimum must be STRICTLY to the right of mid.
#       → l = mid + 1   (safe to exclude mid here — it's definitely not min)
#
#     nums[mid] <= nums[r]:
#       mid is in the RIGHT (low-value) portion.
#       The minimum is AT or LEFT of mid.
#       → r = mid       (keep mid as a candidate)
#
#   Why nums[r] and not nums[0]?
#     Using the shrinking right endpoint (nums[r]) is more precise — as the
#     window narrows, the reference tracks the current boundary, not the
#     fixed original endpoint. Both work, but nums[r] is the standard form.
#
# --------------------------------------------------------------------------
# Why `l < r` (not `l <= r`) for find-min
#
#   l <= r risks an infinite loop: if l==r and we always move r=mid, l and r
#   never change (mid == l == r). Stopping at l<r means when l==r we exit
#   with exactly one candidate — the minimum.
#
# --------------------------------------------------------------------------
# User's version: l < r-1 style
#
#   Stops when l and r are adjacent (differ by exactly 1).
#   At exit: two candidates remain → return min(nums[l], nums[r]).
#   Uses fixed nums[-1] as reference instead of moving nums[r].
#   Correct but requires a min() call at exit. The l<r style is cleaner.
#
# --------------------------------------------------------------------------
# Complexity:
#   Time:  O(log n)
#   Space: O(1)
# --------------------------------------------------------------------------


from typing import List


# ---------------------------------------------------------------------------
# Step 1: Standard — l < r, r = mid, return nums[l]
# ---------------------------------------------------------------------------

def find_min(nums: List[int]) -> int:
    """
    Binary search converging on the inflection point.
    At each step, determine which half the minimum lives in.
    Exit when l == r — exactly one candidate remains.

    Trace: nums=[4,5,6,7,0,1,2]
      l=0,r=6, mid=3: nums[3]=7 > nums[6]=2 → l=4
      l=4,r=6, mid=5: nums[5]=1 <= nums[6]=2 → r=5
      l=4,r=5, mid=4: nums[4]=0 <= nums[5]=1 → r=4
      l==r==4 → return nums[4]=0 ✓

    Trace (not rotated): nums=[1,2,3,4,5]
      l=0,r=4, mid=2: nums[2]=3 <= nums[4]=5 → r=2
      l=0,r=2, mid=1: nums[1]=2 <= nums[2]=3 → r=1
      l=0,r=1, mid=0: nums[0]=1 <= nums[1]=2 → r=0
      l==r==0 → return nums[0]=1 ✓

    Complexity:
        Time:  O(log n)
        Space: O(1)
    """
    l, r = 0, len(nums) - 1
    while l < r:
        mid = l + (r - l) // 2
        if nums[mid] > nums[r]:   # mid in left (high) portion → min is right of mid
            l = mid + 1
        else:                      # mid in right (low) portion → min is at or left of mid
            r = mid
    return nums[l]


# ---------------------------------------------------------------------------
# Step 2: User's version — l < r-1, two-candidate exit
# ---------------------------------------------------------------------------

def find_min_two_candidate(nums: List[int]) -> int:
    """
    Stop when l and r are adjacent. Return min of the two remaining candidates.
    Uses fixed nums[-1] as reference (equivalent to nums[r] when array unmodified,
    but stays fixed as the window shrinks — minor distinction, still correct).

    Difference from Step 1:
      l < r-1 exits with TWO candidates → need min()
      l < r   exits with ONE  candidate → just return nums[l]

    Complexity:
        Time:  O(log n)
        Space: O(1)
    """
    l, r = 0, len(nums) - 1
    while l < r - 1:
        mid = l + (r - l) // 2
        if nums[mid] >= nums[-1]:   # mid in left portion → min is right of mid
            l = mid
        else:                        # mid in right portion → min is at or left of mid
            r = mid
    return min(nums[l], nums[r])


# ---------------------------------------------------------------------------
# Paradigm comparison
#
#   Find target (l <= r, exclude mid):
#     mid proven NOT answer → safely exclude with mid±1
#     Exit: range empty (not found) or explicit return when found
#
#   Find min/max (l < r, keep mid):
#     mid might BE the answer → keep it with r=mid
#     Exit: l==r, exactly one candidate
#
#   User's two-candidate (l < r-1, keep mid):
#     Same idea but stops with two candidates, needs min() at exit
#     Equivalent correctness, slightly more code at return
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all() -> None:
    cases = [
        ([3, 4, 5, 1, 2],    1),
        ([4, 5, 6, 7, 0, 1, 2], 0),
        ([11, 13, 15, 17],   11),   # not rotated
        ([1],                 1),   # single element
        ([2, 1],              1),   # two elements, rotated
        ([1, 2],              1),   # two elements, not rotated
        ([3, 1, 2],           1),   # min in middle
        ([5, 6, 7, 8, 1, 2, 3, 4], 1),
    ]
    for nums, expected in cases:
        for fn in (find_min, find_min_two_candidate):
            result = fn(nums)
            assert result == expected, \
                f"{fn.__name__}({nums}) = {result}, expected {expected}"


if __name__ == "__main__":
    test_all()
    print("All tests passed")
