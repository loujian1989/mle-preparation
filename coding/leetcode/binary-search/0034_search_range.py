# 34. Find First and Last Position of Element in Sorted Array (Medium)
#
# Given a sorted array of integers, return [first, last] index of target.
# Return [-1, -1] if not found. Must run in O(log n).
#
# Key insight: one binary search finds the LEFT boundary, another finds
# the RIGHT boundary. They differ by exactly ONE character: >= vs >.
#
#   Left  boundary: when nums[mid] == target, go LEFT  (r = mid-1) → use >=
#   Right boundary: when nums[mid] == target, go RIGHT (l = mid+1) → use >
#
# --------------------------------------------------------------------------
# Approaches
#
#   Step 1: Flag-based (one function, boolean controls behavior)
#   Step 2: Two named functions (clean, interview-preferred)
#   Step 3: bisect module (production code)
#
# --------------------------------------------------------------------------
# Pros and cons of the flag-based approach
#
#   PRO — Code reuse: one function handles both boundaries.
#   PRO — Correct early exit: search left first, skip right if not found.
#   PRO — O(log n): two binary searches, clean complexity.
#
#   CON 1 — Boolean flag anti-pattern (main issue):
#     binarySearch(..., flag=True) — reader must look up the function to
#     know what True means. Robert Martin's Clean Code: "Flag arguments
#     are ugly. Passing a boolean into a function is a truly terrible
#     practice." It signals the function does two different things.
#
#   CON 2 — Duplicated loop body:
#     The two branches differ by ONE character (>= vs >). Any change to
#     the loop logic must be applied twice.
#
#   CON 3 — Subtle mixed return:
#     return l if flag else r
#     Left search returns l (insertion point); right search returns r.
#     This asymmetry is non-obvious without deep algorithm knowledge.
#
#   CON 4 — Right search starts from `left`, not 0:
#     Minor optimization, but couples the two calls — the right search
#     behaves differently depending on where the left search landed.
#
# --------------------------------------------------------------------------
# Complexity:
#   Time:  O(log n)  — two binary searches
#   Space: O(1)
# --------------------------------------------------------------------------


from bisect import bisect_left, bisect_right
from typing import List


# ---------------------------------------------------------------------------
# Step 1: Flag-based (user's version — acceptable but has code smells)
# ---------------------------------------------------------------------------

def search_range_flag(nums: List[int], target: int) -> List[int]:
    """
    Single binarySearch function controlled by a boolean flag.
    flag=True  → find left  boundary, return l
    flag=False → find right boundary, return r

    Complexity:
        Time:  O(log n)
        Space: O(1)
    """
    def binary_search(l: int, r: int, flag: bool) -> int:
        if flag:                           # find left boundary
            while l <= r:
                mid = l + (r - l) // 2
                if nums[mid] >= target:    # equal → go left to find first
                    r = mid - 1
                else:
                    l = mid + 1
        else:                              # find right boundary
            while l <= r:
                mid = l + (r - l) // 2
                if nums[mid] > target:     # equal → go right to find last
                    r = mid - 1
                else:
                    l = mid + 1
        return l if flag else r

    if not nums:
        return [-1, -1]
    left = binary_search(0, len(nums) - 1, True)
    if left >= len(nums) or nums[left] != target:
        return [-1, -1]
    return [left, binary_search(left, len(nums) - 1, False)]


# ---------------------------------------------------------------------------
# Step 2: Two named functions — clean, interview-preferred
# ---------------------------------------------------------------------------

def search_range(nums: List[int], target: int) -> List[int]:
    """
    Two separate binary searches with self-documenting names.
    The >= vs > difference is immediately visible at the call site.

    Trace: nums=[5,7,7,8,8,10], target=8
      find_left:
        l=0,r=5, mid=2, nums[2]=7 < 8  → l=3
        l=3,r=5, mid=4, nums[4]=8 >= 8 → r=3
        l=3,r=3, mid=3, nums[3]=8 >= 8 → r=2
        exit: return l=3 ✓
      find_right:
        l=0,r=5, mid=2, nums[2]=7 > 8? NO → l=3
        l=3,r=5, mid=4, nums[4]=8 > 8? NO → l=5
        l=5,r=5, mid=5, nums[5]=10 > 8   → r=4
        exit: return r=4 ✓
      return [3, 4] ✓

    Complexity:
        Time:  O(log n)
        Space: O(1)
    """
    def find_left(target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = l + (r - l) // 2
            if nums[mid] >= target:    # equal → keep going left
                r = mid - 1
            else:
                l = mid + 1
        return l                       # l = first position target could occupy

    def find_right(target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = l + (r - l) // 2
            if nums[mid] > target:     # equal → keep going right
                r = mid - 1
            else:
                l = mid + 1
        return r                       # r = last position target could occupy

    left = find_left(target)
    if left == len(nums) or nums[left] != target:
        return [-1, -1]
    return [left, find_right(target)]


# ---------------------------------------------------------------------------
# Step 3: bisect — production code
# ---------------------------------------------------------------------------

def search_range_bisect(nums: List[int], target: int) -> List[int]:
    """
    bisect_left  → first position where target could be inserted (left boundary)
    bisect_right → first position AFTER target (right boundary = that - 1)

    No reimplementation needed. In an interview, mention bisect first then
    implement manually to show you understand both.

    Complexity:
        Time:  O(log n)
        Space: O(1)
    """
    left = bisect_left(nums, target)
    if left == len(nums) or nums[left] != target:
        return [-1, -1]
    return [left, bisect_right(nums, target) - 1]


# ---------------------------------------------------------------------------
# Approach comparison
#
#   Approach        Readability   Reuse    Interview signal   Production
#   ─────────────   ───────────   ──────   ────────────────   ──────────
#   Flag-based      Poor          High     Acceptable         Avoid
#   Two functions   Good          Medium   Strong             OK
#   bisect          Best          N/A      Strong + manual    Preferred
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all() -> None:
    cases = [
        ([5, 7, 7, 8, 8, 10], 8,  [3, 4]),
        ([5, 7, 7, 8, 8, 10], 6,  [-1, -1]),
        ([],                   0,  [-1, -1]),
        ([1],                  1,  [0, 0]),
        ([1],                  0,  [-1, -1]),
        ([1, 1, 1, 1],         1,  [0, 3]),
        ([1, 2, 3],            1,  [0, 0]),
        ([1, 2, 3],            3,  [2, 2]),
        ([1, 2, 3],            4,  [-1, -1]),
        ([2, 2],               2,  [0, 1]),
    ]
    for nums, target, expected in cases:
        for fn in (search_range_flag, search_range, search_range_bisect):
            result = fn(nums, target)
            assert result == expected, \
                f"{fn.__name__}({nums}, {target}) = {result}, expected {expected}"


if __name__ == "__main__":
    test_all()
    print("All tests passed")
