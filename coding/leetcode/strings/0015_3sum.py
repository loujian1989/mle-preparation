# 15. 3Sum (Medium)
#
# Given an integer array nums, return all triplets [nums[i], nums[j], nums[k]]
# such that i != j != k and nums[i] + nums[j] + nums[k] == 0.
# The solution set must not contain duplicate triplets.
#
# Approach: Sort + Two-Pointer   O(n²) T / O(1) S  (excluding output)
#
# --------------------------------------------------------------------------
# Why sort first?
#
#   Sorting unlocks two things simultaneously:
#     1. Two-pointer works on the inner pair (sorted array → skip-by-proof)
#     2. Duplicates become adjacent → dedup reduces to "check neighbor"
#
#   Without sorting, dedup requires a set of tuples (O(n²) extra space,
#   hashing overhead). With sorting, all dedup is O(1) pointer comparisons.
#
# --------------------------------------------------------------------------
# Three independent sources of duplicates — each handled separately:
#
#   Source 1: Duplicate anchor (outer i)
#   ─────────────────────────────────────
#   If nums[i] == nums[i-1], we already ran two-pointer with that same
#   anchor value. Running again produces the EXACT same set of triplets.
#   Fix: skip with `if i > 0 and nums[i] == nums[i-1]: continue`
#
#   Interviewer explanation:
#   "Every unique value of the first element yields a unique set of triplets.
#    Same value again → already found everything from it. Skip."
#
#   Source 2 & 3: Duplicate l and r after a hit
#   ────────────────────────────────────────────
#   After recording triplet [-anchor, nums[l], nums[r]], advancing l or r
#   to an EQUAL value would re-record the same triplet.
#   Fix: skip copies of both before moving the pointers.
#
#   Two styles — same result, different readability:
#
#   Style A (skip-then-advance — user's version):
#     while l < r and nums[l] == nums[l+1]: l += 1   # skip to last duplicate
#     while l < r and nums[r] == nums[r-1]: r -= 1
#     l, r = l+1, r-1                                 # then step past it
#
#   Style B (advance-then-skip — cleaner intent):
#     l += 1                                           # step first
#     r -= 1
#     while l < r and nums[l] == nums[l-1]: l += 1   # skip copies of what we left
#     while l < r and nums[r] == nums[r+1]: r -= 1
#
#   Style B is preferred in an interview: "move first, then skip any copies
#   of what you just left" is easier to reason about and explain.
#
#   Interviewer explanation:
#   "After a hit I skip all copies of the left value and all copies of the
#    right value before moving the pointers. Sorted order guarantees
#    duplicates are contiguous, so this is just a while-equal loop."
#
# --------------------------------------------------------------------------
# Alternative: set-based dedup (simpler code, worse performance)
#
#   res = set()
#   # on hit: res.add(tuple([nums[i], nums[l], nums[r]]))  # already sorted
#
#   Works because sorted tuples are hashable and equal when values match.
#   BUT: requires hashing every triplet → O(n²) hash ops + O(n²) extra space.
#   Interviewers usually push back asking for the pointer-skip approach.
#
# --------------------------------------------------------------------------
# 30-second interview explanation:
#
#   "Sort first — O(n log n), unlocks two things: two-pointer for the inner
#    search, and adjacent-duplicate detection for dedup. Fix each element
#    left to right, skipping repeated values. Inside, two pointers find pairs
#    summing to the negative of the anchor. On a hit, skip copies of both the
#    left and right values before moving on. Three dedup points: anchor, left
#    pointer, right pointer — all handled by checking the neighbor in a sorted
#    array."
#
# --------------------------------------------------------------------------
# Complexity:
#   Time:  O(n²)  — outer O(n) * inner two-pointer O(n)
#                   sort is O(n log n), dominated by O(n²)
#   Space: O(1)   — no extra data structures (output list not counted)
# --------------------------------------------------------------------------


from typing import List


# ---------------------------------------------------------------------------
# Style A: skip-then-advance (user's version — accepted)
# ---------------------------------------------------------------------------

def three_sum_a(nums: List[int]) -> List[List[int]]:
    """
    Sort + fix anchor + two-pointer inner search.
    Dedup: skip-then-advance style.

    Trace: nums = [-1, 0, 1, 2, -1, -4]
    After sort: [-4, -1, -1, 0, 1, 2]

    i=0, anchor=-4, target=4: l=1,r=5 → sums never hit 0 → no result
    i=1, anchor=-1, target=1:
      l=2,r=5: nums[2]+nums[5]=-1+2=1 ✓ → append [-1,-1,2]
        skip dup l: nums[2]==nums[3]? -1==0? No → no skip
        skip dup r: nums[5]==nums[4]?  2==1? No → no skip
        l=3,r=4
      l=3,r=4: nums[3]+nums[4]=0+1=1 ✓ → append [-1,0,1]
        l=4,r=3 → stop
    i=2, anchor=-1: nums[2]==nums[1] → skip (dedup anchor)
    i=3, anchor=0, target=0:
      l=4,r=5: nums[4]+nums[5]=1+2=3 > 0 → r=4
      l=4,r=4 → stop
    i=4,i=5: no room for two-pointer
    Result: [[-1,-1,2], [-1,0,1]] ✓
    """
    res = []
    nums.sort()

    def two_sum(target: int, l: int, r: int) -> None:
        while l < r:
            s = nums[l] + nums[r]
            if s < target:
                l += 1
            elif s > target:
                r -= 1
            else:
                res.append([-target, nums[l], nums[r]])
                # skip copies of current l and r before advancing
                while l < r and nums[l] == nums[l + 1]:
                    l += 1
                while l < r and nums[r] == nums[r - 1]:
                    r -= 1
                l += 1
                r -= 1

    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:   # dedup anchor
            continue
        two_sum(-nums[i], i + 1, len(nums) - 1)

    return res


# ---------------------------------------------------------------------------
# Style B: advance-then-skip (cleaner intent — preferred in interviews)
# ---------------------------------------------------------------------------

def three_sum(nums: List[int]) -> List[List[int]]:
    """
    Same algorithm; dedup in advance-then-skip style.
    Easier to reason about: "move first, then skip copies of what I left."

    Complexity:
        Time:  O(n²)
        Space: O(1)
    """
    res = []
    nums.sort()

    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:   # dedup anchor
            continue
        l, r = i + 1, len(nums) - 1
        while l < r:
            s = nums[l] + nums[r]
            if s < -nums[i]:
                l += 1
            elif s > -nums[i]:
                r -= 1
            else:
                res.append([nums[i], nums[l], nums[r]])
                l += 1                                      # advance first
                r -= 1
                while l < r and nums[l] == nums[l - 1]:   # skip dup l
                    l += 1
                while l < r and nums[r] == nums[r + 1]:   # skip dup r
                    r -= 1

    return res


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _sorted_result(result: List[List[int]]) -> List[List[int]]:
    """Normalize output for comparison (order of triplets doesn't matter)."""
    return sorted(sorted(t) for t in result)


def test_all() -> None:
    cases = [
        ([-1, 0, 1, 2, -1, -4], [[-1, -1, 2], [-1, 0, 1]]),
        ([0, 1, 1],              []),
        ([0, 0, 0],              [[0, 0, 0]]),
        ([],                     []),
        ([0],                    []),
        ([-2, 0, 0, 2, 2],      [[-2, 0, 2]]),
        ([-4, -1, -1, 0, 1, 2], [[-1, -1, 2], [-1, 0, 1]]),  # pre-sorted
        ([1, 2, -2, -1],        []),
        ([-1, -1, -1, 0, 1, 2], [[-1, -1, 2], [-1, 0, 1]]),  # triple dup anchor
    ]
    for nums, expected in cases:
        exp = _sorted_result(expected)
        for fn in (three_sum_a, three_sum):
            result = _sorted_result(fn(nums[:]))   # pass a copy — sort mutates
            assert result == exp, \
                f"{fn.__name__}({nums}) = {result}, expected {exp}"


if __name__ == "__main__":
    test_all()
    print("All tests passed")
