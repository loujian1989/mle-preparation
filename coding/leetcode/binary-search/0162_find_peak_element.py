# 162. Find Peak Element (Medium)
#
# A peak element is strictly greater than its neighbors.
# nums[-1] = nums[n] = -∞ (boundaries are always "less than" any element).
# All adjacent elements are distinct (nums[i] != nums[i+1]).
# Return the index of ANY peak. Must run in O(log n).
#
# Approaches:
#   Step 1: Clean slope-based binary search  O(log n) T / O(1) S  ← idiomatic
#   Step 2: User's boundary-check version    O(log n) T / O(1) S  ← works, more complex
#
# --------------------------------------------------------------------------
# Why this is a find-peak problem, not a find-target problem
#
#   Find target (LC 33, 34): looking for a KNOWN VALUE.
#     When nums[mid] != target, mid is proven NOT the answer → exclude with mid±1.
#     Loop: l <= r.  Moves: l=mid+1 or r=mid-1.
#
#   Find peak/min/max: looking for a PROPERTY (inflection point).
#     mid might be the answer → cannot exclude without proof.
#     Loop: l < r.   Moves: l=mid+1 or r=mid (keep mid as candidate).
#     Exit: l == r → exactly one candidate → the answer.
#
# --------------------------------------------------------------------------
# Slope invariant — why single condition works without boundary checks
#
#   Invariant: there is always at least one peak in [l, r].
#
#   At any mid (not at array boundary in a meaningful sense):
#     Case A: nums[mid] < nums[mid+1]
#       Slope rising to the right. Either nums[mid+1] is a peak (if nums[mid+2]
#       is lower) or the slope continues rising. Either way, a peak exists in
#       [mid+1, r]. → l = mid+1 (safe to exclude mid — it can't be the peak
#       because its right neighbor is larger).
#
#     Case B: nums[mid] >= nums[mid+1]  →  nums[mid] > nums[mid+1] (no equals)
#       Slope falling to the right. Either nums[mid] is a peak (if nums[mid-1]
#       is lower) or the slope is rising to the left. Peak exists in [l, mid].
#       → r = mid (keep mid as candidate).
#
#   Why no boundary check needed:
#     nums[mid+1] access is always valid when l < r (mid < r ≤ n-1).
#     nums[-1] = -∞ means the leftmost element is always a valid peak if
#     nums[0] > nums[1], handled naturally by the slope logic.
#
# --------------------------------------------------------------------------
# User's l <= r version — why it needs boundary checks
#
#   With l <= r, we can reach l = r = 0. Then checking nums[mid-1] = nums[-1]
#   would be index -1 (the last element in Python, not -∞). Boundary check
#   patches this. With l < r, mid is always < r ≤ n-1, so mid+1 is valid
#   and we never need nums[mid-1] in the main condition.
#
#   Subtle difference: the "if not nums" guard in the user's version is dead
#   code — the constraint guarantees 1 <= nums.length <= 1000. Both versions
#   with l <= r are functionally identical.
#
# --------------------------------------------------------------------------
# Complexity:
#   Time:  O(log n)  — halve the search space each iteration
#   Space: O(1)
# --------------------------------------------------------------------------


from typing import List


# ---------------------------------------------------------------------------
# Step 1: Clean slope-based binary search — idiomatic find-peak form
# ---------------------------------------------------------------------------

def find_peak_element(nums: List[int]) -> int:
    """
    Single condition: slope rising right → move l; falling → keep mid in r.
    No boundary checks needed. Exits when l == r.

    Trace: nums=[1,2,1,3,5,6,4]
      l=0,r=6, mid=3: nums[3]=3 < nums[4]=5 → l=4
      l=4,r=6, mid=5: nums[5]=6 > nums[6]=4 → r=5
      l=4,r=5, mid=4: nums[4]=5 < nums[5]=6 → l=5
      l==r==5 → return 5 ✓ (nums[5]=6 is a peak)

    Trace: nums=[1,2,3,1]
      l=0,r=3, mid=1: nums[1]=2 < nums[2]=3 → l=2
      l=2,r=3, mid=2: nums[2]=3 > nums[3]=1 → r=2
      l==r==2 → return 2 ✓

    Trace (ascending): nums=[1,2,3,4,5]
      l=0,r=4, mid=2: nums[2]=3 < nums[3]=4 → l=3
      l=3,r=4, mid=3: nums[3]=4 < nums[4]=5 → l=4
      l==r==4 → return 4 ✓ (last element is peak since nums[n]=-∞)

    Complexity:
        Time:  O(log n)
        Space: O(1)
    """
    l, r = 0, len(nums) - 1
    while l < r:
        mid = l + (r - l) // 2
        if nums[mid] < nums[mid + 1]:
            l = mid + 1    # slope rising right → peak is strictly right of mid
        else:
            r = mid        # slope falling → peak at mid or to its left
    return l               # l == r → the peak


# ---------------------------------------------------------------------------
# Step 2: User's boundary-check version — l <= r with explicit edge cases
# ---------------------------------------------------------------------------

def find_peak_element_boundary(nums: List[int]) -> int:
    """
    Handles boundaries explicitly: check mid==0 and mid==n-1 separately.
    Requires three-branch logic instead of one condition.
    Works correctly, but more code surface for bugs.

    Why boundary checks are needed here (not in Step 1):
      With l <= r, l can equal r, and mid can be 0 or n-1.
      Accessing nums[mid-1] when mid=0 gives nums[-1] = last element,
      NOT -∞. The explicit boundary check avoids this incorrect access.

    Note: "if not nums" guard is dead code (constraint: 1 <= n <= 1000).

    Complexity:
        Time:  O(log n)
        Space: O(1)
    """
    if len(nums) == 1:
        return 0
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = l + (r - l) // 2
        # Boundary: left edge is peak if larger than its only neighbor
        if mid == 0 and nums[mid] > nums[mid + 1]:
            return mid
        # Boundary: right edge is peak if larger than its only neighbor
        if mid == len(nums) - 1 and nums[mid] > nums[mid - 1]:
            return mid
        if nums[mid] < nums[mid + 1]:      # slope rising → peak to the right
            l = mid + 1
        elif nums[mid] < nums[mid - 1]:    # slope rising left → peak to the left
            r = mid - 1
        else:                              # both neighbors smaller → mid is peak
            return mid
    return -1                              # unreachable given constraints


# ---------------------------------------------------------------------------
# Step 3: l < r-1 style — two-candidate exit
# ---------------------------------------------------------------------------

def find_peak_element_two_candidate(nums: List[int]) -> int:
    """
    Stop when l and r are adjacent. At exit, exactly two candidates remain.
    Check which one is a peak (the higher side of the slope).

    Same slope logic as Step 1, but the loop exits one step earlier.
    Needs a final check instead of simply returning nums[l].

    Why two candidates at exit:
      l < r-1 stops when r - l == 1 (adjacent). Mid always equals l
      in this state (mid = l + (r-l)//2 = l + 0 = l when r=l+1),
      so the last step either moves l up or r down, leaving l and r adjacent.

    At exit: nums[l] and nums[r] are the remaining candidates.
      - nums[l] > nums[r]: the slope is falling right, l might be the peak.
        But we also need nums[l] > its left neighbor. Since the binary search
        drove us here following a rising slope, nums[l] > nums[l-1] is
        guaranteed unless l==0 (always a valid peak with -∞ boundary).
      - nums[l] < nums[r]: slope rising, r is the candidate.
        Similarly, nums[r] > nums[r+1] is guaranteed unless r==n-1.
      Simpler: just return the index of the larger value — it is the peak.

    Trace: nums=[1,2,1,3,5,6,4]
      l=0,r=6, mid=3: nums[3]=3 < nums[4]=5 → l=4
      l=4,r=6, mid=5: nums[5]=6 > nums[6]=4 → r=5
      l=4,r=5: r-l==1 → exit
      nums[4]=5 vs nums[5]=6 → return 5 ✓

    Trace: nums=[1,2,3,1]
      l=0,r=3, mid=1: nums[1]=2 < nums[2]=3 → l=2
      l=2,r=3: r-l==1 → exit
      nums[2]=3 vs nums[3]=1 → return 2 ✓

    Complexity:
        Time:  O(log n)
        Space: O(1)
    """
    if len(nums) == 1:
        return 0
    l, r = 0, len(nums) - 1
    while l < r - 1:
        mid = l + (r - l) // 2
        if nums[mid] < nums[mid + 1]:
            l = mid + 1
        else:
            r = mid
    return l if nums[l] > nums[r] else r


# ---------------------------------------------------------------------------
# Comparison
#
#   Approach           Loop      Condition(s)   Boundary check   Return
#   ─────────────      ──────    ────────────   ──────────────   ──────────
#   Slope (Step 1)     l < r     1              No               nums[l]
#   Boundary (Step 2)  l <= r    3              Yes              mid
#   Two-candidate (3)  l < r-1   1              No               larger of l,r
#
#   For find-peak: prefer Step 1. Step 3 is equivalent but needs a final
#   comparison at exit instead of a direct return.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all() -> None:
    cases = [
        # (nums, valid peak indices)
        ([1, 2, 3, 1],           {2}),
        ([1, 2, 1, 3, 5, 6, 4], {1, 5}),
        ([1],                    {0}),
        ([1, 2],                 {1}),
        ([2, 1],                 {0}),
        ([1, 2, 3],              {2}),
        ([3, 2, 1],              {0}),
        ([3, 1, 2],              {0, 2}),
        ([1, 2, 1, 2, 1],       {1, 3}),
    ]
    for nums, valid in cases:
        for fn in (find_peak_element, find_peak_element_boundary, find_peak_element_two_candidate):
            result = fn(nums[:])
            assert result in valid, \
                f"{fn.__name__}({nums}) = {result}, expected one of {valid}"


if __name__ == "__main__":
    test_all()
    print("All tests passed")
