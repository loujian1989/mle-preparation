# 11. Container With Most Water (Medium)
#
# Given n non-negative integers height[0..n-1], where each represents a
# vertical line at position i, find two lines that together with the x-axis
# form a container holding the most water.
#
# Approaches:
#   Step 1: Brute force — try every pair          O(n²) T / O(1) S
#   Step 2: Two-pointer — shrink from both ends   O(n)  T / O(1) S
#
# --------------------------------------------------------------------------
# Why the two-pointer works — proof for the interviewer
#
# Setup: left pointer i=0, right pointer j=n-1.
# Current area = min(height[i], height[j]) * (j - i).
#
# WLOG assume height[i] <= height[j]  (i is the bottleneck).
# We move i += 1, skipping all pairs (i, k) for k < j.
#
# Claim: every skipped pair (i, k) with i < k < j has area <= current area.
#
# Proof:
#   area(i, k) = min(height[i], height[k]) * (k - i)
#              <= height[i]               * (k - i)    [min <= either arg]
#              <  height[i]               * (j - i)    [k < j so k-i < j-i]
#              =  current area                          [height[i] is bottleneck]
#
# Therefore no skipped pair can beat the current candidate.
# The same argument applies symmetrically when height[j] < height[i].
#
# Intuition: when i is the bottleneck, ANY pair involving i is capped at
# height[i] * width. Moving inward only shrinks width, so the only hope
# is to find a TALLER line — which means moving past the bottleneck side.
#
# --------------------------------------------------------------------------
# Two-pointer decision rule:
#   Move the pointer on the SHORTER side.
#   Reason: the shorter side is the bottleneck. Keeping it gains nothing
#   (width shrinks, height cap stays the same). Only advancing past it
#   can find something taller.
#
# --------------------------------------------------------------------------
# Similar problems using the same two-pointer "skip by proof" pattern:
#
#   Problem                     Key condition to skip          Move rule
#   ─────────────────────────   ─────────────────────────────  ──────────────────
#   LC 42  Trapping Rain Water  blocked side can't contribute  advance blocked
#   LC 167 Two Sum II (sorted)  sum too small → left too small advance left
#                               sum too large → right too big  advance right
#   LC 15  3Sum                 after fixing one, reduce to    advance inner ptr
#          (Two Sum variant)    sorted two-sum                 toward target
#   LC 16  3Sum Closest         same structure as 3Sum         advance inner ptr
#
# All share the invariant: one side is provably dominated — advancing it
# can only help, never hurt.
# --------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Step 1: Brute force — try every pair
# ---------------------------------------------------------------------------

def max_area_brute(height: list[int]) -> int:
    """
    Try all O(n²) pairs; track maximum area.

    Complexity:
        Time:  O(n²)
        Space: O(1)
    """
    n = len(height)
    res = 0
    for i in range(n):
        for j in range(i + 1, n):
            res = max(res, min(height[i], height[j]) * (j - i))
    return res


# ---------------------------------------------------------------------------
# Step 2: Two-pointer — O(n)
# ---------------------------------------------------------------------------

def max_area(height: list[int]) -> int:
    """
    Two-pointer shrink from both ends.

    Invariant: at each step, the current pair is the best candidate for
    this width. Moving the shorter side is the only way to improve.

    Trace: height = [1, 8, 6, 2, 5, 4, 8, 3, 7]
      i=0 (h=1), j=8 (h=7): area=min(1,7)*8=8,   shorter=i → i=1
      i=1 (h=8), j=8 (h=7): area=min(8,7)*7=49,  shorter=j → j=7
      i=1 (h=8), j=7 (h=3): area=min(8,3)*6=18,  shorter=j → j=6
      i=1 (h=8), j=6 (h=8): area=min(8,8)*5=40,  tie→j     → j=5
      i=1 (h=8), j=5 (h=4): area=min(8,4)*4=16,  shorter=j → j=4
      i=1 (h=8), j=4 (h=5): area=min(8,5)*3=15,  shorter=j → j=3
      i=1 (h=8), j=3 (h=2): area=min(8,2)*2=4,   shorter=j → j=2
      i=1 (h=8), j=2 (h=6): area=min(8,6)*1=6,   shorter=j → j=1
      i >= j: stop. Answer = 49 ✓

    Complexity:
        Time:  O(n)  — each pointer moves inward at most n times total
        Space: O(1)
    """
    res = 0
    i, j = 0, len(height) - 1
    while i < j:
        res = max(res, min(height[i], height[j]) * (j - i))
        if height[i] < height[j]:
            i += 1
        else:
            j -= 1
    return res


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all() -> None:
    cases = [
        ([1, 8, 6, 2, 5, 4, 8, 3, 7], 49),   # classic example
        ([1, 1],                        1),    # minimum input
        ([4, 3, 2, 1, 4],              16),   # equal heights at ends
        ([1, 2, 1],                     2),
        ([2, 3, 4, 5, 18, 17, 6],      17),
        ([1, 8, 100, 2, 100, 4, 8, 3, 7], 200),  # tall middle pair
    ]
    for height, expected in cases:
        for fn in (max_area_brute, max_area):
            result = fn(height)
            assert result == expected, \
                f"{fn.__name__}({height}) = {result}, expected {expected}"


if __name__ == "__main__":
    test_all()
    print("All tests passed")
