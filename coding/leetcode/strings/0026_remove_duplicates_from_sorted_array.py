# 26. Remove Duplicates from Sorted Array (Easy)
#
# Given a sorted array nums, remove duplicates in-place so each unique
# element appears only once. Return k = number of unique elements.
# The first k elements of nums must contain the unique values in order.
#
# Approach: Two-pointer (read pointer i, write pointer j)   O(n) T / O(1) S
#
# --------------------------------------------------------------------------
# Why it works
#
#   i scans every element; j only advances when a NEW unique element is found.
#
#   Key invariant: j ≤ i always.
#     j increments only when i increments AND the condition is true.
#     So writing nums[j] = nums[i] never overwrites an unread element.
#
#   Duplicate detection: since the array is sorted, duplicates are adjacent.
#     nums[i] != nums[i-1]  → new unique value  → write and advance j
#     nums[i] == nums[i-1]  → duplicate         → skip (i still advances)
#     i == 0                → first element, always write
#
#   Trace: [0, 0, 1, 1, 2]
#     i=0: i==0       → nums[0]=0, j=1
#     i=1: 0==0       → skip
#     i=2: 1!=0       → nums[1]=1, j=2
#     i=3: 1==1       → skip
#     i=4: 2!=1       → nums[2]=2, j=3
#     return 3,  nums=[0,1,2,_,_]
#
# --------------------------------------------------------------------------
# Interview extensions
#
#   Extension 1 — LC 80: allow at most 2 duplicates
#     Use the WRITE pointer lookback: nums[j-2] != x
#       if j < 2 or nums[j-2] != x:
#     NOTE: nums[i-2] is WRONG for k=2. When j falls behind i due to skips,
#     nums[i-2] may point to an already-overwritten position, giving a
#     corrupted value. nums[j-2] always points into the valid result prefix.
#
#   Extension 2 — General k: allow at most k duplicates (Staff question)
#     Use the write pointer itself to avoid the i-k index arithmetic:
#
#       def remove_duplicates_k(nums, k):
#           j = 0
#           for x in nums:
#               if j < k or nums[j - k] != x:
#                   nums[j] = x
#                   j += 1
#           return j
#
#     Why nums[j-k] != x works:
#       nums[0:j] is the result so far. nums[j-k] is the element placed
#       k positions back. If it equals x, we already have k copies → skip.
#       If it differs, x is new enough → write.
#       j < k: haven't placed k elements yet → always write.
#
#     k=1 → LC 26.  k=2 → LC 80.  Same code, one parameter.
#
#   Extension 3 — Unsorted array
#     Can't use this pattern (duplicates not adjacent).
#     Option A: set for seen elements → O(n) extra space.
#     Option B: sort first O(n log n), then apply in-place pattern.
#
#   Extension 4 — Can't modify in-place
#     Return a new array. Space O(k) for output.
#     Interviewer usually follows up asking for the in-place version.
#
# --------------------------------------------------------------------------
# Complexity:
#   Time:  O(n)  — single pass
#   Space: O(1)  — in-place, no extra data structures
# --------------------------------------------------------------------------


from typing import List


# ---------------------------------------------------------------------------
# LC 26: at most 1 duplicate allowed (each unique element appears once)
# ---------------------------------------------------------------------------

def remove_duplicates(nums: List[int]) -> int:
    """
    Two-pointer: i reads, j writes. Write only on new unique values.

    Complexity:
        Time:  O(n)
        Space: O(1)
    """
    j = 0
    for i in range(len(nums)):
        if i == 0 or nums[i] != nums[i - 1]:
            nums[j] = nums[i]
            j += 1
    return j


# ---------------------------------------------------------------------------
# LC 80: at most 2 duplicates allowed
# ---------------------------------------------------------------------------

def remove_duplicates_ii(nums: List[int]) -> int:
    """
    Must use the WRITE pointer lookback nums[j-2], NOT nums[i-2].

    Why nums[i-2] breaks:
      When a skip occurs, j falls behind i. We then write to nums[j],
      which overwrites a position that nums[i-2] may later point to.
      That corrupted value makes the condition incorrect.

      Trace: [1,1,1,2,2,3]
        i=3: write nums[j=2]=2   ← overwrites nums[2] from 1 to 2
        i=4: check nums[i-2]=nums[2]=2 (OVERWRITTEN!), current=2 → skip  ✗
             should have written the second 2.

    Why nums[j-2] is safe:
      nums[0:j] is the valid result so far. nums[j-2] is the element
      placed 2 slots ago in the RESULT — exactly what we need to check.
      It is never corrupted because j only advances forward.

    Complexity:
        Time:  O(n)
        Space: O(1)
    """
    j = 0
    for x in nums:
        if j < 2 or nums[j - 2] != x:
            nums[j] = x
            j += 1
    return j


# ---------------------------------------------------------------------------
# General: at most k duplicates allowed
# ---------------------------------------------------------------------------

def remove_duplicates_k(nums: List[int], k: int) -> int:
    """
    Use write pointer j to check k positions back instead of read pointer i.
    nums[j-k] is the element placed k slots ago in the result.
    If it equals current x, we already have k copies → skip.

    k=1 reproduces LC 26.
    k=2 reproduces LC 80.

    Complexity:
        Time:  O(n)
        Space: O(1)
    """
    j = 0
    for x in nums:
        if j < k or nums[j - k] != x:
            nums[j] = x
            j += 1
    return j


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all() -> None:
    # LC 26 cases
    lc26_cases = [
        ([1, 1, 2],                   2, [1, 2]),
        ([0, 0, 1, 1, 1, 2, 2, 3, 3, 4], 5, [0, 1, 2, 3, 4]),
        ([1],                         1, [1]),
        ([1, 1],                      1, [1]),
        ([1, 2, 3],                   3, [1, 2, 3]),   # no duplicates
    ]
    for nums, exp_k, exp_vals in lc26_cases:
        arr = nums[:]
        k = remove_duplicates(arr)
        assert k == exp_k and arr[:k] == exp_vals, \
            f"remove_duplicates({nums}) → k={k}, nums={arr[:k]}, expected k={exp_k}, {exp_vals}"
        # general k=1 must match
        arr2 = nums[:]
        k2 = remove_duplicates_k(arr2, 1)
        assert k2 == exp_k and arr2[:k2] == exp_vals, \
            f"remove_duplicates_k({nums},1) → k={k2}, nums={arr2[:k2]}"

    # LC 80 cases (at most 2)
    lc80_cases = [
        ([1, 1, 1, 2, 2, 3],          5, [1, 1, 2, 2, 3]),
        ([0, 0, 1, 1, 1, 1, 2, 3, 3], 7, [0, 0, 1, 1, 2, 3, 3]),
        ([1, 1],                       2, [1, 1]),
        ([1, 1, 1],                    2, [1, 1]),
    ]
    for nums, exp_k, exp_vals in lc80_cases:
        arr = nums[:]
        k = remove_duplicates_ii(arr)
        assert k == exp_k and arr[:k] == exp_vals, \
            f"remove_duplicates_ii({nums}) → k={k}, nums={arr[:k]}, expected k={exp_k}, {exp_vals}"
        arr2 = nums[:]
        k2 = remove_duplicates_k(arr2, 2)
        assert k2 == exp_k and arr2[:k2] == exp_vals, \
            f"remove_duplicates_k({nums},2) → k={k2}, nums={arr2[:k2]}"


if __name__ == "__main__":
    test_all()
    print("All tests passed")
