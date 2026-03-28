"""
Koko Eating Bananas / Binary Search on Answer (LeetCode 875) — Medium
=======================================================================

Problem:
    Koko has piles of bananas. Guards return in h hours.
    She eats at speed k bananas/hour (one pile per hour, k <= pile size → remainder wasted).
    Find the minimum k such that she finishes all bananas in <= h hours.

Key insight — "Binary search on the answer":
    Instead of searching in an array, search in the answer space [1, max(piles)].
    Predicate: can_finish(k) = Σ ceil(pile/k) <= h?
    Predicate is monotone: if k works, k+1 also works → binary search applies.

Template (generalizable to many "minimum/maximum feasible" problems):
    left = min_possible_answer
    right = max_possible_answer
    while left < right:
        mid = (left + right) // 2
        if feasible(mid): right = mid    # look for smaller feasible
        else:             left = mid + 1

Similar problems using the same template:
    - Capacity to ship packages in D days (LC 1011)
    - Split array largest sum (LC 410)
    - Minimum days to make m bouquets (LC 1482)

Complexity:
    Time:  O(N log M) where N = len(piles), M = max(piles)
    Space: O(1)
"""

import math
from typing import List


def min_eating_speed(piles: List[int], h: int) -> int:
    """Find minimum eating speed k to finish all bananas in h hours.

    Args:
        piles: Number of bananas in each pile.
        h:     Hours available (h >= len(piles)).

    Returns:
        Minimum integer speed k.

    Raises:
        ValueError: If h < len(piles) (can't even visit each pile).

    Complexity:
        Time:  O(N log M) where M = max(piles)
        Space: O(1)
    """
    if h < len(piles):
        raise ValueError(f"h={h} must be >= len(piles)={len(piles)}")

    def can_finish(k: int) -> bool:
        """True if speed k allows finishing all piles in h hours."""
        return sum(math.ceil(pile / k) for pile in piles) <= h

    left, right = 1, max(piles)
    while left < right:
        mid = (left + right) // 2
        if can_finish(mid):
            right = mid     # try smaller speed
        else:
            left = mid + 1  # need higher speed

    return left


# ---------------------------------------------------------------------------
# Generic "binary search on answer" template
# ---------------------------------------------------------------------------

def binary_search_answer(
    lo: int,
    hi: int,
    feasible,  # Callable[[int], bool] — monotone predicate
    find_min: bool = True,
) -> int:
    """Generic binary search on answer space.

    Finds minimum feasible value if find_min=True,
    or maximum feasible value if find_min=False.

    Args:
        lo:        Minimum candidate answer.
        hi:        Maximum candidate answer.
        feasible:  Monotone predicate. For find_min=True: feasible is True for
                   all values >= some threshold.
        find_min:  If True, find min feasible. If False, find max feasible.

    Returns:
        Optimal answer.

    Complexity:
        Time:  O(log(hi - lo) * cost(feasible))
        Space: O(1)
    """
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            if find_min:
                hi = mid         # feasible, look smaller
            else:
                lo = mid         # feasible, look larger
        else:
            if find_min:
                lo = mid + 1     # not feasible, increase
            else:
                hi = mid - 1     # not feasible, decrease
    return lo


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    assert min_eating_speed([3, 6, 7, 11], 8) == 4
    assert min_eating_speed([30, 11, 23, 4, 20], 5) == 30
    assert min_eating_speed([30, 11, 23, 4, 20], 6) == 23
    assert min_eating_speed([1], 1) == 1

    # Generic template: same problem re-expressed
    piles = [3, 6, 7, 11]
    h = 8
    result = binary_search_answer(
        lo=1,
        hi=max(piles),
        feasible=lambda k: sum(math.ceil(p / k) for p in piles) <= h,
        find_min=True,
    )
    assert result == 4

    print("  koko_eating / binary_search_on_answer: all tests passed")


if __name__ == "__main__":
    _test()
