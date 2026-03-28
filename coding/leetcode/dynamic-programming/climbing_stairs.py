"""
Climbing Stairs / Coin Change — 1D DP Foundation
=================================================

Two classic 1D DP problems that share the same "unbounded knapsack" structure.
Understanding both is prerequisite for all DP interview problems.

Problem A — Climbing Stairs (LeetCode 70):
    Reach the n-th stair taking 1 or 2 steps at a time.
    How many distinct ways?

Problem B — Coin Change (LeetCode 322):
    Given coins of denominations coins[] and a target amount,
    find the fewest coins to make the amount. Return -1 if impossible.

Problem C — Coin Change II (LeetCode 518):
    Count the number of combinations that make up amount.
    (Order doesn't matter — combinations, not permutations.)

Key DP patterns demonstrated:
    - Bottom-up tabulation (faster, no call stack)
    - State: dp[i] = answer for subproblem of size i
    - Transition: dp[i] = f(dp[i-1], dp[i-2], ...)

Edge cases:
    - n = 0 or amount = 0 → base case returns
    - coins larger than amount → skip
    - No valid combination → -1

Complexity:
    Climbing Stairs:  Time O(N),          Space O(1) with rolling vars
    Coin Change:      Time O(N * amount), Space O(amount)
    Coin Change II:   Time O(N * amount), Space O(amount)
"""

from typing import List


# ---------------------------------------------------------------------------
# Problem A: Climbing Stairs
# ---------------------------------------------------------------------------

def climb_stairs(n: int) -> int:
    """Count distinct ways to climb n stairs (1 or 2 steps at a time).

    Args:
        n: Number of stairs. Must be >= 1.

    Returns:
        Number of distinct ways to reach stair n.

    Raises:
        ValueError: If n < 1.

    Examples:
        >>> climb_stairs(3)
        3   # (1+1+1), (1+2), (2+1)

    Complexity:
        Time:  O(N)
        Space: O(1) — rolling two variables
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if n <= 2:
        return n
    prev2, prev1 = 1, 2
    for _ in range(3, n + 1):
        prev2, prev1 = prev1, prev1 + prev2
    return prev1


# ---------------------------------------------------------------------------
# Problem B: Coin Change (min coins)
# ---------------------------------------------------------------------------

def coin_change(coins: List[int], amount: int) -> int:
    """Find fewest coins to make amount. Returns -1 if impossible.

    Args:
        coins:  List of coin denominations (positive integers).
        amount: Target amount (non-negative integer).

    Returns:
        Minimum number of coins, or -1 if not achievable.

    Examples:
        >>> coin_change([1, 5, 6, 9], 11)
        2   # 5 + 6

    Complexity:
        Time:  O(N * amount) where N = len(coins)
        Space: O(amount)
    """
    INF = amount + 1
    dp = [INF] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] < INF else -1


# ---------------------------------------------------------------------------
# Problem C: Coin Change II (number of combinations)
# ---------------------------------------------------------------------------

def change(amount: int, coins: List[int]) -> int:
    """Count combinations of coins that sum to amount.

    IMPORTANT: Outer loop = coins, inner loop = amount.
    Swapping the loops counts permutations instead of combinations.

    Args:
        amount: Target sum.
        coins:  List of coin denominations.

    Returns:
        Number of distinct combinations.

    Examples:
        >>> change(5, [1, 2, 5])
        4   # (5), (1+2+2), (1+1+1+2), (1+1+1+1+1)

    Complexity:
        Time:  O(N * amount)
        Space: O(amount)
    """
    dp = [0] * (amount + 1)
    dp[0] = 1   # one way to make 0: use no coins

    for coin in coins:                         # outer = coin ensures combinations
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]

    return dp[amount]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    # Climbing stairs (Fibonacci)
    assert climb_stairs(1) == 1
    assert climb_stairs(2) == 2
    assert climb_stairs(3) == 3
    assert climb_stairs(5) == 8
    assert climb_stairs(10) == 89

    # Coin change
    assert coin_change([1, 5, 6, 9], 11) == 2    # 5+6
    assert coin_change([1, 2, 5], 11) == 3        # 5+5+1
    assert coin_change([2], 3) == -1              # impossible
    assert coin_change([1], 0) == 0               # amount=0

    # Coin change II
    assert change(5, [1, 2, 5]) == 4
    assert change(3, [2]) == 0                    # impossible
    assert change(0, [1, 2, 3]) == 1              # empty selection

    print("  1D DP (climbing stairs, coin change): all tests passed")


if __name__ == "__main__":
    _test()
