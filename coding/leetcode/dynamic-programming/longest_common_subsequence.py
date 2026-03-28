"""
Longest Common Subsequence (LeetCode 1143) — Medium
=====================================================

Problem:
    Given two strings text1 and text2, return the length of their LCS.
    A subsequence is derived by deleting characters without changing order.

Follow-ups covered:
    A. LCS length (classic)
    B. Reconstruct the actual LCS string
    C. Edit Distance (LeetCode 72) — closely related 2D DP

Edge cases:
    - Empty string → 0
    - No common characters → 0
    - One string is a substring of the other

DP formulation:
    dp[i][j] = LCS length of text1[:i] and text2[:j]

    Transition:
        if text1[i-1] == text2[j-1]: dp[i][j] = dp[i-1][j-1] + 1
        else:                         dp[i][j] = max(dp[i-1][j], dp[i][j-1])

Space optimization: only need two rows at a time (prev + curr).

Complexity:
    LCS:          Time O(M*N), Space O(M*N) full table | O(min(M,N)) optimized
    Edit Distance: Time O(M*N), Space O(M*N)
"""

from typing import List


# ---------------------------------------------------------------------------
# LCS — length only (space-optimized)
# ---------------------------------------------------------------------------

def longest_common_subsequence(text1: str, text2: str) -> int:
    """Compute the length of the longest common subsequence.

    Args:
        text1: First string.
        text2: Second string.

    Returns:
        Length of LCS.

    Complexity:
        Time:  O(M * N)
        Space: O(min(M, N)) — rolling two rows
    """
    if not text1 or not text2:
        return 0

    # Ensure text2 is the shorter string for space optimization
    if len(text1) < len(text2):
        text1, text2 = text2, text1

    M, N = len(text1), len(text2)
    prev = [0] * (N + 1)

    for i in range(1, M + 1):
        curr = [0] * (N + 1)
        for j in range(1, N + 1):
            if text1[i - 1] == text2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr

    return prev[N]


# ---------------------------------------------------------------------------
# LCS — reconstruct the actual sequence
# ---------------------------------------------------------------------------

def lcs_reconstruct(text1: str, text2: str) -> str:
    """Return the actual LCS string (not just length).

    Args:
        text1: First string.
        text2: Second string.

    Returns:
        One valid LCS string (may not be unique).

    Complexity:
        Time:  O(M * N)
        Space: O(M * N) — must keep full table for backtracking
    """
    M, N = len(text1), len(text2)
    dp: List[List[int]] = [[0] * (N + 1) for _ in range(M + 1)]

    for i in range(1, M + 1):
        for j in range(1, N + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to reconstruct
    result = []
    i, j = M, N
    while i > 0 and j > 0:
        if text1[i - 1] == text2[j - 1]:
            result.append(text1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return "".join(reversed(result))


# ---------------------------------------------------------------------------
# Edit Distance (LeetCode 72) — same 2D DP structure
# ---------------------------------------------------------------------------

def min_distance(word1: str, word2: str) -> int:
    """Compute minimum edit distance (Levenshtein distance).

    Operations: insert, delete, replace — each costs 1.

    Args:
        word1: Source string.
        word2: Target string.

    Returns:
        Minimum number of operations to transform word1 → word2.

    Complexity:
        Time:  O(M * N)
        Space: O(N) — rolling row optimization
    """
    M, N = len(word1), len(word2)
    # dp[j] = min edits to transform word1[:i] to word2[:j]
    dp = list(range(N + 1))   # base case: empty word1, insert j chars

    for i in range(1, M + 1):
        prev_diag = dp[0]
        dp[0] = i             # delete i chars from word1 to match ""
        for j in range(1, N + 1):
            temp = dp[j]
            if word1[i - 1] == word2[j - 1]:
                dp[j] = prev_diag       # no operation needed
            else:
                dp[j] = 1 + min(
                    prev_diag,          # replace
                    dp[j],              # delete from word1
                    dp[j - 1],          # insert into word1
                )
            prev_diag = temp

    return dp[N]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    # LCS length
    assert longest_common_subsequence("abcde", "ace") == 3
    assert longest_common_subsequence("abc", "abc") == 3
    assert longest_common_subsequence("abc", "def") == 0
    assert longest_common_subsequence("", "abc") == 0

    # LCS reconstruction
    seq = lcs_reconstruct("abcde", "ace")
    assert seq == "ace"
    seq2 = lcs_reconstruct("abc", "abc")
    assert seq2 == "abc"

    # Edit distance
    assert min_distance("horse", "ros") == 3
    assert min_distance("intention", "execution") == 5
    assert min_distance("", "abc") == 3
    assert min_distance("abc", "") == 3
    assert min_distance("abc", "abc") == 0

    print("  LCS + edit distance: all tests passed")


if __name__ == "__main__":
    _test()
