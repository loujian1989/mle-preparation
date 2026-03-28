"""
Word Break (LeetCode 139 + 140) — Medium / Hard
================================================

Problem A (LC 139):
    Given string s and dictionary wordDict, return True if s can be segmented
    into words from wordDict (words may be reused).

Problem B (LC 140 — Word Break II):
    Return all valid segmentations (as space-separated sentences).

Edge cases:
    - Empty string → one valid segmentation: ""
    - Word longer than remaining string → skip
    - Overlapping words in dictionary

Approach A (LC 139):
    1D DP: dp[i] = True if s[:i] can be segmented.
    Transition: dp[i] = any(dp[j] and s[j:i] in word_set for j < i)
    Optimization: cap inner loop at max_word_len to avoid unnecessary checks.

Approach B (LC 140):
    DFS with memoization: at each position, try all words starting at pos.
    Cache pos → [suffixes] to avoid recomputation.

Complexity:
    LC 139: Time O(N^2) or O(N * max_len), Space O(N)
    LC 140: Time O(N * 2^N) worst case (exponential output), Space O(N * 2^N)
"""

from functools import lru_cache
from typing import List, Set


# ---------------------------------------------------------------------------
# Problem A: Word Break — can it be segmented?
# ---------------------------------------------------------------------------

def word_break(s: str, word_dict: List[str]) -> bool:
    """Determine if s can be segmented using words from word_dict.

    Args:
        s:         Input string.
        word_dict: List of valid words.

    Returns:
        True if valid segmentation exists.

    Complexity:
        Time:  O(N * max_word_len)
        Space: O(N)
    """
    word_set: Set[str] = set(word_dict)
    max_len = max((len(w) for w in word_set), default=0)
    N = len(s)

    dp = [False] * (N + 1)
    dp[0] = True   # empty prefix is always segmentable

    for i in range(1, N + 1):
        for j in range(max(0, i - max_len), i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break

    return dp[N]


# ---------------------------------------------------------------------------
# Problem B: Word Break II — return all segmentations
# ---------------------------------------------------------------------------

def word_break_ii(s: str, word_dict: List[str]) -> List[str]:
    """Return all valid space-separated segmentations of s.

    Args:
        s:         Input string.
        word_dict: List of valid words.

    Returns:
        List of valid sentences.

    Complexity:
        Time:  O(N * 2^N) worst case (e.g., s="aaa", dict=["a","aa","aaa"])
        Space: O(N * 2^N) for output storage
    """
    word_set: Set[str] = set(word_dict)
    N = len(s)
    memo: dict = {}

    def backtrack(start: int) -> List[str]:
        """Return list of valid segmentations of s[start:].

        Args:
            start: Current position in s.

        Returns:
            List of suffixes (space-separated words) from s[start:].
        """
        if start in memo:
            return memo[start]
        if start == N:
            return [""]

        results: List[str] = []
        for end in range(start + 1, N + 1):
            word = s[start:end]
            if word in word_set:
                suffixes = backtrack(end)
                for suffix in suffixes:
                    sentence = word if not suffix else word + " " + suffix
                    results.append(sentence)

        memo[start] = results
        return results

    return backtrack(0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    # Word Break I
    assert word_break("leetcode", ["leet", "code"]) is True
    assert word_break("applepenapple", ["apple", "pen"]) is True
    assert word_break("catsandog", ["cats", "dog", "sand", "and", "cat"]) is False
    assert word_break("", ["a"]) is True     # empty string

    # Word Break II
    result = set(word_break_ii("catsanddog", ["cat", "cats", "and", "sand", "dog"]))
    assert result == {"cats and dog", "cat sand dog"}

    result_empty = word_break_ii("", ["a"])
    assert result_empty == [""]   # empty string → one empty sentence

    result_none = word_break_ii("abc", ["de"])
    assert result_none == []

    print("  word_break (I + II): all tests passed")


if __name__ == "__main__":
    _test()
