"""
Longest Substring Without Repeating Characters (LeetCode 3) — Medium
=====================================================================

Problem:
    Find the length of the longest substring with all unique characters.

Follow-ups covered:
    A. Max length (this problem)
    B. Return the actual substring

Edge cases:
    - Empty string → 0
    - All same characters (e.g., "aaaa") → 1
    - All unique → len(s)
    - Unicode characters (not just ASCII)

Approach — Sliding window with HashMap:
    Maintain left pointer. For each right, if s[right] seen and within window,
    move left past the previous occurrence.
    HashMap stores char → most recent index.

Complexity:
    Time:  O(N) — each character processed at most twice (added + removed)
    Space: O(min(N, charset)) — at most 128 for ASCII
"""


def length_of_longest_substring(s: str) -> int:
    """Find length of longest substring without repeating characters.

    Args:
        s: Input string.

    Returns:
        Length of longest valid substring.

    Complexity:
        Time:  O(N)
        Space: O(min(N, charset))
    """
    last_seen: dict = {}   # char → most recent index
    left = 0
    max_len = 0

    for right, char in enumerate(s):
        if char in last_seen and last_seen[char] >= left:
            # Move left to just past previous occurrence
            left = last_seen[char] + 1
        last_seen[char] = right
        max_len = max(max_len, right - left + 1)

    return max_len


def longest_substring_without_repeating(s: str) -> str:
    """Return the actual longest substring without repeating characters.

    Args:
        s: Input string.

    Returns:
        The longest valid substring (first one if tied).

    Complexity:
        Time:  O(N)
        Space: O(min(N, charset))
    """
    last_seen: dict = {}
    left = 0
    best_start = 0
    best_len = 0

    for right, char in enumerate(s):
        if char in last_seen and last_seen[char] >= left:
            left = last_seen[char] + 1
        last_seen[char] = right
        if right - left + 1 > best_len:
            best_len = right - left + 1
            best_start = left

    return s[best_start:best_start + best_len]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    assert length_of_longest_substring("abcabcbb") == 3   # "abc"
    assert length_of_longest_substring("bbbbb") == 1       # "b"
    assert length_of_longest_substring("pwwkew") == 3      # "wke"
    assert length_of_longest_substring("") == 0
    assert length_of_longest_substring("abcde") == 5       # all unique

    substr = longest_substring_without_repeating("abcabcbb")
    assert len(set(substr)) == len(substr), "Must have all unique chars"
    assert len(substr) == 3

    print("  length_of_longest_substring: all tests passed")


if __name__ == "__main__":
    _test()
