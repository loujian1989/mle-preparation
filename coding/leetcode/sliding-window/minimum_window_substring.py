"""
Minimum Window Substring (LeetCode 76) — Hard
==============================================

Problem:
    Given strings s and t, find the minimum window substring of s that contains
    all characters of t (including duplicates). Return "" if no such window exists.

Edge cases:
    - t longer than s → ""
    - s == t → s
    - t has duplicate characters (e.g., t="AA") — must count occurrences
    - No valid window → ""

Approach — Sliding window + frequency counter:
    1. Count frequency of each char in t.
    2. Expand right: add char to window, decrement need counter if fully satisfied.
    3. When all chars satisfied (formed == required), contract left to minimize window.
    4. Track minimum window across all valid states.

    Key detail: "formed" tracks how many unique chars in t are satisfied in window.
    A char is "satisfied" when window_count[c] >= t_count[c].

Complexity:
    Time:  O(|S| + |T|) — each character in S visited at most twice
    Space: O(|S| + |T|) for frequency maps
"""

from collections import Counter


def min_window(s: str, t: str) -> str:
    """Find minimum window substring of s containing all of t.

    Args:
        s: Source string to search.
        t: Target string (all chars must be present in window).

    Returns:
        Minimum valid window substring, or "" if none exists.

    Complexity:
        Time:  O(|S| + |T|)
        Space: O(|S| + |T|)
    """
    if not s or not t or len(s) < len(t):
        return ""

    t_count = Counter(t)
    required = len(t_count)   # number of unique chars in t that must be satisfied

    window: dict = {}
    formed = 0     # unique chars in t currently satisfied in window

    left = 0
    best_len = float("inf")
    best_left = 0

    for right, char in enumerate(s):
        window[char] = window.get(char, 0) + 1

        # Check if this char's requirement is now met
        if char in t_count and window[char] == t_count[char]:
            formed += 1

        # Contract left while window is valid
        while formed == required:
            # Update best window
            window_len = right - left + 1
            if window_len < best_len:
                best_len = window_len
                best_left = left

            # Remove left character
            left_char = s[left]
            window[left_char] -= 1
            if left_char in t_count and window[left_char] < t_count[left_char]:
                formed -= 1
            left += 1

    return "" if best_len == float("inf") else s[best_left:best_left + best_len]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    assert min_window("ADOBECODEBANC", "ABC") == "BANC"
    assert min_window("a", "a") == "a"
    assert min_window("a", "aa") == ""          # t longer than possible
    assert min_window("abc", "d") == ""         # char not in s
    assert min_window("aa", "aa") == "aa"       # duplicate in t
    assert min_window("", "a") == ""            # empty s

    print("  min_window: all tests passed")


if __name__ == "__main__":
    _test()
