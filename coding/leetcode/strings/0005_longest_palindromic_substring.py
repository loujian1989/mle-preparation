# 5. Longest Palindromic Substring (Medium)
#
# Given a string s, return the longest palindromic substring in s.
#
# Interview progression:
#   Step 1: Brute force — try all substrings, check each  O(n³) T / O(1) S
#   Step 2: DP — build palindrome table bottom-up          O(n²) T / O(n²) S
#   Step 3: Expand around center — O(1) space win          O(n²) T / O(1) S
#   Step 4: Manacher's algorithm (mention only)            O(n)  T / O(n) S
#
# Key insight for Step 3:
#   Every palindrome has a center. 2n-1 possible centers exist:
#     - n   odd-length  centers (each character)
#     - n-1 even-length centers (each gap between adjacent characters)
#   Expand outward from each center while s[l] == s[r].
#   Track the longest window seen.


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def is_palindrome(s: str, l: int, r: int) -> bool:
    """Check if s[l:r+1] is a palindrome in O(r-l) time, O(1) space."""
    while l < r:
        if s[l] != s[r]:
            return False
        l += 1
        r -= 1
    return True


# ---------------------------------------------------------------------------
# Step 1: Brute Force
# ---------------------------------------------------------------------------

def longest_palindrome_brute(s: str) -> str:
    """
    Try all O(n²) substrings; palindrome-check each in O(n).

    Complexity:
        Time:  O(n³)
        Space: O(1)
    """
    n, best = len(s), s[0]
    for i in range(n):
        for j in range(i + 1, n):
            if j - i + 1 > len(best) and is_palindrome(s, i, j):
                best = s[i:j + 1]
    return best


# ---------------------------------------------------------------------------
# Step 2: Dynamic Programming
# ---------------------------------------------------------------------------

def longest_palindrome_dp(s: str) -> str:
    """
    dp[i][j] = True if s[i:j+1] is a palindrome.

    Traversal: i from n-1 down to 0, j from i up to n-1.
    This guarantees dp[i+1][j-1] is already filled when we compute dp[i][j]
    (because i+1 > i was processed in an earlier outer iteration).

    Single condition covers all cases:
      i == j             → length 1, always a palindrome
      i+1 == j           → length 2, palindrome iff s[i]==s[j]
                           (short-circuits before reading dp[i+1][j-1])
      dp[i+1][j-1]       → length 3+, inner substring already computed

    Complexity:
        Time:  O(n²)
        Space: O(n²)  ← downside vs. expand-around-center
    """
    n = len(s)
    dp = [[False] * n for _ in range(n)]
    max_len, start = 0, -1

    for i in range(n - 1, -1, -1):
        for j in range(i, n):
            if i == j or (s[i] == s[j] and (i + 1 == j or dp[i + 1][j - 1])):
                dp[i][j] = True
                if j - i + 1 > max_len:
                    max_len, start = j - i + 1, i

    return s[start:start + max_len]


# ---------------------------------------------------------------------------
# Step 3: Expand Around Center  ← preferred answer
# ---------------------------------------------------------------------------

def _expand(s: str, l: int, r: int) -> tuple[int, int]:
    """
    Expand outward from center (l, r) while characters match.
    Returns (l, r) of the largest palindrome centered here.

    Note: after the loop, we've gone one step past the valid boundary,
    so we return (l+1, r-1).
    """
    while l >= 0 and r < len(s) and s[l] == s[r]:
        l -= 1
        r += 1
    return l + 1, r - 1


def longest_palindrome(s: str) -> str:
    """
    For each of 2n-1 centers, expand and track the longest palindrome.

    Trace on "babad":
      i=0 'b': odd → "b",   even(b,a) mismatch → ""
      i=1 'a': odd → "bab", even(a,b) mismatch → ""   ← best so far
      i=2 'b': odd → "aba", even(b,a) mismatch → ""   (same length, keep "bab")
      i=3 'a': odd → "a",   even(a,d) mismatch → ""
      i=4 'd': odd → "d",   no even center
      → "bab"  ✓

    Complexity:
        Time:  O(n²)  — 2n-1 centers, each expands at most O(n)
        Space: O(1)   — only track indices, no table
    """
    start, end = 0, 0
    for i in range(len(s)):
        l1, r1 = _expand(s, i, i)       # odd-length: center at i
        l2, r2 = _expand(s, i, i + 1)   # even-length: center between i and i+1
        if r1 - l1 > end - start:
            start, end = l1, r1
        if r2 - l2 > end - start:
            start, end = l2, r2
    return s[start:end + 1]


# ---------------------------------------------------------------------------
# Step 4: Manacher's Algorithm — O(n) (for awareness)
#
# Idea: preprocess string with separators → "#b#a#b#a#d#"
# Maintain array P where P[i] = radius of palindrome centered at i.
# Use a "current rightmost palindrome" window [c, r] to mirror previously
# computed radii and avoid redundant comparisons.
# Any center inside the window has a known mirror — use it as a lower bound,
# then try to expand further.
# Time: O(n) — each character is the right boundary at most once.
# Space: O(n) — for the transformed string and radius array.
# Usually considered advanced; mention to show awareness; implement only if asked.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _is_valid_palindrome(result: str, s: str) -> bool:
    """Check result is in s, is a palindrome, and nothing longer exists."""
    if result not in s:
        return False
    if result != result[::-1]:
        return False
    n = len(s)
    for i in range(n):
        for j in range(i + len(result), n + 1):
            sub = s[i:j]
            if sub == sub[::-1] and len(sub) > len(result):
                return False
    return True


def test_all() -> None:
    cases = [
        ("babad", {"bab", "aba"}),   # two valid answers
        ("cbbd",  {"bb"}),
        ("a",     {"a"}),
        ("ac",    {"a", "c"}),
        ("racecar", {"racecar"}),
        ("abcba", {"abcba"}),
        ("aacabdkacaa", {"aca"}),
    ]
    for s, valid_set in cases:
        for fn in (longest_palindrome_brute, longest_palindrome_dp, longest_palindrome):
            result = fn(s)
            assert result in valid_set or _is_valid_palindrome(result, s), \
                f"{fn.__name__}({s!r}) = {result!r}, not in {valid_set}"


if __name__ == "__main__":
    test_all()
    print("All tests passed")
