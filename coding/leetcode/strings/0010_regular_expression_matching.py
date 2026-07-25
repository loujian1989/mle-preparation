# 10. Regular Expression Matching (Hard)
#
# Given string s and pattern p, implement regex matching with '.' and '*'.
#   '.' matches any single character.
#   '*' matches zero or more of the preceding element.
# Return True if p matches the ENTIRE string s.
#
# Approach: 2D DP
#   dp[i][j] = True if s[0:i] fully matches p[0:j]
#   Goal: dp[m][n]
#
# --------------------------------------------------------------------------
# DP Definition & Base Cases
#
#   dp[0][0] = True    empty string matches empty pattern
#   dp[i][0] = False   non-empty string can't match empty pattern (i > 0)
#   dp[0][j]:          empty string vs pattern p[0:j]
#                      only True if pattern is all "x*" pairs (zero-match each)
#                      e.g. "a*b*" → dp[0][2]=True, dp[0][4]=True
#                      handled automatically in the '*' branch of the loop
#
# --------------------------------------------------------------------------
# Transitions
#
# Case 1: p[j-1] != '*'  (literal char or '.')
#
#   dp[i][j] = i > 0
#              and dp[i-1][j-1]           # previous state must match
#              and (p[j-1] == '.'         # '.' matches anything
#                   or s[i-1] == p[j-1])  # exact char match
#
# Case 2: p[j-1] == '*'  (always refers to p[j-2], the preceding element)
#
#   Two sub-choices:
#
#   a) Use '*' ZERO times — skip "x*" entirely:
#        dp[i][j-2]
#
#   b) Use '*' ONE MORE time — consume s[i-1] with "x*":
#        dp[i-1][j]                              # stay at j (x* can match more)
#        and (p[j-2] == '.' or p[j-2] == s[i-1])# s[i-1] must match x
#
#   dp[i][j] = (j > 1 and dp[i][j-2])           # zero occurrences
#            or (i > 0 and dp[i-1][j]            # one more occurrence
#               and (p[j-2] == '.' or p[j-2] == s[i-1]))
#
# --------------------------------------------------------------------------
# Why dp[i-1][j] (not dp[i-1][j-2]) for "one more" case?
#
#   x* can consume MULTIPLE characters — we don't move past the pattern.
#   dp[i-1][j] means "x* already matched s[0:i-1] using p[0:j],
#   and now x* matches one more char s[i-1]".
#   This chains backwards, letting x* consume as many chars as needed.
#
#   Example: s="aaa", p="a*"
#     dp[3][2]: one more → dp[2][2] AND 'a'=='a'
#       dp[2][2]: one more → dp[1][2] AND 'a'=='a'
#         dp[1][2]: one more → dp[0][2] AND 'a'=='a'
#           dp[0][2] = True  (zero occurrences of a*)
#         → True
#       → True
#     → True  ✓
#
# --------------------------------------------------------------------------
# Full trace: s="aab", p="c*a*b"
#
#        ""   c    *    a    *    b
#   ""  [ T    F    T    F    T    F ]   c* and a* each match empty
#   a   [ F    F    F    T    T    F ]
#   a   [ F    F    F    F    T    F ]
#   b   [ F    F    F    F    F    T ]   dp[3][5] = True ✓
#
#   dp[1][4] (s[0]='a', p[3]='*', p[2]='a'):
#     zero: dp[1][2]=False
#     one more: dp[0][4]=True AND 'a'=='a' → True
#
#   dp[3][5] (s[2]='b', p[4]='b', not '*'):
#     dp[2][4]=True AND 'b'=='b' → True ✓
# --------------------------------------------------------------------------


def is_match(s: str, p: str) -> bool:
    """
    2D DP regex matching with '.' and '*'.

    Args:
        s: Input string (lowercase letters only).
        p: Pattern (lowercase letters, '.', '*').

    Returns:
        True if p matches the entire string s.

    Complexity:
        Time:  O(m * n)  — fill every cell of the dp table once
        Space: O(m * n)  — dp table
    """
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    for i in range(0, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] != '*':
                # Case 1: literal char or '.'
                dp[i][j] = (i > 0
                             and dp[i - 1][j - 1]
                             and (p[j - 1] == '.' or s[i - 1] == p[j - 1]))
            else:
                # Case 2: '*' — zero occurrences OR one more occurrence
                dp[i][j] = (j > 1 and dp[i][j - 2]) \
                            or (i > 0
                                and dp[i - 1][j]
                                and (p[j - 2] == '.' or p[j - 2] == s[i - 1]))

    return dp[m][n]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all() -> None:
    cases = [
        # (s, p, expected)
        ("aa",  "a",    False),   # 'a' doesn't match "aa"
        ("aa",  "a*",   True),    # 'a*' matches "aa" (two a's)
        ("ab",  ".*",   True),    # '.*' matches anything
        ("aab", "c*a*b",True),    # c* zero times, a* twice, b once
        ("",    "",     True),    # both empty
        ("",    "a*",   True),    # a* zero times
        ("",    "a*b*", True),    # chained zero matches
        ("a",   ".",    True),    # '.' matches single char
        ("a",   "a*",   True),    # a* matches one a
        ("aaa", "a*",   True),    # a* matches multiple
        ("aaa", "a",    False),   # single 'a' vs "aaa"
        ("ab",  "a.",   True),    # 'a' then any char
        ("ab",  "a*b",  True),    # a* zero times, then b
        ("abc", "a.c",  True),    # '.' matches b
        ("abc", "a*bc", True),    # a once via a*, then bc
        ("mississippi", "mis*is*p*.", False),  # classic hard case
    ]
    for s, p, expected in cases:
        result = is_match(s, p)
        assert result == expected, \
            f"is_match({s!r}, {p!r}) = {result}, expected {expected}"


if __name__ == "__main__":
    test_all()
    print("All tests passed")
