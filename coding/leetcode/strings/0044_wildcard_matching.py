# 44. Wildcard Matching (Hard)
#
# Given string s and pattern p, implement wildcard matching with '?' and '*':
#   '?' matches any single character.
#   '*' matches any sequence of characters (including empty).
# Matching must cover the ENTIRE string.
#
# Compare with LC 10 (Regular Expression Matching) — same DP skeleton,
# one key difference in what '*' means.
#
# --------------------------------------------------------------------------
# LC 10 vs LC 44 — Side-by-Side
#
#   Aspect              LC 10 Regex             LC 44 Wildcard
#   ─────────────────   ────────────────────    ────────────────────
#   '*' stands for      zero+ of preceding x    any sequence (standalone)
#   Pattern unit        2 chars: "x*"           1 char: "*"
#   Any single char     '.'                     '?'
#   Harder?             Yes (coupled to x)      Simpler (independent)
#
#   Single-char wildcard:
#     LC 10 '.':  dp[i][j] = dp[i-1][j-1]                  (always)
#     LC 44 '?':  dp[i][j] = dp[i-1][j-1]                  (always)
#
#   Star — zero occurrences:
#     LC 10:  dp[i][j-2]   skip TWO chars "x*"
#     LC 44:  dp[i][j-1]   skip ONE char  "*"
#
#   Star — one more occurrence:
#     LC 10:  dp[i-1][j] AND s[i-1] matches p[j-2]   (must match preceding x)
#     LC 44:  dp[i-1][j]                              (any char, no constraint)
#
#   Base case dp[0][j] (empty string vs pattern):
#     LC 10:  handled automatically in loop (i=0 hits '*' branch, skips j-2)
#     LC 44:  needs explicit init: dp[0][j] = dp[0][j-1] if p[j-1]=='*'
#
# The entire difference reduces to ONE question:
#   Does '*' consume a preceding pattern char?
#     YES → skip j-2, check p[j-2] on one-more  (LC 10)
#     NO  → skip j-1, no char constraint          (LC 44)
#
# --------------------------------------------------------------------------
# Interview Framework for Pattern Matching DP
#
#   1. dp[i][j] = s[0:i] matches p[0:j]
#   2. Base: dp[0][0] = True
#            dp[0][j]: True only if all pattern chars so far are skip-able
#   3. Single-char wildcard → dp[i-1][j-1]
#   4. Multi-char '*':
#        Coupled (LC 10): zero=dp[i][j-2], one-more=dp[i-1][j] AND match p[j-2]
#        Standalone(LC44): zero=dp[i][j-1], one-more=dp[i-1][j]
#   5. Return dp[m][n]
# --------------------------------------------------------------------------


def is_match(s: str, p: str) -> bool:
    """
    Wildcard matching: '?' = any single char, '*' = any sequence (incl. empty).

    dp[i][j] = True if s[0:i] matches p[0:j].

    Transitions:
      p[j-1] == '*':
        zero occurrences: dp[i][j-1]          skip '*' (one pattern char)
        one more char:    dp[i-1][j]           consume s[i-1], stay at '*'
      p[j-1] == '?' or p[j-1] == s[i-1]:
        dp[i][j] = dp[i-1][j-1]               advance both

    Trace: s="adceb", p="*a*b"
           ""   *    a    *    b
      ""  [ T    T    F    F    F ]   '*' matches empty → dp[0][1]=dp[0][0]=T
      a   [ F    T    T    T    F ]
      d   [ F    T    F    T    F ]
      c   [ F    T    F    T    F ]
      e   [ F    T    F    T    F ]
      b   [ F    T    F    T    T ]   dp[5][4] = True ✓

    Complexity:
        Time:  O(m * n)
        Space: O(m * n)
    """
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    # Empty string matches leading '*'s (each '*' skips one pattern char)
    for j in range(1, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 1]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[i][j] = (dp[i][j - 1]      # zero: skip '*'
                             or dp[i - 1][j])  # one more: '*' eats s[i-1]
            elif p[j - 1] == '?' or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]   # single char match

    return dp[m][n]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all() -> None:
    cases = [
        # (s, p, expected)
        ("aa",    "a",     False),   # 'a' doesn't match "aa"
        ("aa",    "*",     True),    # '*' matches "aa"
        ("cb",    "?a",    False),   # '?' matches 'c' but 'a'!='b'
        ("adceb", "*a*b",  True),    # classic example
        ("acdcb", "a*c?b", False),
        ("",      "",      True),    # both empty
        ("",      "*",     True),    # '*' matches empty
        ("",      "**",    True),    # multiple '*' match empty
        ("",      "?",     False),   # '?' needs exactly one char
        ("a",     "?",     True),
        ("abc",   "???",   True),    # three '?' match three chars
        ("abc",   "**",    True),    # multiple '*' match anything
        ("abc",   "*bc",   True),
        ("abc",   "a*c",   True),
        ("abc",   "a*d",   False),
        ("abcde", "a*e",   True),
        ("abcde", "a*f",   False),
    ]
    for s, p, expected in cases:
        result = is_match(s, p)
        assert result == expected, \
            f"is_match({s!r}, {p!r}) = {result}, expected {expected}"


if __name__ == "__main__":
    test_all()
    print("All tests passed")
