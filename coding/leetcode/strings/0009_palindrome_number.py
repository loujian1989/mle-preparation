# 9. Palindrome Number (Easy)
#
# Given an integer x, return true if x is a palindrome, false otherwise.
# Follow-up: solve it without converting to a string.
#
# Approaches:
#   Step 1: String conversion — trivial, O(n) space
#   Step 2: Half-reverse    — O(1) space, no overflow risk  ← preferred
#
# --------------------------------------------------------------------------
# Why NOT reverse the full number (like LC 7)?
#   Reversing the full number risks overflow for large x (e.g. x=1000000001).
#   In a true 32-bit environment you'd overflow before finishing.
#
# Key insight: only reverse the SECOND HALF and compare with the first half.
#   - Safe: reversed half has at most n//2 digits, well within 32-bit range.
#   - Correct: a palindrome satisfies first_half == reverse(second_half).
#
# Stop condition: keep popping digits from x into rev until rev >= x.
#   At that point rev holds exactly half the digits (even length)
#   or half+1 digits (odd length — middle digit ends up in rev).
#
# Even length: x == rev           (1221 → x=12, rev=12)
# Odd  length: x == rev // 10    (121  → x=1,  rev=12, drop middle digit)
# --------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Step 1: String conversion (trivial)
# ---------------------------------------------------------------------------

def is_palindrome_str(x: int) -> bool:
    """
    Convert to string and check if it reads the same forwards and backwards.
    Negative numbers fail immediately (str includes '-').

    Complexity:
        Time:  O(n)
        Space: O(n)  ← string allocation
    """
    s = str(x)
    return s == s[::-1]


# ---------------------------------------------------------------------------
# Step 2: Half-reverse — no string, no overflow  ← preferred
# ---------------------------------------------------------------------------

def is_palindrome(x: int) -> bool:
    """
    Reverse only the second half of x; compare with the first half.

    Early exits:
      x < 0              → never a palindrome (leading '-' breaks symmetry)
      x % 10 == 0, x ≠ 0 → can't be palindrome (reversed would have leading 0)

    Trace examples:
      x=121:  rev: 0→1 (x=12) → 12 (x=1). rev>x: stop. 1 == 12//10=1  ✓
      x=1221: rev: 0→1 (x=122) → 12 (x=12). rev==x: stop. 12==12      ✓
      x=123:  rev: 0→3 (x=12) → 32 (x=1). rev>x: stop. 1 ≠ 32//10=3  ✗
      x=10:   ends in 0, x≠0 → False                                    ✓
      x=0:    0%10==0 but x==0 → allowed. loop skips. 0==0              ✓

    Complexity:
        Time:  O(log x)  — number of digits / 2
        Space: O(1)
    """
    if x < 0 or (x % 10 == 0 and x != 0):
        return False

    rev = 0
    while x > rev:                   # stop when rev has consumed half the digits
        rev = rev * 10 + x % 10
        x //= 10

    return x == rev or x == rev // 10


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all() -> None:
    cases = [
        (121,        True),
        (-121,       False),   # negative
        (10,         False),   # ends in 0
        (0,          True),    # 0 is a palindrome
        (1,          True),    # single digit
        (11,         True),    # even length
        (1221,       True),    # even length
        (12321,      True),    # odd length
        (123,        False),
        (1000000001, True),    # large — would overflow if full reverse attempted
        (2147483647, False),   # INT_MAX
    ]
    for x, expected in cases:
        for fn in (is_palindrome_str, is_palindrome):
            result = fn(x)
            assert result == expected, \
                f"{fn.__name__}({x}) = {result}, expected {expected}"


if __name__ == "__main__":
    test_all()
    print("All tests passed")
