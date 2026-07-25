# 7. Reverse Integer (Medium)
#
# Given a signed 32-bit integer x, return x with its digits reversed.
# If reversing x causes the value to go outside [-2^31, 2^31-1], return 0.
# Assume the environment does not allow storing 64-bit integers.
#
# Key interview angles:
#   1. Solve without string conversion (math: extract digits via % and //)
#   2. Handle Python's % behavior on negatives (strip sign first)
#   3. Overflow handling: post-check works in Python (arbitrary precision),
#      but pre-check is required in C++/Java (no overflow safety net)
#
# --------------------------------------------------------------------------
# Python-specific gotcha: % on negative numbers
#
#   Python uses floor division, not truncation toward zero:
#     -123 % 10  →  7   (not -3 like C++/Java)
#     -123 // 10 → -13  (not -12 like C++/Java)
#
#   Fix: strip the sign, work with abs(x), reapply sign at the end.
# --------------------------------------------------------------------------

INT_MAX =  2**31 - 1   #  2147483647
INT_MIN = -2**31       # -2147483648


# ---------------------------------------------------------------------------
# Version 1: Post-check (Python only — accepted on LeetCode)
#
# Works because Python integers have arbitrary precision.
# res * 10 + mod never actually overflows — it just becomes a big int.
# The overflow check runs AFTER the operation and correctly catches it.
#
# In C++/Java this would be WRONG: res * 10 + mod overflows BEFORE the
# check runs, causing undefined behavior or wraparound.
# ---------------------------------------------------------------------------

def reverse_postcheck(x: int) -> int:
    """
    Reverse digits of x. Check overflow after each multiply.
    Safe in Python (arbitrary precision); NOT safe in C++/Java.

    Complexity:
        Time:  O(log x)  — number of digits
        Space: O(1)
    """
    sign = 1 if x >= 0 else -1
    res, x = 0, abs(x)
    while x:
        x, mod = x // 10, x % 10
        res = res * 10 + mod
        if res > INT_MAX:         # post-check: fine in Python, wrong in C++/Java
            return 0
    return sign * res


# ---------------------------------------------------------------------------
# Version 2: Pre-check (language-agnostic, simulates 32-bit environment)
#
# Check BEFORE multiplying whether the next operation would overflow.
# Condition: res * 10 + digit > INT_MAX
#          → res > (INT_MAX - digit) // 10
#
# This is the correct approach for C++/Java where intermediate overflow
# is undefined behavior (C++) or silent wraparound (Java).
#
# Why post-check fails in C++/Java:
#   res = 214748364, digit = 8
#   res * 10 + 8 = 2147483648 → overflows to -2147483648 in Java
#   if (res > INT_MAX) → False (already wrong) → returns incorrect result
#
# Why pre-check works:
#   if (214748364 > (2147483647 - 8) // 10)
#      214748364 > 214748363  → True → return 0  ✓
# ---------------------------------------------------------------------------

def reverse(x: int) -> int:
    """
    Reverse digits of x. Check overflow BEFORE each multiply.
    Correct in all languages including C++/Java (simulates 32-bit).

    Complexity:
        Time:  O(log x)  — number of digits
        Space: O(1)
    """
    sign = 1 if x >= 0 else -1
    x = abs(x)
    rev = 0

    while x != 0:
        digit = x % 10
        x //= 10

        # Pre-check: would rev * 10 + digit exceed INT_MAX?
        if rev > (INT_MAX - digit) // 10:
            return 0

        rev = rev * 10 + digit

    return sign * rev


# ---------------------------------------------------------------------------
# Trivial (string-based) — shown to be improved upon
# ---------------------------------------------------------------------------

def reverse_trivial(x: int) -> int:
    """
    Convert to string, reverse, convert back. Violates the spirit of the
    problem ('no 64-bit integers' implies no string tricks either).
    Interviewer will ask for the math approach immediately after.

    Complexity:
        Time:  O(n)
        Space: O(n)
    """
    sign = -1 if x < 0 else 1
    rev = int(str(abs(x))[::-1]) * sign
    return rev if INT_MIN <= rev <= INT_MAX else 0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all() -> None:
    cases = [
        (123,        321),
        (-123,      -321),
        (120,         21),   # trailing zero dropped
        (0,            0),
        (1534236469,   0),   # overflows: reversed = 9646324351 > INT_MAX
        (-2147483648,  0),   # INT_MIN reversed overflows
        (1,            1),
        (-1,          -1),
    ]
    for x, expected in cases:
        for fn in (reverse_trivial, reverse_postcheck, reverse):
            result = fn(x)
            assert result == expected, \
                f"{fn.__name__}({x}) = {result}, expected {expected}"


if __name__ == "__main__":
    test_all()
    print("All tests passed")
