# 1071. Greatest Common Divisor of Strings (Easy)
#
# For two strings s and t, we say "t divides s" if and only if
# s = t + t + ... + t (t concatenated with itself one or more times).
# Given str1 and str2, return the largest string x that divides both.
#
# Key insight progression (interview path):
#   1. Brute force: try all prefix lengths (longest to shortest)
#   2. Validity shortcut: a solution exists iff str1+str2 == str2+str1
#   3. GCD insight: the answer length is gcd(len1, len2) — Euclidean from scratch


# ---------------------------------------------------------------------------
# Helper: divisibility check
# ---------------------------------------------------------------------------

def divides(t: str, s: str) -> bool:
    """
    Return True if t divides s (s = t repeated k times).

    Complexity:
        Time:  O(n)   — string multiply + compare
        Space: O(n)   — temporary repeated string
    """
    if len(s) % len(t) != 0:
        return False
    k = len(s) // len(t)
    return t * k == s


# ---------------------------------------------------------------------------
# Step 1: Brute Force — O(min(n,m) * (n+m))
# ---------------------------------------------------------------------------

def gcd_of_strings_brute(str1: str, str2: str) -> str:
    """
    Try every prefix of str1 (length min_len down to 1).
    A valid divisor must be a prefix of both strings.
    Return the first (longest) that divides both.

    Complexity:
        Time:  O(min(n,m) * (n+m))  — for each candidate length, divides() is O(n+m)
        Space: O(n+m)
    """
    min_len = min(len(str1), len(str2))
    for length in range(min_len, 0, -1):   # longest first → return immediately
        candidate = str1[:length]
        if divides(candidate, str1) and divides(candidate, str2):
            return candidate
    return ""


# ---------------------------------------------------------------------------
# Step 2 & 3: Observation + GCD — O(n+m)
#
# Observation: if any solution exists, both concatenation orders are equal:
#   str1 + str2 == str2 + str1
# Why: if x divides both, str1 = x*a, str2 = x*b, so str1+str2 = x*(a+b) = str2+str1.
#
# GCD insight: all valid lengths must divide each other (if p and q both work,
# then gcd(p, q) also works). So the longest valid length is gcd(len1, len2).
# We only need to check ONE candidate: str1[:gcd(len1, len2)].
# ---------------------------------------------------------------------------

def my_gcd(a: int, b: int) -> int:
    """
    Euclidean algorithm: gcd(a, b) = gcd(b, a % b), base case gcd(a, 0) = a.

    Why it works: any common divisor of a and b also divides a % b = a - k*b,
    so the set of common divisors is unchanged when replacing (a,b) with (b, a%b).

    Trace examples:
        gcd(6, 4): (6,4) -> (4,2) -> (2,0) -> 2
        gcd(9, 6): (9,6) -> (6,3) -> (3,0) -> 3
        gcd(7, 3): (7,3) -> (3,1) -> (1,0) -> 1

    Complexity:
        Time:  O(log(min(a, b)))  — proven by Lame's theorem; steps <= 5 * digits(min)
        Space: O(1)
    """
    while b:
        a, b = b, a % b
    return a


def gcd_of_strings(str1: str, str2: str) -> str:
    """
    Return the largest string x such that x divides both str1 and str2.

    Algorithm:
        1. Check if str1+str2 == str2+str1 (necessary and sufficient for a solution).
        2. If yes, answer is str1[:gcd(len1, len2)].

    Args:
        str1: First input string (1 <= len <= 1000, uppercase letters).
        str2: Second input string.

    Returns:
        Largest dividing string, or "" if none exists.

    Complexity:
        Time:  O(n+m)      — concatenation check dominates; gcd is O(log min)
        Space: O(n+m)      — temporary concat strings for validity check
    """
    if str1 + str2 != str2 + str1:
        return ""
    return str1[:my_gcd(len(str1), len(str2))]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_divides() -> None:
    assert divides("ABC", "ABCABC") is True
    assert divides("AB", "ABABAB") is True
    assert divides("AB", "ABCABC") is False
    assert divides("AAA", "AAAAAB") is False


def test_gcd() -> None:
    assert my_gcd(6, 4) == 2
    assert my_gcd(9, 6) == 3
    assert my_gcd(7, 3) == 1
    assert my_gcd(12, 8) == 4
    assert my_gcd(1, 1) == 1


def test_brute_and_optimal() -> None:
    cases = [
        ("ABCABC", "ABC", "ABC"),
        ("ABABAB", "ABAB", "AB"),
        ("LEET", "CODE", ""),
        ("AAAAAB", "AAA", ""),
        ("A", "A", "A"),
        ("AAAA", "AA", "AA"),
        ("ABABABAB", "ABAB", "ABAB"),
    ]
    for str1, str2, expected in cases:
        assert gcd_of_strings_brute(str1, str2) == expected, \
            f"brute failed: ({str1!r}, {str2!r}) -> expected {expected!r}"
        assert gcd_of_strings(str1, str2) == expected, \
            f"optimal failed: ({str1!r}, {str2!r}) -> expected {expected!r}"


if __name__ == "__main__":
    test_divides()
    test_gcd()
    test_brute_and_optimal()
    print("All tests passed")
