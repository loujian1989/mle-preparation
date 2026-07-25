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
# --- FORWARD direction (easy) ---
# If x divides both: str1 = x*a, str2 = x*b
#   str1+str2 = x*(a+b) = str2+str1  →  concat order doesn't matter.
#
# --- REVERSE direction (the key proof) ---
# Given: str1+str2 == str2+str1, prove str1[:gcd(n,m)] divides both.
# Assume n = len(str1) >= m = len(str2).
#
# Step 1 — str2 is a prefix of str1:
#   Compare first m chars of each side:
#     LHS[:m] = str1[0:m]   (n >= m so str1 is long enough)
#     RHS[:m] = str2
#   → str1[0:m] == str2  →  str2 is a prefix of str1.
#   Write str1 = str2 + str1[m:].
#
# Step 2 — Recurse on the remainder:
#   Substitute str1 = str2 + str1[m:] into the original equality:
#     (str2 + str1[m:]) + str2 == str2 + (str2 + str1[m:])
#   Cancel str2 from both fronts:
#     str1[m:] + str2 == str2 + str1[m:]       ← same property, smaller strings
#
# Step 3 — Base case:
#   Lengths reduce to (g, g) where g = gcd(n, m); both strings equal → done.
#   The base string (length g) is the repeating unit that builds every
#   intermediate string all the way back up to str1 and str2.
#
# Analogy with Euclidean algorithm:
#   gcd(n, m) = gcd(m, n-m)   ↔   gcd_str(s1, s2) = gcd_str(s2, s1[m:])
#   "subtract smaller"         ↔   "peel one copy of str2 from str1"
#   base: gcd(g, g) = g        ↔   base: both strings equal → trivially divides
#
# Concrete trace — str1="ABABAB", str2="ABAB":
#   (ABABAB, ABAB): ABAB is prefix of ABABAB → remainder "AB"
#   (ABAB,   AB  ): AB   is prefix of ABAB   → remainder "AB"
#   (AB,     AB  ): equal → base, unit = "AB" = str1[:gcd(6,4)=2]  ✓
#
# Conclusion: we only need ONE check (concat equality) and ONE candidate
#   (str1[:gcd(len1, len2)]); the proof guarantees it divides both.
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
