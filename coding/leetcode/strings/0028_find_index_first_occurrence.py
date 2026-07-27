# 28. Find the Index of the First Occurrence in a String (Easy)
#
# Given strings haystack and needle, return the index of the first
# occurrence of needle in haystack, or -1 if not found.
#
# Approaches:
#   Step 1: Brute force      O(n·m) T / O(1) S   — slide and compare
#   Step 2: Rabin-Karp       O(n+m) avg T / O(1) S — rolling hash
#   Step 3: KMP              O(n+m) T / O(m) S   — failure function, never re-scan
#
# n = len(haystack), m = len(needle)
#
# --------------------------------------------------------------------------
# Algorithm comparison
#
#   Algorithm      Time          Space         Key idea
#   ────────────   ──────────    ───────────   ──────────────────────────────
#   Brute force    O(n·m)        O(1)          slide window, compare each char
#   Rabin-Karp     O(n+m) avg    O(1)          rolling hash, full compare on hit
#   KMP            O(n+m)        O(m)          failure fn, never re-scan haystack
#   Boyer-Moore    O(n/m) best   O(alphabet)   right-to-left scan, large skips
#
# KMP is the standard interview answer for O(n+m).
# Rabin-Karp is preferred when you need to find ANY of k patterns
# (compute k hashes once, compare against rolling window).
#
# --------------------------------------------------------------------------
# KMP — how the failure function (lps) works
#
#   lps[j] = length of longest proper prefix of needle[0:j+1]
#            that is also a suffix.
#
#   "Proper" means not the full string itself.
#   One-line intuition: for each prefix of needle, how much does the
#   START overlap with the END?
#
#   needle = "aabaa" — position by position:
#
#   j=0 → substring "a"
#     No proper prefix exists for a single character.
#     lps[0] = 0
#
#   j=1 → substring "aa"
#     a  a
#     ↑  ↑
#     │  └── suffix "a" (last 1 char)
#     └───── prefix "a" (first 1 char)
#     "a" appears at both start AND end → length 1
#     lps[1] = 1
#
#   j=2 → substring "aab"
#     Prefix candidates: "a", "aa"
#     End chars: "b" (len 1), "ab" (len 2) — no match with any prefix
#     lps[2] = 0
#
#   j=3 → substring "aaba"
#     a  a  b  a
#     ↑           ← prefix "a"
#                ↑ ← suffix "a" (last char)
#     "a" matches at start and end (length 1).
#     "aa" vs end "ba" → no.
#     lps[3] = 1
#
#   j=4 → substring "aabaa"
#     a  a  b  a  a
#     ↑  ↑           ← prefix "aa"
#              ↑  ↑  ← suffix "aa" (last 2 chars)
#     "a"  matches (length 1)
#     "aa" matches (length 2) ← longest
#     "aab" vs "baa" → no
#     lps[4] = 2
#
#   Summary table:
#     needle:     a    a    b    a    a
#     substring:  a    aa   aab  aaba aabaa
#     overlap:    –    a    –    a    aa
#     lps:        0    1    0    1    2
#
#   On mismatch at pattern position j, jump to lps[j-1].
#   Meaning: you've already matched lps[j-1] characters from the start
#   of the pattern — no need to recheck them.
#   Haystack pointer i NEVER moves backward → O(n) haystack pass.
#
# --------------------------------------------------------------------------
# Rabin-Karp — rolling hash
#
#   Hash the needle window once. Slide across haystack:
#     - Drop leftmost character: subtract its contribution
#     - Add rightmost character: append to hash
#   O(1) per slide. Full string comparison ONLY on hash match (to handle
#   collisions).
#
#   Rolling update:
#     new_hash = (old_hash - char_leaving * BASE^(m-1)) * BASE + char_entering
#
#   MOD prevents integer overflow and keeps arithmetic fast.
#
# --------------------------------------------------------------------------
# Why KMP is correct — full proof intuition
#
#   The problem with brute force:
#     haystack: a  a  a  a  a  b
#     needle:   a  a  a  a  b
#                           ^ mismatch at j=4, i=4
#     Brute force restarts at i=1, j=0 — but we JUST learned haystack[0..3]="aaaa".
#     Starting at i=1 will immediately re-match "aaa" — wasted work.
#
#   The core question KMP answers on every mismatch:
#     After matching needle[0..j-1] against haystack[i-j..i-1], and mismatching
#     at haystack[i] vs needle[j] — what is the next possible start?
#
#     Any valid match starting at position s (where i-j < s < i) requires:
#       needle[0..i-1-s] == haystack[s..i-1]
#     We know haystack[i-j..i-1] == needle[0..j-1], so substituting:
#       needle[0..i-1-s] == needle[s-(i-j)..j-1]
#     This means: a PREFIX of needle must equal a SUFFIX of needle[0..j-1].
#     The longest such prefix-suffix is lps[j-1].
#
#     Jumping to j = lps[j-1] tries the longest possible restart.
#     Every shorter restart is subsumed — no valid start is skipped.
#
#   Why i never moves backward:
#     lps[j-1] < j always (PROPER prefix — can't equal full length).
#     So the new window start i - lps[j-1] always moves RIGHT.
#     i stays put, j shrinks. Window shifts forward.
#
#   Concrete trace:
#     haystack: a  a  a  a  a  b
#     needle:   a  a  a  a  b        lps = [0, 1, 2, 3, 0]
#
#     i=0,j=0: 'a'=='a' → i=1, j=1
#     i=1,j=1: 'a'=='a' → i=2, j=2
#     i=2,j=2: 'a'=='a' → i=3, j=3
#     i=3,j=3: 'a'=='a' → i=4, j=4
#     i=4,j=4: 'a'!='b' → j = lps[3] = 3    ← i stays at 4!
#     i=4,j=3: 'a'=='a' → i=5, j=4
#     i=5,j=4: 'b'=='b' → i=6, j=5=m → FOUND at 6-5=1 ✓
#
#     At the mismatch: we matched "aaaa". lps[3]=3 means "aaa" is already
#     a suffix of what we matched — the new window at position 1 already has
#     "aaa" in place. Continue from j=3. Haystack never rescanned.
#
#   Why the lps fallback chain is safe:
#     When lps[j-1] still doesn't match, fall back again: j = lps[lps[j-1]-1].
#     Each fallback tries a shorter valid prefix-suffix. The set of all borders
#     of a string forms a chain (each is a border of the previous) — no valid
#     restart is ever skipped.
#
#   Why lps construction mirrors the search:
#     The build loop uses the same fallback logic:
#       if needle[i] == needle[length]: extend match
#       elif length: length = lps[length-1]   ← same chain fallback
#     Correct by induction on the border chain structure.
#
#   30-second interview summary:
#     "KMP precomputes for each pattern position the longest prefix that is
#      also a suffix of the matched portion. On mismatch at j, jump to lps[j-1]
#      — that many characters are already implicitly matched at the new window
#      start. Any skipped starting position would require a shorter prefix-suffix,
#      which is subsumed. Haystack pointer never goes backward → O(n) scan."
#
# --------------------------------------------------------------------------
# Interview extensions
#
#   Extension                         Algorithm            Complexity
#   ──────────────────────────────    ─────────────────    ─────────────────────
#   Find ALL occurrences              KMP (don't stop)     O(n+m)
#   Find ANY of k patterns            Aho-Corasick         O(n + sum(|p_i|))
#   k mismatches allowed              Bitap                O(n·m/word_size)
#   Pattern is a regex                NFA simulation       O(n·m)
#   2D pattern in 2D grid             KMP row+col          O(R·C·r·c)
#   Distributed haystack              Rabin-Karp at seams  hash overlapping windows
#
#   Aho-Corasick (Staff-level signal):
#     Build a trie of all k patterns. Add failure links (cross-pattern lps).
#     Single O(n) pass finds ALL k patterns simultaneously.
#     Used in: spam filters, antivirus, network intrusion detection.
# --------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Step 1: Brute force
# ---------------------------------------------------------------------------

def str_str_brute(haystack: str, needle: str) -> int:
    """
    Slide a window of length m across haystack, compare character by character.

    Trace: haystack="sadbutsad", needle="sad"
      i=0: haystack[0:3]="sad" == "sad" → return 0 ✓

    Complexity:
        Time:  O(n·m)  — n windows, each comparison up to m chars
        Space: O(1)
    """
    n, m = len(haystack), len(needle)
    for i in range(n - m + 1):
        if haystack[i:i + m] == needle:
            return i
    return -1


# ---------------------------------------------------------------------------
# Step 2: Rabin-Karp — rolling hash
# ---------------------------------------------------------------------------

def str_str_rabin_karp(haystack: str, needle: str) -> int:
    """
    Hash needle and the first window. Slide: drop leftmost, add rightmost.
    Full comparison only on hash match (handles collisions).

    Hash formula: sum(ord(c) * BASE^(m-1-i) for i,c in needle) mod MOD
    Rolling update:
      new = (old - ord(leaving) * power) * BASE + ord(entering)  (mod MOD)

    Complexity:
        Time:  O(n+m) average  — O(n·m) worst case on hash collisions
        Space: O(1)
    """
    n, m = len(haystack), len(needle)
    if m > n:
        return -1

    BASE, MOD = 26, 10 ** 9 + 7

    # Compute BASE^(m-1) — the weight of the leftmost character
    power = 1
    for _ in range(m - 1):
        power = power * BASE % MOD

    # Initial hashes for needle and first window
    needle_hash = window_hash = 0
    for i in range(m):
        needle_hash = (needle_hash * BASE + ord(needle[i])) % MOD
        window_hash = (window_hash * BASE + ord(haystack[i])) % MOD

    if needle_hash == window_hash and haystack[:m] == needle:
        return 0

    # Slide the window
    for i in range(1, n - m + 1):
        window_hash = (window_hash - ord(haystack[i - 1]) * power) % MOD
        window_hash = (window_hash * BASE + ord(haystack[i + m - 1])) % MOD
        if window_hash == needle_hash and haystack[i:i + m] == needle:
            return i

    return -1


# ---------------------------------------------------------------------------
# Step 3: KMP — O(n+m) guaranteed
# ---------------------------------------------------------------------------

def _build_lps(needle: str) -> list[int]:
    """
    Build the failure function (lps = longest proper prefix that is suffix).

    What `length` means:
      length = how many characters of the prefix we've already matched
               against the current suffix position.
      At start: length=0 means "no overlap yet."
      i starts at 1 (not 0) because lps[0]=0 always — a single character
      has no proper prefix, nothing to overlap.

    What needle[i] == needle[length] means at length=0:
      needle[length] = needle[0] = first character of needle (the prefix).
      needle[i]                  = current character (the suffix candidate).
      If they match → overlap of length 1 exists for needle[0..i].

      IMPORTANT: lps[i] is NOT about the full needle — it is about the
      SUBSTRING needle[0..i]. At i=1, the substring is just "aa", not "aabaa".

        full needle:  a  a  b  a  a
                         ^
                         i=1, substring = needle[0..1] = "aa"

      For substring "aa":
        needle[0] = 'a'  → first char of "aa"  (prefix side)
        needle[1] = 'a'  → last  char of "aa"  (suffix side)
      Both ends ARE checked. They happen to be needle[0] and needle[i]
      because i always points to the last character of the current substring.

      The last char of the full needle (needle[4]) is only checked when i=4,
      which is exactly when we compute lps for the full string "aabaa".

      General rule — for any substring needle[0..i]:
        first character = always needle[0]
        last  character = always needle[i]    ← i IS the last index

      So needle[length] == needle[i] asks: "does the prefix end match the
      suffix end of the current substring?" Both ends, one comparison.

      Table:
        i   substring   checking              question
        1   "aa"        needle[0] vs [1]      first == last of "aa"?
        2   "aab"       needle[0] vs [2]      first == last of "aab"?
        3   "aaba"      needle[0] vs [3]      first == last of "aaba"?
        4   "aabaa"     needle[1] vs [4]      second == last of "aabaa"? (length=1)

    Why length += 1 BEFORE lps[i] = length (not after):
      The two lines are equivalent to writing lps[i] = length + 1 then length += 1.
      Increment first is just a code style — the assignment captures the NEW value.
        length += 1       # extend overlap by 1
        lps[i] = length   # record the extended length  (= old_length + 1)
      Same result, different order.

    Key invariant:
      At the start of each iteration, needle[0..length-1] already matches
      needle[i-length..i-1]. So `length` simultaneously marks:
        - where the prefix currently ends  (needle[length] = next prefix char)
        - how long the current overlap is  (length chars already matched)

      Checking needle[i] == needle[length] tests BOTH sides in one comparison:
        needle[length] = next char of the prefix
        needle[i]      = next char of the suffix candidate
      If they match, the overlap extends by 1. This is how a single comparison
      confirms both prefix and suffix alignment.

    Why fallback length = lps[length-1] (not 0):
      When needle[i] != needle[length], the prefix of length `length` can't
      extend. But needle[0..length-1] has its own internal overlap: lps[length-1].
      That shorter prefix is still aligned on the suffix side (a suffix of a suffix
      is still a suffix). So we try to extend from there instead of restarting.

    Visual — needle = "aabaa", step i=4, length=1:
      We know needle[0]=='a' matches needle[3]=='a' (that's why length=1).
      Now extend by one:
        prefix: a  a          needle[length=1] = 'a'  (next prefix char)
                0  1
        suffix: a  a          needle[i=4]      = 'a'  (next suffix char)
                3  4
      'a'=='a' → prefix "aa" matches suffix "aa". lps[4]=2. ✓

    Trace: needle = "aabaa"
      i=1, length=0: needle[1]='a'==needle[0]='a' → lps[1]=1, length=1, i=2
      i=2, length=1: needle[2]='b'!=needle[1]='a' → length=lps[0]=0
      i=2, length=0: needle[2]='b'!=needle[0]='a' → lps[2]=0, i=3
      i=3, length=0: needle[3]='a'==needle[0]='a' → lps[3]=1, length=1, i=4
      i=4, length=1: needle[4]='a'==needle[1]='a' → lps[4]=2, length=2, i=5
      lps = [0, 1, 0, 1, 2]
    """
    m = len(needle)
    lps = [0] * m
    length, i = 0, 1
    while i < m:
        if needle[i] == needle[length]:
            length += 1
            lps[i] = length
            i += 1
        elif length:
            length = lps[length - 1]   # fall back without advancing i
        else:
            lps[i] = 0
            i += 1
    return lps


def str_str(haystack: str, needle: str) -> int:
    """
    KMP: build lps in O(m), then scan haystack in O(n) — i never goes backward.

    On match: advance both i and j.
    On full pattern match (j==m): found, return i-j.
    On mismatch: if j>0 jump j to lps[j-1]; else advance i.

    Trace: haystack="aabaabaab", needle="aab"
      lps = [0,1,0]
      i=0,j=0: 'a'=='a' → i=1,j=1
      i=1,j=1: 'a'=='a' → i=2,j=2
      i=2,j=2: 'b'=='b' → i=3,j=3 → j==m → return 3-3=0 ✓

    Complexity:
        Time:  O(n+m)  — lps build O(m) + haystack scan O(n)
        Space: O(m)    — lps array
    """
    n, m = len(haystack), len(needle)
    if not needle:
        return 0
    lps = _build_lps(needle)
    i = j = 0
    while i < n:
        if haystack[i] == needle[j]:
            i += 1
            j += 1
        if j == m:
            return i - j
        elif i < n and haystack[i] != needle[j]:
            if j:
                j = lps[j - 1]   # don't advance i — retry with shorter prefix
            else:
                i += 1
    return -1


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all() -> None:
    cases = [
        ("sadbutsad",  "sad",   0),
        ("leetcode",   "leeto", -1),
        ("",           "",      0),
        ("a",          "",      0),
        ("a",          "a",     0),
        ("a",          "b",     -1),
        ("aabaabaab",  "aab",   0),
        ("mississippi","issip", 4),
        ("hello",      "ll",    2),
        ("aaaaaa",     "bba",   -1),
        ("aabaa",      "aabaa", 0),   # needle equals haystack
        ("abcabc",     "abc",   0),
        ("abcabc",     "cab",   2),
    ]
    for haystack, needle, expected in cases:
        for fn in (str_str_brute, str_str_rabin_karp, str_str):
            result = fn(haystack, needle)
            assert result == expected, \
                f"{fn.__name__}({haystack!r}, {needle!r}) = {result}, expected {expected}"


if __name__ == "__main__":
    test_all()
    print("All tests passed")
