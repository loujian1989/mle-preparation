# 76. Minimum Window Substring (Hard)
#
# Given strings s and t, return the minimum window substring of s that
# contains every character of t (including duplicates). Return "" if none.
#
# Approach: Sliding window (two pointers)   O(m+n) T / O(n) S
#
# --------------------------------------------------------------------------
# Core logic — the cnt trick
#
#   dic[c] = how many more of character c we still NEED in the window.
#   cnt    = total characters still needed (sum of positive dic values).
#
#   Expanding (j moves right):
#     If dic[s[j]] > 0 before decrement → we needed this char → cnt -= 1
#     Always decrement dic[s[j]] (even if we now "over-collect" it → goes negative)
#
#   Contracting (i moves right, when cnt == 0 = valid window):
#     Always increment dic[s[i]]
#     If dic[s[i]] > 0 after increment → we lost a needed char → cnt += 1
#
#   Why dic can go negative:
#     Negative means we have MORE of that char than needed — surplus.
#     Surplus chars don't affect cnt. Only when dic[c] crosses back above 0
#     (on either side) does cnt change.
#
#   Example: t="ABC", window has "AABC"
#     dic['A'] = -1 (needed 1, have 2 → surplus of 1)
#     dic['B'] = 0
#     dic['C'] = 0
#     cnt = 0 → valid window
#     When we shrink past one 'A': dic['A'] becomes 0 → still valid (cnt stays 0)
#     When we shrink past second 'A': dic['A'] becomes 1 → cnt becomes 1 → invalid
#
# --------------------------------------------------------------------------
# Optimized variant — filtered sliding window
#
#   When t has few unique characters but s is very long, most characters
#   in s are irrelevant. Pre-filter s to only positions where s[j] is in t.
#   Run the sliding window on this filtered list instead.
#
#   Benefit: inner loop iterations = O(filtered) instead of O(m).
#   Asymptotic complexity is still O(m+n), but practical speedup is large
#   when |charset(t)| << m.
#
#   Example: t="XY", s has 100,000 chars but only 50 are 'X' or 'Y'.
#     Standard: inner loop runs up to 100,000 times.
#     Filtered: inner loop runs at most 50 times.
#
# --------------------------------------------------------------------------
# No better asymptotic solution exists:
#   O(m+n) is optimal — you must read every character of both s and t
#   at least once just to know what's needed and what's available.
#
# --------------------------------------------------------------------------
# Complexity:
#   Time:  O(m+n)  — each char in s added and removed at most once
#   Space: O(n)    — dic stores at most |charset(t)| entries
# --------------------------------------------------------------------------


import sys
from collections import defaultdict
from typing import Optional


# ---------------------------------------------------------------------------
# Standard sliding window
# ---------------------------------------------------------------------------

def min_window(s: str, t: str) -> str:
    """
    Sliding window with cnt trick. Expand j until valid, then shrink i.

    Trace: s="ADOBECODEBANC", t="ABC"
      dic={'A':1,'B':1,'C':1}, cnt=3
      j expands: A(cnt=2), D, O, B(cnt=1), E, C(cnt=0) → window "ADOBEC"
      i shrinks: remove A → dic['A']=1, cnt=1 → invalid. min="ADOBEC"
      j expands: O, D, E, B(dic['B'] was 0 → surplus, cnt stays), A(cnt=0)
      window "DOBECODEBA" valid? No wait — cnt=0 again at 'A'
      i shrinks to find minimum...
      Final minimum: "BANC" ✓

    Complexity:
        Time:  O(m+n)
        Space: O(n)
    """
    m, n = len(s), len(t)
    if n == 0 or n > m:
        return ''

    dic: dict = defaultdict(int)
    for c in t:
        dic[c] += 1
    cnt = n

    min_len, min_index = sys.maxsize, 0
    i = 0
    for j, c in enumerate(s):
        if dic[c] > 0:          # char is needed → satisfies one requirement
            cnt -= 1
        dic[c] -= 1             # consume it (may go negative = surplus)
        while cnt == 0:         # valid window — try to shrink
            if j - i + 1 < min_len:
                min_len, min_index = j - i + 1, i
            dic[s[i]] += 1      # release leftmost char
            if dic[s[i]] > 0:   # char is needed again → window invalid
                cnt += 1
            i += 1

    return s[min_index: min_index + min_len] if min_len != sys.maxsize else ''


# ---------------------------------------------------------------------------
# Optimized: filtered sliding window
# ---------------------------------------------------------------------------

def min_window_filtered(s: str, t: str) -> str:
    """
    Pre-filter s to only positions where s[j] appears in t.
    Run sliding window on the filtered list — far fewer iterations when
    |charset(t)| << m.

    Complexity:
        Time:  O(m+n)  — filter pass O(m), window on filtered list O(filtered)
        Space: O(m+n)  — filtered list O(m) worst case + dic O(n)
    """
    m, n = len(s), len(t)
    if n == 0 or n > m:
        return ''

    t_chars = set(t)
    filtered = [(i, c) for i, c in enumerate(s) if c in t_chars]

    dic: dict = defaultdict(int)
    for c in t:
        dic[c] += 1
    cnt = n

    min_len, min_index = sys.maxsize, 0
    i = 0
    for j, (pos_j, c) in enumerate(filtered):
        if dic[c] > 0:
            cnt -= 1
        dic[c] -= 1
        while cnt == 0:
            pos_i = filtered[i][0]
            if pos_j - pos_i + 1 < min_len:
                min_len, min_index = pos_j - pos_i + 1, pos_i
            left_c = filtered[i][1]
            dic[left_c] += 1
            if dic[left_c] > 0:
                cnt += 1
            i += 1

    return s[min_index: min_index + min_len] if min_len != sys.maxsize else ''


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all() -> None:
    cases = [
        ("ADOBECODEBANC", "ABC",  "BANC"),
        ("a",             "a",    "a"),
        ("a",             "aa",   ""),
        ("aa",            "aa",   "aa"),
        ("ab",            "b",    "b"),
        ("bba",           "ab",   "ba"),
        ("cabwefgewcwaefgcf", "cae", "cwae"),
        ("A",             "B",    ""),
    ]
    for s, t, expected in cases:
        for fn in (min_window, min_window_filtered):
            result = fn(s, t)
            assert result == expected, \
                f"{fn.__name__}({s!r}, {t!r}) = {result!r}, expected {expected!r}"


if __name__ == "__main__":
    test_all()
    print("All tests passed")
