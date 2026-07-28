# 30. Substring with Concatenation of All Words (Hard)
#
# Given string s and array words (all same length n), return all starting
# indices of substrings that are a concatenation of all words (any order).
#
# Approaches:
#   Step 1: Brute force          O(S·m·n) T / O(m) S
#   Step 2: Sliding window       O(S·n)   T / O(m) S  ← O(m) faster
#
# S = len(s), m = len(words), n = len(words[0])
#
# --------------------------------------------------------------------------
# Why the sliding window is faster
#
#   Brute force tries every starting position i (O(S) of them), then
#   extracts m words (O(m) each, O(n) per extraction) → O(S·m·n) total.
#
#   Key insight: since all words have the same length n, valid windows are
#   always word-aligned. There are exactly n distinct alignments:
#     offset 0: positions 0, n, 2n, 3n, ...
#     offset 1: positions 1, 1+n, 1+2n, ...
#     ...
#     offset n-1: positions n-1, n-1+n, ...
#
#   Run ONE sliding window per offset. For each offset, the right pointer
#   advances through S/n word positions; each word is processed at most
#   twice (once added, once removed). Word extraction costs O(n).
#   Total per offset: O(S/n · n) = O(S).
#   Total for all n offsets: O(S·n).
#
#   Speedup over brute force: O(m) — critical when m is large (up to 5000).
#
# --------------------------------------------------------------------------
# Sliding window logic (per offset)
#
#   dic1 = target word counts (from words[])
#   dic2 = current window word counts
#   count = number of valid words in current window
#   left = left boundary of window (char index)
#
#   Right pointer j advances in steps of n:
#
#     word = s[j:j+n]
#
#     Case 1: word NOT in dic1
#       Useless word — reset window entirely. left = j+n.
#
#     Case 2: word in dic1, not over-quota
#       Add to window. count += 1.
#       If count == m: found a valid start at `left`. Shrink one word from left.
#
#     Case 3: word in dic1, over-quota (dic2[word] > dic1[word])
#       Shrink left until dic2[word] drops back to dic1[word].
#
#   Why shrink after finding a match (count==m):
#     We record `left` as a valid start, then shrink by one word to allow
#     the window to shift right and potentially find the next match.
#
# --------------------------------------------------------------------------
# Complexity:
#   Brute force:     Time O(S·m·n), Space O(m)
#   Sliding window:  Time O(S·n),   Space O(m)
# --------------------------------------------------------------------------


from collections import defaultdict
from typing import List


# ---------------------------------------------------------------------------
# Step 1: Brute force
# ---------------------------------------------------------------------------

def find_substring_brute(s: str, words: List[str]) -> List[int]:
    """
    For each starting position, extract m words and compare to dic1.

    Trace: s="barfoothefoobarman", words=["foo","bar"]
      i=0: "bar","foo" → matches → append 0
      i=9: "foo","bar" → matches → append 9

    Complexity:
        Time:  O(S·m·n)  — S starts × m words × O(n) per extraction
        Space: O(m)      — dic1 and dic2
    """
    if not words or not s:
        return []
    m, n = len(words), len(words[0])
    if len(s) < m * n:
        return []

    dic1: dict = defaultdict(int)
    for w in words:
        dic1[w] += 1

    res = []
    for i in range(len(s) - m * n + 1):
        dic2: dict = defaultdict(int)
        for j in range(m):
            cur = s[i + j * n: i + (j + 1) * n]
            dic2[cur] += 1
            if dic2[cur] > dic1[cur]:
                break
        else:
            res.append(i)
    return res


# ---------------------------------------------------------------------------
# Step 2: Sliding window — O(S·n)
# ---------------------------------------------------------------------------

def find_substring(s: str, words: List[str]) -> List[int]:
    """
    n offset passes, each a sliding window at the word level.

    Trace: s="barfoothefoobarman", words=["foo","bar"], n=3, m=2

    offset r=0:
      j=0:  word="bar" ∈ dic1, count=1
      j=3:  word="foo" ∈ dic1, count=2=m → append left=0
            shrink: remove "bar", count=1, left=3
      j=6:  word="the" ∉ dic1 → reset, left=9
      j=9:  word="foo" ∈ dic1, count=1
      j=12: word="bar" ∈ dic1, count=2=m → append left=9
            shrink: remove "foo", count=1, left=12
      j=15: word="man" ∉ dic1 → reset
    offset r=1,2: no matches (words misaligned)
    Result: [0, 9] ✓

    Complexity:
        Time:  O(S·n)  — n offsets, each O(S) amortized
        Space: O(m)    — dic1, dic2
    """
    if not s or not words:
        return []
    m, n = len(words), len(words[0])
    if len(s) < m * n:
        return []

    dic1: dict = defaultdict(int)
    for w in words:
        dic1[w] += 1

    res = []

    for r in range(n):                          # n alignment offsets
        dic2: dict = defaultdict(int)
        count = 0                               # valid words in window
        left = r

        for j in range(r, len(s) - n + 1, n):  # advance right pointer
            word = s[j:j + n]

            if word not in dic1:               # Case 1: useless word
                dic2.clear()
                count = 0
                left = j + n

            else:
                dic2[word] += 1
                count += 1

                while dic2[word] > dic1[word]: # Case 3: over-quota — shrink
                    left_word = s[left:left + n]
                    dic2[left_word] -= 1
                    count -= 1
                    left += n

                if count == m:                 # Case 2: valid window found
                    res.append(left)
                    left_word = s[left:left + n]  # shrink one to continue
                    dic2[left_word] -= 1
                    count -= 1
                    left += n

    return res


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all() -> None:
    cases = [
        ("barfoothefoobarman",        ["foo", "bar"],             [0, 9]),
        ("wordgoodgoodgoodbestword",   ["word", "good", "best", "word"], []),
        ("barfoofoobarthefoobarman",   ["bar", "foo", "the"],     [6, 9, 12]),
        ("a",                          ["a"],                      [0]),
        ("aa",                         ["aa"],                     [0]),
        ("ab",                         ["aa"],                     []),   # no match
        ("aaa",                        ["a", "a"],                 [0, 1]),
        ("aaaaaaaa",                   ["aa", "aa", "aa"],         [0, 1, 2]),
        ("",                           ["foo"],                    []),
    ]
    for s, words, expected in cases:
        for fn in (find_substring_brute, find_substring):
            result = sorted(fn(s, words))
            assert result == sorted(expected), \
                f"{fn.__name__}({s!r}, {words}) = {result}, expected {sorted(expected)}"


if __name__ == "__main__":
    test_all()
    print("All tests passed")
