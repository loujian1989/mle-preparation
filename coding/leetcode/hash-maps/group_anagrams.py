"""
Group Anagrams (LeetCode 49) — Medium
======================================

Problem:
    Given a list of strings, group the anagrams together.
    Return groups in any order.

Follow-up covered:
    B. Minimum Window Substring uses similar frequency-map technique
    C. Valid Anagram (LC 242) — simplified single-pair version

Edge cases:
    - Empty list → []
    - Single string → [[string]]
    - Strings of different lengths (can never be anagrams)
    - Empty string "" is its own anagram group

Approach A — Sorted key:
    Sort each string; anagrams share the same sorted form.
    Key: tuple(sorted(s)) → O(K log K) per string where K = len(s).

Approach B — Frequency count key:
    Count char frequencies; anagrams share the same frequency vector.
    Key: tuple of 26 counts → O(K) per string.
    Faster when strings are long with many characters.

Complexity:
    Approach A: Time O(N * K log K), Space O(N * K)
    Approach B: Time O(N * K),       Space O(N * K)
"""

from collections import defaultdict
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Approach A: Sort-based key
# ---------------------------------------------------------------------------

def group_anagrams_sorted(strs: List[str]) -> List[List[str]]:
    """Group anagrams using sorted string as key.

    Args:
        strs: List of strings.

    Returns:
        List of anagram groups (each group is a list of strings).

    Complexity:
        Time:  O(N * K log K) where N = len(strs), K = max string length
        Space: O(N * K)
    """
    groups: dict = defaultdict(list)
    for s in strs:
        key: Tuple = tuple(sorted(s))
        groups[key].append(s)
    return list(groups.values())


# ---------------------------------------------------------------------------
# Approach B: Frequency count key (faster for long strings)
# ---------------------------------------------------------------------------

def group_anagrams(strs: List[str]) -> List[List[str]]:
    """Group anagrams using character frequency as key.

    Args:
        strs: List of strings (assumed lowercase letters only).

    Returns:
        List of anagram groups.

    Complexity:
        Time:  O(N * K)
        Space: O(N * K)
    """
    groups: dict = defaultdict(list)
    for s in strs:
        count = [0] * 26
        for c in s:
            count[ord(c) - ord("a")] += 1
        groups[tuple(count)].append(s)
    return list(groups.values())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    # Standard case
    result = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
    # Convert to sets for order-independent comparison
    groups_as_sets = {frozenset(g) for g in result}
    assert {"eat", "tea", "ate"} in groups_as_sets
    assert {"tan", "nat"} in groups_as_sets
    assert {"bat"} in groups_as_sets

    # Edge cases
    assert group_anagrams([""]) == [[""]]
    assert group_anagrams(["a"]) == [["a"]]
    assert len(group_anagrams([])) == 0

    # Sorted approach
    result2 = group_anagrams_sorted(["eat", "tea", "tan", "ate"])
    groups2 = {frozenset(g) for g in result2}
    assert {"eat", "tea", "ate"} in groups2

    print("  group_anagrams: all tests passed")


if __name__ == "__main__":
    _test()
