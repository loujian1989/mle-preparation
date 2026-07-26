# 14. Longest Common Prefix (Easy)
#
# Find the longest common prefix string amongst an array of strings.
# Return "" if there is no common prefix.
#
# Approaches:
#   Step 1: Horizontal scan  — shrink prefix against each string  O(S) T / O(m) S
#   Step 2: Vertical scan    — check column by column             O(S) T / O(1) S
#   Step 3: Binary search    — binary search on prefix length     O(S log m) T / O(1) S
#   Step 4: Trie             — for multiple queries               O(S) build + O(m)/query
#   Step 5: Divide & conquer — parallelizable, distributeable     O(S) T / O(m log n) S
#
# S = total characters across all strings
# m = length of shortest string
# n = number of strings
#
# --------------------------------------------------------------------------
# Interview Extensions (Staff bar — these are the real questions)
#
#   Q: "What if there are N queries, not just one?"
#   A: Build a Trie once in O(S), answer each query in O(m). Amortized cost
#      matters; recomputing per query is wasteful.
#
#   Q: "Strings arrive as a stream — you don't have them all upfront."
#   A: Horizontal scan handles this online. Maintain current prefix, update
#      with each new string. O(1) extra space beyond the prefix itself.
#
#   Q: "Strings are distributed across 10 machines."
#   A: MapReduce: each machine computes its local LCP → coordinator reduces.
#      LCP is associative: lcp(A,B,C) = lcp(lcp(A,B), C).
#
#   Q: "The alphabet is Unicode (emoji, CJK), not ASCII."
#   A: Operate on list(s) not bytes. Raw byte operations break multi-byte
#      characters. Vertical scan is safer than byte-level slicing.
#
#   Q: "Generalize to Longest Common Substring (not prefix)."
#   A: Completely different problem. O(n²) DP or O(S) with suffix arrays
#      + RMQ. Mention you know the distinction.
#
# Complexity comparison:
#
#   Approach          Time          Space       Best for
#   ──────────────    ──────────    ─────────   ───────────────────────────
#   Horizontal scan   O(S)          O(m)        single query, stream
#   Vertical scan     O(S)          O(1)        single query, short strings
#   Binary search     O(S log m)    O(1)        when prefix check is cheap
#   Trie              O(S)+O(m)/q   O(S)        multiple queries
#   D&C               O(S)          O(m log n)  distributed / parallel
# --------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Step 1: Horizontal scan
# ---------------------------------------------------------------------------

def lcp_horizontal(strs: list[str]) -> str:
    """
    Start with strs[0] as the prefix. Shrink it until it is a prefix of
    every subsequent string.

    Works naturally as an online/streaming algorithm — process one string
    at a time, update prefix in place.

    Trace: ["flower","flow","flight"]
      prefix = "flower"
      vs "flow":   "flower" not prefix → shrink → "flowe" → ... → "flow"  ✓
      vs "flight": "flow" not prefix → shrink → "flo" → "fl"  ✓
      return "fl"

    Complexity:
        Time:  O(S)  — each character examined at most once across all strings
        Space: O(m)  — prefix string (m = shortest string length)
    """
    if not strs:
        return ""
    prefix = strs[0]
    for s in strs[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


# ---------------------------------------------------------------------------
# Step 2: Vertical scan
# ---------------------------------------------------------------------------

def lcp_vertical(strs: list[str]) -> str:
    """
    Check character by character across all strings in parallel.
    Stop as soon as any string ends or characters diverge.

    Better early-exit than horizontal when short strings appear early
    (avoids repeatedly shrinking a long prefix string).

    Trace: ["flower","flow","flight"]
      col 0: f,f,f → match
      col 1: l,l,l → match
      col 2: o,o,i → mismatch → return strs[0][:2] = "fl"

    Complexity:
        Time:  O(S)  — stop at first mismatch
        Space: O(1)  — no prefix string stored, just return a slice
    """
    if not strs:
        return ""
    for col, char in enumerate(strs[0]):
        for s in strs[1:]:
            if col >= len(s) or s[col] != char:
                return strs[0][:col]
    return strs[0]


# ---------------------------------------------------------------------------
# Step 3: Binary search on prefix length
# ---------------------------------------------------------------------------

def lcp_binary_search(strs: list[str]) -> str:
    """
    Binary search on the answer length L in [0, min_len].
    For each candidate L, check if all strings share the first L characters.

    Useful when the "does all strings share prefix of length L?" check is
    cheap (e.g., pre-hashed strings, rolling hash comparison).

    Complexity:
        Time:  O(S log m)  — O(log m) iterations, each O(S/m * m) = O(S)
        Space: O(1)
    """
    if not strs:
        return ""
    min_len = min(len(s) for s in strs)

    def all_share(length: int) -> bool:
        prefix = strs[0][:length]
        return all(s[:length] == prefix for s in strs[1:])

    lo, hi = 0, min_len
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if all_share(mid):
            lo = mid
        else:
            hi = mid - 1

    return strs[0][:lo]


# ---------------------------------------------------------------------------
# Step 4: Trie — for multiple queries
# ---------------------------------------------------------------------------

class TrieNode:
    def __init__(self):
        self.children: dict[str, "TrieNode"] = {}
        self.is_end: bool = False


class LCPTrie:
    """
    Build a Trie from all strings once. For any query string, walk the
    trie to find the longest prefix shared by at least one inserted string.

    For the LCP-of-all problem specifically: walk the trie from root until
    a node has >1 child or marks end_of_word — that's where the common
    prefix ends.

    Build: O(S)  — insert every character once
    Query: O(m)  — walk at most m characters for a query of length m
    Space: O(S)  — trie stores every character

    Use case: you have a fixed dictionary of strings and need to answer
    many LCP queries efficiently. Amortizes the build cost over queries.
    """

    def __init__(self, strs: list[str]):
        self.root = TrieNode()
        for s in strs:
            self._insert(s)

    def _insert(self, s: str) -> None:
        node = self.root
        for c in s:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_end = True

    def longest_common_prefix(self) -> str:
        """
        Walk from root while each node has exactly one child and is not
        an end-of-word marker (which would mean a string ended here,
        cutting off the common prefix).
        """
        node = self.root
        prefix = []
        while len(node.children) == 1 and not node.is_end:
            char = next(iter(node.children))
            prefix.append(char)
            node = node.children[char]
        return "".join(prefix)


def lcp_trie(strs: list[str]) -> str:
    """
    Build LCPTrie and extract the common prefix.

    Complexity:
        Time:  O(S) build + O(m) query
        Space: O(S)
    """
    if not strs:
        return ""
    trie = LCPTrie(strs)
    return trie.longest_common_prefix()


# ---------------------------------------------------------------------------
# Step 5: Divide & conquer — parallelizable / distributed
# ---------------------------------------------------------------------------

def lcp_divide_conquer(strs: list[str]) -> str:
    """
    Split the array in half recursively; LCP of full array = LCP of the
    two halves' results.

    LCP is associative: lcp(A,B,C) = lcp(lcp(A,B), C)
    This makes the operation naturally parallelizable: each machine handles
    a partition, coordinator reduces the partial results.

    Complexity:
        Time:  O(S)          — each char examined once across all levels
        Space: O(m log n)    — recursion stack depth log n, each frame O(m)
    """
    def common(a: str, b: str) -> str:
        i = 0
        while i < len(a) and i < len(b) and a[i] == b[i]:
            i += 1
        return a[:i]

    def solve(lo: int, hi: int) -> str:
        if lo == hi:
            return strs[lo]
        mid = (lo + hi) // 2
        left  = solve(lo, mid)
        right = solve(mid + 1, hi)
        return common(left, right)

    if not strs:
        return ""
    return solve(0, len(strs) - 1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all() -> None:
    cases = [
        (["flower", "flow", "flight"],  "fl"),
        (["dog", "racecar", "car"],      ""),
        (["a"],                          "a"),       # single string
        (["", "b"],                      ""),        # empty string in input
        (["abc", "abc", "abc"],          "abc"),     # all identical
        (["abc", "abcd"],                "abc"),     # one is prefix of other
        (["ab", "a"],                    "a"),
        (["interspecies", "interstellar", "interstate"], "inters"),
        (["throne", "throne"],           "throne"),
    ]
    fns = [lcp_horizontal, lcp_vertical, lcp_binary_search,
           lcp_trie, lcp_divide_conquer]
    for strs, expected in cases:
        for fn in fns:
            result = fn(strs)
            assert result == expected, \
                f"{fn.__name__}({strs}) = {result!r}, expected {expected!r}"


if __name__ == "__main__":
    test_all()
    print("All tests passed")
