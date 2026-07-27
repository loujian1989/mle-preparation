# 208. Implement Trie (Prefix Tree) (Medium)
#
# Implement a Trie with insert, search, and startsWith operations.
#   insert(word)          — insert word into the trie
#   search(word)          — True if word was inserted (exact match)
#   startsWith(prefix)    — True if any inserted word starts with prefix
#
# --------------------------------------------------------------------------
# Trie structure recap
#
#   Each NODE represents one character. Each PATH from root to a marked
#   node spells out one complete string. All strings sharing a prefix
#   share the same path from the root.
#
#   Example after inserting "apple" and "app":
#
#       root
#        └─ a
#            └─ p
#                └─ p  (is_end=True  ← "app" ends here)
#                    └─ l
#                        └─ e  (is_end=True  ← "apple" ends here)
#
#   search("app")       → reach 'p' (depth 3), is_end=True  → True
#   search("apple")     → reach 'e' (depth 5), is_end=True  → True
#   search("ap")        → reach 'p' (depth 2), is_end=False → False
#   startsWith("app")   → reach 'p' (depth 3), exists       → True
#   startsWith("apz")   → 'z' not in children              → False
#
# --------------------------------------------------------------------------
# children: dict vs array[26]
#
#   dict[str, TrieNode]:
#     Space: O(actual children only) — sparse, good for large alphabets
#     Lookup: O(1) average (hash)
#     Use when: arbitrary characters, unicode, sparse data
#
#   array[26]:
#     Space: always 26 pointers per node — dense
#     Lookup: O(1) exact (index arithmetic: ord(c) - ord('a'))
#     Use when: lowercase English only, cache performance matters
#
#   Interview default: dict — simpler, handles any character.
#   Follow-up: "for lowercase English only I'd switch to array[26]."
#
# --------------------------------------------------------------------------
# search vs startsWith — one-line difference
#
#   Both walk the trie character by character, returning False immediately
#   if a character is missing. They diverge only at the return:
#     search:     return node.is_end   ← must be a complete word
#     startsWith: return True          ← reaching the end of prefix is enough
#
# --------------------------------------------------------------------------
# Common bug — wrong variable in startsWith
#
#   def startsWith(self, prefix):
#       for c in word:       # BUG: NameError — should be `prefix`, not `word`
#
#   Classic copy-paste from search(). Always check the loop variable name
#   when duplicating methods.
#
# --------------------------------------------------------------------------
# Complexity:
#   insert:     Time O(m), Space O(m)  — m = word length
#   search:     Time O(m), Space O(1)
#   startsWith: Time O(m), Space O(1)
#
#   Total space: O(N * m) where N = number of words, m = average length
#                (worst case — no shared prefixes)
# --------------------------------------------------------------------------


from typing import Optional


class TrieNode:
    def __init__(self) -> None:
        self.children: dict[str, "TrieNode"] = {}
        self.is_end: bool = False


class Trie:
    """
    Prefix tree supporting insert, exact search, and prefix search.

    All three operations share the same walk pattern:
      node = self.root
      for c in word/prefix:
          if c not in node.children: return False / create node
          node = node.children[c]
      # then diverge: is_end check, True, or insert marker
    """

    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """
        Walk the trie, creating missing nodes. Mark last node as end-of-word.

        Complexity: Time O(m), Space O(m)  m = len(word)
        """
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_end = True

    def search(self, word: str) -> bool:
        """
        Walk the trie. Return True only if the last node is marked is_end.
        A word that is a prefix of an inserted word returns False.

        Complexity: Time O(m), Space O(1)
        """
        node = self.root
        for c in word:
            if c not in node.children:
                return False
            node = node.children[c]
        return node.is_end      # exact match required

    def startsWith(self, prefix: str) -> bool:
        """
        Walk the trie. Return True if we can traverse all prefix characters.
        No is_end check — reaching the end of prefix is sufficient.

        Complexity: Time O(m), Space O(1)
        """
        node = self.root
        for c in prefix:        # NOTE: must be `prefix`, not `word`
            if c not in node.children:
                return False
            node = node.children[c]
        return True             # prefix exists — don't care about is_end


# ---------------------------------------------------------------------------
# Variant: array[26] children — for lowercase English, better cache perf
# ---------------------------------------------------------------------------

class TrieNodeArray:
    def __init__(self) -> None:
        self.children: list[Optional["TrieNodeArray"]] = [None] * 26
        self.is_end: bool = False


class TrieArray:
    """
    Same logic as Trie but uses array[26] instead of dict.
    Lookup: ord(c) - ord('a') gives index 0-25 directly.
    """

    def __init__(self) -> None:
        self.root = TrieNodeArray()

    def _idx(self, c: str) -> int:
        return ord(c) - ord('a')

    def insert(self, word: str) -> None:
        node = self.root
        for c in word:
            i = self._idx(c)
            if node.children[i] is None:
                node.children[i] = TrieNodeArray()
            node = node.children[i]
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self.root
        for c in word:
            i = self._idx(c)
            if node.children[i] is None:
                return False
            node = node.children[i]
        return node.is_end

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for c in prefix:
            i = self._idx(c)
            if node.children[i] is None:
                return False
            node = node.children[i]
        return True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _run_ops(trie_cls):
    """Run the LC 208 example sequence and additional edge cases."""
    t = trie_cls()
    t.insert("apple")
    assert t.search("apple")    is True,  "exact match"
    assert t.search("app")      is False, "prefix only, not inserted"
    assert t.startsWith("app")  is True,  "app is prefix of apple"
    t.insert("app")
    assert t.search("app")      is True,  "now inserted"

    # additional cases
    t2 = trie_cls()
    t2.insert("ab")
    t2.insert("abc")
    assert t2.search("a")       is False
    assert t2.search("ab")      is True
    assert t2.search("abc")     is True
    assert t2.search("abcd")    is False
    assert t2.startsWith("a")   is True
    assert t2.startsWith("ab")  is True
    assert t2.startsWith("b")   is False
    assert t2.startsWith("")    is True   # empty prefix always true


def test_all() -> None:
    for cls in (Trie, TrieArray):
        _run_ops(cls)


if __name__ == "__main__":
    test_all()
    print("All tests passed")
