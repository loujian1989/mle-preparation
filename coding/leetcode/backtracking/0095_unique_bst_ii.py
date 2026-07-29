# 95. Unique Binary Search Trees II (Medium)
#
# Given n, return all structurally unique BSTs with nodes 1..n.
#
# Approaches:
#   Step 1: Recursion           O(n·Cₙ) T / O(n·Cₙ) S
#   Step 2: Recursion + memo    O(n²·Cₙ) T / O(n²·Cₙ) S  ← avoids recompute
#
# Cₙ = nth Catalan number = C(2n,n)/(n+1) ≈ 4ⁿ / (n^(3/2) · √π)
# C₁=1, C₂=2, C₃=5, C₄=14, C₅=42
#
# --------------------------------------------------------------------------
# Core insight
#
#   For any root value i in [start, end]:
#     Left subtree  uses values [start .. i-1]  (must all be < i for BST)
#     Right subtree uses values [i+1  .. end]   (must all be > i for BST)
#
#   Recursively generate all left and right subtree lists, then
#   cross-combine: for every (left, right) pair, create a new node(i).
#
#   generate(start, end) → list of all valid BST roots for that range.
#
# --------------------------------------------------------------------------
# Why return [None] (not []) when start > end
#
#   [None] represents "one way to build an empty subtree."
#
#   We need it because the combination loop is:
#     for left in lefts:
#         for right in rights:
#             root = TreeNode(i, left, right)
#
#   If either lefts or rights is [], the loop body NEVER executes and
#   we produce zero trees — wrong when the other side has valid trees.
#
#   Example: root=1, n=3
#     left  = generate(1, 0) → must be [None]  so right trees are used
#     right = generate(2, 3) → [Node(2,_,3), Node(3,2,_)]
#     With [None]: produces 2 trees.  With []: produces 0 trees — bug.
#
# --------------------------------------------------------------------------
# Common mistake: redundant `if l == r` branch
#
#   When l == r, the main loop runs once with i = l:
#     generate(l, l-1) → l > r → [None]
#     generate(l+1, l) → l > r → [None]
#     → TreeNode(l, None, None) appended ✓
#
#   The general case already handles the single-node leaf correctly.
#   Adding an explicit `if l == r: return [TreeNode(l)]` is dead code.
#   Same pattern: the `if not nums` guard in LC 162 was also dead code
#   under the given constraints.
#
# --------------------------------------------------------------------------
# Return value vs accumulator parameter — when to use each
#
#   Pattern A — Return (this problem):
#     def generate(l, r):
#         ...
#         for left in generate(l, i-1):    # result used inline
#             for right in generate(i+1, r):
#                 res.append(TreeNode(i, left, right))
#         return res
#
#   Pattern B — Accumulator (backtracking, e.g. LC 22):
#     def backtrack(path, res):            # res shared, mutated in place
#         if done: res.append(path[:]); return
#         for choice in choices:
#             path.append(choice)
#             backtrack(path, res)
#             path.pop()
#
#   Decisive question: do you need to USE the result of one recursive call
#   INSIDE another recursive call?
#
#     YES → return. LC 95 needs left_trees and right_trees as separate
#       lists to cross-combine. An accumulator dumps everything into one
#       pool — you lose the ability to pair left with right correctly.
#
#     NO, all branches just add to one pool → accumulator. LC 22 branches
#       don't combine with each other; every path independently appends
#       a completed string to res.
#
#   Summary table:
#     Pattern       Each call produces       Typical shape
#     ──────────    ─────────────────────    ───────────────────────────
#     Return        Its own independent list  D&C, cross-combine results
#     Accumulator   Writes to shared list     DFS / backtracking / traversal
#
# --------------------------------------------------------------------------
# Complexity:
#   Time:  O(n · Cₙ)   — Cₙ trees, each node takes O(n) work to build
#   Space: O(n · Cₙ)   — total nodes across all Cₙ trees, each with n nodes
#   Memo adds O(n²) unique subproblems on top (distinct (start, end) pairs)
# --------------------------------------------------------------------------


from __future__ import annotations
from typing import List, Optional
from functools import lru_cache


class TreeNode:
    def __init__(self, val: int = 0,
                 left: Optional[TreeNode] = None,
                 right: Optional[TreeNode] = None):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        """Level-order string for debugging."""
        if self is None:
            return "None"
        result, queue = [], [self]
        while queue:
            node = queue.pop(0)
            if node:
                result.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else:
                result.append("_")
        # strip trailing "_"
        while result and result[-1] == "_":
            result.pop()
        return "[" + ",".join(result) + "]"


# ---------------------------------------------------------------------------
# Step 1: Pure recursion
# ---------------------------------------------------------------------------

def generate_trees(n: int) -> List[Optional[TreeNode]]:
    """
    For each root i in [start, end]: cross-combine all left and right subtrees.
    Return [None] for empty range (one way to build an empty subtree).

    Trace: n=3
      generate(1,3):
        i=1: L=generate(1,0)=[None], R=generate(2,3)
               generate(2,3):
                 i=2: L=[None], R=generate(3,3)=[Node(3)]
                      → Node(2, None, Node(3))
                 i=3: L=generate(2,2)=[Node(2)], R=[None]
                      → Node(3, Node(2), None)
               R = [Node(2,None,Node(3)), Node(3,Node(2),None)]
             → Node(1,None,Node(2,None,Node(3))),
               Node(1,None,Node(3,Node(2),None))
        i=2: L=[Node(1)], R=[Node(3)]
             → Node(2,Node(1),Node(3))
        i=3: L=generate(1,2), R=[None]
               generate(1,2):
                 i=1: L=[None], R=[Node(2)] → Node(1,None,Node(2))
                 i=2: L=[Node(1)], R=[None] → Node(2,Node(1),None)
             → Node(3,Node(1,None,Node(2)),None),
               Node(3,Node(2,Node(1),None),None)
      Total: 5 trees ✓

    Complexity:
        Time:  O(n · Cₙ)
        Space: O(n · Cₙ)
    """
    def generate(start: int, end: int) -> List[Optional[TreeNode]]:
        if start > end:
            return [None]        # one way to build empty subtree
        trees = []
        for i in range(start, end + 1):
            lefts  = generate(start, i - 1)
            rights = generate(i + 1, end)
            for left in lefts:
                for right in rights:
                    trees.append(TreeNode(i, left, right))
        return trees

    return generate(1, n)


# ---------------------------------------------------------------------------
# Step 2: Recursion + memoization
# ---------------------------------------------------------------------------

def generate_trees_memo(n: int) -> List[Optional[TreeNode]]:
    """
    Same logic but cache generate(start, end) → list of trees.
    There are O(n²) unique (start, end) pairs, each computed once.

    Note on correctness: cached node objects ARE shared across calls with
    the same (start, end). This is safe here because we never mutate nodes
    after creation. If nodes could be mutated, deep-copying would be required.

    Complexity:
        Time:  O(n² · Cₙ)  — O(n²) subproblems × O(Cₙ) work per subproblem
        Space: O(n² · Cₙ)  — cached lists across all subproblems
    """
    @lru_cache(maxsize=None)
    def generate(start: int, end: int) -> tuple:
        if start > end:
            return (None,)       # tuple for hashability in lru_cache
        trees = []
        for i in range(start, end + 1):
            for left in generate(start, i - 1):
                for right in generate(i + 1, end):
                    trees.append(TreeNode(i, left, right))
        return tuple(trees)

    return list(generate(1, n))


# ---------------------------------------------------------------------------
# Helper: collect all node values via in-order traversal (for testing)
# ---------------------------------------------------------------------------

def inorder(root: Optional[TreeNode]) -> List[int]:
    if not root:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)


def is_valid_bst(root: Optional[TreeNode],
                 lo: int = float('-inf'),
                 hi: int = float('inf')) -> bool:
    if not root:
        return True
    if not (lo < root.val < hi):
        return False
    return (is_valid_bst(root.left, lo, root.val) and
            is_valid_bst(root.right, root.val, hi))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all() -> None:
    from math import comb
    # Catalan numbers: C(n) = C(2n,n)/(n+1)
    catalan = {0: 1, 1: 1, 2: 2, 3: 5, 4: 14, 5: 42}

    for n in range(1, 6):
        for fn in (generate_trees, generate_trees_memo):
            trees = fn(n)

            # Correct count (Catalan number)
            assert len(trees) == catalan[n], \
                f"{fn.__name__}({n}): got {len(trees)} trees, expected {catalan[n]}"

            # Every tree is a valid BST containing exactly 1..n
            for tree in trees:
                assert inorder(tree) == list(range(1, n + 1)), \
                    f"{fn.__name__}({n}): in-order not 1..n"
                assert is_valid_bst(tree), \
                    f"{fn.__name__}({n}): not a valid BST"

            # All trees are structurally distinct (different level-order reprs)
            reprs = [repr(t) for t in trees]
            assert len(reprs) == len(set(reprs)), \
                f"{fn.__name__}({n}): duplicate tree structures"

    # n=0 edge case: problem says n>=1, but generate should handle gracefully
    # generate(1, 0) returns [None] — used internally, not exposed at top level


if __name__ == "__main__":
    test_all()
    print("All tests passed")

    # Visualize n=3
    print("\nn=3 trees:")
    for t in generate_trees(3):
        print(" ", t)
