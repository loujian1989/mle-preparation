# 98. Validate Binary Search Tree (Medium)
#
# Determine if a binary tree is a valid BST:
#   - Left subtree contains only nodes with keys STRICTLY LESS than node's key
#   - Right subtree contains only nodes with keys STRICTLY GREATER than node's key
#   - Both subtrees are also valid BSTs
#
# Approaches:
#   Step 1: Value-based bounds      O(n) T / O(h) S  ← cleanest
#   Step 2: Node-based bounds       O(n) T / O(h) S  ← user's version, more indirection
#   Step 3: Iterative in-order      O(n) T / O(h) S  ← different angle, no bounds
#
# h = tree height. O(log n) balanced, O(n) worst case (skewed).
#
# --------------------------------------------------------------------------
# Key insight: propagate RANGE constraints top-down
#
#   Each node must satisfy lo < node.val < hi.
#   Initial range: (-∞, +∞) — root has no constraint.
#
#   When going LEFT  from node with val v: hi becomes v  (all left nodes < v)
#   When going RIGHT from node with val v: lo becomes v  (all right nodes > v)
#
#   Common mistake: only checking parent-child relationship (node > left child,
#   node < right child). This fails for:
#
#       5
#      / \
#     1   6
#        / \
#       3   7
#
#   Node 3 < parent 6 ✓, but 3 < root 5 violates the BST property.
#   The constraint must propagate: 3 must be in (5, 6), which it isn't.
#
# --------------------------------------------------------------------------
# In-order traversal equivalence
#
#   BST property ↔ in-order traversal is strictly ascending.
#   This is an equivalence — checking sorted in-order = checking BST validity.
#
#   Why: in-order visits left → node → right.
#     For BST, left < node < right at every level.
#     Composing across all levels → in-order sequence is globally sorted.
#
#   Implementation: track the previously visited value; if current ≤ prev → invalid.
#
# --------------------------------------------------------------------------
# Node-based vs value-based bounds
#
#   Node-based (user's version):
#     helper(root, minNode, maxNode)
#     check: minNode and minNode.val >= root.val   ← None guard + .val access
#
#   Value-based (Step 1):
#     helper(root, lo, hi)
#     check: lo < root.val < hi                   ← direct comparison, no guard
#
#   In Python there's no integer overflow, so value-based with float('-inf')/float('inf')
#   as sentinels is simpler. Node-based offers no advantage here.
#
# --------------------------------------------------------------------------
# Complexity (all approaches):
#   Time:  O(n)  — visit every node once
#   Space: O(h)  — recursion stack / explicit stack depth
# --------------------------------------------------------------------------


from __future__ import annotations
import math
from typing import Optional


class TreeNode:
    def __init__(self, val: int = 0,
                 left: Optional[TreeNode] = None,
                 right: Optional[TreeNode] = None):
        self.val = val
        self.left = left
        self.right = right


# ---------------------------------------------------------------------------
# Step 1: Value-based bounds — cleanest form
# ---------------------------------------------------------------------------

def is_valid_bst_bounds(root: Optional[TreeNode]) -> bool:
    """
    Pass (lo, hi) as numeric bounds. Each node must satisfy lo < val < hi.
    No None-guard or .val indirection needed — sentinels handle the edges.

    Going left:  new hi = node.val  (left subtree < node)
    Going right: new lo = node.val  (right subtree > node)

    Trace: root=[5,1,6,null,null,3,7]
      is_valid(5, -inf, inf):
        is_valid(1, -inf, 5): -inf<1<5 ✓ → True
        is_valid(6, 5, inf):
          is_valid(3, 5, 6): 5<3<6? NO → False ✗
      → False ✓ (invalid BST)

    Complexity:
        Time:  O(n)
        Space: O(h)
    """
    def helper(node: Optional[TreeNode], lo: float, hi: float) -> bool:
        if not node:
            return True
        if not (lo < node.val < hi):
            return False
        return (helper(node.left,  lo,       node.val) and
                helper(node.right, node.val, hi))

    return helper(root, -math.inf, math.inf)


# ---------------------------------------------------------------------------
# Step 2: Node-based bounds — user's version
# ---------------------------------------------------------------------------

def is_valid_bst_nodes(root: Optional[TreeNode]) -> bool:
    """
    Pass min/max as TreeNode references instead of values.
    Requires None-guard before accessing .val.

    Functionally equivalent to Step 1, but more verbose:
      Value-based: lo < root.val < hi
      Node-based:  (minNode is None or minNode.val < root.val) and
                   (maxNode is None or root.val < maxNode.val)

    No advantage in Python (no integer overflow). Node-based is common
    in C++ where sentinel values could risk overflow with INT_MIN/INT_MAX.

    Complexity:
        Time:  O(n)
        Space: O(h)
    """
    def helper(node: Optional[TreeNode],
               min_node: Optional[TreeNode],
               max_node: Optional[TreeNode]) -> bool:
        if not node:
            return True
        if min_node and min_node.val >= node.val:
            return False
        if max_node and max_node.val <= node.val:
            return False
        return (helper(node.left,  min_node, node) and
                helper(node.right, node,     max_node))

    return helper(root, None, None)


# ---------------------------------------------------------------------------
# Step 3: Iterative in-order traversal — no bounds, different angle
# ---------------------------------------------------------------------------

def is_valid_bst_inorder(root: Optional[TreeNode]) -> bool:
    """
    In-order traversal (left → node → right) of a valid BST is strictly
    ascending. Track `prev`; if current node ≤ prev → invalid.

    Uses an explicit stack to simulate recursion (avoids Python recursion
    limit for large trees; also cleaner than recursive in-order with
    instance variable for prev).

    Iterative in-order pattern:
      curr starts at root. Push lefts until null, then pop and process,
      then move to right child.

    Trace: root=[5,1,6,null,null,3,7]
      Push 5 → push 1 → pop 1 (prev=-inf, 1>-inf ✓, prev=1)
      → right of 1 is null → pop 5 (1<5 ✓, prev=5)
      → right of 5 is 6 → push 6 → push 3
      → pop 3 (prev=5, 3≤5 → return False ✓)

    Why iterative beats recursive-with-instance-variable:
      Recursive approach needs `self.prev` or a mutable container to share
      state across calls — breaks the pure function model. Iterative keeps
      state local (stack + prev variable).

    Complexity:
        Time:  O(n)
        Space: O(h)  — stack holds at most h nodes (one path to leaf)
    """
    stack = []
    prev = -math.inf
    curr = root
    while curr or stack:
        while curr:                 # push all left nodes
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()          # process node
        if curr.val <= prev:        # strict: equal is also invalid for BST
            return False
        prev = curr.val
        curr = curr.right           # move to right subtree
    return True


# ---------------------------------------------------------------------------
# Comparison
#
#   Approach          Angle         State passed       Sentinel / guard
#   ───────────────   ───────────   ────────────────   ─────────────────────
#   Value bounds      top-down      (lo, hi) floats    -inf / +inf sentinels
#   Node bounds       top-down      (minNode, maxNode)  None guard + .val
#   Iterative inorder bottom-up     stack + prev        -inf initial prev
#
#   For interviews: Step 1 is the clearest to write and explain.
#   Step 3 is impressive as a follow-up ("can you do it without recursion?")
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def build(values: list) -> Optional[TreeNode]:
    """Level-order build from list. None = missing node."""
    if not values:
        return None
    root = TreeNode(values[0])
    queue = [root]
    i = 1
    while queue and i < len(values):
        node = queue.pop(0)
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    return root


def test_all() -> None:
    cases = [
        ([2, 1, 3],                       True),
        ([5, 1, 4, None, None, 3, 6],     False),  # 3 violates root=5 constraint
        ([5, 1, 6, None, None, 3, 7],     False),  # 3 < 5 but in right subtree
        ([2, 2, 2],                        False),  # equal → not strict
        ([1],                              True),
        ([1, None, 2],                     True),
        ([3, 1, 5, None, 2],              True),
        ([3, 1, 5, None, 4],              False),  # 4 > 3 but in left subtree
        ([-2147483648],                    True),   # INT_MIN edge case
        ([2147483647],                     True),   # INT_MAX edge case
    ]
    for values, expected in cases:
        tree = build(values)
        for fn in (is_valid_bst_bounds, is_valid_bst_nodes, is_valid_bst_inorder):
            result = fn(tree)
            assert result == expected, \
                f"{fn.__name__}({values}) = {result}, expected {expected}"


if __name__ == "__main__":
    test_all()
    print("All tests passed")
