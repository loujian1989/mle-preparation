# 99. Recover Binary Search Tree (Medium)
#
# Exactly two nodes of a BST were swapped by mistake.
# Recover the tree in-place (do not change structure, only fix values).
#
# Approaches:
#   Step 1: Recursive inorder    O(n) T / O(h) S  — easiest to write
#   Step 2: Iterative inorder    O(n) T / O(h) S  — no recursion limit risk
#   Step 3: Morris traversal     O(n) T / O(1) S  — optimal space
#
# --------------------------------------------------------------------------
# Core insight: find inversions in the inorder sequence
#
#   A valid BST's inorder traversal is strictly ascending.
#   Two swapped nodes create AT MOST TWO inversions (prev.val > curr.val).
#
#   Case 1 — non-adjacent swap (two inversions):
#     Original inorder: [1, 2, 3, 4, 5]  — nodes 2 and 4 swapped
#     Corrupted inorder: [1, 4, 3, 2, 5]
#                              ^     ^
#                         inv1: 4>3  inv2: 3>2 (second updates to curr=2)
#     first=4, second=2 → swap values ✓
#
#   Case 2 — adjacent swap (one inversion):
#     Original inorder: [1, 2, 3, 4, 5]  — nodes 2 and 3 swapped
#     Corrupted inorder: [1, 3, 2, 4, 5]
#                              ^
#                         inv1: 3>2 only
#     first=3, second=2 → swap values ✓
#
# --------------------------------------------------------------------------
# Why first = prev and second = curr (not the other way)
#
#   At an inversion, prev.val > curr.val:
#     prev is too LARGE  — should be smaller (it's first)
#     curr is too SMALL  — should be larger  (it's second)
#
#   First inversion:
#     first  = prev   (the large node that was swapped left/up)
#     second = curr   (candidate small node — might update later)
#
#   Second inversion (non-adjacent case only):
#     second = curr   (this is the actual small node, further right)
#     first stays — it was correctly identified at inversion 1
#
#   Why second always updates but first doesn't:
#     The large node appears at the FIRST inversion (prev = wrong large).
#     The small node might appear at the FIRST inversion too (adjacent case)
#     or at the SECOND inversion (non-adjacent). Always take the latest curr.
#
# --------------------------------------------------------------------------
# Two-inversion structure — why at most two
#
#   Swapping nodes X and Y in a sorted sequence of n elements:
#     - At X's new position: X > successor(X)  →  inversion 1
#     - At Y's new position: predecessor(Y) > Y → inversion 2
#     - If X and Y were adjacent, these overlap into one inversion.
#
# --------------------------------------------------------------------------
# Complexity (all approaches):
#   Time:  O(n)  — full inorder traversal
#   Space: O(h) recursive/iterative, O(1) Morris
# --------------------------------------------------------------------------


from __future__ import annotations
from typing import Optional


class TreeNode:
    def __init__(self, val: int = 0,
                 left: Optional[TreeNode] = None,
                 right: Optional[TreeNode] = None):
        self.val = val
        self.left = left
        self.right = right


# ---------------------------------------------------------------------------
# Step 1: Recursive inorder
# ---------------------------------------------------------------------------

def recover_tree_recursive(root: Optional[TreeNode]) -> None:
    """
    Inorder traversal with nonlocal state tracking three pointers:
      prev:   the previously visited node
      first:  the wrong-large node (first inversion's prev)
      second: the wrong-small node (latest inversion's curr)

    Trace: tree with swapped nodes producing inorder [1, 6, 3, 4, 5, 2, 7]
      prev=1,  curr=6: 1<6  → no inversion
      prev=6,  curr=3: 6>3  → inv1: first=6, second=3
      prev=3,  curr=4: 3<4  → no inversion
      prev=4,  curr=5: 4<5  → no inversion
      prev=5,  curr=2: 5>2  → inv2: second=2   (first stays 6)
      prev=2,  curr=7: 2<7  → no inversion
      Swap first.val(6) ↔ second.val(2) ✓

    Complexity:
        Time:  O(n)
        Space: O(h)  — call stack
    """
    first = second = prev = None

    def inorder(node: Optional[TreeNode]) -> None:
        nonlocal first, second, prev
        if not node:
            return
        inorder(node.left)
        if prev and prev.val > node.val:
            if not first:
                first = prev       # wrong-large node: first inversion's prev
            second = node          # wrong-small node: always update to latest curr
        prev = node
        inorder(node.right)

    inorder(root)
    first.val, second.val = second.val, first.val  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Step 2: Iterative inorder — same logic, explicit stack
# ---------------------------------------------------------------------------

def recover_tree_iterative(root: Optional[TreeNode]) -> None:
    """
    Standard iterative inorder (LC 94 pattern) with inversion tracking.
    Avoids Python recursion depth limit for very deep trees.

    Complexity:
        Time:  O(n)
        Space: O(h)  — explicit stack
    """
    first = second = prev = None
    stack = []
    curr = root

    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()

        if prev and prev.val > curr.val:
            if not first:
                first = prev
            second = curr

        prev = curr
        curr = curr.right

    first.val, second.val = second.val, first.val  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Step 3: Morris traversal — O(1) space
# ---------------------------------------------------------------------------

def recover_tree_morris(root: Optional[TreeNode]) -> None:
    """
    Morris inorder (LC 94 Step 3) with inversion tracking.
    Threads predecessors → no stack, no recursion.

    The "visit" point is identical to LC 94 Morris — only difference is
    we check for inversions at each visit rather than appending to a list.

    Complexity:
        Time:  O(n)  — each node visited at most twice
        Space: O(1)  — no stack
    """
    first = second = prev = None
    curr = root

    while curr:
        if not curr.left:
            # Case A: no left child → visit curr directly
            if prev and prev.val > curr.val:
                if not first:
                    first = prev
                second = curr
            prev = curr
            curr = curr.right
        else:
            # Find inorder predecessor
            pred = curr.left
            while pred.right and pred.right is not curr:
                pred = pred.right

            if pred.right is None:
                pred.right = curr        # thread: create return path
                curr = curr.left
            else:
                pred.right = None        # unthread: restore tree
                # Case B: visit curr (returned via thread)
                if prev and prev.val > curr.val:
                    if not first:
                        first = prev
                    second = curr
                prev = curr
                curr = curr.right

    first.val, second.val = second.val, first.val  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def build(values: list) -> Optional[TreeNode]:
    if not values or values[0] is None:
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


def inorder_values(root: Optional[TreeNode]) -> list:
    res, stack, curr = [], [], root
    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        res.append(curr.val)
        curr = curr.right
    return res


def test_all() -> None:
    import copy

    cases = [
        # (swapped BST level-order, expected inorder after recovery)
        ([1, 3, None, None, 2],      [1, 2, 3]),    # adjacent swap
        ([3, 1, 4, None, None, 2],   [1, 2, 3, 4]), # non-adjacent swap: 3↔2
        ([2, 3, 1],                  [1, 2, 3]),    # non-adjacent: inorder [3,2,1], swap 3↔1
    ]
    for values, expected in cases:
        for fn in (recover_tree_recursive, recover_tree_iterative,
                   recover_tree_morris):
            tree = build(values)
            fn(tree)
            result = inorder_values(tree)
            assert result == expected, \
                f"{fn.__name__}({values}) = {result}, expected {expected}"


if __name__ == "__main__":
    test_all()
    print("All tests passed")
