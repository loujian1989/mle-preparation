# 173. Binary Search Tree Iterator (Medium)
#
# Implement BSTIterator over the inorder traversal of a BST.
#   __init__: pointer starts before the smallest element
#   next():   move pointer right, return the value
#   hasNext(): True if more elements exist
#
# Approach: lazy iterative inorder (stack)   O(h) space, O(1) amortized per next()
#
# --------------------------------------------------------------------------
# Core insight — this IS the LC 94 iterative inorder, split across methods
#
#   LC 94 standard iterative:
#     curr = root
#     while curr or stack:
#         while curr:               # ← __init__ does this first time
#             stack.append(curr)
#             curr = curr.left
#         curr = stack.pop()        # ← next() does this
#         res.append(curr.val)
#         curr = curr.right         # ← next() does this too, then inner while
#
#   BSTIterator splits this into:
#     __init__  → first inner while (push left chain from root)
#     next()    → pop + push right child's left chain
#     hasNext() → outer condition check
#
#   The iterator "pauses" after each pop, resumes on the next() call.
#
# --------------------------------------------------------------------------
# Stack invariant: top is always the next node to return
#
#   The stack holds nodes that are "ready to visit" or "waiting for their
#   left subtrees to finish." After __init__ or each next():
#     - Stack top = smallest unvisited node
#     - Nodes below = ancestors still waiting
#
#   Example: BST [7,3,15,1,5,9,20]
#
#         7
#        / \
#       3   15
#      / \  / \
#     1   5 9  20
#
#   After __init__: push 7 → push 3 → push 1   stack = [7, 3, 1]
#   next() call 1:  pop 1 (val=1). 1.right=None → nothing pushed.
#                   stack = [7, 3]
#   next() call 2:  pop 3 (val=3). 3.right=5 → push 5 (no left chain).
#                   stack = [7, 5]
#   next() call 3:  pop 5 (val=5). 5.right=None → nothing pushed.
#                   stack = [7]
#   next() call 4:  pop 7 (val=7). 7.right=15 → push 15 → push 9.
#                   stack = [15, 9]
#   next() call 5:  pop 9 (val=9). 9.right=None → nothing.
#                   stack = [15]
#   next() call 6:  pop 15 (val=15). 15.right=20 → push 20.
#                   stack = [20]
#   next() call 7:  pop 20 (val=20). 20.right=None.
#                   stack = []   hasNext() → False ✓
#
# --------------------------------------------------------------------------
# Space complexity: O(h), not O(n)
#
#   Stack holds at most one path from root to a leaf — h nodes.
#   Lazy approach: only pushes nodes as needed, not all n upfront.
#   For a balanced BST: O(log n). For skewed: O(n).
#
#   Eager alternative (precompute full inorder list in __init__):
#     Space: O(n) always. next() is O(1). Simpler code, worse space.
#
# --------------------------------------------------------------------------
# Amortized O(1) per next()
#
#   Each node is pushed exactly once and popped exactly once across ALL
#   next() calls. n calls → O(n) total work → O(1) amortized per call.
#   (Same argument as a dynamic array's amortized O(1) append.)
#
# --------------------------------------------------------------------------
# User's version vs cleaned-up version
#
#   User's next():
#     if node.right:
#         node = node.right
#         self.stack.append(node)
#         while node.left:
#             node = node.left
#             self.stack.append(node)
#
#   Cleaned-up next():
#     curr = node.right       # None if no right child
#     while curr:             # handles None naturally — no if guard needed
#         self.stack.append(curr)
#         curr = curr.left
#
#   User's hasNext():
#     return True if self.stack else False   # redundant ternary
#
#   Cleaned-up hasNext():
#     return bool(self.stack)                # direct
#
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
# User's version — correct, minor style issues
# ---------------------------------------------------------------------------

class BSTIteratorUser:
    def __init__(self, root: Optional[TreeNode]) -> None:
        self.stack: list = []
        node = root
        while node:
            self.stack.append(node)
            node = node.left

    def next(self) -> int:
        node = self.stack.pop()
        val = node.val
        if node.right:                   # redundant guard — while handles None
            node = node.right
            self.stack.append(node)
            while node.left:
                node = node.left
                self.stack.append(node)
        return val

    def hasNext(self) -> bool:
        return True if self.stack else False   # redundant ternary


# ---------------------------------------------------------------------------
# Cleaned-up version
# ---------------------------------------------------------------------------

class BSTIterator:
    """
    Lazy iterative inorder traversal. Stack top is always the next value.

    __init__: push left chain from root   → O(h)
    next():   pop + push right's left chain → O(h) worst, O(1) amortized
    hasNext(): stack non-empty check        → O(1)

    Space: O(h) — at most one root-to-leaf path on the stack at any time.
    """

    def __init__(self, root: Optional[TreeNode]) -> None:
        self.stack: list = []
        self._push_left(root)

    def _push_left(self, node: Optional[TreeNode]) -> None:
        """Push node and its entire left chain onto the stack."""
        while node:
            self.stack.append(node)
            node = node.left

    def next(self) -> int:
        node = self.stack.pop()
        self._push_left(node.right)   # no if guard: _push_left handles None
        return node.val

    def hasNext(self) -> bool:
        return bool(self.stack)


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


def test_all() -> None:
    cases = [
        # (tree level-order, expected next() sequence)
        ([7, 3, 15, 1, 5, 9, 20], [1, 3, 5, 7, 9, 15, 20]),
        ([1],                       [1]),
        ([2, 1, 3],                 [1, 2, 3]),
        ([3, 1, None, None, 2],     [1, 2, 3]),
        ([5, 3, 7, 2, 4, 6, 8],    [2, 3, 4, 5, 6, 7, 8]),
    ]
    for values, expected in cases:
        for cls in (BSTIteratorUser, BSTIterator):
            tree = build(values)
            it = cls(tree)
            result = []
            while it.hasNext():
                result.append(it.next())
            assert result == expected, \
                f"{cls.__name__}({values}) = {result}, expected {expected}"


if __name__ == "__main__":
    test_all()
    print("All tests passed")
