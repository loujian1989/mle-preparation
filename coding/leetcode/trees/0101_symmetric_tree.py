# 101. Symmetric Tree (Easy)
#
# Check if a binary tree is a mirror of itself (symmetric around center).
#
# Approaches:
#   Step 1: Recursive — isMirror(root, root)        O(n) T / O(h) S
#   Step 2: Recursive — isMirror(left, right)        O(n) T / O(h) S  ← cleaner
#   Step 3: Iterative BFS with queue                 O(n) T / O(n) S  ← interview preferred
#
# --------------------------------------------------------------------------
# Core insight
#
#   A tree is symmetric if its left and right subtrees are MIRRORS of each other.
#   Two trees are mirrors when:
#     1. Both null                    → True
#     2. One null, one not            → False
#     3. Both non-null:               → roots equal AND
#                                        outer pairs mirror (r1.left ↔ r2.right)
#                                        inner pairs mirror (r1.right ↔ r2.left)
#
#        mirror check:
#              r1        r2
#             /  \      /  \
#           L1   R1   L2   R2
#
#        outer pair: L1 ↔ R2  (both outermost — must match)
#        inner pair: R1 ↔ L2  (both innermost — must match)
#
# --------------------------------------------------------------------------
# The isMirror(root, root) trick — why it works
#
#   User's version calls isMirror(root, root).
#   First call: r1=root, r2=root
#     - r1.val == r2.val: trivially True (same node)
#     - recurse: isMirror(root.left, root.right)   ← left vs right
#               isMirror(root.right, root.left)    ← right vs left (redundant but equal)
#
#   This works because isMirror(left, right) == isMirror(right, left) by symmetry.
#   Calling both is harmless but does double the work at the top level.
#
#   Cleaner: call isMirror(root.left, root.right) directly.
#   The only cost: need a guard for root=None at the entry point.
#
# --------------------------------------------------------------------------
# Iterative BFS — how it works
#
#   Process pairs of nodes that should be mirrors of each other.
#   Start with (root.left, root.right).
#   At each step, enqueue the OUTER pair and INNER pair of the current two nodes:
#     outer: (left.left,  right.right)
#     inner: (left.right, right.left)
#
#   If any pair mismatches → not symmetric.
#   If queue empties without mismatch → symmetric.
#
# --------------------------------------------------------------------------
# Complexity (all approaches):
#   Time:  O(n)  — every node visited once
#   Space: O(h) recursive (call stack), O(n) BFS (queue width at widest level)
# --------------------------------------------------------------------------


from __future__ import annotations
from collections import deque
from typing import Optional


class TreeNode:
    def __init__(self, val: int = 0,
                 left: Optional[TreeNode] = None,
                 right: Optional[TreeNode] = None):
        self.val = val
        self.left = left
        self.right = right


# ---------------------------------------------------------------------------
# Step 1: User's recursive — isMirror(root, root)
# ---------------------------------------------------------------------------

def is_symmetric_root_root(root: Optional[TreeNode]) -> bool:
    """
    Calls isMirror with root twice. Works because:
      - First comparison r1.val==r2.val is trivially True (same node)
      - Recurses into (left, right) and (right, left) — both equivalent

    Minor: isMirror(right, left) is the same check as isMirror(left, right),
    so the top level does redundant work. Correct but not the clearest intent.

    Complexity:
        Time:  O(n)
        Space: O(h)
    """
    def is_mirror(r1: Optional[TreeNode], r2: Optional[TreeNode]) -> bool:
        if not r1 and not r2:
            return True
        if not r1 or not r2:
            return False
        return (r1.val == r2.val
                and is_mirror(r1.left,  r2.right)   # outer pair
                and is_mirror(r1.right, r2.left))    # inner pair

    return is_mirror(root, root)


# ---------------------------------------------------------------------------
# Step 2: Cleaner recursive — isMirror(left, right) directly
# ---------------------------------------------------------------------------

def is_symmetric_recursive(root: Optional[TreeNode]) -> bool:
    """
    Call is_mirror with the two halves directly. Intent is explicit:
    "are the left and right subtrees mirrors of each other?"

    Complexity:
        Time:  O(n)
        Space: O(h)
    """
    def is_mirror(r1: Optional[TreeNode], r2: Optional[TreeNode]) -> bool:
        if not r1 and not r2:
            return True
        if not r1 or not r2:
            return False
        return (r1.val == r2.val
                and is_mirror(r1.left,  r2.right)
                and is_mirror(r1.right, r2.left))

    if not root:
        return True
    return is_mirror(root.left, root.right)


# ---------------------------------------------------------------------------
# Step 3: Iterative BFS — interview preferred
# ---------------------------------------------------------------------------

def is_symmetric_iterative(root: Optional[TreeNode]) -> bool:
    """
    Process mirror-pairs using a queue. Each dequeue gives two nodes
    that must be mirrors. Enqueue their outer and inner child pairs.

    Avoids recursion depth limit. Makes the pair-comparison explicit.

    Trace: root=[1,2,2,3,4,4,3]
         1
        / \\
       2   2
      / \\ / \\
     3  4 4  3
      queue starts: [(2,2)]
      pop (2,2): vals equal. enqueue (3,3) outer, (4,4) inner
      pop (3,3): vals equal. enqueue (None,None), (None,None)
      pop (4,4): vals equal. enqueue (None,None), (None,None)
      pop (None,None): both null → skip (continue)
      ... all None pairs skipped → return True ✓

    Trace: root=[1,2,2,None,3,None,3]
         1
        / \\
       2   2
        \\   \\
         3   3
      queue: [(2,2)]
      pop (2,2): vals equal. enqueue (None,None) outer, (3,3) inner
      pop (None,None): both null → skip
      pop (3,3): vals equal. enqueue (None,None), (None,None) → True ✓

    Trace: root=[1,2,2,None,3,3,None]  ← asymmetric
         1
        / \\
       2   2
        \\  /
         3 3
      queue: [(2,2)]
      pop (2,2): vals equal. enqueue (None,3) outer, (3,None) inner
      pop (None,3): one null → return False ✓

    Complexity:
        Time:  O(n)
        Space: O(n)  — queue holds up to n/2 pairs at the widest level
    """
    if not root:
        return True
    queue: deque = deque([(root.left, root.right)])
    while queue:
        left, right = queue.popleft()
        if not left and not right:
            continue
        if not left or not right:
            return False
        if left.val != right.val:
            return False
        queue.append((left.left,  right.right))   # outer pair
        queue.append((left.right, right.left))    # inner pair
    return True


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
        ([1, 2, 2, 3, 4, 4, 3],          True),
        ([1, 2, 2, None, 3, None, 3],     False),
        ([1],                              True),
        ([1, 2, 2],                        True),
        ([1, 2, 3],                        False),
        ([1, 2, 2, 2, None, 2],           False),
        ([1, 2, 2, None, 3, 3, None],     True),
    ]
    for values, expected in cases:
        for fn in (is_symmetric_root_root, is_symmetric_recursive,
                   is_symmetric_iterative):
            result = fn(build(values))
            assert result == expected, \
                f"{fn.__name__}({values}) = {result}, expected {expected}"


if __name__ == "__main__":
    test_all()
    print("All tests passed")
