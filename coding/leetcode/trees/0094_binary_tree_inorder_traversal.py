# 94. Binary Tree Inorder Traversal (Easy)
#
# Return inorder traversal (left → node → right) of a binary tree's values.
#
# Approaches:
#   Step 1: Recursive               O(n) T / O(h) S  — simplest
#   Step 2: Standard iterative      O(n) T / O(h) S  — interview default
#   Step 3: User's iterative        O(n) T / O(h) S  — works, has redundancies
#   Step 4: Morris traversal        O(n) T / O(1) S  — optimal space
#
# h = tree height. O(log n) balanced, O(n) worst case (skewed).
#
# --------------------------------------------------------------------------
# Why the standard iterative works — two mental models
#
#   curr  = "the node I'm about to descend into" (None if nothing to descend)
#   stack = "nodes passed through going left that still need visiting"
#
#   Inner while: exhausts the full left chain from curr, pushing every node.
#     After it exits: curr=None, stack holds the left-chain nodes.
#
#   Pop: top of stack is always the leftmost unvisited node — in inorder,
#     the next node to record.
#
#   curr = curr.right (no push): the right child is handled on the NEXT
#     outer iteration. If it's not None, inner while descends and pushes it
#     and its own left chain. If None, curr=None → outer condition checks
#     stack for remaining nodes.
#
#   Why `while curr or stack` (not just `while stack`):
#     After recording a node and setting curr=curr.right, the stack may be
#     empty but curr is not None (e.g. last-popped node had a right child).
#     Checking only `while stack` would exit early and miss that right subtree.
#
#   Trace: root=[2,1,3] → inorder [1,2,3]
#     curr=2, stack=[]
#     Outer 1: inner pushes 2,1 → stack=[2,1], curr=None
#       pop=1, res=[1], curr=1.right=None
#     Outer 2: curr=None, stack=[2]
#       inner skips; pop=2, res=[1,2], curr=2.right=3
#     Outer 3: curr=3, stack=[]
#       inner pushes 3 → stack=[3], curr=None
#       pop=3, res=[1,2,3], curr=3.right=None
#     curr=None, stack=[] → exit ✓
#
# --------------------------------------------------------------------------
# User's solution vs standard iterative — what's different
#
#   User's version:
#     stack.append(root)          # ← pre-push root
#     node = root
#     while stack:
#         while node and node.left:
#             node = node.left
#             stack.append(node)
#         cur = stack.pop()
#         res.append(cur.val)
#         if cur.right:
#             node = cur.right
#             stack.append(node)  # ← push right child explicitly
#
#   Standard version:
#     curr = root                 # ← no pre-push; curr IS the "to-visit" pointer
#     while curr or stack:
#         while curr:
#             stack.append(curr)
#             curr = curr.left
#         curr = stack.pop()
#         res.append(curr.val)
#         curr = curr.right       # ← just SET; next outer iteration handles push
#
#   Two redundancies in user's version:
#
#   1. stack.append(root) before the loop:
#      Standard doesn't pre-push. It just sets curr=root and the inner
#      while pushes root (and its left chain) on the first iteration.
#
#   2. stack.append(node) when assigning right child:
#      Standard sets curr=curr.right and lets the NEXT outer iteration's
#      inner while push it (and its left descendants).
#      User pushes it explicitly, then the inner while pushes its left
#      descendants on top. Both produce the same stack contents, but
#      standard does it uniformly in one place.
#
#   Both are correct. Standard is simpler because ALL pushes happen
#   inside the inner while — no extra push elsewhere.
#
#   Outer condition difference:
#     User:     while stack      — if curr is None but stack non-empty, works
#                                  because inner while does nothing (curr=node=None
#                                  after the last left-chain exhaustion)
#     Standard: while curr or stack — explicit: continue if pointer or stack has items
#
# --------------------------------------------------------------------------
# Morris traversal — how threading works
#
#   Goal: visit nodes in inorder without a stack.
#   Problem: after visiting a node, how do we return to its parent?
#   Solution: temporarily make the node the RIGHT child of its
#             inorder predecessor, creating a "thread" back to it.
#
#   Inorder predecessor of node X = rightmost node in X's LEFT subtree.
#   (The last node visited before X in inorder order.)
#
#   Algorithm (curr starts at root):
#     Case A: curr has no left child
#       → Visit curr (it has no predecessor to thread through)
#       → Move right: curr = curr.right
#
#     Case B: curr has a left child → find predecessor
#       If pred.right is None (not yet threaded):
#         → Thread: pred.right = curr  (save "return address")
#         → Descend left: curr = curr.left
#       If pred.right == curr (already threaded = we've returned):
#         → Unthread: pred.right = None  (restore tree)
#         → Visit curr
#         → Move right: curr = curr.right
#
#   Why the second visit to curr is the right time to record it:
#     First visit: threading and going left. curr not recorded yet.
#     Second visit: returned via thread after exhausting left subtree.
#                   All left descendants visited → now record curr. ✓
#
#   Example: tree [4,2,6,1,3,5,7]
#     curr=4: pred=3 (rightmost of left subtree 2→3). pred.right=None → thread 3→4, go left
#     curr=2: pred=1 (rightmost of left subtree 1).   pred.right=None → thread 1→2, go left
#     curr=1: no left child → visit 1, go right (thread back to 2)
#     curr=2: pred=1, pred.right==2 → unthread, visit 2, go right
#     curr=3: no left child → visit 3, go right (thread back to 4)
#     curr=4: pred=3, pred.right==4 → unthread, visit 4, go right
#     curr=6: pred=5. thread 5→6, go left
#     curr=5: no left → visit 5, go right (thread back to 6)
#     curr=6: pred=5, pred.right==6 → unthread, visit 6, go right
#     curr=7: no left → visit 7, done
#     Result: [1,2,3,4,5,6,7] ✓
#
# --------------------------------------------------------------------------
# Complexity comparison:
#   Approach          Time    Space   Notes
#   ─────────────     ──────  ──────  ──────────────────────────────────────
#   Recursive         O(n)    O(h)    Call stack is implicit
#   Standard iter.    O(n)    O(h)    Explicit stack; interview default
#   Morris            O(n)    O(1)    Modifies tree temporarily; restores it
# --------------------------------------------------------------------------


from __future__ import annotations
from typing import Optional, List


class TreeNode:
    def __init__(self, val: int = 0,
                 left: Optional[TreeNode] = None,
                 right: Optional[TreeNode] = None):
        self.val = val
        self.left = left
        self.right = right


# ---------------------------------------------------------------------------
# Step 1: Recursive — simplest baseline
# ---------------------------------------------------------------------------

def inorder_recursive(root: Optional[TreeNode]) -> List[int]:
    """
    Direct recursion. Simple but uses O(h) call stack space.

    Complexity:
        Time:  O(n)
        Space: O(h)  — call stack
    """
    res: List[int] = []

    def dfs(node: Optional[TreeNode]) -> None:
        if not node:
            return
        dfs(node.left)
        res.append(node.val)
        dfs(node.right)

    dfs(root)
    return res


# ---------------------------------------------------------------------------
# Step 2: Standard iterative — interview default
# ---------------------------------------------------------------------------

def inorder_iterative(root: Optional[TreeNode]) -> List[int]:
    """
    Explicit stack replaces the call stack. ALL pushes happen inside the
    inner while — no pre-push, no explicit right-child push.

    Pattern:
      Outer: while curr or stack  (curr=None but stack has items → still going)
      Inner: push curr and descend left until None
      Pop, record, move to right child (don't push — let next iteration handle it)

    Trace: root=[4,2,6,1,3,5,7]
      Descend left: push 4,2,1. Pop 1 → res=[1]. curr=None.
      Stack=[4,2]. Pop 2 → res=[1,2]. curr=3.
      Descend: push 3. Pop 3 → res=[1,2,3]. curr=None.
      Stack=[4]. Pop 4 → res=[1,2,3,4]. curr=6.
      Descend left: push 6,5. Pop 5 → res=[...,5]. curr=None.
      Stack=[6]. Pop 6 → res=[...,6]. curr=7.
      Push 7. Pop 7 → res=[1,2,3,4,5,6,7] ✓

    Complexity:
        Time:  O(n)
        Space: O(h)
    """
    res, stack = [], []
    curr = root
    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        res.append(curr.val)
        curr = curr.right    # don't push; next outer iteration handles it
    return res


# ---------------------------------------------------------------------------
# Step 3: Morris traversal — O(1) space
# ---------------------------------------------------------------------------

def inorder_morris(root: Optional[TreeNode]) -> List[int]:
    """
    Thread the tree: link each node's inorder predecessor back to it,
    allowing return to parent without a stack. Restore links after use.

    Two passes through each node:
      1st: thread predecessor → curr, go left (do NOT visit yet)
      2nd: arrived via thread → unthread, visit curr, go right

    Tree is temporarily modified but fully restored before return.

    Complexity:
        Time:  O(n)  — each node visited at most twice (find pred + visit)
        Space: O(1)  — no stack, no recursion
    """
    res = []
    curr = root
    while curr:
        if not curr.left:                       # Case A: no left subtree
            res.append(curr.val)
            curr = curr.right
        else:
            # Find inorder predecessor (rightmost of left subtree)
            pred = curr.left
            while pred.right and pred.right is not curr:
                pred = pred.right

            if pred.right is None:              # Case B1: not yet threaded
                pred.right = curr               # create thread
                curr = curr.left
            else:                               # Case B2: already threaded → returned
                pred.right = None               # remove thread (restore tree)
                res.append(curr.val)
                curr = curr.right
    return res


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def build(values: list) -> Optional[TreeNode]:
    """Level-order build. None = missing node."""
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
        ([1, None, 2, 3],               [1, 3, 2]),   # right-skewed with left child
        ([1, None, 2],                   [1, 2]),
        ([],                             []),
        ([1],                            [1]),
        ([4, 2, 6, 1, 3, 5, 7],         [1, 2, 3, 4, 5, 6, 7]),
        ([3, 1, None, None, 2],          [1, 2, 3]),
        ([5, 3, 7, 2, 4, 6, 8],         [2, 3, 4, 5, 6, 7, 8]),
    ]
    for values, expected in cases:
        tree = build(values)
        for fn in (inorder_recursive, inorder_iterative, inorder_morris):
            result = fn(tree)
            assert result == expected, \
                f"{fn.__name__}({values}) = {result}, expected {expected}"


if __name__ == "__main__":
    test_all()
    print("All tests passed")
