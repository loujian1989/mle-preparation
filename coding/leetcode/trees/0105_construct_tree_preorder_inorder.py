# 105. Construct Binary Tree from Preorder and Inorder Traversal (Medium)
#
# Given preorder and inorder traversal arrays, reconstruct the binary tree.
#
# Approaches:
#   Step 1: self.pre pointer (user's version)    O(n) T / O(n) S
#   Step 2: Explicit preorder bounds             O(n) T / O(n) S  ← cleaner
#
# --------------------------------------------------------------------------
# Core insight
#
#   Preorder: root → left subtree → right subtree
#   Inorder:  left subtree → root → right subtree
#
#   Key properties used together:
#     1. preorder[pre_l] is ALWAYS the root of the current subtree
#     2. In inorder, root splits the array:
#          [in_l .. idx-1] = left subtree nodes
#          [idx+1 .. in_r] = right subtree nodes
#     3. left_size = idx - in_l  (count of left subtree nodes)
#        This tells us exactly how many elements in preorder belong to left.
#
#   Preorder slices:
#     Left subtree:  preorder[pre_l+1 .. pre_l+left_size]
#     Right subtree: preorder[pre_l+left_size+1 .. pre_r]
#
# --------------------------------------------------------------------------
# Why self.pre works (user's approach)
#
#   Preorder visits nodes: root, then all of left subtree, then all of right.
#   Our recursion processes: root, then recurse left, then recurse right.
#   These two orderings are IDENTICAL — so preorder[self.pre] always gives
#   the root of whatever subtree we're currently building.
#
#   self.pre just increments by 1 for each node we consume as a root.
#   No need to compute preorder bounds explicitly.
#
# --------------------------------------------------------------------------
# Why explicit bounds are cleaner
#
#   self.pre is a stateful side-effect that's easy to break (e.g. if you
#   accidentally call helper twice, or reorder left/right calls).
#
#   Explicit preorder bounds make the preorder slice visible at each call:
#     build(pre_l, pre_r, in_l, in_r)
#   The mapping between preorder and inorder ranges is explicit — easier to
#   reason about, and the approach generalizes to LC 106 (postorder + inorder)
#   by simply using pre_r as the root instead of pre_l.
#
# --------------------------------------------------------------------------
# Complexity:
#   Time:  O(n)  — each node built once; O(1) dict lookup for inorder index
#   Space: O(n)  — dict + recursion stack O(h), O(n) worst case skewed tree
# --------------------------------------------------------------------------


from __future__ import annotations
from typing import List, Optional


class TreeNode:
    def __init__(self, val: int = 0,
                 left: Optional[TreeNode] = None,
                 right: Optional[TreeNode] = None):
        self.val = val
        self.left = left
        self.right = right


# ---------------------------------------------------------------------------
# Step 1: User's version — self.pre pointer
# ---------------------------------------------------------------------------

def build_tree_pre_pointer(preorder: List[int],
                           inorder: List[int]) -> Optional[TreeNode]:
    """
    Global self.pre pointer advances linearly through preorder.
    Works because preorder order matches our recursion order exactly.

    Trace: preorder=[3,9,20,15,7], inorder=[9,3,15,20,7]
      pre=0: val=3, idx=1 in inorder → left=[9], right=[15,20,7]
        pre=1: val=9, idx=0 → left=[], right=[]   → Node(9)
        pre=2: val=20, idx=3 → left=[15], right=[7]
          pre=3: val=15, idx=2 → Node(15)
          pre=4: val=7,  idx=4 → Node(7)
      Result: 3 → left:9, right:20(left:15, right:7) ✓

    Complexity:
        Time:  O(n)
        Space: O(n)
    """
    dic = {val: i for i, val in enumerate(inorder)}
    pre = [0]   # use list for nonlocal-style mutation

    def helper(l: int, r: int) -> Optional[TreeNode]:
        if l > r:
            return None
        val = preorder[pre[0]]
        idx = dic[val]
        pre[0] += 1
        root = TreeNode(val)
        root.left  = helper(l, idx - 1)
        root.right = helper(idx + 1, r)
        return root

    return helper(0, len(inorder) - 1)


# ---------------------------------------------------------------------------
# Step 2: Explicit preorder bounds — cleaner, generalizes to LC 106
# ---------------------------------------------------------------------------

def build_tree(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    """
    Pass both preorder and inorder window bounds explicitly.
    left_size bridges the two arrays: same nodes, different positions.

    Preorder bounds:
      Root:              preorder[pre_l]
      Left subtree:      preorder[pre_l+1 .. pre_l+left_size]
      Right subtree:     preorder[pre_l+left_size+1 .. pre_r]

    Inorder bounds:
      Left subtree:      inorder[in_l .. idx-1]
      Root:              inorder[idx]
      Right subtree:     inorder[idx+1 .. in_r]

    Trace: preorder=[3,9,20,15,7], inorder=[9,3,15,20,7]
      build(0,4, 0,4): root=3, idx=1, left_size=1
        build(1,1, 0,0): root=9, left_size=0 → Node(9)
        build(2,4, 2,4): root=20, idx=3, left_size=1
          build(3,3, 2,2): root=15 → Node(15)
          build(4,4, 4,4): root=7  → Node(7)
      Result: 3(9, 20(15,7)) ✓

    LC 106 adaptation (postorder + inorder):
      Root is postorder[post_r] (last element).
      Left:  postorder[post_l .. post_l+left_size-1]
      Right: postorder[post_l+left_size .. post_r-1]

    Complexity:
        Time:  O(n)
        Space: O(n)
    """
    dic = {val: i for i, val in enumerate(inorder)}

    def build(pre_l: int, pre_r: int,
              in_l:  int, in_r:  int) -> Optional[TreeNode]:
        if pre_l > pre_r:
            return None
        val       = preorder[pre_l]
        idx       = dic[val]
        left_size = idx - in_l             # nodes in left subtree
        root      = TreeNode(val)
        root.left  = build(pre_l + 1,              pre_l + left_size,
                           in_l,                   idx - 1)
        root.right = build(pre_l + left_size + 1,  pre_r,
                           idx + 1,                in_r)
        return root

    return build(0, len(preorder) - 1, 0, len(inorder) - 1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def inorder_vals(root: Optional[TreeNode]) -> List[int]:
    res, stack, curr = [], [], root
    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        res.append(curr.val)
        curr = curr.right
    return res


def preorder_vals(root: Optional[TreeNode]) -> List[int]:
    if not root:
        return []
    return [root.val] + preorder_vals(root.left) + preorder_vals(root.right)


def test_all() -> None:
    cases = [
        ([3, 9, 20, 15, 7],  [9, 3, 15, 20, 7]),
        ([1],                 [1]),
        ([1, 2],              [2, 1]),
        ([1, 2, 3],           [2, 1, 3]),
        ([1, 2, 4, 5, 3, 6], [4, 2, 5, 1, 3, 6]),
    ]
    for preorder, inorder in cases:
        for fn in (build_tree_pre_pointer, build_tree):
            tree = fn(preorder, inorder)
            assert preorder_vals(tree) == preorder, \
                f"{fn.__name__}: preorder mismatch"
            assert inorder_vals(tree) == inorder, \
                f"{fn.__name__}: inorder mismatch"


if __name__ == "__main__":
    test_all()
    print("All tests passed")
