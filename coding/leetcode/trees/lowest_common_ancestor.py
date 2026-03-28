"""
Lowest Common Ancestor (LeetCode 236 + 235) — Medium
======================================================

Problem A (LC 236): LCA of Binary Tree (arbitrary tree, not BST)
Problem B (LC 235): LCA of Binary Search Tree (use BST property)

LCA(p, q) = the lowest (deepest) node that has both p and q as descendants
            (a node is a descendant of itself).

Edge cases:
    - p or q is the root → root is the LCA
    - p is an ancestor of q → p is the LCA
    - p and q in different subtrees → LCA is their split point

Approach A (Binary Tree):
    Post-order DFS. If current node equals p or q, return it.
    If both left and right subtrees return a non-None → current is LCA.
    If only one returns non-None → propagate that node up.

Approach B (BST):
    Use BST property: if both < root, go left; both > root, go right;
    otherwise root is LCA. Iterative (no recursion needed).

Complexity:
    Binary Tree: Time O(N), Space O(H) call stack
    BST:         Time O(H), Space O(1) iterative
"""

from typing import Optional


class TreeNode:
    """Binary tree node."""

    def __init__(
        self,
        val: int = 0,
        left: Optional["TreeNode"] = None,
        right: Optional["TreeNode"] = None,
    ) -> None:
        self.val = val
        self.left = left
        self.right = right


# ---------------------------------------------------------------------------
# Problem A: LCA in Binary Tree
# ---------------------------------------------------------------------------

def lowest_common_ancestor(
    root: Optional[TreeNode],
    p: TreeNode,
    q: TreeNode,
) -> Optional[TreeNode]:
    """Find LCA in a binary tree (not necessarily BST).

    Args:
        root: Root of the binary tree.
        p:    First target node.
        q:    Second target node.

    Returns:
        LCA node, or None if not found.

    Complexity:
        Time:  O(N)
        Space: O(H) where H = tree height (O(N) worst, O(log N) balanced)
    """
    if root is None or root is p or root is q:
        return root

    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)

    # Both found in different subtrees → current node is LCA
    if left and right:
        return root
    # Propagate whichever subtree found something
    return left or right


# ---------------------------------------------------------------------------
# Problem B: LCA in BST (iterative — O(H), O(1) space)
# ---------------------------------------------------------------------------

def lowest_common_ancestor_bst(
    root: Optional[TreeNode],
    p: TreeNode,
    q: TreeNode,
) -> Optional[TreeNode]:
    """Find LCA in a BST using BST property.

    Args:
        root: Root of the BST.
        p:    First target node.
        q:    Second target node.

    Returns:
        LCA node.

    Complexity:
        Time:  O(H)
        Space: O(1)
    """
    node = root
    while node:
        if p.val < node.val and q.val < node.val:
            node = node.left    # both in left subtree
        elif p.val > node.val and q.val > node.val:
            node = node.right   # both in right subtree
        else:
            return node         # split point = LCA
    return None


# ---------------------------------------------------------------------------
# Helper: build tree from level-order list
# ---------------------------------------------------------------------------

def _build_tree(vals: list) -> Optional[TreeNode]:
    """Build tree from level-order list (None = missing node)."""
    if not vals:
        return None
    nodes = [TreeNode(v) if v is not None else None for v in vals]
    for i, node in enumerate(nodes):
        if node is None:
            continue
        left_idx = 2 * i + 1
        right_idx = 2 * i + 2
        if left_idx < len(nodes):
            node.left = nodes[left_idx]
        if right_idx < len(nodes):
            node.right = nodes[right_idx]
    return nodes[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test_binary_tree() -> None:
    #        3
    #      /   \
    #     5     1
    #    / \   / \
    #   6   2 0   8
    #      / \
    #     7   4
    root = _build_tree([3, 5, 1, 6, 2, 0, 8, None, None, 7, 4])
    node5 = root.left          # type: ignore
    node1 = root.right         # type: ignore
    node6 = root.left.left     # type: ignore
    node4 = root.left.right.right  # type: ignore

    lca = lowest_common_ancestor(root, node5, node1)
    assert lca.val == 3        # type: ignore

    lca2 = lowest_common_ancestor(root, node5, node4)
    assert lca2.val == 5       # type: ignore

    lca3 = lowest_common_ancestor(root, node6, node4)
    assert lca3.val == 5       # type: ignore

    print("  LCA (binary tree): all tests passed")


def _test_bst() -> None:
    #        6
    #      /   \
    #     2     8
    #    / \   / \
    #   0   4 7   9
    #      / \
    #     3   5
    root = _build_tree([6, 2, 8, 0, 4, 7, 9, None, None, 3, 5])
    node2 = root.left          # type: ignore
    node8 = root.right         # type: ignore
    node4 = root.left.right    # type: ignore

    lca = lowest_common_ancestor_bst(root, node2, node8)
    assert lca.val == 6        # type: ignore

    lca2 = lowest_common_ancestor_bst(root, node2, node4)
    assert lca2.val == 2       # type: ignore

    print("  LCA (BST): all tests passed")


if __name__ == "__main__":
    _test_binary_tree()
    _test_bst()
