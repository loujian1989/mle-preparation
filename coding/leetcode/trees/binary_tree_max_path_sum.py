"""
Binary Tree Maximum Path Sum (LeetCode 124) — Hard
===================================================

Problem:
    A path in a binary tree is a sequence of nodes where each pair of adjacent
    nodes has an edge connecting them. A node can appear at most once.
    The path does NOT need to pass through the root.
    Find the maximum sum of any path.

Edge cases:
    - All negative values: answer is max single node (not 0)
    - Single node: answer is that node's value
    - Path must have at least one node

Key insight:
    At each node, the maximum path THROUGH the node = node.val + max(left_gain, 0) + max(right_gain, 0)
    But a path that CONTINUES upward can only use ONE child (left or right, not both).
    So: return node.val + max(left_gain, right_gain, 0) for the parent.

    Maintain a global max updated at each node.

Complexity:
    Time:  O(N) — each node visited once
    Space: O(H) — recursion stack (O(N) worst, O(log N) balanced)
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


def max_path_sum(root: Optional[TreeNode]) -> int:
    """Find the maximum path sum in the binary tree.

    Args:
        root: Root of the binary tree.

    Returns:
        Maximum path sum.

    Raises:
        ValueError: If root is None.

    Complexity:
        Time:  O(N)
        Space: O(H)
    """
    if root is None:
        raise ValueError("Tree must be non-empty")

    best = [root.val]  # use list for mutable closure

    def dfs(node: Optional[TreeNode]) -> int:
        """Return max gain from node going in ONE direction (for parent).

        Also updates best[] with the max path THROUGH node.

        Args:
            node: Current tree node.

        Returns:
            Max gain achievable going through this node (one direction only).
        """
        if node is None:
            return 0

        # Gain from children: 0 if negative (ignore negative branches)
        left_gain = max(dfs(node.left), 0)
        right_gain = max(dfs(node.right), 0)

        # Max path THROUGH this node (can use both branches)
        path_through = node.val + left_gain + right_gain
        best[0] = max(best[0], path_through)

        # Can only extend path in ONE direction upward
        return node.val + max(left_gain, right_gain)

    dfs(root)
    return best[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _build(vals: list) -> Optional[TreeNode]:
    if not vals:
        return None
    nodes = [TreeNode(v) if v is not None else None for v in vals]
    for i, n in enumerate(nodes):
        if n:
            li, ri = 2 * i + 1, 2 * i + 2
            n.left = nodes[li] if li < len(nodes) else None
            n.right = nodes[ri] if ri < len(nodes) else None
    return nodes[0]


def _test() -> None:
    # [1,2,3] → path 2→1→3 = 6
    assert max_path_sum(_build([1, 2, 3])) == 6

    # [-10,9,20,null,null,15,7] → path 15→20→7 = 42
    assert max_path_sum(_build([-10, 9, 20, None, None, 15, 7])) == 42

    # All negative → single max node
    assert max_path_sum(_build([-3, -2, -1])) == -1

    # Single node
    assert max_path_sum(TreeNode(5)) == 5

    print("  max_path_sum: all tests passed")


if __name__ == "__main__":
    _test()
