"""
Serialize and Deserialize Binary Tree (LeetCode 297) — Hard
============================================================

Problem:
    Design a Codec class with:
    - serialize(root): convert tree to string
    - deserialize(data): reconstruct tree from string

    The format is flexible as long as round-trip works.

Edge cases:
    - Empty tree (None root) → serialize to ""
    - Single node
    - Full binary tree
    - Skewed tree (all left or all right)

Approach — Level-order BFS (most intuitive):
    Serialize: BFS, include "null" for missing children.
    Format: "1,2,3,null,null,4,5" (LeetCode tree format)
    Deserialize: BFS reconstruction with a queue of parent nodes.

Alternative: Pre-order DFS with sentinel
    Simpler code, same complexity.

Complexity:
    Time:  O(N) serialize, O(N) deserialize
    Space: O(N) for string + queue
"""

from collections import deque
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


SENTINEL = "null"
SEP = ","


class Codec:
    """BFS-based binary tree serializer/deserializer."""

    def serialize(self, root: Optional[TreeNode]) -> str:
        """Serialize tree to string.

        Args:
            root: Root of tree.

        Returns:
            Comma-separated level-order string.

        Complexity:
            Time:  O(N)
            Space: O(N)
        """
        if root is None:
            return ""

        result = []
        queue: deque = deque([root])

        while queue:
            node = queue.popleft()
            if node is None:
                result.append(SENTINEL)
            else:
                result.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)

        # Trim trailing nulls for compact representation
        while result and result[-1] == SENTINEL:
            result.pop()

        return SEP.join(result)

    def deserialize(self, data: str) -> Optional[TreeNode]:
        """Reconstruct tree from serialized string.

        Args:
            data: String produced by serialize().

        Returns:
            Root of reconstructed tree.

        Complexity:
            Time:  O(N)
            Space: O(N)
        """
        if not data:
            return None

        tokens = data.split(SEP)
        root = TreeNode(int(tokens[0]))
        queue: deque = deque([root])
        i = 1

        while queue and i < len(tokens):
            node = queue.popleft()

            # Left child
            if i < len(tokens):
                if tokens[i] != SENTINEL:
                    node.left = TreeNode(int(tokens[i]))
                    queue.append(node.left)
                i += 1

            # Right child
            if i < len(tokens):
                if tokens[i] != SENTINEL:
                    node.right = TreeNode(int(tokens[i]))
                    queue.append(node.right)
                i += 1

        return root


# ---------------------------------------------------------------------------
# DFS variant (pre-order with sentinel) — simpler code
# ---------------------------------------------------------------------------

class CodecDFS:
    """Pre-order DFS serializer/deserializer."""

    def serialize(self, root: Optional[TreeNode]) -> str:
        """Pre-order DFS serialization."""
        parts = []

        def dfs(node: Optional[TreeNode]) -> None:
            if node is None:
                parts.append(SENTINEL)
                return
            parts.append(str(node.val))
            dfs(node.left)
            dfs(node.right)

        dfs(root)
        return SEP.join(parts)

    def deserialize(self, data: str) -> Optional[TreeNode]:
        """Pre-order DFS deserialization."""
        tokens = iter(data.split(SEP))

        def dfs() -> Optional[TreeNode]:
            token = next(tokens)
            if token == SENTINEL:
                return None
            node = TreeNode(int(token))
            node.left = dfs()
            node.right = dfs()
            return node

        return dfs()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _trees_equal(a: Optional[TreeNode], b: Optional[TreeNode]) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return a.val == b.val and _trees_equal(a.left, b.left) and _trees_equal(a.right, b.right)


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
    for CodecClass in [Codec, CodecDFS]:
        codec = CodecClass()

        # Standard tree [1,2,3,null,null,4,5]
        root = _build([1, 2, 3, None, None, 4, 5])
        assert _trees_equal(codec.deserialize(codec.serialize(root)), root)

        # Empty tree
        assert codec.deserialize(codec.serialize(None)) is None

        # Single node
        single = TreeNode(42)
        result = codec.deserialize(codec.serialize(single))
        assert result is not None and result.val == 42

        # Left-skewed
        skewed = _build([1, 2, None, 3, None])
        assert _trees_equal(codec.deserialize(codec.serialize(skewed)), skewed)

    print("  serialize/deserialize binary tree: all tests passed")


if __name__ == "__main__":
    _test()
