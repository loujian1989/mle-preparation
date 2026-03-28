"""
Clone Graph (LeetCode 133) — Medium
=====================================

Problem:
    Given a reference to a node in a connected undirected graph,
    return a deep copy (clone) of the graph.
    Each node has a val (int) and a neighbors list (List[Node]).

Edge cases:
    - Node is None → return None
    - Single node, no neighbors (no self-loops in test inputs)
    - Cycle: node A ↔ node B — must avoid infinite recursion / re-cloning

Approach (BFS + visited map):
    - visited: original_node → cloned_node (prevents re-cloning and breaks cycles)
    - BFS: process original neighbors, create clones as needed, wire neighbors

Complexity:
    Time:  O(V + E) — each node and edge visited once
    Space: O(V) for visited map + BFS queue
"""

from collections import deque
from typing import Optional


class Node:
    """Graph node with value and adjacency list."""

    def __init__(self, val: int = 0, neighbors: Optional[list] = None) -> None:
        self.val = val
        self.neighbors: list = neighbors if neighbors is not None else []


def clone_graph(node: Optional[Node]) -> Optional[Node]:
    """Deep-copy a connected undirected graph.

    Args:
        node: Reference to any node in the graph, or None.

    Returns:
        Reference to the corresponding node in the cloned graph, or None.

    Complexity:
        Time:  O(V + E)
        Space: O(V)
    """
    if node is None:
        return None

    visited: dict = {}   # original node → clone

    # Clone the start node
    visited[node] = Node(node.val)
    queue: deque = deque([node])

    while queue:
        original = queue.popleft()
        for neighbor in original.neighbors:
            if neighbor not in visited:
                visited[neighbor] = Node(neighbor.val)
                queue.append(neighbor)
            # Wire the cloned neighbor to cloned original
            visited[original].neighbors.append(visited[neighbor])

    return visited[node]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _build_graph(adj: list) -> Optional[Node]:
    """Build graph from 1-indexed adjacency list."""
    if not adj:
        return None
    nodes = [Node(i + 1) for i in range(len(adj))]
    for i, neighbors in enumerate(adj):
        nodes[i].neighbors = [nodes[j - 1] for j in neighbors]
    return nodes[0]


def _graph_equal(a: Optional[Node], b: Optional[Node]) -> bool:
    """Check structural equality of two graphs via BFS."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False

    visited_a: set = set()
    queue_a: deque = deque([a])
    queue_b: deque = deque([b])

    while queue_a:
        na, nb = queue_a.popleft(), queue_b.popleft()
        if na.val != nb.val or len(na.neighbors) != len(nb.neighbors):
            return False
        if id(na) == id(nb):
            return False    # clone must be a distinct object
        if na.val in visited_a:
            continue
        visited_a.add(na.val)
        for n_a, n_b in zip(sorted(na.neighbors, key=lambda x: x.val),
                            sorted(nb.neighbors, key=lambda x: x.val)):
            queue_a.append(n_a)
            queue_b.append(n_b)
    return True


def _test() -> None:
    # Standard 4-node cycle
    original = _build_graph([[2, 4], [1, 3], [2, 4], [1, 3]])
    cloned = clone_graph(original)
    assert _graph_equal(original, cloned)
    assert original is not cloned  # different objects

    # Single node, no neighbors
    single = Node(1)
    cloned_single = clone_graph(single)
    assert cloned_single is not None
    assert cloned_single.val == 1
    assert cloned_single.neighbors == []
    assert cloned_single is not single

    # None input
    assert clone_graph(None) is None

    print("  clone_graph: all tests passed")


if __name__ == "__main__":
    _test()
