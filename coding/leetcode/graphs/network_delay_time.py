"""
Network Delay Time (LeetCode 743) — Medium
===========================================

Problem:
    You have n nodes (1 to n) and a list of directed weighted edges times[i] = [u, v, w]
    meaning it takes w time for signal to travel from u to v.
    Send a signal from node k. Return the time for ALL nodes to receive the signal.
    Return -1 if not all nodes reachable.

    Essentially: shortest path from k to all nodes; answer = max shortest path.

Approach:
    Dijkstra's algorithm from source k.
    Min-heap: (dist, node). Relax edges greedily.

Edge cases:
    - k not reachable to all nodes → -1
    - Negative weights: Dijkstra fails (use Bellman-Ford instead)
    - Self-loops: fine, will never be shortest path

Complexity:
    Time:  O((V + E) log V) with min-heap
    Space: O(V + E) for adjacency list + heap
"""

import heapq
from collections import defaultdict
from typing import List


def network_delay_time(times: List[List[int]], n: int, k: int) -> int:
    """Dijkstra's shortest path from k to all nodes.

    Args:
        times: Directed weighted edges [u, v, w].
        n:     Number of nodes (1-indexed, 1 to n).
        k:     Source node.

    Returns:
        Max time for all nodes to receive signal, or -1 if not all reachable.

    Complexity:
        Time:  O((V + E) log V)
        Space: O(V + E)
    """
    adj: dict = defaultdict(list)
    for u, v, w in times:
        adj[u].append((v, w))

    # dist[node] = shortest known distance from k
    INF = float("inf")
    dist = {i: INF for i in range(1, n + 1)}
    dist[k] = 0

    # Min-heap: (distance, node)
    heap = [(0, k)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue  # stale entry — skip
        for v, w in adj[u]:
            new_dist = d + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))

    max_dist = max(dist.values())
    return max_dist if max_dist < INF else -1


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    # Standard case
    assert network_delay_time([[2, 1, 1], [2, 3, 1], [3, 4, 1]], 4, 2) == 2

    # All nodes reachable, different path lengths
    assert network_delay_time([[1, 2, 1]], 2, 1) == 1

    # Node not reachable
    assert network_delay_time([[1, 2, 1]], 2, 2) == -1  # 2 can't reach 1

    # Single node
    assert network_delay_time([], 1, 1) == 0

    print("  network_delay_time: all tests passed")


if __name__ == "__main__":
    _test()
