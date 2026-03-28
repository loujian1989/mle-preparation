"""
Course Schedule (LeetCode 207) — Medium
========================================

Problem:
    There are numCourses courses (0 to numCourses-1). Given prerequisites[i] = [a, b]
    meaning you must take course b before course a, determine if you can finish all courses.
    (Equivalently: detect if the directed graph has a cycle.)

Follow-up (LeetCode 210 — Course Schedule II):
    Return a valid ordering of courses. Return [] if impossible.

Edge cases:
    - No prerequisites → always possible; return topological order
    - Self-loop: course a requires course a → cycle
    - Disconnected graph: multiple connected components still need full traversal

Approach:
    Topological sort via Kahn's algorithm (BFS, in-degree based):
    1. Compute in-degree for each node
    2. Initialize queue with all nodes of in-degree 0
    3. Process queue: decrement in-degree of neighbors; enqueue if in-degree hits 0
    4. If processed count == numCourses: no cycle (return order); else cycle detected

    Alternative: DFS with 3-color marking (WHITE/GRAY/BLACK).
    Kahn's chosen here: iterative, produces topological order naturally.

Complexity:
    Time:  O(V + E) where V = numCourses, E = len(prerequisites)
    Space: O(V + E) for adjacency list + in-degree array
"""

from collections import defaultdict, deque
from typing import List


def can_finish(num_courses: int, prerequisites: List[List[int]]) -> bool:
    """Determine if all courses can be finished (no cycle in prereq graph).

    Args:
        num_courses:   Number of courses (0 to num_courses-1).
        prerequisites: List of [course, prereq] pairs.

    Returns:
        True if a valid ordering exists; False if there is a cycle.

    Complexity:
        Time:  O(V + E)
        Space: O(V + E)
    """
    in_degree = [0] * num_courses
    adj: dict = defaultdict(list)

    for course, prereq in prerequisites:
        adj[prereq].append(course)
        in_degree[course] += 1

    queue: deque = deque(c for c in range(num_courses) if in_degree[c] == 0)
    processed = 0

    while queue:
        node = queue.popleft()
        processed += 1
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return processed == num_courses


def find_order(num_courses: int, prerequisites: List[List[int]]) -> List[int]:
    """Return a valid course ordering, or [] if impossible (LeetCode 210).

    Args:
        num_courses:   Number of courses.
        prerequisites: List of [course, prereq] pairs.

    Returns:
        List of courses in valid completion order, or empty list if cycle detected.

    Complexity:
        Time:  O(V + E)
        Space: O(V + E)
    """
    in_degree = [0] * num_courses
    adj: dict = defaultdict(list)

    for course, prereq in prerequisites:
        adj[prereq].append(course)
        in_degree[course] += 1

    queue: deque = deque(c for c in range(num_courses) if in_degree[c] == 0)
    order: List[int] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return order if len(order) == num_courses else []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    # Basic solvable cases
    assert can_finish(2, [[1, 0]]) is True
    assert can_finish(2, [[1, 0], [0, 1]]) is False  # cycle: 0→1→0
    assert can_finish(3, [[1, 0], [2, 1]]) is True
    assert can_finish(1, []) is True

    # find_order
    order = find_order(4, [[1, 0], [2, 0], [3, 1], [3, 2]])
    assert len(order) == 4
    # Verify order is valid: every prereq must appear before its course
    pos = {c: i for i, c in enumerate(order)}
    assert pos[0] < pos[1] and pos[0] < pos[2]
    assert pos[1] < pos[3] and pos[2] < pos[3]

    assert find_order(2, [[1, 0], [0, 1]]) == []  # cycle → no valid order

    print("  course_schedule: all tests passed")


if __name__ == "__main__":
    _test()
