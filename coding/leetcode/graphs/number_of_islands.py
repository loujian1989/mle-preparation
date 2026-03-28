"""
Number of Islands (LeetCode 200) — Medium
==========================================

Problem:
    Given an m×n grid of '1' (land) and '0' (water), count the number of islands.
    An island is surrounded by water and formed by connecting adjacent lands
    horizontally or vertically.

Edge cases:
    - Empty grid → 0
    - All water → 0
    - All land → 1 (one connected island)
    - Single cell grid

Approach:
    BFS / DFS flood-fill: when we find a '1', increment count and flood-fill
    the entire island to '0' (mark visited). Repeat until no '1' remains.

    BFS chosen here: avoids recursion stack overflow on large grids.

Complexity:
    Time:  O(M * N) — each cell visited at most once
    Space: O(min(M, N)) — BFS queue at most bounded by the shorter dimension
                         (worst case: O(M*N) for a diagonal snake island)
"""

from collections import deque
from typing import List


DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def num_islands(grid: List[List[str]]) -> int:
    """Count the number of islands using BFS flood-fill.

    Args:
        grid: m×n binary grid of '1' (land) and '0' (water).
              Modified in-place during traversal.

    Returns:
        Number of distinct islands.

    Raises:
        ValueError: If grid is empty.

    Complexity:
        Time:  O(M * N)
        Space: O(min(M, N))
    """
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid), len(grid[0])
    count = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1":
                count += 1
                _bfs_flood(grid, r, c, rows, cols)

    return count


def _bfs_flood(
    grid: List[List[str]],
    start_r: int,
    start_c: int,
    rows: int,
    cols: int,
) -> None:
    """BFS flood-fill: mark all connected '1's as visited ('0').

    Args:
        grid:    Grid modified in-place.
        start_r: Starting row.
        start_c: Starting column.
        rows:    Grid height.
        cols:    Grid width.
    """
    queue: deque = deque([(start_r, start_c)])
    grid[start_r][start_c] = "0"  # mark visited immediately on enqueue

    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == "1":
                grid[nr][nc] = "0"
                queue.append((nr, nc))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    assert num_islands([
        ["1", "1", "1", "1", "0"],
        ["1", "1", "0", "1", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
    ]) == 1

    assert num_islands([
        ["1", "1", "0", "0", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "1", "0", "0"],
        ["0", "0", "0", "1", "1"],
    ]) == 3

    assert num_islands([["0"]]) == 0          # all water
    assert num_islands([["1"]]) == 1          # single land cell
    assert num_islands([]) == 0               # empty grid

    print("  num_islands: all tests passed")


if __name__ == "__main__":
    _test()
