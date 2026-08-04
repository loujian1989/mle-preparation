"""
Maze Solver — starter codebase.

Grid legend:
  S  start
  E  end
  #  wall
  .  open cell
"""

from collections import deque
from typing import Optional


DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def parse_maze(raw: str) -> list[list[str]]:
    """Convert a multi-line string into a 2-D character grid."""
    return [list(row) for row in raw.strip().split("\n")]


def find_cell(grid: list[list[str]], target: str) -> Optional[tuple[int, int]]:
    """Return (row, col) of the first occurrence of *target*, or None."""
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == target:
                return (r, c)
    return None


def get_neighbors(grid: list[list[str]], r: int, c: int) -> list[tuple[int, int]]:
    """Return valid (row, col) neighbors that are not walls."""
    rows, cols = len(grid), len(grid[0])
    result = []
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != "#":
            result.append((nr, nc))
    return result


def bfs(grid: list[list[str]], start: tuple[int, int], end: tuple[int, int]) -> Optional[list[tuple[int, int]]]:
    """
    BFS from start to end.

    Returns the path as a list of (row, col) tuples, or None if unreachable.
    """
    queue = deque()
    queue.append((start, [start]))
    # BUG Q2: visited set is missing — will loop forever on any cycle

    while queue:
        (r, c), path = queue.popleft()

        if (r, c) == end:
            return path

        for neighbor in get_neighbors(grid, r, c):
            queue.append((neighbor, path + [neighbor]))

    return None


def mark_path(grid: list[list[str]], path: list[tuple[int, int]]) -> list[list[str]]:
    """Return a copy of grid with the path marked as '*'."""
    import copy
    marked = copy.deepcopy(grid)
    for (r, c) in path:
        # BUG Q1: overwrites 'S' and 'E' with '*'
        marked[r][c] = "*"
    return marked


def solve(raw: str) -> Optional[list[tuple[int, int]]]:
    """Parse maze, run BFS, return path or None."""
    grid = parse_maze(raw)
    start = find_cell(grid, "S")
    end = find_cell(grid, "E")
    if start is None or end is None:
        raise ValueError("Maze must contain both 'S' and 'E'")
    return bfs(grid, start, end)
