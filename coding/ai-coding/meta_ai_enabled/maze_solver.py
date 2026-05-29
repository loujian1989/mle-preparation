"""
Maze Solver — Meta AI-Enabled Round (Confirmed Pool #1)
=======================================================

Problem:
    Find the shortest path from 'S' to 'E' in a 2D grid maze.

    Cell types:
        '.' — open path
        '#' — wall (impassable)
        'S' — start position
        'E' — end position
        '>' — one-way door: traversable only while moving right  (dr=0, dc=+1)
        '<' — one-way door: traversable only while moving left   (dr=0, dc=-1)
        'v' — one-way door: traversable only while moving down   (dr=+1, dc=0)
        '^' — one-way door: traversable only while moving up     (dr=-1, dc=0)

Checkpoint structure (mirrors actual Meta AI-enabled interview):
    Checkpoint 1: BFS returning shortest path as List[(row, col)]
    Checkpoint 2: Visited set added — without it the loop is infinite
    Checkpoint 3: Directional doors — restrict entry direction per cell type

Key insight:
    Standard BFS is sufficient. For doors, add one guard before enqueueing:
    if the target cell is a door, verify the direction of travel matches.
    No change to BFS structure needed.

Complexity:
    Time:  O(M * N)  — each cell enqueued at most once
    Space: O(M * N)  — visited set + BFS queue
"""

from collections import deque
from typing import Dict, List, Optional, Tuple


DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# Maps door character -> (dr, dc) required to legally enter that cell
DOOR_ENTRY: Dict[str, Tuple[int, int]] = {
    ">": (0, 1),
    "<": (0, -1),
    "v": (1, 0),
    "^": (-1, 0),
}

PASSABLE = {".", "S", "E", ">", "<", "v", "^"}


def solve_maze(grid: List[List[str]]) -> Optional[List[Tuple[int, int]]]:
    """Find shortest path from 'S' to 'E' using BFS.

    Supports directional door cells that restrict the direction of entry.

    Args:
        grid: 2D grid of cell chars. Not modified.

    Returns:
        Ordered list of (row, col) from S to E inclusive, or None if unreachable.

    Raises:
        ValueError: If grid is empty or missing 'S' or 'E'.

    Complexity:
        Time:  O(M * N)
        Space: O(M * N)
    """
    if not grid or not grid[0]:
        raise ValueError("grid must be non-empty")

    rows, cols = len(grid), len(grid[0])
    start: Optional[Tuple[int, int]] = None
    end: Optional[Tuple[int, int]] = None

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "S":
                start = (r, c)
            elif grid[r][c] == "E":
                end = (r, c)

    if start is None:
        raise ValueError("Grid missing 'S'")
    if end is None:
        raise ValueError("Grid missing 'E'")

    # BFS — Checkpoint 2: visited set prevents infinite loops
    visited: set[Tuple[int, int]] = {start}
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    queue: deque = deque([start])

    while queue:
        r, c = queue.popleft()

        if (r, c) == end:
            return _reconstruct_path(parent, end)

        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if (nr, nc) in visited:
                continue

            cell = grid[nr][nc]
            if cell == "#":
                continue

            # Checkpoint 3: directional door guard
            if cell in DOOR_ENTRY and DOOR_ENTRY[cell] != (dr, dc):
                continue

            visited.add((nr, nc))
            parent[(nr, nc)] = (r, c)
            queue.append((nr, nc))

    return None  # no path exists


def _reconstruct_path(
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
    end: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """Trace parent pointers from end back to start and reverse.

    Args:
        parent: Maps each reached cell to the cell it was reached from.
        end:    End cell coordinates.

    Returns:
        Path as ordered list from start to end.
    """
    path: List[Tuple[int, int]] = []
    node: Optional[Tuple[int, int]] = end
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    # Checkpoint 1: basic shortest path
    grid1 = [
        list("S.#"),
        list("..."),
        list("#.E"),
    ]
    path1 = solve_maze(grid1)
    assert path1 is not None
    assert path1[0] == (0, 0)
    assert path1[-1] == (2, 2)
    assert len(path1) == 5, f"Expected 5 steps, got {len(path1)}: {path1}"

    # Checkpoint 2: no path (S and E cut off)
    grid2 = [
        list("S#E"),
        list("###"),
        list("..."),
    ]
    assert solve_maze(grid2) is None

    # Single cell path (S == E is contrived, use adjacent)
    grid3 = [list("SE")]
    path3 = solve_maze(grid3)
    assert path3 == [(0, 0), (0, 1)]

    # Checkpoint 3a: '>' door traversable moving right
    # S > E — moving right from (0,0) to (0,1) enters '>' correctly
    grid4 = [list("S>E")]
    path4 = solve_maze(grid4)
    assert path4 == [(0, 0), (0, 1), (0, 2)], f"Expected direct path, got {path4}"

    # Checkpoint 3b: '>' door blocks downward entry
    # Grid:  S .
    #        > E
    # Can S go down to '>' (moving down, dr=1)? '>' requires dc=1 — blocked.
    # So path must go S(0,0)->right(0,1)->down(1,1)=E
    grid5 = [
        list("S."),
        list(">E"),
    ]
    path5 = solve_maze(grid5)
    assert path5 is not None
    assert path5[-1] == (1, 1)
    assert (1, 0) not in path5, "Should not traverse '>' moving downward"

    # Checkpoint 3c: one-way door makes path impossible
    # S is blocked going right by '#' and the only other route requires entering '>'
    # from the wrong direction
    grid6 = [
        list("S#"),
        list(">E"),
    ]
    # Only route: down from (0,0) to (1,0)='>' — but '>' requires moving right
    assert solve_maze(grid6) is None

    # Checkpoint 3d: '^' door (upward only)
    # S . .
    # . ^ E
    # Can move: S->right->right, then check if we can enter '^' from below (moving up)
    # or go around it. Here '^' at (1,1); to enter it we'd move up (from row 2, but no row 2).
    # So the path must go around: (0,0)->(0,1)->(0,2) then down to E at (1,2)
    grid7 = [
        list("S.."),
        list(".^E"),
    ]
    path7 = solve_maze(grid7)
    assert path7 is not None
    assert path7[-1] == (1, 2)
    # Verify '^' at (1,1) was not traversed going right (dc=1, not allowed for '^')
    if (1, 1) in path7:
        idx = path7.index((1, 1))
        prev = path7[idx - 1]
        assert prev[0] - 1 == 1 and prev[1] == 1, "'^' entered from wrong direction"

    print("  solve_maze: all tests passed")


if __name__ == "__main__":
    _test()
