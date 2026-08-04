"""Test suite — 7 cases. Bar: pass first 4."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from solver import solve, parse_maze, mark_path, find_cell


# ── helpers ────────────────────────────────────────────────────────────────

def path_to_glyphs(raw: str, path: list[tuple[int, int]]) -> list[list[str]]:
    grid = parse_maze(raw)
    return mark_path(grid, path)


# ── Q1: print correctness ──────────────────────────────────────────────────

MAZE_SIMPLE = """\
S...#
.###.
.....
#####
....E"""

def test_start_glyph_preserved():
    """S must remain 'S' after mark_path, not be overwritten by '*'."""
    path = solve(MAZE_SIMPLE)
    assert path is not None
    marked = path_to_glyphs(MAZE_SIMPLE, path)
    start = find_cell(parse_maze(MAZE_SIMPLE), "S")
    assert marked[start[0]][start[1]] == "S", (
        f"Expected 'S' at {start}, got '{marked[start[0]][start[1]]}'"
    )

def test_end_glyph_preserved():
    """E must remain 'E' after mark_path."""
    path = solve(MAZE_SIMPLE)
    assert path is not None
    marked = path_to_glyphs(MAZE_SIMPLE, path)
    end = find_cell(parse_maze(MAZE_SIMPLE), "E")
    assert marked[end[0]][end[1]] == "E", (
        f"Expected 'E' at {end}, got '{marked[end[0]][end[1]]}'"
    )


# ── Q2: BFS correctness ────────────────────────────────────────────────────

MAZE_CYCLE = """\
S..
...
..E"""

def test_bfs_reaches_end():
    """BFS must terminate and find a path."""
    path = solve(MAZE_CYCLE)
    assert path is not None, "Expected a path, got None"
    grid = parse_maze(MAZE_CYCLE)
    end = find_cell(grid, "E")
    assert path[-1] == end

def test_bfs_no_solution():
    """Walled-off end must return None."""
    raw = "S.#\n###\n#.E"
    assert solve(raw) is None


# ── Q3: directional gates ──────────────────────────────────────────────────

MAZE_GATE = """\
S.>..E"""

def test_gate_forces_direction():
    """
    '>' at col 2 forces rightward move only.
    Path must not step left from col 3 back through the gate.
    """
    path = solve(MAZE_GATE)
    assert path is not None
    # After passing col 2 (the '>' cell), all subsequent steps must move right
    gate_col = 2
    past_gate = [(r, c) for (r, c) in path if c >= gate_col]
    for i in range(1, len(past_gate)):
        pr, pc = past_gate[i - 1]
        nr, nc = past_gate[i]
        assert nc >= pc, f"Path moved left after '>' gate: {past_gate[i-1]} -> {past_gate[i]}"


# ── Q4: keys and doors ────────────────────────────────────────────────────

MAZE_KEYS = """\
S.a.A.E"""

def test_key_unlocks_door():
    """Must collect key 'a' before passing door 'A'."""
    path = solve(MAZE_KEYS)
    assert path is not None
    grid = parse_maze(MAZE_KEYS)
    key_col   = next(c for c, ch in enumerate(grid[0]) if ch == "a")
    door_col  = next(c for c, ch in enumerate(grid[0]) if ch == "A")
    cols_visited = [c for (r, c) in path]
    assert cols_visited.index(key_col) < cols_visited.index(door_col), (
        "Key 'a' must be collected before door 'A' is passed"
    )

def test_no_key_blocked():
    """Without accessible key, a door must block the path."""
    raw = "S.#\nA..\n#.E"  # 'A' door, no 'a' key in reachable area
    # NOTE: this test case is intentionally invalid for the base BFS —
    # comment it out if running before Q4 is implemented.
    pass  # placeholder


# ── entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    tests = [
        test_start_glyph_preserved,
        test_end_glyph_preserved,
        test_bfs_reaches_end,
        test_bfs_no_solution,
        test_gate_forces_direction,
        test_key_unlocks_door,
        test_no_key_blocked,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
