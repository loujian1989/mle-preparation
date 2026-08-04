"""
Maze Solver — Meta AI-Enabled Round (Full Q1–Q5 Ladder)
========================================================

Grid legend:
    '.'       open cell
    '#'       wall (impassable)
    'S'       start
    'E'       end
    '>' '<' 'v' '^'   directional gate — forces entry direction
    'a'–'z'   key (collected on entry)
    'A'–'Z'   door (requires matching lowercase key)
    'B'       bomb (one-shot; destroys walls within Chebyshev radius on trigger)

Checkpoint ladder (mirrors actual Meta interview):
    Q1  Bug fix — mark_path overwrites S/E glyphs  (AI usually disallowed)
    Q2  Bug fix — BFS loops forever without visited set
    Q3  Directional gates — one-way entry constraint
    Q4  Keys and doors — expand state to (r, c, key_bitmask)
    Q5a Bombs — further expand state to (r, c, keys, bombs_used_bitmask)
    Q5b Energy / weighted shortest path — Dijkstra instead of BFS

==========================================================================
AI-CODING SOP  (keep this in your head during the interview)
==========================================================================

PHASE 1 — Orient (2 min, no AI)
  1. Run the tests immediately to see which fail.
  2. Read every function signature. Map what each does in one sentence.
  3. Identify which checkpoint the failure belongs to.

PHASE 2 — Algorithm decision (speak aloud BEFORE touching the AI)
  Q1/Q2: trivial — state the fix verbally, then write it yourself.
  Q3:    "I need a gate guard in get_neighbors — check direction vs cell char."
  Q4:    "State must become (r, c, bitmask) — BFS over a larger state space."
  Q5a:   "Each bomb is an index; bombs_used is a second bitmask in state."
  Q5b:   "Dijkstra with (cost, r, c) heap — BFS is wrong for weighted cells."

PHASE 3 — Decompose into named functions BEFORE prompting AI
  Name each function and write its signature + docstring stub yourself.
  Only then hand the stub to Claude/GPT and say:
    "Fill the body of this function. Do not change the signature or docstring."

PHASE 4 — Read & explain every line the AI wrote
  The interviewer WILL ask "walk me through what this does."
  If you cannot explain a line, delete it and regenerate more narrowly.

PHASE 5 — Test incrementally
  After each function: run the suite. One red test at a time.

TRAP: AI models (even Opus 4.6) rarely produce the optimal Q4 solution
  unprompted. You must explicitly tell it: "state is (r, c, key_bitmask),
  visited is a set of those tuples, keys are picked up by OR-ing the bit."

==========================================================================
Complexity summary
==========================================================================
    Q1–Q3  BFS     Time O(M*N)           Space O(M*N)
    Q4     BFS     Time O(M*N*2^K)       Space O(M*N*2^K)   K = distinct keys
    Q5a    BFS     Time O(M*N*2^K*2^B)   Space O(M*N*2^K*2^B) B = bombs
    Q5b    Dijkstra Time O(M*N*log(M*N)) Space O(M*N)
"""

import heapq
from collections import deque
from typing import Dict, FrozenSet, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

DIRECTIONS: List[Tuple[int, int]] = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# Gate char -> required (dr, dc) to legally enter that cell
DOOR_ENTRY: Dict[str, Tuple[int, int]] = {
    ">": (0, 1),
    "<": (0, -1),
    "v": (1, 0),
    "^": (-1, 0),
}

PASSABLE: FrozenSet[str] = frozenset(". S E > < v ^ B".split())


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def parse_maze(raw: str) -> List[List[str]]:
    """Convert a newline-delimited string into a 2-D character grid.

    Args:
        raw: Multi-line maze string.

    Returns:
        2-D list of single characters.

    Complexity:
        Time:  O(M * N)
        Space: O(M * N)
    """
    return [list(row) for row in raw.strip().split("\n")]


def find_cells(grid: List[List[str]], targets: FrozenSet[str]) -> Dict[str, List[Tuple[int, int]]]:
    """Return a mapping of char -> [(row, col)] for every char in *targets*.

    Args:
        grid:    2-D character grid.
        targets: Set of characters to locate.

    Returns:
        Dict mapping each found character to its list of positions.

    Complexity:
        Time:  O(M * N)
        Space: O(M * N)
    """
    result: Dict[str, List[Tuple[int, int]]] = {}
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch in targets:
                result.setdefault(ch, []).append((r, c))
    return result


# ---------------------------------------------------------------------------
# Q1 fix — mark_path must not overwrite S / E
# ---------------------------------------------------------------------------

def mark_path(
    grid: List[List[str]],
    path: List[Tuple[int, int]],
    marker: str = "*",
) -> List[List[str]]:
    """Return a deep copy of grid with interior path cells marked.

    S and E glyphs are preserved (display priority over the path marker).

    Args:
        grid:   Original 2-D grid. Not modified.
        path:   Ordered list of (row, col) from S to E inclusive.
        marker: Character used to mark interior path cells. Default '*'.

    Returns:
        New 2-D grid with path marked.

    Complexity:
        Time:  O(M * N + P)   P = path length
        Space: O(M * N)
    """
    import copy
    marked = copy.deepcopy(grid)
    for r, c in path:
        # Q1 FIX: skip start/end so their glyphs are never overwritten
        if marked[r][c] not in ("S", "E"):
            marked[r][c] = marker
    return marked


def print_solution(grid: List[List[str]], path: List[Tuple[int, int]]) -> None:
    """Pretty-print the maze with the solution path highlighted.

    Args:
        grid: Original 2-D grid.
        path: Solution path from solve_maze / solve_maze_keys / etc.
    """
    marked = mark_path(grid, path)
    for row in marked:
        print("".join(row))


# ---------------------------------------------------------------------------
# Q2 + Q3 — BFS with visited set + directional gate guards
# ---------------------------------------------------------------------------

def solve_maze(grid: List[List[str]]) -> Optional[List[Tuple[int, int]]]:
    """Find shortest path from 'S' to 'E' using BFS.

    Handles:
      - Open cells, walls (Q2)
      - Directional gates >, <, v, ^ (Q3)

    Args:
        grid: 2-D grid of cell chars. Not modified.

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
    positions = find_cells(grid, frozenset("SE"))

    if "S" not in positions:
        raise ValueError("Grid missing 'S'")
    if "E" not in positions:
        raise ValueError("Grid missing 'E'")

    start = positions["S"][0]
    end   = positions["E"][0]

    # Q2 FIX: visited set prevents re-enqueuing, which would cause infinite loops
    visited: set = {start}
    parent: Dict = {start: None}
    queue: deque = deque([start])

    while queue:
        r, c = queue.popleft()

        if (r, c) == end:
            return _reconstruct(parent, end)

        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if (nr, nc) in visited:
                continue

            cell = grid[nr][nc]
            if cell == "#":
                continue

            # Q3: directional gate guard — entry direction must match gate
            if cell in DOOR_ENTRY and DOOR_ENTRY[cell] != (dr, dc):
                continue

            visited.add((nr, nc))
            parent[(nr, nc)] = (r, c)
            queue.append((nr, nc))

    return None


# ---------------------------------------------------------------------------
# Q4 — Keys and doors (bitmask state expansion)
# ---------------------------------------------------------------------------
#
# KEY INSIGHT for Q4:
#   Standard (r, c) visited set is wrong. With a new key you may legally
#   revisit cells that were previously blocked by a door.
#   Fix: state = (r, c, key_bitmask). Each unique (position, key-set)
#   combination is treated as a distinct node in BFS.
#
#   key_bitmask: bit i is set iff key chr(ord('a') + i) has been collected.
#   A door 'A' (index 0) is passable iff bit 0 is set in key_bitmask.
#
#   Max state space: M * N * 2^26, but in practice K << 26.

def solve_maze_keys(grid: List[List[str]]) -> Optional[List[Tuple[int, int]]]:
    """BFS over expanded state (row, col, key_bitmask) for key/door mazes.

    Lowercase letters a-z are keys (collected on entry).
    Uppercase letters A-Z are doors (require corresponding lowercase key).

    Args:
        grid: 2-D grid containing '.', '#', 'S', 'E', lowercase keys,
              uppercase doors, and optionally directional gates.

    Returns:
        Shortest path as list of (row, col), or None if unreachable.

    Raises:
        ValueError: If grid is empty or missing 'S' or 'E'.

    Complexity:
        Time:  O(M * N * 2^K)   K = number of distinct key types in grid
        Space: O(M * N * 2^K)
    """
    if not grid or not grid[0]:
        raise ValueError("grid must be non-empty")

    rows, cols = len(grid), len(grid[0])
    positions = find_cells(grid, frozenset("SE"))

    if "S" not in positions:
        raise ValueError("Grid missing 'S'")
    if "E" not in positions:
        raise ValueError("Grid missing 'E'")

    start = positions["S"][0]
    end   = positions["E"][0]

    # Precompute which uppercase cells are doors.
    # All uppercase alpha chars are doors EXCEPT reserved glyphs (S, E)
    # and directional gate chars (none of which are alpha-uppercase anyway).
    # A door with no matching key in the grid is permanently impassable.
    RESERVED = frozenset({"S", "E"}) | frozenset(DOOR_ENTRY)
    valid_doors: set = {
        ch for row in grid for ch in row
        if ch.isalpha() and ch.isupper() and ch not in RESERVED
    }

    # Initial state: at start, no keys collected
    init_state = (start[0], start[1], 0)
    visited: set = {init_state}
    # parent maps state -> (parent_state, (row, col)) for path reconstruction
    parent: Dict = {init_state: None}
    queue: deque = deque([init_state])

    while queue:
        r, c, keys = queue.popleft()

        if (r, c) == end:
            return _reconstruct_keys(parent, (r, c, keys))

        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            cell = grid[nr][nc]
            if cell == "#":
                continue

            # Directional gate guard (gates still apply in Q4)
            if cell in DOOR_ENTRY and DOOR_ENTRY[cell] != (dr, dc):
                continue

            # Door guard: only letters with a matching key in the grid are doors
            if cell in valid_doors:
                key_index = ord(cell.lower()) - ord("a")
                if not (keys >> key_index & 1):
                    continue  # blocked — don't have the key yet

            # Pick up key if this cell has one
            new_keys = keys
            if cell.islower():
                key_index = ord(cell) - ord("a")
                new_keys = keys | (1 << key_index)

            new_state = (nr, nc, new_keys)
            if new_state in visited:
                continue

            visited.add(new_state)
            parent[new_state] = (r, c, keys)
            queue.append(new_state)

    return None


def _reconstruct_keys(
    parent: Dict,
    end_state: Tuple,
) -> List[Tuple[int, int]]:
    """Reconstruct (row, col) path from key-state parent map.

    Args:
        parent:    Maps (r, c, keys) -> parent (r, c, keys) or None.
        end_state: Terminal state at goal.

    Returns:
        Ordered list of (row, col) from start to end.
    """
    path: List[Tuple[int, int]] = []
    state: Optional[Tuple] = end_state
    while state is not None:
        r, c, _ = state
        path.append((r, c))
        state = parent[state]
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Q5a — Bombs (blast radius, one-shot, bitmask state)
# ---------------------------------------------------------------------------
#
# KEY INSIGHT for Q5a:
#   A bomb 'B' at index i is a new dimension in the state bitmask.
#   Triggering bomb i destroys walls in a Chebyshev-radius-2 area.
#   State: (r, c, keys_bitmask, bombs_used_bitmask).
#   The destroyed walls set is DERIVED from bombs_used — not stored separately.
#   This keeps state compact and hashable.
#
#   Clarification to ask the interviewer:
#     "Does the bomb trigger on entry, or does the player choose to detonate?"
#   Default here: triggered on entry (one-shot, persistent wall destruction).

BOMB_RADIUS = 2  # Chebyshev distance


def get_affected_area(
    r: int,
    c: int,
    rows: int,
    cols: int,
    radius: int = BOMB_RADIUS,
) -> List[Tuple[int, int]]:
    """Return all cells within Chebyshev *radius* of (r, c), clamped to grid.

    Chebyshev distance: max(|dr|, |dc|) <= radius  (square blast zone).

    Args:
        r, c:   Bomb center.
        rows, cols: Grid dimensions.
        radius: Blast radius (inclusive). Default 2.

    Returns:
        List of (row, col) cells affected, including the bomb cell itself.

    Complexity:
        Time:  O(radius^2)
        Space: O(radius^2)
    """
    affected = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                affected.append((nr, nc))
    return affected


def _compute_destroyed_walls(
    grid: List[List[str]],
    bomb_positions: List[Tuple[int, int]],
    bombs_used: int,
) -> FrozenSet[Tuple[int, int]]:
    """Derive the set of destroyed wall cells from the bombs_used bitmask.

    Args:
        grid:           Original grid (unmodified).
        bomb_positions: List of (r, c) for each bomb in index order.
        bombs_used:     Bitmask of which bombs have been triggered.

    Returns:
        Frozenset of wall (r, c) cells that have been destroyed.

    Complexity:
        Time:  O(B * radius^2)   B = number of triggered bombs
        Space: O(B * radius^2)
    """
    rows, cols = len(grid), len(grid[0])
    destroyed: set = set()
    for i, (br, bc) in enumerate(bomb_positions):
        if bombs_used >> i & 1:
            for cell in get_affected_area(br, bc, rows, cols):
                if grid[cell[0]][cell[1]] == "#":
                    destroyed.add(cell)
    return frozenset(destroyed)


def solve_maze_bombs(grid: List[List[str]]) -> Optional[List[Tuple[int, int]]]:
    """BFS over (r, c, key_bitmask, bombs_used_bitmask) state space.

    Bombs ('B') trigger on entry and permanently destroy walls in a
    Chebyshev-radius-2 blast zone. Each bomb is one-shot.

    Args:
        grid: 2-D grid with '.', '#', 'S', 'E', lowercase keys,
              uppercase doors, directional gates, and 'B' bomb cells.

    Returns:
        Shortest path as list of (row, col), or None if unreachable.

    Raises:
        ValueError: If grid is empty or missing 'S' or 'E'.

    Complexity:
        Time:  O(M * N * 2^K * 2^B)
        Space: O(M * N * 2^K * 2^B)
    """
    if not grid or not grid[0]:
        raise ValueError("grid must be non-empty")

    rows, cols = len(grid), len(grid[0])
    positions = find_cells(grid, frozenset("SEB"))

    if "S" not in positions:
        raise ValueError("Grid missing 'S'")
    if "E" not in positions:
        raise ValueError("Grid missing 'E'")

    start = positions["S"][0]
    end   = positions["E"][0]
    # Enumerate bomb positions so each gets a fixed bitmask index
    bomb_positions: List[Tuple[int, int]] = positions.get("B", [])
    bomb_index: Dict[Tuple[int, int], int] = {pos: i for i, pos in enumerate(bomb_positions)}

    # All uppercase alpha chars (excluding S, E, B, gates) are doors.
    RESERVED = frozenset({"S", "E", "B"}) | frozenset(DOOR_ENTRY)
    valid_doors: set = {
        ch for row in grid for ch in row
        if ch.isalpha() and ch.isupper() and ch not in RESERVED
    }

    init_state = (start[0], start[1], 0, 0)  # (r, c, keys, bombs_used)
    visited: set = {init_state}
    parent: Dict = {init_state: None}
    queue: deque = deque([init_state])

    while queue:
        r, c, keys, bombs_used = queue.popleft()

        if (r, c) == end:
            return _reconstruct_bombs(parent, (r, c, keys, bombs_used))

        destroyed = _compute_destroyed_walls(grid, bomb_positions, bombs_used)

        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            cell = grid[nr][nc]

            # Wall passable only if destroyed by a prior bomb
            if cell == "#":
                if (nr, nc) not in destroyed:
                    continue
                effective_cell = "."
            else:
                effective_cell = cell

            if effective_cell in DOOR_ENTRY and DOOR_ENTRY[effective_cell] != (dr, dc):
                continue

            if effective_cell in valid_doors:
                key_index = ord(effective_cell.lower()) - ord("a")
                if not (keys >> key_index & 1):
                    continue

            new_keys = keys
            if effective_cell.islower():
                key_index = ord(effective_cell) - ord("a")
                new_keys = keys | (1 << key_index)

            # Trigger bomb on entry
            new_bombs = bombs_used
            if (nr, nc) in bomb_index:
                new_bombs = bombs_used | (1 << bomb_index[(nr, nc)])

            new_state = (nr, nc, new_keys, new_bombs)
            if new_state in visited:
                continue

            visited.add(new_state)
            parent[new_state] = (r, c, keys, bombs_used)
            queue.append(new_state)

    return None


def _reconstruct_bombs(parent: Dict, end_state: Tuple) -> List[Tuple[int, int]]:
    """Reconstruct (row, col) path from bomb-state parent map."""
    path: List[Tuple[int, int]] = []
    state: Optional[Tuple] = end_state
    while state is not None:
        r, c, _, _ = state
        path.append((r, c))
        state = parent[state]
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Q5b — Energy / weighted shortest path (Dijkstra)
# ---------------------------------------------------------------------------
#
# KEY INSIGHT for Q5b:
#   When cells have different costs, BFS finds shortest *hop* path, not
#   minimum-cost path. Use Dijkstra (min-heap on cumulative cost).
#   Each cell's cost is looked up from a cost_map; default cost = 1.
#
#   Prompt template for AI:
#     "Implement Dijkstra on this grid. cost_map[(r,c)] gives the cost to
#      ENTER cell (r,c). Return (total_cost, path). Use heapq."

def solve_maze_energy(
    grid: List[List[str]],
    cost_map: Optional[Dict[Tuple[int, int], float]] = None,
) -> Optional[Tuple[float, List[Tuple[int, int]]]]:
    """Dijkstra: find minimum-energy path from 'S' to 'E'.

    Args:
        grid:     2-D grid. '#' cells are impassable.
        cost_map: Optional dict mapping (r, c) -> entry cost.
                  Cells absent from the map default to cost 1.0.

    Returns:
        (total_cost, path) tuple, or None if unreachable.

    Raises:
        ValueError: If grid is empty or missing 'S' or 'E'.

    Complexity:
        Time:  O(M * N * log(M * N))
        Space: O(M * N)
    """
    if not grid or not grid[0]:
        raise ValueError("grid must be non-empty")

    rows, cols = len(grid), len(grid[0])
    positions = find_cells(grid, frozenset("SE"))

    if "S" not in positions:
        raise ValueError("Grid missing 'S'")
    if "E" not in positions:
        raise ValueError("Grid missing 'E'")

    start = positions["S"][0]
    end   = positions["E"][0]
    cost_map = cost_map or {}

    # heap: (cumulative_cost, row, col)
    heap: List = [(0.0, start[0], start[1])]
    dist: Dict[Tuple[int, int], float] = {start: 0.0}
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

    while heap:
        cost, r, c = heapq.heappop(heap)

        if (r, c) == end:
            return (cost, _reconstruct(parent, end))

        if cost > dist.get((r, c), float("inf")):
            continue  # stale entry

        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            cell = grid[nr][nc]
            if cell == "#":
                continue

            if cell in DOOR_ENTRY and DOOR_ENTRY[cell] != (dr, dc):
                continue

            entry_cost = cost_map.get((nr, nc), 1.0)
            new_cost = cost + entry_cost

            if new_cost < dist.get((nr, nc), float("inf")):
                dist[(nr, nc)] = new_cost
                parent[(nr, nc)] = (r, c)
                heapq.heappush(heap, (new_cost, nr, nc))

    return None


# ---------------------------------------------------------------------------
# Shared path reconstruction (for Q2/Q3 and Q5b)
# ---------------------------------------------------------------------------

def _reconstruct(
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
    end: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """Trace parent pointers from *end* back to start.

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
# Tests — 7 cases; bar = first 4
# ---------------------------------------------------------------------------

def _test() -> None:
    errors: List[str] = []

    def check(name: str, condition: bool, msg: str = "") -> None:
        if not condition:
            errors.append(f"FAIL  {name}: {msg}")
            print(f"  FAIL  {name}: {msg}")
        else:
            print(f"  PASS  {name}")

    # ── Q1: mark_path glyph priority ──────────────────────────────────────
    raw_simple = "S...\n####\nE..."  # no reachable path from S to E
    grid_simple = parse_maze(raw_simple)
    dummy_path = [(0, 0), (0, 1), (0, 2)]
    marked = mark_path(grid_simple, dummy_path)
    check("Q1_start_glyph", marked[0][0] == "S", f"got '{marked[0][0]}'")
    check("Q1_interior_marked", marked[0][1] == "*", f"got '{marked[0][1]}'")

    # ── Q2: BFS basic + no-path ────────────────────────────────────────────
    grid_basic = parse_maze("S.#\n...\n#.E")
    path_basic = solve_maze(grid_basic)
    check("Q2_finds_path",   path_basic is not None)
    check("Q2_start_end",    path_basic is not None and path_basic[0] == (0, 0) and path_basic[-1] == (2, 2))
    check("Q2_length",       path_basic is not None and len(path_basic) == 5, f"got {len(path_basic) if path_basic else '?'}")

    grid_blocked = parse_maze("S#E\n###\n...")
    check("Q2_no_path",      solve_maze(grid_blocked) is None)

    # ── Q3: directional gates ──────────────────────────────────────────────
    # '>' at (0,1): entry only rightward. Path S->(0,1)->(0,2)=E is valid.
    grid_gate_ok = parse_maze("S>E")
    path_gate = solve_maze(grid_gate_ok)
    check("Q3_gate_passable", path_gate == [(0, 0), (0, 1), (0, 2)])

    # '>' requires rightward; only access is from above (downward) — should fail
    grid_gate_block = parse_maze("S#\n>E")
    check("Q3_gate_blocks",   solve_maze(grid_gate_block) is None)

    # ── Q4: keys and doors ────────────────────────────────────────────────
    # Layout: S . a . A . E  (row 0)
    # Must collect 'a' before passing 'A'.
    grid_keys = parse_maze("S.a.A.E")
    path_keys = solve_maze_keys(grid_keys)
    check("Q4_key_unlocks_door", path_keys is not None and path_keys[-1] == (0, 6))
    if path_keys:
        cols = [c for _, c in path_keys]
        key_col  = 2  # 'a'
        door_col = 4  # 'A'
        check("Q4_key_before_door",
              cols.index(key_col) < cols.index(door_col),
              f"key at col {cols.index(key_col)}, door at {cols.index(door_col)}")

    # Door with no accessible key — unreachable
    grid_locked = parse_maze("S.A.E")  # 'A' present, no 'a' key
    check("Q4_locked_door",  solve_maze_keys(grid_locked) is None)

    # ── Q5a: bombs ────────────────────────────────────────────────────────
    # S . B # E  — wall at col 3 blocked, bomb at col 2 blasts it open.
    grid_bomb = parse_maze("S.B#E")
    path_bomb = solve_maze_bombs(grid_bomb)
    check("Q5a_bomb_clears_wall", path_bomb is not None and path_bomb[-1] == (0, 4),
          f"got {path_bomb}")

    # No bomb — same layout — should be unreachable
    grid_no_bomb = parse_maze("S..#E")
    check("Q5a_wall_without_bomb", solve_maze_bombs(grid_no_bomb) is None)

    # ── Q5b: Dijkstra ─────────────────────────────────────────────────────
    # Grid:  S . E
    # Costs: (0,1) = 10, (1,0) = 1, (1,1) = 1, (1,2) = 1, (0,2) = 1
    # Cheap route: S -> down -> right -> right -> up -> E  cost=5
    # Expensive:   S -> right -> right  cost=11
    grid_energy = parse_maze("S.E\n...")
    cost_map = {(0, 1): 10.0}  # straight path is costly
    result = solve_maze_energy(grid_energy, cost_map)
    check("Q5b_dijkstra_finds_path", result is not None)
    if result:
        total_cost, path_e = result
        check("Q5b_avoids_expensive_cell",
              (0, 1) not in path_e,
              f"path used costly cell: {path_e}")
        check("Q5b_cost_correct", total_cost == 4.0, f"expected 4.0, got {total_cost}")

    # get_affected_area correctness
    affected = get_affected_area(2, 2, 5, 5, radius=1)
    check("Q5a_blast_radius_1",
          len(affected) == 9,  # 3x3 = 9 cells
          f"expected 9 cells, got {len(affected)}")

    print()
    if errors:
        print(f"{len(errors)} test(s) failed.")
    else:
        print("All tests passed.")


if __name__ == "__main__":
    _test()
