"""Print helpers for the maze."""

from solver import parse_maze, mark_path


def print_maze(grid: list[list[str]]) -> None:
    """Print the grid as-is."""
    for row in grid:
        print("".join(row))


def print_solution(raw: str, path: list[tuple[int, int]]) -> None:
    """
    Print the maze with the solution path marked.

    Expected: S and E glyphs must remain visible; interior path cells become '*'.
    """
    grid = parse_maze(raw)
    marked = mark_path(grid, path)
    print_maze(marked)
