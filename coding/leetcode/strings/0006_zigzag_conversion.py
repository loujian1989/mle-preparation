# 6. Zigzag Conversion (Medium)
#
# Write "PAYPALISHIRING" in a zigzag pattern over numRows rows, then read
# line by line.
#
# numRows=3:            numRows=4:
#   P   A   H   N         P     I    N
#   A P L S I I G         A   L S  I G
#   Y   I   R             Y A   H R
#                         P     I
#
# Approaches:
#   Step 1: Simulation — assign each char to its row, concatenate  O(n) T / O(n) S
#   Step 2: Math       — directly index chars per row via cycle     O(n) T / O(n) S
#
# Both are O(n) — simulation is cleaner to derive in an interview.
#
# --------------------------------------------------------------------------
# Common mistake: missing edge-case guard for numRows == 1
#
# BUG: when numRows == 1, both boundary conditions fire at index 0:
#     index == 0          → step = 1
#     index == numRows-1  → (also true, but elif skips it)
#   step stays 1, so index goes to 1 on the next character,
#   but res only has 1 element → IndexError.
#
# Trace on s="AB", numRows=1:
#   res = ['']
#   c='A': res[0]+='A', index==0 → step=1, index=1
#   c='B': res[1]+='B'  ← IndexError
#
# Fix: guard at the top.
#   if numRows == 1 or numRows >= len(s): return s
#
# numRows >= len(s) is also worth guarding: every char lands in its own row,
# so reading row by row just gives back the original string.
# --------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Step 1: Simulation (row builder)
# ---------------------------------------------------------------------------

def convert_simulation(s: str, numRows: int) -> str:
    """
    Maintain one string per row. Walk s with a bouncing index (0→numRows-1→0).
    Reverse direction at the top (index==0) and bottom (index==numRows-1).
    Concatenate all rows at the end.

    Complexity:
        Time:  O(n)  — single pass through s
        Space: O(n)  — row strings collectively hold all characters
    """
    if numRows == 1 or numRows >= len(s):   # edge-case guard (see mistake log above)
        return s

    rows = [[] for _ in range(numRows)]
    index, step = 0, 1

    for c in s:
        rows[index].append(c)
        if index == 0:
            step = 1
        elif index == numRows - 1:
            step = -1
        index += step

    return ''.join(''.join(row) for row in rows)


# ---------------------------------------------------------------------------
# Step 2: Math (cycle indexing)
# ---------------------------------------------------------------------------

def convert_math(s: str, numRows: int) -> str:
    """
    In a zigzag with numRows rows, characters repeat in cycles of length
    cycle = 2 * numRows - 2.

    For each row r, the characters are at positions:
      - Main diagonal:      r, r+cycle, r+2*cycle, ...
      - Inner diagonal:     cycle-r, cycle-r+cycle, ...  (only for 0 < r < numRows-1)

    Build output by iterating rows and collecting the right indices directly.

    Complexity:
        Time:  O(n)  — each character visited once
        Space: O(n)  — output string
    """
    if numRows == 1 or numRows >= len(s):
        return s

    n = len(s)
    cycle = 2 * numRows - 2
    result = []

    for r in range(numRows):
        for j in range(r, n, cycle):           # main diagonal positions
            result.append(s[j])
            inner = j + cycle - 2 * r          # inner diagonal position
            if 0 < r < numRows - 1 and inner < n:
                result.append(s[inner])

    return ''.join(result)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all() -> None:
    cases = [
        ("PAYPALISHIRING", 3, "PAHNAPLSIIGYIR"),
        ("PAYPALISHIRING", 4, "PINALSIGYAHRPI"),
        ("A",              1, "A"),              # numRows == 1 edge case
        ("AB",             1, "AB"),             # triggered the bug
        ("ABC",            2, "ACB"),
        ("ABCD",           3, "ABDC"),
        ("A",              2, "A"),              # numRows >= len(s) edge case
    ]
    for s, numRows, expected in cases:
        for fn in (convert_simulation, convert_math):
            result = fn(s, numRows)
            assert result == expected, \
                f"{fn.__name__}({s!r}, {numRows}) = {result!r}, expected {expected!r}"


if __name__ == "__main__":
    test_all()
    print("All tests passed")
