# 74. Search a 2D Matrix (Medium)
#
# Given an m×n matrix where each row is sorted and the first integer of each
# row is greater than the last integer of the previous row, return True if
# target exists, False otherwise. Must run in O(log(m×n)).
#
# Approaches:
#   Step 1: Binary search on flattened index  O(log(m×n)) T / O(1) S  ← optimal
#   Step 2: Staircase search (top-right)      O(m+n)      T / O(1) S  ← for LC 240
#
# --------------------------------------------------------------------------
# Why binary search works here
#
#   The matrix is fully sorted when read row by row:
#     row 0 ends before row 1 starts, row 1 ends before row 2, etc.
#   → Treat as a 1D sorted array of length m×n.
#
#   Index mapping (key insight):
#     1D index mid → 2D position (mid//n, mid%n)
#     No extra space needed — just arithmetic.
#
# --------------------------------------------------------------------------
# Staircase search — O(m+n), for LC 240 comparison
#
#   Start at top-right corner (0, n-1).
#   At each step, one of three cases:
#     matrix[r][c] == target → found
#     matrix[r][c] >  target → move left  (c--)  eliminates entire column c
#     matrix[r][c] <  target → move down  (r++)  eliminates entire row r
#
#   Why it works: top-right is the largest in its row AND smallest in its
#   column. Moving left/down always eliminates exactly one row or column.
#
#   Why it's SLOWER for LC 74:
#     O(m+n) > O(log(m×n)) when the matrix is fully sorted (rows chain).
#     Binary search exploits the total ordering; staircase doesn't.
#
#   When staircase wins: LC 240 — each row and column is sorted, but
#   rows don't chain (row i+1's first element may be < row i's last).
#   Binary search on flattened index breaks there; staircase still works.
#
# --------------------------------------------------------------------------
# Complexity:
#   Binary search: Time O(log(m×n)), Space O(1)
#   Staircase:     Time O(m+n),      Space O(1)
# --------------------------------------------------------------------------


from typing import List


# ---------------------------------------------------------------------------
# Step 1: Binary search on flattened index — optimal for LC 74
# ---------------------------------------------------------------------------

def search_matrix(matrix: List[List[int]], target: int) -> bool:
    """
    Flatten 2D matrix to 1D via index arithmetic. Standard binary search.

    Key: mid//n gives row, mid%n gives column.

    Trace: matrix=[[1,3,5,7],[10,11,16,20],[23,30,34,60]], target=3
      m=3, n=4, search [0..11]
      mid=5: matrix[5//4][5%4] = matrix[1][1] = 11 > 3 → r=4
      mid=2: matrix[2//4][2%4] = matrix[0][2] = 5  > 3 → r=1
      mid=0: matrix[0//4][0%4] = matrix[0][0] = 1  < 3 → l=1
      mid=1: matrix[1//4][1%4] = matrix[0][1] = 3 == 3 → return True ✓

    Complexity:
        Time:  O(log(m×n))
        Space: O(1)
    """
    if not matrix or not matrix[0]:
        return False
    m, n = len(matrix), len(matrix[0])
    l, r = 0, m * n - 1
    while l <= r:
        mid = l + (r - l) // 2
        val = matrix[mid // n][mid % n]
        if val == target:
            return True
        elif val > target:
            r = mid - 1
        else:
            l = mid + 1
    return False


# ---------------------------------------------------------------------------
# Step 2: Staircase search — O(m+n), shown for contrast (better for LC 240)
# ---------------------------------------------------------------------------

def search_matrix_staircase(matrix: List[List[int]], target: int) -> bool:
    """
    Start at top-right. Move left if too big, down if too small.
    Each step eliminates one full row or column.

    Works for both LC 74 AND LC 240 (weaker sorted property).
    But O(m+n) is worse than O(log(m×n)) for LC 74 specifically.

    Trace: matrix=[[1,3,5,7],[10,11,16,20],[23,30,34,60]], target=3
      r=0, c=3: 7 > 3 → c=2
      r=0, c=2: 5 > 3 → c=1
      r=0, c=1: 3 == 3 → return True ✓

    Complexity:
        Time:  O(m+n)
        Space: O(1)
    """
    if not matrix or not matrix[0]:
        return False
    r, c = 0, len(matrix[0]) - 1
    while r < len(matrix) and c >= 0:
        val = matrix[r][c]
        if val == target:
            return True
        elif val > target:
            c -= 1
        else:
            r += 1
    return False


# ---------------------------------------------------------------------------
# LC 74 vs LC 240 — when to use which
#
#   LC 74: rows chain (row[i][-1] < row[i+1][0])
#     → Binary search O(log(m×n)) ← preferred
#     → Staircase O(m+n)          ← works but slower
#
#   LC 240: rows/cols sorted, rows do NOT chain
#     → Binary search on flat index: BROKEN (no total order)
#     → Staircase O(m+n)          ← only correct approach
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all() -> None:
    matrix1 = [[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]]
    matrix2 = [[1]]
    matrix3 = [[1, 1]]

    cases = [
        (matrix1, 3,  True),
        (matrix1, 13, False),
        (matrix1, 1,  True),
        (matrix1, 60, True),
        (matrix1, 61, False),
        (matrix2, 1,  True),
        (matrix2, 2,  False),
        (matrix3, 1,  True),
        (matrix3, 2,  False),
    ]
    for matrix, target, expected in cases:
        for fn in (search_matrix, search_matrix_staircase):
            result = fn(matrix, target)
            assert result == expected, \
                f"{fn.__name__}({matrix}, {target}) = {result}, expected {expected}"


if __name__ == "__main__":
    test_all()
    print("All tests passed")
