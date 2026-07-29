# 69. Sqrt(x) (Easy)
#
# Return the integer square root of x (rounded down). No built-in sqrt.
#
# Approaches:
#   Step 1: Binary search    O(log x)     T / O(1) S  — easy to derive
#   Step 2: Bit manipulation O(log x)     T / O(1) S  — cleaner constants
#   Step 3: Newton's method  O(log log x) T / O(1) S  — production approach
#
# --------------------------------------------------------------------------
# Comparison
#
#   Approach          Time          Iterations for x=2³¹   Derive in interview
#   ──────────────    ──────────    ─────────────────────   ───────────────────
#   Binary search     O(log x)      ~31                     Easy
#   Bit manipulation  O(log x)      ~16                     Medium
#   Newton's method   O(log log x)  ~5                      Hard but impressive
#
#   Interview strategy: implement binary search, then mention Newton's method
#   as the production approach — it's how hardware floating-point sqrt works.
#
# --------------------------------------------------------------------------
# Newton's method — intuition first, math second
#
#   Core idea — two complementary guesses:
#     If t is your guess and t is TOO BIG   → x/t is TOO SMALL.
#     If t is your guess and t is TOO SMALL → x/t is TOO BIG.
#     The truth is always sandwiched between t and x/t.
#     Best single estimate from two bracketing values → take their average:
#
#       t_next = (t + x/t) / 2
#
#   One-line intuition:
#     "If your guess is too high, x divided by your guess is too low —
#      average them. Repeat."
#
#   Example: sqrt(16), starting from t=6
#     t=6:    x/t=16/6≈2.67    →  avg=(6+2.67)/2≈4.33
#     t=4.33: x/t=16/4.33≈3.70 →  avg=(4.33+3.70)/2≈4.01
#     t=4.01: x/t=16/4.01≈3.99 →  avg=(4.01+3.99)/2=4.00 ✓
#     3 steps from t=6 to answer.
#
#   Why it never undershoots (always approaches from above):
#     AM-GM: (t + x/t)/2 ≥ sqrt(t · x/t) = sqrt(x)
#     So t_next ≥ sqrt(x) always. Loop stops as soon as t²≤x.
#
#   Why convergence is quadratic (error SQUARES each step):
#     step 1: error ≈ 10%
#     step 2: error ≈ 1%      (100× smaller)
#     step 3: error ≈ 0.01%   (100× smaller again)
#     Binary search halves error. Newton's squares it.
#
#   Math derivation (for completeness):
#     f(t) = t²-x,  f'(t) = 2t
#     t_new = t - f(t)/f'(t) = t - (t²-x)/(2t) = (t+x/t)/2
#
#   For x=2³¹ (~2 billion), Newton's needs ~5 iterations.
#   Binary search needs ~31 iterations.
#   Used in hardware FPU sqrt implementations.
#
# --------------------------------------------------------------------------
# Bit manipulation — how it works
#
#   Build the answer bit by bit from the highest possible bit downward.
#   Since sqrt(2³¹) ≈ 46341 < 2^16, the highest bit we need is 2^15.
#
#   At each step: "can I set this bit in the result without exceeding sqrt(x)?"
#   If (result + bit)² <= x → set the bit.
#
# --------------------------------------------------------------------------
# Binary search — why `return r` (not `l`)
#
#   Loop invariant: r always holds the largest mid where mid²<=x.
#   When the loop ends, l > r. r = l-1 = last valid floor sqrt.
#
#   Loop condition collapses `mid*mid <= x → l = mid+1` meaning l overshoots.
#   r is left behind at the correct answer.
# --------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Step 1: Binary search
# ---------------------------------------------------------------------------

def my_sqrt_binary(x: int) -> int:
    """
    Binary search on answer in [0, x].
    When mid²<=x, l moves right (mid could be answer, try larger).
    When mid²>x,  r moves left  (mid too big).
    At exit, r = floor(sqrt(x)).

    Trace: x=8
      l=0,r=8, mid=4: 16>8  → r=3
      l=0,r=3, mid=1: 1<=8  → l=2
      l=2,r=3, mid=2: 4<=8  → l=3
      l=3,r=3, mid=3: 9>8   → r=2
      l=3>r=2: exit. return r=2 ✓

    Complexity:
        Time:  O(log x)
        Space: O(1)
    """
    if x == 0 or x == 1:
        return x
    l, r = 0, x
    while l <= r:
        mid = l + (r - l) // 2
        if mid * mid <= x:
            l = mid + 1
        else:
            r = mid - 1
    return r


# ---------------------------------------------------------------------------
# Step 2: Bit manipulation
# ---------------------------------------------------------------------------

def my_sqrt_bit(x: int) -> int:
    """
    Build result bit by bit from highest possible bit (2^15) down to 1.
    At each step: if adding this bit keeps result² <= x, keep it.

    sqrt(2^31) ≈ 46341 < 2^16, so starting bit = 1<<15 = 32768.

    Trace: x=8
      bit=32768..4: (0+bit)² > 8 → skip
      bit=4: (0+4)²=16 > 8 → skip
      bit=2: (0+2)²=4  ≤ 8 → result=2
      bit=1: (2+1)²=9  > 8 → skip
      return 2 ✓

    Complexity:
        Time:  O(log x)   — 16 iterations max for x < 2^31
        Space: O(1)
    """
    result = 0
    bit = 1 << 15          # highest bit: sqrt(2^31) < 2^16
    while bit > 0:
        if (result + bit) * (result + bit) <= x:
            result += bit
        bit >>= 1
    return result


# ---------------------------------------------------------------------------
# Step 3: Newton's method
# ---------------------------------------------------------------------------

def my_sqrt(x: int) -> int:
    """
    Newton's iteration: t = (t + x//t) // 2
    Starts from t=x (upper bound), converges from above.
    Loop terminates when t² <= x (floor condition satisfied).

    Why always converges from above (for t > sqrt(x)):
      AM-GM: (t + x/t)/2 >= sqrt(t * x/t) = sqrt(x)
      So t_new >= sqrt(x) always, pulling closer each step without undershoot.

    Trace: x=8, start t=8
      t=8:  (8 + 8//8)//2 = 9//2 = 4
      t=4:  (4 + 8//4)//2 = 6//2 = 3
      t=3:  (3 + 8//3)//2 = 5//2 = 2
      t=2:  2*2=4 ≤ 8 → stop. return 2 ✓

    Trace: x=2147483647 (2³¹-1): converges in ~5 iterations.

    Complexity:
        Time:  O(log log x)  — quadratic convergence
        Space: O(1)
    """
    if x == 0:
        return 0
    t = x
    while t * t > x:
        t = (t + x // t) // 2
    return t


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all() -> None:
    cases = [
        (0,           0),
        (1,           1),
        (2,           1),
        (3,           1),
        (4,           2),
        (8,           2),
        (9,           3),
        (15,          3),
        (16,          4),
        (25,          5),
        (26,          5),
        (100,        10),
        (2147395600, 46340),   # near 2^31, floor sqrt = 46340
        (2147483647, 46340),   # 2^31 - 1
    ]
    for x, expected in cases:
        for fn in (my_sqrt_binary, my_sqrt_bit, my_sqrt):
            result = fn(x)
            assert result == expected, \
                f"{fn.__name__}({x}) = {result}, expected {expected}"


if __name__ == "__main__":
    test_all()
    print("All tests passed")
