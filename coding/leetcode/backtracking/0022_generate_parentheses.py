# 22. Generate Parentheses (Medium)
#
# Given n pairs of parentheses, generate all combinations of well-formed
# parentheses.
#
# Approach: Backtracking / DFS
#   At each step, we can add '(' if open count < n,
#   or ')' if close count < open count (keeps string valid at every prefix).
#
# --------------------------------------------------------------------------
# Why this pruning is correct:
#
#   A valid sequence requires:
#     1. Never more than n '(' total.
#     2. Never more ')' than '(' at any prefix (no premature close).
#
#   These two rules are both necessary and sufficient for well-formedness.
#   The recursion enforces them structurally — no post-hoc filtering needed.
#
# --------------------------------------------------------------------------
# res.append(out) vs res.append(out[::])
#
#   In this code, both are identical. out[::] is a full-slice copy of the
#   string, but Python strings are IMMUTABLE — they can never be mutated
#   by subsequent recursive calls. `out + '('` always creates a NEW string
#   object; it does not modify out. So the copy is unnecessary.
#
#   The out[::] pattern is cargo-culted from LIST-based backtracking, where
#   a copy IS required:
#
#     # List backtracking — copy needed
#     def helper(res, path, ...):
#         if done:
#             res.append(path[:])   # MUST copy — path is mutated by pop()
#             return
#         path.append(x)
#         helper(res, path, ...)
#         path.pop()                # mutates in place → corrupts res without copy
#
#   If you wrote res.append(path) there, every entry in res would point to
#   the SAME list object and end up as whatever path looks like at the end.
#
#   Rule of thumb:
#     String in recursion  → res.append(out)     # immutable, no copy needed
#     List in backtracking → res.append(path[:]) # mutable, copy required
#
# --------------------------------------------------------------------------
# Complexity:
#   Time:  O(4^n / sqrt(n))  — nth Catalan number of valid sequences,
#                               each of length 2n
#   Space: O(n)              — recursion depth is at most 2n
# --------------------------------------------------------------------------


from typing import List


def generate_parenthesis(n: int) -> List[str]:
    """
    Backtracking: build the string character by character.
    l = number of '(' placed so far.
    r = number of ')' placed so far.

    Trace: n=2
      helper('',  l=0, r=0)
        add '(' → helper('(',  l=1, r=0)
          add '(' → helper('((',  l=2, r=0)
            add ')' → helper('(()',  l=2, r=1)
              add ')' → helper('(())', l=2, r=2) → append '(())'
          add ')' → helper('()',  l=1, r=1)  [r<l so allowed]
            add '(' → helper('()(',  l=2, r=1)
              add ')' → helper('()()', l=2, r=2) → append '()()'
      Result: ['(())', '()()']

    Complexity:
        Time:  O(4^n / sqrt(n))
        Space: O(n)  recursion stack
    """
    if n <= 0:
        return []
    res = []

    def helper(out: str, l: int, r: int) -> None:
        if l == n and r == n:
            res.append(out)       # strings are immutable — no copy needed
            return
        if l < n:
            helper(out + '(', l + 1, r)
        if r < l:
            helper(out + ')', l, r + 1)

    helper('', 0, 0)
    return res


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all() -> None:
    cases = [
        (1, ["()"]),
        (2, ["(())", "()()"]),
        (3, ["((()))", "(()())", "(())()", "()(())", "()()()"]),
        (0, []),
    ]
    for n, expected in cases:
        result = sorted(generate_parenthesis(n))
        assert result == sorted(expected), \
            f"generate_parenthesis({n}) = {result}, expected {sorted(expected)}"


if __name__ == "__main__":
    test_all()
    print("All tests passed")
