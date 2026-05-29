"""
Compiler Optimization — Meta AI-Enabled Round (Confirmed Pool #5)
==================================================================

Problem:
    Given a program as a list of assignment instructions, compute the total
    execution cost using the following cost model:

        addition (+)       -> 1 unit
        subtraction (-)    -> 1 unit
        assignment (=)     -> 1 unit  (for simple variable-to-variable copies)
        multiplication (*) -> 5 units
        division (/)       -> 5 units

    Instructions format:
        "var = expr"   where expr is one of:
            - literal integer:          "x = 5"
            - another variable:         "y = x"
            - binary operation:         "z = x + y"   or   "z = x * 3"

    Extended tasks:
        Checkpoint 2: Track variable values through the program (constant folding)
        Checkpoint 3: Identify dead code — assignments to variables never read

Checkpoint structure (mirrors actual Meta AI-enabled interview):
    Checkpoint 1: Parse instructions and compute total cost
    Checkpoint 2: Evaluate final variable values (constant propagation)
    Checkpoint 3: Report dead assignments (variables written but never read)

Key insight (ambiguity handling):
    Interviewers may give incomplete cost rules mid-problem. Always ask:
    "What is the cost of a simple assignment like x = 5?" before implementing.
    Guard against unknown operators by raising a clear ValueError.

Complexity:
    parse_program:  Time O(N * L) where L = avg instruction length
    compute_cost:   Time O(N)
    evaluate:       Time O(N)
    find_dead_code: Time O(N)
    Space: O(N) for all operations
"""

from typing import Dict, List, Optional, Tuple


# Operation cost model
OP_COSTS: Dict[str, int] = {
    "+": 1,
    "-": 1,
    "=": 1,  # simple variable copy
    "*": 5,
    "/": 5,
}

BINARY_OPS = frozenset(OP_COSTS.keys()) - {"="}


# ---------------------------------------------------------------------------
# Instruction representation
# ---------------------------------------------------------------------------

class Instruction:
    """Parsed single assignment instruction.

    Attributes:
        target:  Left-hand side variable name.
        op:      Operator ('+', '-', '*', '/') or None for literal/copy.
        lhs:     Left operand (variable name or int literal).
        rhs:     Right operand (variable name or int literal), or None.
    """

    __slots__ = ("target", "op", "lhs", "rhs")

    def __init__(
        self,
        target: str,
        op: Optional[str],
        lhs: "int | str",
        rhs: "Optional[int | str]",
    ) -> None:
        self.target = target
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self) -> str:
        if self.op is None:
            return f"Instruction({self.target} = {self.lhs})"
        return f"Instruction({self.target} = {self.lhs} {self.op} {self.rhs})"


def _parse_token(token: str) -> "int | str":
    """Parse a token as int literal or variable name.

    Args:
        token: String token from instruction.

    Returns:
        Integer if numeric, else the token as a variable name string.
    """
    try:
        return int(token)
    except ValueError:
        return token


def parse_instruction(line: str) -> Instruction:
    """Parse a single assignment instruction string.

    Supported formats:
        "x = 5"       — assign literal
        "y = x"       — variable copy
        "z = x + y"   — binary op with variable operands
        "z = x + 3"   — binary op with mixed operands

    Args:
        line: Instruction string (stripped of whitespace).

    Returns:
        Parsed Instruction object.

    Raises:
        ValueError: If line does not match expected format or operator unknown.

    Complexity:
        Time:  O(L) where L = len(line)
        Space: O(1)
    """
    line = line.strip()
    if "=" not in line:
        raise ValueError(f"Not an assignment: {line!r}")

    eq_idx = line.index("=")
    target = line[:eq_idx].strip()
    expr = line[eq_idx + 1:].strip()

    if not target:
        raise ValueError(f"Missing assignment target in: {line!r}")

    # Detect binary operation in expr
    for op in BINARY_OPS:
        parts = expr.split(op, 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            lhs = _parse_token(parts[0].strip())
            rhs = _parse_token(parts[1].strip())
            return Instruction(target, op, lhs, rhs)

    # Simple assignment: literal or variable copy
    lhs = _parse_token(expr)
    return Instruction(target, None, lhs, None)


def parse_program(lines: List[str]) -> List[Instruction]:
    """Parse a list of instruction strings into Instruction objects.

    Args:
        lines: List of instruction strings.

    Returns:
        List of parsed Instructions, in order.

    Raises:
        ValueError: If any line is malformed.

    Complexity:
        Time:  O(N * L)
        Space: O(N)
    """
    return [parse_instruction(line) for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# Checkpoint 1: Total cost computation
# ---------------------------------------------------------------------------

def compute_cost(instructions: List[Instruction]) -> int:
    """Compute total execution cost of a program.

    Args:
        instructions: Parsed program.

    Returns:
        Total cost in units.

    Complexity:
        Time:  O(N)
        Space: O(1)
    """
    total = 0
    for instr in instructions:
        if instr.op is None:
            # Literal assignment: free (constant load) or simple copy: 1 unit
            total += 0 if isinstance(instr.lhs, int) else OP_COSTS["="]
        else:
            total += OP_COSTS[instr.op]
    return total


# ---------------------------------------------------------------------------
# Checkpoint 2: Constant propagation / value evaluation
# ---------------------------------------------------------------------------

def evaluate_program(instructions: List[Instruction]) -> Dict[str, Optional[int]]:
    """Evaluate final variable values through constant propagation.

    Tracks integer values where determinable. Variables with non-constant
    dependencies are recorded as None.

    Args:
        instructions: Parsed program.

    Returns:
        Dict mapping variable name -> final integer value or None if unknown.

    Complexity:
        Time:  O(N)
        Space: O(V) where V = number of unique variables
    """
    env: Dict[str, Optional[int]] = {}

    def _resolve(operand: "int | str") -> Optional[int]:
        if isinstance(operand, int):
            return operand
        return env.get(operand)  # None if variable not yet assigned or unknown

    for instr in instructions:
        if instr.op is None:
            env[instr.target] = _resolve(instr.lhs)
        else:
            lval = _resolve(instr.lhs)
            rval = _resolve(instr.rhs)
            if lval is None or rval is None:
                env[instr.target] = None
            elif instr.op == "+":
                env[instr.target] = lval + rval
            elif instr.op == "-":
                env[instr.target] = lval - rval
            elif instr.op == "*":
                env[instr.target] = lval * rval
            elif instr.op == "/":
                env[instr.target] = lval // rval if rval != 0 else None
            else:
                env[instr.target] = None

    return env


# ---------------------------------------------------------------------------
# Checkpoint 3: Dead code detection
# ---------------------------------------------------------------------------

def find_dead_assignments(instructions: List[Instruction]) -> List[str]:
    """Find variables that are assigned but never read afterward.

    A variable is "dead" if its last write is never followed by a read
    before the program ends (or before the next overwrite).

    Args:
        instructions: Parsed program.

    Returns:
        List of variable names that have at least one dead assignment,
        in the order they are first detected.

    Complexity:
        Time:  O(N)
        Space: O(V)
    """
    # Track whether each variable is read after its last assignment
    last_write: Dict[str, int] = {}   # variable -> instruction index of last write
    read_after_write: set = set()     # variables that are read after their last write

    for i, instr in enumerate(instructions):
        # Record reads
        for operand in [instr.lhs, instr.rhs]:
            if isinstance(operand, str):
                read_after_write.add(operand)

        # Record write (comes after reads in the same instruction)
        last_write[instr.target] = i
        # A new write resets read status for this variable
        read_after_write.discard(instr.target)

    # Variables written but never read after their last write
    dead = [var for var in last_write if var not in read_after_write]
    return sorted(dead)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    # Checkpoint 1: parse and cost
    prog1 = [
        "x = 5",        # literal assign: cost 0
        "y = 3",        # literal assign: cost 0
        "z = x + y",    # addition: cost 1
        "w = z * 2",    # multiplication: cost 5
        "v = w",        # copy: cost 1
    ]
    instrs1 = parse_program(prog1)
    assert len(instrs1) == 5
    cost1 = compute_cost(instrs1)
    assert cost1 == 0 + 0 + 1 + 5 + 1, f"Expected cost 7, got {cost1}"

    # Checkpoint 2: evaluate
    env1 = evaluate_program(instrs1)
    assert env1["x"] == 5
    assert env1["y"] == 3
    assert env1["z"] == 8
    assert env1["w"] == 16
    assert env1["v"] == 16

    # Division and subtraction
    prog2 = ["a = 10", "b = 3", "c = a - b", "d = a / b"]
    instrs2 = parse_program(prog2)
    env2 = evaluate_program(instrs2)
    assert env2["c"] == 7
    assert env2["d"] == 3  # integer division

    # Checkpoint 3: dead code
    prog3 = [
        "x = 5",
        "y = x",    # reads x (x is alive)
        "z = 3",    # z is never read -> dead
        "result = y + 1",  # reads y (y is alive); result is never read -> dead
    ]
    instrs3 = parse_program(prog3)
    dead3 = find_dead_assignments(instrs3)
    assert "z" in dead3, f"z should be dead: {dead3}"
    assert "result" in dead3, f"result should be dead: {dead3}"
    assert "x" not in dead3, f"x is read by y: {dead3}"
    assert "y" not in dead3, f"y is read by result: {dead3}"

    # Cost of division is 5
    prog4 = ["x = 10", "y = x / 2"]
    instrs4 = parse_program(prog4)
    assert compute_cost(instrs4) == 5, f"Division cost wrong: {compute_cost(instrs4)}"

    print("  compiler_optimization: all tests passed")


if __name__ == "__main__":
    _test()
