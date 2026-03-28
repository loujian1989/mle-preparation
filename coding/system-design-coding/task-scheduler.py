"""
Task Scheduler
==============

Problem A — CPU Task Scheduler (LeetCode 621):
    Given a list of CPU tasks (characters A–Z) and a cooldown n,
    find the minimum intervals to finish all tasks.
    Tasks of the same type must be at least n+1 intervals apart.
    CPU can be idle during cooldown.

    Example: tasks = ["A","A","A","B","B","B"], n = 2
    Output:  8  → A→B→idle→A→B→idle→A→B

    Key insight: Most frequent task drives the minimum time.
    Formula: max(len(tasks), (max_count - 1) * (n + 1) + count_of_max_tasks)

    Complexity:
        Time:  O(T) where T = number of tasks
        Space: O(1) — at most 26 unique task types

Problem B — Production Task Scheduler (min-heap, generic):
    Schedule tasks with priorities and deadlines.
    pop_next() returns the highest-priority ready task.
    Used in: Celery workers, ML job queues, batch inference schedulers.

    Complexity:
        Time:  O(log N) push/pop, O(N) for priority-ordered drain
        Space: O(N) for N pending tasks
"""

import heapq
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional


# ---------------------------------------------------------------------------
# Problem A: CPU Task Scheduler (LeetCode 621)
# ---------------------------------------------------------------------------

def least_interval(tasks: List[str], n: int) -> int:
    """Compute minimum intervals to execute all CPU tasks with cooldown n.

    Args:
        tasks: List of task identifiers (single uppercase letters).
        n:     Cooldown — same task type must wait at least n intervals.

    Returns:
        Minimum number of CPU intervals (including idle slots).

    Raises:
        ValueError: If n < 0.

    Examples:
        >>> least_interval(["A","A","A","B","B","B"], 2)
        8
        >>> least_interval(["A","A","A","B","B","B"], 0)
        6

    Complexity:
        Time:  O(T) where T = len(tasks)
        Space: O(1) — frequency array of size 26
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if not tasks:
        return 0

    freq = [0] * 26
    for t in tasks:
        freq[ord(t) - ord("A")] += 1

    freq.sort()
    max_count = freq[25]                        # highest frequency
    count_of_max = sum(1 for f in freq if f == max_count)

    # Slots filled by most-frequent task(s) + idle
    formula_result = (max_count - 1) * (n + 1) + count_of_max
    return max(len(tasks), formula_result)


# ---------------------------------------------------------------------------
# Problem B: Production Task Scheduler (priority + deadline aware)
# ---------------------------------------------------------------------------

@dataclass(order=True)
class Task:
    """A schedulable unit of work.

    Ordering: lower priority number = higher urgency (like UNIX nice level).
    For equal priority, earlier deadline wins.
    """

    priority: int            # 0 = highest urgency
    deadline: float          # Unix timestamp; float('inf') = no deadline
    task_id: str = field(compare=False)
    payload: Any = field(default=None, compare=False)


class PriorityTaskScheduler:
    """Priority + deadline-aware task scheduler backed by a min-heap.

    Used pattern: ML training job queues, async inference pipelines,
    batch feature computation schedulers.

    Args:
        None

    Notes:
        - Not thread-safe as written; wrap push/pop in a Lock for production.
        - For distributed use, replace heap with Redis sorted set (ZADD/ZPOPMIN).
    """

    def __init__(self) -> None:
        self._heap: List[Task] = []
        self._task_count = 0   # tie-breaker for equal priority + deadline

    def push(self, task: Task) -> None:
        """Enqueue a task.

        Args:
            task: Task to schedule.

        Complexity:
            Time:  O(log N)
            Space: O(1)
        """
        heapq.heappush(self._heap, task)
        self._task_count += 1

    def pop_next(self) -> Optional[Task]:
        """Return and remove the highest-priority ready task.

        Returns:
            Highest-priority Task, or None if queue is empty.

        Complexity:
            Time:  O(log N)
            Space: O(1)
        """
        if not self._heap:
            return None
        return heapq.heappop(self._heap)

    def peek(self) -> Optional[Task]:
        """Return the highest-priority task without removing it.

        Returns:
            Highest-priority Task, or None if queue is empty.

        Complexity:
            Time:  O(1)
            Space: O(1)
        """
        return self._heap[0] if self._heap else None

    def drain_overdue(self, now: Optional[float] = None) -> List[Task]:
        """Return all tasks past their deadline, sorted by priority.

        Args:
            now: Current time (default: time.time()).

        Returns:
            List of overdue tasks, highest-priority first.

        Complexity:
            Time:  O(N log N)
            Space: O(N)
        """
        if now is None:
            now = time.time()
        overdue = [t for t in self._heap if t.deadline <= now]
        for t in overdue:
            self._heap.remove(t)
        heapq.heapify(self._heap)
        return sorted(overdue)

    def __len__(self) -> int:
        return len(self._heap)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test_least_interval() -> None:
    assert least_interval(["A", "A", "A", "B", "B", "B"], 2) == 8
    assert least_interval(["A", "A", "A", "B", "B", "B"], 0) == 6
    assert least_interval(["A", "A", "A", "A", "A", "A", "B", "C", "D", "E", "F", "G"], 2) == 16
    assert least_interval([], 5) == 0
    print("  least_interval: all tests passed")


def _test_priority_scheduler() -> None:
    sched = PriorityTaskScheduler()
    assert sched.pop_next() is None  # empty queue

    t_low  = Task(priority=5, deadline=float("inf"), task_id="low")
    t_high = Task(priority=1, deadline=float("inf"), task_id="high")
    t_mid  = Task(priority=3, deadline=float("inf"), task_id="mid")

    sched.push(t_low)
    sched.push(t_high)
    sched.push(t_mid)

    assert sched.pop_next().task_id == "high"   # type: ignore[union-attr]
    assert sched.pop_next().task_id == "mid"    # type: ignore[union-attr]
    assert sched.pop_next().task_id == "low"    # type: ignore[union-attr]
    assert sched.pop_next() is None

    # Overdue drain
    past = time.time() - 100
    sched.push(Task(priority=2, deadline=past,         task_id="overdue1"))
    sched.push(Task(priority=1, deadline=past,         task_id="overdue2"))
    sched.push(Task(priority=1, deadline=float("inf"), task_id="future"))

    overdue = sched.drain_overdue()
    assert len(overdue) == 2
    assert {t.task_id for t in overdue} == {"overdue1", "overdue2"}
    assert len(sched) == 1   # "future" remains

    print("  PriorityTaskScheduler: all tests passed")


if __name__ == "__main__":
    print("Task Scheduler tests")
    _test_least_interval()
    _test_priority_scheduler()
