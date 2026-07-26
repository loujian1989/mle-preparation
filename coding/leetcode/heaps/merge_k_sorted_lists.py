# 23. Merge K Sorted Lists (Hard)
#
# Given an array of k sorted linked lists, merge them into one sorted list.
#
# Two approaches — both O(N log k):
#   Method 1: Min-heap   — O(N log k) T / O(k) S
#   Method 2: Divide & conquer — O(N log k) T / O(log k) S  (recursion stack)
#
# N = total nodes across all lists, k = number of lists
#
# --------------------------------------------------------------------------
# Method 1: Min-Heap
#
#   Push the head of each non-empty list into a min-heap.
#   Repeatedly pop the minimum node, append to result, push its successor.
#   Heap size stays ≤ k at all times.
#
#   Why O(N log k):
#     Each of the N nodes is pushed and popped exactly once.
#     Each push/pop on a heap of size k costs O(log k).
#     Total: O(N log k).
#
# --------------------------------------------------------------------------
# Method 2: Divide & Conquer
#
#   Merge lists in pairs, halving the problem each round.
#   Round 1: k lists → k/2 merged lists
#   Round 2: k/2    → k/4
#   ...
#   log k rounds total; each round processes all N nodes once → O(N log k).
#
#   Implemented iteratively with a step pointer:
#     step=1: merge (0,1), (2,3), (4,5), ...
#     step=2: merge (0,2), (4,6), ...
#     step=4: merge (0,4), ...
#
# --------------------------------------------------------------------------
# Comparison:
#
#   Method        Time        Space       Notes
#   ──────────    ──────────  ─────────   ──────────────────────────────────
#   Min-heap      O(N log k)  O(k)        Heap holds one node per list
#   D&C           O(N log k)  O(log k)    Recursion stack for mergeTwoLists
#
#   Both are O(N log k). D&C has slightly better space in theory.
#   Min-heap is more intuitive and easier to derive in an interview.
#
# --------------------------------------------------------------------------
# Bugs encountered
#
#   Bug 1 (D&C) — discarding the return value of mergeTwoLists:
#
#     WRONG:   self.mergeTwoLists(lists[i], lists[i+step])
#     CORRECT: lists[i] = self.mergeTwoLists(lists[i], lists[i+step])
#
#     mergeTwoLists returns the HEAD of the merged list, which is NOT always
#     lists[i]. When l2.val < l1.val the function returns l2 — a different
#     node. Without the assignment, lists[i] still points to the old head
#     (now somewhere in the middle of the merged result). The next round
#     starts from the wrong node and loses the prefix.
#
#     Concrete trace: merge [3,4,5] and [1,2]
#       l1.val=3 > l2.val=1 → returns Node(1) as new head
#       Without lists[i]=...: lists[i] still points to Node(3) → [1,2] lost
#
#   Bug 2 (D&C) — wrong empty list guard placement:
#
#     WRONG:
#       if not lists: return None   # early return at top
#       ...
#       return lists[0]             # unguarded — fine here but fragile
#
#     CORRECT (preferred):
#       return lists[0] if sz else None   # single exit point, always safe
#
#     Note: lists=[[]] (one empty list) is NOT caught by `if not lists`
#     because lists has one element. The while loop is skipped (step=1,
#     step<1 is False) and lists[0]=None is returned correctly either way.
#
#   Bug 3 (Min-heap) — naive (val, node) tuple breaks in Python 3:
#
#     WRONG:   heapq.heappush(heap, (l.val, l))
#     CORRECT: heapq.heappush(heap, (l.val, counter, l))
#
#     When two nodes have equal val, Python 3 tries ListNode < ListNode
#     via __lt__. ListNode has no __lt__ → TypeError.
#     Python 2 fell back to comparing by memory address (id) — no error.
#     Fix: add a unique counter as tiebreaker so comparison never reaches
#     the node object.
#
#   Bug 4 (Min-heap) — dead statement in initialization loop:
#
#     for l in lists:
#         if l:
#             heapq.heappush(heap, (l.val, counter, l))
#             l = l.next   # BUG: dead — reassigns local var after push;
#                          # the next for-iteration overwrites l anyway.
#                          # Remove this line.
# --------------------------------------------------------------------------


import heapq
from typing import List, Optional


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None) -> None:
        self.val = val
        self.next = next


# ---------------------------------------------------------------------------
# Method 1: Min-Heap
# ---------------------------------------------------------------------------

def merge_k_lists_heap(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """
    Min-heap: always pop the globally smallest node.

    Complexity:
        Time:  O(N log k)
        Space: O(k)  — heap holds at most one node per list
    """
    heap: list = []
    counter = 0

    for node in lists:
        if node:
            heapq.heappush(heap, (node.val, counter, node))
            counter += 1

    dummy = ListNode(0)
    cur = dummy
    while heap:
        _, _, node = heapq.heappop(heap)
        cur.next = node
        cur = cur.next
        if node.next:
            heapq.heappush(heap, (node.next.val, counter, node.next))
            counter += 1

    return dummy.next


# ---------------------------------------------------------------------------
# Method 2: Divide & Conquer
# ---------------------------------------------------------------------------

def _merge_two(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    """Merge two sorted lists recursively. Returns the new head."""
    if not l1:
        return l2
    if not l2:
        return l1
    if l1.val <= l2.val:
        l1.next = _merge_two(l1.next, l2)
        return l1
    else:
        l2.next = _merge_two(l1, l2.next)
        return l2


def merge_k_lists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """
    Divide & conquer: merge lists in pairs, halving each round.

    step=1: merge (0,1),(2,3),(4,5),...
    step=2: merge (0,2),(4,6),...
    step=4: merge (0,4),...

    CRITICAL: lists[i] = _merge_two(...) — must store the returned head.
    _merge_two may return a different node than lists[i] when l2 < l1.

    Complexity:
        Time:  O(N log k)  — log k rounds, each touching all N nodes once
        Space: O(log k)    — recursion stack of _merge_two
    """
    sz = len(lists)
    step = 1
    while step < sz:
        for i in range(0, sz - step, step * 2):
            lists[i] = _merge_two(lists[i], lists[i + step])  # assignment required
        step *= 2
    return lists[0] if sz else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_list(vals: List[int]) -> Optional[ListNode]:
    dummy = ListNode()
    cur = dummy
    for v in vals:
        cur.next = ListNode(v)
        cur = cur.next
    return dummy.next


def _to_array(head: Optional[ListNode]) -> List[int]:
    result = []
    while head:
        result.append(head.val)
        head = head.next
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all() -> None:
    cases = [
        ([[1, 4, 5], [1, 3, 4], [2, 6]], [1, 1, 2, 3, 4, 4, 5, 6]),
        ([],                              []),
        ([[]],                            []),
        ([[1, 2, 3]],                     [1, 2, 3]),
        ([[], [1]],                       [1]),
        ([[1], [0]],                      [0, 1]),   # l2 head < l1 head — tests bug 1
    ]
    for raw, expected in cases:
        for fn in (merge_k_lists_heap, merge_k_lists):
            inp = [_make_list(l) for l in raw]
            result = _to_array(fn(inp))
            assert result == expected, \
                f"{fn.__name__}({raw}) = {result}, expected {expected}"


if __name__ == "__main__":
    test_all()
    print("All tests passed")
