"""
Merge K Sorted Lists (LeetCode 23) — Hard
==========================================

Problem:
    Merge k sorted linked lists and return one sorted list.

Edge cases:
    - k == 0 or all lists empty → return None
    - k == 1 → return that list unchanged
    - Lists of different lengths

Approach — Min-heap:
    Push the head of each non-empty list into a min-heap.
    Pop the minimum node; advance its list and push next node.
    Result: globally sorted merged list.

    Tie-breaking: heap compares (val, tie_index, node) to avoid comparing Node objects.

Alternative: Divide and conquer (merge pairs repeatedly) — O(N log k) same complexity.

--------------------------------------------------------------------------
Python 2 vs Python 3 — why naive (val, node) tuples break in Python 3:

    The heap stores tuples (val, node). When two nodes have equal val,
    the heap must break the tie by comparing the second element — the
    ListNode object.

    Python 2: falls back to comparing objects by memory address (id).
              Always works, arbitrary but stable order. No error.
    Python 3: tries ListNode < ListNode via __lt__. ListNode has no
              __lt__ defined → TypeError: '<' not supported between
              instances of 'ListNode' and 'ListNode'.

    Fix: add a counter as a tiebreaker: (val, counter, node).
    The counter is always unique, so comparison is resolved before
    reaching the node — the node is never compared directly.

--------------------------------------------------------------------------
Common mistake — dead statement in initialization loop:

    for l in lists:
        if l:
            heapq.heappush(heap, (l.val, l))
            l = l.next   # BUG: dead statement
                         # reassigns local var AFTER the push;
                         # the next for-iteration overwrites l anyway.
                         # The heap already holds the correct head node.
                         # Remove this line.

Complexity:
    Time:  O(N log k) where N = total nodes, k = number of lists
    Space: O(k) for the heap
"""

import heapq
from typing import List, Optional


class ListNode:
    """Singly-linked list node."""

    def __init__(self, val: int = 0, next: Optional["ListNode"] = None) -> None:
        self.val = val
        self.next = next


def merge_k_lists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """Merge k sorted linked lists using a min-heap.

    Args:
        lists: List of heads of sorted linked lists.

    Returns:
        Head of merged sorted linked list.

    Complexity:
        Time:  O(N log k)
        Space: O(k)
    """
    dummy = ListNode(0)
    current = dummy
    heap: list = []
    tie = 0  # tie-breaker index to avoid comparing ListNode objects

    # Initialize heap with heads
    for node in lists:
        if node:
            heapq.heappush(heap, (node.val, tie, node))
            tie += 1

    while heap:
        val, _, node = heapq.heappop(heap)
        current.next = node
        current = current.next
        if node.next:
            heapq.heappush(heap, (node.next.val, tie, node.next))
            tie += 1

    return dummy.next


# ---------------------------------------------------------------------------
# Helper: build list from array, convert back
# ---------------------------------------------------------------------------

def _make_list(vals: List[int]) -> Optional[ListNode]:
    dummy = ListNode()
    cur = dummy
    for v in vals:
        cur.next = ListNode(v)
        cur = cur.next
    return dummy.next


def _list_to_array(head: Optional[ListNode]) -> List[int]:
    result = []
    while head:
        result.append(head.val)
        head = head.next
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    lists = [
        _make_list([1, 4, 5]),
        _make_list([1, 3, 4]),
        _make_list([2, 6]),
    ]
    result = _list_to_array(merge_k_lists(lists))
    assert result == [1, 1, 2, 3, 4, 4, 5, 6]

    # Empty lists
    assert merge_k_lists([]) is None
    assert merge_k_lists([None, None]) is None

    # Single list
    result2 = _list_to_array(merge_k_lists([_make_list([1, 2, 3])]))
    assert result2 == [1, 2, 3]

    print("  merge_k_lists: all tests passed")


if __name__ == "__main__":
    _test()
