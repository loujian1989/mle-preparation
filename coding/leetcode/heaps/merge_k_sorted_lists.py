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

    Divide and conquer pattern:
        step = 1
        while step < n:
            for i in range(0, n - step, step * 2):
                lists[i] = mergeTwoLists(lists[i], lists[i + step])  # ← assignment critical
            step *= 2
        return lists[0] if n else None

    --------------------------------------------------------------------------
    Common mistake — discarding the return value of mergeTwoLists:

        # WRONG
        self.mergeTwoLists(lists[i], lists[i+step])   # return value thrown away

        # CORRECT
        lists[i] = self.mergeTwoLists(lists[i], lists[i+step])

        mergeTwoLists returns the HEAD of the merged list, which is NOT always
        lists[i]. When l2.val < l1.val, the function returns l2 — a different
        node. Without the assignment, lists[i] still points to the old head,
        which is now somewhere in the middle of the merged result. The next
        round of merging starts from the wrong node and loses the prefix.

        Concrete example: merge [3,4,5] and [1,2].
          l1.val=3 > l2.val=1 → returns l2 (head of [1,2,3,4,5])
          Without assignment: lists[i] still points to Node(3) → lost [1,2].

    --------------------------------------------------------------------------
    Empty list guard — two correct styles:

        # Style A: early return at top
        if not lists:
            return None
        ...
        return lists[0]

        # Style B: guard at return (preferred — one exit point)
        return lists[0] if sz else None

        Both are correct. lists=[[]] (one empty list) is NOT caught by
        `if not lists` because lists has one element; the while loop skips
        (step=1, step<1 is False) and lists[0]=None is returned correctly.

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
