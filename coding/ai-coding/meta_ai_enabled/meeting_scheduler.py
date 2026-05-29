"""
Meeting Scheduler — Meta AI-Enabled Round (Confirmed Pool #10)
==============================================================

Problem:
    Given multiple attendees with their busy time intervals, find all available
    time slots for a meeting of a given minimum duration within working hours.

    Input:
        schedules: List of attendees, each with a list of [start, end] busy intervals
        duration:  Minimum meeting duration in minutes
        day_start: Working hours start (default 9 * 60 = 540 = 9:00)
        day_end:   Working hours end   (default 17 * 60 = 1020 = 17:00)

    Times are represented as integers (minutes since midnight).

Checkpoint structure (mirrors actual Meta AI-enabled interview):
    Checkpoint 1: Find free slots for a single attendee
    Checkpoint 2: Merge multiple attendees' schedules into combined busy intervals
    Checkpoint 3: Find all free slots across all attendees of >= min_duration

Key insight:
    1. Merge all busy intervals across all attendees into one sorted list
    2. Find gaps between merged intervals that fit within working hours
    3. Any gap >= duration is a valid meeting slot

Complexity:
    Time:  O(N log N) where N = total number of busy intervals (sort dominates)
    Space: O(N)
"""

from typing import List, Tuple


Interval = Tuple[int, int]  # (start_minutes, end_minutes)

WORK_START = 9 * 60   # 9:00 AM in minutes
WORK_END = 17 * 60    # 5:00 PM in minutes


def merge_intervals(intervals: List[Interval]) -> List[Interval]:
    """Merge overlapping or adjacent intervals into non-overlapping sorted list.

    Args:
        intervals: List of (start, end) tuples. May be unsorted, may overlap.

    Returns:
        Sorted, non-overlapping list of merged intervals.

    Raises:
        ValueError: If any interval has start > end.

    Complexity:
        Time:  O(N log N)
        Space: O(N)
    """
    if not intervals:
        return []
    for s, e in intervals:
        if s > e:
            raise ValueError(f"Invalid interval: start {s} > end {e}")

    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged: List[Interval] = [sorted_intervals[0]]

    for start, end in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:  # overlapping or adjacent
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def find_free_slots(
    schedules: List[List[Interval]],
    duration: int,
    day_start: int = WORK_START,
    day_end: int = WORK_END,
) -> List[Interval]:
    """Find all available meeting slots across all attendees.

    Args:
        schedules: List of attendee schedules; each schedule is a list of
                   (start, end) busy intervals (in minutes from midnight).
        duration:  Minimum slot duration in minutes.
        day_start: Start of working hours in minutes.
        day_end:   End of working hours in minutes.

    Returns:
        List of available (start, end) slots, each of length >= duration,
        clipped to working hours, in chronological order.

    Raises:
        ValueError: If duration < 1 or day_start >= day_end.

    Complexity:
        Time:  O(N log N) — N = total busy intervals across all attendees
        Space: O(N)
    """
    if duration < 1:
        raise ValueError(f"duration must be >= 1, got {duration}")
    if day_start >= day_end:
        raise ValueError(f"day_start={day_start} must be < day_end={day_end}")

    # Flatten all busy intervals and clip to working hours
    all_busy: List[Interval] = []
    for schedule in schedules:
        for start, end in schedule:
            clipped_start = max(start, day_start)
            clipped_end = min(end, day_end)
            if clipped_start < clipped_end:
                all_busy.append((clipped_start, clipped_end))

    if not all_busy:
        # Everyone is free all day
        free_duration = day_end - day_start
        if free_duration >= duration:
            return [(day_start, day_end)]
        return []

    merged_busy = merge_intervals(all_busy)

    # Find gaps between merged busy intervals
    free_slots: List[Interval] = []
    cursor = day_start

    for busy_start, busy_end in merged_busy:
        if busy_start > cursor:
            gap_start = cursor
            gap_end = busy_start
            if gap_end - gap_start >= duration:
                free_slots.append((gap_start, gap_end))
        cursor = max(cursor, busy_end)

    # Check gap after last busy interval
    if cursor < day_end and day_end - cursor >= duration:
        free_slots.append((cursor, day_end))

    return free_slots


def format_time(minutes: int) -> str:
    """Convert minutes-from-midnight to HH:MM string.

    Args:
        minutes: Minutes since midnight.

    Returns:
        Time string in "HH:MM" format.
    """
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def format_slots(slots: List[Interval]) -> List[str]:
    """Format a list of (start, end) slots as human-readable strings.

    Args:
        slots: List of (start_min, end_min) tuples.

    Returns:
        List of "HH:MM - HH:MM" strings.
    """
    return [f"{format_time(s)} - {format_time(e)}" for s, e in slots]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    # Checkpoint 1: single attendee
    schedule_a = [(9 * 60, 10 * 60), (12 * 60, 13 * 60), (16 * 60, 17 * 60)]
    slots = find_free_slots([schedule_a], duration=30)
    # Free gaps: 10:00-12:00 (120 min), 13:00-16:00 (180 min)
    assert len(slots) == 2, f"Expected 2 slots, got {slots}"
    assert slots[0] == (600, 720), f"First slot wrong: {slots[0]}"   # 10:00-12:00
    assert slots[1] == (780, 960), f"Second slot wrong: {slots[1]}"  # 13:00-16:00

    # Checkpoint 2: two attendees, overlapping busy times
    schedule_a2 = [(9 * 60, 11 * 60), (14 * 60, 16 * 60)]
    schedule_b2 = [(10 * 60, 12 * 60), (15 * 60, 17 * 60)]
    slots2 = find_free_slots([schedule_a2, schedule_b2], duration=30)
    # Combined busy: 9:00-12:00 (merged), 14:00-17:00 (merged)
    # Free: 12:00-14:00 = 120 min
    assert len(slots2) == 1, f"Expected 1 slot, got {format_slots(slots2)}"
    assert slots2[0] == (720, 840), f"Slot wrong: {format_slots(slots2)}"

    # Checkpoint 3: no available slot (everyone busy all day)
    schedule_full = [(9 * 60, 17 * 60)]
    slots3 = find_free_slots([schedule_full], duration=30)
    assert slots3 == [], f"Expected no slots, got {slots3}"

    # Minimum duration filter: slot exists but too short
    schedule_c = [(9 * 60, 16 * 60 + 45)]  # busy until 16:45
    slots4 = find_free_slots([schedule_c], duration=30)
    # Free: 16:45-17:00 = 15 min — too short
    assert slots4 == [], f"Expected no slots (too short), got {format_slots(slots4)}"
    slots5 = find_free_slots([schedule_c], duration=10)
    # Free: 16:45-17:00 = 15 min — fits 10 min minimum
    assert len(slots5) == 1

    # No attendees — entire day is free
    slots6 = find_free_slots([], duration=60)
    assert slots6 == [(WORK_START, WORK_END)]

    # Merge test: overlapping intervals
    merged = merge_intervals([(1, 3), (2, 5), (6, 8), (7, 9)])
    assert merged == [(1, 5), (6, 9)], f"Merge wrong: {merged}"

    print("  meeting_scheduler: all tests passed")
    print("    Example:", format_slots(slots2))


if __name__ == "__main__":
    _test()
