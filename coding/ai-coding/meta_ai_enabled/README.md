# Meta AI-Enabled Coding — Problem Index

Run any file directly: `python <filename>.py` — inline tests at the bottom.

---

## Study Order (priority-ranked)

| # | File | Core Algorithm | Q4/Q5 Hardest Extension | Status |
|---|------|---------------|--------------------------|--------|
| 1 | [maze_solver.py](maze_solver.py) | BFS | Bitmask keys + bombs + Dijkstra | Complete (Q1–Q5b) |
| 2 | [max_unique_chars_subset.py](max_unique_chars_subset.py) | Backtrack → bitmask DP | State compression DP | Complete |
| 3 | [friend_recommendation.py](friend_recommendation.py) | Graph BFS | Top-K mutual friends | Complete |
| 4 | [lru_cache_progressive.py](lru_cache_progressive.py) | DLL + hashmap | TTL expiry | Complete |
| 5 | [meeting_scheduler.py](meeting_scheduler.py) | Interval scheduling | Multi-attendee conflict | Complete |
| 6 | [service_dependency_impact.py](service_dependency_impact.py) | Graph DFS | Cascade blast radius | Complete |
| 7 | [compiler_optimization.py](compiler_optimization.py) | Cost model | Ambiguous op costs | Complete |
| 8 | [card_game_15.py](card_game_15.py) | Combination search | Strategy to 90% win rate | Complete |

**Bar:** Reach Q4 = Strong Hire signal. Q5 is reach territory.

---

## Algorithm Map (what to have memorized cold)

| Pattern | Problems | Why it appears |
|---------|----------|----------------|
| BFS + visited set | maze, friends, service-dep | Interviewer adds visited bug intentionally |
| State expansion `(r,c)` → `(r,c,bitmask)` | maze Q4, max-unique-chars | Core Q4 unlock — AI will not get this right unprompted |
| `heapq` Dijkstra | maze Q5b | Triggered when cells have cost; BFS gives wrong answer |
| Bitmask DP | max-unique-chars Q3–Q4 | Only way to handle 10k+ words in time |
| DLL + O(1) hashmap | lru_cache | Always asked as progressive — get/put → capacity → TTL |
| Interval `[start, end)` overlap | meeting_scheduler | `a.end > b.start` is the correct overlap condition |

---

## AI Prompt Templates (copy-paste during interview)

### BFS state expansion (Q4)
```
Implement BFS on a 2D grid where state = (row, col, key_bitmask).
Keys are lowercase letters; doors are uppercase. A door at char C
requires bit (ord(C.lower()) - ord('a')) set in key_bitmask.
Picking up key c ORs bit (ord(c) - ord('a')) into bitmask.
visited is a set of (row, col, bitmask) tuples.
Return List[Tuple[int,int]] path or None. Do not change the signature.
```

### Dijkstra (Q5b energy)
```
Implement Dijkstra on a 2D grid. cost_map: Dict[(r,c), float] gives
entry cost; default 1.0. Use heapq with (cumulative_cost, r, c).
Return (total_cost, List[Tuple[int,int]]) or None. Stale-entry guard:
skip if popped cost > dist[node].
```

### Bomb blast radius (Q5a)
```
Write get_affected_area(r, c, rows, cols, radius=2) returning all
(row, col) cells within Chebyshev distance `radius` of (r,c), clamped
to grid bounds. Chebyshev: max(|dr|, |dc|) <= radius.
```

---

## Clarifying questions to ask the interviewer (Q5)

- "Does the bomb trigger on **entry** to the cell, or does the player choose to detonate?"
- "Is wall destruction **persistent** (stays destroyed) or one-shot per traversal?"
- "Can the player carry multiple bombs, or is it one?"
- "For energy/cost: is the cost to **enter** a cell or to **leave** it?"
