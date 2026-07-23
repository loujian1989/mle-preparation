# Netflix Interview Prep Guide
**Target role:** Research Scientist 5, Marketplace - Ads DSE
**Sources:** 66 candidate interview reports (1point3acres) + 21 implemented problems (`netflix_coding.py`)

---

## 1. Interview Pipeline

```
Recruiter Screen (30 min)
    ↓
Technical Phone Screen (45–60 min) — Coding (1 problem, medium-hard)
    ↓
Virtual Onsite (3–5 rounds, same day or split across 2 days)
    ├── Coding           (1 round) — algorithmic, medium-hard
    ├── System Design    (1 round) — Ads Frequency Cap or Billing System
    ├── Data Modeling    (1 round) — Ads demand schema
    ├── Behavioral       (2 rounds) — both with manager/director
    └── Research Deep-Dive (RS5 only) — present + defend past project *(unconfirmed for RS5 — verify with recruiter)*
```

**Timeline:** Result usually within 2 business days of each round.
**Culture signal woven throughout:** Freedom & Responsibility — every round tests whether you can operate autonomously, make judgment calls, and communicate decisions at exec level without needing guardrails.

---

## 2. Coding Patterns

### Bucket 1 — Cache & TTL

**Problems:** `ExpireCache`, `WeightedCache`, LRU variants (`# 4 Timed Caching` in `netflix_coding.py`)

| Problem | Key insight | Time | Space |
|---|---|---|---|
| LRU Cache | `OrderedDict.move_to_end()` + `popitem(last=False)` | O(1) get/put | O(capacity) |
| Expire Cache (TTL) | `SortedDict` by expiry; clean on access or background thread | O(log n) | O(n) |
| Weighted Cache | Evict largest-weight items; `SortedDict` with negative weights as keys | O(log n) | O(capacity) |
| Rate Limiter | Sliding window log: deque of timestamps, pop stale entries | O(1) amortized | O(window) |

**RS5 angle:** This is the online feature store problem. When discussing cache eviction, bridge: "LRU eviction maps to feature staleness risk — stale features silently degrade model quality in prod."

---

#### Cache & TTL — MLE Deep Dive

**Why this matters to you as an MLE:**
You interact with caches constantly — your online feature store (Redis) caches user features so model serving doesn't hit the DB on every prediction; your embedding lookup table caches entity vectors between requests; your model result cache avoids re-running inference for repeated inputs. Netflix is asking you to implement a simplified version of exactly this from scratch.

---

**How Netflix escalates the problem (typical interview arc):**

```
Step 1: "Implement a key-value cache"           → just a dict
Step 2: "Keys should expire after TTL seconds"  → need to track time
Step 3: "Memory is limited — evict when full"   → need eviction policy (LRU)
Step 4: "High concurrent reads?"                → thread-safety discussion
Step 5: "Items have different costs/weights?"   → WeightedCache variant
```

Start at Step 1 and let the interviewer push you forward. Don't over-engineer upfront.

---

**Step 1: Simplest cache**

```python
cache = {}
cache['user_123_features'] = [0.5, 1.2, 3.1]   # set
value = cache.get('user_123_features')           # get
```

Problem: grows forever → memory leak (OOM in prod).

---

**Step 2: TTL (Time-To-Live)**

TTL = "this entry is only valid for N seconds after I store it."
ML analogy: "last clicked item" features go stale after 10 minutes.

Simplest approach — **lazy deletion** (check expiry on read, not proactively):

```python
import time
cache = {}  # key → (value, expiry_timestamp)

def set(key, value, ttl_seconds):
    cache[key] = (value, time.time() + ttl_seconds)

def get(key):
    if key not in cache:
        return None
    value, expiry = cache[key]
    if time.time() > expiry:
        del cache[key]
        return None
    return value
```

Problem: stale keys accumulate memory until someone reads them. Under low traffic, expired keys pile up → OOM.

---

**Step 3: LRU eviction — the key tool is `OrderedDict`**

When memory is full, evict the **Least Recently Used** entry.
ML analogy: keep hot user embeddings in cache; drop cold ones.

Python's `OrderedDict` maintains insertion order AND lets you reorder in O(1):

```python
from collections import OrderedDict
lru = OrderedDict()

lru['user_001'] = features_a   # inserted first
lru['user_099'] = features_b
lru['user_042'] = features_c   # inserted last

# Someone accesses user_001 → promote it to "most recent"
lru.move_to_end('user_001')    # O(1)

# Cache is full → evict least recently used (front of dict)
evicted_key, evicted_val = lru.popitem(last=False)  # pops user_099
```

Mental model — think of OrderedDict as a queue where you can jump any item to the back:
```
front [least recent] ←————————————→ [most recent] back
       'user_099'      'user_042'     'user_001'
           ↑
      evict this one if full
```

---

**Step 4: Combining TTL + LRU — how `ExpireCache` works**

This is exactly what `netflix_coding.py` implements. Two data structures work together:

```python
self.key_to_values = OrderedDict()   # key → (value, expiry_ts)   ← for LRU order
self.ttl_to_keys   = SortedDict()    # expiry_ts → set(keys)       ← for expiry cleanup
```

**Why two structures?**

| Structure | Answers |
|---|---|
| `key_to_values` (OrderedDict) | "Which key was used least recently?" |
| `ttl_to_keys` (SortedDict) | "Which keys have already expired?" |

**What is `SortedDict`?**
Like a regular dict but keys stay in sorted order automatically (Python's equivalent of Java's `TreeMap`). Comes from `sortedcontainers` library.

```python
from sortedcontainers import SortedDict
d = SortedDict()
d[1000] = {'key_a'}   # expires at t=1000
d[2000] = {'key_b'}   # expires at t=2000
d[500]  = {'key_c'}   # expires at t=500
# Iteration order: 500, 1000, 2000  (always sorted)
```

The efficiency win: when sweeping for expired keys, iterate from smallest expiry upward and **break as soon as you hit a non-expired key** — everything after is still valid:

```python
def _clean_expired_keys(self, cur_time):
    for ttl in self.ttl_to_keys.keys():     # smallest → largest
        if ttl < cur_time:
            for key in self.ttl_to_keys[ttl]:
                del self.key_to_values[key]
            del self.ttl_to_keys[ttl]
        else:
            break   # all remaining keys are still valid
```

**`set` walkthrough:**
```python
cache.set('user_features', [0.5, 1.2], ttl=100)
# cur_time = 1000
# 1. Clean expired keys
# 2. If key exists already → remove from both structures (refresh)
# 3. expiry_ts = 1000 + 100 = 1100
# 4. key_to_values['user_features'] = ([0.5, 1.2], 1100)  ← added to END of OrderedDict
# 5. ttl_to_keys[1100].add('user_features')
# 6. If over capacity → _clean_lru(): popitem(last=False) from OrderedDict
```

**`get` walkthrough:**
```python
cache.get('user_features')
# 1. Clean expired keys
# 2. If not found → return None
# 3. key_to_values.move_to_end('user_features')  ← promote to "recently used"
# 4. Return value
```

---

**Step 5: WeightedCache — evict by cost, not recency**

Instead of LRU, evict the **heaviest** item when over budget.
ML analogy: feature store where large embedding vectors cost more memory than scalar features.

```python
self.content       = {}            # key → (value, weight)
self.weight_to_key = SortedDict()  # -weight → set(keys)   ← negative = largest first
```

Trick: store weights as **negative** numbers. SortedDict sorts ascending, so `-weight` puts the heaviest item first in iteration order.

```python
# Evict heaviest item until under budget:
while self.cur_weight > self.max_weight:
    largest_neg_weight, key_set = next(iter(self.weight_to_key.items()))
    # largest_neg_weight = most negative = heaviest item
    key_to_evict = key_set.pop()
    del self.content[key_to_evict]
    self.cur_weight -= (-largest_neg_weight)
```

---

**Step 6: Rate Limiter — same pattern, different semantics**

"Has this user made too many requests in the last N seconds?"
ML analogy: your model serving API rate-limits callers; your feature pipeline rate-limits upstream data sources.

Same `SortedDict` skeleton — tracks request timestamps instead of cache values:

```
On each request:
  1. Clean timestamps older than (now - TTL)   ← same as cleaning expired keys
  2. If count >= capacity → reject
  3. Else → log this request's timestamp → allow
```

---

**Interview cheat sheet for this bucket:**

| Interviewer prompt | Your opening move |
|---|---|
| "Implement a key-value cache" | "I'll start with a dict. Two questions: do entries expire? Is memory bounded?" |
| "Keys should expire" | "Lazy deletion on read is simplest. If memory leaks are a concern, I'll add a `SortedDict` on expiry timestamps and sweep on every write." |
| "Memory is limited" | "I'll add LRU using `OrderedDict` — `move_to_end` on access, `popitem(last=False)` when over capacity." |
| "Thread safety?" | "Wrap `set`/`get` in a lock. For higher throughput, shard the cache into N buckets each with its own lock to reduce contention." |
| "How does this look in a real ML system?" | "This is exactly the online feature store pattern — TTL maps to feature staleness SLA, LRU maps to hot-feature retention. Redis implements this natively; we'd use this in-process pattern when sub-millisecond latency matters and a network hop to Redis is too expensive." |

**The one line that impresses:** *"The risk is OOM from keys set with long TTLs that are never read again. The `SortedDict` on expiry lets us sweep efficiently — O(log n) to find the next expired batch, O(1) per deleted key — without a background thread."*

---

### Bucket 2 — Intervals & Scheduling
**Problems:** `Meeting Rooms I/II/III`, `Budget Cap`

| Problem | Key insight | Time | Space |
|---|---|---|---|
| Can attend all meetings (I) | Sort by start; check `end[i] > start[i+1]` | O(n log n) | O(1) |
| Min meeting rooms (II) | Min-heap of end times; push/pop on overlap | O(n log n) | O(n) |
| Most booked room (III) | Two heaps: available rooms + occupied rooms with (end, room_id) | O(n log k) | O(k) |
| Budget Cap | Binary search on cap value; check total spend ≤ budget | O(n log(max)) | O(1) |

**Watch for:** Meeting Rooms III follow-ups probe your heap mechanics — know how to handle ties on room index.

---

#### Intervals & Scheduling — MLE Deep Dive

**Why this matters to you as an MLE:**
Interval scheduling is the abstraction behind GPU/CPU job scheduling for training runs, experiment slot allocation (only N A/B tests can run concurrently), and ad slot booking (overlapping campaigns competing for the same inventory window).

---

**Meeting Rooms I — can one person attend all meetings?**

```python
# Sort by start time, then check if any meeting ends after the next one starts
sorted_intervals = sorted(intervals, key=lambda x: x[0])
for i in range(1, len(sorted_intervals)):
    if sorted_intervals[i-1][1] > sorted_intervals[i][0]:  # overlap
        return False
return True
# Time: O(n log n)  Space: O(1)
```

Edge cases to state: empty list (return True), single meeting (return True), exact touch `[1,2],[2,3]` is NOT an overlap.

---

**Meeting Rooms II — minimum rooms needed (most common variant)**

Mental model: think of it as a timeline. Each new meeting either reuses a room that just freed up, or needs a new one. Track freed rooms with a **min-heap of end times**.

```python
from queue import PriorityQueue

def minMeetingRooms(intervals):
    pq = PriorityQueue()  # min-heap of end times (= rooms in use)
    room_count = 0

    for start, end in sorted(intervals, key=lambda x: x[0]):
        # free rooms whose meetings ended before this one starts
        while not pq.empty() and pq.queue[0] <= start:
            pq.get()

        pq.put(end)
        room_count = max(room_count, pq.qsize())

    return room_count
# Time: O(n log n)  Space: O(n)
```

Key insight: the heap size at any point = rooms currently occupied = rooms needed right now. The answer is the max heap size observed.

---

**Meeting Rooms III — which room gets the most bookings?**

Harder variant: N rooms exist (numbered 0..N-1). Each meeting must go into the lowest-numbered available room. If no room is free, the meeting waits for the earliest-ending room (room number as tiebreak). Which room ends up with the most meetings?

Two heaps:
- `available`: min-heap of free room IDs
- `occupied`: min-heap of `(end_time, room_id)` for rooms in use

```python
import heapq

def mostBooked(n, meetings):
    available = list(range(n))    # all rooms start free
    heapq.heapify(available)
    occupied = []                 # (end_time, room_id)
    count = [0] * n

    for start, end in sorted(meetings):
        # release rooms that ended before this meeting starts
        while occupied and occupied[0][0] <= start:
            _, room = heapq.heappop(occupied)
            heapq.heappush(available, room)

        if available:
            room = heapq.heappop(available)
            heapq.heappush(occupied, (end, room))
        else:
            # no room free — wait for earliest-ending room
            earliest_end, room = heapq.heappop(occupied)
            heapq.heappush(occupied, (earliest_end + (end - start), room))

        count[room] += 1

    return count.index(max(count))
# Time: O(n log k) where k = N rooms  Space: O(k)
```

Tie on room index is handled automatically because `heapq` breaks ties on the second element (`room_id`).

---

**Budget Cap — how Netflix distributes limited budget across shows**

Problem: you have a list of show budgets and a total cap. Find the per-show cap value such that the total spend equals the budget exactly. (Shows under the cap keep their full budget; shows over the cap are trimmed to the cap.)

Key formula after sorting:
```
If we set cap = C, total spend = sum(min(budget_i, C))
                               = prefix_sum(budgets below C) + C * (count of budgets above C)
```

```python
def find_budget_cap(costs, total_budget):
    sorted_cost = sorted(costs)
    prefix_sum = 0

    for idx in range(len(sorted_cost)):
        # if cap falls between sorted_cost[idx-1] and sorted_cost[idx], solve for it
        remaining = len(sorted_cost) - idx
        cur_cap = (total_budget - prefix_sum) / remaining
        if 0 < cur_cap <= sorted_cost[idx]:
            return cur_cap
        prefix_sum += sorted_cost[idx]

    return sorted_cost[-1]   # budget >= sum of all costs, no cap needed
# Time: O(n log n)  Space: O(1) after sort
```

**RS5 angle:** This is budget pacing. The cap = the bid multiplier that exhausts your daily budget by midnight. Same binary search intuition: find the multiplier where `sum(min(bid_i * multiplier, max_bid))` equals your budget.

---

**Interview cheat sheet:**

| Prompt | Your move |
|---|---|
| "Meeting Rooms II" | "Min-heap of end times — heap size = rooms in use right now. Track the max." |
| "What if N fixed rooms exist?" | "Two heaps: available room IDs + occupied (end_time, room_id). Python heapq handles ties on room_id automatically." |
| "Budget cap" | "Sort, then use prefix sum to find where cap = (remaining_budget) / (remaining_shows) lands inside the sorted array." |

---

### Bucket 3 — Sliding Window
**Problems:** Longest unique substring, consecutive same show, unique pair count

| Problem | Key insight | Time | Space |
|---|---|---|---|
| Longest substring, no repeats | Two pointers + `{char: last_index}` map; jump left pointer | O(n) | O(min(n,charset)) |
| Longest consecutive same show | Extend right; when content changes, reset window | O(n) | O(1) |
| Unique pair count | Bitmask on character sets; XOR two windows to find unique pairs | O(n) | O(n) |

**Template:**
```python
left = 0
seen = {}
best = 0
for right, val in enumerate(arr):
    if val in seen and seen[val] >= left:
        left = seen[val] + 1
    seen[val] = right
    best = max(best, right - left + 1)
```

---

#### Sliding Window — MLE Deep Dive

**Why this matters to you as an MLE:**
Sliding windows are the algorithmic backbone of time-windowed feature engineering: "count of user clicks in last 10 minutes", "CTR over a 7-day rolling window", "error rate in the last 1000 requests". The same pattern appears in event dedup, real-time monitoring, and anomaly detection.

**Core mental model — two pointers, one window:**
```
left                right
  ↓                   ↓
[ a, b, c, d, e, f, g ]
  |←— valid window —→|

Invariant: everything in [left, right] satisfies your constraint.
When right violates the constraint → move left forward until it's fixed.
```

---

**Problem 1 — Longest consecutive same show**

Simple case: find the longest run of the same element.

```python
def longest_consecutive_same_show(shows):
    if not shows: return 0
    low, high, max_len = 0, 0, 0
    cur_show = shows[0]

    while high < len(shows):
        if shows[high] != cur_show:
            cur_show = shows[high]
            low = high          # reset window start
        max_len = max(max_len, high - low + 1)
        high += 1

    return max_len
# Time: O(n)  Space: O(1)
```

No hashmap needed — the window only needs to know "what's the current run's value".

---

**Problem 2 — Longest substring without repeating characters (most common variant)**

```python
def lengthOfLongestSubstring(s):
    window = {}   # char → last seen index
    left = 0
    best = 0

    for right, c in enumerate(s):
        if c in window and window[c] >= left:
            # c is inside our current window — jump left past it
            left = window[c] + 1
        window[c] = right
        best = max(best, right - left + 1)

    return best
# Time: O(n)  Space: O(min(n, charset))
```

Key detail: `window[c] >= left` — you might have seen `c` before but outside the current window, which is fine. Only jump left if the duplicate is *inside* the window.

The implementation in `netflix_coding.py` uses a different approach (iterating left forward to delete), which is O(n) too but less clean. The `window[c] + 1` jump is the standard interview version.

---

**Problem 3 — Count unique string pairs with no shared characters (bitmask trick)**

Netflix-specific: given a list of show titles (strings), count pairs where the two titles share zero common characters.

Brute force: O(n² × m) where m = average string length.

Optimized: represent each string as a **bitmask** over the alphabet (bit i = 1 if character i is present). Two strings have no common chars iff `mask1 & mask2 == 0`.

```python
def _to_mask(s):
    mask = 0
    for c in s:
        mask |= 1 << (ord(c) - ord('a'))
    return mask

def find_unique_pair_count(shows):
    from collections import Counter
    masks = Counter(_to_mask(s) for s in shows)
    result = 0
    mask_list = list(masks.keys())

    for i in range(len(mask_list)):
        for j in range(i + 1, len(mask_list)):
            if mask_list[i] & mask_list[j] == 0:
                result += masks[mask_list[i]] * masks[mask_list[j]]
    return result
# Time: O(n*m + k²) where k = unique masks (≤ 2^26, but practically small)
# Space: O(k)
```

The `Counter` is key: if multiple shows have the same bitmask (same set of chars), multiplying their counts avoids redundant pairwise checks.

**RS5 angle:** Bitmask encoding is the same idea as one-hot feature encoding for categorical tags. "Does this item have any of these content tags in common with that item?" is a bitmask AND at serving time.

---

**Interview cheat sheet:**

| Prompt | Your move |
|---|---|
| "Consecutive same show" | "Two pointers — reset left to right when the element changes. O(n), O(1)." |
| "Longest unique substring" | "`{char: last_index}` map, jump left to `window[c] + 1` on collision. O(n)." |
| "Pairs with no shared chars" | "Bitmask each string — `mask |= 1 << (ord(c) - ord('a'))`. Count pairs where `mask1 & mask2 == 0`." |
| "Follow-up: what if strings are long?" | "Bitmask is O(m) per string to build, O(1) per pair to compare — much better than set intersection O(m)." |

---

### Bucket 4 — Graph / DAG
**Problems:** `DAGProcessor` (critical path), Org tree analysis, `Parallel Courses`

| Problem | Key insight | Time | Space |
|---|---|---|---|
| Critical path in DAG | Topological sort + DP: `finish[v] = max(finish[u] + cost[v]) for u in parents` | O(V+E) | O(V) |
| Org level analysis | BFS level-order; count nodes per level; measure imbalance | O(N) | O(N) |
| Alien dictionary order | Build directed graph from adjacent word pairs; topo sort | O(C) where C=total chars | O(1) (26 nodes) |
| Sequence reconstruction | Check if topo sort is unique (at most 1 node with indegree=0 at any time) | O(V+E) | O(V+E) |

**RS5 angle:** DAG critical path = Metaflow pipeline critical path analysis. If asked about ML platform design, you can map this directly: "I'd model the training DAG with the same critical path algorithm to identify bottlenecks in the feature engineering step."

---

#### Graph / DAG — MLE Deep Dive

**Why this matters to you as an MLE:**
DAGs appear constantly in ML: your training pipeline is a DAG of steps (data ingestion → feature engineering → training → evaluation → deployment), ML workflow orchestrators (Metaflow, Airflow, Kubeflow) schedule tasks via topological order, and model dependency graphs (e.g., a downstream model requires upstream embeddings) must be validated for cycles before deployment.

---

**Core tool: Topological Sort via BFS (Kahn's Algorithm)**

Topological sort = linear ordering of nodes where every edge `u → v` means `u` comes before `v`. Only possible if the graph has no cycles.

**The indegree BFS pattern — memorize this:**

```python
from collections import defaultdict, deque

def topo_sort(n, edges):
    graph = defaultdict(list)
    indegree = defaultdict(int)

    for u, v in edges:          # u must come before v
        graph[u].append(v)
        indegree[v] += 1

    # Start with all nodes that have no prerequisites
    queue = deque(node for node in range(n) if indegree[node] == 0)
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:   # all prerequisites done
                queue.append(neighbor)

    return order if len(order) == n else []  # [] = cycle detected
# Time: O(V + E)  Space: O(V + E)
```

**Cycle detection is free:** if `len(order) < n`, some nodes were never added to the queue (they still had unsatisfied prerequisites = there's a cycle).

---

**Problem 1 — Course Schedule (can you finish all courses?)**

Direct application of the template above. `canFinish` = `len(topo_sort(n, prerequisites)) == n`.

**Problem 2 — Course Schedule II (return the order)**

Same template, return `order` instead of just checking length.

**Problem 3 — Parallel Courses (minimum semesters)**

Each BFS level = one semester (all courses whose prerequisites are done can be taken simultaneously).

```python
def minimumSemesters(n, relations):
    # ... build graph + indegree ...
    semester = 0
    while queue:
        semester += 1
        for _ in range(len(queue)):   # process one full level = one semester
            node = queue.popleft()
            for neighbor in graph[node]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)
    return semester if visited == n else -1
# Time: O(V + E)  Space: O(V + E)
```

---

**Problem 4 — DAG Critical Path (minimum total time to finish all tasks)**

This is the `DAGProcessor` in `netflix_coding.py`. Key insight: tasks can run **in parallel** if they have no dependency between them. The total time = the longest path through the DAG (the critical path).

```python
# max_time[v] = the earliest time at which v can finish
# = max(max_time[u] for u in parents of v) + task_time[v]

def minimum_process_time(task_times, dependencies):
    # ... build graph + indegree ...
    max_time = {t: task_times[t] for t in task_times}   # initialize with own time

    while queue:
        cur = queue.popleft()
        for neighbor in graph[cur]:
            # this task can start only after cur finishes
            max_time[neighbor] = max(max_time[neighbor],
                                     max_time[cur] + task_times[neighbor])
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    return max(max_time.values())
# Time: O(V + E)  Space: O(V)
```

**Visual example:**
```
A(3) → C(2) → E(1)
B(4) → D(1) ↗

A finishes at t=3, C starts → finishes at t=5, E at t=6
B finishes at t=4, D starts → finishes at t=5 (feeds into E but arrives before C's path)
Critical path: A→C→E = 6
```

**RS5 angle:** This is your Metaflow pipeline: "feature_extraction(10min) → model_training(30min) → evaluation(5min)" runs in sequence, but if "feature_extraction" and "data_validation" are independent, they run in parallel. The critical path tells you which step is your bottleneck.

---

**Problem 5 — Alien Dictionary (inferring ordering from word list)**

Given words in alien alphabetical order, determine the character ordering.

```python
# Compare adjacent words: find first differing character → that's an edge
# Then topo sort on those edges

def alienOrder(words):
    graph = defaultdict(set)
    indegree = {c: 0 for word in words for c in word}

    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i+1]
        min_len = min(len(w1), len(w2))
        if len(w1) > len(w2) and w1[:min_len] == w2[:min_len]:
            return ""   # invalid: longer word comes first
        for j in range(min_len):
            if w1[j] != w2[j]:
                if w2[j] not in graph[w1[j]]:
                    graph[w1[j]].add(w2[j])
                    indegree[w2[j]] += 1
                break

    # topo sort on graph
    queue = deque(c for c in indegree if indegree[c] == 0)
    result = []
    while queue:
        c = queue.popleft()
        result.append(c)
        for neighbor in graph[c]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    return "".join(result) if len(result) == len(indegree) else ""
# Time: O(C) total chars  Space: O(1) — at most 26 nodes
```

---

**Interview cheat sheet:**

| Prompt | Your move |
|---|---|
| "Course scheduling" | "Indegree BFS. Cycle = some nodes never reach indegree 0. Process one level per semester for parallel variant." |
| "Minimum time to complete all tasks" | "Critical path via topo sort: `max_time[v] = max(max_time[parents]) + time[v]`. Answer = max across all nodes." |
| "How do you detect a cycle?" | "After BFS, if `visited < total_nodes`, nodes with remaining indegree > 0 are in a cycle." |
| "Alien dictionary" | "Build a directed graph from adjacent word pairs' first differing character. Topo sort = the ordering." |

---

### Bucket 5 — String & Parsing
**Problems:** String to Integer (`myAtoi`), Timer Function, Number Pairs

| Problem | Key insight | Edge cases |
|---|---|---|
| String to Integer | Strip → sign → digits → clamp to INT bounds | Leading zeros, no digits, `+`/`-` only, overflow |
| Timer Function (seconds → human) | Greedy modulo: `months = secs // (30*24*3600)` then recurse | Zero case, singular vs plural labels |
| Number Pairs with op | Try all pairs; apply `+,-,*,/`; check ≈ target (float precision) | Division by zero, target=0 |

**Netflix gotcha:** They always ask about edge cases before you start. List them explicitly: empty string, all whitespace, sign with no digits, INT_MAX/INT_MIN overflow.

---

#### String & Parsing — MLE Deep Dive

**Why this matters to you as an MLE:**
String parsing problems test code quality under ambiguity — same muscle as parsing model config files, deserializing feature payloads, or writing a log parser for monitoring pipelines. Netflix cares more about edge case discipline than algorithmic cleverness here.

---

**Problem 1 — String to Integer (`myAtoi`)**

Not about algorithm, about *completeness*. The pattern:

```python
def myAtoi(s: str) -> int:
    MAX_INT, MIN_INT = 2**31 - 1, -(2**31)
    s = s.strip()
    if not s: return 0

    sign = 1
    i = 0
    if s[0] in ('+', '-'):
        sign = -1 if s[0] == '-' else 1
        i = 1

    result = 0
    while i < len(s):
        digit = ord(s[i]) - ord('0')
        if digit < 0 or digit > 9:
            break                           # stop at first non-digit
        # overflow check BEFORE updating result
        if result > (MAX_INT - digit) // 10:
            return MAX_INT if sign == 1 else MIN_INT
        result = result * 10 + digit
        i += 1

    return sign * result
# Time: O(n)  Space: O(1)
```

**Edge cases to state before writing a single line:**
1. Empty string / all whitespace → `0`
2. `"  +42"` → `42` (leading whitespace + sign)
3. `"-"` or `"+"` only → `0`
4. `"42abc"` → `42` (stop at non-digit)
5. `"abc"` → `0` (no leading digits)
6. `"2147483648"` → `2147483647` (INT_MAX overflow)

The overflow check `result > (MAX_INT - digit) // 10` catches overflow *before* it happens. Don't do `result * 10 + digit > MAX_INT` — that expression itself overflows in many languages (not Python, but state this).

---

**Problem 2 — Timer Function (seconds → human-readable)**

Pattern: define a unit ladder from largest to smallest, then greedily extract each unit.

```python
def timer(seconds: int) -> str:
    units = [
        ('months',  30 * 24 * 3600),
        ('weeks',    7 * 24 * 3600),
        ('days',         24 * 3600),
        ('hours',             3600),
        ('minutes',             60),
        ('seconds',              1),
    ]
    parts = []
    for name, size in units:
        count = seconds // size
        seconds %= size
        if count > 0 or name == 'seconds':   # always show seconds
            parts.append(f'{count} {name}')
    return ', '.join(parts)
# Time: O(1) — fixed 6 units  Space: O(1)
```

**Edge cases:**
- `0` → `"0 seconds"` (the `name == 'seconds'` guard handles this)
- `60` → `"1 minutes"` (singular vs plural: the implementation uses singular — confirm with interviewer)
- The month definition `30 days` is an approximation — say so upfront

**Follow-up (recursive version):** Netflix has asked for both iterative and recursive implementations. Recursive: base case = empty units list, otherwise extract the first unit and recurse on the remainder.

---

**Problem 3 — Number Pairs Matching Target**

Find all pairs `(a, b)` from a list where `a op b == target` for any of `+, -, *, /`.

```python
from collections import Counter

def findPairs(nums, target):
    counts = Counter(nums)
    result = set()

    for a in counts:
        # addition:  a + b = target  →  b = target - a
        b = target - a
        if b in counts and (a != b or counts[b] >= 2):
            result.add(f'{min(a,b)}+{max(a,b)}')   # canonical form

        # subtraction: a - b = target  →  b = a - target
        b = a - target
        if b in counts and (a != b or counts[b] >= 2):
            result.add(f'{a}-{b}')

        # multiplication: a * b = target
        if a != 0 and target % a == 0:
            b = target // a
            if b in counts and (a != b or counts[b] >= 2):
                result.add(f'{min(a,b)}*{max(a,b)}')

        # division: a / b = target  →  b = a / target (integer)
        if target != 0 and a % target == 0:
            b = a // target
            if b != 0 and b in counts and (a != b or counts[b] >= 2):
                result.add(f'{a}/{b}')

    return list(result)
# Time: O(n)  Space: O(n)
```

**Edge cases to state:**
- `target = 0` with multiplication: any `a * 0 = 0` — need special handling
- Division by zero: `b = 0` is not a valid divisor
- `a == b`: can only form a pair if `counts[a] >= 2` (two separate elements)
- Float precision: integer division only here; if floats allowed, use `math.isclose`

**Follow-up (triples):** precompute all valid pairs into `pair_map[result] = [(expr, idx1, idx2)]`, then for each third number check if `target op pair_result` matches. O(n²) pairs, O(n) third numbers.

---

**Interview cheat sheet:**

| Prompt | Your move |
|---|---|
| "String to int" | "List edge cases first. Overflow check: `result > (MAX_INT - digit) // 10` before updating." |
| "Timer function" | "Unit ladder from largest to smallest. Greedy modulo. Always show seconds even if 0." |
| "Number pairs" | "`Counter` for O(1) lookup. Check all 4 ops. Guard: same element needs `count >= 2`. Deduplicate with `set`." |

---

### Bucket 6 — System-Flavored Data Structures
**Problems:** Command Undo with Tags, Tag Co-occurrence, Randomized Set

| Problem | Key insight | Time | Space |
|---|---|---|---|
| Randomized Set (O(1) remove) | Swap-with-last trick: list + two hashmaps (val→idx, idx→val) | O(1) all ops | O(n) |
| Command Undo with Tags | Doubly linked list + hashmap of tag → list of nodes; O(1) undo-by-tag | O(1) execute/undo | O(n) |
| Tag Co-occurrence tracker | `SortedDict` of timestamps per tag pair; `bisect` for range query | O(log n) query | O(n) |

**Command Undo detail:**
```
execute(cmd, tag):  create Node; append to tail of doubly-linked list; hashmap[tag].append(node_pointer)
undo():             pop from tail of linked list; remove its pointer from tag's list
undo(tag):          get last node_pointer from hashmap[tag]; use it to splice node out of linked list in O(1)
                    — O(1) removal is only possible because you hold the pointer, not an index
```

---

#### System-Flavored Data Structures — MLE Deep Dive

**Why this matters to you as an MLE:**
These are "mini-system" problems — each one mimics a real production component (command history, analytics tracker, probabilistic sampler). They test whether you can pick the right data structure for the job rather than just knowing algorithms.

---

**Problem 1 — Randomized Set (O(1) insert / remove / getRandom)**

Challenge: sets support O(1) insert/lookup but not O(1) random access. Lists support O(1) random access but not O(1) remove-by-value.

Solution: **combine a list + two hashmaps**. The list gives O(1) random access; the hashmap gives O(1) lookup. The trick for O(1) removal: **swap the target with the last element, then pop the last**.

```python
import random

class RandomizedSet:
    def __init__(self):
        self.idx_to_val = {}   # index → value
        self.val_to_idx = {}   # value → index

    def insert(self, val):
        if val in self.val_to_idx: return False
        idx = len(self.idx_to_val)
        self.idx_to_val[idx] = val
        self.val_to_idx[val] = idx
        return True

    def remove(self, val):
        if val not in self.val_to_idx: return False
        idx = self.val_to_idx[val]
        last_idx = len(self.idx_to_val) - 1

        if idx != last_idx:
            last_val = self.idx_to_val[last_idx]   # get last element
            self.idx_to_val[idx] = last_val         # move last to gap
            self.val_to_idx[last_val] = idx         # update its index

        del self.idx_to_val[last_idx]
        del self.val_to_idx[val]
        return True

    def getRandom(self):
        idx = random.randint(0, len(self.idx_to_val) - 1)
        return self.idx_to_val[idx]
# Time: O(1) all operations  Space: O(n)
```

**With duplicates (`RandomizedCollection`):** `val_to_idx` maps to a *set* of indices, not a single index. On remove, pop any index from that set; swap with last.

---

**Problem 2 — Command Undo with Tags**

API: `execute(cmd, tags)`, `undo()` (undo most recent), `undo(tag)` (undo most recent cmd with that tag).

**Version 1 — Simple (interview starting point):**
Use a global `history` list and per-tag stacks. Mark nodes as `alive=False` when undone instead of physically removing them (lazy deletion).

```python
class HistoryNode:
    def __init__(self, cmd, tags):
        self.cmd = cmd; self.tags = tags; self.alive = True

class CommandSystem:
    def __init__(self):
        self.history = []
        self.tag_history = defaultdict(list)

    def execute(self, cmd, tags):
        node = HistoryNode(cmd, tags)
        self.history.append(node)
        for tag in tags:
            self.tag_history[tag].append(node)

    def undo(self, tag=None):
        stack = self.history if tag is None else self.tag_history.get(tag, [])
        while stack and not stack[-1].alive:
            stack.pop()   # skip already-undone nodes
        if not stack: return None
        node = stack.pop()
        node.alive = False
        return f'undo {node.cmd}'
```

Problem: `undo()` after `undo(tag)` may encounter dead nodes — the `while` loop cleans them lazily. Worst case O(n) per undo.

**Version 2 — Optimized with doubly linked list (follow-up):**
Store all commands in a global doubly linked list. Per-tag lists also use doubly linked lists. Each node holds **pointers to its counterpart nodes** in the tag lists (and vice versa).

When you `undo(tag)`:
1. Get the tail of `tag_history[tag]` (the tag-list node) — O(1)
2. From that node, follow the pointer to the **global history node** — O(1)
3. Call `eject_from_list()` on both the global node and all its linked tag nodes — O(k) where k = number of tags on the command

```python
# eject_from_list in O(1) — standard doubly linked list splice:
def eject_from_list(self):
    if self.prev: self.prev.next = self.next
    if self.next: self.next.prev = self.prev
    self.prev = self.next = None
```

**The key insight:** you can remove a node from the middle of a linked list in O(1) **only if you hold a direct pointer to the node**. This is what `linked_nodes` stores — it's not a list of values but of node references.

```
Time: execute O(k), undo O(k)   — k = number of tags
Space: O(n * k)
```

---

**Problem 3 — Tag Co-occurrence Tracker**

API: `record_interaction(tags, timestamp)`, `get_co_occurrence_count(tag1, tag2, valid_after)`, `get_most_frequent_pair(tag, valid_after)`.

**Data structures:**

```python
self.tag_pair_history = defaultdict(list)  # (tag1, tag2) → [timestamps, ...]  sorted
self.tag_neighbors = defaultdict(set)      # tag → set of co-occurring tags
```

```python
def record_interaction(self, tags, timestamp):
    for tag1, tag2 in combinations(tags, 2):
        pair = (min(tag1,tag2), max(tag1,tag2))   # canonical order
        self.tag_pair_history[pair].append(timestamp)   # assume timestamps are sorted
        self.tag_neighbors[tag1].add(tag2)
        self.tag_neighbors[tag2].add(tag1)
# Time: O(k²) per call — k tags per interaction

def get_co_occurrence_count(self, tag1, tag2, valid_after):
    pair = (min(tag1,tag2), max(tag1,tag2))
    history = self.tag_pair_history.get(pair)
    if not history: return None
    idx = bisect_left(history, valid_after)   # first index >= valid_after
    return len(history) - idx
# Time: O(log n) — binary search on sorted timestamps
```

`bisect_left(history, valid_after)` returns the index of the first timestamp ≥ `valid_after`. Everything from that index to the end is within the valid window.

**RS5 angle:** This is an item-item co-occurrence matrix with a time filter — exactly how collaborative filtering computes "users who watched A also watched B, within the last 30 days". The `SortedDict` / `bisect` pattern is the same as your real-time feature store's time-windowed aggregation.

---

**Interview cheat sheet:**

| Prompt | Your move |
|---|---|
| "O(1) remove from a set" | "List + two hashmaps. Swap-with-last trick to avoid O(n) removal." |
| "Command undo with tags" | "Start with alive-flag approach. Follow-up: doubly linked lists with cross-pointers for O(k) undo." |
| "Why doubly linked vs singly?" | "Need to splice a node out from the middle — requires access to `prev`. Singly linked = O(n) traversal to find prev." |
| "Tag co-occurrence query" | "`defaultdict(list)` of sorted timestamps per tag pair. `bisect_left` for O(log n) range count." |

---

### Bucket 7 — Recommendation & Deduplication
**Problems:** Viewport dedup, Event dedup, Movie similarity (`findFriends`)

| Problem | Key insight | Time | Space |
|---|---|---|---|
| Viewport dedup | Global set for first-K slots; local set per row beyond K | O(n) | O(K) |
| Event dedup (10s window) | `{event_id: timestamp}`; check if already seen within TTL | O(1) amortized | O(events in window) |
| Friends by movie overlap | Inverted index (movie → users); count pair co-occurrences | O(u·k + pairs) | O(u²) |
| Best group top-K scores | Min-heap of size k per group; sum heap at end | O(g·m·log k) | O(k) |
| Weighted random (pickIndex) | Prefix sum + `bisect_left` on random(1, total) | O(log n) | O(n) |

**RS5 angle:** Viewport dedup = candidate generation dedup in two-tower retrieval. Event dedup = deduplication layer for impression/click tracking pipelines. Weighted random = Thompson sampling / negative sampling for training.

---

#### Recommendation & Deduplication — MLE Deep Dive

**Why this matters to you as an MLE:**
This is the bucket where your ML background is most directly applicable. Deduplication = inference-time candidate pruning. Movie similarity = collaborative filtering. Weighted random = exploration strategies in bandits. These problems look like coding exercises but are actually mini-versions of your daily work.

---

**Problem 1 — Viewport Deduplication**

Netflix homepage: show K globally-visible slots across all rows, dedup within them; beyond K, dedup only within each row.

```python
def dedupe(rows, K):
    global_visited = set()   # tracks the first K unique items across all rows
    result = []

    for row in rows:
        local_visited = set()
        current = []
        for title in row:
            if len(global_visited) < K and title not in global_visited:
                current.append(title)
                global_visited.add(title)
                local_visited.add(title)
            elif len(global_visited) >= K and title not in local_visited:
                current.append(title)
                local_visited.add(title)
        result.append(current)
    return result
# Time: O(total items)  Space: O(K + row_size)
```

**Clarify before coding:** Is K the total count of visible items across all rows, or the number of rows? Does "below the fold" count? Ask explicitly — this changes the implementation.

**RS5 angle:** Post-retrieval candidate dedup. After ANN search returns candidates, you dedup before reranking — same global/local set logic.

---

**Problem 2 — Event Deduplication (10-second window)**

Suppress an event if the same event fired within 10 seconds of its previous occurrence.

```python
from collections import OrderedDict

class EventDedup:
    def __init__(self):
        self.seen = OrderedDict()   # event_name → last_timestamp

    def process(self, name, ts):
        is_new = name not in self.seen or ts - self.seen[name] >= 10
        self.seen[name] = ts
        self.seen.move_to_end(name)

        # cleanup stale events from front (oldest first in OrderedDict)
        for key in list(self.seen.keys()):
            if self.seen[key] < ts - 10:
                del self.seen[key]
            else:
                break
        return name if is_new else None
# Time: O(1) amortized  Space: O(events in window)
```

`OrderedDict` here mirrors the `ExpireCache` pattern: `move_to_end` on update, iterate from front to clean stale. Same muscle.

**RS5 angle:** Impression dedup in ad serving — the same ad impression should only count once if the user sees the same ad twice in a short window (e.g., page refresh).

---

**Problem 3 — Friends by Common Movies**

**Version 1 — exact K-suffix match:**
```python
def findFriends(customerIds, movies, k):
    groups = defaultdict(list)
    for uid, movie_list in enumerate(movies):
        key = '-'.join(sorted(movie_list[-k:]))   # canonical suffix key
        groups[key].append(customerIds[uid])
    return [g for g in groups.values() if len(g) >= 2]
# Time: O(u * k log k)  Space: O(u)
```

**Version 2 — at least M movies in common from last K:**
Build inverted index, then count pair co-occurrences.
```python
def findFriendsWithM(customerIds, movies, k, m):
    movie_to_users = defaultdict(list)
    for uid, movie_list in enumerate(movies):
        for movie in movie_list[-k:]:
            movie_to_users[movie].append(uid)

    pair_count = defaultdict(int)
    for users in movie_to_users.values():
        for i in range(len(users)):
            for j in range(i+1, len(users)):
                pair_count[(users[i], users[j])] += 1

    return [[customerIds[u1], customerIds[u2]]
            for (u1, u2), cnt in pair_count.items() if cnt >= m]
# Time: O(u*k + movies * users_per_movie²)  Space: O(u²)
```

**RS5 angle:** Inverted index (movie → users) is the standard representation for user-item collaborative filtering. The pair co-occurrence count is item-item similarity in disguise.

---

**Problem 4 — Best Group by Top-K Scores**

```python
import heapq

def bestGroupIndex(scores, groups, k):
    best_idx, best_score = -1, -1
    for g_idx, group in enumerate(groups):
        heap = []
        for movie in group:
            heapq.heappush(heap, scores[movie])
            if len(heap) > k:
                heapq.heappop(heap)    # keep only top-k
        total = sum(heap)
        if total > best_score:
            best_score, best_idx = total, g_idx
    return best_idx
# Time: O(g * m * log k)  Space: O(k)
```

Min-heap of size k: push each score; when size exceeds k, pop the smallest (evict the worst). What remains is the top-k.

---

**Problem 5 — Weighted Random (pickIndex)**

```python
import random, bisect

class WeightedRandom:
    def __init__(self, weights):
        self.prefix = []
        total = 0
        for w in weights:
            total += w
            self.prefix.append(total)
        self.total = total

    def pick(self):
        target = random.randint(1, self.total)
        return bisect.bisect_left(self.prefix, target)
# Build: O(n)  Pick: O(log n)  Space: O(n)
```

Mental model: weights `[1, 3, 2]` → number line `|1|333|22|`. Pick a random point in `[1, 6]`; binary search which segment contains it. `bisect_left(prefix, target)` finds the leftmost index where `prefix[i] >= target`.

**RS5 angle:** This is Thompson sampling (sample arm proportional to posterior probability) and negative sampling for recommendation model training (sample non-clicked items proportional to item frequency).

---

**Interview cheat sheet:**

| Prompt | Your move |
|---|---|
| "Viewport dedup" | "Global set for first-K, local set per row beyond-K. Clarify K definition before writing." |
| "Event dedup in time window" | "`OrderedDict` of event → last_ts. `move_to_end` on update, clean stale from front." |
| "Users with common movies" | "Inverted index (movie → users), count pair co-occurrences. Threshold on pair count for ≥ M." |
| "Weighted random pick" | "Prefix sum + `bisect_left` on random int in `[1, total]`. O(n) build, O(log n) pick." |

---

## 3. System Design

### Design 1 — Ads Frequency Cap (most common)
**Problem:** Limit how many times a user sees the same ad in a time window (e.g., max 3 times per 24h).

**Components:**
```
Write path:  Ad served → publish to Kafka → consumer increments Redis counter
Read path:   Before serving → read Redis counter → gate if ≥ cap
```

**Key decisions to state:**
- **Redis key:** `freq:{user_id}:{ad_id}:{window_bucket}` (bucket by hour/day)
- **TTL:** Set TTL = window size on key creation; auto-expires — no cleanup needed
- **Atomicity:** Use `INCR` (atomic) not `GET + SET`; for strict cap use `SET NX` + Lua script
- **Read/write split:** Async write path (Kafka) for throughput; sync read via Redis for latency
- **Strictly 1 cap:** Lua script that atomically checks + increments in one round-trip — avoids distributed lock latency (a lock per impression would blow the SLA)

**Netflix-specific angle:** Tie to subscriber experience — over-capping an ad annoys users and hurts retention. Under-capping wastes advertiser budget. State the tension explicitly.

**Monitoring hook (say this proactively):** "I'd track `p99 read latency`, `cap hit rate per ad`, and `counter drift` (Redis vs ground truth from Kafka lag). Alert if drift > 5%."

---

### Design 2 — Billing System (300M users)
**Problem:** Charge subscribers on their renewal date; handle failures gracefully.

**Components:**
```
Cron job (daily) → picks up users whose renewal_date = today
    → publishes to delay queue (SQS/RocketMQ) with delivery_time = renewal_timestamp
    → billing worker consumes → charges → writes receipt → updates next_renewal_date
```

**Key decisions:**
- **Idempotency:** Each billing attempt has a `billing_id`; guard with DB unique constraint to prevent double charge
- **Retry semantics:** Exponential backoff (1h, 6h, 24h); after 3 failures → grace period → suspension
- **Scale:** 300M users / 30 days = ~10M/day average = ~115/sec avg — but renewal dates cluster (many users renew on the 1st of the month), so design for 10–20x burst: ~1,000–2,000/sec sustained peak; use Redis rate limiter on billing worker pool
- **Failure modes:** Payment processor down → dead-letter queue → alerting → manual review queue

---

### Design 3 — Ad-Break Prediction (ML system design)
**Problem:** Predict the optimal moment to insert an ad break without degrading watch experience.

**ML framing:**
- **Label:** Ad break inserted + user continued watching (positive) vs churned session (negative)
- **Features:** Content type, episode position %, engagement signal (pause/rewind), content segment boundaries (scene changes)
- **Model:** Gradient boosted tree for low-latency inference; ensemble with content-aware model for cold-start
- **Serving:** Inference at stream-start for pre-computed break schedule; update in-session via lightweight signal

**Latency:** <200ms SLA — precompute break schedule at content ingestion time; serve from cache.

**Monitoring (proactively):** "Shadow score new model vs current for 48h; holdout 5% of streams; primary metric is post-ad continuation rate, guardrail is overall session length."

---

## 4. Data Modeling — Ads Demand Schema

**Core hierarchy:**
```
Business Account
    └── Campaign (budget, flight dates, objective)
            └── Ad Group (targeting: geo, demo, interest; bid)
                    └── Ad (creative ref, call-to-action)
                            └── Creative (video asset, thumbnail, copy)
```

**Schema sketch:**
```sql
campaigns(id, account_id, name, total_budget, start_date, end_date, objective)
ad_groups(id, campaign_id, daily_budget, bid_amount, targeting_json)
ads(id, ad_group_id, creative_id, status)
creatives(id, type ENUM(video,image,banner), asset_url, copy, cta)
impressions(id, ad_id, user_id, ts, placement, device)
clicks(id, impression_id, ts)
```

**What to discuss:**
- `targeting_json` vs normalized targeting tables — tradeoff: flexibility vs queryability
- Separate `impressions` table at high write volume — partition by `ts` (daily/hourly)
- Read path for serving: join `ads → ad_groups → campaigns` to enforce budget caps at query time or via pre-computed cache
- Mention intake flow, serving flow, impression tracking flow if the interviewer seems interested in breadth

---

## 5. ML / RS5 Framing Layer

These SDE problems have direct ML analogs — use them to bridge when the interviewer asks follow-ups:

| Coding Problem | ML/RS5 Bridge |
|---|---|
| Expire Cache / TTL | Online feature store: feature staleness = TTL; eviction policy = freshness SLA |
| Ads Frequency Cap | Pacing: frequency cap is a hard constraint; Lagrangian relaxation generalizes to soft budget pacing |
| Budget Cap (binary search) | Bid shading / throttling: binary search on pacing multiplier to hit target spend |
| Tag Co-occurrence | Co-occurrence matrix → item-item collaborative filtering; same `SortedDict` range query pattern |
| DAG Critical Path | Metaflow training DAG optimization; critical path = bottleneck step in feature pipeline |
| Viewport Dedup | Candidate generation dedup in two-tower retrieval; dedup after ANN search before reranking |
| Event Dedup (10s window) | Impression dedup pipeline; same TTL-window pattern for click dedup |
| Weighted Random | Exploration in bandits: weighted sampling for epsilon-greedy or Thompson sampling |
| Meeting Rooms / Scheduling | ML training job deduplication: detect overlapping pipeline runs with the same config before scheduling |

---

## 6. Research Deep-Dive Template (RS5-specific round)

Use this structure for presenting past work. Keep total time to ~10 min, then open for questions.

> **Replace the example below with one of your own projects before rehearsing. Do not walk into the interview with someone else's story.**

```
1. Problem statement (1 sentence)
   EXAMPLE: "We needed to predict Dasher supply at the zone level 30 minutes ahead to pre-position incentives."

2. Why existing approaches failed
   EXAMPLE: "Simple time-series models didn't capture supply-demand coupling — high demand would pull in supply,
    so the baseline overestimated supply shortfalls."

3. Your method + key design decision
   EXAMPLE: "We used a causal graph to model supply response to incentive signals, separating the demand effect.
    The key decision was to treat incentive spend as a do-operator intervention, not a covariate."

4. Tradeoffs you made
   EXAMPLE: "We chose a simpler structural model over a deep learning approach for interpretability — we needed
    to explain predictions to ops teams and justify incentive spend to finance."

5. Deployed impact
   EXAMPLE: "Reduced incentive overspend by 18% while maintaining Dasher fill rate. Served 50K zones,
    real-time inference at <100ms p99."

6. What you'd do differently
   EXAMPLE: "I'd invest earlier in better offline evaluation — our proxy metric (MAPE) didn't fully correlate
    with the incentive efficiency metric we actually cared about in prod."
```

**What Netflix evaluates in this round:**
- Can you defend your methodology under pressure? (They will poke at assumptions)
- Do you know where your model fails? (They value intellectual honesty)
- Can you tie research to business outcome? (Every design decision must trace to a metric)

---

## 7. Behavioral Framework

Top 5 patterns from 66 candidate reports (high-frequency, from report synthesis):

| Theme | What they want to see | Format |
|---|---|---|
| Conflict with manager/coworker | You addressed it directly, not around it; outcome was constructive | STAR: Situation → Task → Action → Result |
| Feedback given/received | Specific feedback with measurable effect; you were receptive, not defensive | SBI: Situation → Behavior → Impact (delivery framework) |
| Prioritization (long vs short-term) | You made a principled call under competing pressures; explained tradeoffs to stakeholders | Decision log format |
| Cross-functional influence | You moved people without authority; used data and framing, not rank | Before/after: what changed in their behavior |
| Freedom & Responsibility | You operated autonomously on a high-stakes decision; how you decided without escalating | Situation → your framework for deciding → outcome |

**Director-level rounds (L5/RS5):** These are NOT behavioral-only. Directors assess whether you communicate technical decisions at exec level. Practice giving a 2-sentence summary of a complex technical decision before explaining the details.

---

## 8. Quick Drill List — Top 10 by Frequency

Ranked by frequency (from report synthesis — not exact counts):

| Priority | Problem | Where in `netflix_coding.py` | Drill focus |
|---|---|---|---|
| 1 | Topological Sort (course scheduling) | `# 16 Topological Sorts` | BFS indegree; handle cycle detection; runtime variants |
| 2 | LRU / Expire Cache with TTL | `# 4 Timed Caching` | TTL cleanup, LRU eviction, thread-safety follow-up |
| 3 | Ads Frequency Cap | System design section above | Redis key design, atomic INCR, strict cap variant |
| 4 | Viewport / Recommendation Dedup | `# 5 Recommendation` | Clarify K, dedupe logic, follow-up: pagination |
| 5 | Meeting Rooms II / III | `# 2 Concurrent Users` | PriorityQueue mechanics; III variant needs two heaps |
| 6 | Command Undo with Tags | `# 17 Command Undo with Tags` | Doubly linked list + hashmap; O(1) undo-by-tag |
| 7 | Sliding Window (unique substring) | `# 15 Sliding Windows` | Jump left pointer variant; bitmask variant |
| 8 | Ads Data Modeling | Data modeling section above | Campaign hierarchy; targeting schema; impression table |
| 9 | Weighted Random (pickIndex) | `# 14 Weighted Cache` (see also pickIndex inline) | Prefix sum + bisect |
| 10 | Latency Percentile Tracker | `netflix.py` experience reports | Concurrent reads, SortedDict, TTL cleanup |

**Drill protocol:** For each problem — implement from scratch in 20 min → state time/space complexity → handle 2 follow-up variants → add one RS5 bridge sentence.

---

## 9. Day-Before Checklist

- [ ] Re-read the JD: ad tech (targeting, ranking, bidding) — have a concrete example for each
- [ ] Run through your research deep-dive out loud (10 min, timed)
- [ ] Review `netflix_coding.py` problems 4, 16, 17 (highest-frequency coding problems from candidate reports)
- [ ] Prepare the monitoring hook for each system design: "Here's how I'd know it's working in prod..."
- [ ] Have one autonomy story ready: a high-stakes decision you made without escalating
- [ ] Know the campaign hierarchy schema cold — draw it without looking
