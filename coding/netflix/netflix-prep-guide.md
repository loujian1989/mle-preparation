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

### Bucket 5 — String & Parsing
**Problems:** String to Integer (`myAtoi`), Timer Function, Number Pairs

| Problem | Key insight | Edge cases |
|---|---|---|
| String to Integer | Strip → sign → digits → clamp to INT bounds | Leading zeros, no digits, `+`/`-` only, overflow |
| Timer Function (seconds → human) | Greedy modulo: `months = secs // (30*24*3600)` then recurse | Zero case, singular vs plural labels |
| Number Pairs with op | Try all pairs; apply `+,-,*,/`; check ≈ target (float precision) | Division by zero, target=0 |

**Netflix gotcha:** They always ask about edge cases before you start. List them explicitly: empty string, all whitespace, sign with no digits, INT_MAX/INT_MIN overflow.

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

### Bucket 7 — Recommendation & Deduplication
**Problems:** Viewport dedup, Event dedup, Movie similarity (`findFriends`)

| Problem | Key insight | Time | Space |
|---|---|---|---|
| Viewport dedup | Limit list to K visible; use set to track seen; skip duplicates | O(n) | O(K) |
| Event dedup (10s window) | `{event_id: timestamp}`; on new event, check if already seen within TTL | O(1) | O(events in window) |
| Friends by movie overlap | `{user: set(movies)}`; find pairs with non-empty intersection; top-k by size | O(U² × M) | O(U × M) |
| Weighted random (pickIndex) | Prefix sum array + `bisect_left` on random(0, total) | O(log n) | O(n) |

**RS5 angle:** Viewport dedup = candidate generation dedup in two-tower retrieval. Event dedup = deduplication layer for impression/click tracking pipelines.

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
