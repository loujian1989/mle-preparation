# Coding Interview Prep

Staff/L6 bar across two tracks: LeetCode-style algorithmic + practical ML coding.

---

## Directory Structure

```
coding/
├── system-design-coding/   ← P0: universal, asked by every company
├── leetcode/
│   ├── graphs/             ← P0
│   ├── dynamic-programming/← P0
│   ├── sliding-window/     ← P0
│   ├── heaps/              ← P0
│   ├── hash-maps/          ← P0
│   ├── binary-search/      ← P1
│   └── trees/              ← P1
└── ml-coding/
    ├── fundamentals/       ← OpenAI bar: backprop, BN, focal loss
    ├── applied/            ← Stripe/Reddit bar: dataset → model
    └── take-home/          ← Shopify bar: FastAPI service
```

---

## Topic Priority Matrix

| Topic | Priority | Companies |
|---|---|---|
| Graphs (BFS, DFS, topological sort, Dijkstra) | **P0** | Meta, Uber, Whatnot, Pinterest |
| Dynamic Programming (1D, 2D, knapsack, LIS) | **P0** | Meta, Netflix, Whatnot |
| Sliding Window / Two Pointers | **P0** | All |
| Heaps / Priority Queues | **P0** | Meta, Uber, Roblox (real-time) |
| Hash Maps / LFU/LRU | **P0** | All |
| Binary Search (on value, rotated array) | P1 | Pinterest, Uber |
| Trees (LCA, path sum, serialization) | P1 | Meta, Pinterest |
| System design coding (LRU, rate limiter, task scheduler) | **P0** | All |
| ML coding (backprop, focal loss, BN) | **P0** | OpenAI |
| Applied ML (dataset → model, metrics, calibration) | **P0** | Stripe, Reddit, Shopify |

---

## System Design Coding (P0 — Universal)

| File | Problem | Key Pattern |
|---|---|---|
| `system-design-coding/lru-cache.py` | LRU Cache | OrderedDict + DLL+HashMap variants |
| `system-design-coding/rate-limiter.py` | Rate Limiter | Token bucket + sliding window + fixed window |
| `system-design-coding/task-scheduler.py` | Task Scheduler | Min-heap + LeetCode 621 formula |

---

## LeetCode Problems by Topic

### Graphs (P0)

| File | Problem | Difficulty | Key Technique |
|---|---|---|---|
| `leetcode/graphs/number_of_islands.py` | Number of Islands (LC 200) | Medium | BFS flood-fill |
| `leetcode/graphs/course_schedule.py` | Course Schedule I+II (LC 207, 210) | Medium | Kahn's topological sort |
| `leetcode/graphs/word_ladder.py` | Word Ladder (LC 127) | Hard | BFS + pattern map + bidirectional BFS |
| `leetcode/graphs/clone_graph.py` | Clone Graph (LC 133) | Medium | BFS + visited map |
| `leetcode/graphs/network_delay_time.py` | Network Delay Time (LC 743) | Medium | Dijkstra's algorithm |

### Dynamic Programming (P0)

| File | Problem | Difficulty | Key Pattern |
|---|---|---|---|
| `leetcode/dynamic-programming/climbing_stairs.py` | Climbing Stairs + Coin Change I+II (LC 70, 322, 518) | Easy/Medium | 1D DP, unbounded knapsack |
| `leetcode/dynamic-programming/longest_common_subsequence.py` | LCS + Edit Distance (LC 1143, 72) | Medium | 2D DP + backtracking |
| `leetcode/dynamic-programming/word_break.py` | Word Break I+II (LC 139, 140) | Medium/Hard | 1D DP + DFS memo |
| `leetcode/dynamic-programming/knapsack.py` | 0/1 Knapsack + Partition Sum + Target Sum (LC 416, 494) | Medium | 0/1 knapsack variants |
| `leetcode/dynamic-programming/longest_increasing_subsequence.py` | LIS (LC 300) | Medium | O(N²) DP + O(N log N) patience sort |

### Sliding Window / Two Pointers (P0)

| File | Problem | Difficulty | Key Technique |
|---|---|---|---|
| `leetcode/sliding-window/longest_substring_without_repeating.py` | Longest Substring No Repeat (LC 3) | Medium | Sliding window + HashMap |
| `leetcode/sliding-window/minimum_window_substring.py` | Minimum Window Substring (LC 76) | Hard | Variable window + formed counter |
| `leetcode/sliding-window/sliding_window_maximum.py` | Sliding Window Max (LC 239) | Hard | Monotonic deque |

### Heaps / Priority Queues (P0)

| File | Problem | Difficulty | Key Technique |
|---|---|---|---|
| `leetcode/heaps/kth_largest_element.py` | Kth Largest + Top-K Frequent (LC 215, 347, 703) | Medium/Hard | Min-heap of size k; online stream |
| `leetcode/heaps/merge_k_sorted_lists.py` | Merge K Sorted Lists (LC 23) | Hard | Min-heap with tie-breaker |
| `leetcode/heaps/find_median_from_data_stream.py` | Median from Stream (LC 295) | Hard | Two heaps (max + min) |

### Hash Maps (P0)

| File | Problem | Difficulty | Key Technique |
|---|---|---|---|
| `leetcode/hash-maps/two_sum.py` | Two Sum + 3Sum + 4Sum (LC 1, 15, 18) | Easy–Medium | O(N) lookup; sort + two-pointer |
| `leetcode/hash-maps/group_anagrams.py` | Group Anagrams (LC 49) | Medium | Sort key vs. freq count key |
| `leetcode/hash-maps/lfu_cache.py` | LFU Cache (LC 460) | Hard | freq_map + key_map + min_freq |

### Binary Search (P1)

| File | Problem | Difficulty | Key Technique |
|---|---|---|---|
| `leetcode/binary-search/search_in_rotated_array.py` | Search Rotated Array (LC 33, 81) | Medium | Which half is sorted |
| `leetcode/binary-search/find_minimum_in_rotated_array.py` | Find Min Rotated (LC 153, 154) | Medium | Compare mid vs. right |
| `leetcode/binary-search/koko_eating_bananas.py` | Koko Eating Bananas (LC 875) | Medium | Binary search on answer space |

### Trees (P1)

| File | Problem | Difficulty | Key Technique |
|---|---|---|---|
| `leetcode/trees/lowest_common_ancestor.py` | LCA Binary Tree + BST (LC 236, 235) | Medium | Post-order DFS; BST property |
| `leetcode/trees/serialize_deserialize_binary_tree.py` | Serialize/Deserialize (LC 297) | Hard | BFS level-order; DFS pre-order |
| `leetcode/trees/binary_tree_max_path_sum.py` | Max Path Sum (LC 124) | Hard | Post-order DFS + global max |

---

## ML Coding

### Fundamentals (OpenAI bar — architecture/training, not abstract algos)

| File | Problem | Key Insight |
|---|---|---|
| `ml-coding/fundamentals/logistic_regression_edge_cases.py` | Logistic Regression from scratch | Separable data → divergence; L2 fixes it |
| `ml-coding/fundamentals/backprop_from_scratch.py` | 2-layer MLP forward + backward | Gradient check vs. numerical; He init |
| `ml-coding/fundamentals/batch_normalization.py` | Batch Norm: train vs. inference | BN fails at batch_size=1; LayerNorm alternative |
| `ml-coding/fundamentals/focal_loss.py` | Focal loss implementation + gradient | Down-weights easy negatives; gradient verified numerically |

### Applied ML (Stripe/Reddit bar — 45–60 min dataset → model)

| File | Company Style | Key Topics |
|---|---|---|
| `ml-coding/applied/fraud_detection.py` | Stripe | Velocity features, GBT, AUPRC, threshold as business decision |
| `ml-coding/applied/content_ranking.py` | Reddit | NDCG@K, spam pre-filter, IPS position bias |
| `ml-coding/applied/churn_prediction.py` | Shopify | Time-based split, leakage detection, Platt calibration |

### Take-Home (Shopify bar — FastAPI service)

| File | Description |
|---|---|
| `ml-coding/take-home/shopify_ml_service_template.py` | FastAPI + Pydantic + SHAP + time-series CV + ModelService DI |

---

## Company → Track Mapping

| Company | Coding Style | Key Topics |
|---|---|---|
| **Meta** | LeetCode medium/hard (hardest bar) | Graphs, DP, arrays |
| **Pinterest** | LeetCode medium (graphs, trees, sliding window, backtracking) | 2 coding rounds |
| **Whatnot** | LeetCode medium–hard (trie, graphs, DP) | Working solution first |
| **Uber** | LeetCode-style + data manipulation | Graph traversal, rolling metrics |
| **Netflix** | Algorithmic (array manipulation, DP, bit ops, linked lists) | Medium to hard |
| **Roblox** | Online assessment (HackerRank) | Standard algorithmic |
| **OpenAI** | ML architecture coding (NOT abstract algos) | `ml-coding/fundamentals/` |
| **Stripe** | Real dataset → model (NOT LeetCode) + debug round | `ml-coding/applied/fraud_detection.py` |
| **Reddit** | Live ML model building from dataset | `ml-coding/applied/content_ranking.py` |
| **Shopify** | CoderPad commerce data + take-home FastAPI | `ml-coding/applied/churn_prediction.py` + take-home |

---

## Code Standards (All Files)

- Type hints on all public function signatures
- Docstrings: Args / Returns / Raises / Complexity blocks
- Named constants instead of magic numbers
- No bare `except:` — specific exception types only
- Tests at bottom of each file: run `python <file>` to verify
- ML files: output shape assertions + `predict_batch` path
