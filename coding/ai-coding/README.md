# AI-Coding Prep — Meta Interview

Two parallel tracks:
- **Track A** `meta_ai_enabled/` — Meta's AI-Enabled Coding round (onsite, 60 min, LLM-assisted CoderPad)
- **Track B** `transformer/` — ML architecture from scratch (FAIR/AI Research roles, 60 min, NumPy)

---

## Track A: Meta AI-Enabled Coding Round

### Format

| Aspect | Detail |
|--------|--------|
| Duration | 60 minutes |
| Tool | CoderPad with embedded LLM assistant (since October 2025) |
| Format | Multi-file Python project with runnable tests |
| Structure | 3–4 checkpoints per problem; must clear >= 3 to pass |
| Evaluated on | Engineering judgment — not prompt quality |

### Checkpoint Flow

1. **Checkpoint 1**: Buggy starter code — debug and fix
2. **Checkpoint 2**: Extend with a new constraint or feature
3. **Checkpoint 3**: Optimize for scale or handle an edge case
4. **Checkpoint 4 (bonus)**: Harder extension, time permitting

### Strategy

**Precise prompts are the signal. Vague prompts are a red flag.**

```
Weak:   "write a maze solver"
Strong: "implement BFS maze solver on a 2D List[List[str]] grid.
         State = (row, col). Return List[Tuple[int,int]] path from 'S' to 'E',
         or None if unreachable. Use iterative deque-based BFS with a visited set."
```

**After AI generates code — always:**
1. Read every line before running. Catch wrong variable names, missing edge cases, off-by-ones.
2. Run tests immediately. Never say "done" without tests passing.
3. Fix small bugs manually — faster than re-prompting; shows engineering judgment.
4. Explain complexity when asked. BFS = O(M*N); bitmask DP = O(U^2) over unique masks.

**Time allocation (~20 min per checkpoint):**
- 2 min: read problem, ask one clarifying question if needed
- 3 min: design approach verbally before prompting
- 8 min: generate + review + fix
- 5 min: run tests, discuss complexity
- 2 min: buffer/transition

**Red flags interviewers watch for:**
- Running AI output without reading it first
- Re-prompting for bugs a 30-second manual fix would solve
- Not running tests before declaring completion
- Saying "it should work" instead of actually running it

---

## Confirmed Problem Pool (12 Problems)

| # | Problem | Core Algorithm | Difficulty | Key Checkpoint |
|---|---------|---------------|------------|----------------|
| 1 | [Maze Solver](meta_ai_enabled/maze_solver.py) | BFS + path reconstruction | Medium | Directional one-way doors |
| 2 | [Max Unique Chars Subset](meta_ai_enabled/max_unique_chars_subset.py) | Bitmask DP | Hard | Scaling: backtrack -> bitmask |
| 3 | [Card Game Sum-to-15](meta_ai_enabled/card_game_15.py) | Combination search + simulation | Medium | Backtracking to ~90% win rate |
| 4 | [Friend Recommendation](meta_ai_enabled/friend_recommendation.py) | Graph BFS | Medium | Rank by mutual friend count |
| 5 | [Compiler Optimization](meta_ai_enabled/compiler_optimization.py) | Cost model parsing | Medium | Ambiguous op costs from interviewer |
| 6 | Card Hand Comparator | Comparison + state machine | Medium | Tie-breaking rules |
| 7 | Delivery Cost Dashboard | Data aggregation | Easy-Med | Grouping + filtering |
| 8 | Expense Rule Engine | Rule-based logic | Medium | Conflict resolution between rules |
| 9 | [Service Dependency Impact](meta_ai_enabled/service_dependency_impact.py) | Graph DFS | Medium-Hard | Cascade blast radius |
| 10 | [Meeting Scheduler](meta_ai_enabled/meeting_scheduler.py) | Interval scheduling | Medium | Multi-attendee conflict detection |
| 11 | [LRU Cache (Progressive)](meta_ai_enabled/lru_cache_progressive.py) | DLL + hashmap | Medium | TTL expiry extension |
| 12 | Crawler Frontier Queue | Priority queue + dedup | Medium | Politeness delay + deduplication |

8/12 problems implemented. Problems 6, 7, 8, 12 are close variants of patterns already covered.

---

## Track B: Transformer Study Path

Study in this order — each file builds on the prior:

| Step | File | Key Concepts |
|------|------|-------------|
| 1 | [layer_normalization.py](transformer/layer_normalization.py) | Feature-wise norm, learnable gamma/beta, backward derivation |
| 2 | [positional_encoding.py](transformer/positional_encoding.py) | Sinusoidal PE, wavelength, learned vs fixed |
| 3 | [scaled_dot_product_attention.py](transformer/scaled_dot_product_attention.py) | Q/K/V, 1/sqrt(d_k) scale, causal mask, stable softmax |
| 4 | [multi_head_attention.py](transformer/multi_head_attention.py) | h parallel heads, split/merge, output projection W_O |
| 5 | [transformer_encoder_block.py](transformer/transformer_encoder_block.py) | Pre-LN residual, MHA + FFN + dropout |
| 6 | [transformer_decoder_block.py](transformer/transformer_decoder_block.py) | Masked self-attn + cross-attn + FFN |
| 7 | [adam_optimizer.py](transformer/adam_optimizer.py) | m/v moments, bias correction, epsilon stability |
| 8 | [beam_search.py](transformer/beam_search.py) | Greedy vs beam, length penalty, temperature sampling |

**Validation rule**: Every transformer file overfits a small toy batch to loss < 0.01 in `_test()`.
If you cannot overfit, the forward pass is wrong — do not move on.

---

## ML Debugging Cheatsheet

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| Loss is NaN from epoch 1 | log(0) in cross-entropy | `np.clip(p, 1e-12, 1.0)` or log-sum-exp |
| Loss completely flat | LR too low, dead ReLU, bad init | LR * 10; He init; print grad magnitudes |
| Loss explodes (>100) | LR too high, no gradient clipping | LR / 10; `np.clip(grad, -5, 5)` |
| All predictions same class | Vanishing grads or dead neurons | Check backward; reduce depth; He init |
| Attention weights all uniform | Missing 1/sqrt(d_k) scaling | Divide QK^T by `sqrt(d_k)` before softmax |
| Train loss down, val loss flat | Overfitting | dropout(0.1), weight_decay=1e-4 |
| Gradient check fails | Wrong backward derivation | Check chain rule step by step; print shapes |

---

## Priority Matrix

| Track | 1 Week | 3 Weeks |
|-------|--------|---------|
| AI-Enabled | Problems 1–4 (maze, bitmask, card, friends) | All 12 confirmed problems |
| Transformer | Steps 1–3 (LayerNorm + PE + Attention) | Full encoder + decoder + beam search |
| Classic Coding | Meta tag: graphs, trees, DP | NeetCode 150 Meta-filtered list |
| ML System Design | News Feed ranking end-to-end | PYMK + Ads + Reels (3 full designs) |

---

## Running Files

```bash
# From repo root — stdlib + numpy only, no venv needed
python coding/ai-coding/meta_ai_enabled/maze_solver.py
python coding/ai-coding/meta_ai_enabled/max_unique_chars_subset.py
python coding/ai-coding/meta_ai_enabled/lru_cache_progressive.py
python coding/ai-coding/transformer/layer_normalization.py
python coding/ai-coding/transformer/scaled_dot_product_attention.py
python coding/ai-coding/transformer/multi_head_attention.py

# Adam optimizer (PyTorch variant requires venv)
cd coding && source .venv/bin/activate
python ai-coding/transformer/adam_optimizer.py
```
