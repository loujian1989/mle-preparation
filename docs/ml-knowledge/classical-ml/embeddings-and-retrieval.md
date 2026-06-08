# Embeddings & Retrieval — ML Knowledge Q&A

P1: Meta, Pinterest, Roblox, Netflix.

---

## Two-Tower Architecture

### Q: Describe the two-tower model for retrieval. Why is it efficient at serving time?

**Answer (Staff level):**
- **Architecture**: two separate encoder networks.
  - **User tower**: encodes user context (history, demographics, session) into a dense embedding `u ∈ R^d`.
  - **Item tower**: encodes item features (content, metadata) into a dense embedding `v ∈ R^d`.
  - **Similarity score**: `s(u, v) = u · v` (dot product) or `cosine(u, v)`. Simple inner product scoring.
- **Training**: contrastive loss. For each user-item positive pair, sample K in-batch negatives. Loss pushes positive item's embedding closer to user, negatives further away. **In-batch negative sampling** (batch softmax) is the standard: all other items in the batch serve as negatives. Scale with `scale_pos_weight` if needed.
- **Serving efficiency**:
  - Item embeddings are precomputed offline and indexed (FAISS, ScaNN).
  - At query time: compute user embedding (fast, single forward pass) → nearest neighbor search in the item index → retrieve top-K candidates.
  - This is O(d × index_complexity) instead of O(N × model_complexity) per request.
- **Limitation**: no cross-features between user and item. Dot product can't capture interactions like "user prefers action movies" AND "this movie has explosions." That requires a ranking model.

**Company context:** Meta (production retrieval for News Feed, Instagram), Pinterest (home feed retrieval), Roblox, Netflix.

**Common wrong answer:** "I'd score all items with the full ranking model." — Fails latency SLA. Two-tower enables fast retrieval; the ranking model handles the top-K candidates.

---

## Two-Tower Architecture (Deep Dive)

### Why Separate Towers — The Precomputation Insight

The core design choice is **decoupling user from item at inference time**. Consider the alternative — a single joint model:

```
Cross-attention model: f(user_features, item_features) → score

Serving 1B items per user request:
  1B × f(user, item) forward passes
  At 1ms per pass: 1B ms = 11 days per request  ← completely infeasible
```

Two-tower breaks the dependency:

```
User tower: g(user_features) → u          ← computed once per request (~5ms)
Item tower: h(item_features) → v          ← precomputed offline for all items
Score:      u · v                          ← ANN search (~5ms)

Total: ~10ms regardless of catalog size
```

The key constraint this imposes: **the score must decompose as a function of u alone and v alone — no interaction between user and item features inside the model.** The dot product is the only interaction, and it happens outside the network.

---

### What Each Tower Encodes

**User tower** — must capture intent for this specific request, not just who the user is:

```
Long-term signals (stable across sessions):
  - Watch history: [video_id₁, video_id₂, ..., video_idₙ]  → mean-pooled embedding
  - Demographic features: age_bucket, country, device_type
  - Explicit preferences: subscribed channels, liked topics

Short-term signals (this session):
  - Last 5 videos watched in current session
  - Search query (if present)
  - Time of day, day of week
```

The user tower fuses all of this into a single d-dim vector. The challenge: the same user has different intent at 8am (commute, short clips) vs. 10pm (relaxing, long-form). The session context must dominate when present.

**Item tower** — must capture what the item is about, independent of any user:

```
Content signals:
  - Title, description: text embeddings (BERT or trained from scratch)
  - Tags, categories: sparse categorical features
  - Audio/visual features: thumbnail embeddings, audio spectrograms

Engagement signals (aggregated, not user-specific):
  - Watch-through rate: avg % of video watched across all users
  - CTR: click rate from impressions
  - Engagement velocity: how fast it's gaining interactions
```

Engagement signals are aggregate statistics — they don't leak per-user information and can be precomputed. They encode item quality, which helps the model learn that quality items should be in many users' neighborhoods.

---

### The No-Cross-Feature Limitation — Concrete Impact

Because user and item are only joined at the dot product layer, the model cannot learn:

```
"This user watches cooking videos AND this video is a cooking tutorial"
  → requires knowing both simultaneously inside the model
  → not possible in two-tower (u and v computed independently)
```

What two-tower CAN learn:

```
User tower learns: "cooking-interested users have embeddings near [0.3, 0.8, ...]"
Item tower learns: "cooking tutorials have embeddings near [0.3, 0.8, ...]"
Score: high because they're in the same neighborhood
```

The model learns to place users and items in the same neighborhood if the user would engage with the item — but it can only do so based on general affinity, not on specific feature combinations.

**Why this matters in practice:**

```
User: watches action movies + is in France
Item A: French action movie  → user would love it (intersection of both interests)
Item B: American action movie → user likes it (matches one interest)
Item C: French documentary    → user might watch (matches one interest)

Two-tower: probably retrieves B and C, may miss A
           because it can't model "French AND action" jointly
Ranking model (gets top-K from retrieval): can model this interaction
           → reranks A to the top
```

This is the fundamental reason the retrieval-ranking pipeline exists: two-tower handles scale, ranking model handles cross-feature interactions.

---

### Training: What the Model Is Actually Learning

Batch softmax trains the model to answer: *"which item in this batch did this user engage with?"*

```
For user uᵢ with positive item vᵢ in a batch of B=256:

The model must learn:
  u₁ · v₁ > u₁ · v₂, u₁ · v₃, ..., u₁ · v₂₅₆

This means: user 1's embedding must be closer to item 1's embedding
            than to any of the other 255 items' embeddings
```

At convergence, the embedding space self-organizes so that users cluster near the items they engage with:

```
Embedding space visualization:
  [action cluster]  u_action_fan₁, u_action_fan₂, v_avengers, v_terminator
  [cooking cluster] u_chef₁, u_chef₂, v_cooking101, v_pasta_recipe
  [music cluster]   u_musician₁, v_jazz_intro, v_guitar_lesson
```

The contrastive loss is what creates this cluster structure — it pulls positive pairs together and pushes everything else apart.

---

### Feature Engineering Choices

**Handling user history (variable length)**:

```
Option 1: Mean pooling of item embeddings in history
  h = (1/n) Σ v_i  for all items i in history
  Fast. Loses ordering. Works well for general taste.

Option 2: Weighted mean (recent items weighted higher)
  h = Σ wᵢ · vᵢ  where wᵢ = exp(-λ · age_in_days)
  Better for capturing recency.

Option 3: Transformer over history
  h = Transformer([v₁, v₂, ..., vₙ])[:, 0]  ← CLS token
  Captures sequential patterns and item interactions.
  Expensive — but YouTube, Netflix use this for long-horizon modeling.
```

**Handling new users (cold start)**:

```
No history → user tower gets zero or mean-of-all-items embedding
→ retrieves popular items (broadest appeal)
→ as interactions accumulate, embedding personalizes

After 1 interaction:  embedding shifts toward that item's neighborhood
After 10 interactions: reasonable personalization
After 100 interactions: full personalization
```

---

### The Full Serving Pipeline

```
Request arrives: user_id=12345, context={time=9am, device=mobile}

1. User tower inference (~2ms):
   - Fetch user features from feature store (last 50 interactions, demographics)
   - Run user tower forward pass → u ∈ R^256

2. ANN retrieval (~5ms):
   - Query FAISS IVF+PQ index with u
   - Retrieve top-500 candidate items

3. Candidate scoring + ranking (~20ms):
   - Fetch item features for 500 candidates
   - Run full cross-feature ranking model
   - Output: ranked list of 500 items

4. Business logic filters (~1ms):
   - Remove already-watched items
   - Apply content policy filters
   - Inject diversity (no 10 consecutive cooking videos)

5. Return top-20 to client
```

Two-tower retrieval (steps 1–2) takes ~7ms. The ranking model (step 3) takes the bulk. The two-tower exists to reduce the ranking model's input from 1B items to 500 — a 2,000,000× reduction.

---

### Summary

| Concept | Intuition |
|---|---|
| Why separate towers | Enables offline item precomputation; decouples serving cost from catalog size |
| Score decomposition | u·v is the only user-item interaction — forces embeddings to carry all signal |
| No cross-features | Can't model "user likes X AND item is X" jointly — retrieval finds neighborhoods, ranking finds exact matches |
| User tower | Fuses long-term taste + short-term session intent into one vector |
| Item tower | Encodes content + aggregate quality signals, precomputed offline |
| Contrastive training | Creates cluster structure: users near items they engage with |
| Serving pipeline | Two-tower reduces 1B → 500 candidates; ranking model handles the rest |

---

## Contrastive Loss & In-Batch Negative Sampling (Deep Dive)

### The Goal

Train user embedding `u` and item embedding `v` so that:
- `u · v` is **high** for items the user engaged with (positives)
- `u · v` is **low** for items they didn't (negatives)

### The Loss Function (Batch Softmax)

For a batch of B user-item positive pairs `{(u₁,v₁), (u₂,v₂), ..., (uB,vB)}`:

For user `uᵢ`, treat all **other items in the batch** as negatives:

```
Loss(uᵢ) = -log [ exp(uᵢ·vᵢ / τ) / Σⱼ exp(uᵢ·vⱼ / τ) ]
```

- `τ` = temperature (usually 0.05–0.1). Lower τ = sharper distribution = harder task.
- Numerator: score of the positive item.
- Denominator: score of positive + all B-1 negatives.

This is **cross-entropy where the "correct class" is the positive item** out of B candidates.

### Concrete Example (Batch Size = 4)

| | item₁ (Avengers) | item₂ (Inception) | item₃ (Cooking tutorial) | item₄ (Jazz playlist) |
|---|---|---|---|---|
| user₁ (likes action) | ✅ positive | ❌ negative | ❌ negative | ❌ negative |
| user₂ (likes thrillers) | ❌ negative | ✅ positive | ❌ negative | ❌ negative |
| user₃ (likes cooking) | ❌ negative | ❌ negative | ✅ positive | ❌ negative |
| user₄ (likes music) | ❌ negative | ❌ negative | ❌ negative | ✅ positive |

For **user₁**: the model must score Avengers higher than Inception, Cooking tutorial, and Jazz playlist.
You get **3 negatives for free** from the same batch — no extra data needed.
With batch size 256, every user gets **255 negatives per step**.

### Why "In-Batch" Is Efficient

**Naive alternative**: for each positive pair, explicitly sample K random items from the catalog as negatives. Requires K extra forward passes per sample.

**In-batch trick**: the B items already in the batch *are* the negatives. The item tower already computed their embeddings for their own positive pairs — reuse them. Zero extra compute.

### The `scale_pos_weight` / Log-Frequency Correction

**Problem**: popular items appear as negatives much more often than rare items (because popular items are disproportionately sampled into batches as positives for other users). The model learns to push popular items' scores down — **popularity bias**.

**Fix**: correct for sampling frequency. If item `v` appears in the batch with probability `p(v)`, subtract a log-frequency correction:

```
corrected score = uᵢ · vⱼ - log(p(vⱼ))
```

This debiases the loss so the model doesn't unfairly penalize popular items just for being common in batches.

| Concept | What it does |
|---|---|
| Contrastive loss | Trains embeddings to rank positive above negatives |
| In-batch negatives | Reuses other batch items as free negatives — O(B²) pairs from B forward passes |
| Temperature τ | Controls how "hard" the task is; lower = sharper penalties |
| Log-frequency correction | Removes popularity bias from in-batch sampling |

---

## Temperature Parameter — The Math

### Effect on the Distribution

The softmax output assigns probability to each item. Let `sᵢⱼ = uᵢ·vⱼ`:

```
P(j | uᵢ) = exp(sᵢⱼ / τ) / Σₖ exp(sᵢₖ / τ)
```

Two items with scores `s₊ = 0.8` (positive) and `s₋ = 0.6` (hard negative):

| τ | P(positive) | P(hard negative) | Effect |
|---|---|---|---|
| 1.0 | 0.55 | 0.45 | Soft — barely distinguishes them |
| 0.1 | 0.98 | 0.02 | Sharp — confident separation |
| 0.01 | ~1.00 | ~0.00 | Extremely sharp — near one-hot |

For the two-item case this reduces to a sigmoid:

```
P(positive) = 1 / (1 + exp(-(s₊ - s₋) / τ))
```

As τ → 0, the gap `(s₊ - s₋)` gets amplified by `1/τ` → sigmoid saturates to 1.

### Effect on Gradients

Gradient w.r.t. positive score `s₊`:

```
∂Loss/∂s₊ = -(1 - P(positive)) / τ
```

Gradient w.r.t. a negative score `sⱼ`:

```
∂Loss/∂sⱼ = P(j | uᵢ) / τ
```

Both gradients are scaled by `1/τ`:
- **Low τ** → large gradients → strong updates → large margins forced between positive and negatives.
- **High τ** → small gradients → weak updates → model tolerates ambiguous scores.

### Two Failure Modes

**τ too high (e.g. 1.0):** distribution near-uniform → tiny gradients → model barely learns.

**τ too low (e.g. 0.001):** gradients vanish for all but the single hardest negative → training unstable, collapses to trivial solutions.

### Geometric Interpretation

Dividing by τ is equivalent to rescaling the embedding space:

```
uᵢ·vⱼ / τ  =  (uᵢ/√τ) · (vⱼ/√τ)
```

Low τ stretches the space → points that were close together get pulled far apart → model must learn tighter, more separated clusters.

### Production Values

| System | τ |
|---|---|
| SimCLR (vision) | 0.07 |
| Meta EBR / DPR | 0.05 |
| Google YouTube two-tower | 0.05–0.1 |

**Rule of thumb**: start at 0.07, tune on recall@K on a validation set. τ is one of the highest-leverage hyperparameters — a poorly chosen value can hurt recall@10 by 5–10% even with a correct architecture.

---

## Hard Negative Mining

### Q: What is hard negative mining and why is it necessary for two-tower training?

**Answer (Staff level):**
- **Problem with random negatives**: in-batch random negatives (typical at early training) are "easy" — they're clearly different from the positive. The model quickly learns to separate easy negatives, but doesn't learn to distinguish genuinely similar but non-relevant items.
- **Hard negatives**: items that are semantically close to the user's interest but were not engaged with. Example: user interested in Python programming → a JavaScript tutorial is a hard negative (similar domain, not preferred).
- **How to mine hard negatives**:
  1. **In-batch hard negatives**: for each query, rank the other items in the batch by current model score; use the top-ranked non-positives as negatives. Computationally free (no extra forward pass).
  2. **Offline hard negative mining**: periodically retrieve top-K candidates from the index for each user; samples from K+1 to 2K position as hard negatives.
  3. **Semi-hard negatives (margin-based)**: items further from positive than negative but within a margin. Avoids "false hard negatives" (items the user might actually like but hasn't seen).
- **Risk**: "false hard negatives" — items sampled as negatives that the user would have engaged with if shown. This corrupts the training signal. Mitigation: filter by explicit disengagement signals (user blocked, user thumbs-down).

**Company context:** Meta (DPR, FAISS-based retrieval), Pinterest (home feed two-tower), Roblox (game retrieval).

**Common wrong answer:** "I'd use all non-clicked items as negatives." — This includes false hard negatives (items never shown, or shown but not yet interacted with). Must use only observed-negative items.

---

## Hard Negative Mining (Deep Dive)

### The Problem With Random Negatives

With random in-batch negatives, the batch looks like:

| User | Positive | Negatives (random) |
|---|---|---|
| user₁ (likes Python ML content) | PyTorch tutorial | Justin Bieber music, cat videos, cooking recipes |

The model sees scores like:
```
s(u₁, pytorch_tutorial) = 0.82
s(u₁, justin_bieber)    = 0.10
s(u₁, cat_video)        = 0.08
s(u₁, cooking_recipe)   = 0.12
```

Softmax loss here is nearly zero — the model already confidently ranks the positive first. **Gradient ≈ 0. No learning happens.**

This is the **"easy negative" problem**. The model saturates early and stops improving.

### What Makes a Negative "Hard"

A hard negative is an item that is **semantically close to the user's interest but was not engaged with**:

```
user₁ (likes Python ML content)
  Easy negative:   Justin Bieber music     → clearly irrelevant, score = 0.1
  Hard negative:   JavaScript tutorial     → same domain, wrong language, score = 0.75
  Hardest:         PyTorch docs (unseen)   → almost identical, score = 0.80
```

The loss on the hard negative:
```
s(u₁, pytorch_tutorial)    = 0.82  ← positive
s(u₁, javascript_tutorial) = 0.75  ← hard negative

P(positive) = exp(0.82/τ) / (exp(0.82/τ) + exp(0.75/τ))
            ≈ 0.57 at τ=0.1   ← model barely prefers the positive

Loss = -log(0.57) = 0.56   ← large loss → large gradient → real learning
```

Compare to easy negative: Loss ≈ -log(0.99) = 0.01.

### The Three Mining Strategies

**1. In-Batch Hard Negatives (free)**

After each forward pass, rank the B-1 negatives by current model score. Use the top-ranked non-positives:

```
Batch scores for user₁ against all items:
  [pytorch_tutorial=0.82✅, javascript_tutorial=0.75, tensorflow_guide=0.72, cat_video=0.10 ...]

→ Use javascript_tutorial and tensorflow_guide as negatives instead of random ones
```

Cost: zero extra forward passes. Just reorder the existing score matrix.

**2. Offline Hard Negative Mining**

Periodically (e.g. nightly), run ANN retrieval for each user against the full index. Sample negatives from positions K+1 to 2K:

```
Top-K retrieved for user₁:
  Rank 1:       pytorch_tutorial   ← positive (shown, clicked)
  Rank 2:       tensorflow_guide   ← hard negative candidate
  Rank 3:       jax_tutorial       ← hard negative candidate
  ...
  Rank K+1–2K:  sampled as training negatives
```

Why K+1 and not rank 1? Because rank 1–K items might be positives the user hasn't seen yet — see "false hard negatives" below.

**3. Semi-Hard Negatives (margin-based)**

Mine negatives that satisfy:

```
s(u, v⁻) < s(u, v⁺)   AND   s(u, v⁺) - s(u, v⁻) < margin
```

The negative is already ranked below the positive, but only barely. Forces the model to increase the margin without pushing past the point where gradients vanish.

| Strategy | Cost | Quality | When to use |
|---|---|---|---|
| In-batch hard | Free | Medium | Always — baseline |
| Offline mining | Expensive (full retrieval) | High | After model is warm |
| Semi-hard | Medium | Medium-high | Stable training, avoid collapse |

### The False Hard Negative Problem

The most dangerous failure mode:

```
user₁ has watched 50 Python videos.
The catalog has 10,000 Python videos.
user₁ has only seen 50 of them.

If you sample "Python tutorial (unseen)" as a negative:
  → The user would likely engage with it if shown
  → You're training the model to push it away
  → This corrupts the embedding space
```

**Mitigation strategies:**
1. **Only use observed negatives**: items explicitly shown but not clicked (impressions without engagement).
2. **Filter by negative signals**: user blocked, thumbs-down, skip within 2 seconds.
3. **Skip-K buffer**: don't sample from the top-K retrieved items as negatives.
4. **Confidence threshold**: only use items where `s(u, v⁻) > threshold` AND there's an observed non-engagement signal.

### Training Curriculum

Don't start with hard negatives. The model needs to learn basic structure first:

```
Phase 1 (epoch 1–5):   Random in-batch negatives only
                        → model learns coarse structure
Phase 2 (epoch 6–15):  In-batch hard negatives added
                        → model learns fine-grained distinctions
Phase 3 (epoch 16+):   Offline mined hard negatives
                        → model pushed to recall@K ceiling
```

Starting with hard negatives on an untrained model causes training instability — the signal is too noisy before the embeddings are meaningful.

### Summary

| Concept | Intuition |
|---|---|
| Easy negative | Score gap is large → loss ≈ 0 → no gradient → no learning |
| Hard negative | Score gap is small → large loss → large gradient → real learning |
| False hard negative | Item user would like but hasn't seen — corrupts training if used |
| In-batch hard | Free reuse of batch scores; always use |
| Offline mining | Best quality; expensive; use after warm-up |
| Curriculum | Random → in-batch hard → offline mined |

---

## ANN Algorithms

### Q: Compare HNSW, IVF (FAISS), and ScaNN for approximate nearest neighbor search.

**Answer (Staff level):**

| Algorithm | Structure | Recall | QPS | Build time | Memory |
|---|---|---|---|---|---|
| **HNSW** | Navigable small-world graph, hierarchical | Very high (>98% @10) | Very high | Slow | High (stores graph edges) |
| **IVF (Inverted File, FAISS)** | Voronoi cell partitioning + quantization (PQ) | High (configurable) | High | Fast | Low with compression |
| **ScaNN (Google)** | Anisotropic quantization | High (slightly better than FAISS) | High | Medium | Medium |

- **HNSW**: best recall/QPS trade-off for static indexes. Build once, query many times. Poor for frequently updated indexes (insertion is O(log N) per element, expensive at scale).
- **IVF**: better for dynamic indexes. Partitions the space into nlist clusters; query searches nprobe clusters. `nprobe` controls recall-speed trade-off. With Product Quantization (PQ), compresses vectors for memory efficiency at slight recall cost.
- **Production choice**: FAISS IVF+PQ for large-scale systems where memory is constrained and index is rebuilt periodically (nightly). HNSW for smaller indexes where recall matters and index is relatively static.
- **Dimensionality**: 128–256 dimensions is the standard sweet spot. Higher = better recall but slower search. Reduce via PCA or trained projection before indexing.

**Company context:** Meta (FAISS is Meta open-source — built for their use case), Pinterest, Roblox.

**Common wrong answer:** "I'd use exact nearest neighbor search." — Exact search is O(N × d) per query. At 1B items, this is infeasible at p99 <100ms. ANN is the only option at scale.

---

## ANN Algorithms (Deep Dive)

### Why Exact Search Fails at Scale

At 1B items, 256 dimensions, float32:

```
Memory: 1B × 256 × 4 bytes = 1TB  (doesn't fit in RAM)
Compute per query: 1B × 256 multiply-adds = 256B operations
Modern CPU: ~10B float ops/sec (single core)
Latency: 256B / 10B = 25 seconds per query  ← 250× over p99 SLA of 100ms
```

ANN trades a small amount of recall for orders-of-magnitude speedup. At recall@10 = 95%, you're serving users 19/20 correct results — imperceptible quality loss, query in <10ms.

---

### HNSW — How It Works

Intuition: "six degrees of separation." Any two people are connected by ~6 acquaintances. HNSW builds a graph with this property over your item vectors.

**Structure — layered graph:**

```
Layer 2 (few nodes):  A ——————————————— F   (long-range connections, coarse navigation)
Layer 1 (more nodes): A ——— C ——— E ——— F   (medium-range)
Layer 0 (all nodes):  A-B-C-D-E-F-G-H-I-J  (short-range, dense)
```

Each node exists in layer 0. A random subset also exists in layer 1. A smaller subset in layer 2. Higher layers have fewer nodes and longer edges.

**Search for query q:**

```
1. Start at the single entry point in the top layer
2. Greedily navigate: at each node, move to whichever neighbor is closest to q
3. When no neighbor is closer → drop to the next layer, start from current node
4. Repeat until layer 0 → return top-K neighbors found
```

Analogy: Google Maps navigation. Start zoomed out (continent → country), progressively zoom in (city → street). Long hops first, fine-grained at the end.

**Why recall is very high**: layer 0 is densely connected — every node has `efConstruction` neighbors. Once you reach the right neighborhood in the top layers, layer 0 exhaustively searches the local cluster.

**Why build is slow**: inserting a new node requires finding its neighbors at every layer it appears in — each insertion is an ANN search itself: `O(log N)` layers × `O(ef)` search per layer. For 1B items, building from scratch takes hours.

**Why memory is high**: stores the graph edges explicitly. Each node stores ~M=16 neighbor pointers per layer. At 1B nodes: `1B × 16 × 8 bytes ≈ 128GB` just for edges.

---

### IVF (Inverted File Index) — How It Works

Intuition: partition the vector space into neighborhoods, then only search the relevant neighborhoods.

**Build phase — K-Means clustering:**

```
Train K-Means on all N vectors → K centroids (Voronoi cells)
Assign each vector to its nearest centroid
Store: centroid → [list of vectors in this cell]   ← the "inverted file"
```

**Search phase:**

```
Query q arrives
1. Find the nprobe nearest centroids to q  (cheap: compare q to K centroids only)
2. Search all vectors in those nprobe cells (full dot product within cells)
3. Return top-K from the union of results
```

**The `nprobe` knob:**

```
nprobe=1:   search 1 cell  → fastest, misses items near cell boundaries → low recall
nprobe=10:  search 10 cells → 10× slower, catches boundary cases → higher recall
nprobe=K:   search all cells → exact search, slowest
```

Typical: `nprobe = sqrt(K)` as a starting point. Tune by plotting recall@10 vs. QPS.

**Why IVF is better for dynamic indexes than HNSW**: adding a new vector just requires computing its nearest centroid and inserting it into that cell's list — no graph edges to update, no index rebuild.

**Product Quantization (PQ) — memory compression:**

1B × 256-dim float32 vectors = 1TB. PQ compresses this drastically:

```
Split each 256-dim vector into M=8 subvectors of 32 dims each
For each of the 8 subspaces, train K=256 centroids
Encode each subvector as its nearest centroid ID → 1 byte (log₂256 = 8 bits)

Result: 256 floats × 4 bytes = 1024 bytes  →  8 bytes (128× compression)
1TB → ~8GB  ←  fits in memory
```

Distance computation with PQ: precompute distances from the query to all 256 centroids in each subspace (256 × 8 = 2048 lookups), then approximate full distance via table lookups instead of dot products. Fast and memory-efficient at slight recall cost (~1–2% recall drop).

---

### ScaNN — Anisotropic Quantization

Standard PQ minimizes L2 reconstruction error **uniformly** across all directions — it treats all reconstruction errors equally.

**The problem**: for maximum inner product search (MIPS), not all errors are equal. An error in the direction **parallel** to the query vector changes the dot product (affects ranking). An error **perpendicular** to the query has no effect on the dot product at all.

```
Item vector v, query q:

Reconstruction error ε parallel to q:      q·(v+ε) = q·v + q·ε  ← changes ranking
Reconstruction error ε perpendicular to q:  q·(v+ε) = q·v + 0   ← no effect
```

**ScaNN's fix — anisotropic quantization**: weight the quantization error by its impact on the dot product. Penalize errors in the query direction heavily; tolerate errors perpendicular to the query:

```
Standard PQ:  minimize  ||v - v̂||²
ScaNN:        minimize  Σ wᵢ · (vᵢ - v̂ᵢ)²   where wᵢ ∝ importance for ranking
```

Result: better recall at the same compression ratio — the quantized vectors are "wrong" in ways that don't affect ranking, not in ways that do.

---

### Recall-Speed-Memory Triangle

You can only optimize two of the three:

```
High recall + high speed  → needs full vectors in memory (high memory)
High recall + low memory  → needs more cells/graph edges to search (low speed)
Low memory + high speed   → accepts lower recall (fewer vectors scanned)
```

| Algorithm | Optimizes | Sacrifices |
|---|---|---|
| HNSW | Recall + speed | Memory (graph edges) |
| IVF flat | Recall + memory efficiency | Speed (linear scan within cells) |
| IVF+PQ | Speed + memory | Slight recall (quantization error) |
| ScaNN | Recall + memory (vs IVF+PQ) | Build complexity |

---

### Production Decision Guide

| Situation | Choice | Reason |
|---|---|---|
| Static catalog, recall critical, <100M items | HNSW | Best recall/QPS; rebuild cost amortized |
| Dynamic catalog (new items daily), >100M items | FAISS IVF+PQ | Cheap insertion; memory-efficient |
| Memory severely constrained | FAISS IVF+PQ | 128× compression via PQ |
| Google infra / want slightly better recall than PQ | ScaNN | Anisotropic quantization advantage |
| Nightly full rebuild acceptable | FAISS IVF+PQ | Simpler ops; Meta's production default |

**Meta's choice (FAISS)**: catalog is rebuilt nightly with updated item embeddings anyway (model retraining). Full index rebuild is acceptable. IVF+PQ fits the 1B-item catalog in ~8GB. HNSW would require >100GB just for edges.

---

### Summary

| Concept | Intuition |
|---|---|
| Why ANN | Exact search at 1B items = 25s/query; ANN = <10ms at 95% recall |
| HNSW layers | Coarse → fine navigation; long hops at top, dense search at layer 0 |
| HNSW weakness | Build is slow (O(log N) per insert); memory-heavy (explicit graph edges) |
| IVF nprobe | Recall-speed knob: more cells searched = higher recall, lower QPS |
| PQ compression | Split vector into subspaces, encode each as centroid ID → 128× smaller |
| ScaNN insight | Quantization errors perpendicular to query don't affect ranking — tolerate them |

---

## Dot Product vs. Cosine vs. L2

### Q: Which similarity function do you use for ANN, and why does the choice matter?

**Answer (Staff level):**
- **Dot product (inner product)**: `u · v = ||u|| · ||v|| · cos(θ)`. Captures both magnitude and angle. Used when magnitude is informative (popular items have higher-magnitude embeddings naturally).
- **Cosine similarity**: `u · v / (||u|| · ||v||)`. Pure angle, magnitude-normalized. Equivalent to dot product on L2-normalized embeddings. Used when magnitude should not affect ranking (all items should have equal "base score").
- **L2 distance** (Euclidean): `||u - v||`. Equivalent to cosine for normalized vectors. Used in K-Means clustering, less common for retrieval.
- **Why the choice matters**:
  - For recommendation: dot product allows popular items to be more easily retrieved (their embeddings often have higher magnitude due to more training signal). This can help cold-start popular items or hurt by over-fetching popular.
  - Cosine treats all items equally regardless of popularity. Fairer for long-tail items.
  - **Common practice**: L2-normalize item embeddings, use inner product search. This collapses dot product to cosine but allows fast FAISS inner product index (faster than L2 search).

**Company context:** Meta, Pinterest, Roblox.

**Common wrong answer:** "I'd use cosine similarity because it's always better." — Dot product is often preferred in production because it implicitly encodes item quality/popularity, which is usually a valid signal.

---

## Dot Product vs. Cosine vs. L2 (Deep Dive)

### What Each Metric Measures Geometrically

```
Dot product:       u · v = ||u|| · ||v|| · cos(θ)   ← magnitude AND angle
Cosine similarity: u · v / (||u|| · ||v||) = cos(θ) ← angle only
L2 distance:       ||u - v||                         ← Euclidean distance
```

Visualized in 2D:

```
         v₁ (magnitude=5, θ=10°)
        /
       /   ← dot product with u is HIGH (close angle, large magnitude)
      /
u ——→
      \
       \   ← dot product with u is LOWER (same angle as v₂, but v₂ magnitude=1)
        \
         v₂ (magnitude=1, θ=10°)
```

Cosine(u, v₁) = Cosine(u, v₂) — they're the same angle. Dot product(u, v₁) > dot product(u, v₂) — magnitude of v₁ matters.

**Key question**: is item magnitude a meaningful signal? That determines which metric to use.

---

### Why Magnitude Emerges During Training

In a two-tower model trained with batch softmax, items that appear frequently in the training data receive more gradient updates. More updates → larger weight adjustments → naturally larger embedding magnitude.

```
Popular item  (1M interactions): many gradient steps → ||v_popular|| ≈ 3.2
Rare item     (100 interactions): few gradient steps  → ||v_rare||   ≈ 0.8
```

Under dot product search, popular items get a ~4× head start in ranking before angle even factors in. This is not a bug — it encodes a real signal: popular items are broadly appealing and likely relevant to many users.

Under cosine search, both items are treated as equally relevant candidates at baseline. The rare item competes on angle alone.

---

### Concrete Retrieval Comparison

User embedding: `u = [0.6, 0.8]` (normalized, ||u|| = 1.0)

Three candidate items:
```
v₁ = [0.58, 0.81]  popular   → ||v₁|| = 1.0 (normalized)  θ ≈ 1°
v₂ = [0.55, 0.77]  popular   → ||v₂|| = 2.8 (high magnitude) θ ≈ 3°
v₃ = [0.59, 0.80]  long-tail → ||v₃|| = 0.3 (low magnitude) θ ≈ 0.5°
```

Scores under each metric:
```
                    v₁       v₂       v₃
Dot product:       0.999    2.77     0.299   → ranking: v₂ > v₁ > v₃
Cosine:            0.999    0.998    0.999   → ranking: v₁ ≈ v₃ > v₂
```

- Dot product: popular v₂ wins even though v₃ has the closest angle to u
- Cosine: long-tail v₃ competes fairly; popular v₂ slightly penalized for lower angle

Neither is wrong — the choice is a business decision about how much to weight popularity.

---

### The Mathematical Relationship

L2 distance for normalized vectors:

```
||u - v||² = ||u||² + ||v||² - 2(u·v)
           = 1 + 1 - 2(u·v)        (if ||u|| = ||v|| = 1)
           = 2(1 - cos(θ))
```

For L2-normalized vectors: **minimizing L2 distance = maximizing cosine similarity = maximizing dot product**. They're the same ranking. This is why the standard pattern is:

```python
# L2-normalize item embeddings before indexing
item_embs = item_embs / np.linalg.norm(item_embs, axis=1, keepdims=True)
# Build FAISS inner product index (faster than L2 index)
index = faiss.IndexFlatIP(d)
index.add(item_embs)
# At query time: L2-normalize user embedding too
user_emb = user_emb / np.linalg.norm(user_emb)
scores, ids = index.search(user_emb, k=100)
```

You get cosine similarity semantics (magnitude-invariant) but use FAISS's fast inner product implementation. Best of both worlds.

---

### When to Use Each

**Use dot product (unnormalized) when:**
- Popularity is a valid relevance signal (most production recommendation systems)
- You want the model to naturally learn to promote broadly-appealing content
- Items have very different amounts of training signal and you want that reflected

**Use cosine (or normalized dot product) when:**
- Long-tail fairness matters — every item should compete on angle alone
- You're doing semantic search where relevance = topic match, not popularity
- Items have been trained with equal exposure (e.g., content-based embeddings with no collaborative signal)
- You're comparing embeddings from different models or modalities

**L2 distance:** rarely used for retrieval. Common in clustering (K-Means) and anomaly detection (distance from cluster centroid). For normalized vectors, identical to cosine — no practical distinction.

---

### The Popularity Bias Trade-off

Dot product implicitly promotes popular items. Whether this is good or bad depends on the use case:

| Use case | Dot product behavior | Good or bad? |
|---|---|---|
| New feed ranking | Popular posts surfaced more | Good — popular = broadly relevant |
| Music discovery | Popular songs dominate | Bad — defeats the purpose of discovery |
| E-commerce search | Popular products ranked higher | Neutral — depends on query intent |
| Long-tail creator support | Popular creators over-indexed | Bad — suppresses niche creators |

Production systems often apply an **explicit popularity correction** on top of dot product retrieval: `score = u·v - α·log(popularity)` to partially offset magnitude bias without fully discarding the signal.

---

### Summary

| Metric | Formula | Captures | Use when |
|---|---|---|---|
| Dot product | `u·v` | Magnitude + angle | Popularity is a valid signal; production default |
| Cosine | `u·v / (‖u‖‖v‖)` | Angle only | Long-tail fairness; semantic search |
| L2 | `‖u-v‖` | Euclidean distance | Clustering; equivalent to cosine on normalized vectors |
| Normalized dot product | L2-normalize then `u·v` | Angle only (cosine) + fast FAISS IP index | Best of both: cosine semantics + IP speed |

---

## Embedding Dimensionality

### Q: How do you choose embedding dimensionality for a two-tower model?

**Answer (Staff level):**
- **Tradeoffs**:
  - Higher dimension: richer representation, better recall for complex queries, but slower ANN search and more memory.
  - Lower dimension: faster search, less memory, but information bottleneck may hurt recall.
- **Rule of thumb**: `d ≈ 4 × ⁴√N` where N = number of items (from heuristics in Google's paper). For 10M items: d ≈ 178 → round to 256.
- **Empirical tuning**: train models at 64, 128, 256, 512 and plot offline recall@K. The "elbow" in the recall-dimension curve is the optimal dimension.
- **Important**: the dimension is a joint function of model capacity AND ANN search. A 512-dim model may have better recall than 256-dim, but if ANN search is significantly slower, the serving SLA may not justify it.
- **Production sizes**: Google YouTube DNN: 256 dims. Meta EBR: 256 dims. Most production two-towers land at 128–256.

**Company context:** Meta, Pinterest, Netflix.

**Common wrong answer:** "I'd use 512 because more dimensions = better." — Diminishing returns after a certain dimension; the serving latency cost may exceed the quality gain.

---

## Embedding Dimensionality (Deep Dive)

### Why Dimension Has Diminishing Returns

Think of embedding dimensions as **degrees of freedom** for the model to encode distinctions between items.

```
d=2:   model can only encode 2 independent axes of variation
       (e.g. "action vs. drama" and "old vs. new")
       Two items that differ on a third axis → same embedding → retrieval failure

d=64:  64 axes → can encode genre, mood, pacing, era, language, ...
       Most meaningful item-to-item distinctions are captured

d=256: 256 axes → captures finer-grained nuances
       Marginal gain: distinguishes "fast-paced action thriller" from
       "slow-burn action thriller" — real but small improvement

d=1024: 1024 axes → axes start encoding noise in the training data
        Overfits to idiosyncrasies of the training set
        Recall@10 may actually drop vs. d=256
```

The recall-dimension curve follows an S-shape: slow improvement at low d (too few axes to capture structure), steep rise in the middle, then plateau + noise at high d.

### The ANN Search Cost Penalty

Increasing dimension doesn't just cost more memory — it directly slows ANN search:

**HNSW**: each distance computation during graph traversal is `O(d)`. More hops × more dims = slower query.

**IVF+PQ**: PQ splits the vector into M subspaces of `d/M` dims each. Doubling d with fixed M doubles the subspace size → slower distance table lookups.

**FAISS flat (exact)**: scan cost is exactly `O(N × d)`. Doubling d doubles query time.

```
d=128, 1B items, FAISS flat: 1B × 128 = 128B ops  → ~13ms
d=256, 1B items, FAISS flat: 1B × 256 = 256B ops  → ~26ms
d=512, 1B items, FAISS flat: 1B × 512 = 512B ops  → ~51ms  ← may breach SLA
```

This is why dimension is a **joint optimization between model quality and serving latency** — not a pure quality decision.

### The `d ≈ 4 × ⁴√N` Heuristic — Where It Comes From

From the Johnson-Lindenstrauss lemma: to preserve pairwise distances between N points under random projection, you need at least `O(log N)` dimensions. The `⁴√N` heuristic is a tighter empirical fit:

```
N = 1M items:   d ≈ 4 × ⁴√(10⁶) = 4 × 31.6 ≈ 126 → round to 128
N = 10M items:  d ≈ 4 × ⁴√(10⁷) = 4 × 56.2 ≈ 178 → round to 256
N = 1B items:   d ≈ 4 × ⁴√(10⁹) = 4 × 177  ≈ 178 → 256 (already sufficient)
```

This gives a starting point — always validate empirically. The heuristic assumes a uniform distribution of items; structured catalogs (e.g. mostly English-language movies) may need less.

### Practical Tuning Protocol

```
Step 1: Train models at d ∈ {64, 128, 256, 512} — same architecture, same data
Step 2: For each d, build ANN index and measure:
          - Offline recall@10 on held-out queries
          - p99 query latency under target QPS
Step 3: Plot recall vs. latency per dimension
Step 4: Pick the point on the Pareto frontier that satisfies your SLA
```

**Elbow detection**: recall improvement from 64→128 is usually large. 128→256 is moderate. 256→512 is often small. If 256→512 gives <1% recall gain but 2× latency increase → stay at 256.

### Dimensionality Reduction Before Indexing

If your model outputs 512-dim embeddings but latency requires 128-dim:

**PCA projection**: fit PCA on item embeddings, keep top 128 components. Lossless in the sense of preserving maximum variance, but loses fine-grained directions.

**Trained linear projection**: add a linear layer `W ∈ R^{512×128}` at the end of each tower and train end-to-end with the contrastive loss. The model learns which 128 directions are most discriminative for retrieval — better than PCA because it's task-aware.

```python
# Trained projection in two-tower model
user_emb = user_tower(user_features)        # [B, 512]
user_emb = F.normalize(user_proj(user_emb)) # [B, 128]  ← learned projection

item_emb = item_tower(item_features)        # [B, 512]
item_emb = F.normalize(item_proj(item_emb)) # [B, 128]

scores = user_emb @ item_emb.T              # [B, B]
```

### Summary

| Concept | Intuition |
|---|---|
| Why dimensions help | More axes of variation → finer item distinctions representable |
| Diminishing returns | After ~256, new axes encode noise rather than signal |
| ANN latency cost | Query time scales linearly with d — dimension is a latency parameter too |
| `4 × ⁴√N` heuristic | Empirical fit from JL lemma; use as starting point, validate empirically |
| Elbow method | Plot recall@K vs. d; pick inflection before plateau |
| Trained projection | Better than PCA — learns which dimensions matter for retrieval specifically |

---

## Online Index Update

### Q: How do you handle new items being added to a retrieval index in production?

**Answer (Staff level):**
- **Periodic full rebuild**: rebuild the entire ANN index nightly. Simple, consistent, handles full re-embedding with updated model. Acceptable when new item latency ≤ 24h.
- **Online incremental update**:
  - HNSW supports inserting new embeddings without rebuild. Insert is O(log N).
  - FAISS IVF requires assignment to the nearest Voronoi cluster at insert time — works but cluster imbalance can degrade recall over time.
- **Two-tier serving**:
  - **Tier 1**: large, periodically rebuilt index (offline items).
  - **Tier 2**: small, fast exhaustive search on items added in the last 24h (< 1% of catalog). New items brute-force searched; after 24h they graduate to the main index.
- **Cold start at retrieval level**: new item has no embedding from collaborative signal. Use content-tower embedding only until the item accumulates enough interactions for the collaborative signal to be reliable.

**Company context:** Pinterest (new pins added in real-time), Roblox (new games), Netflix (new titles).

**Common wrong answer:** "I'd rebuild the index hourly." — Hourly rebuilds are expensive for large catalogs and introduce index lag during build time. The two-tier pattern is the standard production solution.

---

## Online Index Update (Deep Dive)

### Why This Problem Is Hard

A retrieval index isn't just a list of vectors — it's a data structure optimized for fast search. HNSW stores graph edges. IVF stores Voronoi cluster assignments. Adding a new item requires updating the structure, not just appending a row.

The core tension:
```
Users expect new items to be retrievable immediately (Pinterest pin, new YouTube video)
  ↕
Index rebuild at 1B items takes hours; you can't rebuild on every insert
```

### Option 1: Periodic Full Rebuild

```
Schedule: nightly at 2am (low traffic window)
Process:  1. Re-embed all items with current model
          2. Build fresh index from scratch
          3. Atomic swap: new index goes live, old index discarded
```

**Why this is the production default**: it's the only option that guarantees index consistency after model retraining. When you retrain the embedding model, ALL item embeddings change — you must rebuild anyway. Nightly rebuild aligns with the retraining cadence.

**Limitation**: new items published at 3am aren't retrievable until the next night's rebuild — up to 23h lag. Acceptable for Netflix (new titles are announced in advance) but not for Pinterest (pins created in real time).

### Option 2: HNSW Online Insert

HNSW supports inserting a new vector without full rebuild:

```
Insert(new_item_embedding):
  1. Assign new node to layers probabilistically (same as during build)
  2. For each layer the node appears in:
       a. Run ANN search to find M nearest neighbors at this layer
       b. Add bidirectional edges: new_node ↔ each neighbor
  Cost: O(log N) ANN searches × O(ef) work per search
```

For N=1B, this is ~30 ANN searches — fast enough for real-time inserts.

**The degradation problem**: HNSW's graph quality depends on the global structure of all nodes when edges were built. Late inserts can't retroactively add edges to nodes that were built earlier. Over time, the graph develops "dead ends" — regions where new items are reachable from their neighbors but the neighbors don't have back-edges to other well-connected nodes.

```
Fresh HNSW (at build time): recall@10 = 98%
After 10% new inserts:       recall@10 = 96%
After 30% new inserts:       recall@10 = 91%  ← noticeable degradation
```

Mitigation: periodic rebuild (weekly or monthly) to restore graph quality, with HNSW inserts filling the gap between rebuilds.

### Option 3: IVF Online Insert

Inserting into IVF is cheaper but has a different failure mode:

```
Insert(new_item_embedding):
  1. Find nearest centroid: O(K) distance computations (K = number of clusters)
  2. Append new vector to that cluster's list
  Cost: O(K) — very cheap
```

**The cluster imbalance problem**: K-Means centroids are fixed at build time. If a new content category emerges (e.g. a viral new format), hundreds of new items all map to the same centroid — that cluster grows unboundedly while others stay small.

```
At build time:  all K=1000 clusters have ~1000 items each
After 6 months: cluster 42 (trending category) has 50,000 items
                nprobe=10 searches 10 clusters
                If cluster 42 is one of them → 50,000-item linear scan → slow
                If cluster 42 isn't searched → 50,000 relevant items missed
```

Mitigation: periodic rebuild with fresh K-Means clustering to re-balance cells.

### Option 4: Two-Tier Serving (Production Standard)

The pattern used at Pinterest, YouTube, Meta:

```
Tier 1 (main index):  all items older than 24h → large ANN index, rebuilt nightly
Tier 2 (new items):   items added in the last 24h → small flat (exact) index

Query time:
  1. ANN search on Tier 1 → top-K₁ candidates
  2. Exhaustive search on Tier 2 → top-K₂ candidates (brute force, but tiny)
  3. Merge and re-rank → final top-K
```

Why Tier 2 can be brute-force: new items are < 1% of the catalog. If the total catalog is 1B items and 24h adds 1M new items (0.1%), brute-force over 1M items at 256 dims is fast:

```
1M × 256 dot products = 256M ops ≈ 0.03ms  ← negligible
```

After 24h, items graduate from Tier 2 to Tier 1 in the next nightly rebuild.

### Cold Start: The Embedding Problem for New Items

A new item has zero interaction history. The collaborative tower (trained on user-item co-engagement) produces a near-random embedding — it has no signal to work with.

```
Item tower inputs:
  Collaborative signal: user IDs who engaged → empty for new items
  Content signal:       title, description, tags, thumbnail → available immediately
```

**Solutions in priority order:**

1. **Content-only embedding at launch**: use the content tower alone until the item has ≥ N interactions (N=100 is typical). Index under this embedding.

2. **Warm start from similar items**: find the K most similar items by content features, average their collaborative embeddings, use as the new item's initial collaborative embedding.

3. **Propagation**: as the first users interact with the new item, incrementally update its embedding using online learning (rare in production — adds serving complexity).

```
t=0 (publish):    embedding = content_tower(title, tags, thumbnail)
t=100 interactions: embedding = α·content_tower() + (1-α)·collab_tower()
t=1000 interactions: embedding = collab_tower()  ← fully collaborative
```

### Failure Mode: Index Serving During Rebuild

Full rebuild takes hours. During this window you need to keep serving:

```
Naive: take index offline during rebuild → retrieval outage
Better: blue-green index swap

  Blue index (current, live):  serving traffic
  Green index (new, building): being rebuilt in background

  When green is ready:
    1. Load green into memory
    2. Atomic pointer swap: route traffic to green
    3. Deallocate blue

  Zero downtime. Memory spike during swap: 2× index memory required.
```

At 1B items with IVF+PQ: each index ≈ 8GB → peak memory during swap ≈ 16GB. Plan capacity accordingly.

### Summary

| Approach | Latency to serve new item | Recall degradation | Cost |
|---|---|---|---|
| Nightly full rebuild | Up to 23h | None (fresh index) | High compute, low ops complexity |
| HNSW online insert | Immediate | Grows with insert volume | Low compute, graph degrades over time |
| IVF online insert | Immediate | Cluster imbalance over time | Very low compute, degrades differently |
| Two-tier (standard) | Immediate (Tier 2) | None | Moderate — dual index, brute-force on Tier 2 |

**Cold start rule**: never index a new item with a purely collaborative embedding. Always fall back to content-tower embedding until sufficient interactions accumulate.
