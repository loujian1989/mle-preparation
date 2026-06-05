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
