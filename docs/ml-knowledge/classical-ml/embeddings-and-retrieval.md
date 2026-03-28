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
