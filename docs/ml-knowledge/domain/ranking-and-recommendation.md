# Ranking & Recommendation — ML Knowledge Q&A

P0: Reddit (Staff), Netflix, Meta, Pinterest, Roblox, Whatnot.

---

## Position Bias

### Q: What is position bias in ranking, and how do you correct for it?

**Answer (Staff level):**
- **Position bias**: users are more likely to click on items shown in higher positions, regardless of actual relevance. A CTR model trained on naive click logs learns to score items shown at position 1 highest — a feedback loop.
- **Inverse Propensity Scoring (IPS)**: re-weight each click by 1/P(click | position). Estimated from randomized position experiments (shuffle a fraction of traffic randomly). The IPS-corrected loss de-biases gradient updates.
- **Randomization in serving**: inject a small fraction of randomly ranked slates to measure position propensity P(click | position=k). Typically a 1–5% holdout used for propensity estimation.
- **Unbiased learning-to-rank (Joachims et al.)**: propensity-weighted pairwise loss. Jointly learns propensity + relevance using EM — works without explicit randomization but requires more data.
- **Practical note**: IPS introduces high variance when propensity is small (items in low positions are rarely clicked even when shown). Clip propensity weights (IPS-C) or use self-normalized IPS (SNIPS) for variance reduction.

**Company context:** Reddit (live model building, explicitly probes this), Meta (News Feed), Pinterest (home feed, ads).

**Common wrong answer:** "I'd exclude position from features." — That prevents the model from learning position bias but doesn't debias the training labels. The click data itself is biased — you need IPS or randomization to fix the labels, not just feature exclusion.

---

## Pointwise vs. Pairwise vs. Listwise Ranking

### Q: Compare pointwise, pairwise, and listwise ranking approaches. When do you start with pointwise?

**Answer (Staff level):**

| Approach | Target | Loss | Pros | Cons |
|---|---|---|---|---|
| **Pointwise** | Binary engagement per item | Log-loss or MSE | Simple, scales to any binary classifier | Ignores inter-item order; optimizes per-item CTR, not list quality |
| **Pairwise** | Preference: item A > item B | Hinge loss (RankSVM), log-loss (RankNet) | Directly optimizes relative ordering | Quadratic sample complexity (O(N²) pairs per query) |
| **Listwise** | Full ranking of a slate | LambdaLoss, ListNet, NDCG-optimized | Directly optimizes list-level metric | Complex to implement; needs full query-level batching |

- **Start pointwise** in an interview: it maps to standard binary classification, is easy to explain, and works well as a baseline. The model output (engagement probability) is a valid ranking score even if the loss didn't explicitly optimize rank.
- **Upgrade to LambdaMART / LambdaLoss** when: you have dense per-query label data, and offline NDCG matters more than raw throughput.

**Company context:** Reddit (they expect you to start pointwise and then narrate when you'd upgrade), Netflix, Meta.

**Common wrong answer:** Starting with pairwise or listwise in an interview without justification — interviewers expect you to justify complexity. Pointwise + narrate trade-offs > jumping straight to LambdaMART.

---

## Cold Start

### Q: How do you handle cold start for new users and new items in a recommendation system?

**Answer (Staff level):**
- **New user cold start**:
  - Fallback: popularity-based ranking (global or segment-specific). Return top-K items for the user's demographic/geo/signup-context.
  - Onboarding: collect explicit preferences (interest selection during signup). Use as side features in the retrieval model.
  - Transfer: if user signed in via social (e.g., Reddit account history), use cross-platform signals.
- **New item cold start**:
  - Content-based features only (text, image embeddings) — collaborative filtering embeddings are uninitialized.
  - Exploration budget: inject new items into feeds with low frequency; use Thompson sampling or ε-greedy to balance explore/exploit.
  - Warm-up period: wait for N impressions before relying on collaborative signal.
- **Unified handling**: a two-tower model where the item tower is content-based (works at item cold start) and the user tower is behavior-based. At user cold start, substitute the user tower with a fallback embedding learned from demographic features.

**Company context:** Roblox (anchor question on cold start), Netflix (new title launch), Pinterest (new pin), Whatnot (new seller or new item listing).

**Common wrong answer:** "I'd use collaborative filtering." — CF requires interaction data which doesn't exist at cold start. Must specify the fallback chain explicitly.

---

## Diversity vs. Relevance

### Q: Why does optimizing purely for relevance produce a diversity problem, and how do you fix it?

**Answer (Staff level):**
- **Echo chamber / filter bubble**: a relevance-only ranker returns more of what a user has already engaged with. Over time, this narrows exposure and can reduce retention (users get bored with same-type content).
- **Determinantal Point Processes (DPP)**: penalize similar items in the same slate. Formally, DPP defines a distribution over subsets where the probability is proportional to det(L_S), where L encodes item similarity. Computationally expensive for large N; use greedy approximations.
- **MMR (Maximal Marginal Relevance)**: greedy re-ranking. Iteratively pick the item that maximizes `λ * relevance − (1−λ) * max_similarity_to_already_selected`. Cheap and explainable.
- **Slot-based diversity injection**: reserve K slots in the top-N for items from categories not already represented (rule-based, production-friendly).
- **Metric**: pair-wise dissimilarity in the top-K slate = intra-list diversity (ILD). Report alongside NDCG.

**Company context:** Netflix (serendipity is explicit in their metric set), Roblox, Whatnot (auction fairness), Meta.

**Common wrong answer:** "I'd add a diversity feature to the model." — Adding a diversity feature doesn't enforce diversity at the slate level; it's a single-item signal. Slate-level diversity requires post-ranking re-scoring (MMR, DPP) or slot injection.

---

## Feedback Loops

### Q: Describe a feedback loop in recommendation systems and how to break it.

**Answer (Staff level):**
- **Mechanism**: model recommends popular items → users click them → more clicks on popular items → model reinforces popularity → long-tail items never get exposure → rich-get-richer spiral.
- **Consequences**: reduced catalog coverage, stale recommendations, eventual user disengagement (users find recommendations predictable).
- **Breaks**:
  1. **Exploration**: ε-greedy or Thompson sampling to surface non-popular items with positive probability.
  2. **IPS correction**: weight clicks by inverse of position/popularity propensity.
  3. **Counterfactual logging**: log what would have been shown by a randomized policy, train on that.
  4. **Causal model**: disentangle "item quality" from "exposure probability" in the model. Popularity bias correction as a feature, not implicit signal.
- **Metric signal**: track catalog coverage (% of catalog recommended at least once per month) and tail item CTR as health metrics separate from overall CTR.

**Company context:** Netflix (explicitly mentioned in eng blog), Meta (News Feed has heavy feedback loop history), Roblox.

**Common wrong answer:** "I'd add a freshness feature." — Freshness helps with temporal stale content but doesn't fix popularity feedback loops or long-tail suppression.

---

## Two-Stage Ranking

### Q: Why do production ranking systems use a two-stage (retrieval + ranking) architecture?

**Answer (Staff level):**
- **Scale constraint**: a full ranking model (deep neural net with user × item features) cannot score 1B items per request within latency budget (p99 <100ms).
- **Stage 1 — Retrieval (ANN)**: lightweight model (two-tower) creates user and item embeddings. Approximate nearest neighbor search (FAISS/ScaNN) retrieves top-K (100–10K) candidates in O(log N) or O(1) time.
- **Stage 2 — Ranking**: expensive model (DLRM, cross-feature MLP) scores only the K candidates with full feature interactions. Can use expensive signals (user history cross-features) that retrieval stage omits.
- **Why not just retrieval?** The retrieval model's dot-product scoring cannot capture feature crosses (e.g., user × item affinity interactions) — those require the full ranking model.
- Optional **Stage 3 — Re-ranking**: business logic, diversity injection, policy constraints (e.g., no more than 3 consecutive same-creator posts).

**Company context:** Meta (DLRM is two-stage), Pinterest (ads + home feed), Roblox, Netflix.

**Common wrong answer:** "I'd rank all items with the scoring model." — This fails the latency requirement. Must explain the retrieval/ranking split and why each stage exists.

---

## Collaborative Filtering vs. Content-Based

### Q: When does content-based beat collaborative filtering, and when is the reverse true?

**Answer (Staff level):**

| Scenario | Better Approach | Reason |
|---|---|---|
| New item (no interactions) | **Content-based** | CF embeddings uninitialized |
| New user (no history) | **Content-based** | CF user embedding uninitialized |
| Niche item with few interactions | **Content-based** | CF signal too sparse |
| Dense interaction data | **Collaborative** | Captures latent taste patterns beyond content |
| Cross-domain recommendation | **Collaborative** | Shared interaction structure across domains |
| Content is low-signal (generic) | **Collaborative** | Metadata doesn't differentiate quality |

- **Hybrid**: item tower = content features + CF embedding. At cold start, content features dominate. As item accumulates interactions, CF embedding weight increases. This is the production default.
- CF requires regularization at inference for long-tail items (embeddings from few observations are high-variance).

**Company context:** Roblox, Netflix, Pinterest. Expected to enumerate both approaches and arrive at hybrid.

**Common wrong answer:** "I'd use collaborative filtering because it's more powerful." — CF fails at cold start. Staff answer always includes fallback and hybrid strategies.

---

## Position Bias (Deep Dive)

### Why Naive CTR Training Creates a Feedback Loop

When you train a click model on logged data without correcting for position:

```
Position 1: shown 10,000 times, clicked 800 times → CTR = 8%
Position 5: shown 10,000 times, clicked 200 times → CTR = 2%
```

The model learns: "items at position 1 are 4× more relevant." But these items are at position 1 *because* the current production model put them there. The high CTR is partly relevance, partly position advantage.

**The feedback loop:**
```
Current model ranks item A at position 1 → item A gets high CTR
→ next model trained on this data → item A ranked at position 1 again
→ item B (equally relevant, shown at position 3) never gets a fair chance
→ rich-get-richer: position 1 items accumulate more training signal
```

After 5 retraining cycles, items that were randomly placed in position 1 early on can permanently dominate the top positions regardless of their true relevance.

### IPS — The Math

For each observed click, we want to estimate what the click probability would be regardless of position. IPS re-weights each click by the inverse of how likely it was to be shown at that position:

```
Unbiased loss = Σ_i [click_i / P(shown at position_i)] × loss(score_i, label_i)
```

If `P(click | position=1) = 0.08` and `P(click | position=5) = 0.02`:

```
A click at position 1 is weighted by: 1 / 0.08 = 12.5
A click at position 5 is weighted by: 1 / 0.02 = 50.0
```

A click at position 5 contributes 4× more to the gradient — correcting for the fact that items at position 5 are underexposed.

### IPS-C — Why You Need to Clip

Problem: items shown at position 10 have very low P(shown), so their IPS weight is huge. A single click at a low-ranked position can dominate the entire loss:

```
P(click | position=10) = 0.005
IPS weight = 1/0.005 = 200  ← one click multiplied by 200
```

**IPS-C (clipped IPS)**: cap the weight at some maximum `W_max`:

```
weight_i = min(1/P(position_i), W_max)
```

Trades variance reduction (fewer extreme weights) for slight bias (low-position items slightly underweighted). `W_max = 10` is a common default.

### Propensity Estimation — The Randomization Holdout

To estimate `P(click | position=k)`, you need randomized traffic:

```
1% of queries → randomize the ranking completely
For each position k: P(click | position=k) = clicks_at_k / impressions_at_k in holdout

Typical position propensities:
  Position 1: 0.08
  Position 2: 0.05
  Position 3: 0.03
  Position 5: 0.015
  Position 10: 0.005
```

The randomized holdout is essential — you can't estimate propensity from the biased production logs.

---

## Pointwise vs. Pairwise vs. Listwise (Deep Dive)

### Why Pointwise Works as a Ranker

Pointwise trains a binary classifier: `P(click | user, item)`. The output probability is a valid ranking score — you sort items by this probability descending.

**Why this works even though the loss doesn't optimize rank:**
The probability `P(click | user, item)` is monotone with relevance (higher relevance → higher P(click)). Sorting by probability gives the same order as sorting by relevance. Log-loss pushes the model to give high scores to clicked items and low scores to unclicked items — which is exactly what you want for ranking.

**When pointwise fails:** it ignores inter-item interactions. Two items can each have `P(click) = 0.5`, but one is much more valuable to show first. Pointwise can't represent this — items are scored independently.

### Pairwise — The Inversion Intuition

For a query with items A (relevant) and B (not relevant), a good ranker should score A > B. Pairwise loss penalizes when B > A (an inversion):

```
RankNet loss: L = log(1 + exp(s_B - s_A))
  If s_A >> s_B: loss ≈ 0  (correct order, no penalty)
  If s_A ≈ s_B: loss ≈ log(2) ≈ 0.69  (uncertain, moderate penalty)
  If s_B >> s_A: loss ≈ s_B - s_A  (wrong order, large penalty)
```

The number of pairs is O(N²) per query. For N=100 items per query: 4,950 pairs. This is expensive but gives a richer training signal than pointwise.

### LambdaMART — Why It's the Production Default

LambdaMART is a GBT-based listwise ranker that approximates NDCG gradients via "lambdas":

```
λ_ij = -∂NDCG/∂(s_i - s_j)  ← how much does swapping items i and j affect NDCG?
     = (pairwise loss gradient) × |ΔNDCG from swapping i and j|
```

The key insight: weight each pairwise gradient by the NDCG change from swapping. Swaps that move a highly relevant item from rank 10 to rank 1 get large lambdas. Swaps that move items at ranks 8 and 9 get small lambdas.

This means GBT trees focus on the high-value swaps, effectively directly optimizing NDCG.

**Why start with pointwise in an interview:**
- Maps to standard binary classification — any interviewer follows it
- Works well in practice (calibrated probabilities are good ranking signals)
- Easier to debug, explain, and serve
- Upgrade to LambdaMART when you have a working pointwise baseline and want to squeeze the last few NDCG points

---

## Cold Start (Deep Dive)

### New User — The Explore/Exploit Framing

A new user with zero history gives you no signal. The question is: how do you go from zero to personalized as fast as possible while not showing garbage?

```
Day 0 (0 interactions):  show global popular content
                         → high CTR baseline (popular = broadly appealing)
                         → zero personalization

Day 0 (5 interactions):  user clicked 3 sci-fi videos, skipped 2 romantic comedies
                         → item tower embeddings of clicked items → crude user vector
                         → retrieve nearest neighbors in item space
                         → shift toward sci-fi neighborhood

Day 0 (20 interactions): enough signal for a rough user embedding
                         → two-tower model starts working
                         → personalization is real but noisy

Day 30 (200 interactions): collaborative signal is reliable
                           → full recommendation quality
```

**Thompson sampling for exploration:**

Instead of a fixed popularity ranking, sample from a Beta(α, β) posterior per item:
```
α = clicks_on_item + 1
β = (impressions - clicks) + 1

Sample θᵢ ~ Beta(αᵢ, βᵢ) for each item
Rank by θᵢ
```

Items with few impressions have high variance → high probability of large θ → get shown. Items with many impressions have tight posteriors → exploration naturally decreases as data accumulates.

### New Item — The Content-to-Collaborative Transition

```
t=0 (no interactions):
  embedding = content_tower(title, description, genre, thumbnail)
  → purely semantic, no popularity or taste signal

t=100 interactions:
  embedding = α × content_tower() + (1-α) × collab_tower()
  α = max(0, 1 - n_interactions / N_threshold)  e.g. N_threshold = 500

t=500+ interactions:
  embedding = collab_tower()  ← collaborative signal dominates
```

The blending ensures a smooth transition. Without it: items with 1 interaction jump to a noisy collaborative embedding; items never transition off the content embedding.

---

## Diversity vs. Relevance (Deep Dive)

### Why Pure Relevance Narrows Over Time

Suppose a user's click history is 70% action movies and 30% sci-fi. A pure relevance ranker scores:
```
Action movie A: P(click) = 0.72
Action movie B: P(click) = 0.68
Action movie C: P(click) = 0.65
Sci-fi movie X: P(click) = 0.60
Documentary Y:  P(click) = 0.35
```

Top-5: all action movies. Next retraining cycle: user's history is now 95% action → the model "learns" this user only likes action → sci-fi items are never recommended → user's taste narrowing is self-fulfilling → eventually bored, churns.

### MMR — Concrete Walkthrough

Maximal Marginal Relevance: at each step, pick the item maximizing `λ × relevance − (1−λ) × similarity_to_selected`.

```
λ = 0.7 (relevance weight), embeddings in 2D for clarity

Candidates (relevance, embedding):
  A: rel=0.90, emb=[0.8, 0.1]
  B: rel=0.85, emb=[0.7, 0.2]  (similar to A)
  C: rel=0.80, emb=[0.1, 0.9]  (different from A)
  D: rel=0.75, emb=[0.2, 0.8]  (similar to C)

Step 1: pick highest relevance → A selected. selected = [A]

Step 2: for each remaining item, score = 0.7×rel − 0.3×max_sim_to_selected
  B: 0.7×0.85 − 0.3×cos(B,A) = 0.595 − 0.3×0.98 = 0.595 − 0.294 = 0.301
  C: 0.7×0.80 − 0.3×cos(C,A) = 0.560 − 0.3×0.09 = 0.560 − 0.027 = 0.533  ← pick C
  D: 0.7×0.75 − 0.3×cos(D,A) = 0.525 − 0.3×0.12 = 0.525 − 0.036 = 0.489

selected = [A, C]

Step 3: B vs. D
  B: 0.7×0.85 − 0.3×max(cos(B,A), cos(B,C)) = 0.595 − 0.3×0.98 = 0.301
  D: 0.7×0.75 − 0.3×max(cos(D,A), cos(D,C)) = 0.525 − 0.3×0.97 = 0.234

Pick B. Final slate: [A, C, B, D]
```

Result: the most relevant item (A) leads, but C (different genre) is injected early, before another action movie (B).

---

## Two-Stage Ranking (Deep Dive)

### Why the Dot Product Can't Replace the Ranker

Two-tower scores items with a single dot product `u·v`. This means the user and item representations must each independently encode everything relevant — there's no interaction between user and item features inside the model.

```
Things dot product CAN capture:
  "This user generally likes cooking content"
  "This item is a cooking tutorial"
  → high score because both vectors are in the cooking neighborhood

Things dot product CANNOT capture:
  "This user just watched 3 French cuisine videos in this session"
  "This item is a French cuisine tutorial"
  → the session context × item specificity interaction requires seeing both together
```

The ranking model (Stage 2) takes the top-K candidates and scores each with full cross-features:

```
Ranking model input for (user, item) pair:
  user_history_embedding × item_embedding interaction features
  user_location × item_delivery_time (Uber)
  user_budget × item_price_tier (e-commerce)
  current_session_signals × item_topic_match
```

These cross-features are what justify the ranking stage — it handles everything two-tower can't.

### Latency Math — Why the Split Is Necessary

```
Full ranking model for 1B items:
  - Model: 5M parameter MLP, 10ms per item
  - 1B × 10ms = 10B ms = 115 days per request  ← impossible

Two-tower + ANN (Stage 1):
  - User embedding: 5ms
  - FAISS search over 1B items: 10ms
  - Total retrieval: 15ms, returns top-500

Ranking model (Stage 2):
  - 500 items × 10ms = 5,000ms  ← still too slow

Ranking with distilled/quantized model:
  - 500 items × 0.5ms = 250ms  ← marginal
  - Or: vectorized batch inference: 500 items in 20ms total ← acceptable
```

The key: Stage 2 runs the expensive model on 500 items, not 1B. The 2,000,000× reduction in candidate set is what makes the architecture work.

---

## Feedback Loops (Deep Dive)

### The Rich-Get-Richer Math

Let `p_i(t)` = probability that item i is recommended at time t. In a popularity-biased system:

```
p_i(t+1) ∝ p_i(t) × (1 + α × clicks_i(t))
```

Items with more historical clicks get recommended more → get more clicks → get recommended more. This is a power-law accumulation.

**Catalog coverage degradation:**
```
Time 0:  2000 items each get ~0.05% of recommendations
Time 6mo: top 50 items get 80% of recommendations
          bottom 1950 items get 20%
          → user sees the same 50 items repeatedly → disengagement
```

### Measuring the Feedback Loop

**Catalog coverage metric:** fraction of catalog recommended at least once per week:

```
Healthy system:   80% of catalog items seen by at least one user per week
Feedback loop:    5% of catalog items account for 90% of all recommendations
Alert threshold:  if top-1% items account for >50% of recommendations
```

**Tail item CTR:** track click-through rate for items in the bottom quintile of popularity. A healthy system maintains tail CTR ≥ 30% of head item CTR. If tail CTR → 0, the model has stopped showing tail items entirely.

---

## Collaborative Filtering vs. Content-Based (Deep Dive)

### Why CF Captures What Content Misses

Content features describe the item: genre, director, keywords. But they can't capture:
```
"This film has slow pacing but rewards patient viewers"  ← not in metadata
"Users who liked Arrival also like Annihilation"        ← latent taste pattern
"This item is high quality despite niche content"       ← quality signal
```

CF embeddings learn latent dimensions from co-engagement patterns. If users A, B, C all click items [X, Y, Z] and also click [W], the CF model places W near X, Y, Z in embedding space — even if their content metadata is completely different.

### Regularization for Long-Tail CF Embeddings

A long-tail item with 10 interactions has a high-variance embedding — 10 data points aren't enough to reliably place it in the embedding space. Its embedding may be near items that had the 10 interactions by chance, not by taste affinity.

Fix:
```
Loss = contrastive_loss + λ × ||v_item - v_content(item)||²
```

Anchor the CF embedding to the content embedding. As more interactions accumulate, the contrastive loss dominates and the item drifts to its true latent position. With few interactions, it stays near the content-based anchor.
