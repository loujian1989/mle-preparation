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
