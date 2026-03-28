# Meta News Feed Ranking — ML System Design

**Domain:** `ranking`
**Target Company:** Meta
**Difficulty Bar:** L6 (E6)
**Date:** 2026-03-27
**Related Designs:** `netflix-recommendation-system.md`, `../templates/ml-system-design-template.md`

---

## 0. Scorecard

| Axis | Score | Weakest Point |
|---|---|---|
| Requirements | ★★★★★ | — |
| Data Modeling | ★★★★☆ | Cross-regional feature consistency not fully addressed |
| ML Pipeline | ★★★★★ | — |
| Failure Modes | ★★★★★ | — |
| Capacity | ★★★★★ | — |

**Overall:** `STRONG HIRE`
**Top Gap:** Cross-regional feature store consistency — probe how user embeddings stay consistent across US-East, EU, APAC shards.

---

## 1. Requirements

#### Functional Requirements
1. Rank and surface the top-K posts from a user's friend/follow graph and interest graph for the home feed
2. Blend organic content (friends, groups, pages) with paid content (ads) in a single ranked feed
3. Support multiple feed surfaces: Home, Watch, Reels, Groups — each with different ranking objectives
4. Near-real-time response to new posts and interactions (within minutes of publication)

#### Non-Functional Requirements

| Dimension | Target | Business Tie |
|---|---|---|
| Latency (p99 serving) | ≤ 100ms | Beyond this, feed load abandonment increases measurably |
| Availability | 99.99% | Feed downtime = direct DAU and ad revenue loss at 3B user scale |
| Consistency | Eventual (minutes) | Feed staleness < 5 min acceptable; strong consistency would require cross-region locking |
| Throughput | ~580K peak QPS | 3B DAU × ~17 feed loads/day / 86,400 × 1.1× steady peak |
| Feature freshness (user) | ≤ 5 min | Reaction/share signals need to propagate quickly for viral content |
| Feature freshness (content) | ≤ 1 min | Newly published posts must enter ranking within 60s of publication |

#### Scale Numbers (stated upfront)
- **DAU / MAU:** 3B DAU / 3.3B MAU
- **Peak QPS:** ~580K sustained (Meta feeds billions of requests/day; no sharp peak — rolling global load)
- **Posts published/day:** ~100M new posts
- **Ad impressions/day:** ~10B across surfaces
- **Graph edges (friend/follow):** ~150B edges (social graph)

#### Out of Scope
- Ad auction pricing and bidding (separate system)
- Content creation and publishing pipeline
- Integrity/misinformation classification (feeds as input but not designed here)
- Video transcoding and CDN delivery

> **Meta rubric:** Scale mechanisms named explicitly. "It's distributed" is not sufficient — sharding key, replication factor, and hot partition strategy are required answers.

---

## 2. Data Modeling

#### Input Signals

| Feature | Type | Freshness | Source | Notes |
|---|---|---|---|---|
| User interest embedding | `batch` | 1 hour | Spark on interaction logs → TAO/Hive | 256-d dense vector across topic taxonomy |
| User social graph (friend/follow list) | `batch` | 15 min | TAO (social graph store) → candidate generator | Fan-out during candidate generation, not serving |
| Post engagement velocity (likes/comments/shares in last 30 min) | `real-time` | < 1 min | Scribe → Flink → Memcache | Most predictive of virality; Flink aggregates sliding window |
| Post freshness score | `batch` | 1 min | Publication timestamp + content type decay curve | Reels decay faster than articles |
| User × post affinity features | `real-time` | request-time | Computed at serving from user embedding + post embedding | Cross features via DLRM interaction layer |
| Author affinity (how much user interacts with this author) | `batch` | 1 hour | Aggregated interaction history | Strong predictor; decays toward global author popularity |
| Content type preference | `batch` | daily | User's click distribution across video/photo/link/text | Session-level override available |
| Ad quality score (ads only) | `batch` | 15 min | Ad auction system → Memcache | Relevance score from ad ranking system |
| Integrity score | `batch` | 1 hour | Integrity ML pipeline → Hive | Lower bound filter: posts below threshold excluded entirely |

#### Label Definition
- **Label:** Engagement event (like, comment, share, click, video watch ≥ 50%) as positive; scroll-past as negative
- **Collection strategy:** Implicit feedback via Scribe event log; multi-label (can predict multiple engagement types jointly)
- **Positive/negative ratio:** ~1:50 (most posts are scrolled past)
- **Label delay:** Click/like immediate; comments observable within minutes; long-form video completion ~10 min lag
- **Bias risks:**
  - **Position bias:** Posts ranked first get 3–5× more engagement regardless of quality → inverse propensity scoring (IPS) in training loss
  - **Feedback loop:** Highly ranked posts generate more signal → reinforces ranking → filter bubble; address with diversity injection and exploration budget (ε-greedy, 3% of slots)
  - **Author popularity bias:** Viral authors dominate signals; normalize by author reach (impressions-corrected engagement rate)

#### Storage Engine Choice

| Layer | Engine | Justification |
|---|---|---|
| Social graph | TAO (Meta's custom graph cache, backed by MySQL + Memcache) | Low-latency traversal of friend graph; horizontal sharding by user_id |
| Feature store (online, user) | Memcache cluster (regional) | Sub-ms reads; regionally sharded; TTL-based invalidation |
| Feature store (online, content) | Memcache + TAO | Post features cached per post_id; TAO for graph-structured post metadata |
| Feature store (offline) | Hive on HDFS + Presto | Petabyte-scale; analytical workloads; time-partitioned |
| Training data | ORC/Parquet on HDFS | Columnar; Spark-readable; daily snapshots + streaming joins |
| Logs / labels | Scribe → Kafka → Hive | Scribe is Meta's internal log transport; high-throughput, low-latency |
| Model artifacts | HDFS + custom model registry | Versioned; supports A/B experiment model isolation |

#### Online vs. Offline Split

```
Offline (batch, Spark + Presto)                Online (real-time, < 100ms budget)
───────────────────────────────────────        ──────────────────────────────────────────
Scribe events → Hive (daily snapshots)         Request: user_id + device + surface
Spark: compute user interest embeddings        TAO: fetch friend/follow list (fan-out)
Spark: post embeddings (text + visual)         Candidate generator: ~2K posts from graph
Spark: author affinity aggregation             Memcache: user features + post features (5ms)
Daily DLRM training (organic + ads jointly)    DLRM scoring: rank 2K candidates (30ms)
Evaluation: NDCG@10, engagement rate lift      Diversity + integrity filter (5ms)
Champion/challenger promotion                  Response + async Scribe log
```

**Hot partition strategy:** Celebrity users (Beyoncé, Obama, Messi) have 100M+ followers. Fan-out on write at publication time is infeasible. Solution: **fan-out on read** for high-follower accounts — their posts are fetched at request time from a celebrity post cache (Memcache, TTL 5 min), not pre-distributed to follower feeds. Threshold: fan-out on write for < 10K followers; fan-out on read above.

#### Schema (key entities)

```
User: {
  user_id:              int64        # primary key; sharded by user_id % N across TAO regions
  interest_embedding:   float[256]   # updated hourly by Spark
  friend_ids:           int64[]      # TAO adjacency list; sharded
  country_region:       string       # for regional data residency
  updated_at:           timestamp
}

Post: {
  post_id:              int64        # primary key
  author_id:            int64
  content_embedding:    float[256]   # text + image, updated at publish time
  content_type:         enum[PHOTO, VIDEO, LINK, TEXT, REEL]
  published_at:         timestamp
  engagement_velocity:  float        # real-time; Flink sliding window
  integrity_score:      float        # from integrity pipeline
}

FeedEvent: {
  user_id:              int64
  post_id:              int64
  event_type:           enum[IMPRESSION, LIKE, COMMENT, SHARE, CLICK, VIDEO_COMPLETE]
  rank_position:        int          # position in feed (for IPS)
  surface:              enum[HOME, WATCH, REELS, GROUPS]
  timestamp:            timestamp
}
```

> **Meta rubric:** Hot partition strategy explicitly named (fan-out on read for celebrity accounts). Sharding key stated (user_id % N). Geographic replication for EU data residency addressed via regional TAO clusters.

---

## 3. ML Pipeline

### 3a. Offline Training

#### Data Pipeline
- **Ingestion:** Scribe (log transport) → Kafka → Hive (hourly partitions); daily training data join of impressions + labels
- **Feature engineering:**
  - User interest embeddings: Transformer-based sequence model over last 1000 interactions → 256-d vector (Spark, hourly)
  - Post embeddings: CLIP for images/video thumbnails + RoBERTa for text → 256-d (at publish time, async)
  - Engagement velocity: Flink sliding window aggregation (1 min / 10 min / 60 min buckets) → Memcache
- **Train/val/test split:** Time-based — train D-7 to D-1, val D-1, test D-0. Never shuffle across days.
- **Pipeline orchestration:** Internal DAG scheduler (similar to Airflow); idempotent partitioned Spark jobs; retry ×3 on transient failure.

#### Model Architecture

**Organic feed ranking (DLRM):**

| Option | Pros | Cons | Decision |
|---|---|---|---|
| DLRM (Meta's own) | Handles mixed sparse + dense features; embedding tables for user/item IDs; production-proven at Meta | Requires FSDP for large embedding tables; complex to tune | **Chosen** |
| Two-tower + re-ranking | Simpler serving; good for retrieval stage | Less expressive for interaction features at ranking stage | Two-tower used at retrieval; DLRM at ranking |
| Transformer over history | Rich sequential context | Inference too slow at 580K QPS for ranking 2K candidates | Rejected for online ranking; used for offline embedding precomputation |

**Selected pipeline:**
1. **Candidate generation** (two-tower ANN): narrows social graph posts to ~2K candidates using user embedding × post embedding dot product
2. **Ranking** (DLRM): scores each of 2K candidates using dense features + sparse embedding interactions (user_id, author_id, post_id embeddings + cross features)
3. **Post-ranking** (lightweight re-ranker): applies diversity, integrity filters, and ads blending

**Training objective:** Multi-task — jointly train on:
- P(like) — primary engagement signal
- P(comment) — deeper engagement; weighted 3× vs. like
- P(share) — social amplification; weighted 5× vs. like
- P(negative_feedback) — "Hide post" / "See less of this"; weighted −10× (strong downrank signal)

Loss: `L = Σ wᵢ × BCE(engagement_i) + λ × L2_regularization`

#### Training Infrastructure
- **Framework:** PyTorch + FSDP — embedding tables for 3B users × 256-d exceed single-GPU memory; FSDP shards across 256 A100 GPUs
- **Scale:** 256× A100 80GB; ~12 hours/full retrain; incremental hourly updates via online learning on recent interactions
- **Mixed precision:** Yes (bfloat16) — embedding table lookups in fp16; gradient accumulation in fp32
- **Gradient checkpointing:** Yes — user interaction sequence in Transformer-based embedding precomputation
- **Eval metrics:** NDCG@10, per-engagement-type AUC, engagement rate lift vs. champion on held-out slice

---

### 3b. Online Serving

#### Inference Path

```
Client (iOS/Android/Web)
  → CDN (static assets)
  → Feed API (Load Balancer)
  → Feed Ranking Service
      ├─ Candidate Generation
      │    ├─ TAO: fetch friend/follow list (fan-out for <10K followers)
      │    ├─ Celebrity post cache (Memcache, for >10K followers)
      │    └─ Interest-based candidates (ANN on user embedding, ~500 posts)
      │         → merged: ~2,000 candidates
      ├─ Feature Fetch
      │    ├─ Memcache: user features + post features (5ms)
      │    └─ Real-time engagement velocity (Flink → Memcache, 2ms)
      ├─ DLRM Scoring
      │    └─ Batch inference over 2K candidates (GPU, 30ms)
      ├─ Post-ranking
      │    ├─ Integrity filter: remove posts below threshold
      │    ├─ Diversity injection: max 3 posts from same author per 10-post block
      │    ├─ Ads blending: inject ranked ads at positions 4, 9, 15...
      │    └─ Surface-specific rules (Reels: video-only; Watch: long-form only)
      └─ Response + async Scribe log (impression + ranks)
```

#### Latency Budget Breakdown

| Step | p50 | p99 | Notes |
|---|---|---|---|
| Candidate generation (TAO + cache) | 8ms | 20ms | Fan-out bounded by follower cap strategy |
| Feature fetch (Memcache) | 3ms | 8ms | Parallel reads; regional Memcache |
| DLRM scoring (2K candidates, GPU) | 20ms | 45ms | TensorRT optimization; batched |
| Post-ranking + ads blending | 3ms | 8ms | CPU; lightweight |
| Network + serialization | 5ms | 15ms | gRPC + protobuf |
| **Total** | **39ms** | **96ms** | Budget: ≤ 100ms p99 ✓ |

#### Caching Strategy
- **User feature cache:** Memcache per region; TTL 5 min; invalidated on new interaction events via Scribe trigger
- **Post feature cache:** Memcache; TTL 1 min for high-velocity posts; 1 hour for older posts
- **Celebrity post cache:** Pre-ranked list of top-N posts from celebrity accounts; rebuilt every 5 min by background job; serves fan-out-on-read

---

### 3c. Monitoring

> **Designed upfront — Meta will ask "how do you know the feed quality is improving, not just engagement metrics gaming?"**

#### Drift Detection

| Signal | Method | Threshold | Action |
|---|---|---|---|
| User embedding drift | PSI daily | PSI > 0.2 | Retrain trigger |
| Engagement rate by content type | Rolling 24hr vs. 7-day baseline | > −5% relative per content type | Page on-call; investigate label pipeline |
| DLRM score distribution | Rolling mean ± 2σ | > 15% shift sustained 2hr | Shadow model comparison |
| Integrity filter hit rate | Daily audit | > 2× baseline | Integrity system incident escalation |
| Feed diversity (Herfindahl index on author distribution) | Daily computation | > 0.4 (monopolization signal) | Diversity injection parameter tuning |
| p99 latency | Real-time APM | > 100ms sustained 5min | Circuit breaker; alert on-call |

#### Shadow Scoring
- Challenger model (e.g., new DLRM architecture) runs on 1% traffic; results logged but not served
- Comparison cadence: daily offline on shadow slice; metrics: NDCG@10, per-task AUC, diversity index
- Promotion criteria: +1% relative NDCG@10, no regression on negative feedback rate, no p99 degradation, 7-day shadow validation

#### A/B Holdout Design
- **Unit of randomization:** `user_id` (consistent cross-device experience)
- **Holdout size:** 2% permanent holdout (Meta scale → 60M users; statistically robust even for rare events)
- **Treatment size:** Typical experiment 5% treatment vs. 5% control
- **Primary metric:** Engaged time per session (not raw clicks — guards against clickbait)
- **Guardrail metrics:** Negative feedback rate ("hide post"), unsubscribe rate, p99 latency
- **Duration:** Minimum 2 weeks (accounts for novelty effect and day-of-week patterns)

> **Meta rubric:** Monitoring covers both ML drift and business health (diversity index, negative feedback). The wellbeing guardrail (negative feedback) is the key Meta-specific addition — optimizing only for engagement creates filter bubbles.

---

## 4. Failure Modes

| Scenario | Impact | Mitigation | Fallback |
|---|---|---|---|
| TAO social graph unavailable | No candidate generation from friend graph | Retry ×2 (10ms timeout); serve from cached friend list (Redis, 1hr TTL) | Interest-based candidates only (ANN on user embedding, no social graph) |
| Memcache regional outage | Missing user/post features | Auto-failover to secondary region Memcache (cross-region replication, 50ms overhead) | Default features (global averages); degrade gracefully |
| DLRM model server timeout | No personalized ranking | Pre-ranked feeds cached per user (hourly batch job, S3) | Reverse-chronological feed from friend graph; no personalization |
| Celebrity post cache stale | Top influencer posts not surfaced | Rebuild cache triggered by publication event (< 5 min); dual cache (hot + warm) | Fetch directly from TAO (higher latency, acceptable for < 0.1% of requests) |
| Flink pipeline lag (engagement velocity stale) | Real-time virality signals unavailable | Monitor Kafka consumer lag; alert at > 5 min lag | Use 1-hour cached velocity scores; mark as degraded in logs |
| Training job failure | Model does not update | Auto-retry ×3; champion stays live; page on-call | Champion model continues; staleness SLA: 48hr max before forced retrain |
| Regional data center degraded | Latency spike in affected region | Geographic load balancing; traffic shifted to adjacent region within 30s (BGP-level) | Serve from adjacent region with up to 50ms additional latency |

#### Circuit Breaker Thresholds
- **Error rate:** Trip at > 0.5% 5xx over 60s → serve cached feeds; alert on-call
- **Latency:** Trip at p99 > 120ms sustained 3min → bypass DLRM ranking; serve interest-based ANN output only
- **Recovery:** Half-open after 30s; restore full path on 5 consecutive successes

#### Degraded-Mode Behavior
1. **Level 1** — Personalized cached feed (Memcache/S3, up to 1 hour stale) — still personalized
2. **Level 2** — Interest-based ANN ranking (no social graph, no real-time signals) — reduced personalization
3. **Level 3** — Reverse-chronological feed from friend graph (no ML, no ranking) — fully rule-based

> **Meta rubric:** Geographic distribution failure explicitly addressed (regional failover via BGP load balancing). Hot partition strategy (celebrity accounts) and cross-region replication stated upfront.

---

## 5. Capacity Estimates

> **Assumptions:**
> - DAU: 3B
> - Feed loads per DAU: ~17/day (high engagement platform; mobile background refresh included)
> - Sustained peak QPS: ~580K (no sharp peak — rolling global load across time zones)
> - Candidates per request: 2,000 (social graph + interest-based)
> - User/post feature vector: 256 floats = 1KB
> - DLRM model size: 12GB (large embedding tables for 3B users)
> - Log retention: 30 days hot (Hive), 2 years cold (cold storage)
> - Serving replicas: A100 GPU nodes

#### Estimates

| Metric | Calculation | Result |
|---|---|---|
| Avg QPS | 3B × 17 loads/day / 86,400 | **~590K QPS** |
| Peak QPS | ~590K (rolling global; no sharp peak) | **~590K QPS** |
| Memcache reads/s (user features) | 590K QPS × 1 read | **~590K reads/s** |
| Memcache reads/s (post features) | 590K × 2K candidates (batched, ~100/call) | **~11.8M lookups/s → ~5.9K batch calls/s** |
| User embedding storage (Memcache) | 3B users × 1KB | **~3TB** (distributed Memcache cluster) |
| Post embedding storage (active posts) | 100M active posts × 1KB | **~100GB** |
| Scribe ingest throughput | 590K requests × ~5 events/request avg | **~3M events/s → ~300GB/hr log volume** |
| Training data (daily) | 3M events/s × 86,400s × 200B/event | **~52TB/day** (compressed ~10TB) |
| DLRM embedding table (3B users × 256-d) | 3B × 256 × 4B | **~3TB** (sharded across GPU memory + CPU RAM with FSDP) |
| Serving replicas (DLRM) | 590K QPS / 200 QPS per A100 (2K candidates) | **~2,950 A100 GPUs** (in practice spread across many data centers) |
| Training replicas | 256 A100 80GB (FSDP sharding for 3TB embedding table) | **256 GPUs, ~12hr/run** |

---

## 6. Open Questions & Follow-ups

#### Unresolved Design Decisions
- [ ] **Online learning vs. daily retraining:** Meta uses online learning (hourly gradient updates on recent interactions) for the ranking model. Should we add this? Trade-off: higher signal freshness vs. training instability risk
- [ ] **Wellbeing objective:** How to weight negative signals (hide post, report) vs. engagement? Currently weighted heuristically — should it be a constrained optimization (maximize engagement subject to wellbeing floor)?
- [ ] **Cross-surface feature sharing:** Home feed and Reels share user embeddings — does cross-surface contamination hurt Reels-specific ranking? Needs multi-head architecture evaluation
- [ ] **Advertiser fairness:** Small advertisers get fewer impressions due to lower bids — is there a floor for auction-based ad delivery fairness?

#### Company Rubric Gaps (self-assessment)
- [x] Requirements: scale mechanisms named (TAO sharding, fan-out-on-read for celebrities, geographic replication)
- [x] Data Modeling: hot partition strategy explicit; online/offline consistency addressed
- [x] ML Pipeline: monitoring includes wellbeing guardrail (negative feedback), not just engagement
- [x] Failure Modes: geographic failover addressed; 3-level degraded mode defined
- [x] Capacity: embedding table size calculated; GPU replica count derived

#### Recommended Follow-up Problems
- Ads CTR Prediction (Meta) — same infrastructure but pure precision/recall optimization with auction pricing
- Pinterest Ads Ranking — MMOE-DCN multi-objective variant of this same problem

---

## 7. References

| Resource | Type | Relevance |
|---|---|---|
| Naumov et al., "Deep Learning Recommendation Model for Personalization and Recommendation Systems" (2019) | Paper | DLRM architecture used at Meta for ads and organic ranking |
| He et al., "Practical Lessons from Predicting Clicks on Ads at Facebook" (KDD 2014) | Paper | Foundational Meta ads CTR paper; GBDT + logistic regression → DNN evolution |
| Bronstein et al., "Geometric Deep Learning" (2021) | Paper | Graph neural networks for social graph-based features |
| Meta AI Blog: "Powered by AI: Instagram's Explore recommender system" | Blog | Two-tower retrieval + ranking at Meta scale; directly applicable |
| Meta Engineering: "Scaling data ingestion for machine learning training at Meta" | Blog | Scribe → Hive pipeline; feature freshness at 3B user scale |
| Agarwal et al., "Overlap in News Feed: An analysis of Facebook's feed algorithm" | Paper | Feed diversity metrics; Herfindahl index as diversity guardrail |
| Meta AI Blog: "DLRM: An advanced, open source deep learning recommendation model" | Blog | Production DLRM details; embedding table sharding with FSDP |
| Zhao et al., "Recommending What Video to Watch Next: A Multitask Ranking System" (RecSys 2019) | Paper | Multi-task ranking with engagement + satisfaction objectives — directly applicable |
| TAO: Facebook's Distributed Data Store for the Social Graph (ATC 2013) | Paper | TAO internals: social graph storage, fan-out, regional replication |
| Chapelle & Li, "An Empirical Evaluation of Thompson Sampling" (NeurIPS 2011) | Paper | Exploration strategy for feed diversity injection |
