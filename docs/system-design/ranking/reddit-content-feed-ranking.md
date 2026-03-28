# Reddit Content Feed Ranking — ML System Design

**Domain:** `ranking`
**Target Company:** Reddit
**Difficulty Bar:** L6 (Staff MLE)
**Date:** 2026-03-27
**Related Designs:** `meta-news-feed-ranking.md`, `../templates/ml-system-design-template.md`

---

## 0. Scorecard

| Axis | Score | Weakest Point |
|---|---|---|
| Requirements | ★★★★★ | — |
| Data Modeling | ★★★★☆ | Cross-subreddit interest modeling partially addressed |
| ML Pipeline | ★★★★★ | — |
| Failure Modes | ★★★★★ | — |
| Capacity | ★★★★★ | — |

**Overall:** `STRONG HIRE`
**Top Gap:** Cross-subreddit interest transfer — a user active in r/Python but new to r/MachineLearning; can history from adjacent subreddits bootstrap their ranking quality?

---

## 1. Requirements

#### Functional Requirements
1. Rank posts on the personalized home feed (r/all equivalent, personalized to subscriptions + interests)
2. Rank posts within individual subreddits by multiple sort modes: Best (personalized), Hot (community velocity), New (chronological), Top (all-time)
3. Integrate spam/low-quality pre-filter: posts below quality threshold excluded before ranking
4. Detect and handle controversial content: bimodal vote distribution → special treatment (not suppression)
5. Maintain community health: diversity of subreddits shown in home feed to prevent echo chamber collapse

#### Non-Functional Requirements

| Dimension | Target | Business Tie |
|---|---|---|
| Latency (p99 serving) | ≤ 150ms | Feed load time; Reddit's audience is highly latency-sensitive (power users with fast internet) |
| Availability | 99.9% | Reddit has historically had outages; investment in resilience is high-priority |
| Consistency | Eventual (seconds for Hot/Best; milliseconds for New) | New sort must be real-time; Best can tolerate seconds of lag |
| Throughput | ~60K peak QPS | 50M DAU × ~10 feed loads/day / 86,400 × 2× evening peak |
| Feature freshness (vote velocity) | ≤ 30s | Flair-hot algorithm requires real-time vote signal for emerging posts |
| Community health | Bottom 50% of subscribed subreddits visible ≥ 20% of slots | Prevents dominant subreddits from crowding out smaller ones |

#### Scale Numbers (stated upfront)
- **DAU / MAU:** 50M DAU / 100M+ MAU
- **Peak QPS:** ~60K
- **Posts/day:** ~4M new posts; ~200M new comments
- **Kafka events/day:** 65B+ (Reddit's documented throughput)
- **Subreddits:** ~3M active subreddits; ~100K with daily posts

#### Out of Scope
- Comment ranking within a post (separate ranking problem)
- Ads ranking (separate system; Reddit Ads is independent)
- Search ranking (separate; different recall/precision trade-offs)
- Spam/account ban decisions (Trust & Safety ops)

> **Reddit rubric:** Reddit's "Best" sort is the personalized feed; distinguish it from "Hot" (community-level, not personalized) upfront. Spam classifier score is a **hard pre-filter** (not a soft ranking feature) — posts below quality threshold never enter ranking. Community health (subreddit diversity) is a guardrail metric.

---

## 2. Data Modeling

#### Input Signals

| Feature | Type | Freshness | Source | Notes |
|---|---|---|---|---|
| User subreddit affinity vector | `batch` | daily | Spark on upvote/comment history → feature store | Per-subreddit engagement rate over last 30 days |
| User interest embedding (cross-subreddit) | `batch` | 1 hour | Spark on content interactions → feature store | 256-d; enables cross-subreddit transfer |
| Post upvote velocity (upvotes in last 5 min) | `real-time` | ≤ 30s | Kafka → Flink → Redis | Primary signal for Hot/Best; Flink sliding window |
| Post comment rate (comments in last 1hr) | `real-time` | ≤ 1 min | Kafka → Flink → Redis | Conversation depth; weighted higher than passive upvotes |
| Post award count | `real-time` | ≤ 5 min | Award service → Kafka → Redis | Community-validated quality signal |
| Post age (time since creation) | `real-time` | request-time | Post metadata | Recency decay: Hot score decays as post ages |
| Post content embedding (title + body) | `batch` | at post creation | BERT on post content → feature store | Cold-start signal; updated once at publish |
| Spam/quality score | `batch` | ≤ 15 min | Spam classifier pipeline → Redis | Hard pre-filter: posts with score < threshold excluded entirely before ranking |
| Vote controversy score | `batch` | 1 hour | Wilson score confidence interval on vote distribution | Bimodal votes = controversial; handled differently from low-quality |
| User × subreddit history | `batch` | daily | Spark on subscription + engagement logs | Subreddit affinity for home feed blending |

#### Label Definition
- **Label (primary):** Upvote event (positive); downvote or scroll-past (negative)
- **Label (secondary):** Comment event (deeper engagement; weighted 3× upvote); award given (community validation; weighted 10×)
- **Collection strategy:** Explicit feedback (votes) + implicit feedback (scroll-past, click-through, time-on-post)
- **Positive/negative ratio:** ~1:15 (most posts in a feed are scrolled past)
- **Label delay:** Upvote/downvote immediate; comment within minutes; full post engagement pattern observed over 24hr for hot posts
- **Bias risks:**
  - **Vote manipulation:** Coordinated upvote/downvote campaigns by subreddit communities; detected by vote spike anomalies and account age signals in spam filter
  - **Recency bias in training:** Hot new posts are over-represented in training data relative to their long-term quality → use time-stratified sampling (sample equally from posts at 1hr, 6hr, 24hr, 7day age)
  - **Subreddit size bias:** Large subreddits (r/AskReddit, r/WorldNews) generate most training signal; small niche subreddits are underrepresented → per-subreddit-tier evaluation

#### Storage Engine Choice

| Layer | Engine | Justification |
|---|---|---|
| Vote velocity + comment rate (real-time) | Redis (key: post_id → {upvotes_5min, comments_1hr}) | Sub-ms reads; INCR + sliding window expiry; high write rate |
| Spam/quality scores | Redis (key: post_id → quality_score) | Updated every 15 min; fast lookup for pre-filter |
| User feature store (online) | Redis Cluster | Sub-ms; user affinity vectors; sharded by user_id |
| Post content embeddings | Redis (hot posts) + S3 (all posts) | Hot posts (< 24hr) loaded in Redis; older posts fetched from S3 on demand |
| Feature store (offline) | Hive on HDFS (Kafka → Flink → Hive) | Reddit's documented stack; 65B events/day; partitioned by date + subreddit |
| Training data | Parquet on S3 (date + subreddit-tier partitioned) | Columnar; Spark-efficient |
| Logs / labels | Kafka (Confluent Cloud) → Flink → Hive | Reddit's production stack; 65B events/day |
| Model artifacts | S3 + internal registry | Versioned; supports per-sort-mode model variants |

#### Online vs. Offline Split

```
Offline (batch, Spark + Hive)                       Online (real-time, < 150ms)
────────────────────────────────────────────         ──────────────────────────────────────────────
Kafka events → Flink → Hive                          Request: user_id + subreddit (or home feed)
Spark: user subreddit affinity vectors               Spam pre-filter: Redis quality_score check (<1ms)
Spark: post content embeddings (BERT)                Redis: vote velocity + comment rate (2ms)
Spam classifier: quality score (Flink, 15 min lag)   Redis: user affinity vectors (2ms)
Daily GBDT training (Best sort)                      Candidate pool: recent posts (< 48hr) from subscribed subreddits
Heuristic: Hot score formula (Wilson + decay)        GBDT scoring (Best mode) or formula (Hot/New/Top)
Subreddit diversity audit                            Subreddit diversity injection (home feed)
                                                     Response + async Kafka log
```

**Sort mode differentiation:**
- **New:** Pure reverse-chronological; no ML; latency ≤ 10ms
- **Hot:** Deterministic formula: `Hot = log10(max(upvotes - downvotes, 1)) + age_seconds / 45000`; real-time; no ML
- **Top:** Aggregated upvote count over selected time window; Spark batch; no real-time ML
- **Best (home feed):** Full ML personalization; GBDT ranking model; the focus of this design

#### Schema

```
Post: {
  post_id:          string
  subreddit_id:     string
  author_id:        string
  title:            string
  content_embedding:float[256]      # BERT; updated at publish
  created_at:       timestamp
  upvote_velocity:  float           # upvotes in last 5 min; Flink-computed
  comment_rate:     float           # comments in last 1hr
  award_count:      int
  quality_score:    float           # spam classifier; [0, 1]; hard filter threshold: 0.3
  controversy_score:float           # Wilson score bimodality; [0, 1]
}

UserFeedState: {
  user_id:          string
  subreddit_affinity: {subreddit_id: float}  # engagement rate per subreddit
  interest_embedding: float[256]              # cross-subreddit
  subscribed_subreddits: string[]
  updated_at:       timestamp
}

FeedEvent: {
  user_id:          string
  post_id:          string
  event_type:       enum[IMPRESSION, UPVOTE, DOWNVOTE, COMMENT, SCROLL_PAST]
  sort_mode:        enum[BEST, HOT, NEW, TOP]
  rank_position:    int
  timestamp:        timestamp
}
```

> **Reddit rubric:** Spam classifier score is pre-filter (not a ranking feature). Controversy score is computed and used for *special treatment* (not suppression). Vote velocity at multiple time windows is the primary real-time signal — Flink sliding windows required.

---

## 3. ML Pipeline

### 3a. Offline Training

#### Data Pipeline
- **Ingestion:** Reddit's Kafka (65B events/day) → Flink (real-time feature computation: vote velocity, comment rate) → Hive (daily snapshots)
- **Feature engineering:**
  - Vote velocity: Flink sliding window (5 min, 30 min, 2 hr) per post_id → Redis
  - User affinity: Spark daily aggregation of (user, subreddit) engagement rates, weighted by recency and interaction type (comment > upvote > view)
  - Controversy detection: Wilson score lower bound on upvotes; bimodal distribution detected when upvote_rate ∈ (0.3, 0.7) with high total vote count
- **Train/val/test split:** Time-based; stratified by subreddit_tier (top-100 subreddits, top-1000, long-tail) to ensure representation; train D-30 to D-1; val D-1; test D-0

#### Model Architecture

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Wilson score Hot formula | No ML; real-time; community-interpretable | Not personalized; same ranking for all users | **Hot sort only** |
| Logistic Regression | Interpretable; fast | Cannot capture non-linear interactions between velocity + user affinity | Baseline; rejected for Best |
| GBDT (LightGBM, chosen) | Handles mixed feature types (velocity + embeddings + categorical); fast inference; SHAP interpretable | No sequential modeling; embeddings must be precomputed | **Chosen for Best sort** |
| Deep learning ranker | Better on embedding interactions | Slower inference; marginal gain for tabular + precomputed embeddings | Future direction |

**Selected: LightGBM for Best sort**
- **Input features:**
  - Real-time: `upvote_velocity_5min`, `upvote_velocity_30min`, `comment_rate_1hr`, `award_count`, `post_age_hours`
  - User × post: `user_subreddit_affinity[post.subreddit_id]`, cosine_similarity(user_embedding, post_embedding), `user_historical_upvote_rate_in_subreddit`
  - Post quality: `quality_score` (used as feature for soft ranking, after hard filter already applied), `controversy_score`
- **Objective:** Binary cross-entropy on upvote label; comment events added as weighted positive labels (3× weight); award events (10× weight)
- **Post-ranking:** Subreddit diversity injection for home feed: if top-10 posts are all from 2 subreddits → inject posts from 3rd most-affined subreddit

#### Training Infrastructure
- **Framework:** LightGBM (CPU cluster)
- **Scale:** 8× CPU nodes, ~30min/run on 30 days of engagement data
- **Eval metrics:** NDCG@10 (Best sort on held-out users), community health Gini (subreddit diversity in served feed), spam pre-filter recall (what % of spam posts are caught before ranking)

---

### 3b. Online Serving

#### Inference Path

```
Client (iOS/Android/Web)
  → API Gateway (auth, rate limit)
  → Feed Service
      ├─ Spam Pre-filter (mandatory, before all else)
      │    └─ Redis: quality_score lookup for candidate posts (<1ms)
      │    → Filter out posts with quality_score < 0.3
      ├─ Sort Mode Routing
      │    ├─ NEW: return reverse-chronological (no ML, < 5ms)
      │    ├─ HOT: compute Hot formula (no ML, < 5ms)
      │    ├─ TOP: fetch Spark-aggregated scores from Redis (< 5ms)
      │    └─ BEST: full ML pipeline →
      │         ├─ Redis: user affinity + interest embedding (2ms)
      │         ├─ Redis: vote velocity + comment rate for candidates (3ms)
      │         └─ LightGBM scoring (5ms for 200 candidates)
      ├─ Controversy Handling
      │    └─ Posts with controversy_score > 0.7: add "Controversial" label + allow but deprioritize by 5 positions
      ├─ Subreddit Diversity Injection (home feed only)
      │    └─ Ensure ≥ 3 distinct subreddits in top-10 posts
      └─ Response + async Kafka log
```

#### Latency Budget Breakdown

| Step | p50 | p99 | Notes |
|---|---|---|---|
| Spam pre-filter (Redis) | < 1ms | 2ms | Batch Redis lookup for candidate posts |
| User feature fetch (Redis) | 2ms | 5ms | Affinity vector + interest embedding |
| Post feature fetch (Redis) | 2ms | 8ms | Vote velocity + comment rate per candidate |
| LightGBM scoring (200 candidates) | 3ms | 8ms | CPU; fast tabular model |
| Diversity injection | 1ms | 3ms | CPU arithmetic |
| Network + serialization | 5ms | 15ms | gRPC |
| **Total (Best sort)** | **14ms** | **41ms** | Budget: ≤ 150ms p99 ✓ |

#### Caching Strategy
- **Hot/Top cached scores:** Pre-computed per subreddit every 5 min (Spark hot job); served from Redis; no per-user computation
- **User affinity cache:** Redis; TTL 1 hour; invalidated on new upvote or subscription event
- **Spam score cache:** Redis; TTL 15 min; rebuilt by spam classifier Flink pipeline

---

### 3c. Monitoring

#### Drift Detection

| Signal | Method | Threshold | Action |
|---|---|---|---|
| Spam pre-filter hit rate | Rolling 24hr | > 2× baseline → vote manipulation attack | Alert Trust & Safety; tighten quality threshold |
| Subreddit diversity (home feed Gini) | Daily | Gini > 0.6 (top 5 subreddits > 60% of slots) | Increase diversity injection; alert product team |
| Vote velocity feature distribution | PSI daily | PSI > 0.25 on velocity features | Check Flink pipeline health; possible event lag |
| Best sort engagement rate vs. Hot sort | Weekly comparison on same user cohort | Best engagement < Hot by > 5% | ML ranking worse than heuristic → investigate feature importance |
| NDCG@10 on held-out users | Weekly offline eval | < 0.25 | Retrain trigger |
| Community health (unique subreddits per user/week) | Weekly | < 5 unique subreddits for median user | Echo chamber forming; strengthen diversity injection |

#### Shadow Scoring
- Challenger LightGBM runs on 5% of users (logged, not served)
- Comparison: NDCG@10 + community diversity Gini; promote if +1% NDCG with no community health regression

#### A/B Holdout Design
- **Unit:** `user_id`
- **Holdout:** 5% permanent holdout (Hot-only ranking; no personalization)
- **Treatment:** Best (ML personalized) vs. holdout (Hot heuristic)
- **Primary metric:** Sessions per day per user (personalization should increase return visits)
- **Guardrail metrics:** Community health Gini, spam filter hit rate, p99 latency
- **Note:** Reddit also runs the ML vs. heuristic experiment for new users (< 7 days tenure) to measure cold-start effectiveness

---

## 4. Failure Modes

| Scenario | Impact | Mitigation | Fallback |
|---|---|---|---|
| Spam classifier pipeline lag (> 15 min) | Spam posts may enter ranking before filter | Monitor Flink consumer lag; alert at > 15 min lag; use last-valid Redis cache | Tighten spam filter threshold conservatively; flag all borderline posts for deferred review |
| Redis (vote velocity) unavailable | Stale real-time signals | Retry ×1 (5ms); circuit breaker | Use cached velocities from 30 min ago; Hot formula still works (age-based decay) |
| LightGBM model server failure | Best sort unavailable | Auto-reload from S3 (< 2s); failover to hot standby | Serve Hot sort for all users; no personalization |
| Vote manipulation attack (coordinated brigading) | Fake viral posts injected into hot | Spike detection: if post velocity > 3σ above subreddit baseline within 5 min → quarantine for human review | Hard velocity cap: posts cannot exceed 99th percentile velocity of their subreddit in first 30 min |
| Training data gap (Flink lag > 2hr) | LightGBM not updated; gradual staleness | Monitor Kafka consumer lag; block retraining if gap detected | Champion model continues; alert data engineering |

#### Circuit Breaker Thresholds
- **Error rate:** Trip at > 1% 5xx → Hot sort for all users; alert on-call
- **Latency:** Trip at p99 > 200ms sustained 3min → disable ML ranking; serve Hot sort
- **Recovery:** Half-open after 30s; full restore on 3 consecutive successes

#### Degraded-Mode Behavior
1. **Level 1** — Best (ML personalized) — full experience
2. **Level 2** — Hot sort (community-based, no personalization; always available from formula)
3. **Level 3** — New sort (reverse-chronological; zero dependencies; always available)

> **Reddit rubric:** Reddit has historically served Hot as the fallback — it's always computable from post age and net votes without any external dependencies. New is the ultimate fallback (no computation at all).

---

## 5. Capacity Estimates

> **Assumptions:**
> - DAU: 50M
> - Feed loads per DAU: 10/day
> - Peak QPS: 60K (2× average; evening US peak)
> - Candidate posts per request: 200 (from subscribed subreddits, last 48hr)
> - Kafka events: 65B/day (Reddit's documented throughput)
> - LightGBM model size: ~100MB (fast tabular; loaded in RAM)
> - Log retention: 90 days hot (Hive), 2 years cold

#### Estimates

| Metric | Calculation | Result |
|---|---|---|
| Avg QPS | 50M × 10 / 86,400 | **~5,800 QPS** |
| Peak QPS | 5,800 × 2× | **~11,600 QPS** |
| Redis reads/s (vote velocity, user features) | 11,600 × 5 reads | **~58K reads/s** |
| Kafka ingest throughput | 65B / 86,400 | **~750K events/s** (documented; matches 65B/day) |
| Flink workers (vote velocity aggregation) | 750K events/s / 10K per worker | **~75 Flink workers** |
| Vote velocity Redis storage | 4M active posts/day × 3 windows × 8B | **~100MB** (trivially small) |
| User affinity storage (Redis) | 50M users × ~100 subreddits × 4B score | **~20GB** |
| Post content embeddings (hot posts, Redis) | 4M posts/day × 2 days × 1KB | **~8GB in Redis hot cache** |
| LightGBM serving replicas | 11,600 QPS / 3,000 QPS per CPU node | **~4 CPU nodes** (fast tabular model) |
| Training data (30 days) | 50M × 10 events × 30 × 300B | **~450GB** |

---

## 6. Open Questions & Follow-ups

#### Unresolved Design Decisions
- [ ] **Reddit ML vs. traditional SD interview boundary:** Reddit's generic SD round covers distributed systems; if asked "design Reddit's feed" in that round, the answer blends infrastructure (fan-out, sharding) with ML ranking. Confirm with recruiter which round this question falls in
- [ ] **Cross-subreddit interest transfer:** A user active in r/Python moving to r/MachineLearning — can their programming subreddit signals bootstrap ML ranking quality? Would require cross-subreddit topic embedding (e.g., subreddit2vec) and explicit transfer mechanism
- [ ] **Comments ranking:** Comment ranking within a post is a separate, interesting problem — Wilson score lower bound is Reddit's current approach; ML-based comment ranking is a future direction
- [ ] **Long-tail subreddit cold-start:** A new subreddit with < 100 members has no training signal. Should cross-subreddit content embeddings bootstrap ranking for new subreddits?

#### Company Rubric Gaps (self-assessment)
- [x] Requirements: spam filter as hard pre-filter (not soft feature); sort mode differentiation; community health as guardrail
- [x] Data Modeling: vote velocity at multiple windows; controversy detection; Kafka + Flink stack named
- [x] ML Pipeline: LightGBM for Best sort; formula-based Hot/New/Top; subreddit diversity injection
- [x] Failure Modes: Hot sort is always available fallback (formula-only; zero dependencies)
- [x] Capacity: Kafka 65B/day throughput acknowledged; 75 Flink workers calculated

#### Recommended Follow-up Problems
- Reddit Generic System Design — distributed feed architecture (fan-out on write vs. read), sharding, caching (separate round)
- Spam Detection Pipeline — two-stage classifier feeding the pre-filter in this design

---

## 7. References

| Resource | Type | Relevance |
|---|---|---|
| Reddit Engineering Blog: "How We Built r/Place" (2017) | Blog | Reddit's distributed system architecture; event streaming at scale |
| Reddit Engineering Blog: "Scaling the Reddit Community Points" | Blog | Redis usage patterns at Reddit |
| Wilson, "Probable Inference, the Law of Succession, and Statistical Inference" (1927) | Paper | Wilson score lower bound; used for Reddit Hot/Controversial sort |
| Reddit Code: Reddit's Hot Ranking Algorithm (Randall Munroe analysis) | Blog | Hot formula derivation; log10 scaling for upvote magnitude |
| Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" (NeurIPS 2017) | Paper | LightGBM for Best sort ranking |
| Amatriain et al., "Past, Present, and Future of Recommender Systems" (RecSys 2016) | Paper | Survey covering community-based ranking vs. personalized ranking |
| Chakrabarti et al., "Dynamic Personalized Ranking of Social Media Streams" (WWW 2011) | Paper | Real-time personalized ranking of social streams; directly applicable |
| Yin et al., "Silence of the Hams: Identifying Spam Posts in Social Media" (2010) | Paper | Spam detection in social media; velocity-based features |
| Aiello et al., "Friendship Prediction and Homophily in Social Media" (TWEB 2012) | Paper | Subreddit affinity modeling; community interest transfer |
| Kafka Documentation: "Confluent Platform" | Docs | Reddit uses Confluent Kafka; stream processing architecture |
