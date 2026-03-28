# Netflix Homepage Recommendation — ML System Design

**Domain:** `ranking`
**Target Company:** Netflix
**Difficulty Bar:** L6 (E6)
**Date:** 2026-03-27
**Related Designs:** `../templates/ml-system-design-template.md`

---

## 0. Scorecard

| Axis | Score | Weakest Point |
|---|---|---|
| Requirements | ★★★★★ | — |
| Data Modeling | ★★★★☆ | Cold-start for new titles is partially addressed |
| ML Pipeline | ★★★★★ | — |
| Failure Modes | ★★★★☆ | Geographic failover not explicitly addressed |
| Capacity | ★★★★★ | — |

**Overall:** `STRONG HIRE`
**Top Gap:** Cold-start new title strategy and multi-region failover — probe these first in a real interview.

---

## 1. Requirements

#### Functional Requirements
1. Personalized homepage row ordering (Top Picks, Continue Watching, Trending Now, etc.)
2. Within-row title ranking per user × context (device, time of day)
3. Cold-start handling for new users (no history) and new titles (no interactions)
4. Near-real-time response to session-recent viewing behavior (last 1–3 titles watched)

#### Non-Functional Requirements

| Dimension | Target | Business Tie |
|---|---|---|
| Latency (p99 serving) | ≤ 100ms | Netflix research: engagement drops measurably beyond this threshold |
| Availability | 99.99% | Homepage unavailability is a direct subscriber churn signal |
| Consistency | Eventual (minutes) | Stale recs within 5 min are acceptable; strong consistency would add latency |
| Throughput | ~17,400 peak QPS | 100M DAU × 5 loads/day / 86,400 × 3× evening peak |
| Feature freshness (user) | ≤ 5 min | Session-level recency drives click-through on recently-started titles |
| Feature freshness (item) | ≤ 1 hour | Trending/popularity signals; not real-time |

#### Scale Numbers (stated upfront)
- **DAU / MAU:** 100M DAU / 270M+ members (~37% daily active)
- **Peak QPS:** ~17,400 (8–10 PM local — rolling across time zones)
- **Events/day:** ~1B interactions (plays, completions, skips, thumbs)
- **Catalog:** ~15K titles globally (varies by region and licensing)

#### Out of Scope
- Search ranking (separate system with different latency and recall requirements)
- Social features (shared profiles, friend activity)
- Content acquisition and licensing decisions
- Regional content availability logic (licensing layer sits above recommendation)

> **Netflix rubric:** Every NFR above is tied to a user or business metric — latency → engagement, availability → churn, freshness → click-through. The SLA is not arbitrary.

---

## 2. Data Modeling

#### Input Signals

| Feature | Type | Freshness | Source | Notes |
|---|---|---|---|---|
| User viewing history embedding | `batch` | 1 hour | Spark job on interaction logs → Delta Lake | Dense 256-d vector; captures long-term taste |
| Session-recent interactions (last 3 titles) | `real-time` | < 5 min | Flink stream → Redis | Most predictive of immediate intent |
| User genre affinity vector | `batch` | daily | Offline aggregation pipeline | Sparse; used in DCN cross features |
| Device type | `real-time` | request-time | Request context | TV vs. mobile → different row layouts |
| Time of day, day of week | `real-time` | request-time | Request context | Weeknight vs. weekend viewing patterns differ |
| Title embedding (visual + metadata) | `batch` | 24 hours | Offline embedding pipeline (CLIP + NLP) | Content-based; enables cold-start for new titles |
| Title global popularity (decay-weighted) | `batch` | 1 hour | Spark aggregation on play events | Exponential decay: recent plays weighted 10× vs. 7-day-old |
| Title freshness score | `batch` | 1 hour | Release date + engagement curve | New titles get exploration boost for first 2 weeks |
| Completion rate by user × title type | `batch` | daily | Offline aggregation | Proxy for quality signal; guards against clickbait |

#### Label Definition
- **Label:** Play event (binary) as primary; completion rate (continuous) as secondary
- **Collection strategy:** Implicit feedback via interaction event log (Kafka → Flink → Delta Lake); explicit thumbs up/down is available but sparse (~2% of plays)
- **Positive/negative ratio:** ~1:20 (most impressions do not result in play)
- **Label delay:** Play is immediate (< 1s); completion rate observed ~40 min after play start
- **Bias risks:**
  - **Position bias** — titles shown first get more plays regardless of quality; mitigate with inverse propensity scoring (IPS) weighting in training loss
  - **Exposure bias** — only shown titles generate feedback; address with exploration (ε-greedy row injection or Thompson sampling for new titles)
  - **Survivorship bias** — users who churned generate no labels; their taste is underrepresented

#### Storage Engine Choice

| Layer | Engine | Justification |
|---|---|---|
| Feature store (online, user) | Redis Cluster | Sub-ms reads; TTL-based expiry for session features; horizontal sharding by `user_id % N` |
| Feature store (online, item) | DynamoDB | Low-latency point reads; item catalog is write-rare (hourly batch updates) |
| Feature store (offline) | Delta Lake on S3 | ACID transactions; time-travel queries for reproducible training data cuts |
| Training data | Parquet on S3 (date-partitioned) | Columnar; efficient for Spark jobs; partition pruning by date range |
| Logs / labels | Kafka → Flink → Delta Lake | Streaming ingest for low-latency label availability; batch join for training |
| Model artifacts | S3 with versioned path (`s3://models/<name>/<version>/`) | Cheap; supports champion/challenger rollback in < 1 min |

#### Online vs. Offline Split

```
Offline (batch, Spark/Metaflow DAGs)          Online (real-time, < 5ms budget)
──────────────────────────────────────        ──────────────────────────────────────
Raw events → Flink → Delta Lake               Request arrives (user_id + context)
Spark: compute user history embeddings        Redis fetch: user embedding + session (2ms)
Spark: compute item embeddings (CLIP+NLP)     DynamoDB fetch: item metadata (3ms)
Spark: aggregate popularity + freshness       FAISS ANN search: 1K candidates (5ms)
Daily training runs (retrieval + ranking)     DCN-v2 batch ranking: score 1K (20ms)
Evaluation: Recall@1K, NDCG@10               Post-processing: diversity + rules (5ms)
Champion/challenger promotion logic           Response → async Kafka log
```

**Consistency note:** User embeddings may be up to 1 hour stale in online serving vs. offline training. This is acceptable — the session-recent signals (< 5 min, real-time) carry the most predictive weight for short-term intent.

#### Schema (key entities)

```
User: {
  user_id:              string       # partition key in Redis
  history_embedding:    float[256]   # dense vector, updated hourly
  genre_affinity:       float[20]    # sparse genre vector, updated daily
  session_recent:       string[3]    # last 3 title_ids, updated < 5 min
  country:              string       # for regional catalog filtering
  updated_at:           timestamp
}

Title: {
  title_id:             string       # partition key in DynamoDB
  content_embedding:    float[256]   # CLIP + NLP embedding, updated daily
  genre_tags:           string[]     # multi-label
  language:             string
  maturity_rating:      string
  popularity_score:     float        # decay-weighted, updated hourly
  freshness_score:      float        # release-date curve, updated hourly
  updated_at:           timestamp
}

InteractionEvent: {
  user_id:              string
  title_id:             string
  event_type:           enum[PLAY, SKIP, COMPLETE, THUMB_UP, THUMB_DOWN]
  position_rank:        int          # position in row at time of impression (for IPS)
  device_type:          string
  timestamp:            timestamp
}
```

> **Netflix rubric:** Online/offline feature consistency is explicitly addressed above. Stale offline features are a known failure mode — session-recent signals in Redis are the real-time correction layer.

---

## 3. ML Pipeline

### 3a. Offline Training

#### Data Pipeline
- **Ingestion:** Kafka (interaction events) → Flink (stream processing, deduplication) → Delta Lake (raw event log)
- **Feature engineering:**
  - User history embedding: mean-pool last 500 watched title embeddings (Spark job, hourly)
  - Popularity score: exponential decay `score = Σ plays × e^(-λt)`, λ = 0.1/day (Spark, hourly)
  - Negative sampling: random negatives from catalog (1:20 ratio) + hard negatives from impressed-but-not-played
- **Train/val/test split:** Time-based — train on D-30 to D-1, val on D-1 to D-0 (yesterday), test on D-0 (today's holdout). Never shuffle across time — prevents leakage.
- **Pipeline orchestration:** Metaflow DAG; idempotent by design (re-running same date produces same output). Retry semantics: auto-retry ×3 on transient failures; alert on-call on 3rd failure.

#### Model Architecture

Two-stage pipeline: retrieval narrows the search space; ranking maximizes precision on the shortlist.

**Stage 1 — Candidate Retrieval**

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Matrix Factorization (MF) | Simple, fast to train | No cold-start; no context features | Rejected |
| Two-tower (chosen) | Cold-start via content embeddings; context-aware; ANN-compatible | Two inference calls; embedding freshness dependency | **Chosen** |
| BM25 / text retrieval | No training required | Not personalized; no behavioral signal | Rejected |

**Selected:** Two-tower model
- **User tower:** `[history_embedding, genre_affinity, session_recent_embeddings, device_type, hour_of_day]` → 2-layer MLP → 256-d user vector
- **Item tower:** `[content_embedding, genre_tags, maturity_rating, popularity_score]` → 2-layer MLP → 256-d item vector
- **Training objective:** In-batch softmax; batch negatives + hard negatives from impressed-but-skipped
- **Serving:** Item vectors pre-computed and indexed in FAISS (IVF-PQ, d=256); user vector computed at request time; ANN search returns top-1000

**Stage 2 — Ranking**

| Option | Pros | Cons | Decision |
|---|---|---|---|
| LightGBM | Fast inference; interpretable | No deep feature crosses; no embedding inputs | Rejected |
| Deep Cross Network v2 (chosen) | Explicit + implicit feature crosses; handles sparse + dense | More complex; GPU serving required | **Chosen** |
| Transformer over user history | Rich sequential modeling | 10–100× inference cost; unacceptable for 1K candidates | Rejected for now; revisit for top-50 re-ranking |

**Selected:** DCN-v2 (Deep & Cross Network)
- **Input:** Concatenation of user features + item features + user×item interaction features (e.g., user_genre_affinity · item_genre_tags dot product)
- **Multi-task heads:**
  - Head 1: P(play) — primary objective
  - Head 2: P(completion rate > 80%) — quality guardrail; prevents clickbait titles from ranking first
- **Loss:** Weighted sum: `L = 0.7 × BCE(play) + 0.3 × MSE(completion)`

#### Training Infrastructure
- **Framework:** PyTorch + FSDP — embedding tables are large (user_id embeddings for 100M users require FSDP sharding across GPUs)
- **Scale:** 16× A100 80GB GPUs; ~6 hours/full retrain; incremental daily fine-tune ~45 min
- **Mixed precision:** Yes (bfloat16) — 2× memory reduction on embedding tables; no accuracy regression observed
- **Gradient checkpointing:** Yes on ranking model — user history sequence inputs increase activation memory
- **Eval metrics:**
  - Retrieval: Recall@1000 on held-out users (target ≥ 90%)
  - Ranking: NDCG@10 on held-out impressions
  - Business proxy: offline play-rate on held-out slice (compared to current champion)

---

### 3b. Online Serving

#### Inference Path

```
Client (TV/Mobile/Web)
  → CDN (static assets cached)
  → API Gateway (auth, rate limit)
  → Recommendation Service
      ├─ Feature Fetch
      │    ├─ Redis: user_embedding + session_recent  (2ms p50)
      │    └─ DynamoDB: top-1K item metadata          (3ms p50, batched)
      ├─ Retrieval Service
      │    └─ FAISS ANN search (IVF-PQ, d=256)        (5ms p50)
      │         → returns 1,000 candidate title_ids
      ├─ Ranking Service
      │    └─ DCN-v2 batch inference (GPU)            (15ms p50 for 1K)
      │         → returns scored + ranked list
      ├─ Post-processing
      │    ├─ Diversity injection (max 3 same genre per row)
      │    ├─ Business rules (parental controls, regional availability)
      │    └─ Row assembly (Top Picks, Continue Watching, etc.)   (5ms)
      └─ Response
           └─ Async: log impression + ranks to Kafka
```

#### Latency Budget Breakdown

| Step | p50 | p99 | Notes |
|---|---|---|---|
| Feature fetch (Redis + DDB) | 3ms | 8ms | Parallel fetch; DDB batch read for items |
| Retrieval (FAISS ANN) | 5ms | 12ms | IVF-PQ; index fits in memory (~15MB) |
| Ranking (DCN-v2, 1K candidates) | 15ms | 50ms | GPU batch; TensorRT optimization applied |
| Post-processing + assembly | 5ms | 10ms | CPU-side rules |
| Network + serialization | 3ms | 10ms | gRPC + protobuf |
| **Total** | **31ms** | **90ms** | Budget: ≤ 100ms p99 ✓ |

#### Caching Strategy
- **What is cached:** Pre-computed ranked lists per user (warmed by batch job every hour)
- **Cache key:** `user_id + hour_bucket` (not per-request, to avoid cold cache on new sessions)
- **TTL:** 5 minutes (bounded by session-recent freshness budget)
- **Cache hit rate target:** ~40% (covers users who load homepage multiple times in 5 min window)
- **FAISS index:** Rebuilt hourly from DynamoDB; blue/green swap (new index built offline, atomically swapped in)

---

### 3c. Monitoring

> **Designed upfront — not as an afterthought. "How do you know it's working in prod?" is answered here.**

#### Drift Detection

| Signal | Method | Alert Threshold | Action |
|---|---|---|---|
| User embedding drift | PSI (Population Stability Index) daily | PSI > 0.2 | Auto-trigger incremental retrain |
| Play-rate shift | Rolling 24hr vs. 7-day baseline | > −3% relative sustained 2hr | Page on-call; compare to holdout |
| Prediction score distribution | Rolling mean ± 2σ on P(play) scores | > 15% shift sustained 1hr | Shadow model comparison; inspect features |
| Retrieval recall (offline) | Daily held-out eval, Recall@1000 | < 90% | Alert + retrain retrieval model |
| Label pipeline lag | Kafka consumer lag | > 2hr lag | Block retraining; alert; backfill |
| p99 latency | Real-time APM (Prometheus + Grafana) | > 100ms sustained 5min | Circuit breaker (see §4) |
| GPU utilization | Per-replica monitoring | < 20% sustained (over-provisioned) or > 95% (under-provisioned) | Scale down/up replicas |

#### Shadow Scoring
- **Setup:** Challenger model serves predictions in parallel to 1% of traffic; results logged but not shown to users
- **Comparison cadence:** Daily offline comparison on the 1% shadow slice; primary metric NDCG@10 + play-rate
- **Promotion criteria:** Challenger must show +1% relative NDCG@10 AND no regression on completion rate AND no p99 latency degradation over 7 consecutive shadow days
- **Demotion:** Challenger auto-demoted if it degrades guardrail metrics; champion resumes

#### A/B Holdout Design
- **Unit of randomization:** `user_id` (not session — avoids within-user inconsistency across devices)
- **Holdout size:** 5% permanent holdout (never receives any model update) — used as long-term guardrail against Goodhart's Law / metric inflation
- **Treatment size:** Typical experiment is 10% treatment vs. 10% control (from 90% non-holdout pool)
- **Primary metric:** Play-rate per homepage load (minimum detectable effect: +0.5% relative, 80% power)
- **Guardrail metrics:** p99 latency, error rate, 30-day retention rate (lagging signal — checked post-experiment)
- **Duration:** Minimum 2 weeks — accounts for novelty effect and weekly viewing pattern cycles

> **Netflix rubric:** Monitoring is designed before serving is built. The 5% permanent holdout is the key Netflix-specific pattern — it catches cases where optimizing proxy metrics (play-rate) diverges from true value (retention).

---

## 4. Failure Modes

| Scenario | Impact | Mitigation | Fallback |
|---|---|---|---|
| Redis unavailable | Missing user embedding + session features | Retry 1× (10ms timeout); serve stale from in-process LRU cache (10s TTL, 10K users) | Batch-precomputed ranked list from S3 (up to 1hr stale) |
| DynamoDB throttled | Missing item metadata for ranking | Exponential backoff ×2; DDB on-demand capacity auto-scales | Retrieval-only output (no ranking); serve top-1K by popularity |
| FAISS index stale or corrupt | Poor candidate quality; recall degradation | Blue/green index swap; old index kept warm for 1hr post-swap | Previous index version served; alert on stale index age > 2hr |
| Ranking model timeout (GPU OOM) | No personalized ranking | Pre-ranked lists cached per user_id (1hr TTL, warmed by hourly batch job) | Global popularity ranking by genre (no personalization) |
| Training job failure | Model does not update; gradual staleness | Auto-retry ×3; champion stays live; page on-call on 3rd failure | Champion model continues; SLA: on-call must respond within 24hr |
| Label pipeline delay (Flink lag > 2hr) | Training data gap; silent staleness | Monitor Kafka consumer lag in real time; block retrain if gap detected | Log gap; backfill when pipeline recovers; no silent data loss |
| Cold-start: new user | No history embedding | Content-based retrieval only (item tower still works); genre preferences from onboarding quiz | Country-level trending titles; prompt user to rate 3 titles |
| Cold-start: new title | No interaction signal | Content embedding (CLIP + NLP) enables ranking via item tower; freshness boost for first 2 weeks | Include in exploration bucket (ε-greedy: 10% of slots reserved for new titles) |

#### Circuit Breaker Thresholds
- **Error rate:** Trip at > 1% 5xx over 60s rolling window → route to fallback recs; alert on-call
- **Latency:** Trip at p99 > 150ms sustained 3min → bypass ranking stage; serve retrieval output only (FAISS top-N by similarity score)
- **Recovery:** Half-open after 30s; probe with 1% traffic; full restoration on 3 consecutive successes

#### Degraded-Mode Behavior
When full ML pipeline is unavailable, serve in order:

1. **Level 1** — Session-personalized cached recs (Redis, up to 5 min stale) — still personalized
2. **Level 2** — Daily batch-precomputed recs (S3, up to 1 hour stale) — personalized but not session-aware
3. **Level 3** — Country-level popularity ranking by genre (rule-based, no ML) — no personalization

> **Netflix rubric:** The full 3-level fallback chain is defined explicitly. Netflix will probe this — "what does the user see when your model is down?" is a required answer.

---

## 5. Capacity Estimates

> **Assumptions:**
> - DAU: 100M active users (of 270M+ total members)
> - Homepage loads per DAU: 5/day (mix of browsing sessions)
> - Evening peak multiplier: 3× average (8–10 PM local; rolling across time zones → sustained, not spiked)
> - Feature vector size: 256 floats × 4 bytes = 1KB per user/item entity
> - Catalog size: 15K titles
> - Model sizes: retrieval user tower 200MB, item tower 200MB; ranking model (DCN-v2) 2GB
> - Log retention: 90 days hot (Delta Lake), 2 years cold (S3 Glacier)
> - Ranking throughput: 1 A10G GPU replica handles ~500 QPS (1K candidates per request)

#### Estimates

| Metric | Calculation | Result |
|---|---|---|
| Avg QPS | 100M × 5 loads/day / 86,400 | **~5,800 QPS** |
| Peak QPS | 5,800 × 3× evening multiplier | **~17,400 QPS** |
| Redis reads/s (user features) | 17,400 QPS × 1 read/request | **~17,400 reads/s** (well within Redis Cluster capacity) |
| DynamoDB reads (items) | 17,400 QPS × 1K candidates (batched, 20 items/batch call) | **~870K batch calls/s → ~50 DDB requests/s** |
| Feature storage — users (Redis) | 100M users × 1KB/user | **~100GB** |
| Item embedding index (FAISS) | 15K titles × 256 floats × 4B | **~15MB** (trivially in-memory, replicated per serving node) |
| Daily interaction logs | 1B events × 200B/event | **~200GB/day** |
| Training data (1 year) | 200GB/day × 365 days | **~73TB** (S3; columnar compression ~3×: ~24TB effective) |
| Model artifact storage | 10 versions × 2.4GB | **~24GB** |
| Serving replicas — ranking (GPU) | 17,400 QPS / 500 QPS per A10G | **~35 A10G GPU replicas** (+ 20% headroom = 42 replicas) |
| Serving replicas — retrieval (CPU) | 17,400 QPS / 2,000 QPS per CPU node | **~9 CPU nodes** (FAISS IVF-PQ is CPU-efficient) |
| Kafka throughput (impression logs) | 17,400 requests × 1K impressions × 100B | **~1.7GB/s ingest** (manageable with 20-partition topic) |

---

## 6. Open Questions & Follow-ups

#### Unresolved Design Decisions
- [ ] **Retrieval index:** FAISS IVF-PQ vs. ScaNN — IVF-PQ is chosen here, but ScaNN offers better recall/latency tradeoff at 15K scale; revisit if catalog grows to 100K+ titles
- [ ] **Row-level ordering:** Should row type ordering (which row appears first — Top Picks vs. Continue Watching) be a separate model or a feature in the ranking model? Separate model is cleaner; same model avoids training pipeline duplication
- [ ] **New title cold-start depth:** Content embedding only handles genre/style similarity. Should we add Thompson sampling or UCB1 for exploration of truly novel title types (no close neighbors in embedding space)?
- [ ] **Multi-profile households:** Current design personalizes per `user_id`. Shared household profiles require profile-switching logic upstream — out of scope but worth flagging

#### Company Rubric Gaps (self-assessment)
- [x] Requirements: scale mechanisms named explicitly (Redis sharding, DDB on-demand scaling)
- [x] Data Modeling: online/offline feature consistency explicitly addressed (session-recent as real-time correction layer)
- [x] ML Pipeline: monitoring designed upfront; permanent holdout defined
- [x] Failure Modes: full 3-level fallback chain defined; circuit breaker thresholds stated
- [x] Capacity: all estimates have stated assumptions; serving replicas calculated

#### Recommended Follow-up Problems
- Meta News Feed Ranking — same two-stage pattern but at 10× scale with DLRM and geographically distributed serving
- Uber ETA Prediction — shifts from ranking to regression; introduces prediction intervals and geospatial features
