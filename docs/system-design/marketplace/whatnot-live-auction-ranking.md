# Whatnot Live Auction Feed Ranking — ML System Design

**Domain:** `marketplace`
**Target Company:** Whatnot
**Difficulty Bar:** L6 (Staff MLE)
**Date:** 2026-03-27
**Related Designs:** `../templates/ml-system-design-template.md`

---

## 0. Scorecard

| Axis | Score | Weakest Point |
|---|---|---|
| Requirements | ★★★★★ | — |
| Data Modeling | ★★★★★ | — |
| ML Pipeline | ★★★★★ | — |
| Failure Modes | ★★★★☆ | Rockset partial degradation (slow queries vs. full outage) not differentiated |
| Capacity | ★★★★★ | — |

**Overall:** `STRONG HIRE`
**Top Gap:** Rockset degradation vs. outage — if Rockset is slow (200ms queries) but not down, the system needs a latency budget override path, not just a binary circuit breaker.

---

## 1. Requirements

#### Functional Requirements
1. Rank live auction shows on the Whatnot discovery feed for each user by predicted engagement probability
2. Surface shows where user intent (bidding, watching) is highest given current live context (active bids, show momentum, seller reputation)
3. Integrate fraudulent bidding detection as a hard filter: shows with fraud signals are suppressed from recommendation
4. Ensure new seller discoverability: prevent established sellers from monopolizing all recommendation slots

#### Non-Functional Requirements

| Dimension | Target | Business Tie |
|---|---|---|
| Latency (p99 serving) | ≤ 200ms | Whatnot's documented ML serving requirement; beyond this, show discovery feels laggy during live browsing |
| Availability | 99.99% | Live auctions are time-critical; discovery downtime = sellers lose bidders mid-auction |
| Consistency | Strong for fraud filter (synchronous); Eventual for ranking (seconds) | Fraud filter must run before show appears in feed; ranking can be slightly stale |
| Throughput | ~30K peak QPS | 5M DAU × ~15 discovery requests/day / 86,400 × 3× evening peak |
| Feature freshness (live signals) | ≤ 5s | Bid activity, viewer count, and price momentum are the primary real-time signals |
| New seller discoverability | ≥ 10% of top-20 slots reserved for sellers < 6 months old | Guards against monopolization; documented Whatnot product priority |

#### Scale Numbers (stated upfront)
- **DAU / MAU:** 5M DAU / 15M+ MAU
- **Peak QPS:** ~30K
- **Live auctions simultaneously:** ~500–2,000 at any time (most activity peaks evenings and weekends)
- **GMV/year:** ~$2B+ (every second of downtime has direct financial impact)
- **Categories:** Sports cards, sneakers, Pokémon, collectibles, fashion (each with distinct price dynamics)

#### Out of Scope
- Live streaming infrastructure (video encoding, CDN delivery)
- Bidding engine and auction settlement
- Payment processing and shipping logistics
- Seller onboarding and verification

> **Whatnot rubric:** The system design round is hybrid — infrastructure (Kafka, Rockset, Redis) + ML (GBDT, inference serving). Name Whatnot's actual stack explicitly. Fraudulent bidding detection is a hard filter (same pattern as spam pre-filter in Reddit's design), not a soft ranking feature.

---

## 2. Data Modeling

#### Input Signals

| Feature | Type | Freshness | Source | Notes |
|---|---|---|---|---|
| Show bid rate (bids/min, last 5 min) | `real-time` | ≤ 5s | Kafka → KSQL → Rockset | Primary real-time engagement signal; high bid rate = hot auction |
| Show viewer count (concurrent) | `real-time` | ≤ 5s | Presence service → Kafka → Rockset | Audience size; growing viewer count = discovery opportunity |
| Current item price vs. estimated fair value | `real-time` | ≤ 5s | Auction engine → Kafka → Rockset | Price momentum: item trading below fair value = high conversion signal |
| Seller reputation score | `batch` | daily | Spark on historical ratings + dispute history → Redis | Trust signal; low reputation = deprioritized even if high bid rate |
| Seller historical show GMV (category-specific) | `batch` | daily | Spark on transaction logs → Redis | Predicts future GMV contribution; category-specific (card seller ≠ sneaker seller) |
| User category affinity (sports cards, sneakers, etc.) | `batch` | daily | Spark on user bidding history → Redis | Primary personalization feature |
| User recent bidding activity (last session) | `real-time` | ≤ 2 min | Kafka → KSQL → Redis | Session-level intent; user who just bid on cards → more card auctions |
| Fraudulent bidding score (per show) | `batch` | 15 min | Fraud detection pipeline → Redis | **Hard filter**: shows with score > threshold excluded before ranking |
| Show category and item type | `static` | at show creation | Catalog service | Category matching to user affinity |
| Seller tenure (months on platform) | `static` | at account creation | User profile | New seller boost: < 6 months → exploration injection |

#### Label Definition
- **Label (primary):** Bid event within 10 min of feed impression (positive); scroll-past as negative
- **Label (secondary):** Purchase (item won) event — highest-value conversion (weighted 10×)
- **Collection strategy:** Implicit feedback via bidding event log and purchase confirmation; bid is a strong signal (requires deliberate action + financial commitment)
- **Positive/negative ratio:** ~1:40 (most show impressions don't result in bids)
- **Label delay:** Bid is near-immediate (seconds after impression); purchase confirmed after auction close (minutes to hours)
- **Bias risks:**
  - **Price anchoring bias:** High-priced items generate more bid activity but may attract different user segments → normalize bid rate by category price tier
  - **Timing bias:** Evening/weekend shows have higher bid rates regardless of quality → time-of-day normalization in features
  - **New seller cold-start:** New sellers have no interaction history → model assigns low predicted bid rate → ε-greedy exploration required

#### Storage Engine Choice

| Layer | Engine | Justification |
|---|---|---|
| Real-time show signals (bid rate, viewer count, price) | **Rockset** (real-time analytics DB, Whatnot's documented stack) | Convergent index: real-time ingest from Kafka + low-latency analytical queries; ideal for fast-changing auction signals |
| Session-level user features | Redis (key: user_id → recent bids; TTL 30 min) | Sub-ms reads; short TTL; high write rate |
| Seller reputation + user affinity | Redis (key: seller_id / user_id → feature_vector) | Daily batch updates; fast lookup at serving |
| Fraudulent bidding scores (per show) | Redis (key: show_id → fraud_score) | Updated every 15 min; fast lookup for pre-filter |
| Feature store (offline) | Snowflake (Whatnot's documented data warehouse) | Analytical workloads; historical bidding data; ML training data |
| Training data | Parquet on S3 (via Snowflake export) | Columnar; Spark-compatible |
| Logs / labels | Kafka (Confluent Cloud) → KSQL → Snowflake | Real-time stream processing → data warehouse |
| Model artifacts | S3 + internal registry | GBDT compiled to C++ for serving (< 200ms requirement) |

#### Online vs. Offline Split

```
Offline (batch, Spark/Snowflake)                    Online (real-time, < 200ms)
────────────────────────────────────────────         ──────────────────────────────────────────────
Bid events → Kafka → KSQL → Snowflake               Request: user_id + device
Spark: user category affinity (daily)               Fraud pre-filter: Redis fraud_score per live show (<1ms)
Spark: seller reputation + GMV history (daily)      Redis: user category affinity + session bids (2ms)
Daily GBDT training (LightGBM → C++ compiled)       Rockset: live show signals (bid_rate, viewers, price) (10ms)
Fraud classifier: LightGBM on bid patterns          LightGBM C++ model: score live shows (5ms)
Weekly fairness audit: seller size distribution     New seller exploration: inject ε-greedy slots (1ms)
                                                    Response + async Kafka log
```

**Why GBDT → compiled C++:**
- Whatnot's documented architecture: train LightGBM, compile to C++ via TreeLite, serve via FastAPI or gRPC
- Compiled C++ inference: ~0.5ms for 20 features × 200 live shows = ~100μs total scoring time
- This leaves ample budget for Rockset queries (10ms) and Redis reads (2ms) within the 200ms p99 target
- Alternative (neural network) was rejected: compilation path not available, inference 10× slower

#### Schema

```
LiveShow: {
  show_id:           string
  seller_id:         string
  category:          enum[SPORTS_CARDS, SNEAKERS, POKEMON, COLLECTIBLES, FASHION, ...]
  started_at:        timestamp
  bid_rate_5min:     float           # bids per minute, last 5 min (Rockset)
  viewer_count:      int             # concurrent viewers (Rockset)
  current_item_price:float           # current bid on active item
  estimated_value:   float           # AI price estimation for item
  fraud_score:       float           # from fraud pipeline; hard filter if > 0.7
  seller_tenure_months: int
  updated_at:        timestamp       # must be ≤ 5s old at serving time
}

UserProfile: {
  user_id:           string
  category_affinity: {category: float}   # bid rate by category, normalized
  session_recent_bids: string[]          # last 5 show_ids where user bid; TTL 30 min
  updated_at:        timestamp
}

BidEvent: {
  bid_id:            string
  show_id:           string
  bidder_id:         string
  bid_amount:        float
  timestamp:         timestamp
  is_fraud:          bool?              # labeled ex-post by fraud pipeline
}
```

> **Whatnot rubric:** Rockset is named explicitly as the real-time analytics layer. GBDT → compiled C++ is Whatnot's actual inference approach. Fraudulent bidding score is a hard pre-filter (not a soft feature). Seller tenure is a first-class field for new seller exploration.

---

## 3. ML Pipeline

### 3a. Offline Training

#### Data Pipeline
- **Ingestion:** Bid events + purchase events → Kafka → KSQL (stream joins) → Snowflake; daily export to S3 Parquet for Spark training
- **Feature engineering:**
  - User category affinity: Spark daily aggregation of bid count and GMV by category per user; normalized as bid_rate/impressions
  - Seller category GMV: Spark weekly rolling average GMV per category per seller
  - Bid rate normalization: normalize by category median and time-of-day bucket to remove timing bias
  - Fraudulent bid labels: ex-post labeled by fraud pipeline; joined to training data as label signal (fraudulent bids excluded from positive labels)
- **Train/val/test split:** Time-based — train D-60 to D-1; val D-7 to D-1; test D-0 held-out shows; stratify by category and seller_tenure_bucket
- **Orchestration:** Airflow DAG; idempotent; daily retrain; model compiled to C++ after training

#### Model Architecture

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Simple popularity ranking (bid count) | < 1ms; no ML | Not personalized; monopolized by established sellers | Fallback only |
| LightGBM → compiled C++ (chosen) | < 1ms inference; personalized; fast to retrain; handles mixed feature types | Requires compilation step; no sequential modeling | **Chosen** |
| Two-tower + ranking (neural) | Better embedding interactions | Compilation not available; 10ms+ inference | Rejected (violates 200ms budget with Rockset latency) |
| Transformer over bid history | Rich sequential bidding patterns | 50ms+ inference; overkill for 2K simultaneous shows | Future direction |

**Selected: LightGBM → TreeLite compiled C++**
- **Input features:**
  - User: `user_category_affinity[show.category]`, `user_bid_rate_last7d`, `user_session_bids_overlap_with_show_category`
  - Show (real-time from Rockset): `bid_rate_5min`, `viewer_count`, `price_vs_estimated_value_ratio`, `show_age_minutes`
  - Seller: `seller_reputation_score`, `seller_category_gmv_30d`, `seller_tenure_months`
  - Context: `hour_of_day`, `day_of_week`, `category` (one-hot)
- **Objective:** Binary cross-entropy on bid event label; purchase events as 10× weighted positives
- **Compilation:** After training, compile to C++ via TreeLite; serve via gRPC endpoint (Whatnot uses FastAPI + gRPC)

**Fraudulent bidding detection (separate model, feeds the hard pre-filter):**
- Signals: bid velocity spike (anomalously high bid rate from single user), bid-then-retract pattern, new account bidding on high-value items, IP/device clustering with known fraudulent accounts
- Model: LightGBM binary classifier; output score → Redis; shows with score > 0.7 excluded from discovery feed
- Retrain: every 6 hours (fraud patterns evolve faster than recommendation patterns)

#### Training Infrastructure
- **Framework:** LightGBM (CPU) + TreeLite (C++ compilation)
- **Scale:** 4× CPU nodes, ~15 min/run (Whatnot's catalog is small — only ~500–2K live shows at peak)
- **Eval metrics:** NDCG@5 (ranking quality on held-out users), bid rate lift vs. popularity baseline, new seller slot coverage (% of top-20 slots occupied by sellers < 6 months old)

---

### 3b. Online Serving

#### Inference Path

```
User Opens Discovery Feed (iOS/Android)
  → API Gateway
  → Discovery Ranking Service
      ├─ Live Show Inventory
      │    └─ Fetch all currently live shows (max ~2K) — lightweight catalog query
      ├─ Fraud Pre-filter (mandatory, synchronous)
      │    └─ Redis: fraud_score per show (<1ms, batch lookup)
      │    → Remove shows with fraud_score > 0.7
      ├─ Feature Assembly
      │    ├─ Redis: user category affinity + session bids (2ms)
      │    └─ Rockset: live signals (bid_rate, viewer_count, price_ratio) for filtered shows (10ms)
      ├─ LightGBM C++ Scoring
      │    └─ Score all eligible shows (~200–2K) (< 2ms, compiled C++)
      ├─ Post-ranking
      │    ├─ New seller exploration: if slot 1-20 has < 2 sellers with tenure < 6mo → inject one
      │    └─ Category diversification: max 5 same-category shows in top-20
      └─ Response: top-N ranked shows + async Kafka log
```

#### Latency Budget Breakdown

| Step | p50 | p99 | Notes |
|---|---|---|---|
| Fraud pre-filter (Redis batch) | < 1ms | 2ms | Redis MGET for all live shows |
| User feature fetch (Redis) | 2ms | 5ms | Parallel reads |
| Rockset real-time show signals | 8ms | 25ms | Convergent index; real-time Kafka-ingested data |
| LightGBM C++ scoring (~500 shows) | 1ms | 3ms | Compiled C++; extremely fast |
| Post-ranking + diversity | 1ms | 3ms | CPU arithmetic |
| Network + serialization | 5ms | 15ms | gRPC + protobuf |
| **Total** | **18ms** | **53ms** | Budget: ≤ 200ms p99 ✓ (large headroom) |

#### Caching Strategy
- **Fraud scores (Redis):** Updated every 15 min; TTL 20 min; fraud pipeline writes to Redis directly
- **User feature cache (Redis):** TTL 2 min for session features; 1 hour for category affinity
- **Rockset real-time data:** Not cached — Rockset is designed for low-latency real-time queries; caching would defeat the purpose of real-time show signals
- **Pre-scored fallback (S3):** Hourly batch job pre-scores all live shows using batch features; written to Redis; used as fallback if Rockset is unavailable

---

### 3c. Monitoring

#### Drift Detection

| Signal | Method | Threshold | Action |
|---|---|---|---|
| Bid rate feature distribution | PSI daily | PSI > 0.3 | Check KSQL pipeline; possible Kafka lag |
| Fraud pre-filter hit rate | Rolling 24hr | > 3× baseline → active fraud attack | Alert Trust & Safety; fraud model emergency retrain |
| New seller slot coverage | Daily audit | < 8% of top-20 slots for sellers < 6 months | Increase ε-greedy exploration rate |
| Model score distribution | Rolling mean ± 2σ | > 25% shift sustained 1hr | Shadow model comparison; check Rockset freshness |
| Rockset query latency | Real-time APM | > 50ms p99 sustained 2min | Alert + fallback to Redis pre-scored cache |
| Bid rate lift (ML vs. popularity baseline) | Weekly A/B holdout | ML < +5% bid rate vs. popularity | Model underperforming; retrain or revisit features |

#### Shadow Scoring
- Challenger model (e.g., improved seller reputation features) runs on 5% of users; bid rate logged
- Comparison: daily NDCG@5 + bid rate + new seller coverage; promoted after 7 days
- Whatnot's small scale means experiments reach statistical significance faster (~3–5 days vs. 2 weeks at Netflix scale)

#### A/B Holdout Design
- **Unit:** `user_id`
- **Holdout:** 5% permanent holdout (popularity ranking = bid_rate_5min descending, no personalization)
- **Primary metric:** Bids placed per discovery session (not impressions — measures actual conversion)
- **Guardrail metrics:** Fraud pre-filter breach rate, new seller slot coverage, p99 latency, GMV per session
- **Duration:** Minimum 1 week (accounts for weekly auction schedule patterns — Sunday night is highest-traffic evening)

---

## 4. Failure Modes

| Scenario | Impact | Mitigation | Fallback |
|---|---|---|---|
| Rockset slow (> 50ms queries, not down) | Latency budget exceeded before fraud filter | Rockset query timeout (30ms); circuit breaker at p99 > 50ms | Redis pre-scored cache (hourly batch; no real-time bid_rate signal); show pre-scored list |
| Rockset fully unavailable | No real-time show signals | Redis pre-scored fallback (immediate); alert Rockset ops | Popularity ranking (bid_rate from Redis, last update ≤ 15 min) |
| Redis (user features + fraud scores) unavailable | No personalization + fraud filter cannot run | Retry ×1 (5ms timeout); circuit breaker | **If fraud filter unavailable: block all discovery recommendations entirely** (fail-safe for fraud); alert ops immediately |
| Fraud model pipeline lag (> 30 min) | Fraudulent shows may appear in discovery | Alert at 15 min lag; tighten fraud filter threshold conservatively | Block any show with high bid_rate from accounts < 7 days old (rule-based proxy for fraud) |
| LightGBM C++ binary corrupted | Model scoring unavailable | Auto-reload from S3 (< 2s); TreeLite compilation is deterministic → re-compile if needed | Popularity ranking (bid_rate_5min descending) |
| New seller exploration injection bug | New sellers not surfaced | Injection runs as isolated post-processing step; failure isolated; fallback to ranked list without injection | Alert creator ops; manual seller spotlight feature (editorial) |

#### Circuit Breaker Thresholds
- **Error rate:** Trip at > 1% 5xx → popularity ranking fallback; alert on-call
- **Rockset latency:** Trip at p99 > 50ms sustained 1min → Redis pre-scored fallback (graceful degradation, not full fallback)
- **Fraud filter unavailable:** Immediately stop discovery recommendations (fail-safe); this is non-negotiable
- **Recovery:** Half-open after 20s; restore full pipeline on 3 successes

#### Degraded-Mode Behavior
1. **Level 1** — Full ML: personalized GBDT + Rockset real-time + fraud filter + seller exploration
2. **Level 2** — ML with stale show signals: GBDT + Redis pre-scored cache (no real-time bid_rate; up to 1hr stale)
3. **Level 3** — Popularity ranking: bid_rate_5min descending (Rockset query or Redis cache); no personalization
4. **Level 0 (safety)** — No discovery: if fraud filter is unavailable → no discovery feed; redirect users to their seller following list only

> **Whatnot rubric:** Level 0 (no discovery) is explicitly defined. Fraud filter unavailability cannot fall back to "show all shows anyway" — that's an unacceptable fraud risk for a marketplace handling $2B+ GMV.

---

## 5. Capacity Estimates

> **Assumptions:**
> - DAU: 5M
> - Discovery requests per DAU: 15/day (live platform; frequent refresh during browsing sessions)
> - Peak QPS: 30K (3× average; Sunday evenings)
> - Live shows simultaneously: max ~2K at peak
> - GBDT model: ~50MB → ~5MB compiled C++ binary
> - Rockset: convergent index; real-time Kafka ingest; supports ~100K QPS per cluster
> - Log retention: 2 years (financial records requirement)

#### Estimates

| Metric | Calculation | Result |
|---|---|---|
| Avg QPS | 5M × 15 / 86,400 | **~870 QPS** |
| Peak QPS | 870 × 3× | **~2,600 QPS** (well within 30K capacity target) |
| Rockset queries/s | 2,600 QPS × 1 query (batch fetch for ~500 shows) | **~2,600 queries/s** |
| Redis reads/s | 2,600 × 3 reads (user + fraud + seller) | **~7,800 reads/s** (trivial) |
| Fraud score storage (Redis) | 2K live shows × 8B | **~16KB** (effectively zero) |
| User feature storage (Redis) | 5M users × 1KB | **~5GB** |
| Kafka ingest (bid events) | Peak: ~10K concurrent bidders × 2 bids/min | **~340 events/s → ~30MB/hr** (very small vs. Reddit/Meta scale) |
| Snowflake training data (60 days) | ~300M bid events × 200B | **~60GB** (small dataset; fast training) |
| Serving replicas | 2,600 QPS / 5,000 QPS per node (C++ is CPU-efficient) | **~1 node** (with 10× headroom: 3 nodes for HA) |
| Annual GMV per QPS | $2B / 870 avg QPS | **~$2.3M GMV per QPS** — illustrates why 200ms SLA matters |

---

## 6. Open Questions & Follow-ups

#### Unresolved Design Decisions
- [ ] **Price estimation model:** Current design uses `price_vs_estimated_value_ratio` as a feature. The estimated fair value comes from a separate price estimation model (comps-based: what did similar items sell for?). That model is a second ML system design problem worth prepping separately
- [ ] **Category-specific models:** Sports cards and fashion have very different bid dynamics. Should we train separate LightGBM models per category? Trade-off: better accuracy per category vs. cold-start for new categories
- [ ] **Live video quality signals:** High production quality (good lighting, engaging commentary) predicts viewer retention. Computer vision features on live video frames would improve ranking but require significant infra investment
- [ ] **Seller fairness and long-term platform health:** Optimizing for bid rate may concentrate successful sellers (rich get richer). Should there be a seller Gini coefficient guardrail similar to Roblox's creator equity metric?

#### Company Rubric Gaps (self-assessment)
- [x] Requirements: Rockset + Redis + Kafka + KSQL + GBDT → C++ named explicitly; fraud filter as hard pre-filter
- [x] Data Modeling: Whatnot's actual stack (Confluent Kafka, KSQL, Snowflake, Rockset, LightGBM) used throughout
- [x] ML Pipeline: GBDT → TreeLite C++ compilation; fraud model retrained every 6hr (faster than main model)
- [x] Failure Modes: Level 0 (no discovery if fraud filter down) explicitly defined; Rockset slow vs. down differentiated
- [x] Capacity: GMV per QPS calculated ($2.3M) to motivate SLA seriousness

#### Recommended Follow-up Problems
- Whatnot Price Estimation — second Whatnot ML problem; predict fair market value for collectibles
- Whatnot Trust & Safety — fraudulent bidding detection model (full system design of the pre-filter used here)

---

## 7. References

| Resource | Type | Relevance |
|---|---|---|
| Whatnot Engineering Blog: "Evolving Feed Ranking at Whatnot" (Medium) | Blog | Production feed ranking architecture; GBDT → C++ compilation; feature importance |
| Whatnot Engineering Blog: "6x Faster ML Inference: Why Online > Batch" (Medium) | Blog | Online serving architecture; Rockset + Redis for real-time features; latency optimization |
| Whatnot Engineering Blog: "Feeds with Real-time Signals" (Medium) | Blog | KSQL + Kafka for real-time feature computation; convergent index via Rockset |
| Whatnot Engineering Blog: "How Whatnot Utilizes Generative AI for Trust & Safety" (Medium) | Blog | Fraudulent bidding detection; LLM-enhanced multimodal moderation |
| Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" (NeurIPS 2017) | Paper | LightGBM foundation; fast CPU training and inference |
| Mu et al., "TreeLite: A Prediction Library for Decision Tree Ensembles" | Docs/Tool | TreeLite C++ compilation of LightGBM; critical to Whatnot's < 200ms architecture |
| Rockset Documentation: "Convergent Indexing" | Docs | Rockset's architecture; real-time Kafka ingest + analytical query; latency characteristics |
| Confluent Documentation: "KSQL: Streaming SQL for Apache Kafka" | Docs | KSQL for real-time bid aggregation; sliding window feature computation |
| Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System" (KDD 2016) | Paper | XGBoost context; comparison with LightGBM for tabular data |
| Akbani et al., "Applying Support Vector Machines to Imbalanced Datasets" (ECML 2004) | Paper | Class imbalance handling; applicable to 1:40 bid-rate imbalance |
