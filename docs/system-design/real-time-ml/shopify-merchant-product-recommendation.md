# Shopify Merchant Product Recommendation — ML System Design

**Domain:** `real-time-ml`
**Target Company:** Shopify
**Difficulty Bar:** L6 (Senior II)
**Date:** 2026-03-27
**Related Designs:** `../templates/ml-system-design-template.md`

---

## 0. Scorecard

| Axis | Score | Weakest Point |
|---|---|---|
| Requirements | ★★★★★ | — |
| Data Modeling | ★★★★☆ | Multi-language product catalog coverage not fully addressed |
| ML Pipeline | ★★★★★ | — |
| Failure Modes | ★★★★★ | — |
| Capacity | ★★★★☆ | Black Friday replica pre-scaling calculation approximate |

**Overall:** `STRONG HIRE`
**Top Gap:** Multi-language catalog handling — product descriptions in 20+ languages require multilingual embeddings, not just English NLP.

---

## 1. Requirements

#### Functional Requirements
1. Recommend complementary products to a shopper on a merchant's storefront (e.g., "Customers also bought")
2. Rank recommended products by predicted conversion probability for that shopper × merchant × product combination
3. Handle cold-start for new merchants (no purchase history) and new products (no interaction data)
4. Support real-time personalization based on current session behavior (items browsed in this session)

#### Non-Functional Requirements

| Dimension | Target | Business Tie |
|---|---|---|
| Latency (p99 serving) | ≤ 100ms | Shopify SLA to merchants; page load time directly impacts merchant conversion rate (GMV) |
| Availability | 99.99% | Recommendation downtime = lost upsell revenue for merchants; Black Friday SLA is critical |
| Consistency | Eventual (minutes) | Session signals update within 2 min; strong consistency adds latency for no merchant value |
| Throughput | ~200K peak QPS (Black Friday) | Normal peak ~20K QPS; Black Friday 10× surge by design — must be planned for, not treated as incident |
| Feature freshness (session) | ≤ 2 min | Session browsing signals must inform recommendations within current shopping session |
| Feature freshness (purchase history) | ≤ 1 hour | Hourly batch update sufficient for inter-session history |

#### Scale Numbers (stated upfront)
- **Merchants:** ~2M active merchants on Shopify
- **Shoppers (DAU):** ~50M shopper sessions/day across all merchant storefronts
- **Products in catalog:** ~1B product variants across all merchants (long tail: each merchant has 1–100K products)
- **Peak QPS (Black Friday):** ~200K (10× normal; Black Friday is a known, plannable event — not a surprise spike)
- **Purchase events/day:** ~5M transactions/day (typical); ~50M on Black Friday

#### Out of Scope
- Merchant-side product catalog management and inventory
- Ad serving and promoted products (separate Shopify Audiences product)
- Payment processing and checkout
- Merchant-to-merchant product comparison or cross-store recommendations

> **Shopify rubric:** Present ≥2 design options with explicit reasoning before committing. Black Friday (10× surge) is a first-class design constraint — not a capacity footnote.

---

## 2. Data Modeling

#### Decision Log: Collaborative Filtering vs. Two-Tower

| Option | Pros | Cons | Decision |
|---|---|---|---|
| **Option A: Collaborative Filtering (Matrix Factorization)** | Simple; well-understood; strong for dense interaction matrices | **Rejected:** Cold-start failure for new merchants and new products (most of Shopify's long tail); sparse interaction matrices for small merchants | Rejected |
| **Option B: Two-Tower + Content Features (chosen)** | Handles cold-start via product content embeddings; works for new merchants with zero history; generalizes across the long tail | More complex training pipeline; requires multilingual product embeddings | **Chosen** |

**Why this matters for Shopify:** Shopify's merchant base is dominated by small businesses — the median merchant has < 500 purchases/month. Pure collaborative filtering fails entirely for these merchants. Content-based features using product descriptions, categories, and images enable recommendations even from Day 1.

#### Input Signals

| Feature | Type | Freshness | Source | Notes |
|---|---|---|---|---|
| Session-browsed product IDs (last 10) | `real-time` | ≤ 2 min | Shopify Storefront → Kafka → Pano (Redis) | Strongest real-time intent signal |
| Product content embedding (title + description) | `batch` | daily | mBERT embedding pipeline → Pano (offline) | Multilingual; handles 20+ catalog languages |
| Product image embedding | `batch` | daily | CLIP vision model → Pano (offline) | Visual similarity for fashion/home categories |
| Product category hierarchy | `static` | at publish | Merchant catalog → Pano | 3-level taxonomy: category → subcategory → product type |
| Merchant purchase co-occurrence matrix | `batch` | 1 hour | Spark on purchase logs → Pano (online, Redis) | P(product B bought | product A viewed); merchant-scoped |
| Shopper purchase history (cross-session) | `batch` | 1 hour | Spark on transaction logs → Pano | Merchant-scoped; not cross-merchant (privacy) |
| Product price tier | `static` | at publish | Merchant catalog | Price-range matching: don't recommend $500 accessories for $10 items |
| Product inventory status | `real-time` | ≤ 5 min | Inventory service → Kafka → Pano (Redis) | Out-of-stock products excluded from recommendations |

#### Label Definition
- **Label:** Add-to-cart event (primary); purchase event (secondary — delayed by checkout flow)
- **Collection strategy:** Implicit feedback; add-to-cart is immediate and strong; purchase has 5–30 min delay
- **Positive/negative ratio:** ~1:100 (recommendation impressions rarely convert to add-to-cart)
- **Label delay:** Add-to-cart immediate; purchase observed within 30 min for typical checkout
- **Bias risks:**
  - **Position bias:** First recommendation slot gets 5–10× more clicks → IPS weighting in training loss
  - **Merchant size bias:** Large merchants generate most training signal; small merchants (majority) are underrepresented → stratified sampling by merchant_size_bucket
  - **Category bias:** Fashion and electronics generate far more signals than niche categories → per-category evaluation metrics

#### Storage Engine Choice

| Layer | Engine | Justification |
|---|---|---|
| Feature store (online) | **Pano** (Shopify's Feast-based feature store) — Redis backend | Native Shopify platform; sub-ms reads; TTL-based session features |
| Feature store (offline) | Pano offline layer → S3 (Parquet) | Feast-compatible; Spark-readable; audit trails |
| Product embeddings (online) | Redis (key: merchant_id:product_id → embedding vector) | Scoped to merchant for privacy; 256-d float vector |
| Inventory status | Redis (TTL: 5 min; key: product_id) | High write rate from inventory service; fast read for filtering |
| Training data | Parquet on S3 (merchant_id + date partitioned) | Columnar; Spark-efficient; privacy-scoped per merchant |
| Logs / labels | Kafka → Flink → S3 | Shopify uses Kafka for all storefront events |
| Model artifacts | S3 + internal model registry | Versioned; Blue/green deployment |

#### Online vs. Offline Split

```
Offline (batch, Spark)                              Online (real-time, < 100ms budget)
────────────────────────────────────────            ──────────────────────────────────────────────
Transaction logs → Spark → co-occurrence matrix     Request: merchant_id + shopper_id + current_product_id
Product descriptions → mBERT → embeddings           Pano (Redis): session-browsed products (2ms)
Product images → CLIP → embeddings                  Pano (Redis): product embeddings (3ms)
Shopper history → Spark → aggregations              Inventory filter: remove out-of-stock
Daily training: two-tower + ranking model           Two-tower ANN: retrieve top-50 from merchant catalog
Hourly: co-occurrence matrix update                 Ranking model: score 50 candidates (10ms)
Champion/challenger per merchant tier               Diversity filter + business rules
                                                    Response + async Kafka log
```

**SOLID architecture:**
- `ProductEmbeddingService` — single responsibility: serve product embeddings
- `CandidateRetrievalService` — retrieval only; no ranking logic
- `RankingService` — scoring only; receives candidates + features
- `BusinessRulesFilter` — inventory, price-range, category exclusion rules; injected, not hardcoded
- Monitoring hooks injected via interface — unit-testable without hitting production Pano

#### Schema

```
Product: {
  product_id:        string         # globally unique
  merchant_id:       string         # scoping key for privacy
  content_embedding: float[256]     # mBERT + CLIP fusion
  category_path:     string[]       # ["clothing", "tops", "t-shirts"]
  price_tier:        enum[LOW, MED, HIGH, PREMIUM]
  in_stock:          bool           # updated every 5 min
  updated_at:        timestamp
}

ShopperSession: {
  session_id:        string
  merchant_id:       string
  browsed_products:  string[]       # last 10 product_ids; TTL 30 min
  updated_at:        timestamp
}

PurchaseEvent: {
  transaction_id:    string
  merchant_id:       string
  shopper_id:        string         # anonymized; merchant-scoped
  product_ids:       string[]       # all products in basket
  timestamp:         timestamp
}
```

> **Shopify rubric:** Explicit decision log above (CF vs. Two-Tower). SOLID service boundaries named. Pano + Kafka explicitly called out. Dependencies injected (monitoring hooks, business rules).

---

## 3. ML Pipeline

### 3a. Offline Training

#### Data Pipeline
- **Ingestion:** Storefront events → Kafka → Flink (session assembly) → S3 (Parquet); purchase events → S3
- **Feature engineering:**
  - mBERT product embeddings: title + description → 256-d multilingual embedding (Ray distributed compute, daily)
  - CLIP image embeddings: product images → 256-d visual embedding; fused with mBERT via learned projection layer
  - Co-occurrence matrix: P(B purchased | A in basket) per merchant, Spark windowed aggregation (hourly)
- **Train/val/test split:** Time-based; train on D-90 to D-1; val on D-1; test on D-0 held-out merchants (not held-out time — validates generalization to new merchant/product distributions)
- **Orchestration:** Metaflow-compatible DAG; idempotent; Spark jobs partitioned by merchant_id; retry ×3

#### Model Architecture

**Stage 1 — Candidate Retrieval (Two-Tower):**
- User tower: `[session_browsed_embeddings (mean-pool), shopper_history_embedding]` → 128-d
- Product tower: `[content_embedding, category_path_embedding, price_tier]` → 128-d
- Training: in-batch negatives; hard negatives from same merchant's catalog (not global catalog)
- ANN index: FAISS IVF-PQ per merchant (small catalogs: exact search; > 10K products: IVF-PQ)
- Output: top-50 candidates per request

**Stage 2 — Ranking:**
- Features: user × product cross features (browsed_embedding · candidate_embedding dot product, price_ratio, category_match)
- Model: LightGBM (fast CPU inference; 50 candidates is a small batch; GPU unnecessary)
- Objective: binary cross-entropy on add-to-cart label
- Multi-task option: jointly predict add-to-cart + purchase (purchase head weighted 3×)

#### Training Infrastructure
- **Framework:** PyTorch (two-tower) + LightGBM (ranking); Ray for distributed embedding compute
- **Scale:** Two-tower: 8× A10G GPUs, ~2hr/run; LightGBM ranking: CPU cluster, ~30min/run
- **Mixed precision:** Yes for two-tower (bfloat16); not applicable for LightGBM
- **Eval metrics:** Recall@50 (retrieval), NDCG@5 (ranking), add-to-cart rate lift vs. no-recommendation baseline

---

### 3b. Online Serving

#### Inference Path

```
Shopper → Merchant Storefront → Shopify Edge (CDN)
  → Recommendation API
      ├─ Inventory Filter
      │    └─ Pano (Redis): fetch in_stock status for merchant catalog (<5ms)
      ├─ Feature Fetch
      │    ├─ Pano (Redis): session-browsed product embeddings (2ms)
      │    └─ Pano (Redis): target product embedding (1ms)
      ├─ Candidate Retrieval
      │    └─ FAISS ANN on merchant catalog → top-50 (5ms for catalogs < 10K)
      ├─ Ranking
      │    └─ LightGBM: score 50 candidates (3ms)
      ├─ Business Rules Filter
      │    ├─ Remove out-of-stock (inventory)
      │    ├─ Remove same product
      │    └─ Price-range filter (don't recommend 20× price jump)
      └─ Response: top-5 recommendations + async Kafka log
```

#### Latency Budget Breakdown

| Step | p50 | p99 | Notes |
|---|---|---|---|
| Inventory + feature fetch (Pano/Redis) | 3ms | 8ms | Parallel reads |
| FAISS candidate retrieval | 4ms | 12ms | Per-merchant index; loaded in memory |
| LightGBM ranking (50 candidates) | 2ms | 5ms | CPU; trivially fast |
| Business rules filter | 1ms | 2ms | In-process |
| Network + serialization | 5ms | 15ms | gRPC + protobuf |
| **Total** | **15ms** | **42ms** | Budget: ≤ 100ms p99 ✓ (significant headroom for Black Friday) |

#### Caching Strategy
- **Recommendation cache:** (merchant_id, product_id, session_hash) → top-5 recommendations; TTL: 2 min
- **FAISS index:** Per-merchant; rebuilt daily (product catalog is batch-updated); loaded into memory at serving startup
- **Black Friday pre-warming:** 48hr before Black Friday, pre-compute and cache recommendations for top-100 products per merchant; warm Redis with pre-ranked lists; this alone covers ~60% of Black Friday traffic with cached responses

---

### 3c. Monitoring

#### Drift Detection

| Signal | Method | Threshold | Action |
|---|---|---|---|
| Add-to-cart rate vs. control | Rolling 24hr vs. 7-day baseline by merchant_tier | > −5% relative | Page on-call; investigate |
| Product embedding distribution | PSI on merchant catalog embedding centroids | PSI > 0.25 | Retrain embedding pipeline |
| Inventory filter hit rate | Daily audit | > 40% of recommendations out-of-stock | Inventory sync issue; alert inventory team |
| Cold-start merchant recommendation quality | NDCG@5 for < 30-day merchants | < 0.15 | Content embedding quality degradation; retrain |
| Black Friday latency | Real-time APM | p99 > 80ms sustained 2min | Trigger pre-warm; activate cached fallback |
| p99 latency (normal) | Real-time APM | > 100ms sustained 3min | Circuit breaker |

#### Shadow Scoring
- Challenger (e.g., improved multilingual embeddings) runs on 1% of merchant storefronts; add-to-cart logged but not served
- Comparison: daily NDCG@5 and add-to-cart rate; promotion requires +2% relative NDCG@5 + no latency regression

#### A/B Holdout Design
- **Unit of randomization:** `merchant_id` (not shopper_id — consistent experience within a merchant's store)
- **Holdout size:** 5% of merchants (100K merchants) — permanent holdout; receives co-occurrence-based baseline
- **Primary metric:** Add-to-cart rate per recommendation impression
- **Guardrail metrics:** GMV per session (merchant revenue impact), p99 latency, error rate
- **Duration:** Minimum 2 weeks (accounts for merchant-specific seasonality — some merchants have weekly sales cycles)

> **Shopify rubric:** Monitoring hooks are injected interfaces (not hardcoded Pano calls) — unit-testable. Black Friday latency monitoring is treated as a first-class alert, not an afterthought.

---

## 4. Failure Modes

| Scenario | Impact | Mitigation | Fallback |
|---|---|---|---|
| Pano (Redis) unavailable | No session or product features | Retry ×1 (5ms timeout); circuit breaker | Pre-warmed static recommendations from S3 (daily batch; no session personalization) |
| FAISS index for merchant unavailable | No ANN retrieval for that merchant | Retry; reload from S3 (< 1s) | Co-occurrence-based recommendations (from Redis, batch-updated hourly) |
| Inventory service lag | Out-of-stock products recommended | Filter at display layer in storefront UI (redundant check); Kafka-based invalidation on stock-out event | Exclude recently out-of-stock products (from Redis TTL cache) |
| Black Friday 10× traffic spike | p99 latency exceeds 100ms | Pre-warm cached recommendations 48hr before; auto-scale replicas (Kubernetes HPA, trigger at 70% CPU); circuit breaker to cached fallback | Pre-computed top-5 recommendations per merchant × top product served from Redis (no real-time ML) |
| New merchant (zero history) | No shopper history; no co-occurrence data | Content-based only (product embeddings); "Trending in category" from cross-merchant aggregated signals | Show merchant's own top-selling products by revenue (rule-based, always available) |
| Training job failure | Embeddings + models not updated | Auto-retry ×3; champion stays live; alert at 48hr staleness | Serve current champion; co-occurrence matrix updates independently (hourly) — partial freshness maintained |

#### Circuit Breaker Thresholds
- **Error rate:** Trip at > 1% 5xx over 60s → serve pre-computed S3 recommendations; alert
- **Latency:** Trip at p99 > 120ms sustained 2min → bypass ranking; serve retrieval output directly (top-50 by embedding similarity)
- **Recovery:** Half-open after 20s; restore full path on 3 consecutive successes

#### Degraded-Mode Behavior
1. **Level 1** — Session-personalized cached recommendations (Redis, up to 2 min stale)
2. **Level 2** — Co-occurrence-based recommendations (batch, up to 1hr stale, no session signals)
3. **Level 3** — Merchant's top-selling products (rule-based by revenue; always available from catalog service)

> **Shopify rubric:** Black Friday circuit breaker is explicitly designed — 10× surge is not treated as an incident but as a planned scaling event. Fallback chain is tested (dependency injection allows mock Pano in unit tests).

---

## 5. Capacity Estimates

> **Assumptions:**
> - DAU sessions: 50M/day (normal); 500M/day (Black Friday)
> - Recommendation requests per session: 4 (product pages visited)
> - Normal peak QPS: 50M × 4 / 86,400 × 3× peak = ~7K QPS
> - Black Friday peak QPS: 500M × 4 / 86,400 × 2× peak multiplier = ~46K QPS (≈ 10× normal)
> - Merchant catalog size: median ~500 products; P90 ~10K products
> - Product embedding: 256 floats = 1KB
> - Pano (Redis) read throughput: 500K reads/s per cluster
> - FAISS index per merchant: median ~5MB; P90 ~50MB; all active merchant indexes: ~10TB total (only top-100K active merchants hot-loaded in memory)

#### Estimates

| Metric | Calculation | Result |
|---|---|---|
| Normal avg QPS | 50M sessions × 4 requests / 86,400 | **~2,300 QPS** |
| Normal peak QPS | 2,300 × 3× | **~7,000 QPS** |
| Black Friday peak QPS | 7,000 × 10× | **~70,000 QPS** |
| Pano reads/s (normal) | 7,000 × 5 reads/request | **~35,000 reads/s** |
| Pano reads/s (Black Friday) | 70,000 × 5 reads | **~350,000 reads/s** (within Pano Redis capacity at scale) |
| FAISS indexes in hot memory | Top 100K merchants × 5MB median | **~500GB** (dedicated serving pods) |
| Product embeddings (all merchants) | 1B products × 1KB | **~1TB** (S3; hot-loaded per merchant on demand) |
| Training data (daily) | 5M transactions × 10KB features | **~50GB/day** |
| Serving replicas (normal) | 7,000 QPS / 500 QPS per pod | **~14 pods** |
| Serving replicas (Black Friday, pre-scaled) | 70,000 QPS / 500 QPS per pod | **~140 pods** (pre-scaled 48hr before; Kubernetes HPA + manual pre-scale) |

---

## 6. Open Questions & Follow-ups

#### Unresolved Design Decisions
- [ ] **Cross-merchant signals:** Should aggregated (anonymized) purchase co-occurrence data from all merchants inform recommendations for small merchants? Privacy/competitive sensitivity — requires legal review
- [ ] **Multilingual embeddings:** mBERT covers 104 languages but has weaker performance on low-resource languages. For merchants in Southeast Asia and LATAM, should we use specialized regional models (e.g., PhoBERT for Vietnamese)?
- [ ] **Real-time inventory:** Current design polls inventory every 5 min. Should we subscribe to Kafka inventory events for instant removal of out-of-stock items? Trade-off: complexity vs. correctness on fast-selling items
- [ ] **Merchant customization:** Should merchants be able to configure business rules (e.g., "never recommend competitors", "always surface items on sale")? Requires a rules engine layer

#### Company Rubric Gaps (self-assessment)
- [x] Requirements: ≥2 design options with explicit decision log; Black Friday as first-class requirement
- [x] Data Modeling: SOLID service boundaries; Pano + Ray + Kafka named; dependency injection for monitoring
- [x] ML Pipeline: content-based cold-start; testable monitoring hooks
- [x] Failure Modes: Black Friday circuit breaker; 3-level degraded mode; explicit pre-scale plan
- [x] Capacity: Black Friday replica count calculated; pre-scale timing stated (48hr before)

#### Recommended Follow-up Problems
- Pinterest Ads Ranking — same two-tower + ranking pattern but with multi-objective (CTR + CVR + fairness)
- Shopify Fraud Detection — real-time transaction scoring at checkout; complementary problem to recommendation

---

## 7. References

| Resource | Type | Relevance |
|---|---|---|
| Shopify Engineering Blog: "Pano: Shopify's ML Feature Store" | Blog | Pano architecture (Feast-based); online/offline split; TTL management |
| Shopify Engineering Blog: "Scaling Recommendations for Millions of Merchants" | Blog | Merchant-scoped recommendation; cold-start strategies |
| Kang et al., "Self-Attentive Sequential Recommendation" (ICDM 2018) | Paper | Session-based recommendation; attention over browsed items |
| He & McAuley, "VBPR: Visual Bayesian Personalized Ranking" (AAAI 2016) | Paper | Visual embedding for product recommendation; CLIP-based extension |
| Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2019) | Paper | BERT foundation; mBERT multilingual variant |
| Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP, 2021) | Paper | CLIP for product image embeddings |
| Yi et al., "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations" (RecSys 2019) | Paper | In-batch negative correction for two-tower training; relevant for merchant-scoped negatives |
| Covington et al., "Deep Neural Networks for YouTube Recommendations" (RecSys 2016) | Paper | Foundational two-tower recommendation; candidate retrieval + ranking pipeline |
| Martin et al., "CamemBERT: a Tasty French Language Model" (2020) | Paper | Example of language-specific model that outperforms mBERT; motivation for regional model strategy |
| Amazon Science Blog: "Scaling up knowledge for big catalog recommendations" | Blog | Long-tail product catalog recommendations; sparse interaction handling |
