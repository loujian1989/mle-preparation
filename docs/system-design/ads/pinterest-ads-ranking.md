# Pinterest Ads Ranking (CTR + Conversion) — ML System Design

**Domain:** `ads`
**Target Company:** Pinterest
**Difficulty Bar:** L6 (IC16 Staff)
**Date:** 2026-03-27
**Related Designs:** `../ranking/meta-news-feed-ranking.md`, `../templates/ml-system-design-template.md`

---

## 0. Scorecard

| Axis | Score | Weakest Point |
|---|---|---|
| Requirements | ★★★★★ | — |
| Data Modeling | ★★★★★ | — |
| ML Pipeline | ★★★★★ | — |
| Failure Modes | ★★★★☆ | Galaxy (feature store) partial failure not fully specified |
| Capacity | ★★★★★ | — |

**Overall:** `STRONG HIRE`
**Top Gap:** Galaxy (online feature store) partial failure — when Galaxy degrades but doesn't fully fail, which features degrade and how does the MMOE-DCN model perform with missing inputs?

---

## 1. Requirements

#### Functional Requirements
1. Rank ads in the Pinterest home feed, search results, and related pins by predicted value (CTR × CVR × bid)
2. Predict three signals jointly: click-through rate (CTR), conversion rate (CVR), and return on ad spend (ROAS)
3. Ensure calibrated predictions: predicted CTR/CVR must match realized rates (required for auction pricing integrity)
4. Enforce algorithmic fairness: avoid disproportionate disadvantage to small advertisers vs. large advertisers
5. Support A/B experimentation on ranking parameters without full model retraining

#### Non-Functional Requirements

| Dimension | Target | Business Tie |
|---|---|---|
| Latency (p99 serving) | ≤ 100ms | Pinterest home feed SLA; ads are injected inline — latency adds directly to page load |
| Availability | 99.99% | Ad serving downtime = direct revenue loss (Pinterest monetizes primarily through ads) |
| Consistency | Eventual (minutes) | User feature freshness ≤ 5 min; ad features ≤ 1 hour |
| Throughput | ~80K peak QPS | 400M MAU × ~5 ad impressions/day / 86,400 × 3× peak |
| CTR calibration error | ≤ 5% relative | Uncalibrated CTR = incorrect auction pricing = advertiser over/underpays |
| Fairness (demographic parity) | FPR deviation ≤ 10% across advertiser size buckets | Small advertiser protection: no systematic disadvantage vs. large budget advertisers |

#### Scale Numbers (stated upfront)
- **MAU / DAU:** 400M MAU / ~100M DAU
- **Peak QPS:** ~80K
- **Active ad campaigns:** ~5M globally
- **Ad impressions/day:** ~500M
- **Advertisers:** ~1M (majority are small businesses — median monthly budget < $500)

#### Out of Scope
- Ad auction mechanism and pricing (consumes predicted CTR/CVR as input; not designed here)
- Ad creative generation and optimization (separate system)
- Advertiser budget management and pacing
- Organic pin ranking (separate ranking model; different objective)

> **Pinterest rubric:** Algorithmic fairness is probed explicitly. State demographic parity targets and how they're measured before designing the model. Read "Handling Online-Offline Discrepancy in Pinterest Ads Ranking" before this interview.

---

## 2. Data Modeling

#### Input Signals

| Feature | Type | Freshness | Source | Notes |
|---|---|---|---|---|
| User interest embedding (visual + text) | `batch` | 1 hour | Spark on pin engagement → Galaxy (offline) | Pinterest users are interest-graph driven; visual content is dominant |
| User's recent ad engagement (last 10 ads) | `real-time` | ≤ 5 min | Flink stream → Galaxy (Redis) | Session-level intent; strong predictor of current browsing mode |
| User's shopping intent signals | `batch` | daily | Purchase event log → Galaxy | History of clicking "Shop" button, saved products, price comparisons |
| Ad creative embedding (pin image + title) | `batch` | daily | CLIP + text encoder → Galaxy (offline) | 256-d; captures visual style and category |
| Advertiser historical CTR (by category) | `batch` | 1 hour | Spark on impression logs → Galaxy | Advertiser-level quality signal; key for fairness audit |
| Ad × user category affinity | `real-time` | request-time | Computed from user interest × ad category | Cross feature via DCN interaction layer |
| Ad bid amount | `real-time` | request-time | Auction system | Combined with predicted CTR/CVR for final ranking score |
| Advertiser budget remaining | `real-time` | ≤ 1 min | Pacing system → Redis | Prevents over-delivery; excludes exhausted budgets |
| Time features (hour, day, holiday) | `real-time` | request-time | Request context | Shopping intent peaks differ by time |
| Ad format (static pin, video, carousel) | `static` | at campaign creation | Campaign management system | Format affects CTR baseline; video typically higher |

#### Label Definition
- **Labels (multi-task):**
  - CTR label: click event within 1 min of impression (immediate)
  - CVR label: purchase/signup event within 30 days of click (long delay!)
  - ROAS label: reported conversion value / ad spend (requires advertiser pixel/API integration; 30-day window)
- **Collection strategy:** Impression-click-conversion funnel; conversion attribution via Pinterest Tag (pixel) or API conversion events
- **Positive/negative ratio:** CTR ~1:50; CVR ~1:200 (conversion is much rarer)
- **Label delay:** CTR is immediate; CVR/ROAS has 30-day delay → use proxy signals (add-to-cart, product save) for faster feedback
- **Online-offline discrepancy (Pinterest-specific probed topic):**
  - Training data features come from offline batch computation; serving features come from Galaxy online store
  - If batch and online feature computation differ (e.g., different aggregation windows, rounding), the model sees a distribution it was never trained on
  - Mitigation: feature consistency tests — run both pipelines on the same data, assert max deviation < 1%; daily automated consistency audit
- **Bias risks:**
  - **Small advertiser bias:** Large advertisers have more historical data → model learns their patterns better → small advertisers get systematically lower predicted CTR → higher effective CPM → less competitive in auction
  - **Visual content bias:** Model may learn visual style correlations (e.g., white background → higher CTR) that advantage large advertisers with professional creative
  - **Conversion window bias:** 30-day attribution window means recent campaigns have censored labels; training on censored data underestimates true CVR

#### Storage Engine Choice

| Layer | Engine | Justification |
|---|---|---|
| Feature store (online) | **Galaxy** (Pinterest's homegrown feature store, Redis backend) | Native Pinterest platform; sub-ms reads; TTL-based session features |
| Feature fetch + inference | **Scorpion** (Pinterest's feature fetching + inference service) | Handles Galaxy reads + model inference in single service; reduces latency by co-locating computation |
| Feature store (offline) | Galaxy offline layer → S3 (Parquet) | Feast-compatible; Spark-readable; daily snapshots |
| Ad embeddings | Redis (key: ad_id → embedding) | High read rate; updated daily from batch pipeline |
| Training data | Parquet on S3 (date-partitioned, campaign-partitioned) | Columnar; Spark-efficient |
| Logs / labels | Kafka → Flink → Hive | Streaming ingest; Flink joins click + conversion events for label construction |
| Model artifacts | S3 + MLEnv (Pinterest's full-stack ML framework) | MLEnv manages training, evaluation, and deployment lifecycle |

#### Online vs. Offline Split

```
Offline (batch, Spark)                              Online (real-time, < 100ms)
────────────────────────────────────────────        ──────────────────────────────────────────────
Click/conversion events → Hive → label join         Request: user_id + surface + ad_candidates
Spark: user interest embeddings (CLIP + NLP)        Scorpion: Galaxy reads (user + ad features, 5ms)
Spark: advertiser CTR history by category           Scorpion: MMOE-DCN inference (15ms)
Spark: ad creative embeddings                       Calibration layer: Platt scaling (1ms)
Daily MMOE-DCN training (CTR + CVR + ROAS)          Fairness filter: small advertiser floor (1ms)
Fairness audit: FPR by advertiser_size_bucket       Auction score: eCPM = predicted_CTR × bid (1ms)
Calibration: Platt scaling on held-out val set      Response + async Kafka log
Champion/challenger: fairness + accuracy gate
```

#### Schema

```
Ad: {
  ad_id:              string
  campaign_id:        string
  advertiser_id:      string
  creative_embedding: float[256]      # CLIP + text fusion
  target_categories:  string[]        # advertiser-specified targeting
  bid_amount:         float
  format:             enum[STATIC, VIDEO, CAROUSEL, SHOPPING]
  advertiser_size:    enum[SMALL, MEDIUM, LARGE]   # for fairness audit
  updated_at:         timestamp
}

AdImpression: {
  impression_id:      string
  user_id:            string
  ad_id:              string
  surface:            enum[HOME_FEED, SEARCH, RELATED_PINS]
  rank_position:      int             # for IPS
  predicted_ctr:      float
  predicted_cvr:      float
  bid_amount:         float
  timestamp:          timestamp
}

ConversionEvent: {
  impression_id:      string          # 30-day attribution window
  event_type:         enum[CLICK, ADD_TO_CART, PURCHASE, SIGNUP]
  conversion_value:   float?          # for ROAS computation
  attributed_at:      timestamp
}
```

> **Pinterest rubric:** Online-offline discrepancy is a documented Pinterest interview topic — consistency audit between Galaxy batch and Galaxy online is named. Advertiser size is a first-class schema field for fairness.

---

## 3. ML Pipeline

### 3a. Offline Training

#### Data Pipeline
- **Ingestion:** Impression logs + click events + conversion events → Kafka → Flink → Hive; 30-day conversion window requires delayed label join
- **Feature engineering:**
  - User interest embedding: mean-pool of CLIP embeddings for last 100 engaged pins (Spark, hourly)
  - Ad creative embedding: CLIP (visual) + SentenceTransformer (title + description) → concatenate → linear projection to 256-d (daily)
  - Online-offline consistency test: daily automated job runs both batch and online feature pipelines on same 10K sample; assert cosine similarity > 0.99 for embedding features; assert deviation < 0.01 for scalar features
- **Train/val/test split:** Time-based — train D-60 to D-1; val D-1; test D-0; conversion labels joined with 30-day lag; recent training data uses proxy CVR (add-to-cart) until full conversion label arrives
- **Orchestration:** MLEnv DAG; idempotent; Spark partitioned by campaign; Flink label-join service runs continuously

#### Model Architecture

**MMOE-DCN (Multi-gate Mixture-of-Experts + Deep Cross Network):**

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Shared bottom multi-task NN | Simple; handles multiple objectives | Shared bottom causes task conflict (CTR and ROAS have different optimal feature interactions) | Rejected |
| MMOE (chosen) | Expert networks specialize per task; gating reduces task conflict | More parameters; slower training | **Chosen** |
| DCN-v2 only (single task) | Fast; strong feature crosses | Cannot jointly optimize CTR + CVR + ROAS | Rejected |
| MMOE + DCN-v2 (full model) | Expert specialization + explicit feature crosses; current Pinterest production architecture | Complex to tune; requires careful gate initialization | **Chosen as full model** |

**Architecture details:**
- **Expert networks:** 8 shared expert MLPs (256 → 128 → 64 units); each expert specializes in different feature interaction patterns
- **Task-specific gates:** Softmax attention over 8 experts per task (CTR gate, CVR gate, ROAS gate)
- **DCN-v2 cross layer:** 4 cross layers on top of task-specific expert output; captures explicit feature crosses (user_category × ad_category, recency × content_type)
- **Task heads:**
  - CTR head: sigmoid output; binary cross-entropy loss
  - CVR head: sigmoid output; binary cross-entropy loss (on click-conditional conversions)
  - ROAS head: regression output; MSE loss on log-normalized conversion value
- **Combined loss:** `L = 0.5 × BCE(CTR) + 0.3 × BCE(CVR) + 0.2 × MSE(ROAS)` — weights tuned to match advertiser objective priorities

**Calibration (Platt scaling):**
- Raw sigmoid output ≠ calibrated probability → auction pricing incorrect
- After training, fit Platt scaling (`σ(a × raw_score + b)`) on held-out val set
- Calibration is per-category (CTR for fashion differs from electronics) and per-surface (home feed vs. search)
- Validation: reliability diagram; expected calibration error (ECE) ≤ 0.02

**Fairness objective:**
- Measure: predicted CTR by advertiser_size_bucket (SMALL/MEDIUM/LARGE); target: no bucket has > 10% lower predicted CTR vs. global average for same category
- Mitigation: add fairness regularization term to loss: `L_fair = λ × Σ |FPR_bucket_i - FPR_global|²`
- Trade-off: fairness regularization reduces overall AUC slightly (~0.3% AUC-ROC); acceptable per product decision

#### Training Infrastructure
- **Framework:** PyTorch + DDP (MMOE-DCN is parameter-heavy; DDP for data parallelism)
- **Scale:** 32× A100 80GB; ~8hr/full retrain; daily incremental fine-tune ~1hr on last 24hr data
- **Mixed precision:** bfloat16
- **Eval metrics:** Per-task AUC-ROC, AUC-PR (CTR/CVR), MAPE on ROAS, ECE (calibration), fairness FPR parity across advertiser_size_bucket

---

### 3b. Online Serving

#### Inference Path

```
Ad Auction Request (user_id + surface + ad_candidate_list)
  → Ad Serving Service
  → Scorpion (Pinterest's feature fetch + inference co-located service)
      ├─ Galaxy (Redis): user features (interest embedding, session signals) (3ms)
      ├─ Galaxy (Redis): ad creative embeddings + advertiser CTR history (2ms)
      ├─ MMOE-DCN inference
      │    → predicted_CTR, predicted_CVR, predicted_ROAS (15ms, GPU)
      ├─ Platt scaling calibration (1ms)
      ├─ Fairness floor: boost small advertiser effective bid by 5% if below threshold (1ms)
      └─ eCPM score: predicted_CTR × bid × quality_score (1ms)
  → Ad Auction (selects winning ads by eCPM)
  → Response + async Kafka log (impression + predictions)
```

#### Latency Budget Breakdown

| Step | p50 | p99 | Notes |
|---|---|---|---|
| Galaxy feature fetch (Scorpion) | 4ms | 10ms | Co-located; Scorpion batches reads |
| MMOE-DCN inference (GPU) | 10ms | 25ms | TensorRT; 50–200 ad candidates per request |
| Calibration + fairness floor | 1ms | 2ms | CPU arithmetic |
| eCPM computation + ranking | 1ms | 3ms | CPU; sort N ads |
| Network + serialization | 3ms | 10ms | gRPC |
| **Total** | **19ms** | **50ms** | Budget: ≤ 100ms p99 ✓ |

#### Caching Strategy
- **Ad embedding cache (Galaxy/Redis):** Updated daily; TTL 25 hours; rebuilt from batch pipeline
- **User feature cache (Galaxy/Redis):** TTL 5 min for session signals; 1 hour for long-term interest embedding
- **No caching of ad scores:** Each request has different user context; ad score for user A ≠ ad score for user B on same ad

---

### 3c. Monitoring

> **Designed upfront — calibration monitoring and fairness monitoring are Pinterest-specific requirements.**

#### Drift Detection

| Signal | Method | Threshold | Action |
|---|---|---|---|
| CTR calibration error (ECE) | Daily reliability diagram | ECE > 0.03 | Recalibrate Platt scaling; alert revenue team |
| Advertiser FPR parity (fairness) | Daily audit by advertiser_size_bucket | > 10% relative deviation | Fairness regularization weight adjustment; alert policy team |
| User interest embedding drift | PSI daily | PSI > 0.2 | Retrain trigger |
| CVR label delay (conversion arrival rate) | Daily monitoring of 30-day label completeness | < 80% of expected conversions arrived | Check Pinterest Tag pixel health; alert data engineering |
| Ad creative embedding distribution | Rolling mean ± 2σ | > 20% shift sustained 4hr | Batch embedding pipeline failure; check CLIP service |
| MMOE gate entropy (expert utilization) | Weekly audit | Any gate collapses to single expert (entropy < 0.5) | Mode collapse in MMOE; reinitialize gating layer |
| p99 latency (Scorpion) | Real-time APM | > 100ms sustained 3min | Circuit breaker |

#### Shadow Scoring
- Challenger MMOE-DCN (e.g., new fairness regularization weight) runs on 2% of traffic; results logged
- Comparison: daily AUC per task + ECE + fairness FPR parity; promotion requires improved fairness without AUC regression
- MLEnv manages champion/challenger lifecycle; automatic rollback if guardrail metrics degrade within 24hr of promotion

#### A/B Holdout Design
- **Unit:** `advertiser_id` for fairness experiments (measure impact on advertiser outcomes); `user_id` for ranking quality experiments
- **Holdout size:** 5% of advertisers (50K advertisers) as permanent holdout
- **Primary metric:** Revenue per impression (RPM) for business; NDCG@10 for ranking quality
- **Guardrail metrics:** CTR calibration ECE, advertiser fairness FPR parity, p99 latency, advertiser churn rate

> **Pinterest rubric:** Fairness and calibration monitoring are designed before the model, not after deployment. MMOE gate collapse is a known failure mode — monitoring gate entropy catches it before it affects accuracy.

---

## 4. Failure Modes

| Scenario | Impact | Mitigation | Fallback |
|---|---|---|---|
| Galaxy (online feature store) unavailable | No user or ad features for serving | Retry ×1 (5ms timeout); circuit breaker | Pre-computed daily scores from S3 (no real-time personalization); rule-based ad selection by bid × category match |
| MMOE-DCN GPU model unavailable | No ML-based ad ranking | Pre-ranked ad list from last successful request (cached per user_id, TTL 5 min) | Bid-only ranking (highest bidder wins; no quality score) |
| Platt calibration stale (> 24hr old) | CTR/CVR predictions uncalibrated | Monitor calibration freshness; alert at > 12hr; force recalibration if ECE > 0.05 | Use uncalibrated model with wider auction margins; alert revenue team |
| Fairness floor not applied (code bug) | Small advertisers systematically disadvantaged | Fairness floor applied in separate service (not inline with ranking); separate monitoring circuit | Alert; retrospective bid adjustment for affected campaign window |
| CVR label pipeline delay (pixel outage) | Training data gap for CVR task | Monitor conversion arrival rate; halt CVR fine-tuning if < 80% label completeness | CTR-only model as fallback; CVR head outputs prior mean until labels recover |
| MLEnv training job failure | Model staleness | Auto-retry ×3; champion stays live; alert at 48hr staleness | Current champion continues; ECE may drift if ad distribution shifts |

#### Circuit Breaker Thresholds
- **Error rate:** Trip at > 0.5% 5xx → bid-only ranking fallback; alert on-call
- **Latency:** Trip at p99 > 120ms sustained 3min → disable MMOE; use lightweight LightGBM backup scorer (< 5ms)
- **Recovery:** Half-open after 30s; restore full pipeline on 3 consecutive successes

#### Degraded-Mode Behavior
1. **Level 1** — Full MMOE-DCN + calibration + fairness (normal)
2. **Level 2** — LightGBM backup scorer (precomputed daily features; no session signals; ~15% AUC regression)
3. **Level 3** — Bid-only ranking (highest effective bid wins; no quality score; revenue-optimal but user experience degrades)

---

## 5. Capacity Estimates

> **Assumptions:**
> - MAU: 400M; DAU: 100M
> - Ad impressions per DAU: 5
> - Peak QPS: 80K (3× average)
> - Ads per auction: 100–200 candidates per request
> - User feature vector: 256 floats = 1KB
> - MMOE-DCN model: 2GB (expert networks + DCN layers)
> - Galaxy Redis: 100M users × 1KB = ~100GB online feature storage
> - Log retention: 90 days hot (Hive), 3 years cold (Pinterest compliance)

#### Estimates

| Metric | Calculation | Result |
|---|---|---|
| Avg QPS | 100M × 5 / 86,400 | **~5,800 QPS** |
| Peak QPS | 5,800 × 3× | **~17,400 QPS** (stated 80K is headroom capacity) |
| Galaxy reads/s (user + ad features) | 17,400 × 5 reads | **~87K reads/s** |
| Ad embedding storage (Galaxy) | 5M active ads × 1KB | **~5GB** (trivially fits in Redis) |
| User feature storage (Galaxy) | 100M users × 1KB | **~100GB** |
| Impression logs ingest (Kafka) | 17,400 × 150 ads × 200B | **~522MB/s → ~45TB/day** |
| Training data (60 days) | 500M impressions/day × 60 days × 500B | **~15TB** |
| MMOE-DCN serving replicas | 17,400 QPS × 150 ads / 50K inferences per A10G | **~52 A10G GPUs** |
| Conversion label join latency | 30-day attribution window requires 30-day Hive partition retention | **~30TB active label data** |

---

## 6. Open Questions & Follow-ups

#### Unresolved Design Decisions
- [ ] **ROAS task weight:** Currently 0.2 in the combined loss. Should ROAS be weighted higher as Pinterest shifts toward conversion-focused advertisers? Trade-off: higher ROAS weight may hurt CTR calibration for brand advertisers
- [ ] **Advertiser lookalike modeling:** Small advertisers with limited budget could benefit from lookalike audience modeling (find users similar to existing converters). Would require cross-advertiser aggregation with differential privacy guarantees
- [ ] **Video ads:** Video pins have different CTR dynamics (completion rate, sound-on rate) than static pins. Should video ads have dedicated model heads or separate MMOE instances?
- [ ] **Delayed CVR proxy:** Using add-to-cart as a 1-day proxy for 30-day conversion reduces label delay but introduces proxy-label gap. Quantify proxy quality by measuring correlation between proxy and true 30-day conversion across categories

#### Company Rubric Gaps (self-assessment)
- [x] Requirements: algorithmic fairness target (±10% FPR); calibration target (ECE ≤ 0.02)
- [x] Data Modeling: online-offline discrepancy documented and audited; Galaxy + Scorpion named
- [x] ML Pipeline: MMOE-DCN architecture; Platt scaling for calibration; fairness regularization in loss
- [x] Failure Modes: bid-only fallback for Galaxy failure; LightGBM backup scorer
- [x] Capacity: GPU count calculated; Galaxy storage realistic

#### Recommended Follow-up Problems
- Meta News Feed Ranking — compare DLRM vs. MMOE-DCN for multi-task ranking at different scale
- Pinterest Visual Search — same CLIP embedding pipeline; different retrieval problem

---

## 7. References

| Resource | Type | Relevance |
|---|---|---|
| Pinterest Engineering Blog: "Beyond Two Towers: Re-architecting the Serving Stack" (Feb 2026) | Blog | Current Pinterest serving architecture; Scorpion service; Galaxy feature store |
| Pinterest Engineering Blog: "Evolution of Ads Conversion Optimization Models" | Blog | MMOE-DCN adoption; multi-task loss weighting; CVR optimization |
| Pinterest Engineering Blog: "Handling Online-Offline Discrepancy in Pinterest Ads Ranking" | Blog | Featured in prep-roadmap as required reading; consistency audit methodology |
| Ma et al., "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts" (MMOE, KDD 2018) | Paper | MMOE architecture; expert gating mechanism; task conflict reduction |
| Wang et al., "DCN V2: Improved Deep & Cross Network" (WWW 2021) | Paper | DCN-v2 for explicit feature crosses; combines with MMOE for Pinterest's architecture |
| Guo et al., "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction" (IJCAI 2017) | Paper | CTR prediction deep learning; baseline comparison for MMOE-DCN |
| Platt, "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods" (1999) | Paper | Platt scaling for CTR calibration |
| Niculescu-Mizil & Caruana, "Predicting Good Probabilities With Supervised Learning" (ICML 2005) | Paper | Calibration methods comparison; reliability diagrams; ECE definition |
| Zafar et al., "Fairness Constraints: Mechanisms for Fair Classification" (AISTATS 2017) | Paper | Fairness regularization in ML; demographic parity constraint formulation |
| Zhao et al., "Recommending What Video to Watch Next: A Multitask Ranking System" (RecSys 2019) | Paper | Multi-task ranking framework; guardrail metric design |
