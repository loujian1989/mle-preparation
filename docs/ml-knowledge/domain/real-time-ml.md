# Real-Time ML — ML Knowledge Q&A

P1: Uber, Shopify, Roblox, Whatnot.

---

## Feature Freshness

### Q: What is feature freshness, and why does it matter more for some features than others?

**Answer (Staff level):**
- **Feature freshness** = age of the feature value used at inference time. Stale features cause prediction error proportional to how much the feature changes over the staleness window.
- **Freshness tiers**:

| Tier | Staleness | Examples | How to serve |
|---|---|---|---|
| **Real-time (<1s)** | Sub-second freshness required | Fraud: txn_count_in_past_1h, auction bid price | Computed inline at request time from event stream |
| **Near-real-time (1–60s)** | Minutes acceptable | Ride demand in this H3 cell, live auction view count | Computed from Kafka consumer, written to Redis |
| **Batch (hours–days)** | Daily refresh acceptable | User historical preferences, merchant risk score | Precomputed overnight, stored in feature store |

- **Mismatch between training and serving**: if features are computed fresh at training time but stale at serving time, you have a **training-serving skew** — the model sees different distributions. Solution: materialize features at training time using the same serving pipeline (point-in-time correct joins).
- **Freshness SLA**: define per-feature: "txn_count_1h must be < 5s old at inference." Monitor staleness distribution in production.

**Company context:** Uber (ETA freshness: driver location must be <2s old), Shopify (merchant activity score: acceptable at 1hr staleness), Roblox (player affinity: 24h batch acceptable), Whatnot (live bid price: sub-second required).

**Common wrong answer:** "I'd precompute all features in batch." — Some features (live prices, real-time counts) change too fast for batch. Must differentiate feature types and match serving strategy to freshness requirement.

---

## Online vs. Batch Inference

### Q: When do you use online inference (real-time scoring) vs. batch inference (precomputed scores)?

**Answer (Staff level):**

| Criterion | Online Inference | Batch Inference |
|---|---|---|
| **Latency requirement** | Sub-second (p99 <100ms) | Seconds to hours acceptable |
| **Personalization** | Request-time context needed (current session, live inventory) | Historical preference sufficient |
| **Feature freshness** | Must use live features | Can use cached features |
| **Scale** | Per-request; limited by QPS capacity | Amortized; score entire user base overnight |
| **Example** | Fraud scoring at checkout | Daily email recommendations |

- **Hybrid**: precompute candidate sets (batch), re-rank online (real-time). Common pattern for feed ranking.
- **Pre-computation when possible**: for cases where context at request time is limited (e.g., email CTR prediction — you know user and content, just not current session), batch scoring is cheaper and simpler.
- **Serving infra for online**: model server (TorchServe, Triton) + feature store (Redis) + load balancer. Target: p99 <100ms, p50 <20ms.

**Company context:** Uber (ETA = online), Shopify (merchant recommendations = hybrid), Roblox (game recs = batch + online re-rank), Whatnot (live auction = online only).

**Common wrong answer:** "I'd always use online inference for accuracy." — Batch is often sufficient and 10–100× cheaper to operate. Over-engineering the serving pipeline for freshness you don't need wastes resources.

---

## Feature Store

### Q: What is a feature store and why do you need one?

**Answer (Staff level):**
- **Feature store**: a system that manages ML features across training and serving, ensuring consistency.
- **Problems it solves**:
  1. **Training-serving skew**: without a feature store, features are computed differently during training (Spark/Python batch) vs. serving (Java/Go microservice). Different implementations → different values → model sees different distribution at inference.
  2. **Feature reuse**: multiple teams recompute the same feature independently. Feature store shares computation.
  3. **Point-in-time correctness**: for training data assembly, you need feature values as they were at each training example's timestamp — not current values. Feature store maintains historical feature snapshots.
- **Two stores**:
  - **Offline store** (S3 + Parquet, BigQuery): historical feature snapshots for training data assembly. Optimized for batch reads.
  - **Online store** (Redis, DynamoDB): latest feature values for low-latency serving. Optimized for p99 <10ms point lookups.
- **Sync**: a pipeline writes newly computed features to both stores simultaneously. Feast, Tecton, Hopsworks, Meta's learned feature store, Uber's Palette.

**Company context:** Uber (Michelangelo has a built-in feature store), Shopify (Pano/Feast-based), Roblox, Whatnot.

**Common wrong answer:** "I'd compute features in the serving code." — Leads to training-serving skew (different implementation), no feature reuse, and no point-in-time correctness for training. Feature store is the standard solution.

---

## Online Learning

### Q: When do you use online learning (continuous model updates) vs. periodic retraining?

**Answer (Staff level):**
- **Online learning**: model parameters updated incrementally as new data arrives (SGD on streaming batches). Model always reflects recent distribution.
- **Periodic retraining**: retrain from scratch (or warm-start) on a fixed schedule (daily, weekly).

| | **Online Learning** | **Periodic Retraining** |
|---|---|---|
| Concept drift speed | Fast (minutes to hours) | Slow (days to weeks) |
| Data volume | Streaming, unlimited | Batch, fits in memory |
| Training stability | Risk of catastrophic forgetting | Stable |
| Engineering complexity | High (streaming infra, monitoring) | Low |
| Use case | Fraud (adversarial drift), ads CTR | Churn, recommendations |

- **Online learning risks**: **catastrophic forgetting** (new updates overwrite old knowledge), **training instability** (no convergence guarantee with streaming), **label latency** (for churn: you don't know if a user churned for 30 days — no immediate label for online training).
- **Common hybrid**: retrain nightly on rolling window; apply lightweight online updates (e.g., update embedding bias terms) for very fresh signal.

**Company context:** Roblox (real-time game recommendation online updates), Whatnot (live auction signal), Uber (ETA: driver speed updated in real-time).

**Common wrong answer:** "I'd retrain daily for freshness." — Daily retraining may be insufficient for adversarial drift (fraud) but overkill for slow-moving signals (churn). Must calibrate retraining frequency to drift speed.

---

## Serving Latency

### Q: Your p99 inference latency is 450ms vs. a 100ms SLA. Walk through how you diagnose and fix it.

**Answer (Staff level):**
- **Diagnosis step 1 — profile the critical path**:
  - Feature retrieval time: Redis round-trip, count of feature lookups per request
  - Model inference time: forward pass
  - Post-processing: score normalization, business rules
  - Network overhead: serialization, TLS, load balancer hops
- **Common culprits and fixes**:

| Root Cause | Fix |
|---|---|
| Sequential feature fetches | Batch all feature lookups into a single Redis pipeline (pipelining reduces N round-trips to 1) |
| Large model (slow forward pass) | Model distillation (smaller student), quantization (FP32 → INT8), pruning |
| Python GIL in serving | Move model server to Triton (C++) or TorchServe with async workers |
| Cold start (no cache) | Warm the model server cache on deploy; use connection pooling |
| Unoptimized pre/post-processing | Profile with cProfile; vectorize with NumPy instead of Python loops |
| p99 tail from garbage collection | Set JVM GC tuning (if Java-based) or use Go-based feature server |

- **Measurement**: always diagnose with percentile histograms (p50, p95, p99, p99.9), not averages. p99 latency is what users experience in the worst 1%.

**Company context:** Uber (sub-second ETA SLA), Shopify (<100ms recommendation SLA), Whatnot (live auction — sub-100ms for bid ranking).

**Common wrong answer:** "I'd use a bigger instance." — Scaling horizontally reduces throughput bottlenecks, not per-request latency. Must profile to find the specific bottleneck.

---

## Model Versioning and A/B in Production

### Q: How do you safely deploy a new model version with minimal risk?

**Answer (Staff level):**
- **Traffic splitting**: route 1–5% of traffic to new model, remainder to current model. Compare key metrics (CTR, revenue, fraud detection rate) between cohorts for 24–72h.
- **Shadow mode (dry run)**: run new model in parallel with no serving impact. Log its scores alongside the current model's scores. Compare offline metrics without risk. Useful for catching bugs before any traffic is served.
- **Canary deployment**: promote shadow → 1% → 5% → 20% → 100%. Each promotion gate has automated metric checks (revenue per request, error rate, latency p99).
- **Rollback trigger**: define rollback conditions upfront (e.g., "if fraud FN rate increases > 10% vs. baseline for 2 consecutive hours, auto-rollback"). Requires near-real-time metric pipelines.
- **Model versioning**: each deployed model has a version tag + feature schema version. The feature store must serve the correct feature schema version to match the model.

**Company context:** Netflix (observability + A/B design is explicitly probed), Shopify, Uber.

**Common wrong answer:** "I'd deploy the new model and monitor it." — No traffic splitting, no rollback criteria, no shadow mode. Staff answer includes specific rollback triggers and promotion criteria.

---

## Feature Freshness (Deep Dive)

### Why Staleness Hurts Some Features More Than Others

The impact of stale features scales with how fast they change and how much they affect the prediction:

```
Feature: user_age (changes ~once/year)
  Staleness of 24h → delta ≈ 0  → zero prediction impact

Feature: merchant_risk_score (changes weekly based on chargebacks)
  Staleness of 24h → delta ≈ small  → minimal impact
  Acceptable at batch (daily refresh)

Feature: txn_count_in_past_1h (changes every transaction)
  Staleness of 5 minutes → could miss 10+ transactions for a fraud pattern
  → model sees txn_count=2 when true value is 15
  → completely misses card-testing pattern
  → must be computed fresh at inference time
```

**Rule of thumb for freshness tier assignment:**
- Feature half-life < 1 minute → real-time (compute at inference)
- Feature half-life 1 min – 1 hour → near-real-time (Kafka + Redis, 30s lag)
- Feature half-life > 1 hour → batch (hourly or daily)

### Training-Serving Skew — The Silent Bug

This is the most common and hardest-to-detect ML production bug:

```
Training pipeline (Spark job):
  merchant_risk_score = AVG(chargebacks) over ENTIRE history
  → computed correctly using all data available in training

Serving pipeline (Java microservice):
  merchant_risk_score = query(merchant_id) from Redis
  → Redis entry updated daily at 2am
  → at 3pm, this feature is 13 hours stale

Result:
  Training sees "true" risk scores
  Serving sees yesterday's risk scores
  For rapidly changing merchants, these differ significantly
  Model trained on current risk scores but served with stale ones
  → systematic prediction errors that don't show up in offline eval
```

Detection: compare feature distributions at training time vs. serving time (log a sample of serving-time features, compare to training distribution using KL divergence).

Fix: use the same feature computation pipeline for both training and serving. The feature store's offline store provides historical feature snapshots for training; the online store provides current values for serving.

---

## Feature Store (Deep Dive)

### Point-in-Time Correctness — Why It Matters

When building a training dataset, you need feature values as they existed at each training example's timestamp — not current values.

```
Training example: user U made a transaction at 2024-01-15 14:32:00
Features needed:  txn_count_30d as of 2024-01-15 14:32:00

WRONG: query current txn_count_30d for user U
  → includes all transactions AFTER 14:32:00 on Jan 15
  → temporal leakage: future transactions contaminate features
  → model learns from a future that wasn't available at prediction time

RIGHT: query offline store for txn_count_30d(user_U, as_of=2024-01-15 14:32:00)
  → feature store maintains historical snapshots
  → returns the value that was in the online store at that exact timestamp
```

Without point-in-time correctness: AUPRC on offline eval looks great (model learned from future data), production performance is much worse (future data unavailable), and you can't explain the gap.

### Dual-Store Architecture — Why Two Stores

**Offline store** (S3 + Parquet / BigQuery):
- Stores full history of feature values: `(entity_id, timestamp, feature_value)`
- Optimized for batch reads during training data assembly
- Low QPS, high throughput, cheap storage
- Query: "give me txn_count_1h for all users between Jan 1 and March 31"

**Online store** (Redis / DynamoDB):
- Stores only the latest value: `entity_id → feature_value`
- Optimized for single-key lookups at serving time
- p99 < 10ms read latency
- Query: "give me txn_count_1h for user 12345 right now"

**Why not just one store?**
- Historical Parquet doesn't give sub-10ms point lookups → can't use for serving
- Redis doesn't store history → can't do point-in-time training joins
- Each store is optimized for exactly one access pattern

---

## Online Learning (Deep Dive)

### Catastrophic Forgetting — The Core Risk

Standard neural networks store knowledge in weights. New training overwrites old weights:

```
Model trained on Jan data: learned that "international + night + high_amount = fraud"
Online update with Feb data: Feb has few international fraud cases (they adapted)
Weights get updated toward "international + night + high_amount ≠ fraud"
→ March: international fraud resumes → model forgot the pattern

This is catastrophic forgetting: new data overwrites old knowledge
```

For tree models (GBT): no forgetting risk because new trees are ADDED, not overwrites. But you need to manage tree count (ensemble grows indefinitely).

**Elastic Weight Consolidation (EWC)**: for neural networks, add a regularization term that prevents weights critical for previous tasks from changing too much:
```
L_new = L_new_data + λ × Σᵢ Fᵢ(θᵢ - θ*ᵢ)²
```
`Fᵢ` = Fisher information (how important is weight i to previous knowledge). Rarely used in production — simpler to use replay buffers (mix old and new data).

### Label Latency — Why Fraud Can't Learn Online

For fraud detection:
```
Transaction occurs at t=0
Customer sees statement at t=30 days
Customer disputes charge at t=35 days
Chargeback confirmed at t=45 days
Label (fraud=1) available at t=45 days
```

Online learning requires labels within minutes. Fraud labels take 45 days. You can't update the model online for fraud — you can only do periodic retraining once labels accumulate.

**What CAN learn online for fraud:** account-level velocity features and session-level signals can be updated in real-time from new transactions, even before labels arrive.

---

## Serving Latency (Deep Dive)

### How to Profile — The Critical Path Approach

p99 = 450ms, SLA = 100ms. Don't guess — measure every component:

```
Add timing instrumentation to each stage:
  t0: request received by serving endpoint
  t1: after authentication / request parsing
  t2: after feature retrieval from Redis
  t3: after model forward pass
  t4: after post-processing and business rules
  t5: response sent

p99 measurements:
  t1 - t0: 5ms   (parsing — fast)
  t2 - t1: 280ms ← CULPRIT: feature retrieval
  t3 - t2: 120ms ← model inference also slow
  t4 - t3: 30ms  (post-processing)
  t5 - t4: 15ms  (serialization)
```

With two culprits found:

**Feature retrieval: 280ms → 15ms**
```
Root cause: 50 sequential Redis round-trips (one per feature)
  50 × 5ms per RTT = 250ms + 30ms overhead = 280ms

Fix: Redis pipelining — batch all 50 reads into one round-trip
  1 × 5ms + 10ms parsing = 15ms  ← 18× improvement
```

**Model inference: 120ms → 25ms**
```
Root cause: FP32 model, no batching, Python GIL

Fix 1: quantize FP32 → INT8
  ~3-4× faster arithmetic, slight accuracy loss (validate with shadow mode)
  120ms → 35ms

Fix 2: switch to TorchServe with async workers
  Parallel requests, no GIL bottleneck
  35ms → 25ms
```

**Combined: 450ms → 5 + 15 + 25 + 30 + 15 = 90ms ← meets SLA**

---

## Model Versioning and A/B in Production (Deep Dive)

### The Canary Promotion Ladder

Each promotion gate has automated metric checks:

```
Shadow mode (0% traffic, logs only):
  Check: model output distribution matches expected range
  Check: no NaN/inf outputs
  Check: latency p99 < SLA
  Gate: human review of score distributions

1% canary:
  Duration: 2 hours
  Check: error rate < 0.1%, latency p99 < 100ms, primary metric direction positive
  Gate: automated (if checks pass → auto-promote)

5% canary:
  Duration: 24 hours
  Check: primary metric, guardrail metrics (all must pass)
  Gate: automated

20% canary:
  Duration: 48 hours
  Full A/B analysis: statistical significance on primary metric
  Gate: human approval

100% rollout:
  Monitor for 7 days before decommissioning old model
```

### Rollback Trigger Design

Define rollback conditions with specific thresholds BEFORE deployment:

```
Auto-rollback if ANY of:
  - Error rate > 0.5% (vs baseline 0.02%) for 5 consecutive minutes
  - p99 latency > 200ms for 10 consecutive minutes
  - Fraud FN rate increases > 15% vs baseline for 2 consecutive hours
  - Revenue per request drops > 5% for 4 consecutive hours
  - Primary metric moves in wrong direction with p < 0.01 (statistically significant degradation)

Why 5-min / 10-min / 2-hour windows:
  Too short → react to noise (natural traffic variance)
  Too long → real problems linger

Error rate: short window (errors are obvious fast)
Fraud FN rate: longer window (need enough fraud events to measure)
Revenue: longer window (need statistical power)
```

Pre-committed rollback criteria prevent the "let's wait and see" trap that allows bad models to run for hours.

---
