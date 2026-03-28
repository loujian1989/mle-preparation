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
