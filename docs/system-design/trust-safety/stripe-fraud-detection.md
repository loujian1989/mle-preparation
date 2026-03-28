# Stripe Transaction Fraud Detection — ML System Design

**Domain:** `trust-safety`
**Target Company:** Stripe
**Difficulty Bar:** L6 (Staff)
**Date:** 2026-03-27
**Related Designs:** `openai-content-moderation.md`, `../templates/ml-system-design-template.md`

---

## 0. Scorecard

| Axis | Score | Weakest Point |
|---|---|---|
| Requirements | ★★★★★ | — |
| Data Modeling | ★★★★★ | — |
| ML Pipeline | ★★★★★ | — |
| Failure Modes | ★★★★☆ | Network graph failure (GNN unavailable) fallback not fully specified |
| Capacity | ★★★★★ | — |

**Overall:** `STRONG HIRE`
**Top Gap:** GNN-based graph features — when the graph feature computation is unavailable, specify which features degrade and how the model compensates.

---

## 1. Requirements

#### Functional Requirements
1. Score every payment transaction in real-time with a fraud probability estimate before authorization
2. Take automated action: ALLOW, BLOCK, or FLAG_FOR_REVIEW based on configurable risk thresholds
3. Surface explainable signals to Stripe's Radar UI: which features triggered the fraud score
4. Support merchant-specific threshold customization (different merchants have different fraud tolerance)
5. Adapt to new fraud patterns within 24hr of detection (adversarial dynamics requirement)

#### Non-Functional Requirements

| Dimension | Target | Business Tie |
|---|---|---|
| Latency (p99 serving) | ≤ 100ms | Fraud scoring is inline before card authorization; beyond 100ms, Stripe's authorization p99 breaches Visa/Mastercard SLA |
| Availability | 99.999% (five nines) | Fraud system down = either all transactions blocked (too safe) or all transactions allowed (too risky) — both are catastrophic |
| Consistency | Strong (synchronous gate) | Decision must be made before authorization; eventual consistency not acceptable |
| Throughput | ~500K peak QPS | Black Friday + global peak; Stripe processes billions of transactions/year |
| False positive rate | ≤ 0.1% on legitimate transactions | Blocking legitimate transactions erodes merchant trust and drives churn |
| False negative rate | ≤ 0.5% across all fraud types | Weighted by fraud type severity (card-not-present > friendly fraud) |

#### Scale Numbers (stated upfront)
- **Transactions/day:** ~1B (Stripe processes hundreds of billions annually)
- **Peak QPS:** ~500K (Black Friday, global)
- **Fraud rate:** ~0.1% of transactions globally; varies dramatically by merchant category and geography
- **Merchant count:** 1M+ active merchants; each with independent threshold configuration
- **Card network SLA:** Stripe must respond to authorization requests within 300ms end-to-end; fraud scoring budget is 100ms of that

#### Out of Scope
- Card issuer fraud detection (bank-side; separate system)
- Chargeback dispute resolution workflow
- Merchant fraud (merchant committing fraud on their own platform)
- Account takeover detection (separate authentication system)

> **Stripe rubric:** Adversarial dynamics are the defining characteristic of fraud detection. State the arms race explicitly — fraud patterns shift when detection improves. Threshold is a *business* decision, not an ML metric decision.

---

## 2. Data Modeling

#### Input Signals

| Feature | Type | Freshness | Source | Notes |
|---|---|---|---|---|
| Velocity: txn count (user, 1m/1h/24h) | `real-time` | ≤ 1s | Kafka → Flink → Redis sliding window | Most predictive real-time signal; burst = card testing |
| Velocity: txn amount (user, 1m/1h/24h) | `real-time` | ≤ 1s | Kafka → Flink → Redis | Sudden high-value transactions after low-value testing |
| Device fingerprint hash | `real-time` | request-time | Stripe.js client → API | Browser/device characteristics; known fraudulent devices blocklisted |
| IP address risk score | `batch` | 15 min | IP reputation database → Redis | Proxy/VPN/Tor exit node detection; geo mismatch |
| Shipping ↔ billing address match | `real-time` | request-time | Transaction payload | Mismatch is strong fraud signal for physical goods |
| Card BIN features | `static` | daily | Card network BIN database → Redis | Issuing bank, country, card type (prepaid = higher risk) |
| Network graph features: shared device/IP/card | `batch` | 1 hour | Graph computation (Spark GNN) → Redis | Graph connectivity between fraudulent accounts |
| Merchant risk profile | `batch` | daily | Merchant history → Redis | High-chargeback merchants raise baseline fraud prior |
| Transaction amount vs. merchant category average | `real-time` | request-time | Merchant stats from Redis | $5,000 txn at avg $50 merchant = anomalous |
| User account age and transaction history | `batch` | 1 hour | Spark on transaction history → Redis | New accounts with no history = higher prior |

#### Label Definition
- **Label:** Confirmed fraud (positive) — determined by:
  1. Chargeback with fraud reason code (strongest signal; ~21-day delay)
  2. Card network fraud report (network-reported; 7-day delay)
  3. Real-time rules engine matches (immediate; lower quality)
- **Positive/negative ratio:** ~1:1000 (0.1% fraud rate; extreme class imbalance)
- **Label delay:** Chargebacks arrive 21 days after transaction → training data is inherently lagged; real-time model must generalize from historical patterns
- **Adversarial dynamics (key Stripe-specific consideration):**
  - When the model improves, fraudsters observe their transactions being blocked and adapt their tactics
  - This creates a **distribution shift** that is *adversarially driven* — not random drift
  - Mitigation: monitor for sudden drop in fraud detection on new card BIN ranges or new device fingerprints (indicator of new attack vector); retrain within 24hr of detection
- **Bias risks:**
  - **Geographic bias:** Fraud rates vary dramatically by country; a global model may over-flag transactions from high-fraud-rate countries, blocking legitimate users. Per-region calibration required
  - **Friendly fraud:** Cardholders claiming legitimate charges are fraud (chargeback abuse) creates false positives in training labels → require card network confirmation, not just cardholder claim

#### Storage Engine Choice

| Layer | Engine | Justification |
|---|---|---|
| Real-time velocity features | Redis Cluster (key: user_id → sliding window counters) | Sub-ms reads/writes; INCR + EXPIRE for sliding window; TTL-based expiry |
| Device/IP risk scores | Redis (key: fingerprint_hash → risk_score) | Blocklist + reputation scores; fast lookup at request time |
| Card BIN database | Redis (key: BIN → {country, type, issuer}) | Static-ish; updated daily; fast lookup |
| Network graph features | Redis (precomputed, key: entity_id → feature_vector) | GNN computed offline hourly; served from Redis at request time |
| Merchant risk profiles | Redis (key: merchant_id → {fraud_rate, threshold_config}) | Updated daily; per-merchant threshold configuration stored here |
| Transaction logs | Kafka → Flink → Hive | Streaming ingest; Flink computes velocity features; Hive for training |
| Chargeback labels | Batch pipeline (card network → S3 → Hive join) | 21-day delayed labels joined to transaction features |
| Model artifacts | S3 + model registry | Versioned; frequent retraining (24hr fraud pattern cycle) |

#### Online vs. Offline Split

```
Offline (batch)                                    Online (synchronous, < 100ms)
────────────────────────────────────────────       ──────────────────────────────────────────
Chargeback labels (21-day lag) → Hive              Transaction request arrives
Spark: network graph computation (GNN)             Rules engine: IP blocklist + device blocklist (<1ms)
Spark: velocity feature aggregation (daily)        Redis: velocity counters (1m/1h/24h) (2ms)
Spark GNN: entity graph embedding                  Redis: device fingerprint + IP risk score (1ms)
Daily model retraining on new fraud labels         Redis: network graph features (1ms)
24hr red team cycle: new pattern → retrain         XGBoost model scoring (5ms)
Threshold calibration (per merchant)               Threshold lookup: merchant config (1ms)
Champion/challenger: PSI + AUC gating              Decision: ALLOW / BLOCK / FLAG_REVIEW
                                                   Async: Kafka log + Flink velocity update
```

#### Schema

```
Transaction: {
  transaction_id:    string         # globally unique
  merchant_id:       string
  user_id:           string         # card holder; anonymized
  amount_usd:        float
  card_bin:          string         # first 6 digits of card
  device_fingerprint:string         # hashed browser/device signature
  ip_address:        string
  billing_country:   string
  shipping_country:  string?
  timestamp:         timestamp
}

FraudDecision: {
  transaction_id:    string
  fraud_score:       float           # [0, 1]; probability of fraud
  action:            enum[ALLOW, BLOCK, FLAG_REVIEW]
  triggered_rules:   string[]        # rule IDs from rules engine
  top_features:      FeatureContrib[] # SHAP top-3 for Radar UI
  model_version:     string
  latency_ms:        int
  timestamp:         timestamp
}

NetworkGraphFeature: {
  entity_id:         string          # card/device/IP/email hash
  shared_fraud_count:int             # number of confirmed fraud entities in 2-hop graph
  graph_risk_score:  float           # GNN-computed embedding → risk score
  updated_at:        timestamp
}
```

> **Stripe rubric:** Adversarial dynamics are in the data modeling section (not just "monitoring"). Velocity features at multiple time windows (1m/1h/24h) are standard fraud detection practice — name them explicitly.

---

## 3. ML Pipeline

### 3a. Offline Training

#### Data Pipeline
- **Ingestion:** Kafka (transaction events) → Flink (velocity computation) → Hive; chargeback labels from card networks → S3 → joined to transaction features (21-day lag join)
- **Feature engineering:**
  - Velocity features: Flink sliding window aggregation (1 min, 1 hour, 24 hour) per user/device/IP
  - Network graph: Spark + GraphX or PyTorch Geometric GNN — build entity graph (nodes: cards, devices, IPs, emails; edges: shared usage); GNN propagates fraud signal across 2-hop neighborhood
  - Class rebalancing: SMOTE or focal loss to handle 1:1000 imbalance; prefer focal loss (avoids overfitting to synthetic examples)
- **Train/val/test split:** Time-based — train D-60 to D-1; val D-1; test D-0; never use future data to predict past (temporal leakage is common and catastrophic for fraud models)
- **Orchestration:** Spark jobs + Airflow DAG; idempotent; 24hr cycle for label arrival and retrain

#### Model Architecture

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Rule engine only | < 1ms; deterministic; interpretable | Easily evaded; requires manual maintenance | Stage 1 only; not sufficient |
| Logistic Regression | Interpretable; fast | Cannot capture non-linear velocity patterns | Baseline; rejected for production |
| XGBoost (chosen) | Handles mixed feature types; fast inference (< 5ms); SHAP natively supported; excellent on tabular fraud features | No sequential modeling; graph features must be pre-computed | **Chosen for main scorer** |
| GNN (PyTorch Geometric) | Captures network fraud rings | 100ms+ inference → offline only; features precomputed into Redis | Used offline; features served from Redis |
| Neural network (deep) | Can capture interaction effects | Less interpretable; marginal gain on tabular features; slower | Rejected |

**Selected pipeline:**
1. **Stage 1 — Rule Engine** (< 1ms): IP/device blocklist; velocity hard rules (> 20 txn in 1 min = auto-block); known fraud card BINs
2. **Stage 2 — XGBoost** (< 10ms): Main fraud scorer; input: velocity features + device features + graph features (precomputed) + transaction context
3. **Stage 3 — Threshold routing** (< 1ms): Compare score to merchant-specific threshold → ALLOW / BLOCK / FLAG_REVIEW

**Key XGBoost features (ordered by SHAP importance from empirical data):**
1. `velocity_txn_count_1m` — card testing behavior
2. `network_graph_risk_score` — connected to known fraudulent entities
3. `device_fingerprint_seen_fraud_rate` — this device previously used in fraud
4. `ip_is_proxy_or_vpn` — anonymization attempt
5. `amount_vs_merchant_category_zscore` — unusual amount for this merchant
6. `shipping_billing_country_mismatch` — card-not-present fraud signal
7. `account_age_days` — new accounts are higher risk
8. `card_bin_country_vs_ip_country_mismatch` — geographic inconsistency

**Threshold as business decision (not ML metric):**
- Stripe provides a Pareto frontier visualization to merchants: at threshold T, expected fraud rate = X% and false positive rate = Y%
- Merchants choose their operating point based on their business model
- High-value merchants (jewelry, electronics): lower threshold (block more → fewer fraud losses)
- High-volume low-margin merchants (digital goods): higher threshold (block less → fewer false positives → higher conversion)

#### Training Infrastructure
- **Framework:** XGBoost (CPU cluster; GBT is CPU-efficient)
- **Scale:** 24× CPU nodes, ~30min/full retrain on 60 days of transaction history
- **Focal loss:** `FL(p) = -α(1-p)^γ log(p)`, γ=2, α=0.25 — down-weights easy negatives (most legitimate transactions); focuses learning on hard-to-classify cases
- **Eval metrics:** AUC-ROC (overall); AUC-PR (precision-recall; more informative for imbalanced class); Fraud Detection Rate at 0.1% false positive rate (FDR@FPR0.1%); SHAP stability (feature importance should be consistent across runs)

---

### 3b. Online Serving

#### Inference Path

```
Transaction Request (merchant → Stripe API)
  → Payment Gateway (auth, merchant lookup)
  → Fraud Scoring Service (inline, synchronous)
      ├─ Stage 1: Rules Engine
      │    ├─ Redis: IP/device blocklist lookup (<1ms)
      │    └─ Hard velocity rule check (> 20 txn/min) (<1ms)
      │    → If hit: BLOCK immediately
      ├─ Stage 2: Feature Assembly
      │    ├─ Redis: velocity counters (1m/1h/24h) (2ms)
      │    ├─ Redis: device fingerprint risk score (1ms)
      │    ├─ Redis: network graph risk score (1ms)
      │    ├─ Redis: card BIN features (1ms)
      │    └─ Redis: merchant risk profile + threshold config (1ms)
      ├─ Stage 3: XGBoost Scoring
      │    └─ Assembled feature vector → XGBoost (5ms)
      ├─ Stage 4: Threshold Routing
      │    └─ Score vs. merchant threshold → ALLOW / BLOCK / FLAG_REVIEW (1ms)
      └─ Response to payment gateway + async:
           Kafka: log transaction + score (velocity update via Flink)
           Radar UI: SHAP features if FLAG_REVIEW
```

#### Latency Budget Breakdown

| Step | p50 | p99 | Notes |
|---|---|---|---|
| Rules engine (Stage 1) | < 1ms | 1ms | Cache hit; deterministic |
| Redis feature fetch (all) | 4ms | 10ms | Parallel reads; 5 Redis calls |
| XGBoost scoring | 3ms | 8ms | CPU; model loaded in memory |
| Threshold routing + SHAP (if review) | 1ms | 5ms | SHAP only for FLAG_REVIEW path |
| Network + serialization | 3ms | 10ms | gRPC |
| **Total** | **12ms** | **34ms** | Budget: ≤ 100ms p99 ✓ (significant headroom) |

#### Caching Strategy
- **No caching of fraud decisions:** Each transaction is unique; caching would allow adversaries to test cached results
- **Model in-memory:** XGBoost model (~50MB) loaded in RAM at startup; no disk reads at serving time
- **Velocity counters:** Written to Redis by Flink within 1s of transaction processing; reading counters at serve time always reflects the latest transactions including the current one (INCR before scoring)

---

### 3c. Monitoring

> **Designed upfront — adversarial dynamics make fraud monitoring fundamentally different from standard ML drift monitoring.**

#### Drift Detection

| Signal | Method | Threshold | Action |
|---|---|---|---|
| Fraud detection rate (FDR) by card BIN range | Rolling 24hr vs. 7-day baseline | > -20% relative on any BIN cluster | Alert: new fraud attack vector; retrain within 24hr |
| False positive rate | Rolling 24hr complaint/dispute rate | > 0.2% (2× baseline) | Threshold recalibration; investigate feature drift |
| Velocity feature distribution (1m window) | PSI daily | PSI > 0.3 on any velocity bucket | Check Flink pipeline; possible lag |
| Network graph risk score distribution | Rolling mean ± 2σ | > 25% shift sustained 2hr | GNN recomputation trigger |
| Adversarial pattern emergence | Clustering of low-score fraud (score < 0.3 but labeled fraud later) | > 50 confirmed fraud cases with low score in 24hr | Red team analysis; emergency retrain |
| p99 latency | Real-time APM | > 100ms sustained 2min | Circuit breaker |

#### Shadow Scoring
- Challenger model runs on 5% of transactions (decisions logged, not acted on; challenger decision applied only if it agrees with champion)
- Comparison cadence: daily; metrics: AUC-PR, FDR@FPR0.1%, SHAP feature stability
- Promotion: challenger must show improvement on adversarial test set (last 2 weeks of red team examples) + no FPR regression + no latency regression

#### A/B Holdout Design
- **Unit:** Not standard user-level A/B — fraud models cannot have holdouts where known fraud passes through
- **Approach:** **Retrospective holdout** — 5% of decisions are independently scored by both champion and challenger on historical data; live traffic uses champion only
- **Primary metric:** AUC-PR on confirmed fraud labels (available after 21-day chargeback lag)
- **Guardrail:** False positive rate (merchant complaint rate proxy), p99 latency

> **Stripe rubric:** Standard A/B is not appropriate for fraud — letting fraud through as a "control" is unacceptable. Retrospective evaluation on historical data is the correct approach.

---

## 4. Failure Modes

| Scenario | Impact | Mitigation | Fallback |
|---|---|---|---|
| Redis unavailable (velocity features) | No real-time velocity context | Retry ×1 (5ms timeout); circuit breaker | Rule engine only + XGBoost with zero-filled velocity features; increased BLOCK rate for high-risk merchants |
| XGBoost model loading failure | No ML scoring | Auto-reload from S3 (< 1s); circuit breaker on persistent failure | Rules engine only; strict rule thresholds applied (block more) |
| Flink velocity pipeline lag (> 1min) | Stale velocity counters | Monitor Kafka consumer lag; alert at > 30s lag | Use last-known counters + flag decision as "degraded confidence" in logs |
| GNN graph feature computation failure | Network graph features unavailable | GNN runs offline (hourly); if computation fails, Redis values stay stale | XGBoost uses last valid graph features from Redis (acceptable up to 2hr staleness) |
| New fraud pattern not in training data | Model fails to detect novel attack | 24hr SLA: detect via FDR anomaly → red team analysis → emergency retrain | Stricter rule engine fallback; alert fraud ops team for manual review |
| Card network outage (chargeback labels delayed) | Training labels not arriving; model goes stale | Alert at 48hr without new labels; halt scheduled retrains | Current champion stays live; no degradation in serving; only training pipeline affected |

#### Circuit Breaker Thresholds
- **Error rate:** Trip at > 0.1% 5xx (very sensitive — fraud system must not silently fail) → rule engine only; block all high-risk transactions; alert on-call immediately
- **Latency:** Trip at p99 > 120ms sustained 2min → bypass XGBoost; rules-only with strict thresholds
- **Recovery:** Half-open after 15s; restore ML path on 3 consecutive successes; monitor FDR for 5 min post-recovery

#### Degraded-Mode Behavior
1. **Level 1** — Full pipeline: Rules + XGBoost + Graph features — full accuracy
2. **Level 2** — Rules + XGBoost (no graph features; last valid graph values) — ~5% lower FDR on network fraud rings
3. **Level 3** — Rules engine only — lowest accuracy; strict thresholds to compensate; high false positive rate acceptable in emergency

> **Stripe rubric:** Fail-safe stance: when in doubt, BLOCK (especially for high-value transactions). The cost of a false positive (merchant friction) is lower than the cost of a false negative (fraud loss + chargeback fee + card network penalty).

---

## 5. Capacity Estimates

> **Assumptions:**
> - Transactions/day: 1B (Stripe scale)
> - Peak QPS: 500K (Black Friday global; ~6× average)
> - Transaction payload: ~2KB (card details, merchant info, device fingerprint)
> - XGBoost model size: 50MB (loaded in RAM)
> - Redis velocity counters: 3 windows × 3 metrics (count, amount, distinct merchants) = 9 counters per user; 1M active users in peak window
> - Network graph: ~100M entities (cards, devices, IPs, emails); GNN embedding 64-d = ~25GB
> - Log retention: 7 years (PCI-DSS compliance)

#### Estimates

| Metric | Calculation | Result |
|---|---|---|
| Avg QPS | 1B / 86,400 | **~11,600 QPS** |
| Peak QPS | 11,600 × 6× Black Friday | **~70K QPS** (conservative; stated 500K is absolute peak capacity target) |
| Redis reads/s (velocity, peak) | 70K QPS × 5 Redis reads | **~350K reads/s** |
| Redis velocity counter storage | 1M active users × 9 counters × 8B | **~72MB** (trivially small) |
| Redis graph feature storage | 100M entities × 64-d × 4B | **~25GB** (per Redis cluster; replicated ×3) |
| Kafka ingest throughput | 70K events/s × 2KB/event | **~140MB/s → ~12TB/day** |
| Training data (60 days, chargeback join) | 1B transactions/day × 60 days × 500B features | **~30TB** |
| XGBoost serving replicas | 70K QPS / 5,000 QPS per CPU core (fast model) | **~14 CPU cores** → **~4 CPU nodes** (with headroom) |
| Flink velocity computation workers | 70K events/s / 10K events per worker | **~7 Flink workers** |
| Network graph GNN recomputation | 100M entities × 1 hop neighbors × compute | **~2hr on 32-node Spark cluster (hourly batch)** |

---

## 6. Open Questions & Follow-ups

#### Unresolved Design Decisions
- [ ] **Real-time GNN inference:** If GNN could run in < 50ms (e.g., GraphSAGE with mini-batch inference), graph features could be computed at request time rather than precomputed hourly. This would detect real-time fraud ring expansion. Current batch approach misses 1-hour window
- [ ] **Friendly fraud (chargeback abuse):** Training labels include both genuine fraud and friendly fraud (legitimate transactions disputed by cardholders). Should these be separate labels? Friendly fraud requires different features (dispute history, merchant return policy) and different model
- [ ] **Merchant-specific models:** High-volume merchants (e.g., Amazon) could have dedicated per-merchant models. Trade-off: better accuracy vs. cold-start problem for new merchant categories
- [ ] **PCI-DSS compliance for features:** Storing card numbers and BINs in Redis requires PCI-DSS scope for the feature store. Tokenized card IDs used in production — verify legal team has reviewed feature storage design

#### Company Rubric Gaps (self-assessment)
- [x] Requirements: adversarial dynamics stated; threshold as business decision (not ML metric)
- [x] Data Modeling: velocity features at 3 time windows; GNN for network fraud rings; chargeback label delay acknowledged
- [x] ML Pipeline: focal loss for class imbalance; SHAP for Radar UI; 24hr adversarial retrain SLA
- [x] Failure Modes: fail-safe stance; 3-level degraded mode; no A/B in live traffic (retrospective only)
- [x] Capacity: PCI-DSS retention (7 years) noted; GNN storage calculated

#### Recommended Follow-up Problems
- Authorization Optimization (Stripe) — second documented Stripe question; minimize decline rates while controlling fraud
- Uber Surge Pricing — same real-time ML infrastructure pattern but different domain

---

## 7. References

| Resource | Type | Relevance |
|---|---|---|
| Stripe Engineering Blog: "How we built Stripe Radar" | Blog | Radar production architecture; rule engine + ML; merchant threshold customization |
| Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System" (KDD 2016) | Paper | XGBoost foundation; handles imbalanced classes with scale_pos_weight |
| Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017) | Paper | Focal loss for extreme class imbalance; directly applicable to fraud (1:1000 ratio) |
| Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (SHAP, NeurIPS 2017) | Paper | SHAP for fraud feature explanation in Radar UI |
| Hamilton et al., "Inductive Representation Learning on Large Graphs" (GraphSAGE, NeurIPS 2017) | Paper | GNN for network fraud ring detection; scalable to 100M entities |
| Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique" (JAIR 2002) | Paper | Class imbalance handling; alternative to focal loss |
| Zhang et al., "FRAUDAR: Bounding Graph Fraud in the Face of Camouflage" (KDD 2016) | Paper | Graph-based fraud detection; camouflage-resistant algorithms |
| Bolton & Hand, "Statistical Fraud Detection: A Review" (Statistical Science 2002) | Paper | Survey of fraud detection methods; velocity features; outlier detection |
| Kou et al., "Survey of Fraud Detection Techniques" (2004) | Paper | Taxonomy of fraud detection approaches; neural network + rule-based hybrid |
| PCI Security Standards Council: PCI-DSS v4.0 | Standard | Data retention requirements (7 years); scope for feature stores containing card data |
