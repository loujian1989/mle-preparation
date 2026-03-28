# Uber ETA Prediction — ML System Design

**Domain:** `real-time-ml`
**Target Company:** Uber
**Difficulty Bar:** L6 (E6)
**Date:** 2026-03-27
**Related Designs:** `../templates/ml-system-design-template.md`

---

## 0. Scorecard

| Axis | Score | Weakest Point |
|---|---|---|
| Requirements | ★★★★★ | — |
| Data Modeling | ★★★★★ | — |
| ML Pipeline | ★★★★★ | — |
| Failure Modes | ★★★★☆ | Cross-city cascade effects partially addressed |
| Capacity | ★★★★★ | — |

**Overall:** `STRONG HIRE`
**Top Gap:** Cascading marketplace effects — probe how a stale ETA model in one city affects pricing, dispatch, and driver supply signals downstream.

---

## 1. Requirements

#### Functional Requirements
1. Predict ETA (estimated time of arrival) from driver's current location to pickup point, and from pickup to destination
2. Provide prediction intervals (p10/p50/p90) — not just a point estimate
3. Support real-time updates: ETA refreshes every 30s during an active trip as conditions change
4. Handle city-level heterogeneity: traffic patterns, road networks, and demand differ radically across cities

#### Non-Functional Requirements

| Dimension | Target | Business Tie |
|---|---|---|
| Latency (p99 serving) | ≤ 200ms | ETA shown before rider confirms trip; beyond 200ms, conversion drops |
| Availability | 99.99% | ETA unavailability = riders cannot price/confirm trips = direct revenue loss |
| Consistency | Eventual (seconds) | Real-time traffic updates propagate within 30s; strong consistency unneeded |
| Throughput | ~50K peak QPS | 30M DAU × ~15 ETA checks/day / 86,400 × 2× peak multiplier (rush hour) |
| Feature freshness (traffic) | ≤ 30s | Traffic conditions change in minutes; 30s budget for sensor → feature store |
| Feature freshness (supply/demand) | ≤ 500ms | Driver supply and surge zone signals must be near-real-time for accurate ETA |

#### Scale Numbers (stated upfront)
- **DAU / MAU:** 30M DAU riders / 100M+ MAU
- **Active drivers (peak):** ~5M globally
- **Peak QPS:** ~50K (rush hour across all cities globally)
- **Cities covered:** 10,000+ cities; 70+ countries
- **ETA refreshes during active trips:** ~200M/day (every 30s per active trip)

#### Out of Scope
- Route planning and navigation (separate system; ETA consumes route graph output)
- Dynamic pricing / surge calculation (consumes ETA as input, not designed here)
- Driver dispatch and matching (consumes ETA as input, not designed here)

> **Uber rubric:** ETA is a cascading dependency — stale or wrong ETA flows into pricing → dispatch → driver supply → marketplace equilibrium. State this dependency graph explicitly; Uber will probe it.

---

## 2. Data Modeling

#### Input Signals

| Feature | Type | Freshness | Source | Notes |
|---|---|---|---|---|
| H3 hexagonal grid traffic speed (resolution 8) | `real-time` | ≤ 30s | Sensor network → Kafka → Flink → Redis | H3 res-8 = ~460m hexagons; ~21M hexagons globally |
| Driver GPS trajectory (last 5 positions) | `real-time` | ≤ 5s | Driver app → Kafka → Redis | Speed, heading, acceleration — short-term momentum features |
| Road segment historical speed distribution | `batch` | daily | Aggregated GPS traces → Hive → feature store | P10/P50/P90 speed by hour-of-day, day-of-week, weather |
| Time features | `real-time` | request-time | Request context | Hour-of-day, day-of-week, holiday flag — major traffic predictors |
| Weather (precipitation, visibility) | `batch` | 15 min | Weather API → Kafka → Redis | Rain adds 15–25% to ETA in empirical data |
| Route distance and turn count | `real-time` | request-time | Routing engine (OSRM) output | Baseline ETA before ML adjustment |
| H3 supply/demand imbalance | `real-time` | ≤ 500ms | Surge pricing system → Kafka → Redis | High demand zones → slower pickup |
| City-level ETA bias (calibration offset) | `batch` | daily | Per-city model fine-tuning residuals | Corrects for systematic under/over-prediction in specific cities |
| Driver type (car, XL, Black, moto) | `static` | per-trip | Trip request | Vehicle type affects speed profile |

#### Label Definition
- **Label (primary):** Actual trip duration (seconds) — difference between trip start and completion timestamps
- **Label (interval targets):** P10 actual, P50 actual, P90 actual — computed from empirical distribution of similar trips
- **Collection strategy:** Every completed trip generates a label; ~15M completed trips/day
- **Positive/negative ratio:** N/A — regression task, not classification
- **Label delay:** Trip completion observed at trip end (~10 min average lag from prediction time)
- **Bias risks:**
  - **Censoring bias:** Cancelled trips have no completion label; their features may indicate unusual conditions → undersample edge cases. Mitigate: train also on partial trip observations
  - **City heterogeneity bias:** Model trained on NYC traffic patterns performs poorly in Lagos or Mumbai; city-level fine-tuning required
  - **Rush hour underrepresentation:** If training data is time-uniform, rush hour trips are undersampled relative to their real-world frequency → stratified sampling by hour-of-day

#### Storage Engine Choice

| Layer | Engine | Justification |
|---|---|---|
| Real-time traffic features (H3 grid) | Redis Cluster (geospatial keys: H3 index → speed vector) | Sub-ms reads; H3 key enables range queries on adjacent hexagons |
| Driver GPS state | Redis (TTL: 30s; key: driver_id) | Very high write rate; short TTL; Redis pub/sub for real-time updates |
| Historical speed distributions | Cassandra (key: (h3_id, hour_bucket, day_type)) | Wide rows; efficient time-partitioned reads; city-level sharding |
| Feature store (offline) | Hive on HDFS (partitioned by city, date) | Analytical workloads; Spark-compatible; petabyte-scale trip history |
| Training data | Parquet on S3 (partitioned by city + date) | Columnar; Spark jobs for feature assembly |
| Logs / labels | Kafka → Flink → Hive | Streaming label generation; Flink joins trip start + completion events |
| Model artifacts | S3 + Michelangelo model registry | Versioned; per-city model variants tracked |

#### Online vs. Offline Split

```
Offline (batch, Spark)                          Online (real-time, < 200ms budget)
──────────────────────────────────────          ──────────────────────────────────────────
GPS traces → H3 aggregation → Hive              Request: (pickup_lat/lng, dropoff_lat/lng)
Spark: historical speed distributions           OSRM routing engine: compute route + baseline ETA
Spark: city-level calibration offsets           Redis: fetch H3 traffic features along route (10ms)
Spark: driver GPS trajectory aggregation        Redis: fetch supply/demand + weather (5ms)
Daily model training (global + per-city FT)     GBR model: predict p10/p50/p90 intervals (15ms)
Eval: MAE, MAPE, interval coverage rate         Calibration layer: apply city offset
Champion/challenger per city                    Response + async Kafka log
```

**H3 spatial indexing rationale:** Uber's H3 hexagonal grid (open-sourced) enables:
- Efficient neighbor lookups (each hexagon has exactly 6 neighbors — no corner artifacts vs. square grids)
- Resolution tuning: resolution 8 (~460m) for traffic speed; resolution 6 (~4km) for surge zone modeling
- Consistent indexing across cities without coordinate system differences

#### Schema

```
H3TrafficCell: {
  h3_index:         string       # H3 cell ID at resolution 8
  city_id:          string       # for sharding and city-level model routing
  speed_p10:        float        # km/h, current 30s window
  speed_p50:        float
  speed_p90:        float
  updated_at:       timestamp    # must be ≤ 30s old at serving time
}

Trip: {
  trip_id:          string
  city_id:          string
  pickup_h3:        string       # H3 index of pickup point
  dropoff_h3:       string
  route_h3_cells:   string[]     # ordered H3 cells along route
  vehicle_type:     enum[UberX, XL, Black, Moto]
  requested_at:     timestamp
  started_at:       timestamp
  completed_at:     timestamp    # label source
  predicted_p50:    float        # stored for calibration analysis
}
```

> **Uber rubric:** H3 hexagonal indexing named explicitly. City-level heterogeneity reflected in schema (city_id as first-class field; per-city model routing). Sub-second feature freshness for supply/demand signals (≤ 500ms) stated.

---

## 3. ML Pipeline

### 3a. Offline Training

#### Data Pipeline
- **Ingestion:** GPS events + trip completion events → Kafka → Flink (real-time joins) → Hive (daily snapshots)
- **Feature engineering:**
  - H3 speed aggregation: mean/p10/p50/p90 speed per H3 cell × hour_bucket × day_type (Spark, daily)
  - Route feature extraction: for each training trip, fetch H3 cells along route → compute speed distribution statistics across route cells
  - Weather join: historical weather API data joined to trip records by city × timestamp
- **Train/val/test split:** Time-based per city — train D-90 to D-1, val D-1 to D-0, test holdout D-0; never cross city boundaries in evaluation (prevent data leakage across cities with correlated events)
- **Pipeline orchestration:** Michelangelo DAG; idempotent city-partitioned Spark jobs; each city trained independently for fine-tuning

#### Model Architecture

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Linear regression on route distance | Trivially fast; interpretable | Cannot capture non-linear traffic dynamics or city heterogeneity | Baseline only (used as Michelangelo fallback) |
| Gradient Boosted Quantile Regression (chosen) | Natively outputs p10/p50/p90; handles mixed feature types; fast inference | No sequential/spatial structure; city fine-tuning needed | **Chosen** |
| LSTM over GPS trajectory | Captures momentum in driver movement | Slow inference; marginal improvement over GBR at L6 QPS | Rejected for online path |
| Graph Neural Network on road network | Rich spatial structure | 100–500ms inference per request; exceeds budget | Research direction; not production-ready |

**Selected: Gradient Boosted Quantile Regression (GBR)**
- Three heads: predict p10 ETA, p50 ETA, p90 ETA simultaneously (pinball loss: `L_α = α × (y - ŷ)₊ + (1-α) × (ŷ - y)₊`)
- Architecture: LightGBM with quantile objective; city-level fine-tuning on top of global model (transfer learning via warm-start)
- **Key features:** route_distance_km, n_turns, n_traffic_lights, avg_h3_speed_p50 along route, speed_variance_along_route, hour_of_day, day_of_week, precipitation_mm, supply_demand_ratio at pickup H3
- **City heterogeneity handling:**
  - Global model: trained on all cities — captures universal patterns (time-of-day, weather, route complexity)
  - City fine-tune: 10 epochs warm-start fine-tuning on city-specific data — captures local road culture (lane discipline in Mumbai, roundabout-heavy cities, etc.)
  - City routing: at serving time, lookup city_id → select city-specific model variant

**Interval coverage validation:** Target: P10 prediction covers ≤ 10% of actual trips (undershoot), P90 covers ≥ 90%. Miscalibrated intervals directly hurt rider trust ("my trip was supposed to take 8–15 min, why did it take 22 min?").

#### Training Infrastructure
- **Framework:** LightGBM (CPU-based; GBR does not benefit from GPU)
- **Scale:** 1 global model (CPU cluster, ~2hr/run on full trip history); per-city fine-tune ~5 min/city × 10,000 cities = parallelized across city-sharded Spark workers
- **Michelangelo integration:** Feature definitions shared with Uber Pricing, Dispatch, and Driver Allocation — the same H3 traffic features feed multiple downstream models
- **Eval metrics:** MAE (mean absolute error in seconds), MAPE (mean absolute percentage error), Interval Coverage Rate (ICR) for p10/p90, city-level calibration residuals

---

### 3b. Online Serving

#### Inference Path

```
Client (Rider App) → API Gateway
  → ETA Service
      ├─ Routing Engine (OSRM/H3-aware)
      │    └─ Compute route → list of H3 cells along path + baseline distance ETA
      ├─ Feature Fetch
      │    ├─ Redis: H3 traffic speed for each cell along route  (10ms)
      │    ├─ Redis: H3 supply/demand imbalance at pickup       (3ms)
      │    └─ Redis: weather at city                            (2ms)
      ├─ Feature Assembly
      │    └─ Aggregate H3 cells into route-level statistics    (2ms, CPU)
      ├─ GBR Model Inference
      │    └─ City-specific LightGBM model → p10/p50/p90        (5ms, CPU)
      ├─ Calibration Layer
      │    └─ Apply per-city offset + recency correction        (1ms)
      └─ Response: {eta_p10, eta_p50, eta_p90}
           + async: log prediction to Kafka for label join
```

#### Latency Budget Breakdown

| Step | p50 | p99 | Notes |
|---|---|---|---|
| Routing engine (H3 path) | 10ms | 30ms | OSRM; H3 cell list computed in-process |
| Feature fetch (Redis) | 5ms | 15ms | Parallel reads for all cells along route |
| Feature assembly | 2ms | 5ms | CPU-side aggregation |
| GBR model inference | 3ms | 8ms | LightGBM; city model loaded in memory |
| Calibration + post-processing | 1ms | 3ms | Simple arithmetic |
| Network + serialization | 5ms | 15ms | gRPC + protobuf |
| **Total** | **26ms** | **76ms** | Budget: ≤ 200ms p99 ✓ (significant headroom) |

#### Caching Strategy
- **Route-level ETA cache:** Cache (pickup_h3, dropoff_h3, time_bucket) → (p10, p50, p90); TTL: 60s
- **Cache hit rate target:** ~30% (many unique origin-destination pairs; time_bucket = 5-min window reduces granularity)
- **H3 traffic cache in Redis:** Written by Flink stream processor every 30s; Redis pub/sub notifies model server of updates → invalidation-free (TTL-based refresh)

---

### 3c. Monitoring

> **Designed upfront — "how do you know ETA is accurate in prod?" is a required Uber answer.**

#### Drift Detection

| Signal | Method | Threshold | Action |
|---|---|---|---|
| ETA prediction error (MAPE) by city | Daily comparison vs. 7-day rolling baseline | > 10% relative degradation | Retrain city-specific model |
| Interval coverage rate (ICR) | Daily: % of trips where actual ≤ predicted p90 | ICR < 85% (should be ≥ 90%) | Recalibrate city offset |
| H3 traffic feature drift | PSI on speed distribution by hour bucket | PSI > 0.3 | Alert; check sensor pipeline |
| Model score distribution | Rolling mean ± 2σ on p50 predictions | > 20% shift sustained 1hr | Shadow model comparison |
| Downstream metric: Rider satisfaction | NPS score correlation with ETA error | > 5% relative NPS drop | Escalate to PM + on-call |
| p99 latency | Real-time APM | > 200ms sustained 5min | Circuit breaker |

#### Shadow Scoring
- Challenger model (e.g., new city fine-tuning approach) runs on 5% of trips; actual vs. predicted logged
- Comparison cadence: daily evaluation on shadow trips; metrics: MAE, ICR, city-specific residuals
- Promotion criteria: lower MAPE + better ICR (coverage rate ≥ 90%) + no p99 latency regression

#### A/B Holdout Design
- **Unit of randomization:** `city_id` (not user_id — ETA accuracy affects all users in a city equally)
- **Holdout:** 2% of cities (50 cities) permanently use baseline model — long-term accuracy guardrail
- **Treatment:** New city model or global model update on treatment cities
- **Primary metric:** P50 ETA MAPE (mean absolute percentage error)
- **Guardrail metrics:** P90 interval coverage rate, rider satisfaction score, cancellation rate (high ETA variance drives cancellations)
- **Duration:** Minimum 1 week (accounts for day-of-week traffic patterns)

> **Uber rubric:** Prediction intervals validated in monitoring (ICR metric). City-level heterogeneity addressed in holdout design (randomize by city, not user). Downstream cascade effects (rider satisfaction, cancellation rate) are guardrail metrics.

---

## 4. Failure Modes

| Scenario | Impact | Mitigation | Fallback |
|---|---|---|---|
| Redis H3 traffic cache stale (> 30s lag) | ETA uses outdated traffic conditions | Monitor Kafka consumer lag; alert at > 30s; Flink auto-restarts on failure | Historical average speed for this H3 cell × time-of-day bucket (cold cache path) |
| Routing engine (OSRM) unavailable | No route → no H3 cell list | Retry ×2 (50ms timeout); fallback to cached common routes | Straight-line distance heuristic + historical city average speed → rough ETA; mark as approximate |
| City model unavailable (disk/load failure) | City-specific accuracy lost | Load global model as fallback; log incident | Global model serves all cities; accuracy degrades ~8% on average (city-specific patterns lost) |
| Label pipeline delay (trip completion lag) | Training data gap for affected cities | Monitor trip completion event lag; alert at > 2hr lag | Log gap; backfill on recovery; block retraining for affected cities until resolved |
| Cascading effect: ETA → Pricing | Stale ETA causes incorrect surge calculation | Pricing system has its own 5-min ETA cache; surge recalculates when ETA delta > 20% | Pricing uses last valid ETA + manual surge override by ops team |
| Cascading effect: ETA → Dispatch | Wrong ETA → driver dispatched too late/early | Dispatch system monitors ETA confidence (p90 - p10 spread); widens matching radius when spread > 5 min | Dispatch uses conservative (p90) ETA as planning horizon |

#### Circuit Breaker Thresholds
- **Error rate:** Trip at > 2% 5xx over 60s → serve cached ETA + historical baseline; alert on-call
- **Latency:** Trip at p99 > 250ms sustained 3min → bypass ML model; serve straight-line distance heuristic
- **Recovery:** Half-open after 60s; probe 5% traffic; full restoration on 3 consecutive successes

#### Degraded-Mode Behavior
1. **Level 1** — Cached route-level ETA (Redis, up to 60s stale) — personalized but not real-time
2. **Level 2** — Global model (no city fine-tune) — reduced accuracy, still ML-based
3. **Level 3** — Historical average ETA (city × origin H3 × destination H3 × hour_bucket) — rule-based, no real-time signals

> **Uber rubric:** Cascading marketplace effects explicitly addressed. ETA → Pricing and ETA → Dispatch failure propagation stated with mitigations.

---

## 5. Capacity Estimates

> **Assumptions:**
> - DAU: 30M riders + 5M active drivers (peak)
> - ETA requests: 15/day average for riders; 200M trip refresh calls/day
> - Peak QPS: 50K (rush hour overlap across time zones)
> - Route H3 cells per trip: ~30 cells on average (urban trip)
> - GBR model size per city: ~50MB; global model: ~200MB
> - Redis H3 traffic cache: 21M H3 cells globally at resolution 8 × 3 float values (p10/p50/p90) × 4B = ~252MB — trivially fits in memory
> - Log retention: 90 days hot (Hive), 3 years cold (S3 Glacier — regulatory compliance)

#### Estimates

| Metric | Calculation | Result |
|---|---|---|
| Avg QPS (total ETA requests) | (30M × 15 + 200M) / 86,400 | **~7,500 QPS avg** |
| Peak QPS | 7,500 × 2× rush hour multiplier | **~15K QPS** (well within 50K capacity target) |
| Redis reads/s (H3 traffic, 30 cells/route) | 15K QPS × 30 reads (batched) | **~450K H3 lookups/s → 15K batch calls/s** |
| H3 traffic cache size (Redis) | 21M H3 cells × ~100 bytes | **~2.1GB** (trivially in memory) |
| Trip event ingest (Kafka) | 15K QPS × 5 events/request + 200M refreshes/day | **~75K events/s → ~7.5GB/hr** |
| Training data (daily) | 15M trips × 2KB features/trip | **~30GB/day** (compressed ~6GB) |
| City model storage | 10K cities × 50MB | **~500GB** (S3; hot-loaded into memory for active cities) |
| Serving replicas (ETA service) | 15K QPS / 1,000 QPS per node (GBR is CPU) | **~15 CPU nodes** (with 2× headroom: 30 nodes) |
| Flink workers (H3 aggregation) | 75K events/s / 10K events per worker | **~8 Flink workers** |

---

## 6. Open Questions & Follow-ups

#### Unresolved Design Decisions
- [ ] **Real-time ETA updates during trip:** Every 30s re-prediction creates 200M refresh calls/day. Should we shift to event-driven updates (only re-predict when H3 cell speed changes > 20%) to reduce load?
- [ ] **Graph Neural Network for road network:** GNN on the road graph could capture intersection-level dynamics (traffic light timing, turn penalties). Viable if inference < 50ms — currently a research direction
- [ ] **Driver behavior features:** Experienced drivers in a city are 15% faster than new drivers on the same route. Should driver_tenure or historical_speed_ratio be added as features? Privacy implications need legal review
- [ ] **Weather forecast (not just current):** For trips > 20 min, weather at completion time may differ from weather at request time. Forward-looking weather API integration could improve long-trip accuracy

#### Company Rubric Gaps (self-assessment)
- [x] Requirements: prediction intervals (not point estimates) as first-class requirement
- [x] Data Modeling: H3 hexagonal indexing; city-level heterogeneity in schema; sub-second supply/demand freshness
- [x] ML Pipeline: quantile regression for intervals; city fine-tuning; ICR monitoring
- [x] Failure Modes: cascading effects to Pricing and Dispatch explicitly addressed
- [x] Capacity: H3 cache size calculated; CPU replica count for GBR (not GPU)

#### Recommended Follow-up Problems
- Uber Surge Pricing — consumes ETA as input; adds demand/supply forecasting and auction mechanism
- Uber Driver-Rider Matching — consumes ETA; adds combinatorial optimization

---

## 7. References

| Resource | Type | Relevance |
|---|---|---|
| Uber Engineering Blog: "DeepETA: How Uber Predicts Arrival Times Using Deep Learning" (2022) | Blog | Production DeepETA architecture; H3 features; city-level modeling |
| Uber Engineering Blog: "H3: Uber's Hexagonal Hierarchical Spatial Index" (2018) | Blog | H3 spatial indexing library; resolution levels; neighbor lookup |
| Uber Engineering Blog: "Meet Michelangelo: Uber's Machine Learning Platform" (2017) | Blog | Feature store architecture; model registry; cross-domain feature reuse |
| Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" (NeurIPS 2017) | Paper | LightGBM used for GBR; quantile regression objective |
| Koenker & Bassett, "Regression Quantiles" (Econometrica 1978) | Paper | Foundational quantile regression; pinball loss derivation |
| Uber Engineering Blog: "Forecasting at Uber: An Introduction" | Blog | Time-series forecasting at Uber; feature engineering for temporal patterns |
| Brockwell & Davis, "Introduction to Time Series and Forecasting" | Book | Background on temporal feature engineering and validation (time-based splits) |
| Uber Engineering Blog: "How Uber Predicts Surge Pricing" | Blog | How ETA feeds into surge; cascading dependency illustration |
| Janowicz et al., "GeoAI: Spatially Explicit Artificial Intelligence" (2020) | Paper | Geospatial ML overview; H3-style spatial indexing context |
| Salinas et al., "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks" (2020) | Paper | Probabilistic forecasting baseline; comparison with quantile GBR |
