# Uber Supply Pricing — ML System Design

**Domain:** `real-time-ml`
**Target Company:** Uber (Supply Pricing team — Chris Mosch's team)
**Difficulty Bar:** L6 (E6)
**Date:** 2026-03-27
**Related Designs:** `uber-eta-prediction.md`, `../templates/ml-system-design-template.md`

---

## 0. Scorecard

| Axis | Score | Weakest Point |
|---|---|---|
| Requirements | ★★★★★ | — |
| Data Modeling | ★★★★★ | — |
| ML Pipeline | ★★★★★ | — |
| Failure Modes | ★★★★★ | — |
| Capacity | ★★★★★ | — |

**Overall:** `STRONG HIRE`
**Top Gap:** Strategic driver response creates endogeneity — naive supply estimation ignores that supply responds to the price you're trying to set. This must be addressed explicitly.

---

## 1. Requirements

#### Functional Requirements
1. Compute a surge price multiplier per H3 zone in real time, updated every 30–60 seconds
2. Model strategic driver behavior: drivers reposition and time their availability in response to surge
3. Support 1M+ pricing decisions per second globally (served from cache; not model inference)
4. Provide 15-min predictive surge for proactive driver positioning notifications
5. Emit pricing decisions and driver response signals for offline A/B evaluation and counterfactual analysis

#### Non-Functional Requirements

| Dimension | Target | Business Tie |
|---|---|---|
| Serving latency (p99) | ≤ 50ms | Within rider request flow; total request budget ~200ms |
| Surge update cadence | Every 30–60s per zone | Faster = oscillation; slower = stale signal after demand spike |
| Availability | 99.99% | Pricing outage = riders can't confirm trips = direct revenue loss |
| Supply signal freshness | ≤ 500ms | Driver GPS and status must be near-real-time for accurate zone supply count |
| Demand signal freshness | ≤ 30s | Trip request rate is the primary demand signal |
| Throughput (zone updates) | ~10M zone refreshes/min | 10K cities × avg 1,000 H3 zones/city × 1 update/min |

#### Scale Numbers (stated upfront)
- **Rider requests**: 1M+ pricing lookups/sec (cached reads)
- **Active zones globally**: ~10M H3 resolution-7 zones; ~500K with non-trivial activity
- **Driver GPS events**: ~500M events/day globally (~5,000 events/sec)
- **Trip requests**: ~30M rider requests/day (~350/sec average, ~1,500/sec peak)
- **Cities**: 10,000+ globally; top 100 cities drive ~80% of volume

#### Out of Scope
- ETA computation (separate system; this system consumes ETA as input to pricing)
- Driver matching/dispatch (separate system; consumes surge as input)
- Rider-facing fare display (consumes surge multiplier output)

> **Uber rubric:** ETA is a cascading dependency — wrong ETA → wrong demand estimate → wrong surge. State this dependency explicitly: surge is downstream of ETA, upstream of dispatch.

---

## 2. Data Modeling

#### Input Signals

| Feature | Type | Freshness | Source | Notes |
|---|---|---|---|---|
| H3 zone online driver count | `real-time` | ≤ 500ms | Driver GPS → Kafka → Flink → Redis | Approx count via HLL per zone |
| H3 zone trip request rate (last 5 min) | `real-time` | ≤ 30s | Trip request stream → Kafka → Flink → Redis | Rolling window count |
| Driver ETA to zone (supply accessibility) | `real-time` | ≤ 60s | ETA system output → Redis | Drivers within 5 min are "accessible supply" |
| Historical demand by zone × hour-of-week | `batch` | daily | Hive → feature store | Baseline demand expectation; detrend from historical |
| Event calendar (concerts, sports, airports) | `batch` | hourly | Events API → feature store | Demand multiplier for known future demand spikes |
| Weather (precipitation, temperature) | `near-real-time` | 15 min | Weather API → Kafka | Rain increases demand ~20%; reduces driver supply |
| Current surge multiplier (feedback) | `real-time` | ≤ 30s | Previous zone state | Dampening input: don't oscillate around equilibrium |
| Driver response elasticity (per zone) | `batch` | daily | Offline estimation from historical data | How many incremental drivers per 0.25× surge increase |

#### Label Definition
- **Label (surge optimization)**: `optimal_multiplier(zone, time)` = multiplier that minimizes `E[unfulfilled_trips] + α × E[rider_wait_time]` subject to `E[driver_earnings_per_hour] ≥ baseline`
- **Label derivation**: counterfactual — cannot be directly observed. Estimated via switchback experiments (zone alternates surge levels) or structural demand/supply models
- **Proxy labels used in practice**:
  1. `trip_completion_rate(zone, t)` under current surge (directly observable)
  2. `driver_acceptance_rate(zone, t)` (observable; proxy for supply sufficiency)
  3. `rider_wait_time_p90(zone, t)` (observable; proxy for demand/supply balance)

#### Storage Engine Choice

| Layer | Engine | Justification |
|---|---|---|
| Zone surge multiplier (read path) | Redis Cluster (key: h3_zone_id → surge, TTL 90s) | Sub-ms reads; 1M+ req/sec is O(1) cache hit; TTL ensures eventual consistency |
| Driver supply state (per zone) | Redis (HyperLogLog + sorted set for ETA-ranked drivers) | Approximate zone driver count; sorted set enables "N closest drivers" lookup |
| Trip request stream | Kafka (partitioned by H3 zone ID) | High-throughput; Flink consumer aggregates to zone-level demand counts |
| Driver GPS stream | Kafka (partitioned by driver_id) | ~5K events/sec; Flink joins to H3 zone and updates zone supply counts |
| Historical demand/supply patterns | Cassandra (key: (h3_id, hour_bucket, day_of_week)) | Wide rows for time-partitioned reads; city-level sharding |
| Offline feature store | Hive on S3 (partitioned by city, date) | Spark-accessible; driver response elasticity estimation runs nightly |
| Pricing decision log | Kafka → S3 (Parquet) | Full audit trail for counterfactual analysis and A/B evaluation |

---

## 3. ML Pipeline

#### Architecture Overview

```
[Driver GPS stream]   ──→ Kafka ──→ Flink (zone aggregation) ──→ Zone Supply Store (Redis)
[Trip request stream] ──→ Kafka ──→ Flink (zone aggregation) ──→ Zone Demand Store (Redis)
[Event calendar]      ──→                                    ──→ Feature Store
[Weather API]         ──→ Kafka                               ──→ Feature Store

                            Zone Supply + Demand
                                    ↓
                       Surge Computation Engine (30s cadence)
                       ┌──────────────────────────────────────┐
                       │  1. Fetch zone supply/demand signals  │
                       │  2. Apply demand elasticity model     │
                       │  3. Compute raw surplus/deficit       │
                       │  4. Run pricing model → raw multiplier│
                       │  5. Apply EMA dampening               │
                       │  6. Snap to discrete tier             │
                       │  7. Write to Redis surge cache        │
                       └──────────────────────────────────────┘
                                    ↓ (every 30s per zone)
                           Redis Surge Cache (h3_id → multiplier)
                                    ↓ (< 1ms read)
                      [Rider App] ← API Gateway ← Surge Lookup
```

#### Surge Computation Pipeline (detail)

**Step 1 — Zone state assembly**: read from Redis: `online_drivers(zone)`, `trip_requests_last_5min(zone)`, `current_multiplier(zone)`.

**Step 2 — Supply/demand balance**:
```
imbalance = trip_requests_rate(zone) / max(online_drivers(zone), 1)
# imbalance > 1.5 → demand-heavy; < 0.5 → supply-heavy
```

**Step 3 — Raw multiplier**:
- Model input: [imbalance, time_of_day, day_of_week, weather, event_flag, adjacent_zone_surge, historical_demand_deviation]
- Model: LightGBM trained on historical (zone, hour) → optimal_multiplier labels derived from switchback experiments
- Output: continuous multiplier (e.g., 1.73)

**Step 4 — Strategic response adjustment**:
```
supply_response = driver_elasticity(zone) × (raw_multiplier - 1.0)
adjusted_supply = online_drivers(zone) + supply_response
# Re-compute imbalance with adjusted supply → corrected multiplier
```
This corrects for the fact that publishing a high surge will bring more drivers online. Without this step, the system systematically oversurges.

**Step 5 — EMA dampening**:
```
final_multiplier = α × corrected_multiplier + (1 - α) × previous_multiplier
# α = 0.3 prevents oscillation; zone needs 3+ consecutive readings to fully update
```

**Step 6 — Discretization**: snap to `[1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]`. Prevents micro-fluctuations that trigger driver repositioning.

#### 15-Minute Predictive Surge Model

- **Purpose**: send push notifications to drivers "Surge expected in Downtown in 15 min — head there now"
- **Model**: time series forecast per zone — LGBM with lag features (demand rate last 1h in 5-min buckets), event flags, historical demand pattern at this zone × hour-of-week
- **Output**: P(surge > 1.5× in next 15 min per zone)
- **Threshold for notification**: P > 0.7 to avoid alert fatigue
- **Feedback loop concern**: notifications bring drivers → reduces surge probability → notifications were "wrong." This is the Hawthorne effect in reverse. Fix: cap notification volume per zone (don't notify more drivers than the expected supply gap)

#### Driver Response Elasticity Estimation

- **Offline nightly job**: for each (city, zone_type) pair, estimate `ΔS/ΔP` from historical surge events
- **Method**: instrument the surge change itself as quasi-random variation (surge changes are triggered by thresholds — use regression discontinuity at threshold boundaries)
- **Output**: elasticity coefficient per zone type stored in feature store, consumed by Step 4 above
- **Update cadence**: nightly; elasticity changes slowly (driver demographics shift over weeks)

---

## 4. Failure Modes

| Failure | Detection | Mitigation |
|---|---|---|
| ML model unavailable | Model health check; timeout on inference | Fall back to threshold rule: `surge = 1.5× if imbalance > 1.5, else 1.0×` |
| Redis surge cache miss | Cache miss rate alert | Recompute on demand (slower path, still < 50ms for single zone) |
| Kafka consumer lag (stale supply/demand) | Consumer lag metric > 60s | Use last known zone state + exponential decay (assume supply drifts offline over time) |
| Supply signal completely lost | Driver count = 0 for zone > 5 min | Use adjacent zone supply as proxy + alert on-call |
| Oscillation (surge overshoots repeatedly) | Zone multiplier std dev > 0.5 over 10 min | Increase EMA dampening (α → 0.1); root cause usually missing strategic response correction |
| Surge overestimation (too many drivers flood zone) | Driver acceptance rate < 30% in zone | Cap maximum surge increase per interval (max +0.25× per 30s step) |
| Surge underestimation (unmet demand) | Rider wait time p90 > 8 min | Alert; may indicate demand spike not captured by 5-min rolling window; shorten window |
| City-level driver data outage | Supply count = 0 for all zones in city | Fall back to city-level historical surge pattern (day-of-week × hour baseline) |

---

## 5. Capacity Estimates

| Component | Estimate | Assumptions |
|---|---|---|
| Rider surge lookups | 1M req/sec | All served from Redis cache; each lookup = 1 Redis GET |
| Zone update rate | ~170K updates/sec | 500K active zones × 1 update/30s |
| Driver GPS ingestion | 5K events/sec | 5M active drivers × 1 GPS event/1000s average |
| Trip request ingestion | 350–1,500 req/sec | Average–peak globally |
| Redis memory (surge cache) | ~10 GB | 500K zones × 20 bytes/entry |
| Redis memory (driver supply) | ~50 GB | 5M drivers × 10 bytes GPS state |
| Surge computation CPU | ~50 cores | 170K zone updates/sec × 0.3ms/update = ~50 core-seconds/sec |
| Pricing log storage | ~10 TB/month | 500K zones × 2 updates/min × 1 KB/record × 60 × 24 × 30 |

---

## 6. Monitoring & Observability

**Real-time dashboards (5-min resolution):**
- Zone surge distribution (histogram): catches systematic over/under-surging
- Trip completion rate by surge tier: monitors if high surge actually improves completion or just raises price
- Driver acceptance rate by zone: proxy for "was supply actually insufficient?"
- Rider wait time p50/p90 by zone: the metric surge is supposed to improve
- Supply fishing rate: % drivers who came online within 2 min of surge increase (leading indicator of gaming)

**Offline model monitoring (daily):**
- Driver elasticity coefficient drift: if elasticity estimates are stale, Step 4 overcorrects or undercorrects
- Surge prediction calibration: does "predicted surge = 1.5×" actually materialize at 1.5× +/- 0.2?
- A/B experiment analysis: switchback experiment results vs. offline model predictions

**Alerting thresholds:**
- Zone-level wait time p90 > 10 min for > 5 min sustained → P1 (demand/supply severe imbalance)
- System-wide surge cache hit rate < 99.9% → P1 (fallback path activated)
- Kafka consumer lag > 2 min → P2 (supply/demand signals stale)
- Oscillating zones (std dev > 0.5) > 1% of active zones → P2

---

## 7. Key Design Decisions

**Why precomputed zone-level surge vs. per-request inference?**
At 1M req/sec, calling the ML model per request requires 1M inferences/sec. At 10ms/inference that's 10,000 CPU cores. Precomputing at zone level (30s cadence) reduces model calls to ~170K/sec, and serving is a Redis cache lookup (< 1ms). This is the fundamental architectural choice.

**Why EMA dampening instead of setting the model output directly?**
The surge system has a feedback loop: high surge → drivers come online → supply increases → surge falls. Without dampening, the system oscillates. EMA smoothing with α = 0.3 ensures the zone needs 3–4 consecutive high-demand readings before fully surging, preventing response to transient demand spikes.

**Why discrete surge tiers instead of continuous pricing?**
Continuous pricing (1.73×, 1.82×) creates micro-incentives for drivers to hover at zone boundaries to capture marginally higher rates. Discrete tiers reduce boundary gaming, make pricing legible to drivers and riders, and reduce oscillation amplitude.

**Why model driver elasticity offline instead of inferring it online?**
Online elasticity estimation would require A/B testing price levels in real time, which is slow and ethically fraught (different prices for similar riders simultaneously). Offline regression discontinuity on historical threshold-triggered surge changes provides a cleaner estimate without real-time experimentation.
