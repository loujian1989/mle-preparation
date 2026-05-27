# Uber Supply Pricing — ML System Design

**Domain:** `real-time-ml`
**Target Company:** Uber (Supply Pricing team)
**Difficulty Bar:** E6
**Related Designs:** `uber-eta-prediction.md`, `../templates/ml-system-design-template.md`

---

## Overview

**What this system does:** Compute a surge price multiplier per H3 geographic zone every 30–60 seconds, served to 1M+ rider requests per second from a Redis cache. The multiplier balances supply (drivers) and demand (trip requests) in real time.

**Three core design principles to state in the first 2 minutes:**

1. **Surge is market-clearing, not revenue maximization.** The objective is `min E[unfulfilled_trips]` subject to `E[driver_earnings] ≥ baseline` — not max revenue. Revenue is a side effect.
2. **Observed supply is endogenous.** Drivers respond to the surge you're about to publish. Naive supply/demand ratio overestimates scarcity. You must correct for strategic driver response before setting the multiplier.
3. **Oscillation is the primary operational risk.** Surge spikes → drivers flood zone → supply overshoots → surge crashes → drivers leave → repeat. EMA dampening, discrete tiers, and capped per-interval increases are all anti-oscillation controls.

**System dependency chain:**
```
ETA System ──→ Supply Pricing ──→ Dispatch / Matching ──→ Rider App
```
Wrong ETA → wrong demand estimate → wrong surge. Surge is downstream of ETA and upstream of dispatch. State this dependency explicitly.

---

## 1. Requirements

### Functional Requirements

1. Compute a surge price multiplier per H3 zone in real time, updated every 30–60 seconds
2. Model strategic driver behavior: drivers reposition and time availability in response to surge
3. Serve 1M+ pricing lookups per second globally (from cache — not live model inference)
4. Provide 15-min predictive surge for proactive driver positioning push notifications
5. Emit pricing decisions and driver response signals for offline A/B evaluation and counterfactual analysis

### Non-Functional Requirements

| Dimension | Target | Business Tie |
|---|---|---|
| Serving latency (p99) | ≤ 50ms | Within rider request flow; total budget ~200ms |
| Surge update cadence | Every 30–60s per zone | Faster → oscillation; slower → stale after demand spike |
| Availability | 99.99% | Pricing outage = trips blocked = direct revenue loss |
| Supply signal freshness | ≤ 500ms | Driver GPS must be near-real-time for accurate zone supply count |
| Demand signal freshness | ≤ 30s | Trip request rate is the primary demand signal |
| Zone update throughput | ~10M refreshes/min | 10K cities × ~1,000 H3 zones/city × 1 update/min |

### Scale Numbers

- **Rider lookups:** 1M+ req/sec (cached reads)
- **Active zones globally:** ~10M H3 resolution-7 zones; ~500K with non-trivial activity
- **Driver GPS events:** ~500M/day (~5K events/sec)
- **Trip requests:** ~30M/day (~350/sec avg, ~1,500/sec peak)
- **Cities:** 10,000+ globally; top 100 drive ~80% of volume

### Out of Scope

- ETA computation (separate system; consumed as input)
- Driver matching/dispatch (separate system; consumes surge as input)
- Rider-facing fare display (consumes surge multiplier output)

---

## 2. Data Modeling

### Input Signals

| Feature | Type | Freshness | Source | Notes |
|---|---|---|---|---|
| H3 zone online driver count | real-time | ≤ 500ms | Driver GPS → Kafka → Flink → Redis | Approx count via HyperLogLog per zone |
| H3 zone trip request rate (last 5 min) | real-time | ≤ 30s | Trip request stream → Kafka → Flink → Redis | Rolling window count |
| Driver ETA to zone (supply accessibility) | real-time | ≤ 60s | ETA system output → Redis | Drivers within 5 min counted as "accessible supply" |
| Historical demand by zone × hour-of-week | batch | daily | Hive → feature store | Baseline demand; used to detrend imbalance ratio |
| Event calendar (concerts, sports, airports) | batch | hourly | Events API → feature store | Demand multiplier for known future spikes |
| Weather (precipitation, temperature) | near-real-time | 15 min | Weather API → Kafka | Rain ↑ demand ~20%; ↓ driver supply |
| Current surge multiplier (feedback) | real-time | ≤ 30s | Previous zone state | EMA dampening input |
| Driver response elasticity (per zone) | batch | daily | Offline RDD estimation | Δdrivers per Δ1× surge; used in endogeneity correction |

### Label Definition

**Optimization target:** `optimal_multiplier(zone, t)` = multiplier that minimizes:
```
E[unfulfilled_trips] + α × E[rider_wait_time]
subject to: E[driver_earnings_per_hour] ≥ baseline_earnings
```

**Label derivation:** counterfactual — cannot be directly observed. Estimated via switchback experiments (zone alternates surge levels on a fixed schedule) or structural demand/supply models.

**Proxy labels used in practice:**

| Proxy | What It Measures | Why It's Imperfect |
|---|---|---|
| `trip_completion_rate(zone, t)` | Did supply meet demand? | Doesn't distinguish "no demand" from "demand met" |
| `driver_acceptance_rate(zone, t)` | Was supply willing? | Acceptance can be high even if wait times are high |
| `rider_wait_time_p90(zone, t)` | Was wait acceptable? | Lags the actual supply/demand event by minutes |

### Storage Engine

| Layer | Engine | Justification |
|---|---|---|
| Zone surge multiplier (read path) | Redis Cluster (`h3_zone_id → multiplier`, TTL 90s) | Sub-ms reads; 1M+ req/sec = O(1) cache hit; TTL ensures eventual consistency |
| Driver supply state (per zone) | Redis (HyperLogLog + sorted set for ETA-ranked drivers) | Approximate zone driver count; sorted set enables "N closest drivers" lookup |
| Trip request stream | Kafka (partitioned by H3 zone ID) | High-throughput; Flink consumer aggregates to zone-level demand counts |
| Driver GPS stream | Kafka (partitioned by driver_id) | ~5K events/sec; Flink joins to H3 zone and updates zone supply counts |
| Historical demand/supply patterns | Cassandra (`(h3_id, hour_bucket, day_of_week)`) | Wide rows for time-partitioned reads; city-level sharding |
| Offline feature store | Hive on S3 (partitioned by city, date) | Spark-accessible; elasticity estimation runs nightly |
| Pricing decision log | Kafka → S3 (Parquet) | Full audit trail for counterfactual analysis and A/B evaluation |

---

## 3. Algorithm Design

### 3.1 Economic Foundation

**Objective (formal):**
```
min_{p(zone,t)}  E[unfulfilled_trips(p)] + α × E[rider_wait_time(p)]
subject to:      E[driver_earnings_per_hour(p)] ≥ w_baseline
                 p(zone, t) ∈ {1.0, 1.25, 1.5, 2.0, 2.5, 3.0}
```

**Why discrete tiers?** Continuous pricing creates micro-incentives for drivers to hover at zone boundaries to capture marginally higher rates. Discrete tiers reduce boundary gaming, make pricing legible to drivers and riders, and reduce oscillation amplitude.

**Multiplicative vs. Additive Surge** *(Garg & Nazerzadeh, Management Science 2021)*

Multiplicative surge — the historical standard — is **not incentive-compatible** in a dynamic setting.

| Mechanism | Driver Payout | Incentive Compatibility Problem |
|---|---|---|
| Multiplicative | `base_fare × surge_multiplier` | A driver on a long trip earns `long_fare × 2×`; a driver who took a short trip earns `short_fare × 2×`. The long-trip driver is penalized for being unavailable during peak. Drivers strategically decline long trips to stay available for the next surge event. |
| Additive | `base_fare + flat_bonus(zone, t)` | Bonus is independent of trip length. A driver earns the same zone-time bonus regardless of which trip they accept. Dominant strategy is to provide supply in the high-bonus zone — the intended behavior. |

**Uber's production system now uses additive driver surge.** Rider-facing pricing may still use a multiplicative display for rider comprehension, but driver payment uses the additive mechanism. In an interview, framing this distinction signals Staff-level awareness of mechanism design vs. UX pricing.

---

### 3.2 Algorithm Evolution

The field has progressed through four generations. Production at Uber is at Gen 2–3; Gen 4 is research-frontier.

| Generation | Core Approach | Key Insight | Production Readiness |
|---|---|---|---|
| **Gen 1: Rule-based** | `surge = 1.5× if drivers < threshold` | Simple, interpretable, safe fallback | Yes — used as circuit-breaker fallback today |
| **Gen 2: Supervised ML** | LightGBM trained on (zone, hour) → optimal_multiplier from switchback labels | Captures nonlinear demand patterns; incorporates weather, events, adjacency | Yes — core production model |
| **Gen 3: Single-Zone RL** | MDP per zone; DDPG with prioritized experience replay (DDPG-PER) | Continuous action space; explicitly models feedback loop via delayed reward | Emerging in production (2023–2024) |
| **Gen 4: MARL** | Each zone is an agent; centralized critic + decentralized actors (MADDPG, R2Pricing) | Spatial-temporal coordination; captures driver repositioning across zones | Research frontier; not yet at scale |

---

### 3.3 MDP Formulation

Formulating surge as a Markov Decision Process unlocks RL-based optimization and provides a principled framework for credit assignment across the demand-supply feedback loop.

**State space** (per zone, per 30s tick):
```
s_t = (
    online_drivers(zone),         # current supply
    trip_request_rate_5min(zone), # current demand
    adj_zone_surge[k=1..6],       # neighboring zone surges (spatial context)
    time_of_week,                 # encoded as (hour, day_of_week) cyclic features
    weather_code,                 # precipitation, temperature bucket
    event_flag,                   # 0/1 for known upcoming event
    current_surge(zone),          # previous action (for dampening)
    driver_elasticity(zone)       # offline-estimated supply response coefficient
)
```

**Action space:** `a_t ∈ {1.0, 1.25, 1.5, 2.0, 2.5, 3.0}` — discrete surge tiers

**Reward:**
```
r_t = −unfulfilled_trips(t)
      − α × rider_wait_time_p90(t)
      + β × driver_earnings_above_baseline(t)
      − γ × |a_t − a_{t-1}|          # oscillation penalty
```

**Transition dynamics:** Stochastic. Driver supply responds with lag `τ ~ Gamma(mean=5min, shape=2)` — actions at time `t` affect state at `t+τ`. This delayed-reward credit assignment is the core difficulty. RL with n-step returns or TD(λ) is required to properly propagate the signal back through the delay.

**Why not standard Q-learning?** Discrete action space with 6 tiers is tractable, but the feedback loop creates a non-stationary MDP (driver populations adapt over time). DDPG-PER (continuous action, discretized at output) with prioritized experience replay handles rare high-surge events that a uniform replay buffer under-represents.

---

### 3.4 Queueing Theory Baseline

Formal grounding for when and why surge pricing achieves equilibrium. *(Banerjee, Johari, Riquelme — Columbia; queueing theory for ridesharing platforms)*

**Model:** Each zone is an M/M/N queue:
- Arrivals: trip requests at rate `λ(p)` — price-elastic demand
- Servers: N = online drivers, service rate `μ` (trips completed per driver per hour)
- Queue discipline: first-come, first-served with abandonment after wait threshold `W_max`

**At steady state**, the optimal price satisfies:
```
surge*(zone) = argmin_p  E[queue_length | λ(p), N(p)]
```
where `N(p) = N_0 × (1 + ε_s × (p − 1))` is the supply response curve.

**Stability condition** (fluid model approximation, large-market limit):

A unique stable equilibrium exists if and only if:
```
|ε_d(zone)| > ε_s(zone)
```
where `ε_d` = demand price elasticity (negative), `ε_s` = supply price elasticity (positive).

**Practical implications:**

| Zone Type | Typical ε_d | Typical ε_s | Implication |
|---|---|---|---|
| Airport pickups | Low (−0.2) | High (+0.8) | Small surge clears market; high surge inefficient |
| Late-night downtown | Low (−0.1) | Low (+0.1) | Large surge required; equilibrium fragile |
| Suburban rush hour | High (−0.7) | Low (+0.2) | Surge destroys demand before attracting supply; use predictive re-positioning instead |
| Event venue exit | Very low (−0.05) | High (+1.2) | High surge is efficient; drivers respond strongly to bonus |

**Why this matters in the interview:** When asked "how do you know your surge model will converge?", cite the fluid model stability condition. It's the theoretical guarantee that the EMA-dampened feedback loop reaches steady state rather than oscillating indefinitely.

---

### 3.5 Contextual Bandits for Online Surge Learning

Alternative to full RL when the feedback loop is short (surge → driver response within minutes) and zone-level independence can be approximately satisfied.

**Framing:**
- Context `x` = zone feature vector (demand, supply, time, weather)
- Arm `a` = surge tier ∈ {1.0, 1.25, 1.5, 2.0, 2.5, 3.0}
- Reward `r` = trip completion rate in zone over next 5-min window

**Algorithm:** Thompson Sampling per zone cluster
- Maintain Beta posterior `Beta(α_a, β_a)` on `P(trip_complete | arm=a, cluster)` per (surge tier, zone cluster)
- Each update tick: sample from posterior → choose surge tier → observe outcome → update posterior
- Zone clusters: group by (city_tier, zone_type, hour_bucket) to share statistical strength across similar zones

**SUTVA violation:** Zone outcomes are not independent — drivers move between zones. Fix:
1. Use zone clusters where geographically separated clusters are treated as independent
2. Restrict exploration to non-adjacent zones simultaneously
3. Use doubly robust reward estimator to correct for spillover (see OPE section in `dynamic-pricing-and-mechanism-design.md`)

**When to use bandits vs. full RL:**

| Criterion | Contextual Bandits | RL (DDPG/MARL) |
|---|---|---|
| Feedback latency | Short (< 10 min) | Long (30+ min, with lag) |
| Spatial coupling | Weak (zones mostly independent) | Strong (driver repositioning across zones) |
| Interpretability required | High (regulator audit) | Lower acceptable |
| Sample efficiency | Lower (needs more exploration) | Higher (model-based planning) |

---

### 3.6 Driver Response Elasticity Estimation

**Problem:** To apply the endogeneity correction in the surge pipeline, we need to know `ε_s(zone)` = how many incremental drivers come online per unit increase in surge. This cannot be estimated naively (drivers who show up at high surge were planning to anyway).

**Method: Regression Discontinuity Design (RDD)**

Surge multipliers are assigned by threshold rules (e.g., the multiplier jumps from 1.5× to 2.0× when imbalance crosses 1.8). Drivers just above vs. just below the threshold are near-identical in unobservables — the discontinuous jump in driver arrivals at the threshold identifies the causal supply response.

```
Estimation:
  - For each tier boundary b ∈ {1.25, 1.5, 2.0, 2.5}:
    - Select observations with surge ∈ [b − bandwidth, b + bandwidth]
    - Fit local linear regression on each side of b
    - ε_s at boundary = (fitted_above − fitted_below) / (upper_tier − lower_tier)
  - Zone-level elasticity = mean over tier boundaries with sufficient data
```

**Production details:**
- Runs nightly as a Spark job over 90 days of historical data
- Output: `elasticity_coefficient` per `(h3_zone_id, day_of_week_bucket)` in Cassandra
- Consumed by Surge Computation Engine at Step 4 (endogeneity correction)
- Elasticity changes slowly (driver demographic shifts over weeks); daily refresh is sufficient

---

### 3.7 15-Minute Predictive Surge

**Purpose:** Notify drivers "Surge expected in Downtown in 15 min — head there now." Proactive repositioning prevents the surge from materializing at all.

**Model:** Per-zone time-series binary classifier
- Features: demand rate last 1h in 5-min buckets (lag features), event flags, historical demand pattern at this zone × hour-of-week
- Output: `P(surge > 1.5× in next 15 min | zone, features)`
- Model: LightGBM with rolling-window lag features; retrained daily

**Hawthorne effect (the key design challenge):**
Notifications bring drivers → reduces surge probability → the notification was "correct but self-defeating." Fix: cap notification volume per zone to the expected supply gap. If you need 10 more drivers, notify at most 15 (accounting for conversion rate). Notifying 200 floods the zone, crashes surge, and erodes driver trust in notifications.

**Threshold for notification:** P > 0.7 to avoid alert fatigue. Monitor notification precision (fraction of notified zones that did surge) and recall (fraction of surge events that had a notification) separately; they trade off via this threshold.

---

## 4. System Architecture

### Data Flow

```
[Driver GPS stream]   ─→ Kafka (partitioned by driver_id)
                          ─→ Flink (H3 zone join, supply aggregation)
                             ─→ Zone Supply Store (Redis HLL + sorted set)

[Trip request stream] ─→ Kafka (partitioned by H3 zone)
                          ─→ Flink (5-min rolling demand count)
                             ─→ Zone Demand Store (Redis)

[Event calendar]      ─→ Feature Store (hourly refresh)
[Weather API]         ─→ Kafka ─→ Feature Store (15-min refresh)
[Historical patterns] ─→ Hive/S3 ─→ Feature Store (daily batch)
```

### Surge Computation Engine (30s cadence)

```
Every 30s, for each active zone:
                                         ┌──────────────────────────────────┐
Zone Supply + Demand (Redis)  ──────────→│ 1. Fetch zone state              │
Feature Store                ──────────→│ 2. Compute raw imbalance ratio    │
                                         │ 3. Apply LightGBM pricing model   │
Elasticity Coefficients      ──────────→│ 4. Endogeneity correction         │
Previous Surge (Redis)       ──────────→│ 5. EMA dampening (α = 0.3)       │
                                         │ 6. Snap to discrete tier          │
                                         │ 7. Write to Redis surge cache     │
                                         └──────────────────────────────────┘
                                                        │
                                         Redis Surge Cache (h3_id → multiplier)
                                                        │  < 1ms read
                                         [Rider App] ← API Gateway ← Surge Lookup
```

### Surge Computation Steps (with rationale)

**Step 1 — Zone state assembly**
Read from Redis: `online_drivers(zone)`, `trip_requests_last_5min(zone)`, `current_multiplier(zone)`, weather code, event flag.

**Step 2 — Supply/demand imbalance**
```
imbalance = trip_request_rate(zone) / max(online_drivers(zone), 1)
# imbalance > 1.5 → demand-heavy; < 0.5 → supply-heavy
```
*Why normalize by baseline:* raw imbalance varies by zone size. Detrend against `historical_baseline_imbalance(zone, hour_of_week)` so the model sees excess demand relative to normal, not absolute counts.

**Step 3 — LightGBM pricing model**
Input features: `[imbalance_vs_baseline, time_of_day, day_of_week, weather, event_flag, adj_zone_surges, historical_demand_deviation]`
Output: continuous multiplier (e.g., 1.73×). Trained on historical (zone, hour) → optimal multiplier labels derived from switchback experiments.

**Step 4 — Endogeneity correction** *(the key Staff-level step)*
```
supply_response = driver_elasticity(zone) × (raw_multiplier − 1.0)
adjusted_supply = online_drivers(zone) + supply_response
# Re-compute imbalance with adjusted_supply → corrected_multiplier
```
*Why this matters:* Publishing surge=2.0× will cause drivers to reposition toward the zone. The current supply count doesn't yet reflect those incoming drivers. Without this correction, the system systematically oversurges. This is the endogeneity problem: the price you're about to set will change the supply you're using to justify the price.

**Step 5 — EMA dampening**
```
dampened = α × corrected_multiplier + (1 − α) × previous_multiplier
# α = 0.3: zone needs 3–4 consecutive high-demand readings to fully update
```
*Why 0.3:* with α = 0.3, the time constant (1/α) ≈ 3 ticks × 30s = 90s. One transient spike (e.g., a bus arriving) dissipates without surging. Three consecutive readings of high demand do trigger the surge. This is a control-theoretic choice, not a hyperparameter to grid-search.

**Step 6 — Discretization**
Snap to `{1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0}`. Prevents micro-fluctuations that trigger driver repositioning across zone boundaries.

**Step 7 — Write to Redis**
`SET h3:{zone_id}:surge {multiplier} EX 90` — TTL of 90s ensures stale surge expires gracefully even if the computation engine fails.

---

## 5. Failure Modes

| Failure | Detection | Mitigation |
|---|---|---|
| ML model unavailable | Model health check; timeout on inference | Fallback to threshold rule: `surge = 1.5× if imbalance > 1.5, else 1.0×` |
| Redis surge cache miss | Cache miss rate alert | Recompute on demand (slower path; still < 50ms for single zone) |
| Kafka consumer lag (stale supply/demand) | Consumer lag metric > 60s | Use last known zone state + exponential decay (assume supply drifts offline over time) |
| Supply signal completely lost | Driver count = 0 for zone > 5 min | Use adjacent zone supply as proxy; alert on-call |
| Oscillation (surge overshoots repeatedly) | Zone multiplier std dev > 0.5 over 10 min | Increase EMA dampening (α → 0.1); root cause usually missing endogeneity correction |
| Surge overestimation (too many drivers flood zone) | Driver acceptance rate < 30% | Cap maximum surge increase per interval (max +0.25× per 30s step) |
| Surge underestimation (unmet demand) | Rider wait time p90 > 8 min | Shorten demand window from 5 min to 2 min; may indicate demand spike not yet captured |
| City-level driver data outage | Supply count = 0 for all zones in city | Fall back to city-level historical surge pattern (day-of-week × hour baseline) |

---

## 6. Capacity Estimates

| Component | Estimate | Assumptions |
|---|---|---|
| Rider surge lookups | 1M req/sec | All served from Redis cache; each lookup = 1 Redis GET |
| Zone update rate | ~170K updates/sec | 500K active zones × 1 update/30s |
| Driver GPS ingestion | 5K events/sec | 5M active drivers × 1 GPS event/1000s average |
| Trip request ingestion | 350–1,500 req/sec | Average–peak globally |
| Redis memory (surge cache) | ~10 GB | 500K zones × 20 bytes/entry |
| Redis memory (driver supply) | ~50 GB | 5M drivers × 10 bytes GPS state |
| Surge computation CPU | ~50 cores | 170K zone updates/sec × 0.3ms/update |
| Pricing log storage | ~10 TB/month | 500K zones × 2 updates/min × 1 KB/record × 60 × 24 × 30 |

---

## 7. Monitoring & Observability

### Real-Time Dashboards (5-min resolution)

| Metric | What It Catches |
|---|---|
| Zone surge distribution (histogram) | Systematic over/under-surging across the fleet |
| Trip completion rate by surge tier | Whether high surge actually improves completion or just raises price |
| Driver acceptance rate by zone | Proxy for "was supply actually insufficient?" |
| Rider wait time p50/p90 by zone | The metric surge is supposed to improve |
| Supply fishing rate | % drivers online within 2 min of surge increase (leading indicator of gaming) |
| Adjacent zone surge correlation | Detecting artificial boundary effects from gaming |

### Offline Model Monitoring (daily)

- **Driver elasticity coefficient drift:** stale elasticity causes Step 4 to over- or under-correct
- **Surge prediction calibration:** does "predicted surge = 1.5×" materialize at 1.5× ± 0.2?
- **A/B experiment analysis:** switchback results vs. offline model predictions; if they diverge > 15%, retrain

### Alerting Thresholds

| Alert | Threshold | Severity |
|---|---|---|
| Zone-level wait time p90 > 10 min for 5+ min sustained | Demand/supply severe imbalance | P1 |
| System-wide surge cache hit rate < 99.9% | Fallback path activated | P1 |
| Kafka consumer lag > 2 min | Supply/demand signals stale | P2 |
| Oscillating zones (std dev > 0.5) > 1% of active zones | EMA tuning issue | P2 |

---

## 8. Key Design Decisions

**Why precomputed zone-level surge vs. per-request inference?**
At 1M req/sec, per-request inference needs 1M inferences/sec. At 10ms/inference that's 10,000 CPU cores. Precomputing at zone level (30s cadence) reduces model calls to ~170K/sec, and serving is a Redis cache lookup (< 1ms). This is the primary architectural choice that makes the scale feasible.

**Why additive driver surge vs. multiplicative?**
Multiplicative surge (`fare × multiplier`) is not incentive-compatible: drivers on long trips miss short high-surge trips. The mechanism selects against the drivers who are already engaged. Additive surge (`fare + flat_bonus`) is incentive-compatible: the bonus is time-independent, so the dominant strategy is to be available in the high-bonus zone regardless of trip length. *(Garg & Nazerzadeh, Management Science 2021)*

**Why EMA dampening instead of setting model output directly?**
Surge has a feedback loop: high surge → drivers come online → supply increases → surge falls. Without dampening, the system oscillates with period ~5–15 min. EMA with α = 0.3 is derived from the control-theoretic stability condition for the feedback loop given typical supply response lag τ ~ Gamma(mean=5min). Increasing α makes the system more responsive but unstable in elastic zones; decreasing α makes it stable but slow to clear genuine demand spikes.

**Why estimate driver elasticity offline (nightly) vs. online?**
Online elasticity estimation requires live A/B testing across surge levels — ethically and commercially problematic (different prices for similar riders simultaneously). Offline regression discontinuity on historical threshold-triggered surge changes provides a causal estimate without real-time experimentation. The cost: elasticity is 24h stale, acceptable because driver demographics shift over weeks, not hours.

**Why discrete surge tiers vs. continuous pricing?**
Continuous pricing (1.73×, 1.82×) creates micro-incentives for drivers to position at zone boundaries to capture marginally higher rates. Discrete tiers reduce boundary gaming, make pricing legible to drivers and riders, and reduce oscillation amplitude (steps are larger, so dampening catches them before the next tier).

**Why LightGBM instead of a deep model for the pricing model?**
LightGBM is interpretable (SHAP attributions), fast at inference (~1ms), robust to missing features, and does not require GPU serving infrastructure. The pricing model is a moderate-complexity tabular problem (zone + time features) — not image/text. Deep models would add serving complexity without meaningful accuracy gain. The harder ML problems (elasticity estimation, OPE) are done offline where training time is not a bottleneck.

---

## 9. Scorecard

| Axis | Score | Weakest Point |
|---|---|---|
| Requirements | ★★★★★ | — |
| Data Modeling | ★★★★★ | — |
| ML Pipeline | ★★★★★ | — |
| Failure Modes | ★★★★★ | — |
| Capacity | ★★★★★ | — |

**Overall:** `STRONG HIRE`

**Top Gap to watch for in candidates:** Strategic driver response creates endogeneity — naive supply estimation ignores that supply responds to the price you're trying to set. Candidates who skip Step 4 (endogeneity correction) are solving a simpler, incorrect version of the problem. The follow-up question: "How do you estimate the elasticity coefficient without running a live experiment on prices?" is the RDD question.

---

## References

- Garg, N. & Nazerzadeh, H. — [Driver Surge Pricing](https://arxiv.org/abs/1905.07544), *Management Science* 2021
- Banerjee, S., Johari, R., Riquelme, C. — [Pricing in Ride-Share Platforms: A Queueing-Theoretic Approach](https://www.columbia.edu/~ww2040/8100F16/Riquelme-Johari-Banerjee.pdf), *EC* 2015
- Scalable RL for dynamic pricing in ride-hailing — [ScienceDirect 2023](https://www.sciencedirect.com/science/article/abs/pii/S019126152300173X)
- MARL for competitive ride-sourcing pricing — [R2Pricing, Transportation Research Part C 2024](https://www.sciencedirect.com/science/article/abs/pii/S0968090X24002183)
