# Driver Response Modeling & Anti-Gaming — ML Knowledge Q&A

P0: Uber Supply Pricing team. Complements `dynamic-pricing-and-mechanism-design.md`
(which covers strategic behavior at a conceptual level). This file covers:
- Implementation-level supply fishing detection
- Oscillation dynamics and damping
- Multi-agent simulation for policy validation
- Fairness metrics for pricing systems

---

## Supply Fishing Detection

### Q: What is supply fishing, and how do you detect it at scale?

**Answer (Staff level):**

**What it is:** Coordinated or individual driver behavior where drivers repeatedly
go offline → trigger zone supply shortage → surge activates → come back online
at the higher fare. At scale, even a small group (10–15 drivers in a dense zone)
can reliably sustain artificial surge.

**Behavioral signatures:**

| Signal | Legitimate Behavior | Fishing Pattern |
|---|---|---|
| `offline_online_cycles / hour` | 1–3 (meal breaks, trip completion) | 5–15+ (deliberate toggling) |
| `cross_boundary_moves / shift` | 2–4 (organic repositioning) | 8–15+ (hop just outside zone and back) |
| `surge_chase_lag_seconds` | 60–300s (organic decision time) | 5–45s (automated or trained response) |
| `zone_entry_timing` | Random within hour | Clustered within 30s of surge event |

**Detection architecture:**
```
Driver GPS stream → Kafka
      ↓
Flink: compute shift-level aggregates per driver per 8-hr window
      ↓
Feature store: cycles_per_hour, boundary_moves, surge_chase_lags
      ↓
Anomaly scorer (LightGBM or rule-based) → flagged_driver_ids
      ↓
Enforcement action: earnings smoothing or deactivation
```

**Why rule-based first, ML second:**
- Rules are interpretable and auditable — required for enforcement (drivers can dispute).
- ML layer catches emerging patterns rules miss; but rules set the floor.
- Always prefer a high-precision, lower-recall rule over a high-recall classifier
  when false positives = wrongly penalizing legitimate drivers.

**Mitigation options:**

| Option | Mechanism | Trade-off |
|---|---|---|
| Earnings smoothing | Replace per-trip surge with shift-level average earnings | Removes incentive to time surge; reduces driver hourly variance | Less reward for genuine availability during high demand |
| Surge delay notification | Don't notify drivers of surge level until they are online | Breaks the information advantage that enables fishing | May reduce overall driver supply (less certainty of earnings) |
| Zone boundary noise | Fuzzy H3 zone assignment (probabilistic 5% boundary overlap) | Raises cost of cross-boundary gaming | Adds slight matching complexity |
| Hard deactivation | Temporary suspension on repeated flagging | Strong deterrent | Requires high-confidence detection; appeals process needed |

**Staff-level framing:** Lead with "why fishing exists" (mechanism design failure — the current surge mechanism has an information advantage that enables gaming) before discussing detection. Fix the mechanism design first (earnings smoothing, delay notifications); use detection as the second line of defense.

---

## Surge Oscillation Dynamics

### Q: Why does surge oscillate, and how do you tune the damping coefficient?

**Answer (Staff level):**

**Why oscillation happens:**
1. Surge spikes → drivers rush to zone (supply response lag ~3–10 min).
2. Over-supply arrives → surge drops to 1×.
3. Drivers leave (lack of trips) → supply shrinks again → surge spikes again.
4. Period of oscillation: typically 5–15 min in dense urban zones.

This is a classic **delayed feedback control system**. The surge multiplier is the
control signal; the objective is market clearing (demand ≈ supply); the plant is the
driver supply response (with a transportation lag).

**Stability analysis:**
Let `s_t` = surge at time `t`, `r_t` = true demand/supply ratio.
Without dampening:
```
s_t = f(r_t)       # naive: map ratio directly to surge tier
r_t = g(s_{t-1})  # supply responds to prior surge with lag τ
```
This system oscillates when the gain `|∂f/∂r × ∂g/∂s| > 1`.

With EMA dampening:
```
s_t = α × f(r_t) + (1 - α) × s_{t-1}
```
The effective gain is multiplied by `α`, so choosing `α < 1/|∂g/∂s|` guarantees
stability at the cost of slower response to genuine demand shocks.

**Tuning α:**

| α Value | Behavior | When to Use |
|---|---|---|
| 0.1 | Very smooth; slow response | Event venues, airports (predictable demand) |
| 0.3 | Balanced (default) | General urban zones |
| 0.5–0.7 | Reactive; may oscillate in elastic zones | Zones with very low driver elasticity |
| 1.0 | No dampening (raw surge) | Never use in production |

**Empirical approach:** Run switchback experiments where `α` is varied by zone.
Monitor `std(surge_over_30min)` (oscillation) vs. `E[unfulfilled_trips]` (clearing
efficiency). Pareto-optimal α is zone-specific; use online Bayesian optimization
to learn per-zone α over time.

**Predictive surge as an alternative to dampening:**
Instead of dampening the *current* surge signal, predict the surge 15 minutes ahead
and use the prediction to pre-position drivers before the demand spike. This avoids
the fundamental oscillation problem by acting before scarcity occurs.

---

## Multi-Agent Simulation for Policy Validation

### Q: How do you validate a new surge policy before deploying to production?

**Answer (Staff level):**

**Why simulation before A/B test:**
- Some surge policies are too risky to test live (extreme multipliers, new zone configurations).
- Simulation lets you run thousands of policy variations in hours, not weeks of A/B traffic.
- Simulation is especially valuable for rare events (concerts, weather) that an A/B test
  would take months to accumulate naturally.

**Simulation architecture:**

```
Historical data (GPS, trip requests, surge logs)
      ↓
Driver behavioral model: P(driver_comes_online | surge, hour_of_week, weather, city)
      ↓                     P(driver_repositions | current_zone_surge, neighboring_surge)
Rider demand model:      P(trip_request | surge, zone, time_of_day)
      ↓                     P(rider_cancels | wait_time_estimate)
Matching layer:          Greedy ETA-weighted matching (no ML needed; just simulate dispatch)
      ↓
Surge policy under test  → surge_multiplier(zone_state) [varies by policy]
      ↓
Outcome metrics: unfulfilled_trips, driver_earnings_gini, rider_wait_time_p90
```

**Driver behavioral model (key component):**
- Fit a logistic regression or LightGBM per driver cohort (city × hour × day) on
  historical online/offline transitions.
- Include: `P(come_online | surge)`, `P(move_to_zone | zone_surge_delta)`.
- Validate: replay holdout days and check that simulated supply distribution matches
  actual. Target <10% RMSE on zone-level supply counts.

**Common failure modes of simulation:**
1. **Overclaims equilibrium** — simulation assumes drivers reach steady state;
   real drivers have heterogeneous response times.
2. **SUTVA violation** — drivers moving between zones break zone-independence
   assumption. Fix: include a zone-adjacency graph in the supply model.
3. **Hawthorne effect** — new policy changes driver behavior in ways historical data
   can't capture. Simulation underestimates adaptation effects; always follow with
   a small live holdout experiment.

---

## Fairness in Surge Pricing

### Q: How do you measure and address fairness issues in dynamic pricing?

**Answer (Staff level):**

**Why fairness matters at Uber Scale:**
- Surge pricing can create systematic earnings disparities between driver cohorts
  (city center vs. suburban, experienced vs. new drivers, full-time vs. part-time).
- Rider price disparities can harm low-income areas if they have lower driver density
  and therefore persistent high surge.
- At Uber's scale, even small bias in surge design affects millions of driver livelihoods.

**Earnings fairness metrics:**

| Metric | Formula | Threshold (indicative) |
|---|---|---|
| Earnings Gini coefficient | Standard Gini on hourly earnings across all active drivers | < 0.35 per city |
| P10/P90 earnings ratio | `p10_hourly_earnings / p90_hourly_earnings` | > 0.4 (top earners shouldn't be 2.5× bottom) |
| City-level earnings CV | `std(city_avg_earnings) / mean(city_avg_earnings)` | < 0.2 across comparable cities |
| Surge exposure by zone type | `mean_surge(low_income_zone) / mean_surge(high_income_zone)` | < 1.3 (should not consistently double-penalize low-income areas) |

**Driver earnings fairness interventions:**

1. **Guaranteed minimum hourly earnings** during surge events (driver gets max of
   surge earnings and guaranteed floor). Prevents new or unlucky drivers from
   missing surge events through no fault of their own.

2. **Surge notification fairness** — if surge notifications are sent, ensure
   notification latency is uniform across driver cohorts (don't accidentally
   advantage drivers with newer phones or better connectivity).

3. **Quest threshold personalization** (see mechanism design doc) — ensures
   incremental effort requirements scale proportionally to each driver's baseline,
   not at a fixed absolute count that advantages high-frequency drivers.

**Rider fairness interventions:**

1. **Surge cap by zone income level** — set lower surge caps in zones with lower
   median income (measured from census data). Riders in those zones are less able
   to absorb demand pricing.

2. **Price lock for vulnerable trips** — airport rides, hospital zones, and late-night
   trips from areas with no transit alternatives should have surge smoothed or capped.

**Staff-level framing:** Fairness is not an add-on — it's a constraint in the surge
optimization objective. Reframe the objective:

```
minimize  E[unfulfilled_trips] + α × E[rider_wait_time]
subject to:
  E[driver_earnings_per_hour] ≥ baseline_earnings        # driver floor
  E[surge_multiplier | low_income_zone] ≤ surge_cap_low  # rider equity
  Gini(hourly_earnings) ≤ 0.35                           # driver equity
```

Adding these constraints doesn't require separate fairness infrastructure —
they enter the optimization as Lagrange multipliers (penalized objectives in
the RL/bandit formulation of surge optimization).

---

## Interview Quick Reference

**3 things to say in the first 2 minutes of a supply pricing ML design:**
1. "Surge is a market-clearing mechanism, not revenue maximization."
2. "Drivers respond strategically — observed supply is endogenous to the surge you're
   trying to set. Must correct for this."
3. "Oscillation is the primary operational risk — design for stability first,
   then optimize for efficiency."

**Questions Chris Mosch's team will ask:**
- "How do you know your surge model is causal?" → IV/RDD/switchback experiment design
- "How do you validate a new surge policy?" → multi-agent simulation + staged rollout
- "What happens when Kafka lag spikes?" → stale supply signal → fallback to historical
  baseline surge + circuit breaker to prevent catastrophic clearing failure
- "How do you handle the Hawthorne effect?" → drivers learn and adapt; need continual
  offline re-estimation of elasticity (daily batch job); simulation is optimistic baseline
- "How do you think about driver fairness in surge?" → earnings constraints in objective;
  Gini metric; quest personalization

**Related prep files:**
- `dynamic-pricing-and-mechanism-design.md` — foundational Q&A (mechanism design, RDD, OPE)
- `docs/system-design/real-time-ml/uber-supply-pricing.md` — full system design with capacity
- `coding/ml-coding/applied/supply_pricing_ml.py` — implementations of all 4 core algorithms
