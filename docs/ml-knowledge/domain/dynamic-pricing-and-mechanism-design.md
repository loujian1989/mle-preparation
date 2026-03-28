# Dynamic Pricing & Mechanism Design — ML Knowledge Q&A

P0: Uber (Supply Pricing team). Directly relevant to Chris Mosch's team description:
"real-time systems that optimize pricing across 1M+ decisions per second while modeling
strategic driver responses."

---

## Surge Pricing Fundamentals

### Q: How does surge pricing work, and what is it actually optimizing for?

**Answer (Staff level):**
- **Common misconception**: surge pricing maximizes revenue. Wrong framing — it's a market-clearing mechanism.
- **What it's actually doing**: when demand > supply in a zone, riders would queue indefinitely at a fixed price. Surge raises price until `demand(price) ≈ supply(price)` — clearing the market so willing riders get matched and drivers earn more for showing up.
- **Why ML over threshold rules**:
  - Threshold rule: `surge = 1.5× if available_drivers < 10 in zone`. Simple but: ignores cancellation rates, ignores predicted supply arriving in 5 min, ignores demand elasticity (some zones have inelastic demand that should get higher surge).
  - ML approach: predict `expected_wait_time(surge_multiplier)` or directly predict `optimal_multiplier` that minimizes a joint objective (wait time + driver earnings + rider completion rate).
- **The real objective**: not max revenue, but min unfulfilled demand subject to driver earnings constraints. Formally:
  ```
  minimize  E[unfulfilled_trips] + α × E[rider_wait_time]
  subject to E[driver_earnings_per_hour] ≥ baseline_earnings
  ```
- **Feedback loop**: surge → more drivers come online → supply increases → surge falls. The system is dynamic, not static. Naive surge overestimation drives too many drivers to the zone, all earn less, and some stop coming. Calibrating the feedback loop is the hard problem.

**Company context:** Uber Supply Pricing. This is the team's core product — understand the economic purpose before the ML.

**Common wrong answer:** "Surge pricing maximizes revenue." — It's a market-clearing tool. Revenue is a side effect. The primary goal is matching supply to demand efficiently. Getting this wrong in the interview signals you haven't thought about the economics.

---

## Mechanism Design for Driver Incentives

### Q: What is mechanism design and how does it apply to driver incentive programs at Uber?

**Answer (Staff level):**
- **Mechanism design**: the reverse of game theory. In game theory, you're given the rules and predict what agents do. In mechanism design, you choose the rules to get the outcome you want — even when agents act in their own self-interest.
- **Incentive compatibility**: a mechanism is incentive-compatible (IC) if agents' dominant strategy (the strategy that maximizes their payoff regardless of what others do) is to behave in the way the mechanism designer wants. You don't need to rely on altruism or compliance.
- **Applied to driver quests**:
  - Quest: "Complete 10 trips by Sunday and earn a $30 bonus."
  - **Design question**: what trip count triggers the quest? Too low → drivers get the bonus anyway (no incremental behavior change). Too high → drivers don't bother trying (effort too large for $30). The IC-optimal threshold is the count just above what the driver would do naturally.
  - **Personalization**: Uber personalizes quest thresholds per driver based on historical trip frequency. A driver who normally does 8 trips/week gets a 10-trip quest; one who normally does 5 gets a 7-trip quest. Both face incremental effort requirements proportional to their baseline.
- **Adverse selection in incentives**:
  - Boost pricing: "Earn 2× fare in this zone from 5pm–7pm." If drivers know this in advance, they all rush to the boost zone before 5pm → supply shortage outside the zone → surge cascades elsewhere.
  - Fix: announce boost zones with shorter lead time or use smooth boost gradients instead of binary on/off.
- **VCG analogy**: in ads, VCG makes truthful bidding the dominant strategy. In driver pricing, the analogous design is: make "providing supply where and when it's needed" the dominant strategy by making earnings exactly proportional to the platform's marginal need for supply. This is the economic ideal; practical approximations (surge multipliers) are step functions toward this ideal.

**Company context:** Uber Supply Pricing (this is mechanism design applied directly). Reddit Staff (auction theory adjacent). The question "how did you design an incentive program?" is explicitly a mechanism design question.

**Common wrong answer:** "We set bonuses empirically based on what worked in A/B tests." — Empirical testing is necessary but insufficient at Staff bar. The mechanism design answer asks: why did that threshold work? What's the theoretical property being satisfied? If you can't answer that, you'll set the wrong threshold in a new city with different driver behavior.

---

## Strategic Driver Response Modeling

### Q: Drivers respond strategically to surge pricing. How do you model strategic behavior instead of treating drivers as passive supply?

**Answer (Staff level):**
- **The naive model**: `P(driver comes online | surge=2.0×) = sigmoid(β₀ + β₁ × surge)`. Treats surge as a static input. Problem: drivers learn the system and respond strategically.
- **Observed strategic behaviors**:
  1. **Supply fishing**: drivers stay offline until surge starts (they know demand will spike at a predictable time, e.g., concert end). Going offline artificially reduces supply → triggers surge → then they come online at the higher rate.
  2. **Surge chasing**: drivers reposition to the edge of a high-surge zone to capture the surge when they accept a trip. Creates zone boundary artifacts — supply clusters just outside the zone.
  3. **Waiting out surge peaks**: experienced drivers learn surge decay curves. Rather than accepting a 2.0× trip, they wait 3 minutes for 2.5× as the surge escalates. Individual-optimal; collectively creates a supply hole.
  4. **Trip cherry-picking**: drivers near airports accept surge trips to premium destinations, ignoring short local trips. Supply in the zone appears sufficient, but trip completion rate falls.
- **Modeling strategic behavior**:
  - **Supply elasticity function**: `ΔS/ΔP` at a given surge level. But elasticity is heterogeneous — part-time drivers (high elasticity) vs. full-time drivers (low elasticity, already online).
  - **Behavioral clustering**: segment drivers into strategic archetypes (supply fishers, passive, maximizers). Model each segment's response function separately. Avoids model averaging that hides bimodal behavior.
  - **Multi-agent simulation**: simulate N drivers, each with a response function calibrated to historical behavior. Run the simulation under different surge policies. Find the policy where no driver has incentive to deviate (Nash equilibrium of the pricing game).
  - **Counterfactual off-policy evaluation**: use logged data from past surge events + off-policy evaluation to estimate what supply would have looked like under a different surge policy without running an A/B test.
- **Why this matters for the pricing model**: if you set surge based on current supply and ignore strategic response, you overshoot — surge goes up, drivers come online, surge falls, drivers leave, surge goes up again. Oscillation. The stable equilibrium requires predicting the supply response, not just observing it.

**Company context:** Uber Supply Pricing — Chris's explicit description: "modeling strategic driver responses." This is the differentiating topic.

**Common wrong answer:** "We'd model driver supply as a function of surge level using historical data." — Correct start, but ignores strategic adaptation. Drivers who know the pricing algorithm will exploit it. Staff answer models the strategic response, not just the historical correlation.

---

## Counterfactual Estimation in Pricing

### Q: How do you estimate the effect of a surge price change when you can only observe what happened at the price you actually charged?

**Answer (Staff level):**
- **The fundamental problem**: you set surge = 2.0×. You observe 8 drivers come online and 15 riders complete trips. You want to know: what would have happened at surge = 2.5×? You don't have that data.
- **Why this matters**: optimizing pricing requires knowing the demand/supply response curve `Q(P)`. But you can't directly observe the counterfactual.
- **Approach 1 — Natural experiments**: use exogenous events that shift price without changing demand: algorithm changes, city-level policy changes, road construction that raises ETA → effectively raises price. Compare outcomes in affected vs. control zones (difference-in-differences). Valid if the instrument is truly exogenous.
- **Approach 2 — Randomized price experiments (A/B test)**:
  - Randomize surge multiplier at zone-hour level. Zone A gets 2.0×, Zone B gets 2.5×, otherwise identical.
  - Problem: SUTVA (Stable Unit Treatment Value Assumption) is violated. If Zone A has low surge, drivers migrate to Zone B (interference). Standard A/B estimates are biased.
  - Fix: switchback experiments — alternate treatment in the same zone over time. Zone A gets 2.0× for 5 min, then 2.5× for 5 min. Reduces interference but introduces temporal autocorrelation.
- **Approach 3 — Structural demand estimation**:
  - Fit a demand model `D(P, X)` where X = observable confounders (time of day, weather, events). Use price variation from historical data as identification.
  - Problem: prices are set endogenously based on demand — OLS is biased (high demand causes high price AND high trips, creating spurious correlation). Fix: instrumental variable (IV) regression with an instrument that shifts price but not demand directly (e.g., driver app outage in adjacent zone).
- **Approach 4 — Off-policy evaluation (OPE)**:
  - Treat historical pricing decisions as a logged bandit problem. Each (zone, hour) is a context; the surge level is an action; the outcome is trips completed.
  - Use importance sampling (IS): `E[Y(π_new)] = E[(π_new(a|x) / π_old(a|x)) × Y]`. High variance when new policy differs significantly from logged policy.
  - Doubly robust estimator: combine IS with a direct model of `E[Y | a, x]`. Consistent if either the propensity model or the outcome model is correct.

**Company context:** Uber Supply Pricing (causal inference is mandatory for pricing teams). This is the question that separates candidates who've actually done this work from those who've only read about it.

**Common wrong answer:** "We'd A/B test different surge levels." — Correct in principle, but SUTVA violation makes naive A/B biased in a marketplace. Staff answer names the interference problem and the fix (switchback or cluster-level randomization).

---

## Spatial Surge Zone Design

### Q: How do you design surge pricing zones, and what are the ML and operational tradeoffs?

**Answer (Staff level):**
- **Why zones?**: continuous spatial pricing (every GPS point gets its own surge) is computationally infeasible and creates pricing instability. Zones aggregate supply/demand into computationally tractable units.
- **H3 hexagonal indexing**: Uber uses Uber H3 (open-sourced), a hierarchical hexagonal grid system. Hexagons (vs. squares) minimize edge effects — every cell has 6 equally distant neighbors, so supply can flow in from any direction symmetrically. Resolution 7 hexagons (~5 km² each) are typical for surge zones.
- **Zone granularity tradeoff**:
  - **Fine-grained zones** (resolution 9, ~0.1 km²): precise signal, but each zone has few trips → high variance in supply/demand estimates → noisy surge. Small zones are also exploited by drivers who hover at zone boundaries.
  - **Coarse zones** (resolution 6, ~36 km²): stable signal, but one concert in a corner of the zone triggers citywide surge — bad for riders who are far from the event.
  - **Dynamic granularity**: use fine zones during high-volume periods (rush hour) where signal is dense; aggregate to coarse zones during low-volume periods (3am) where fine zones have too little data.
- **Zone boundary effects**:
  - Drivers position themselves just outside a high-surge zone, then drive in to accept the surge trip. Results in supply appearing adequate at zone level (drivers counted in the zone) but being concentrated at the perimeter.
  - Fix: compute driver position at trip acceptance, not at zone entry. Or use overlapping zones with soft zone assignment (a driver at zone boundary contributes to both zones proportionally).
- **Zone stability**: zones shouldn't change their boundaries frequently — drivers learn zone shapes and plan accordingly. Rapid zone shape changes (to track demand more precisely) disrupt driver mental models → strategic behavior becomes harder to predict.

**Company context:** Uber Supply Pricing. H3 is Uber's own open-source technology — knowing it signals genuine familiarity with the domain.

**Common wrong answer:** "I'd use a grid-based approach." — H3 hexagonal indexing is the answer Uber expects; knowing why hexagons outperform squares (equal-distance neighbors, no diagonal artifacts) shows real domain depth.

---

## Supply & Demand Forecasting for Pricing

### Q: What does the forecasting pipeline look like for surge pricing, and how far ahead do you need to predict?

**Answer (Staff level):**
- **Why prediction matters**: reactive surge (raise multiplier when supply < demand NOW) is too late — drivers need 5–15 minutes to travel to the zone. You need predictive surge: raise the price before the demand spike so drivers pre-position.
- **Prediction horizons**:
  - **15-min ahead**: actionable for driver pre-positioning. At 15 min, uncertainty is low enough to be useful. This is the primary horizon.
  - **60-min ahead**: useful for driver quest targeting ("if you drive to downtown by 6pm, you'll likely earn $X") and for Uber's demand allocation planning.
  - **24-hour ahead**: day-of forecast used to set quest thresholds and bonus budgets.
- **Demand forecasting features**:
  - **Temporal**: hour of day, day of week, is_holiday, minutes_to_event_end (concerts, games)
  - **Spatial**: zone ID embeddings, historical demand at this zone × time-of-week
  - **Contextual**: weather (rain increases demand), special events (calendar API), nearby POIs
  - **Lag features**: trip requests in last 5/15/30 min (strong predictor; demand autocorrelates)
- **Supply forecasting features**:
  - Current online driver count in zone + adjacent zones
  - Driver status: available, on trip (ETA to trip end), offline
  - Historical supply at zone × hour-of-week (baseline)
  - Driver app engagement signals (surge notifications opened, app foreground)
- **Model architecture**: LightGBM on tabular features for demand. For supply: harder because supply is endogenous (it responds to price, which we're trying to set). Solution: forecast supply at baseline price (no surge), then model supply response as a separate elasticity function applied on top.
- **Demand elasticity as output**: the forecast should output not just `E[demand]` but `E[demand | price]` — a price-demand curve. This requires fitting demand elasticity from historical price variation. Implemented as: predict demand at reference price × apply elasticity function `D(P) = D_ref × (P / P_ref)^ε` where ε = price elasticity of demand (typically -0.3 to -0.8 for ridesharing).

**Company context:** Uber Supply Pricing. Prediction intervals are also required (not just point estimates) — Chris's team needs confidence intervals to set conservative surge that avoids over-surging with high probability.

**Common wrong answer:** "I'd predict demand and set surge proportionally." — Missing the supply response modeling (supply is endogenous to price), the elasticity output (not just demand level), and the prediction horizon rationale. Staff answer covers all three.

---

## Real-Time Pricing System Architecture

### Q: Design the core architecture for a system that makes 1M+ surge pricing decisions per second across 10,000+ cities.

**Answer (Staff level):**

**Scale constraints:**
- 1M decisions/sec globally = ~100 decisions/sec per city on average, but unevenly distributed (NYC, São Paulo, London are 10–100× the average)
- Decision latency budget: price must be computed and served within rider request flow — total budget ~200ms; pricing gets ~50ms
- Surge update frequency: every 30–60s per zone (faster = oscillation risk; slower = stale signal)

**Architecture:**

```
[Rider App Request] → API Gateway → Pricing Service (p99 < 50ms)
                                           ↑
                          Surge Cache (Redis, zone → multiplier)
                                           ↑ (refresh every 30s)
                          Surge Computation Engine
                          ┌────────────────────────────┐
                          │  Supply Signal Aggregator  │ ← Driver location stream (Kafka)
                          │  Demand Signal Aggregator  │ ← Trip request stream (Kafka)
                          │  Zone State Store          │ ← H3 zone supply/demand counts
                          │  Pricing Model (LightGBM)  │ ← Features → optimal multiplier
                          │  Multiplier Rounding       │ ← snap to [1.0, 1.25, 1.5, 2.0, ...]
                          └────────────────────────────┘
                                           ↑
                          Feature Store (Feast / Redis)
                          [historical demand, event calendar, weather]
```

**Key design decisions:**

1. **Decouple surge computation from request serving**: surge multiplier is precomputed every 30s per zone and cached in Redis. Rider request reads from cache (O(1)), not from ML model. This is why 1M+ req/sec is achievable — the model runs at 1 QPS per zone, not 1M.

2. **Zone-level aggregation via streaming**: Kafka consumer aggregates driver GPS events into H3 zone supply counts in a 60s rolling window. Flink or KSQL for real-time aggregation. Key: use approximate counts (HyperLogLog for unique drivers per zone) at scale.

3. **Multiplier discretization**: continuous surge (e.g., 1.83×) is confusing to drivers and riders. Snap to predefined tiers (1.0, 1.25, 1.5, 2.0, 2.5, 3.0+). This also prevents constant small fluctuations that trigger driver repositioning.

4. **Feedback dampening**: surge update uses exponential moving average of raw multiplier: `surge_t = α × computed_multiplier + (1-α) × surge_{t-1}`. Prevents oscillation when supply responds to surge and drives it down sharply.

5. **Failure modes and fallbacks**:
   - **ML model unavailable**: fall back to threshold rule (supply count < N → surge = 1.5×)
   - **Feature store stale**: cache last-known zone state; stale surge for 60s is acceptable
   - **Driver supply signal lost (Kafka lag)**: use last known supply count + apply decay factor (assume online drivers drift offline over time)
   - **Zone state missing (new city)**: use regional average as prior; bootstrap with cold-start surge model trained on similar cities

**Company context:** Uber Supply Pricing — Chris described "1M+ decisions per second" explicitly. This is the system design answer for his team.

**Common wrong answer:** Calling the ML model on every rider request. At 1M req/sec that's 1M model inferences/sec — impossible without massive GPU infrastructure. The correct answer is precomputed zone-level surge cached in Redis. This is a fundamental architecture insight separating Staff from Senior.
