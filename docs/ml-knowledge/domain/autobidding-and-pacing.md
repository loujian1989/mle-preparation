# Autobidding & Pacing — ML Knowledge Q&A

P0: Reddit (Staff ads), Pinterest, Meta. Core ML problem in any ads system with budget management.

---

## Autobidding Fundamentals

### Q: What is autobidding and how does it differ from manual bidding?

**Answer (Staff level):**
- **Manual bidding**: advertiser sets a fixed bid per auction (e.g., $2.00 per click). Simple, but requires constant tuning and doesn't adapt to conversion probability variation across users/contexts.
- **Autobidding**: advertiser specifies a goal (target CPA, target ROAS, maximize conversions within budget), and the system computes bids dynamically per auction. The system bids high when conversion probability is high and low when it's low — same budget, more conversions.
- **Value of autobidding**: bid = f(pConversion | context, user) × conversion_value. For a $50 CPA target, bid high when pCVR is 10% (expected value = $5 per click ≈ matches $50 CPA target), bid low when pCVR is 0.5%.
- **System components**:
  1. **Conversion prediction model**: pCVR model (often separate from pCTR — sparse signal, delayed labels).
  2. **Bid calculator**: maps predicted value to a bid given budget/goal constraints.
  3. **Pacing controller**: ensures budget is spent smoothly, not exhausted in first hour.
  4. **Feedback loop**: update bid multipliers based on observed CPA vs. target CPA.

**Company context:** Reddit (Staff), Meta (Value Optimization), Pinterest (Conversion Objectives), Google (Smart Bidding).

**Common wrong answer:** "Autobidding just maximizes CTR subject to a budget." — Wrong objective. Autobidding optimizes for conversion value, not clicks. Optimizing CTR without conversion signal is bandit-level, not autobidding.

---

## Constrained Optimization Formulation

### Q: How do you formulate autobidding as an optimization problem?

**Answer (Staff level):**
- **Goal**: maximize total conversion value subject to a budget constraint.
- **Primal problem** (over auction opportunities i):
  ```
  maximize  Σᵢ xᵢ · vᵢ
  subject to Σᵢ xᵢ · cᵢ ≤ B    (budget constraint)
             xᵢ ∈ {0, 1}        (win/lose auction)
  ```
  where `vᵢ` = conversion value of impression i, `cᵢ` = cost of winning auction i, `B` = total budget.
- **Lagrangian relaxation**: introduce multiplier λ for the budget constraint:
  ```
  L(x, λ) = Σᵢ xᵢ · (vᵢ − λ · cᵢ) + λB
  ```
  Optimal bid for impression i = `vᵢ / λ`. λ acts as the "price of budget" — if budget is tight, λ is large → bids are shaded down.
- **Practical interpretation**: λ is the target CPA. `bid_i = pCVR_i × conversion_value / λ`. Advertiser sets target CPA = λ; system bids accordingly.
- **Target ROAS variant**: `bid_i = pCVR_i × order_value_i / target_ROAS`. Same structure, revenue-weighted.
- **λ tuning**: λ is updated in a feedback loop. If actual CPA > target, raise λ (shade bids down). If actual CPA < target and budget is not exhausted, lower λ (bid more aggressively).

**Company context:** Reddit (Staff ML — this is the core formulation question), Meta, Google.

**Common wrong answer:** "I'd train a model to predict the right bid directly." — End-to-end bid prediction conflates the conversion prediction problem with the budget allocation problem. Separating pCVR estimation from bid calculation via λ is cleaner, interpretable, and enables budget constraint enforcement.

---

## pCVR Modeling for Autobidding

### Q: How is pCVR modeling different from pCTR modeling in the context of autobidding?

**Answer (Staff level):**
- **Signal sparsity**: CVR is ~10–100× sparser than CTR. Most clicks don't convert. Training data is severely imbalanced (1% conversion rate is high). Requires careful negative sampling and class weighting.
- **Attribution lag**: conversions occur hours/days after the click. A model trained on recent data has truncated conversion labels — clicks from yesterday look like non-conversions even if they convert tomorrow.
  - Fix: apply a survival model or empirical lag distribution correction. Only train on impressions where the attribution window has closed. Use calibration correction for recent impressions.
- **Multi-touch attribution**: one conversion may be influenced by multiple ads. Direct attribution inflates individual ad CVR. Must decide: last-touch, first-touch, or data-driven attribution (Shapley values, Markov chains).
- **Value heterogeneity**: conversions are not equal (a $5 purchase vs. $500 purchase). Model should predict `E[value | click, context]` not just `P(convert | click)`, especially for target ROAS campaigns.
- **Covariate shift**: the population of clicks that reach conversion is not random — high-intent users click AND convert. Training on (click → convert) conflates CTR-selection bias with true pCVR.
  - Fix: train on impression-level with both pCTR and pCVR models; or use importance weighting.

**Company context:** Meta (Value Optimization team), Reddit (Staff), Pinterest.

**Common wrong answer:** "I'd train on (clicked, converted) pairs." — Ignores attribution lag, multi-touch, and selection bias. Staff-level answer addresses all three.

---

## Budget Pacing

### Q: What is budget pacing and why is it a hard ML problem?

**Answer (Staff level):**
- **Budget pacing**: given a daily/lifetime budget B, control the bid or throttle rate over time so the budget is spent smoothly (not exhausted in the first hours of the day).
- **Why it's hard**:
  1. **Traffic is non-stationary**: auction volume follows a diurnal pattern (peaks at 8am–10pm). Without pacing, an ASAP strategy exhausts budget during the morning peak before the afternoon high-value auctions arrive.
  2. **Stochastic**: you can't know future auction opportunities, only past traffic distributions.
  3. **Competing objectives**: under-delivery (unspent budget) = lost revenue for the platform. Over-delivery (overspending) = violates advertiser commitment → trust/legal issue.
  4. **Budget granularity**: different advertisers have daily budgets from $5 to $5M — pacing logic must generalize.
- **Two approaches**:
  1. **Throttling (impression-level)**: sample auctions probabilistically. Throttle rate `p = remaining_budget / expected_remaining_spend`. Simple, doesn't require bid modification.
  2. **Bid shading (value-level)**: multiply bids by a pacing factor `ρ ∈ [0,1]`. When ahead of pace, `ρ < 1` (shade bids down → win fewer auctions). When behind, `ρ → 1`. Preferred: more targeted than throttling since it shades low-value auctions first.

**Company context:** Reddit, Meta, Pinterest, any platform running budget-constrained campaigns.

**Common wrong answer:** "I'd just proportionally distribute the budget by hour." — Fixed hourly allocation ignores the fact that auction quality (pCVR, conversion value) varies within the day and across days. Optimal pacing prioritizes high-value auctions within the budget, not equal hourly spend.

---

## PID Controller for Pacing

### Q: How does a PID controller work for budget pacing, and what are its limitations?

**Answer (Staff level):**
- **PID formulation**: treat pacing as a control problem.
  - **Target**: spend rate = `B / T` (budget divided by day length).
  - **Observed**: actual spend so far vs. expected spend so far.
  - **Error signal**: `e(t) = actual_spend(t) − target_spend(t)`.
  - **PID update**:
    ```
    ρ(t) = Kp·e(t) + Ki·∫e(t)dt + Kd·de(t)/dt
    ```
    Adjust throttle/bid multiplier `ρ` up or down based on spend deviation.
- **P term**: reacts to current over/under-spend. Fast but oscillates.
- **I term**: corrects accumulated error (if consistently underspending all morning, I term increases bids). Risk: wind-up if budget is genuinely unavailable (e.g., no matching auctions).
- **D term**: anticipates future deviation from rate of change. Dampens oscillation.
- **Hyperparameter tuning**: Kp, Ki, Kd are typically tuned per budget tier (large budgets tolerate more oscillation; small budgets need gentler control).
- **Limitations**:
  - PID assumes smooth, continuous dynamics. Ad auctions are discrete and bursty. Traffic spikes (Super Bowl, flash sale) break linear assumptions.
  - PID doesn't model the value distribution of future auctions — it paces for even spend, not optimal value per dollar.
  - Better alternative: **model-based pacing** — use a learned spend forecast model to predict remaining auction value, then solve for optimal λ(t) at each time step.

**Company context:** Reddit (pacing controller design), Meta. PID is the baseline — Staff answer knows its failure modes.

**Common wrong answer:** "PID is standard and works well." — At Staff bar, name the limitations and when you'd replace it with a model-based controller.

---

## PID in Autobidding (CPA Control Loop)

### Q: How is PID used in autobidding itself (not just pacing), and how does it differ from the pacing PID?

**Answer (Staff level):**
- A production autobidding system runs **two nested PID loops** at different timescales:

  **Outer loop — CPA control (autobidding):**
  - Controls: λ (the Lagrangian multiplier = effective CPA target)
  - Error signal: `e(t) = actual_CPA(t) − target_CPA`
  - If actual CPA > target → raise λ → `bid = pCVR × value / λ` decreases → win fewer auctions → CPA falls
  - If actual CPA < target and budget not exhausted → lower λ → bid more aggressively → more conversions
  - Update cadence: **1–6 hours** (conversions are delayed; faster updates react to noise, not signal)

  **Inner loop — spend rate control (pacing):**
  - Controls: bid multiplier ρ or throttle rate
  - Error signal: `e(t) = actual_spend(t) − target_spend(t)`
  - Update cadence: **every 5 minutes** (spend signal is immediate)

- **Why separate loops?** Spend signal is available in seconds (auction wins). Conversion signal has hours of lag (post-click window). Mixing them in one loop creates instability — the controller reacts to noise before the real conversion signal arrives.

- **Attribution lag correction**: actual_CPA at time t is systematically underestimated because many clicks haven't converted yet. Must apply a lag multiplier:
  ```
  corrected_CPA(t) = observed_CPA(t) / attribution_completion_rate(t)
  ```
  where `attribution_completion_rate` = fraction of conversions in a 7-day window that have been observed by hour t of day. Typically 20–30% at hour 2, 60–70% by end of day.

- **Anti-windup**: if budget is exhausted mid-day (no more auctions to win), the integral term would keep accumulating negative error → λ falls to zero → next day starts with dangerously low bids. Fix: clamp I-term accumulation when budget is depleted.

- **Steady-state behavior**: in equilibrium, the outer PID loop should hold `actual_CPA ≈ target_CPA` within ±10–15%, with the inner loop ensuring budget is spent smoothly. CPA overshoot on day 1 for a new campaign is expected (insufficient conversion history for good lag correction).

**Company context:** Reddit Staff (distinguishing the two loops is an L6-level answer), Meta, Google Smart Bidding architecture discussions.

**Common wrong answer:** "PID adjusts the bid to hit the CPA target." — Too vague. Staff answer names the two nested loops, their different timescales, and the attribution lag problem that makes the outer loop non-trivial.

---

## Autobidding & Pacing: Method Comparison

### Q: Compare PID, model-based, gradient descent, LP, and RL approaches for autobidding and pacing — when do you use each?

**Answer (Staff level):**

| Method | What it controls | Update cadence | Pros | Cons | When to use |
|---|---|---|---|---|---|
| **PID** | λ (CPA) or pacing factor | 5 min (pacing), 1–6 hr (CPA) | Simple, interpretable, low-latency, no training data needed | Oscillation, hand-tuned gains, stale signal for CPA loop, doesn't model future value distribution | Production baseline for both pacing and CPA control; robust and debuggable |
| **Model-based pacing** | Budget allocation over time | Per pacing interval (5 min) | Value-aware; can hold budget for high-value future windows | Requires spend forecast model; forecast errors compound; more complex failure modes | High-budget campaigns where value density varies significantly by hour of day |
| **Gradient descent on λ** | λ (CPA) | Per batch (hourly) | Principled convergence; no manual gain tuning; adapts to non-stationarity | Delayed, noisy gradient signal (attribution lag); unstable if learning rate too high | When dense conversion signal is available and you want to eliminate manual Kp/Ki/Kd tuning |
| **Constrained LP** | Budget allocation across portfolio | Batch (hourly/daily) | Globally optimal allocation across many campaigns simultaneously | Not real-time; LP must be re-solved each refresh; ignores intra-day dynamics | Portfolio pacing: allocating a shared budget pool across dozens of campaigns |
| **RL (DQN, PPO, etc.)** | Bid per auction or λ per interval | Per auction or per episode | Handles long-horizon dependencies; adapts to non-stationary traffic and marketplace dynamics | Data hungry; credit assignment hard with delayed reward; difficult to constrain (hard CPA/budget guarantees); hard to debug | Research and long-horizon optimization; not yet standard in production pacing for most platforms |

**Key interview framing — why PID dominates in production:**
- Interpretable: an oncall engineer can understand why λ changed at 2am without reading a neural network.
- Debuggable: when CPA spikes, you can inspect the error signal, integral term, and clamp logic directly.
- Latency: PID update is O(1); RL inference or LP solve adds latency at update time.
- Graceful degradation: PID with a reasonable Kp still functions if Kd is misconfigured. RL with a bad reward function can catastrophically over-bid.

**When RL is worth considering:**
- Long-horizon campaigns where today's bid affects tomorrow's auction win rate (marketplace feedback loops).
- Platforms with dense reward signal (e.g., e-commerce where purchases happen within minutes of click).
- Research teams with the infrastructure to simulate, train offline, and shadow-deploy safely before production.

**Gradient descent on λ vs. PID:**
- Mathematically, PID is a heuristic optimizer for λ; gradient descent is principled.
- In practice, PID often outperforms gradient descent because the gradient signal is corrupted by attribution lag, auction noise, and dataset shift. PID's hand-tuned Kp effectively acts as a robust learning rate that doesn't blindly follow a noisy gradient.
- **Hybrid approach**: use PID for fast pacing corrections + gradient descent for slow λ updates on a 24-hour cycle (once the attribution window closes and gradient is cleaner).

**Company context:** Reddit Staff (method comparison expected at L6), Meta, Pinterest. This is a Staff-differentiating answer — most candidates know PID; few can explain why you'd choose it over RL.

**Common wrong answer:** "RL is the best approach because it can optimize long-term value." — RL is theoretically appealing but practically brittle in budget-constrained, latency-sensitive systems. Staff answer acknowledges RL's theoretical advantage while explaining why PID dominates in production.

---

## Model-Based Pacing

### Q: How does model-based pacing improve on PID, and what does the ML system look like?

**Answer (Staff level):**
- **Core idea**: at time t, predict the distribution of future auction opportunities (volume, quality, pCVR) for the remainder of the day. Use this forecast to solve for the optimal budget allocation over time — bidding more on high-value future windows, less on low-value windows.
- **System components**:
  1. **Spend forecast model**: predicts `spend_rate(t, t+Δ)` given time of day, day of week, advertiser category, historical traffic pattern. Typically a time-series model (LightGBM with lag features, or LSTM for capturing seasonality).
  2. **Value density estimator**: predicts the distribution of `conversion_value / cost` for future auctions. Allows the pacing controller to compute expected ROI for different bid levels.
  3. **Dynamic λ solver**: at each pacing interval (e.g., every 5 min), solve for λ(t) that maximizes expected remaining conversions subject to remaining budget. Closed-form solution using the Lagrangian: `λ(t) = remaining_budget / E[remaining_conversion_value]`.
  4. **Feedback**: update forecast and λ every pacing interval. If conversion rate is trending below forecast, raise λ early to conserve budget for better auctions.
- **Advantage over PID**: explicitly optimizes value per dollar, not just spend rate. Can sacrifice cheap low-converting auctions to save budget for a high-converting evening window.

**Company context:** Reddit Staff (this is the L6-level design expected), Meta, Google Smart Bidding internals.

**Common wrong answer:** Describing PID as the solution without acknowledging value-aware allocation. Model-based pacing is the expected answer for a Staff MLE role with budget optimization in scope.

---

## Exploration vs. Exploitation in Autobidding

### Q: How do you handle exploration in autobidding, particularly for new advertisers or new creatives?

**Answer (Staff level):**
- **Cold start problem**: new advertiser or new creative has no conversion history → pCVR model falls back to category/demographic priors → bids are uncertain → under-delivery or CPA overshoot.
- **Tension**: platform wants to bid conservatively (protect advertiser CPA commitment). Advertiser wants volume (learn quickly). Each auction foregone due to low confidence is lost learning signal.
- **Exploration strategies**:
  1. **Thompson sampling on pCVR estimate**: maintain a posterior over CVR (Beta-Binomial). Sample from posterior each auction → naturally explores when uncertainty is high, exploits when not.
  2. **UCB on CVR**: bid based on upper confidence bound of CVR rather than point estimate. Conservative exploration.
  3. **Exploration budget**: set aside a fixed fraction (e.g., 5%) of budget for exploratory bids (bid at the CPA-target limit rather than shaded). Labeled separately for attribution.
  4. **Cross-advertiser transfer**: use CVR from similar advertisers in the same category as a prior. Reduces cold-start length significantly.
- **Exploitation-aware pacing**: don't explore during high-value windows (end of day if budget is nearly gone). Exploration should happen when budget is plentiful and traffic is lower-value.

**Company context:** Reddit (new advertiser onboarding is a key product metric), Meta (value optimization cold start).

**Common wrong answer:** "Use a default bid until there's enough data." — No active exploration → cold start lasts longer → worse advertiser experience → churn. Staff answer recognizes this as a bandit problem.

---

## Autobidding System Design

### Q: Design an autobidding system for a $500M/year ads platform (e.g., Reddit).

**Answer (Staff level):**

**Components and data flows:**

```
[Conversion Events] → Attribution Pipeline → CVR Label Store
[Auction Logs]      →  ┐
[User Features]     →  ├→ pCVR Training Pipeline → pCVR Model (refreshed daily)
[Ad Features]       →  ┘
                            ↓
[Real-time Auction]  → pCVR Serving (p99 < 10ms) → Bid Calculator
                                                           ↓
[Budget Controller] → λ(t) per campaign  →────────────→  bid = pCVR × value / λ
         ↑                                                 ↓
[Spend Tracker]  ←──────────────────── Auction Result (win/lose, cost)
         ↑
[Forecast Model] → predicted_spend(t+Δ) → λ_adjustment
```

**Key design decisions:**
- **pCVR refresh**: daily batch training on last 30 days with recency weighting. Attribution window: 7-day post-click for most categories (longer for high-consideration purchases).
- **λ update cadence**: every 5 minutes per campaign. Prevents oscillation while still reacting to intra-day traffic shifts.
- **Spend tracking**: use a distributed counter (Redis) per campaign with TTL-based windows. Atomic increment per auction win. Consistency: accept eventual consistency — brief over-spend is acceptable; hard limit enforced via daily reconciliation.
- **Failure modes**:
  - pCVR model serving failure → fall back to category-average CVR + conservative λ.
  - Budget tracker unavailable → throttle to 50% of normal bid rate until recovered.
  - Attribution pipeline delay → use time-discounted labels (down-weight unconfirmed conversions).
- **Monitoring**: CPA deviation by campaign tier (small/medium/large budgets behave differently), under-delivery rate (% campaigns spending < 90% of budget), pCVR calibration by category.

**Company context:** Reddit Staff ML (full system design), Meta, Pinterest.

---

## Delivery vs. Performance Trade-off

### Q: An advertiser is consistently underspending their budget. Walk through your diagnosis and fix.

**Answer (Staff level):**
- **Step 1 — Diagnose root cause** (each has a different fix):
  1. **Audience too narrow**: targeting constraints exclude most auctions. Fix: expand targeting, or flag to account team.
  2. **Bid too low**: λ is too high (over-conservative pacing). Actual CPA is well below target → room to bid more aggressively. Fix: lower λ.
  3. **pCVR underestimation**: model predicts low conversion probability → low bids → low win rate. Fix: check calibration by category; if model is systematically low, apply calibration correction.
  4. **Budget exhausted on wrong auctions**: bid high on low-value morning traffic, budget gone before high-value afternoon traffic. Fix: model-based pacing to preserve budget for high-value windows.
  5. **Frequency cap hit**: creative has been shown to all eligible users up to the cap. Fix: expand creative set or adjust frequency cap policy.
- **Triage order**: check audience size → check pacing logs (spend rate by hour) → check CVR calibration → check bid competitiveness (win rate at current bids) → check frequency data.
- **Metric to monitor**: delivery rate (actual spend / budget), win rate by hour, CPA vs. target CPA.

**Company context:** Reddit (advertiser health is a core product metric for the Staff ads MLE role), Pinterest, Meta.

**Common wrong answer:** "Just raise the bid." — Without diagnosing the root cause, raising the bid may overshoot CPA target or not help at all (if the problem is audience size). Staff answer starts with systematic diagnosis.
