# Ad Inventory Forecasting & Yield Optimization — ML Knowledge Q&A

P0: Netflix (MLE5 Ads Inventory Management & Forecasting team). Publisher-side ad tech.
Contrasts with the demand-side topics in `autobidding-and-pacing.md`.

---

## What Is Ad Inventory?

### Q: Define ad inventory and explain why forecasting it accurately is hard.

**Answer (Staff level):**
- **Ad inventory** = the pool of future ad impressions a publisher (Netflix, Reddit, Pinterest) has available to sell. One impression = one opportunity to show one ad to one user in one context.
- **Three dimensions** of inventory:
  1. **Volume**: how many impressions will be available in a time window?
  2. **Audience**: which users will see those impressions? (Demographic, behavioral, geographic segments)
  3. **Context**: what content is the user watching/browsing? (Genre, title, placement type)
- **Why forecasting is hard**:
  1. **Audience overlap / contention**: the same impression may satisfy multiple campaigns. A user watching a drama who is female, 25–34, in NYC satisfies dozens of targeting criteria simultaneously. Each campaign's "available inventory" overlaps with others'. You can't simply sum each campaign's forecast.
  2. **Hierarchical structure**: you forecast at aggregate level (total impressions/day) and disaggregate to targeting segments. Errors compound across the hierarchy.
  3. **External shocks**: content releases, events, sports seasons drive non-stationary demand spikes that are hard to predict from historical patterns alone.
  4. **Supply-side uncertainty**: for CTV (Netflix), ad-eligible viewing hours depend on member viewing behavior, which varies with content slate. A major series release changes total ad impressions available.
  5. **Booking horizon**: advertisers buy guaranteed inventory 30–90 days ahead. The forecast must be accurate at that horizon — not just for tomorrow.

**Company context:** Netflix MLE5 Inventory (this is the team's core product), Reddit (self-serve ad platform), Pinterest.

**Common wrong answer:** "I'd predict total ad requests and allocate proportionally to segments." — Ignores audience overlap (contention) between targeting segments. Proportional allocation without deduplication double-counts users who satisfy multiple segments and leads to overselling.

---

## Reach and Frequency Forecasting

### Q: How do you forecast reach (unique users) and frequency (impressions per user) for an ad campaign?

**Answer (Staff level):**
- **Reach**: number of unique users exposed to the ad. Not the same as impressions — showing 3 ads to 1 user = 3 impressions but reach of 1.
- **Reach curve**: reach grows sublinearly with impressions due to repeat exposure. Shape: `reach(n) ≈ audience_size × (1 - (1 - p)^n)` where p = probability a random impression hits a new user.
- **Frequency cap interaction**: if the campaign has a frequency cap (max 3 impressions/user/day), reach and frequency are jointly constrained. Higher cap → higher reach potential but more repeat exposure.
- **Forecasting reach at scale**:
  - **HyperLogLog (HLL)**: approximate cardinality estimation for unique users in a targeting segment at scale. Runs on Spark over historical impression logs. Error rate ~1–2% at massive scale.
  - **Bitmap-based approaches**: for smaller segments, exact user bitmaps allow set union/intersection for overlap computation. Memory-prohibitive at full user scale.
  - **Probabilistic models**: fit a Beta-Binomial model to historical reach curves per segment type. Predict reach as a function of planned impressions at booking time.
- **Cross-campaign contention**: when multiple campaigns target overlapping audiences, each campaign's reach reduces the available unique users for others. Must model audience intersection.
  - **Inclusion-exclusion** is exact but exponential in the number of segments.
  - **Approximation**: use pairwise overlap estimates (`|A ∩ B| ≈ |A| × |B| / total_users` under independence, with corrections for known correlations).

**Company context:** Netflix MLE5 Inventory. Advertisers buy campaigns with specific reach goals (e.g., "reach 5M unique viewers"). Accurate reach forecasting is what allows Netflix to sell with delivery guarantees.

**Common wrong answer:** "Reach = impressions / average frequency." — This is circular (frequency is what you're trying to predict) and wrong when frequency is non-uniform across users. Staff answer uses reach curves and HLL for large-scale computation.

---

## Audience Overlap and the Contention Problem

### Q: Multiple campaigns target overlapping audiences. How do you forecast available inventory per campaign without double-counting?

**Answer (Staff level):**
- **The contention problem**: Netflix has 100 campaigns all targeting "women, 25–34, US, who watched dramas in the last 30 days." Each campaign's forecast predicts a pool of users — but it's the same pool. If each campaign's forecast sums independently, you oversell by 100×.
- **Overlap graph**: model campaigns as nodes; edges represent audience overlap (estimated by `|A ∩ B| / |A ∪ B|`). High overlap campaigns compete for the same impressions.
- **Contention-aware forecasting**:
  1. **Simulation-based**: run ad server simulation over historical (or forecasted) traffic. Each simulated impression runs through the real auction logic. Allocate the impression to a winner. Count impressions won per campaign. Repeat N times to get a distribution.
  2. **Analytical approximation**: use a queueing theory model (M/M/c queue where "servers" = impressions, "jobs" = campaigns). Estimate allocation probabilities per campaign given contention.
  3. **Greedy allocation with priority**: rank campaigns by CPM or contract priority. For each simulated user visit, greedily assign the impression to the highest-priority eligible campaign. Tracks remaining available supply as campaigns are served.
- **Overselling risk**: if Netflix commits to 10M impressions across 100 campaigns and contention means only 9M can be delivered, 10% of campaigns under-deliver. This is a contract breach → make-goods (free inventory given as compensation) → revenue loss.
- **Underselling risk**: being too conservative in forecasting → sell 8M when 10M was available → 2M impressions go unsold → lost revenue.
- **Margin buffer**: sell at 80–90% of forecasted capacity to absorb forecasting error. The buffer is calibrated from historical forecast accuracy.

**Company context:** Netflix MLE5 Inventory — "inventory split and yield optimization" from the job description is exactly this problem.

**Common wrong answer:** "I'd forecast each campaign's inventory independently and sum." — Classic overselling bug. The contention problem is why inventory forecasting is harder than just predicting total impressions.

---

## Yield Optimization: Direct vs. Programmatic

### Q: What is the yield optimization problem in publisher ad inventory, and how does it differ from the buyer-side bidding problem?

**Answer (Staff level):**
- **Publisher yield**: total revenue earned per available impression. `Yield = eCPM × fill_rate`. Maximize this subject to advertiser delivery guarantees and member experience constraints.
- **Demand waterfall** (priority order for filling an impression):
  1. **Direct / guaranteed deals**: advertiser bought N impressions at a fixed CPM upfront. Publisher must deliver. Highest priority — revenue is already committed.
  2. **Private Marketplace (PMP)**: pre-negotiated deals with select advertisers bidding in a private auction. Higher floor prices than open auction.
  3. **Programmatic / open auction**: all DSPs compete in real-time. Highest eCPM wins. No delivery guarantee.
  4. **House ads**: Netflix's own promotion. Zero revenue but fills otherwise-empty inventory.
- **The yield optimization decision**: for each impression, which demand source wins? The naive waterfall (guaranteed first, then programmatic) is suboptimal — a guaranteed campaign may win an impression that a programmatic buyer would have bid $20 CPM for, while the guaranteed eCPM is only $5. But guaranteed campaigns must deliver, so you can't always pass them to programmatic.
- **Dynamic allocation (Google's approach)**: allow programmatic bids to compete with guaranteed campaigns by computing the guaranteed campaign's "shadow price" — the eCPM it must match to justify taking this impression over programmatic. If programmatic bid > shadow price, route to programmatic and bank the credit against the guaranteed campaign's future delivery.
- **Shadow price computation**: `shadow_price = guaranteed_CPM × (remaining_impressions_needed / remaining_available_inventory)`. As delivery deadline approaches and remaining inventory shrinks, shadow price rises → fewer impressions get diverted to programmatic → guaranteed delivery rate improves.
- **Floor price optimization**: minimum bid accepted in open auction. Too high → low fill rate (impressions go unmonetized). Too low → revenue left on table.
  - ML approach: fit a hazard model on historical clearing prices to estimate `P(clearing_price > floor | user, context)`. Set floor where `E[revenue | floor] = floor × P(clear > floor)` is maximized.

**Company context:** Netflix MLE5 Inventory — the job description explicitly names "dynamic pricing, rate card management, product packaging, inventory split and yield optimization."

**Common wrong answer:** "I'd give guaranteed campaigns top priority always." — Sub-optimal. Dynamic allocation allows guaranteed campaigns to share inventory with high-value programmatic demand while still meeting delivery commitments. Shadow pricing is the key mechanism.

---

## Ad Inventory Forecasting Models

### Q: What ML models are used for ad inventory forecasting, and what are the unique challenges vs. standard time-series forecasting?

**Answer (Staff level):**
- **What you're forecasting**: future impression volume by targeting segment (e.g., "impressions available for women 25–34 watching drama content in the US next Tuesday").
- **Hierarchical structure**:
  - Level 0: total impressions per day (most stable; seasonal patterns)
  - Level 1: by geography (US vs. global)
  - Level 2: by content genre + day part
  - Level 3: by demographic segment
  - Level 4: targeting combination (the actual booking unit)
  - Bottom level has sparse data; top level is over-aggregated. Forecast top-down with allocation models, or bottom-up with reconciliation.
- **Hierarchical forecasting approaches**:
  1. **Top-down**: forecast total, disaggregate to segments using historical allocation shares. Fast but propagates top-level errors.
  2. **Bottom-up**: forecast each segment independently, aggregate. Captures segment-specific trends but ignores aggregate constraints.
  3. **Optimal reconciliation (MinT)**: forecast all levels, then reconcile to be consistent using a minimum-variance optimal combination. Used by Netflix, Amazon for inventory forecasting.
- **Features**:
  - Lag features: impressions at same segment × day-of-week, 1/2/4 weeks ago
  - Calendar: holidays, content release schedule (a major new season → more viewing → more impressions)
  - Trend: long-term subscriber growth, ad tier adoption rate
  - Seasonality: weekly (weekends > weekdays), annual (Q4 > Q1 for ad spend)
  - Content metadata: known upcoming releases that will drive viewership
- **Models**:
  - **LightGBM with lag features**: fast, handles non-linearity, good for tabular forecasting with many segment-level features
  - **Prophet**: handles seasonality and holidays well; works at aggregate level
  - **LSTM / Temporal Fusion Transformer**: captures long-range temporal dependencies; better when segment count is manageable
  - **Ensemble**: combine LightGBM (short-term accuracy) + Prophet (trend/seasonality) + content release model (event spikes)
- **Forecast horizon**: 30–90 days (advertiser booking lead time). At 90 days, forecast error of ±15% is typical. Must model forecast uncertainty for overbooking decisions.

**Company context:** Netflix MLE5 Inventory (the team's primary product is a "state-of-art realtime inventory forecasting solution").

**Common wrong answer:** "I'd use ARIMA on historical impression counts." — ARIMA doesn't handle hierarchical structure, doesn't incorporate content release calendar, and doesn't produce segment-level forecasts. Staff answer addresses the hierarchical problem and feature engineering.

---

## Ad Server Simulation

### Q: Why do inventory forecasting teams build ad server simulations, and what does the simulation pipeline look like?

**Answer (Staff level):**
- **Purpose**: answer "what if" questions about inventory allocation changes without live experimentation. Examples:
  - "If we increase the floor price from $5 to $8 CPM, what happens to fill rate and total revenue?"
  - "If we add a new targeting dimension, how does contention change?"
  - "If a new campaign books 10M guaranteed impressions, how does it affect delivery for existing campaigns?"
- **Simulation pipeline**:
  1. **Traffic replay**: take historical ad request logs (user visits with features: demographics, content, device, time). These are the simulated "impressions to allocate."
  2. **Synthetic demand**: for "what-if" questions about new campaigns, inject synthetic campaigns with specified targeting criteria, budget, and CPM.
  3. **Auction simulation**: for each historical request, run the modified auction logic (new floor price, new priority rules, new campaigns). Determine winner using the new rules.
  4. **Outcome aggregation**: aggregate wins/losses per campaign, fill rate, total revenue, delivery rate.
  5. **Comparison**: compare simulated outcomes vs. historical actuals. Difference = estimated effect of the proposed change.
- **Monte Carlo extension**: for forward-looking simulations (next 30 days), sample future traffic from the inventory forecast distribution. Run N simulations. Report mean and confidence intervals on revenue and delivery rate.
- **Validation**: before using simulation for decisions, validate against held-out historical data. Run the simulator on last month's traffic with last month's actual campaigns and verify it reproduces actual outcomes within ±5%.
- **High performance requirements**: the job description says "high performance ad server simulations." At Netflix scale, replaying 5M daily impressions × 50 Monte Carlo runs × nightly = 250M simulated auctions per night. Must be parallelized on Spark (impressions are independent — embarrassingly parallel).

**Company context:** Netflix MLE5 Inventory — the job description specifically names "scalable simulation solution to model different inventory scenarios."

**Common wrong answer:** "I'd use A/B tests to evaluate pricing changes." — Live A/B tests take weeks and can't test hypothetical new campaigns. Simulation provides fast directional estimates (hours, not weeks) for decisions that inform the next sales cycle.

---

## CTV-Specific Constraints

### Q: How does Connected TV (CTV) advertising differ from display/mobile, and what ML implications does this create?

**Answer (Staff level):**
- **What is CTV**: advertising on internet-connected television screens (Netflix, Hulu, Disney+, Peacock). Distinguished from linear TV (broadcast) and from mobile/desktop display.
- **Key differences from display advertising**:

  | Dimension | Display (web/mobile) | CTV (Netflix) |
  |---|---|---|
  | **User identification** | Third-party cookies, IDFA/GAID | No cookies; IP-based or first-party ID |
  | **Audience measurement** | Deterministic (1:1 user tracking) | Probabilistic + panel-based (household, not individual) |
  | **Ad completion rate** | 30–70% (skippable) | 90%+ (mid-roll in content; often unskippable) |
  | **Co-viewing** | Individual screen | Multiple viewers per TV → household targeting, not individual |
  | **Ad format** | Banner, video, native | Video only (15s, 30s); VAST standard |
  | **Attribution** | Click-through tracking | View-through attribution (no click available on TV) |

- **ML implications**:
  1. **No third-party cookies**: audience segments built on Netflix's first-party viewing history + account data. No cross-site behavioral data. Lookalike modeling uses Netflix internal signals only.
  2. **Household vs. individual targeting**: a Netflix account has 5 profiles. Who is watching right now? Use active profile + time-of-day + content genre as proxy for the individual viewer. Probabilistic "viewer in household" model.
  3. **View-through attribution**: there's no click on TV. Attribution must be view-through: did the user who saw the ad take an action (subscription, visit brand website) within a window (1–7 days)? Requires probabilistic matching or controlled geo holdout.
  4. **High completion rate is a double-edged sword**: advertisers value the captive audience. But an irrelevant ad with a captive audience causes more brand damage than a skipped ad on mobile. Brand safety and relevance matter more, not less.
  5. **VAST compliance**: Netflix's ad server must conform to VAST (Video Ad Serving Template) standard for compatibility with ad agency workflows. ML models must output VAST-compatible responses.
  6. **Panel reconciliation**: Nielsen and other panels provide a ground truth reach measurement for TV. Netflix must reconcile its deterministic first-party data with panel estimates for reporting to advertisers.

**Company context:** Netflix MLE5 Inventory — "experience working in the CTV space and knowledge of its unique constraints" is listed as a nice-to-have.

**Common wrong answer:** "CTV is just video advertising." — Missing the co-viewing problem, the attribution challenge (no click), and the panel reconciliation requirement. Staff answer knows CTV-specific constraints and their ML implications.
