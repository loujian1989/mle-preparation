# A/B Testing & Experimentation — ML Knowledge Q&A

P0: Universal. Netflix (take-home quiz is A/B-heavy), Meta, Reddit, Stripe. The Netflix
MLS5 Ads role explicitly requires "online and offline evaluation frameworks."

---

## Experiment Design Fundamentals

### Q: How do you design an A/B test for an ML model change? Walk through the full setup.

**Answer (Staff level):**
- **Step 1 — Define the metric hierarchy before running**:
  - **Primary metric**: the one decision is made on. Single metric only. "Improve conversion rate" is a metric; "improve things" is not.
  - **Guardrail metrics**: metrics that must not degrade. If primary metric improves but session depth falls 5%, the experiment fails. Define guardrail thresholds upfront or you'll rationalize them post-hoc.
  - **Diagnostic metrics**: not decision criteria, but explain why the primary metric moved.
- **Step 2 — Randomization unit**: what entity is randomly assigned to treatment/control?
  - User-level: standard. Consistent experience per user. Use when treatment affects user session.
  - Request-level: only valid when treatment is stateless (doesn't affect subsequent user behavior). Risk: same user sees both variants → contamination.
  - Session-level: reasonable compromise for session-scoped changes.
  - **Key rule**: randomization unit must be ≥ the unit of analysis. If you randomize at user level but analyze at request level, you're fine. The reverse (randomize at request, analyze at user) inflates false positives.
- **Step 3 — Sample size / power calculation**:
  - Effect size: what minimum detectable effect (MDE) matters? Smaller MDE → larger sample needed.
  - `n = (z_α/2 + z_β)² × 2σ² / δ²`
    - `z_α/2 = 1.96` (two-tailed, α=0.05), `z_β = 0.84` (power=80%)
    - `σ²` = variance of metric, `δ` = MDE
  - For conversion rates: `σ² ≈ p(1-p)`, so `n ≈ (2.49)² × 2p(1-p) / δ²`
  - **In practice**: use a power calculator. But understand the inputs — variance of your metric is the key unknown, and it often needs to be estimated from holdout data.
- **Step 4 — Runtime**:
  - Minimum: 1–2 full weeks to capture weekly seasonality. Never stop at "we hit significance."
  - Maximum: defined upfront based on opportunity cost of running the experiment.
  - Pre-commit to runtime before looking at results.
- **Step 5 — Sanity checks before launch**:
  - A/A test: verify the randomization is working (no pre-existing difference between groups).
  - Check traffic split is correct (50/50 within ±2% is acceptable; larger imbalance signals a bug).
  - Check sample ratio mismatch (SRM): if assignment ratio is 50/50 but observed is 55/45, randomization is broken. SRM invalidates the experiment.

**Company context:** Universal. Netflix take-home quiz tests A/B design explicitly. Meta, Reddit, and Stripe probe it in system design rounds.

**Common wrong answer:** "I'd run it until p < 0.05." — Stopping at significance is p-hacking. Pre-committed runtime is mandatory. Also: no mention of guardrail metrics is a Staff-bar miss.

---

## CUPED / Variance Reduction

### Q: What is CUPED and why does it matter for A/B testing at scale?

**Answer (Staff level):**
- **Problem**: reducing variance reduces required sample size → shorter experiments → faster iteration. This is a scaling lever, not a statistical nicety.
- **CUPED (Controlled-experiment Using Pre-Experiment Data)**: adjust the outcome metric using a pre-experiment covariate to reduce variance.
  ```
  Y_cuped = Y - θ × (X - E[X])
  ```
  where `Y` = metric during experiment, `X` = same metric for the same user before the experiment, `θ = Cov(Y,X)/Var(X)`.
- **Intuition**: if user A always has high engagement (pre-experiment), and user B always has low engagement, their outcomes differ for reasons unrelated to the treatment. CUPED removes this baseline variation, leaving only treatment-driven variance.
- **Variance reduction**: typically 30–70% reduction. A 50% variance reduction → need half the sample size → run experiment in half the time (or detect effects twice as small).
- **Requirements**: pre-experiment data for the same metric over the same users. Works best when pre/post correlation is high (typically > 0.3 for engagement metrics; usually very high since the same user tends to have similar behavior week-to-week).
- **When CUPED doesn't help**: new users (no pre-experiment history); metrics that are not stable over time (e.g., rare events like subscription cancellation); short experiments where pre-period data is unavailable.
- **Extensions**: CUPAC (CUPED with ML-based covariate) — use a model to predict the outcome from pre-experiment features. Higher explained variance → higher variance reduction. Used at Airbnb, LinkedIn.

**Company context:** Netflix (explicitly used in their experimentation platform). Meta, Reddit. The Netflix MLS5 role mentions "offline evaluation frameworks" — CUPED is the standard variance reduction technique.

**Common wrong answer:** "I'd increase sample size to detect smaller effects." — Correct but expensive. CUPED achieves the same power improvement without more users, which is often the binding constraint.

---

## Multiple Testing

### Q: How do you handle multiple testing in an experiment with many metrics?

**Answer (Staff level):**
- **The problem**: if you test 20 metrics at α=0.05 each, the probability of at least one false positive is `1 - (0.95)^20 = 64%`. Multiple testing inflates false positive rate.
- **Family-wise error rate (FWER) control** — Bonferroni: `α_adjusted = α / m` where m = number of tests. Controls P(any false positive). Too conservative when tests are correlated.
- **False discovery rate (FDR) control** — Benjamini-Hochberg: controls E[false discoveries / total discoveries]. Less conservative than Bonferroni. Appropriate when correlated tests are expected (metrics often move together).
  - Procedure: sort p-values p₁ ≤ p₂ ≤ ... ≤ pₘ. Reject H₀ for all i where pᵢ ≤ (i/m) × α.
- **Hierarchical testing**: pre-define a primary metric and test only it for the decision. Test secondary metrics only if primary is significant. Eliminates most of the multiplicity problem by design.
- **Staff-level answer**: the right solution is metric hierarchy (one primary), not statistical correction. Correction is what you do when you have to test many metrics for regulatory or compliance reasons. In a product experiment: pick one metric, commit to it.
- **Practical rule**: if you find yourself running corrections, you have too many primary metrics. The experiment design is the fix, not the statistics.

**Company context:** Netflix (metrics governance is explicit in their culture), Meta, Stripe.

**Common wrong answer:** "I'd apply Bonferroni correction to all metrics." — Mechanically correct but misses the point. Staff answer is: don't design experiments that need multiple testing correction; if you must, use FDR (not FWER) because your metrics are correlated.

---

## Network Effects and SUTVA Violations

### Q: Standard A/B tests assume SUTVA (no interference between units). When is this violated, and how do you handle it?

**Answer (Staff level):**
- **SUTVA (Stable Unit Treatment Value Assumption)**: a unit's outcome depends only on its own treatment assignment, not on others'. Violated when units interact.
- **When SUTVA is violated**:
  1. **Two-sided marketplaces**: Uber, Airbnb. If treatment improves driver matching, riders in control benefit too (network is shared). Treatment effect is underestimated.
  2. **Social networks**: Meta, Reddit. If you show user A a feature that increases engagement, A's posts affect B's feed — B benefits even in control.
  3. **Shared inventory**: ads auctions. If treatment bidders bid more aggressively, control bidders face higher competition → control outcomes degrade → experiment overestimates treatment effect.
  4. **Shared infrastructure**: if treatment uses more cache/DB resources, control experiences degraded performance.
- **Fixes**:
  1. **Cluster randomization**: randomize at a higher level (city, country, social community) rather than user. Units within a cluster interact but clusters don't. Reduces power (fewer independent units).
  2. **Switchback experiments**: alternate treatment and control in the same unit over time (e.g., 30-min windows). Temporal interference exists but can be modeled. Used at Uber/Lyft for marketplace experiments.
  3. **Graph-based cluster randomization**: partition the social graph into weakly connected communities; randomize at community level. Used at Meta/LinkedIn.
  4. **Two-sided experiment design**: treat supply and demand sides independently. Used when supply/demand interference is the primary concern.
  5. **Ego network experiments**: for social features, randomize the entire ego network (user + friends) to the same condition.

**Company context:** Uber Supply Pricing (switchback is the standard), Meta (graph cluster randomization), Netflix ads (shared inventory effects in auction).

**Common wrong answer:** "I'd run a standard user-level A/B test." — At Staff bar, you should recognize marketplace and social network scenarios where this is invalid, and name the correct alternative.

---

## Novelty Effect

### Q: How do you detect and handle the novelty effect in A/B tests?

**Answer (Staff level):**
- **Novelty effect**: users interact with a new feature more (or less) than they will in steady state simply because it's new. A new button gets more clicks at launch; a changed layout causes temporary confusion. Inflates (or deflates) early metric estimates.
- **Detection**: plot the treatment effect over time. If effect is large early and decays over days/weeks, novelty effect is present. Stable effect = no novelty issue.
- **Handling**:
  1. **Run longer**: novelty effects typically decay within 1–2 weeks. Sufficient experiment runtime is the simplest fix.
  2. **New user cohort analysis**: new users have no prior experience → no novelty effect. Compare treatment effect for users who joined during the experiment vs. existing users. If new users show the same effect, it's real (not novelty).
  3. **Holdout analysis**: keep 5–10% of users in a long-term holdout (never exposed to new feature). Compare holdout vs. fully launched users at 30/90/180 days. Long-term holdouts are the gold standard for measuring true steady-state effect.
  4. **Time-based segmentation**: compute treatment effect for days 1–3 vs. days 8–14. If day 8–14 effect is significantly smaller, flag as novelty-driven.
- **Opposite: primacy effect**: the new experience is worse at first (learning curve) but improves over time. Users in treatment appear to underperform in short experiments. Same techniques apply.

**Company context:** Netflix (long-term holdouts are documented practice at Netflix), Meta (novelty effects common in feed ranking changes).

**Common wrong answer:** "I'd just wait until significance." — Waiting for significance doesn't fix novelty effect; it amplifies it if early data has high novelty signal. The fix is analyzing effect over time, not accumulating more of the same biased data.

---

## Long-Term Holdouts

### Q: What is a long-term holdout and when do you use it?

**Answer (Staff level):**
- **Definition**: a permanently withheld 1–5% of users who never receive a new feature. Used to measure the cumulative long-term effect of shipped features without a formal experiment.
- **Why holdouts matter**: short experiments measure 7-day effects. Users adapt over months. A recommendation change may improve CTR in week 1 but reduce library diversity and increase churn in month 6. Short experiments miss this.
- **How it works**:
  1. At feature launch, withhold 1–5% of users from the new experience.
  2. After 30/90/180 days, compare holdout (old experience) vs. treatment (new experience) on long-term metrics: retention rate, LTV, subscriber churn.
  3. The holdout group has never seen the new feature, so the comparison is clean.
- **Challenges**:
  - Holdout population drifts over time (new users enter the holdout; older users churn). Must control for cohort effects.
  - 1% holdout on 300M Netflix subscribers = 3M users — statistically powerful enough for even small effects.
  - Ethics: withholding a product improvement from 1–5% of users permanently. Netflix justifies this as necessary for measurement integrity.
- **What long-term holdouts catch that short experiments miss**:
  - Churn caused by engagement farming (short-term engagement up, long-term satisfaction down)
  - Catalog diversity degradation (recommendation model converges on popular content over time)
  - Advertiser ecosystem effects (bidding algorithm changes that affect advertiser ROI only visible in 90-day ROAS)

**Company context:** Netflix (explicitly a documented Netflix practice; their culture deck mentions measurement integrity). Critical for the Netflix ads role — ROAS measurement requires 30–90 day attribution windows.

**Common wrong answer:** "I'd use a 7-day post-launch observation period." — That's a monitoring, not a holdout. A holdout requires a clean control group that was never exposed. 7-day observation confounds the new feature with time-of-year effects.

---

## Interleaving for Ranking Evaluation

### Q: What is interleaving and when is it preferred over standard A/B for ranking models?

**Answer (Staff level):**
- **Standard A/B for ranking**: show user A ranking from Model 1, show user B ranking from Model 2. Compare clicks/engagement. Problem: high variance — users differ in base engagement, so detecting small ranking improvements requires large samples.
- **Interleaving**: for each user, merge Model 1 and Model 2 rankings into a single ranked list. Track which model's items get clicked more.
  - **Team-draft interleaving**: alternate between Model 1 and Model 2 "picking" items for the merged list. The model whose items get more clicks "wins" the comparison.
  - **Probabilistic interleaving**: assign each position to Model 1 or Model 2 with a probability, then track click attribution.
- **Why interleaving is more sensitive**: each user sees items from both models simultaneously, so user preference differences cancel out. This gives 10–100× more statistical power for the same number of users. You can detect 0.1% ranking improvements in days vs. weeks.
- **Limitations**:
  - Measures relative preference ("which ranking is better?") not absolute improvement ("by how much?").
  - Can't measure metrics that aren't directly tied to the ranked items (e.g., long-term churn, session depth).
  - Position bias: if both models agree on position 1, the item there will get more clicks regardless.
- **When to use**: rapid iteration during model development (offline → interleaving → full A/B). Interleaving is a cheap filter; full A/B is the final measurement.

**Company context:** Netflix (extensively documented in their tech blog for recommendation ranking), Meta (used in ads ranking), Reddit (content ranking).

**Common wrong answer:** "I'd A/B test all ranking changes." — For ranking, interleaving is the standard first gate because it's dramatically more efficient. Staff answer knows when to use interleaving as a pre-filter before committing to a full A/B.

---

## A/B Testing in Ads (Conversion Delay and Attribution)

### Q: What makes A/B testing for ads bidding algorithms harder than standard product experiments?

**Answer (Staff level):**
- **Problem 1 — Conversion delay**: a click today converts in 7–30 days (for subscription, enterprise sales, or considered purchases). Standard A/B analysis uses short windows → sees clicks, not conversions → optimizes the wrong metric.
  - Fix: use a longer measurement window (14–30 days post-click). Means experiment runtime = treatment period + attribution window. A 7-day test with a 30-day attribution window takes 37 days total.
  - Fix: survival modeling on conversion time. Estimate P(convert | time elapsed) to correct for truncated observations in the attribution window.
- **Problem 2 — Shared auction**: if treatment bidder increases bids, control bidders face higher competition → they win fewer auctions → worse outcomes even in control. Standard A/B underestimates treatment effect (or produces a negative control bias).
  - Fix: holdout at advertiser level, not user level. Advertisers are randomly assigned to new vs. old bidding algorithm. Ad impressions are served from the respective algorithm's bids.
- **Problem 3 — Budget constraints**: a bidding algorithm that bids more aggressively depletes budget faster. The experiment ends early for treatment advertisers → less data, lower statistical power.
  - Fix: normalize outcomes per dollar spent, not per impression. Compare `conversions / $ spent` not `total conversions`.
- **Problem 4 — Advertiser heterogeneity**: small advertisers (< $100/day budget) behave very differently from large advertisers ($10K/day). Pooling them in one experiment conflates effects.
  - Fix: stratified randomization by advertiser size tier. Analyze per stratum and aggregate with appropriate weights.
- **Offline evaluation**: before running a live experiment, evaluate with replay/counterfactual simulation: take historical bid logs, apply the new bidding algorithm's bids, estimate outcomes via off-policy evaluation (doubly robust estimator). Cheaper than live A/B; provides directional signal.

**Company context:** Netflix MLS5 Ads (explicitly: "online and offline evaluation frameworks"), Reddit (ads auction), Meta, Pinterest.

**Common wrong answer:** "I'd run a 7-day A/B test with conversion as the metric." — Truncated attribution window → wrong metric. Shared auction → control contamination. Staff answer addresses both.

---

## Experiment Design Fundamentals (Deep Dive)

### The Power Calculation — Intuition Behind Each Term

```
n = (z_α/2 + z_β)² × 2σ² / δ²

z_α/2 = 1.96:  how far out on the null distribution to set the rejection threshold
z_β = 0.84:    how far the treatment distribution must be to have 80% power
σ²:            variance of your metric per observation
δ:             minimum detectable effect (MDE) — smallest lift that matters
```

The `δ²` in the denominator is the critical relationship: **halving the MDE quadruples the required sample size**. This is why "detect a 0.1% improvement" is 100× more expensive than "detect a 1% improvement."

```
Conversion rate experiment, p=0.05 (5% baseline), δ=0.005 (0.5% absolute lift):
  σ² ≈ p(1-p) = 0.05 × 0.95 = 0.0475
  n = (1.96+0.84)² × 2×0.0475 / (0.005)² = 7.84 × 0.095 / 0.000025 ≈ 29,800 per arm

Same experiment, δ=0.001 (0.1% absolute lift):
  n = 7.84 × 0.095 / (0.001)² ≈ 745,000 per arm  ← 25× more users
```

**The MDE is a business input, not a statistical choice.** "What's the smallest effect we'd actually ship?" — below that effect size, the product change isn't worth the engineering cost regardless of significance.

### Sample Ratio Mismatch (SRM) — Why It Invalidates Everything

If you assign 50/50 but observe 55/45, the experiment is broken:

```
Root causes:
  - Bot/crawler traffic filtered differently per variant
  - Redirect delays causing users to drop out of treatment
  - Feature flags applied incorrectly (some treatment users get control experience)
  - Logging bugs (treatment events logged at lower rate)
```

SRM test: Chi-squared test on the observed vs. expected assignment counts. p < 0.001 → SRM detected → invalidate and debug before interpreting any metric.

**Why SRM invalidates everything**: if assignment probabilities differ, the randomization guarantee breaks. You no longer have a clean causal comparison. Even if the primary metric looks good, you can't trust it.

### Guardrail Metrics — Why You Need Them Pre-Committed

```
Experiment: new recommendation algorithm
Primary: CTR ↑ 3% (significant, exciting)
Guardrail check: session_depth ↓ 8% (users get what they want faster, then leave)
                 p99_latency ↑ 40% (algorithm is slower)

Without pre-committed guardrails: team ships because "CTR improved"
With pre-committed guardrails: experiment fails on session_depth → debug
```

Post-hoc guardrail selection is rationalization. If guardrails are defined after results are seen, they'll be chosen to support the preferred conclusion.

---

## CUPED (Deep Dive)

### Why It Works — The Variance Decomposition

Each user's outcome Y has two components:

```
Y = baseline_behavior + treatment_effect + noise

baseline_behavior: how this user always behaves (their "type")
treatment_effect:  the causal impact of the treatment
noise:             random variation this session
```

Standard A/B: both treatment and control groups have high baseline variance (some users are always heavy engagers, some are always light). This baseline variance makes it hard to detect the treatment effect.

CUPED removes the baseline component:

```
Y_cuped = Y - θ·X   where X = same metric in pre-experiment period

Var(Y_cuped) = Var(Y) + θ²·Var(X) - 2θ·Cov(Y,X)
Minimized at θ* = Cov(Y,X)/Var(X)  →  Var(Y_cuped) = Var(Y)·(1 - ρ²)
```

`ρ` = correlation between pre-experiment and experiment-period metrics. Typical values:

```
Daily active users:   ρ ≈ 0.8 → Var reduction = 1 - 0.64 = 36%
Weekly watch time:    ρ ≈ 0.7 → Var reduction = 1 - 0.49 = 51%
Rare events (churn):  ρ ≈ 0.2 → Var reduction = 1 - 0.04 = 4%  ← barely helps
```

High-frequency stable metrics benefit most. Rare, bursty events have low autocorrelation — CUPED doesn't help.

### CUPAC — ML-Based Extension

CUPED uses a single pre-experiment metric as the covariate. CUPAC replaces it with an ML model's prediction:

```
CUPED:  Y_cuped = Y - θ·X_pre
CUPAC:  Y_cuped = Y - θ·f(X_pre, X_demographics, X_device, ...)
```

`f` is a gradient boosted model trained to predict Y from all available pre-experiment features. If `f` explains 80% of variance in Y (vs. 50% for CUPED), variance reduction is 80% vs. 50%.

**Used at**: Airbnb (reduced experiment runtime by 50%), LinkedIn, Netflix.

---

## Multiple Testing (Deep Dive)

### The 64% False Positive Problem — Concrete

20 metrics tested at α=0.05:

```
P(at least one false positive) = 1 - P(no false positives)
                                = 1 - (1 - 0.05)^20
                                = 1 - 0.95^20
                                = 1 - 0.358
                                = 0.642  ← 64% chance of a spurious finding
```

Every time you add a metric to the dashboard, you're increasing this probability. 50 metrics → `1 - 0.95^50 = 92%` false positive rate.

### Bonferroni vs. FDR — When Each Applies

**Bonferroni** controls FWER (probability of ANY false positive):
```
α_adjusted = 0.05 / 20 = 0.0025 per test
Each metric needs p < 0.0025 to be declared significant
```

Very conservative — assumes tests are independent. For product metrics (CTR, session_depth, latency all tend to move together), this is too strict. You'll miss real effects.

**Benjamini-Hochberg** controls FDR (expected fraction of discoveries that are false):
```
Sort p-values: p₁ ≤ p₂ ≤ ... ≤ pₘ
Reject H₀ for all i where pᵢ ≤ (i/m)·α

Example (5 metrics, α=0.05):
  p₁=0.001 ≤ (1/5)×0.05=0.010  ✓ significant
  p₂=0.013 ≤ (2/5)×0.05=0.020  ✓ significant
  p₃=0.031 ≤ (3/5)×0.05=0.030  ✗ not significant (stops here)
```

More liberal than Bonferroni, appropriate when tests are correlated (product metrics).

**Staff-level answer**: neither correction is the real fix. The fix is **one primary metric, decided before the experiment**. Corrections are band-aids for poor experiment design.

---

## SUTVA Violations (Deep Dive)

### The Marketplace Interference Mechanism

In a two-sided marketplace (Uber, Airbnb), user outcomes depend on the entire system state:

```
Treatment: new matching algorithm → treatment drivers get 20% more rides
Effect on control: same driver pool, but treatment drivers are more efficient
                   → they take rides that would have gone to control drivers
                   → control drivers get fewer rides

Result:
  Treatment: +20% rides
  Control:   -5% rides  ← harmed by treatment
  Naive ATE estimate: +20% - (-5%) = +25%  ← overestimates true effect

True incremental effect: what would happen if ALL drivers were on treatment?
  Answer is somewhere between 0% and 25%, not 25%.
```

### Switchback Experiments — How They Work

Used at Uber, Lyft. Time is the randomization unit instead of user:

```
Time windows (e.g. 30-minute slots):
  8:00-8:30:  Control   (all users see control algorithm)
  8:30-9:00:  Treatment (all users see treatment algorithm)
  9:00-9:30:  Control
  9:30-10:00: Treatment
  ...
```

At any given time, ALL users are in the same condition → no within-period contamination. The interference is temporal (state from control period affects treatment period), but this can be modeled with time-series methods.

**Why this works for marketplaces**: during a treatment window, the marketplace dynamics are entirely governed by the treatment algorithm. No mixing of treatment and control supply/demand.

**Tradeoff**: requires assuming temporal interference decays quickly (no long-term carryover between windows). If a treatment window causes driver positioning changes that persist into the next control window, estimates are biased.

---

## Novelty Effect (Deep Dive)

### How to Detect It — The Time-Sliced Effect Plot

Plot the treatment effect (treatment − control) as a function of day-in-experiment:

```
Day 1:  +8% CTR
Day 2:  +7%
Day 3:  +6%
Day 7:  +4%
Day 10: +3%
Day 14: +2.5%  ← stabilizing
Day 21: +2.5%  ← stable

Conclusion: real effect is ~2.5%, not 8%. First 3 days inflated by novelty.
```

Versus a real effect:
```
Day 1:  +3% CTR
Day 7:  +3%
Day 14: +3%  ← flat line = no novelty effect
```

### New User Cohort Analysis — The Cleanest Test

New users who joined during the experiment have no prior experience with the old product:

```
Existing users in treatment: they notice the change → novelty response
New users in treatment:       they have no baseline → no novelty response

If new users show same effect as existing users → effect is real, not novelty
If new users show smaller effect → novelty effect confirmed
```

This is why new user cohort analysis is the most rigorous novelty test — it directly compares users with and without prior expectations.

---

## Interleaving (Deep Dive)

### Team-Draft Interleaving — The Mechanics

```
Model A ranking:  [item₁, item₂, item₃, item₄, item₅]
Model B ranking:  [item₃, item₁, item₅, item₂, item₄]

Team-draft process:
  Round 1 (A picks first): A picks item₁ → merged list: [item₁]
  Round 1 (B picks):       B picks item₃ → merged list: [item₁, item₃]
  Round 2 (A picks):       A picks item₂ → merged list: [item₁, item₃, item₂]
  Round 2 (B picks):       B picks item₅ → merged list: [item₁, item₃, item₂, item₅]
  ...

User sees merged list: [item₁, item₃, item₂, item₅, item₄]
User clicks item₃ → Model B gets credit (B originally ranked it higher)
User clicks item₁ → Model A gets credit
```

After many queries: A wins if its items get more clicks in the merged list.

### Why It's 10–100× More Sensitive

Standard A/B: user A sees Model A, user B sees Model B. The comparison is confounded by user differences (user A and B have different engagement patterns). You need many users to wash out user variance.

Interleaving: the SAME user sees items from both models in the SAME session. User variance cancels exactly. The only difference is which model's item the user preferred.

```
Variance of A/B treatment effect estimate: σ²_user + σ²_treatment
Variance of interleaving effect estimate:  σ²_treatment only

σ²_user is typically 10-100× σ²_treatment for engagement metrics
→ interleaving is 10-100× more efficient
```

**The limitation**: interleaving only measures preference ("which ranking is better?"), not magnitude ("by how much does CTR improve?"). Use interleaving to filter quickly, then full A/B for magnitude measurement before shipping.

---

## A/B Testing in Ads (Deep Dive)

### The Attribution Window Problem — Mechanics

```
Timeline:
  Day 0:   user clicks ad
  Day 0-7: user considers purchase
  Day 7:   user converts (buys)

Experiment runtime: 7 days
Measurement at day 7: Day 7 conversions only captured for users who clicked on day 0

Users who clicked on day 6: their conversions haven't happened yet
→ treatment group's conversion rate is systematically underestimated
→ the underestimation is WORSE for treatment if treatment drove more late-session clicks
```

The fix:

```
Experiment runtime: 7 days
Attribution window: 30 days
Total wait: 37 days before analysis

During the 7-day experiment: record which users were assigned to which arm
During the 30-day window: wait for conversions to roll in
At day 37: compute conversions per arm using full 30-day attribution
```

**Survival model alternative**: instead of waiting 30 days, model the conversion time distribution and estimate P(convert | time elapsed, treatment) using a survival model. Provides early estimates with uncertainty bounds.

### The Shared Auction Problem — Who Gets Harmed

```
Advertiser A (treatment): new bidding algo bids aggressively
Advertiser B (control):   old bidding algo

In each auction:
  If A and B compete for the same impression:
    A bids higher → A wins more often
    B now faces a more competitive auction → B's win rate drops

Naive ATE:
  Treatment: +15% ROAS
  Control:   -8% ROAS  ← harmed by treatment
  ATE = +23%  ← overestimates true incremental value

True effect: what would happen if ALL advertisers used the new algo?
  No control group to contaminate → true effect is probably ~10-15%, not 23%
```

**Fix — advertiser-level holdout**:

```
Assign advertisers (not users) to treatment/control.
All of advertiser A's impressions use new algo.
All of advertiser B's impressions use old algo.

Advertisers still compete in the same auctions, but:
  - Each advertiser's own performance is internally consistent
  - You measure "does advertiser A do better under new algo vs. advertiser B under old algo?"
  - Contamination exists but is symmetric and estimable
```
