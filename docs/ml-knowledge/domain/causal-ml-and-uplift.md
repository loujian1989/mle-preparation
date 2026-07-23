# Causal ML & Uplift Modeling — ML Knowledge Q&A

P0: Universal. Directly relevant to: whole-page optimization (Netflix, Reddit), incentive
design (Uber Supply Pricing, Whatnot), blending ads and organic (Pinterest, Reddit),
experiment design (all companies), promotion targeting (Meta, Netflix, Shopify).

---

## Potential Outcomes Framework

### Q: What is the potential outcomes (Rubin causal) framework, and why does it matter for ML?

**Answer (Staff level):**
- **Core idea**: for each unit i, there exist two potential outcomes: `Y_i(1)` (outcome if treated) and `Y_i(0)` (outcome if untreated). The **individual treatment effect** is `τ_i = Y_i(1) - Y_i(0)`.
- **The fundamental problem of causal inference**: you can only observe one potential outcome per unit — the one corresponding to the treatment they actually received. The other is the **counterfactual** and is never observed.
- **Why this matters for ML**: standard ML models learn `E[Y | X, T]` — the expected outcome given features and treatment. This is a **predictive** model. But `E[Y | X, T=1] - E[Y | X, T=0]` is NOT the causal effect unless treatment assignment T is independent of X (i.e., randomized). In observational data, T is almost never independent of X.
- **Average Treatment Effect (ATE)**: `E[Y(1) - Y(0)] = E[Y(1)] - E[Y(0)]`. Estimates the effect on the whole population. Identified under randomization or unconfoundedness.
- **Average Treatment Effect on the Treated (ATT)**: `E[Y(1) - Y(0) | T=1]`. Effect on those who received treatment. More relevant for policy questions like "did our promotion help the people we gave it to?"
- **Conditional ATE (CATE)**: `τ(x) = E[Y(1) - Y(0) | X=x]`. Heterogeneous treatment effects — the effect varies by individual features. This is the target of uplift modeling.
- **Identification assumptions**:
  1. **Unconfoundedness**: `T ⊥ (Y(0), Y(1)) | X` — no unmeasured confounders given observed features X.
  2. **Overlap (positivity)**: `0 < P(T=1|X) < 1` — every unit has some probability of being in either treatment or control. Violations cause extrapolation failures.
  3. **SUTVA**: no interference between units. Violated in marketplace/social settings.

**Company context:** Universal prerequisite. Uber (strategic driver response modeling), Netflix (incrementality measurement), Reddit (organic vs. ad blending), Meta (ads measurement).

**Common wrong answer:** "I'd train a model on treated vs. control outcomes and subtract them." — This is S-learner, which conflates selection bias with treatment effect. Without controlling for confounders, observed `E[Y|T=1] - E[Y|T=0]` is not the causal effect.

---

## Uplift Modeling

### Q: What is uplift modeling and how does it differ from standard propensity or response modeling?

**Answer (Staff level):**
- **Standard response model**: `P(convert | user features)`. Predicts who converts — but this includes users who would convert anyway (without treatment). Targeting high-responders wastes budget on "sure things."
- **Propensity model**: `P(T=1 | user features)`. Predicts who received treatment — used for inverse propensity weighting (IPW), not for targeting.
- **Uplift model**: estimates `τ(x) = P(convert | T=1, X=x) - P(convert | T=0, X=x)`. Predicts the **incremental** effect of treatment. Target users with high uplift, not high response.
- **Four segments (the persuasion matrix)**:

  | | Would respond without treatment | Would NOT respond without treatment |
  |---|---|---|
  | **Responds with treatment** | **Sure Things** (wasted spend) | **Persuadables** (target these) |
  | **Does NOT respond with treatment** | **Lost Causes** | **Sleeping Dogs** (treatment hurts) |

  Uplift targets Persuadables. Response model conflates Persuadables + Sure Things.

- **Estimators**:

  **T-Learner**: train two separate models — `μ_1(x) = E[Y|T=1, X=x]` and `μ_0(x) = E[Y|T=0, X=x]`. Estimate uplift as `μ_1(x) - μ_0(x)`. Simple, but each model sees only half the data; doesn't share information across treatment arms.

  **S-Learner**: train one model `μ(x, t) = E[Y|X=x, T=t]`. Estimate uplift as `μ(x, 1) - μ(x, 0)`. Can under-fit — treatment indicator may be regularized away if it has weak marginal signal.

  **X-Learner**: estimate treatment effects using cross-predictions. Particularly effective when treatment and control groups have very different sizes.
  1. Train `μ_1` on treated, `μ_0` on control.
  2. Impute counterfactuals: `D_i^1 = Y_i - μ_0(X_i)` for treated; `D_i^0 = μ_1(X_i) - Y_i` for control.
  3. Train uplift models `τ_1(x)` on `D^1` and `τ_0(x)` on `D^0`.
  4. Combine: `τ(x) = g(x) × τ_1(x) + (1-g(x)) × τ_0(x)` where `g(x) = P(T=1|X=x)`.

  **Causal Forest (Wager & Athey)**: random forest variant where splits maximize heterogeneity in treatment effects (not prediction accuracy). Gold standard for CATE estimation with theoretical guarantees. Available in `econml` and `grf` libraries.

- **Evaluation challenge**: you can't directly evaluate uplift models because you never observe both `Y(1)` and `Y(0)` for the same unit. Use:
  - **Qini curve / uplift curve**: rank users by predicted uplift; compute cumulative incremental conversions by percentile. Compare to random targeting.
  - **AUUC (Area Under Uplift Curve)**: higher = better targeting of persuadables.
  - **Held-out A/B test on model segments**: apply model in a holdout; compare conversion rates of top-uplift vs. bottom-uplift groups in treatment and control.

**Company context:** Uber (incentive targeting — who needs a quest to come online? not everyone), Netflix (promotion targeting), Meta (ads measurement), Shopify (merchant incentive programs).

**Common wrong answer:** "I'd train a model on who converts and target those users." — Response model, not uplift model. This wastes spend on sure things and may even harm sleeping dogs. The incremental effect is the target, not the absolute response rate.

---

## Opportunity Cost in Whole-Page Optimization

### Q: How do you estimate the opportunity cost of showing an ad instead of an organic result on a page?

**Answer (Staff level):**
- **The problem**: a webpage (search results, content feed, product listing) has N slots. Each slot can show an organic item or a paid ad. Showing an ad in slot k displaces the organic item that would have been there — this is the opportunity cost.
- **Why it matters**: naive revenue maximization fills every slot with the highest-paying ad. But displacing high-quality organic results degrades user experience → session depth falls → long-term DAU/retention declines. This is the core tension in whole-page optimization.
- **Opportunity cost estimation**:
  1. **Counterfactual engagement**: what would the user have done if slot k showed the organic item? Estimate engagement value of the displaced organic item using a relevance model: `OC(k) = E[engagement | organic_item_k, user, context]`.
  2. **Revenue equivalence**: convert organic engagement to a dollar value via a reference metric. If 1 organic click has the same long-term value as $X in ad revenue (estimated from holdout), then OC(k) = X × P(organic_click | k).
  3. **Effective eCPM threshold**: only place an ad in slot k if `ad_revenue(k) > OC(k)`. This sets a dynamic floor price per slot based on organic quality.
- **The blending objective**:
  ```
  maximize  Σ_k [ad_revenue(k) × I(ad_k) + organic_value(k) × I(organic_k)]
  subject to Σ_k I(ad_k) ≤ max_ads_per_page
             long_term_engagement ≥ guardrail_threshold
  ```
- **Staff framing**: the optimization isn't "should we show an ad?" but "what's the minimum bid at which the ad creates more value than the organic result it displaces?" This is a reserve price that varies per slot based on organic quality.
- **Measurement**: use long-term holdout (no ads in holdout group) to estimate true organic engagement lift. Compare session depth, return rate, and LTV between ad-exposed and holdout groups. This is how you calibrate the organic value coefficient.

**Company context:** Reddit (content feed blends organic posts with promoted posts), Pinterest (organic pins vs. promoted pins), Netflix (content recommendations vs. ads in homepage), Meta (organic feed vs. sponsored posts).

**Common wrong answer:** "I'd cap ads at 20% of slots to protect user experience." — Arbitrary caps don't account for variation in organic quality by slot. High-quality organic slot 1 should have a higher opportunity cost than low-quality slot 10. Opportunity cost is position- and content-dependent.

---

## Instrumental Variables

### Q: When can't you use a standard regression to estimate causal effects, and how do instrumental variables help?

**Answer (Staff level):**
- **The endogeneity problem**: when the treatment T is correlated with unobserved confounders U that also affect the outcome Y. OLS on `Y ~ T + X` gives a biased estimate of the causal effect of T.
  - Example: surge pricing. High surge correlates with high demand. Regressing trips on surge gives a negative coefficient (fewer trips at high surge). But this conflates the price effect (raises price → fewer trips) with the demand shock (high demand causes both high trips AND high surge). The true causal effect of price is smaller in magnitude.
- **Instrumental variable (IV)**: a variable Z that:
  1. **Relevance**: Z is correlated with T (`Cov(Z, T) ≠ 0`).
  2. **Exclusion restriction**: Z affects Y only through T, not directly (`Cov(Z, Y|T) = 0`).
  3. **Independence**: Z is independent of unobserved confounders U.
- **IV estimation (2SLS)**:
  - Stage 1: regress T on Z and X. Get predicted treatment `T̂ = α₀ + α₁Z + α₂X`.
  - Stage 2: regress Y on `T̂` and X. The coefficient on `T̂` is the IV estimate of the causal effect.
- **Example applications**:
  - **Surge pricing**: use weather events as IV for surge. Weather causes surge (rain → fewer drivers → higher surge) but doesn't directly affect trip demand (same number of people need rides regardless of rain). Identifies causal effect of surge on trip completion.
  - **Driver incentives**: use randomized quest assignment as IV for driver hours. Quest assignment is randomized → exogenous; it affects hours worked but doesn't directly affect trip quality.
  - **Ad exposure**: use ad slot position as IV for ad clicks. Being in slot 1 causes more clicks (purely mechanical) but doesn't directly cause conversion (beyond the click). Identifies causal effect of click on conversion.
- **Limitations**: IV only estimates the Local Average Treatment Effect (LATE) — the effect on "compliers" (those whose treatment changes due to the instrument). Can't generalize to non-compliers. IV estimator has higher variance than OLS, especially when the instrument is weak.

**Company context:** Uber Supply Pricing (pricing counterfactuals), Reddit (ads pricing), Netflix (ad incrementality). Identifying valid instruments is hard — this is a Staff-level answer.

**Common wrong answer:** "I'd control for confounders with a richer feature set." — Controlling for observables doesn't fix unobserved confounders. IV is specifically for the unobservable case. Staff answer knows when to reach for IV vs. when rich features suffice.

---

## Double ML (Partially Linear Model)

### Q: What is Double ML and when is it better than a standard regression approach for causal estimation?

**Answer (Staff level):**
- **Problem with naive regression**: including many controls X in `Y ~ T + X` biases the treatment effect estimate when X is high-dimensional (regularization biases the T coefficient) or when X and T have complex non-linear relationships.
- **Double ML (Chernozhukov et al.)**: a two-step approach that uses ML models for nuisance estimation while remaining valid for causal inference:
  1. **Step 1 — Partial out the treatment**: fit `T̂ = E[T|X]` using any ML model (gradient boosting, neural net). Compute residual `Ṽ = T - T̂`.
  2. **Step 2 — Partial out the outcome**: fit `Ŷ = E[Y|X]` using any ML model. Compute residual `Ũ = Y - Ŷ`.
  3. **Causal estimate**: regress `Ũ` on `Ṽ`. The coefficient is the causal effect estimate: `θ = Cov(Ũ, Ṽ) / Var(Ṽ)`.
- **Why this works**: by partialling out X from both Y and T, you remove confounding. The residual variation in T is (approximately) exogenous given X. The final regression is unbiased even when the ML nuisance models are complex.
- **Cross-fitting**: to avoid overfitting bias, use K-fold cross-fitting — train nuisance models on fold k, predict on held-out fold k. Prevents the nuisance models from memorizing and creating artificial residuals.
- **When to use**:
  - High-dimensional X where regularization would otherwise bias the T coefficient
  - Non-linear relationship between X and T (treatment selection) or X and Y (outcome)
  - You have a large dataset where ML nuisance models can be fit reliably
- **Limitation**: requires unconfoundedness (no unmeasured confounders). Double ML is not magic — it estimates the causal effect efficiently under unconfoundedness but doesn't fix unobserved confounding.

**Company context:** Netflix (ad incrementality with rich viewing history as controls), Meta (ads measurement with thousands of user features as controls), Uber (pricing effects controlling for weather, events, city characteristics).

**Common wrong answer:** "I'd add all features to a linear regression." — In high dimensions, regularization biases the treatment coefficient. Double ML uses ML for nuisance estimation while preserving valid causal inference for the treatment effect.

---

## Doubly Robust Estimator

### Q: What is the doubly robust (AIPW) estimator and why is it preferred for treatment effect estimation?

**Answer (Staff level):**
- **Setup**: estimating ATE from observational data with propensity score `e(x) = P(T=1|X=x)` and outcome model `μ_t(x) = E[Y|T=t, X=x]`.
- **IPW estimator**: `ATE_IPW = E[T×Y/e(x) - (1-T)×Y/(1-e(x))]`. Consistent if propensity model is correctly specified. Fails if propensity model is wrong.
- **Direct estimator**: `ATE_DM = E[μ_1(x) - μ_0(x)]`. Consistent if outcome model is correctly specified. Fails if outcome model is wrong.
- **Doubly Robust / AIPW (Augmented IPW)**:
  ```
  ATE_DR = E[μ_1(x) - μ_0(x)
            + T(Y - μ_1(x)) / e(x)
            - (1-T)(Y - μ_0(x)) / (1-e(x))]
  ```
  **Key property**: consistent if EITHER the propensity model OR the outcome model is correctly specified (not both need to be right — hence "doubly robust").
- **Why this matters in practice**: you're never sure which model is well-specified. DR provides insurance — if your propensity model is off but your outcome model is good (or vice versa), you still get a consistent estimate.
- **Application in off-policy evaluation**: replace ATE with `E[Y(π_new)]` — expected outcome under a new policy. AIPW-based OPE is the standard for evaluating new bidding algorithms, pricing policies, or recommendation changes without running live experiments.
- **Semiparametric efficiency**: DR achieves the semiparametric efficiency bound — it extracts as much information as possible from the data given the model class.

**Company context:** Netflix (OPE for bidding evaluation), Uber (pricing counterfactual estimation), Reddit (ads ranking policy evaluation). The `doubleml` and `econml` Python libraries implement this.

**Common wrong answer:** "I'd use importance sampling for off-policy evaluation." — Standard IS is the IPW estimator — high variance, not doubly robust. AIPW has lower variance and the doubly robust property. Staff answer knows the distinction.

---

## Incrementality and Ads-Organic Blending

### Q: How do you measure the true incremental value of an ad when users may have converted organically anyway?

**Answer (Staff level):**
- **The problem**: an advertiser runs ads. Users see ads and convert. Advertiser sees: 1,000 conversions from 10,000 ad exposures → 10% CVR. But: some of those 1,000 users would have searched for and purchased the product organically anyway. The ad didn't cause all 1,000 conversions — it only caused the incremental ones.
- **Incrementality**: `incremental_conversions = conversions_with_ads - conversions_without_ads`. The counterfactual (conversions without ads) is never directly observed.
- **Incrementality testing (geo holdout)**:
  - Randomly select DMAs (Designated Market Areas / geographic regions) as holdout — no ads served there.
  - Compare conversion rate in ad-exposed DMAs vs. holdout DMAs.
  - `iROAS = (revenue_in_treatment - revenue_in_control) / ad_spend_in_treatment`
  - Controls for organic conversion rate via the holdout baseline.
- **Ghost ads**: instead of withholding ads entirely, serve an ad for a different brand (same format, same targeting) to the holdout group. This controls for the "awareness that an ad slot exists" effect. User sees an ad either way; only the brand changes.
- **Organic cannibalization**: a real phenomenon — ads placed near organic results for the same brand may displace organic clicks with paid clicks (user clicks ad instead of organic result below it). The ad drives no incremental user, but advertiser pays for the click. Detect by looking at organic click rate in geographic areas with vs. without ads for the same brand.
- **Platform-level incrementality**: Netflix's question — "did showing an ad to a member cause them to subscribe at higher rates than non-exposed members (controlling for self-selection)?" This requires geo holdout or matched control group (propensity score matching on member demographics).

**Company context:** Netflix (MLS5 Ads role explicitly: measuring ad effectiveness for subscription conversions), Reddit (promoted content incrementality), Meta (ads measurement), Pinterest.

**Common wrong answer:** "I'd measure ROAS as attributed conversions / ad spend." — Attribution-based ROAS overcounts conversions that would have happened organically. Incrementality is the correct measure, and it requires a counterfactual control group, not just attribution.

---

## Heterogeneous Treatment Effects for Incentive Targeting

### Q: You want to personalize a driver incentive (quest bonus). How do you identify who to target and with what threshold?

**Answer (Staff level):**
- **Goal**: maximize incremental driver hours from the incentive program, subject to a budget constraint. Don't give incentives to drivers who'd work the same hours anyway.
- **Step 1 — Estimate CATE**: for each driver, estimate `τ_i = E[hours(1) - hours(0) | driver features]`. Use causal forest or X-learner on historical randomized incentive data.
- **Step 2 — Targeting rule**: under budget constraint B and cost-per-driver c:
  ```
  target drivers where τ_i ≥ threshold
  threshold = solve for: Σ_{i: τ_i ≥ threshold} c ≤ B
  ```
  This ensures you maximize incremental hours per dollar.
- **Step 3 — Personalize threshold**: within targeted drivers, set quest threshold = `baseline_hours_i + target_increment`. Incentive compatible if increment is achievable but requires genuine effort.
- **Step 4 — Sleeping dog detection**: segment where treatment effect `τ_i < 0` — these drivers work fewer hours when given a quest (possibly because the target feels unachievable, demotivating them). Exclude from treatment.
- **Evaluation**: run A/B test on model segments:
  - High predicted uplift group (treatment vs. control): large positive effect expected
  - Low predicted uplift group (treatment vs. control): near-zero effect expected
  - "Sleeping dog" group (treatment vs. control): negative effect expected
  Verifying these patterns validates the uplift model calibration.

**Company context:** Uber Supply Pricing (driver quests), Whatnot (seller incentives), Netflix (member promotion targeting), Meta (advertiser incentive programs).

**Common wrong answer:** "I'd give incentives to the most active drivers." — Those are sure things (high response, zero uplift). The correct answer targets persuadables — drivers who are on the margin and would respond incrementally to the incentive.

---

## Potential Outcomes (Deep Dive)

### The Fundamental Problem — Made Concrete

```
User A receives a promotional email (T=1). They renew their subscription. Y(1)=1.
Question: would user A have renewed without the email? Y(0) = ???

User B does NOT receive the email (T=0). They don't renew. Y(0)=0.
Question: would user B have renewed with the email? Y(1) = ???

We observe:
  User A: Y(1)=1, Y(0)=?  → only one potential outcome
  User B: Y(0)=0, Y(1)=?  → only one potential outcome

The causal effect for A: τ_A = Y_A(1) - Y_A(0) = 1 - ??? (unobservable)
The causal effect for B: τ_B = Y_B(1) - Y_B(0) = ??? - 0 (unobservable)
```

This is why "causal inference" requires assumptions — you need a model for the unobserved counterfactual. The experiment (random assignment) makes the assumption valid by design.

### ATE vs. ATT — When They Differ

```
Promotion targeting example:
  200 users total: 100 high-intent (would convert anyway), 100 low-intent (need nudge)

  Random assignment (ATE scenario):
    50 high-intent → treatment: 50/50 convert = 100% response rate
    50 high-intent → control:   48/50 convert = 96% response rate
    50 low-intent  → treatment: 20/50 convert = 40% response rate
    50 low-intent  → control:   5/50 convert  = 10% response rate

    ATE = average over all:
    Treatment mean: (50+20)/100 = 70%
    Control mean:   (48+5)/100  = 53%
    ATE = 17%

  Targeted assignment (ATT scenario — only send to low-intent users):
    100 low-intent → treatment: 40/100 convert = 40%
    100 high-intent → control: 96/100 convert = 96%

    ATT = effect ON THE TREATED (low-intent):
    Treatment: 40%, Control for same group: 10%
    ATT = 30%  ← much larger because we targeted persuadables

ATE (17%) underestimates the effect on persuadables because it averages with sure-things.
ATT on the right population (30%) is the actionable number for targeting decisions.
```

---

## Uplift Modeling (Deep Dive)

### The Persuasion Matrix — With Numbers

```
Promotion offer to 1,000 users. Historical conversion rates:

                    Would convert WITHOUT offer   Would NOT convert without offer
Converts WITH offer:  Sure Things (200 users)      Persuadables (150 users)
                      Conversion: 100% always      Conversion: 100% if treated, 0% if not
                                                   Uplift = +1.0 per user

Does NOT convert:     Lost Causes (50 users)       Sleeping Dogs (600 users)
WITH offer:           0% regardless               Baseline: 0%, treatment: LOWER (-0.05)
                                                   Uplift = -0.05 per user
```

**What a response model does:** targets highest P(convert | treated). That's Sure Things + Persuadables. Budget wasted on 200 Sure Things who would have converted anyway.

**What uplift model does:** targets highest τ = P(convert|T=1) - P(convert|T=0). That's Persuadables only. 150 users × $10 offer cost = $1,500. 150 incremental conversions.

Response model: 350 users targeted (200 sure things + 150 persuadables) × $10 = $3,500. Still 150 incremental conversions.

Uplift model: 2.3× more cost-efficient for the same incremental conversions.

### T-Learner vs S-Learner — When Each Fails

**T-Learner failure mode — insufficient data per arm:**
```
Treatment group: 100 samples
Control group:   100 samples

Two separate models, each trained on 100 samples
High-cardinality features (user embeddings) → insufficient data per arm
Both models are noisy → their difference (τ estimate) is doubly noisy

S-Learner uses all 200 samples in one model
  → better statistical power
  → but may regularize away the treatment indicator if treatment effect is small
```

**X-Learner — why it works for imbalanced arms:**
```
Scenario: observational data, 90% control (9,000 users), 10% treatment (1,000 users)

T-Learner:  μ₁ trained on 1,000 samples (weak)
            μ₀ trained on 9,000 samples (strong)
            τ = μ₁ - μ₀: dominated by noise from μ₁

X-Learner step 2: impute counterfactuals using the strong model
  For each treated user i: D_i = Y_i - μ₀(X_i)
    → uses the strong 9,000-sample control model to estimate what treated user would earn in control
    → converts 1,000 treated observations into 1,000 good uplift estimates

  Now train τ_1 on 1,000 well-estimated uplift values
  → much better than T-Learner which uses noisy μ₁ directly
```

---

## Doubly Robust Estimator (Deep Dive)

### Why "Doubly Robust" — The Insurance Property

Standard IPW: if propensity model is wrong, estimate is biased. Direct estimator: if outcome model is wrong, estimate is biased.

AIPW formula:
```
ATE_DR = E[μ₁(x) - μ₀(x)          ← direct model term
          + T(Y - μ₁(x)) / e(x)    ← IPW correction for treated
          - (1-T)(Y - μ₀(x)) / (1-e(x))]  ← IPW correction for control
```

**Case 1: outcome model is correct, propensity model is wrong:**
```
If μ₁(x) = true E[Y|T=1,X=x]:
  Y - μ₁(x) ≈ 0 for treated units
  The IPW correction term ≈ 0 / e(x) ≈ 0
  ATE_DR ≈ E[μ₁(x) - μ₀(x)]  ← the direct estimator, which is correct
```

**Case 2: propensity model is correct, outcome model is wrong:**
```
If e(x) = true P(T=1|X=x):
  The IPW correction exactly debiases the direct model
  ATE_DR → IPW estimator which is consistent
  (the correction term absorbs the direct model's bias)
```

One model correct → consistent estimate. Both wrong → biased, but less so than either alone.

### Off-Policy Evaluation — Using DR for Policy Evaluation

```
Problem: evaluate new bidding policy π_new without running it live
Data: historical logs under old policy π_old

For each impression i with action a_i (bid level) and outcome y_i (click/conversion):

AIPW OPE estimate:
  V(π_new) = (1/n) Σᵢ [μ(x_i, π_new(x_i))           ← direct model
              + (π_new(a_i|x_i)/π_old(a_i|x_i)) × (y_i - μ(x_i, a_i))]  ← IS correction

Doubly robust: consistent if EITHER the direct model μ OR the propensity ratio is correct
```

This lets you evaluate a new bidding algorithm against historical data before running a live experiment — cheaper and faster, with theoretical guarantees.

---

## Incrementality Measurement (Deep Dive)

### Why Attribution Overcounts Conversions

```
Without ads:
  100 users would organically search for "running shoes"
  30 would visit the website
  10 would purchase

With ads (retargeting):
  Same 100 users see an ad → 15 click → 12 visit website → 8 purchase
  + the 30 who would have visited organically anyway also still visit → 10 of those purchase
  Total purchases: 8 (from ad click) + 10 (organic) = 18

Attribution model (last-click):
  8 ad-click conversions → attributed to ads
  10 organic conversions → attributed to organic
  Reports: "ads drove 8 conversions"

But wait — how many of the 8 ad-click conversions would have purchased organically anyway?
  Of the 15 ad clickers, maybe 4 would have organically purchased regardless
  → incremental conversions from ads: 8 - 4 = 4 (not 8)

iROAS with attribution: 8 conversions / ad_spend
iROAS with incrementality: 4 incremental conversions / ad_spend
→ attribution reports 2× the true ROAS
```

This is why geo holdout or ghost ad experiments are necessary for advertisers to know if ads are actually working.

### Ghost Ads — The Cleanest Incrementality Test

```
Standard holdout:
  Treatment group: sees brand's ad
  Control group: sees no ad at all

Problem: control group knows there's "empty space" where an ad would be
  → different page layout → changes browsing behavior slightly
  → confounds the measurement

Ghost ad holdout:
  Treatment group: sees brand's ad (e.g., Nike shoe ad)
  Control group: sees a PUBLIC SERVICE ANNOUNCEMENT or competitor ad
                 (same format, same slot, different brand)

Both groups see an ad. The only difference is which brand.
Comparison is cleaner: isolates the Nike ad's effect from the "any ad in this slot" effect.
```

Used by Google, Meta, and Netflix ads for high-fidelity incrementality measurement.

---

## Double ML (Deep Dive)

### Why Standard Regression Fails in High Dimensions

Imagine estimating the effect of pricing on conversion, controlling for 1,000 user features X:

```
Standard: Y = β₀ + β₁ × price + β₂X₁ + β₃X₂ + ... + β₁₀₀₁X₁₀₀₀ + ε

With regularization (Lasso), β₁₁ is shrunk toward zero along with all other βᵢ.
If price is correlated with some Xᵢ features, the regularization on Xᵢ
leaks into regularization on price → price coefficient β₁ is biased.
```

**Double ML partialling out:**
```
Step 1: T̂ = GBT(X)  ← predict price from features
        Ṽ = price - T̂  ← residual: "price variation unexplained by features"

Step 2: Ŷ = GBT(X)  ← predict conversion from features
        Ũ = conversion - Ŷ  ← residual: "conversion unexplained by features"

Step 3: regress Ũ on Ṽ  ← no features, just two residuals
        coefficient = causal effect of price on conversion
```

Both residuals have X "partialled out". Their covariation is the pure price → conversion effect, free from feature confounding.

**Why this works despite using ML:** the key insight is that the nuisance models (T̂, Ŷ) can be as complex as needed without biasing the causal estimate. The residuals Ṽ and Ũ are approximately independent of X by construction (they're what's LEFT after X explains everything it can). The final regression of Ũ on Ṽ is just a simple univariate regression — no regularization bias.

---
