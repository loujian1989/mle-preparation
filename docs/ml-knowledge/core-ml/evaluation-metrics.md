# Evaluation Metrics — ML Knowledge Q&A

P0: Universal — asked by every company in some form.

---

## Precision, Recall, and the Threshold Trade-Off

### Q: When do you optimize for precision vs. recall, and why is that a business decision not a model decision?

**Answer (Staff level):**
- **Precision** = TP / (TP + FP): of all items flagged positive, what fraction are actually positive. Optimize when false positives are costly (e.g., fraud block that declines a legitimate card → customer lost).
- **Recall** = TP / (TP + FN): of all actual positives, what fraction did we catch. Optimize when false negatives are costly (e.g., missed fraud → chargeback + reputational damage).
- The threshold is a **post-model business decision** — model outputs a probability score; threshold is set by minimizing `FP_cost * FP + FN_cost * FN`. FP:FN cost ratio is not a model hyperparameter.
- Communicate to interviewer: "I'd build the model to maximize AUPRC, then set the operating threshold in a separate step with the business team."

**Company context:** Stripe, Reddit, OpenAI (any trust/safety). Stripe specifically asks: "how do you choose the threshold?" to see if you confuse model selection with business decision.

**Common wrong answer:** "I'd tune the threshold to maximize F1." — Staff answer: F1 assumes FP and FN have equal cost, which is almost never true in production. State the actual cost ratio first.

---

### Q: Why is accuracy a misleading metric at 0.5% fraud rate?

**Answer (Staff level):**
- A model that predicts "not fraud" for every transaction achieves 99.5% accuracy with 0% recall on fraud — it's useless.
- At extreme class imbalance, accuracy collapses to majority-class prevalence.
- Use **AUPRC** (area under precision-recall curve) as the primary metric: it evaluates model discrimination specifically on the positive (minority) class across all thresholds.
- AUROC is also useful but less sensitive at extreme imbalance because it includes TN-heavy operating points in the area calculation.

**Company context:** Stripe (fraud, 0.5% rate), OpenAI (content moderation, rare violations).

**Common wrong answer:** "I'd use F1 score." — Mid-level. F1 is a single-threshold metric; AUPRC aggregates across all thresholds. For skewed classes, always report AUPRC as primary.

---

## AUPRC vs. AUROC

### Q: When does AUROC diverge from AUPRC, and which do you report first?

**Answer (Staff level):**
- **AUROC** = P(score(pos) > score(neg)): rank-ordering of a random positive vs. random negative. Insensitive to class ratio — useful for comparing models, not for setting operating points.
- **AUPRC** = area under precision-recall curve. Penalizes heavily when positives are rare and the model produces many false positives at high recall.
- They diverge significantly when class ratio > 1:20. A model can have AUROC=0.95 and AUPRC=0.30 if it produces many FPs chasing high recall on a rare class.
- **Report order:** AUPRC first for imbalanced classification; AUROC as secondary.

**Company context:** Meta (ads CTR, ~2% click rate — borderline), Stripe (fraud), Reddit (spam).

**Common wrong answer:** "AUROC is my primary metric because it's threshold-independent." — This ignores that AUROC includes the easy TN-dominated region; at high imbalance, AUPRC is strictly more informative.

---

## NDCG, MRR, Precision@K

### Q: Explain NDCG@K. Why is it better than Precision@K for ranking?

**Answer (Staff level):**
- **DCG@K** = Σ_{i=1}^{K} (2^rel_i − 1) / log₂(i+1): higher-relevance items get exponentially more credit; earlier positions get higher weight via the log discount.
- **NDCG@K** = DCG@K / IDCG@K (ideal DCG): normalizes to [0,1] and makes scores comparable across queries with different numbers of relevant items.
- **Precision@K** treats all top-K positions as equal — being relevant at position 1 vs. position K is worth the same. NDCG properly discounts lower positions.
- Use **MRR** when only the first relevant item matters (search: "did I find what I wanted?"). Use **NDCG** when you have graded relevance or care about ordering within K.

**Company context:** Reddit, Netflix, Meta, Pinterest, Roblox (any ranking system). Reddit interview will ask you to implement NDCG live.

**Common wrong answer:** Confusing DCG formula denominator: some use log₂(i+1), others log₂(i+2) for 0-indexed. State which convention you're using. Also: forgetting to handle IDCG=0 (all items irrelevant → return 0.0, don't divide).

---

### Q: How do offline ranking metrics (NDCG) relate to online metrics (CTR, revenue)?

**Answer (Staff level):**
- Offline NDCG measures ranking quality against historical engagement labels — it's a proxy.
- **Surrogate gap**: a model with NDCG +2% can have negative online impact if it over-ranks a type of content users dislike in context (e.g., autoplay video in a reading feed).
- A/B testing is ground truth. Offline metrics should be validated to predict online metrics before being used as development signal.
- **Position bias** in historical data (items ranked higher got more clicks, creating biased labels) means NDCG computed on naive logs underestimates quality of items shown low. Fix: use counterfactual evaluation or IPS-weighted NDCG.

**Company context:** Netflix (explicit A/B design question), Reddit, Meta.

**Common wrong answer:** "I'd optimize NDCG offline and ship the winner." — Staff answer mentions the surrogate gap and requires A/B validation.

---

## Calibration

### Q: What is Expected Calibration Error (ECE) and why does GBT need Platt scaling?

**Answer (Staff level):**
- **ECE** = Σ_b (|B_b| / N) × |mean_score(B_b) − true_rate(B_b)|: weighted average absolute difference between predicted probability and actual frequency, binned by predicted score.
- GBT is a discriminative model optimized for ranking (AUC/log-loss), not probability estimation. Its scores are monotone but poorly calibrated — GBT pushes probabilities toward extremes (overconfident near 0 and 1).
- **Platt scaling** (logistic regression on raw scores) or **isotonic regression** are calibration post-processors. Use Platt when you have limited calibration data; isotonic when you have more data but need non-parametric flexibility.
- Calibration matters when the output probability is used downstream: risk scores fed to pricing, decision thresholds set by another system, or combining scores from multiple models.

**Company context:** Stripe (fraud score feeds risk engine), Shopify (churn score used for intervention budget), OpenAI (content score feeds policy layer).

**Common wrong answer:** "I'd just use predict_proba() from sklearn." — Raw GBT `predict_proba` is not calibrated. Demonstrate awareness of the difference between discrimination (AUC) and calibration (ECE/Brier).

---

## Offline vs. Online Metrics

### Q: Give an example where offline metrics improved but the online metric degraded.

**Answer (Staff level):**
- **Example 1 (Feedback loop):** Ranking model trained on CTR data. Offline: NDCG improved +5%. Online: Users clicked more but spent less time — CTR is a weak proxy for satisfaction. Watch time or return rate is the true north star.
- **Example 2 (Distribution shift):** Fraud model AUPRC improved on held-out data. Online: FP rate spiked on a new merchant category not in training — a covariate shift the offline split didn't capture.
- **Example 3 (Label lag):** Churn model improved AUPRC on 30-day labels. Online: intervention launched at day 15 changed behavior, invalidating the label definition. Counterfactual impact is unobservable without a holdout.
- The pattern: offline labels are historical; online the world is reactive. Always define which online metric the offline metric is a proxy for, and validate the proxy empirically before relying on it.

**Company context:** Netflix (product thinking question), Meta (ranking experiments), Reddit (live model building context).

**Common wrong answer:** Giving a generic answer without a concrete mechanism. Interviewers want the specific causal chain: why did offline improve but online degrade?

---

## Brier Score

### Q: What does Brier score measure, and when is it preferred over ECE?

**Answer (Staff level):**
- **Brier score** = (1/N) Σ (p_i − y_i)²: mean squared error between predicted probabilities and binary outcomes. Lower = better.
- Combines calibration and discrimination: a perfectly calibrated but uninformative model (always predicts base rate) has Brier = p(1-p), while a perfectly calibrated + discriminative model approaches 0.
- **ECE** is more interpretable for calibration alone (units are in probability), but sensitive to binning strategy (n_bins choice). **Brier score** is bin-free and a proper scoring rule — it can't be gamed by miscalibration tricks.
- Use Brier when comparing models holistically on probability quality; use ECE when you specifically want to communicate "how far off are our probability estimates?"

**Company context:** Shopify (churn probability drives intervention budget — probability quality matters), Stripe (fraud score used for dynamic thresholds).

**Common wrong answer:** Treating Brier score as only a calibration metric. It's a proper scoring rule that measures both calibration and sharpness (discrimination).

---

## Precision, Recall, and Threshold (Deep Dive)

### What the Confusion Matrix Actually Encodes

```
                 Predicted Positive    Predicted Negative
Actual Positive:       TP                    FN
Actual Negative:       FP                    TN
```

Every threshold choice traces out a path through this table. Lowering the threshold:
- Flags more items as positive → TP goes up, FP goes up
- Misses fewer positives → FN goes down
- Precision drops (more FPs in the denominator), recall rises

**The threshold is a lever that trades FP for FN.** The model's job is to make that trade-off as efficient as possible — the business decides where on the curve to operate.

### Why It's a Business Decision — The Cost Formula

```
Total cost = FP_cost × FP + FN_cost × FN
```

The optimal threshold minimizes this cost. Plugging in real numbers:

```
Fraud model at Stripe:
  FP: block a legitimate transaction → customer calls support, possible churn
      FP_cost ≈ $15 (support cost + churn probability × LTV)
  FN: miss a fraud → chargeback + fee + reputational damage
      FN_cost ≈ $90 (average fraud amount + $25 chargeback fee)

Optimal threshold: set so that the marginal FP and marginal FN have equal cost
  P(fraud | score=t) × FN_cost = (1 - P(fraud | score=t)) × FP_cost
  t* = FP_cost / (FP_cost + FN_cost) = 15 / (15 + 90) ≈ 0.14
```

The model never sees these dollar amounts — they come from the finance and ops teams. Maximizing F1 implicitly sets `FP_cost = FN_cost`, which is almost never the right business assumption.

### Why Accuracy Fails at Imbalance — Concrete Math

```
Dataset: 100,000 transactions, 0.5% fraud (500 fraud, 99,500 legit)

Naive model: always predict "not fraud"
  TP=0, FP=0, FN=500, TN=99,500
  Accuracy = (0 + 99,500) / 100,000 = 99.5%  ← looks great
  Recall = 0 / 500 = 0%                       ← catches nothing
```

Accuracy rewards the model for being good at the easy task (classifying the majority class) and never penalizes it for failing at the hard task. AUPRC focuses exclusively on the positive class performance across all thresholds.

### The Precision-Recall Curve — What Shape Tells You

```
Perfect model:   curve hugs top-right corner (P=1, R=1 achievable simultaneously)
Random model:    flat line at P = prevalence (0.5% for fraud)
Production goal: high area under curve, especially at high-recall operating points
```

The "area" interpretation: average precision across all recall thresholds. An AUPRC of 0.70 means: if you randomly sample operating points across the recall range, you average 70% precision — 140× better than random (0.5% baseline).

---

## AUPRC vs. AUROC (Deep Dive)

### What AUROC Actually Measures — And Why It Hides Imbalance

AUROC = P(score(random positive) > score(random negative))

For a fraud model at 0.5% rate:
```
Random positive = 1 fraud transaction
Random negative = 1 of 199 legitimate transactions (on average)

AUROC=0.95: 95% of the time, the fraud has a higher score than the legit
             Sounds great — but doesn't say anything about what happens
             when you actually set a threshold and flag transactions
```

The problem: AUROC includes the TN-dominated region. At a fraud rate of 0.5%, most of the ROC curve is about ranking negatives against each other — a task that's easy and irrelevant. The curve area is dominated by the bottom-right region (high recall, high FP rate) where there are thousands of TN-TN comparisons for every TP.

### Where AUROC and AUPRC Diverge — Concrete Example

```
Model A: excellent at high precision, poor at high recall
  AUROC = 0.95   (ranks positives above negatives most of the time)
  AUPRC = 0.60   (when you need high recall, precision tanks)

Model B: mediocre overall but consistent
  AUROC = 0.92
  AUPRC = 0.55

A is strictly better. But the AUROC gap (0.95 vs 0.92) looks small.
The AUPRC gap (0.60 vs 0.55) correctly signals A is meaningfully better
at the high-recall operating points that production actually uses.
```

### Reporting Convention

```
Imbalanced binary classification (fraud, spam, abuse):
  Primary:   AUPRC  ← how well you rank positives among themselves
  Secondary: AUROC  ← overall discrimination quality

Balanced classification or multi-class:
  Primary:   AUROC or macro-averaged F1
  Calibration check: ECE or Brier score separately
```

---

## NDCG, MRR, Precision@K (Deep Dive)

### Why Position Matters — The Log Discount

Users don't read equally. Eye-tracking studies show attention drops off rapidly:
```
Position 1: ~100% of users see it
Position 3: ~80% see it
Position 5: ~50% see it
Position 10: ~20% see it
```

NDCG encodes this with a logarithmic discount:

```
DCG@K = Σ_{i=1}^{K} (2^rel_i − 1) / log₂(i+1)

Position 1: discount = 1/log₂(2) = 1.0
Position 2: discount = 1/log₂(3) = 0.63
Position 3: discount = 1/log₂(4) = 0.50
Position 10: discount = 1/log₂(11) = 0.29
```

A highly relevant item (rel=3) at position 1 is worth `(2³-1)/1.0 = 7.0`.
The same item at position 10 is worth `7.0 × 0.29 = 2.03`. 3.4× less credit.

### Precision@K vs. NDCG@K — Concrete Example

Two ranking systems for the same query (★★★ = highly relevant, ★ = slightly relevant, — = irrelevant):

```
System A: [★★★, —, —, —, ★★★]   top-5
System B: [★★★, ★★★, —, —, —]   top-5

Precision@5: both = 2/5 = 0.40  ← identical, can't distinguish
NDCG@5: A ≈ 0.64, B ≈ 0.87     ← B correctly ranked higher (both relevant items early)
```

Precision@K is position-blind within the top K. NDCG rewards systems that surface the most relevant items earliest.

### MRR — When Only First Matters

```
MRR = (1/|Q|) Σ_q 1/rank_q(first relevant item)

Query 1: first relevant item at rank 3 → 1/3
Query 2: first relevant item at rank 1 → 1/1
Query 3: first relevant item at rank 5 → 1/5
MRR = (1/3 + 1 + 1/5) / 3 = 0.51
```

Use MRR for navigational queries ("find the official site") or question answering ("what is X?") — the user stops at the first good result. Use NDCG when users consume multiple items (feed, search results page with multiple useful entries).

### The Offline-Online Gap

NDCG computed on historical click logs has a fundamental problem: **position bias**. Items shown at rank 1 got more clicks simply because they were shown there, not because they're more relevant. This inflates the NDCG of the current production model (which determined the positions used in the logs).

```
Naive NDCG on logs: biased toward current production model
IPS-weighted NDCG:  weight each click by 1/P(shown at that position)
                    corrects for the fact that position 1 gets shown more
```

In production: always report IPS-corrected offline metrics when evaluating ranking quality, and validate against A/B before deploying.

---

## Calibration (Deep Dive)

### Why GBT Scores Are Systematically Miscalibrated

GBT trains on log-loss (cross-entropy), which rewards correct ranking — not correct probability estimation. The loss is minimized when the model's ordering of predictions is correct, not when the predicted probabilities are accurate.

The result: GBT pushes scores toward extremes.

```
True fraud probability: 0.7
GBT output:             0.95  ← overconfident

True fraud probability: 0.3
GBT output:             0.08  ← overconfident in the other direction
```

A calibration plot (predicted probability vs. observed rate) for GBT typically shows an S-shaped curve — underestimates at low probabilities, overestimates at high.

### ECE — What the Formula Means

```
ECE = Σ_b (|B_b| / N) × |mean_score(B_b) − true_rate(B_b)|
```

Split predictions into B bins (e.g. [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0)):

```
Bin [0.7, 0.8): model predicted ~0.75 for these 200 samples
                actual fraud rate among them: 0.52
                contribution: (200/10000) × |0.75 - 0.52| = 0.0046

Sum over all bins → ECE
```

ECE = 0.05 means: on average, the model's probabilities are off by 5 percentage points. Acceptable for most production systems; ECE > 0.1 is a signal that calibration is needed before using the score as a probability.

### Platt Scaling vs. Isotonic Regression

Both are post-hoc calibration methods applied to a held-out calibration set (separate from the validation set used for model selection):

**Platt scaling** — fit a logistic regression on raw scores:
```
P(y=1 | score) = σ(A·score + B)
```
Two parameters. Works well with < 1000 calibration samples. Assumes a sigmoid relationship between raw score and true probability (reasonable for GBT).

**Isotonic regression** — fit a non-parametric monotone step function:
```
P(y=1 | score) = f(score)  where f is monotone non-decreasing
```
No parametric assumption. Needs ≥ 5000 calibration samples to avoid overfitting the step function. Better when the raw score-probability relationship is non-sigmoid.

**When calibration matters most:**
- Score is multiplied by a dollar amount (fraud chargeback risk = score × average_fraud_value)
- Downstream system sets its own threshold (your score is a component, not the final decision)
- Multiple models' scores are combined (uncalibrated scores from different models aren't comparable)

---

## Offline vs. Online Metrics (Deep Dive)

### The Three Failure Modes — Mechanisms

**1. Proxy metric divergence**: the offline metric is optimizing for a proxy that doesn't match the actual user value:

```
Optimize for:  CTR (clicks per impression)
Actual goal:   user satisfaction (time spent, return visit)

What happens:  model learns clickbait headlines are high-CTR
               users click, immediately leave (pogo-sticking)
               CTR ↑, watch time ↓, churn ↑
```

**2. Distribution shift between offline and online:**

```
Offline: model trained on Jan-March data, validated on April data
Online:  deployed in May → new seasonal patterns, new content categories

AUPRC on April holdout: 0.72  ← looks good
FPR on May production:  2×    ← covariate shift not captured in April holdout
```

**3. Feedback loop — the labels change when you deploy:**

```
Fraud model improved → deployed → blocks more fraudsters
Fraudsters adapt → change transaction patterns
New patterns not in training data → model deteriorates
AUPRC on static holdout: still 0.72
Production recall: declining month over month
```

### The Surrogate Gap — How to Minimize It

The surrogate gap is the difference between offline metric improvement and online metric improvement.

```
ΔOffline > 0 but ΔOnline ≤ 0 → surrogate gap
```

Strategies to reduce it:
1. **Counterfactual offline evaluation**: use IPS-weighted metrics that correct for position bias and exposure bias in logged data
2. **Interleaving tests**: cheaper than full A/B — interleave two rankers' results and measure user preference signals
3. **Proxy validation history**: track the empirical correlation between ΔNDCG and ΔCTR over past experiments; if correlation is low, the metric is unreliable as a development signal
4. **Holdout by time**: always split by time, not randomly — random splits leak future patterns into training

---

## Brier Score (Deep Dive)

### The Decomposition — Why It Captures Both Things

Brier score decomposes as:

```
Brier = Uncertainty − Resolution + Calibration_error

Uncertainty:       p̄(1-p̄)  — inherent difficulty of the task (fixed)
Resolution:        how much your predictions vary around the base rate
                   high resolution = model confidently separates positives from negatives
Calibration_error: how far predicted probabilities are from true frequencies
```

A perfectly calibrated but useless model (always predicts base rate p̄):
```
Brier = p̄(1-p̄) − 0 + 0 = p̄(1-p̄)
For fraud at 0.5%: Brier = 0.005 × 0.995 = 0.00498
```

A perfect model (predicts 1.0 for all frauds, 0.0 for all legits):
```
Brier = 0
```

Your model should be between these two. A useful reference: if your Brier score is worse than `p̄(1-p̄)`, you're doing worse than "always predict the base rate" — a red flag.

### Brier vs. ECE — When Each Is More Useful

```
Use Brier when:
  - Comparing two models holistically (captures both discrimination + calibration)
  - No concern about specific probability bins being well-calibrated
  - Want a single number that's a proper scoring rule (can't be gamed)

Use ECE when:
  - Need to communicate calibration to a stakeholder ("our scores are off by X%")
  - Diagnosing WHERE calibration breaks down (which bin is worst)
  - Setting thresholds for downstream systems that need accurate probabilities
```

**The gaming problem with ECE**: ECE is sensitive to binning strategy. With 5 bins vs. 20 bins you get different ECE values for the same model. Brier is bin-free and therefore more comparable across experiments.
