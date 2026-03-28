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
