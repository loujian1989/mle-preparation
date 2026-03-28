# Fraud & Trust Safety — ML Knowledge Q&A

P0: Stripe, OpenAI. Also Pinterest (spam), Reddit (content moderation).

---

## Velocity Features

### Q: What are velocity features and why are they critical for fraud detection?

**Answer (Staff level):**
- **Velocity feature**: count or sum of an event in a rolling time window. Example: `txn_count_in_past_1h`, `amount_sum_past_24h`, `distinct_merchants_past_7d`.
- **Why critical**: individual transactions look normal in isolation. Fraud patterns emerge at the behavioral level — a stolen card is used 15 times in 2 hours across 8 merchants. No single transaction triggers a rule; the velocity pattern does.
- **Feature taxonomy**:
  - Card-level: transactions/hour on this card
  - User-level: new accounts/day from this device fingerprint
  - Merchant-level: chargebacks/week at this merchant (risk score)
  - Network-level: fraction of this card's recent transactions to high-risk merchant categories
- **Implementation challenge**: velocity features require real-time aggregation with sub-second latency. In production: pre-computed in a feature store (Redis / DynamoDB) with sliding-window counts using approximate counters (Count-Min Sketch for memory efficiency) or exact counts with TTL-keyed storage.
- **Leakage trap**: if velocity features are computed on the full historical dataset during training (not strictly windowed from each observation's cutoff), future transactions contaminate past windows. Always compute windows relative to the observation timestamp.

**Company context:** Stripe (core feature engineering question), OpenAI (content moderation velocity: API call rate).

**Common wrong answer:** "I'd use transaction amount as the primary feature." — Amount alone is insufficient; behavioral patterns across transactions are the signal.

---

## Class Imbalance

### Q: Fraud rate is 0.5%. How do you handle class imbalance? Compare 4 approaches.

**Answer (Staff level):**

| Approach | Mechanism | Pros | Cons |
|---|---|---|---|
| **scale_pos_weight** (XGBoost) or `class_weight='balanced'` | Multiply positive-class loss by `(n_neg/n_pos)` | No data modification; simple | Amplifies noise on mislabeled positives |
| **Oversampling (SMOTE)** | Synthesize new positive examples in feature space | Increases positive class diversity | Doesn't generalize well for high-dimensional sparse features |
| **Undersampling** | Randomly drop negative examples | Reduces training time | Loses information from negatives |
| **Focal loss** | Down-weight easy negatives `FL = −α(1−p)^γ log(p)` | Handles imbalance + hard examples jointly | Requires tuning γ, α; needs calibration afterward |

- **Recommended for GBT**: `scale_pos_weight` (XGBoost) or `class_weight` (sklearn). Fast, no data modification.
- **Recommended for NN**: focal loss with γ=2, α=0.25 (RetinaNet defaults). Then calibrate with Platt scaling because focal loss shifts probability estimates.
- **Critical**: resampling should only occur in the training set. Never resample validation or test — you need natural class distribution to evaluate calibrated metrics (AUPRC, ECE).

**Company context:** Stripe, OpenAI (content moderation). Interview will ask you to name and compare at least 3 approaches.

**Common wrong answer:** "I'd use SMOTE to balance the classes." — SMOTE works for low-dimensional tabular data; it fails for high-dimensional sparse or mixed-type features. Also, mixing it with GBT adds complexity without benefit (GBT's `class_weight` is simpler and more effective).

---

## Threshold as Business Decision

### Q: Your fraud model produces probability scores. How do you set the operating threshold?

**Answer (Staff level):**
- **The threshold is not a model hyperparameter** — it's a business decision based on cost estimates.
- Define: `cost = FP_cost × FP + FN_cost × FN`
  - FP_cost: cost of blocking a legitimate transaction (friction, potential customer churn, revenue loss). E.g., $1.
  - FN_cost: cost of allowing fraud (chargeback fee + operational cost + potential regulatory exposure). E.g., $50.
- Sweep thresholds from 0 to 1, compute expected cost at each threshold, pick the minimum.
- **In practice**: use the precision-recall curve. Find the threshold where the cost-weighted F-beta score is maximized. For FN_cost >> FP_cost: β > 1 (recall-favoring).
- **Stakeholder communication**: "at threshold 0.3, we block 95% of fraud but also decline 8% of legitimate transactions — that's a business decision, not a model decision. I can show you the cost curve and let the business set the acceptable trade-off."
- **Multi-tier thresholds**: instead of binary block/allow, implement a 3-tier system: allow (score < 0.2), review queue (0.2–0.7), auto-block (score > 0.7). Review queue allows human-in-the-loop for ambiguous cases.

**Company context:** Stripe (this is explicitly probed in interviews), OpenAI (content moderation: auto-remove vs. human review vs. allow).

**Common wrong answer:** "I'd tune the threshold to maximize F1." — F1 assumes equal FP/FN cost. Must define the cost ratio first.

---

## Adversarial Drift in Fraud

### Q: Fraud models degrade faster than other ML models. Why, and how do you handle it?

**Answer (Staff level):**
- **Adversarial drift**: fraudsters observe which transactions get blocked and adapt their behavior to evade detection. Unlike natural distribution drift (seasonal, demographic), adversarial drift is actively optimized against your model.
- **Examples**:
  - Fraudsters shift to transaction amounts just below your velocity thresholds (if you block >5 txns/hour at same merchant, they do 4).
  - Fraud ring changes device fingerprints after you add that as a feature.
  - Geographic patterns shift after you deploy geo-risk features.
- **Mitigation strategies**:
  1. **Frequent retraining**: weekly or bi-weekly model refresh with recent labeled data.
  2. **Concept drift detection**: monitor feature distributions and model score distributions. Alert on KL divergence > threshold from baseline.
  3. **Online learning**: partial_fit or streaming gradient updates on recent confirmed fraud cases.
  4. **Behavioral features vs. static features**: velocity and session-level features are harder to spoof than static card metadata.
  5. **Ensemble diversity**: multiple models with different feature sets — adversary must evade all simultaneously.

**Company context:** Stripe. This distinguishes Staff from Senior: Senior builds the initial model; Staff thinks about the deployment lifecycle and adversary.

**Common wrong answer:** "I'd retrain monthly on new data." — Monthly is too slow for adversarial drift. Must specify monitoring (drift detection), frequency reasoning, and the adversarial nature of the problem.

---

## Content Moderation at Scale

### Q: How do you build a content moderation system at 1B posts/day? (OpenAI context)

**Answer (Staff level):**
- **Two-stage pipeline** (same as fraud):
  1. **Heuristic pre-filter**: regex, keyword blocklist, account-level reputation score. O(1), runs first, handles obvious cases. High recall, lower precision.
  2. **ML classifier**: multimodal model (text + image) predicts probability of policy violation per content type (hate speech, CSAM, misinformation). Runs on post-heuristic candidates.
- **Confidence-based routing**:
  - High confidence positive (p > 0.95): auto-remove
  - Ambiguous (0.3 < p < 0.95): route to human review queue with model explanation (SHAP attribution)
  - High confidence negative (p < 0.3): allow
- **Prioritization**: not all content is equal urgency. Viral content (1M impressions/hr) gets expedited review. Use view count / share velocity as urgency signal.
- **Evaluation**: false positive rate on benign content (user experience), false negative rate on policy violations (safety). Separate evaluation per policy category (hate speech vs. CSAM vs. spam — different acceptable FP/FN trade-offs).
- **Model cards + bias monitoring**: moderation models have historically over-flagged certain languages/demographics. Monitor per-demographic false positive rates. Audit quarterly with equity analysis.

**Company context:** OpenAI (content moderation system design, also probed in ML knowledge round).

**Common wrong answer:** "I'd train a classifier and deploy it." — No two-stage architecture, no confidence-based routing, no human review integration, no bias monitoring.

---

## SHAP for Fraud Explainability

### Q: Why is model explainability important in fraud, and how do you use SHAP?

**Answer (Staff level):**
- **Business requirement**: chargeback disputes require explanation ("why was this transaction flagged?"). Some jurisdictions (EU GDPR Article 22) require "meaningful information about the logic involved" in automated decisions affecting users.
- **SHAP (Shapley Additive Explanations)**: assigns each feature a contribution score based on its marginal contribution across all subsets of features (Shapley values from cooperative game theory).
  - `prediction = base_value + Σ SHAP_i(feature_i)`
  - Additive: individual contributions sum to the model output.
  - Consistent: if a feature's contribution increases, its SHAP value increases.
- **TreeExplainer**: O(T × D) per prediction (where T = trees, D = depth). Exact Shapley values for tree models — fast.
- **Use cases**:
  1. Alert messaging: "Transaction blocked because: international merchant (+0.32), unusual hour (+0.21), high velocity (+0.19)."
  2. Model debugging: aggregate SHAP values across false positives to find systematic errors.
  3. Feature selection: identify features with near-zero mean |SHAP| across training set.

**Company context:** Stripe, OpenAI (explainability in safety-critical systems), Shopify (take-home template includes SHAP).

**Common wrong answer:** "I'd use feature importance from the tree." — Split-gain feature importance is a global average and doesn't explain individual predictions. SHAP gives per-prediction, additive attributions.
