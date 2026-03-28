# Feature Engineering — ML Knowledge Q&A

P1: Stripe, Shopify, Uber, Reddit.

---

## Leakage

### Q: Describe three types of feature leakage, give a concrete example of each.

**Answer (Staff level):**

| Leakage Type | Definition | Example |
|---|---|---|
| **Post-observation** | Feature derived from events that happen AFTER the observation cutoff | In churn prediction: "support ticket opened in next 30 days" as a feature (this IS the effect of churn, not a cause) |
| **Target-derived** | Feature that encodes the label, directly or via a proxy | In fraud detection: "charge was disputed" as a feature when the label is also derived from disputes |
| **Temporal** | Using future data in aggregations rolled backward | Computing "average order value over the next 90 days" as a feature for a model predicting current churn |
| **Row-level** | Test rows included in train statistics (e.g., normalization) | Fitting a StandardScaler on train+test combined before split |

- **Detection heuristic**: correlation > 0.5 between a feature and the label in training data — flag and investigate. Leakage features often have implausibly high predictive power.
- **Time-based split is the primary defense** for temporal leakage: all features must be computable using only data available at prediction time (observation cutoff).

**Company context:** Stripe (they test this explicitly), Shopify (churn prediction take-home), Reddit (live model building, will ask "what might leak?").

**Common wrong answer:** "I'd use feature importance to detect leakage." — Feature importance identifies influential features; high importance ≠ leakage. You must trace the feature's derivation back to whether future data was used.

---

## Target Encoding

### Q: What is target encoding, and what goes wrong if you don't apply it correctly?

**Answer (Staff level):**
- **Target encoding**: replace a categorical variable with the mean of the target conditioned on that category. Example: `merchant_category → mean(churn_rate for merchant_category)`.
- **Why it's useful**: handles high-cardinality categoricals better than one-hot (avoids O(K) dimensionality explosion). Captures the signal directly in a single continuous feature.
- **What goes wrong — leakage**: if you compute target encodings using all rows (including the current row's label), you've created direct leakage. The encoding for a single-occurrence category is exactly the label.
- **Correct approaches**:
  1. **Hold-out encoding**: compute encoding on train split only; apply to val/test. Never use the row's own label.
  2. **K-fold target encoding** (sklearn `TargetEncoder`): use out-of-fold predictions. Row i's encoding = mean label of rows not in fold containing i. Reduces leakage while using more training data.
  3. **Smoothing**: blend category mean with global mean proportional to category frequency: `enc_k = (n_k · mean_k + m · global_mean) / (n_k + m)`. Prevents noise from rare categories.

**Company context:** Shopify, Stripe (high-cardinality categoricals in transaction data).

**Common wrong answer:** "I'd use mean target encoding across all rows." — This is the leakage version. Must specify out-of-fold or train-only encoding.

---

## Log Transform

### Q: When do you apply a log transform, and what can go wrong?

**Answer (Staff level):**
- **Apply when**: the feature is right-skewed (long right tail — most values small, a few very large). Examples: transaction amount, social media follower count, time between events.
- **Why it helps**: reduces range of the feature (from 1–10^6 to 0–14 for log₁₀), makes linear models more effective (linear relationship in log-space corresponds to power-law in original space), reduces influence of extreme outliers.
- **`log1p` vs `log`**: use `np.log1p(x)` = log(1+x) to handle x=0 (avoids log(0) = -∞).
- **What can go wrong**:
  1. Negative values: log is undefined for x ≤ 0. Must handle: `log(|x| + 1) · sign(x)` or clip/impute negatives.
  2. Monotone transform hides interaction effects: two features with multiplicative interaction (a × b) become additive after log (log(a) + log(b)), which can help or hurt depending on the model.
  3. For tree models (GBT, XGBoost): log transform is often unnecessary — trees are invariant to monotone transformations. Helps for linear models and distance-based models (KNN, SVM).

**Company context:** Stripe, Shopify, Uber (amount, count, and rate features are ubiquitous).

**Common wrong answer:** "I'd always log-transform numeric features." — Tree models are invariant to monotone transforms; log transform is specifically for linear models and neural nets that benefit from bounded feature ranges.

---

## Missing Values

### Q: How do you handle missing values differently for GBT vs. neural networks?

**Answer (Staff level):**

| Model | Approach | Reason |
|---|---|---|
| **GBT (XGBoost, LightGBM)** | Leave as NaN; let the library handle it | XGBoost/LightGBM have built-in NaN handling: at each split, NaN values go to the direction that minimizes loss. No imputation needed. |
| **Scikit-learn GBT** | Mean/median imputation | sklearn's GBT doesn't handle NaN natively; must impute. |
| **Linear models** | Mean/median imputation + add `is_missing` binary flag | Imputing with mean removes the missing signal. The binary flag retains "was this field missing?" as an informative feature. |
| **Neural networks** | Learned imputation (zero + mask token) or mean imputation | Zero-fill + a "missing" indicator feature. Or use masking in the embedding layer (categorical) / learned imputation layer (continuous). |

- **Never impute before train/test split**: if you compute the mean for imputation on the full dataset, test mean information leaks into the training imputer. Always `fit_transform` on train, `transform` on val/test.
- **Missing not at random (MNAR)**: if missingness correlates with the label (e.g., high-risk transactions have incomplete data), missingness itself is a strong signal. Always add `is_missing` indicators for potentially informative missing values.

**Company context:** Stripe (transaction data has many optional fields), Shopify, Uber.

**Common wrong answer:** "I'd impute with the mean." — Missing the GBT native handling, the MNAR case, and the train/test leakage from fitting the imputer on the full dataset.

---

## Normalization

### Q: StandardScaler vs. MinMaxScaler — when do you use each?

**Answer (Staff level):**
- **StandardScaler** (`x - mean) / std`): centers distribution at 0, unit variance. Robust to outliers compared to MinMax because it doesn't use range. Preferred for: linear models, neural networks, SVM (distance-based).
- **MinMaxScaler** (`x - min) / (max - min`): maps to [0, 1]. Sensitive to outliers (one extreme value compresses all others). Preferred for: image pixel values (known range [0, 255]), algorithms requiring bounded inputs.
- **RobustScaler** (`x - median) / IQR`): uses median and interquartile range — outlier-resistant version of StandardScaler. Best when outliers are expected (transaction amounts, social engagement counts).
- **Tree models**: **no normalization needed**. Trees split on thresholds — monotone transforms don't change split decisions. Normalization adds compute with zero benefit for GBT/RF.

**Company context:** Stripe, Shopify (feature preprocessing pipelines).

**Common wrong answer:** "I always normalize features before training." — For tree models, this is unnecessary. State explicitly that normalization applies to linear/distance-based/neural net models, not tree models.

---

## Temporal Features

### Q: What features do you extract from a timestamp for a model predicting user behavior?

**Answer (Staff level):**
- **Cyclical encoding** for periodic features (hour of day, day of week, month): use `sin(2π·x/period)` and `cos(2π·x/period)` as two features. A single integer (hour=23) has no proximity to hour=0, but sin/cos correctly places them adjacent on the unit circle.
- **Recency**: `days_since_last_event` — captures activity freshness. Log-transform if right-skewed.
- **Velocity**: count of events in rolling windows (1h, 24h, 7d). Critical for fraud (spikes) and engagement models.
- **Seasonality indicators**: is_weekend, is_holiday, is_Q4 (for commerce). Learned from data or domain-injected.
- **Account age**: `log(days_since_account_creation)` — captures maturity curve.
- **Trend features**: slope of a metric over past 30d vs. 90d (revenue_trend). More informative than the absolute level.

**Company context:** Stripe (transaction time patterns), Shopify (merchant activity), Uber (ride timing), Reddit (posting patterns).

**Common wrong answer:** "I'd use the raw timestamp." — Raw timestamp is a non-periodic integer with no relationship encoding. Cyclical encoding, recency, and velocity are the production standard.
