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

---

## Leakage (Deep Dive)

### Why Leakage Is Insidious — It Always Improves Offline Metrics

Leakage makes a model look better in evaluation while being useless or harmful in production:

```
Churn model with leakage feature "support_ticket_in_next_30_days":
  Offline AUPRC: 0.91  ← suspiciously high
  Production:    feature doesn't exist at prediction time
                 model falls back to random guessing when deployed
```

The signal is real — it just comes from the future. The model learns it perfectly, passes all offline tests, and fails at serving time.

### The Temporal Leakage Pattern — Most Common in Practice

The root cause: aggregating data without respecting the prediction cutoff:

```
Prediction task: will this user churn in the next 30 days?
Observation cutoff: day T (when prediction is made)

WRONG — temporal leakage:
  Feature: avg_session_length_last_90_days
  If "last 90 days" is computed relative to label time (day T+30),
  it includes session data from days T to T+30 — after prediction time.

RIGHT:
  Feature: avg_session_length_from_day(T-90)_to_day(T)
  Only uses data available when prediction is made.
```

In code, this typically happens when joins are written carelessly:

```python
# WRONG: computes aggregate over all time including future
features = events.groupby('user_id').agg({'session_length': 'mean'})

# RIGHT: filter to observation window first
features = events[events['event_date'] <= observation_date] \
               .groupby('user_id').agg({'session_length': 'mean'})
```

### Detection Protocol

1. **Implausibly high performance**: AUPRC > 0.90 on a hard problem → investigate. Real fraud models rarely exceed 0.80.
2. **Feature correlation with label**: correlation(feature, label) > 0.5 → trace the feature's derivation.
3. **Feature importance suspicion**: a feature you didn't expect to be strong is the #1 feature. Investigate its construction.
4. **Sanity check**: remove the suspicious feature and re-evaluate. If AUPRC drops dramatically, it was doing the work.
5. **Point-in-time validation**: rebuild the feature at several historical dates using only data available at that date. If feature values differ from the "leaky" version, leakage confirmed.

### Row-Level Leakage — The Preprocessing Trap

```python
# WRONG: fit scaler on full dataset before split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # uses test set statistics!
X_train, X_test = train_test_split(X_scaled)

# RIGHT: fit only on training data
X_train, X_test = train_test_split(X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on train
X_test_scaled = scaler.transform(X_test)          # apply to test
```

The test set's mean and variance influenced the scaler → each test sample is partially described by its own statistics → subtle leakage that inflates performance.

---

## Target Encoding (Deep Dive)

### Why Naive Target Encoding Leaks — The Single-Sample Case

For a category with one training sample:

```
merchant_id = "XYZ123", appears once, label = 1 (churned)
Naive encoding: mean(label for XYZ123) = 1.0/1 = 1.0
```

The model receives `1.0` as the feature value exactly when the label is `1`. For this sample, the feature IS the label. The model learns the trivial mapping `feature=1.0 → label=1`. On new data, `XYZ123` won't have `feature=1.0` unless it also churned.

### K-Fold Target Encoding — How the Out-of-Fold Fix Works

```
Split training data into 5 folds.

For sample i in fold k:
  encoding(i) = mean(label for all samples NOT in fold k with same category)

This means: sample i's encoding never uses its own label.
```

The model trains on encodings that were computed without data leakage. At inference, use the global mean from all training data (all 5 folds combined).

### Smoothing — Preventing Noise From Rare Categories

Without smoothing: a category that appears once with label=1 gets encoding=1.0 — noise masquerading as signal.

Smoothed encoding:

```
enc_k = (n_k · mean_k + m · global_mean) / (n_k + m)

n_k:          count of samples in category k
mean_k:       raw target mean for category k
global_mean:  overall target mean (e.g. 0.05 for 5% churn)
m:            smoothing parameter (typically 10–300)

n_k=1, mean_k=1.0, global_mean=0.05, m=100:
enc_k = (1×1.0 + 100×0.05) / (1+100) = 6.0/101 ≈ 0.059  ← pulled to global mean

n_k=1000, mean_k=0.20, global_mean=0.05, m=100:
enc_k = (1000×0.20 + 100×0.05) / (1000+100) = 205/1100 ≈ 0.186  ← close to category mean
```

Rare categories get pulled toward the global mean. High-frequency categories retain their category-specific mean. `m` controls how many samples you need before trusting the category-specific mean.

---

## Log Transform (Deep Dive)

### When Skewness Actually Hurts — Linear Models vs. Trees

**Linear models** (logistic regression, linear SVM): assume linear relationship between feature and output. For a right-skewed feature:

```
transaction_amount: [1, 5, 10, 50, 500, 10000]
label (fraud):      [0, 0,  0,  0,   1,     1]

Linear model: w × amount + b
  To separate fraud (amount>100) from non-fraud (amount<100):
  w must be large (to create a big decision boundary)
  But then the few extreme values (10000) dominate the loss
  → model is numerically unstable, weights blow up

Log transform: [0, 1.6, 2.3, 3.9, 6.2, 9.2]
  A simple threshold at ~4.6 separates the classes cleanly
  w can be small → stable optimization
```

**Tree models**: split on `amount > 500`. The actual scale of values doesn't matter — only the ordering does. Log transform preserves ordering → same splits → identical model. Zero benefit.

### The Multiplicative Interaction — Help or Hurt?

Two features with a multiplicative relationship:

```
revenue = price × quantity
log(revenue) = log(price) + log(quantity)  ← additive in log space
```

For linear models: instead of needing an interaction term `w_pq × price × quantity`, the model can use two additive terms. Simpler model, easier to regularize.

For tree models: this doesn't matter — trees naturally discover the multiplicative interaction through nested splits (`price > X` then `quantity > Y`).

### The log1p Pattern and Negative Values

```python
# Handles zeros (log(0) = -∞)
np.log1p(x)   # = log(1 + x), defined at x=0, returns 0

# Handles negatives (e.g., P&L, signed returns)
np.sign(x) * np.log1p(np.abs(x))   # symmetric log transform
```

The symmetric log transform preserves the sign while compressing both tails. Useful for financial features where negative values are meaningful (loss-making merchants).

---

## Missing Values (Deep Dive)

### Why MNAR Is the Most Dangerous Pattern

Three types of missingness:

```
MCAR (Missing Completely At Random):
  A sensor randomly glitches. P(missing) independent of everything.
  Simple imputation is fine. No information in missingness.

MAR (Missing At Random):
  Income is more often missing for younger users.
  P(missing | age) is predictable. Missingness is explained by observed data.
  Impute, optionally add is_missing flag.

MNAR (Missing Not At Random):
  High-risk transactions have incomplete merchant data (fraudsters provide less info).
  P(missing) depends on the unobserved true value.
  Missingness IS the signal. Must add is_missing flag.
```

**The MNAR trap**: if you impute MNAR values with the mean and don't add a flag, you've erased the signal. The model sees mean-imputed values for both high-risk (MNAR) and randomly-missing (MCAR) cases — they look identical. The pattern that distinguishes them (missing vs. not) is gone.

```python
# Right way for a potentially MNAR feature
df['amount_missing'] = df['merchant_revenue'].isna().astype(int)
df['merchant_revenue'] = df['merchant_revenue'].fillna(df['merchant_revenue'].median())
# Model now sees both the imputed value AND the missing indicator
```

### XGBoost's Learned Direction — What It Actually Does

XGBoost doesn't just route NaN values to a default branch. It learns which direction (left or right at each split) minimizes loss for NaN values:

```
Split: amount > $500
  Left subtree:  low-amount transactions → mostly legit
  Right subtree: high-amount transactions → more fraud

For missing amount values, XGBoost tries both directions during training:
  Route NaN left → compute gain
  Route NaN right → compute gain
  Pick the direction with higher gain

Result: if NaN amount correlates with fraud, XGBoost learns to route NaN right
        effectively treating "missing amount" as a signal for the fraud branch
```

This is why XGBoost/LightGBM often outperform sklearn's GBT on real data — real data has meaningful missing values, and the learned direction captures the MNAR signal automatically.

---

## Normalization (Deep Dive)

### Why Distance-Based Models Need Normalization — Concrete Math

KNN: distance between sample A `[income=100000, age=30]` and sample B `[income=100001, age=60]`:

```
Without normalization (raw values):
  L2 distance = sqrt((100000-100001)² + (30-60)²)
              = sqrt(1 + 900) ≈ 30
  Age difference of 30 years contributes 900 to distance.
  Income difference of $1 contributes 1 to distance.
  KNN is effectively doing age-only nearest neighbor.

With StandardScaler (both features ~ N(0,1)):
  L2 distance = sqrt((0.001)² + (2.0)²) ≈ 2.0
  Both features now on comparable scale.
```

The model's notion of "nearest" matches the semantics of the problem.

### RobustScaler — When Outliers Are Real

Transaction amounts: `[10, 15, 12, 14, 11, 50000]`

StandardScaler: mean ≈ 8352, std ≈ 20397. Every normal transaction scales to ≈ -0.41. The entire non-outlier distribution is compressed into a tiny range.

RobustScaler: median = 13, IQR = 3. Normal transactions scale to `[-1, 0.67]`. The outlier scales to ≈ 16,662 — still extreme, but the non-outlier distribution is readable.

```
When to use:
  StandardScaler:  no significant outliers (e.g. age, normalized scores)
  RobustScaler:    known outliers that are real (amount, count, duration)
  MinMaxScaler:    bounded-range features with known min/max (pixel values, percentiles)
```

---

## Temporal Features (Deep Dive)

### Why Cyclical Encoding Is Necessary — The Distance Problem

Raw hour encoding: `hour=23` and `hour=0` are 1 apart in real time but 23 apart numerically. A linear model treats them as very different. A distance-based model places them far apart.

Cyclical encoding maps to a unit circle:
```
sin_hour = sin(2π × hour / 24)
cos_hour = cos(2π × hour / 24)

hour=0:  sin=0.00,  cos=1.00
hour=6:  sin=1.00,  cos=0.00
hour=12: sin=0.00,  cos=-1.00
hour=18: sin=-1.00, cos=0.00
hour=23: sin=-0.26, cos=0.97  ← close to hour=0 ✓
```

The Euclidean distance between hour=23 and hour=0 is:
```
sqrt((0.00-(-0.26))² + (1.00-0.97)²) = sqrt(0.068 + 0.001) ≈ 0.26  ← small, correct
```

**Why two features (sin AND cos)?** One feature is ambiguous: `sin(hour=2) = sin(hour=10)`. You need both sin and cos to uniquely identify the position on the circle.

### Velocity Features — The Fraud Detection Workhorse

Velocity = count (or sum) of events in a rolling time window, computed per entity:

```
transactions_1h:   how many transactions by this card in the last 1 hour
transactions_24h:  how many transactions by this card in the last 24 hours
amount_7d:         total spend by this card in the last 7 days
```

These are the most predictive fraud features because:
- Normal behavior: 1-3 transactions per day, $50-200 each
- Card testing: 20+ transactions in 1 hour, all for small amounts
- Account takeover: large transaction spike after months of inactivity

**Velocity ratio** is even more powerful: `transactions_1h / (transactions_24h + 1)`. A ratio near 1.0 means all of today's activity happened in the last hour — strong fraud signal.

**Implementation**: requires a point-in-time feature store to compute at prediction time without leakage. The 1h window must use only events with timestamp < prediction_time.
