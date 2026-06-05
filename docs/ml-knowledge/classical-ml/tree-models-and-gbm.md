# Tree Models & Gradient Boosting — ML Knowledge Q&A

P1: Stripe, Reddit, Shopify, Uber.

---

## GBT vs. Random Forest

### Q: Gradient Boosting vs. Random Forest — what are the key differences, and when does each win?

**Answer (Staff level):**

| | **Gradient Boosting (GBT)** | **Random Forest (RF)** |
|---|---|---|
| Ensemble method | Sequential (each tree corrects residuals of prior) | Parallel (independent trees, majority vote) |
| Bias-variance | Low bias, can overfit (tuning required) | Higher bias, lower variance (bagging reduces variance) |
| Training speed | Slower (sequential) | Faster (parallelizable) |
| Interpretability | Similar (feature importance, SHAP) | Similar |
| Hyperparameter sensitivity | High (lr, n_estimators, max_depth, subsample) | Low (mostly n_estimators, max_features) |
| Missing values | XGBoost/LightGBM handle natively | sklearn RF requires imputation |
| Best for | Tabular data, Kaggle, production fraud/ranking | Quick baseline, noisy data, out-of-bag eval |

- **GBT wins when**: you have clean, well-prepared tabular features and enough time to tune. Lower final error than RF on most structured datasets.
- **RF wins when**: you need a quick, robust baseline with minimal tuning; data is noisy (GBT will memorize noise); you need out-of-bag uncertainty estimates.

**Company context:** Stripe (fraud — GBT default), Reddit (ranking — LightGBM), Shopify (churn — GBT), Uber (ETA — GBT ensemble).

**Common wrong answer:** "I'd always use GBT because it's more accurate." — RF is a better starting point if you need rapid prototyping or have noisy labels (GBT overfits to label noise more aggressively).

---

## GBT Internals

### Q: How does gradient boosting work? Why does it minimize an arbitrary loss function?

**Answer (Staff level):**
- GBT is a functional gradient descent algorithm. Each tree is fit to the **negative gradient** of the loss with respect to the current ensemble predictions.
- For loss `L(y, F(x))`:
  1. Initialize `F_0(x) = argmin_γ Σ L(y_i, γ)` (constant).
  2. At each step m: compute pseudo-residuals `r_im = -∂L/∂F(x_i)|_{F=F_{m-1}}`.
  3. Fit a new tree `h_m(x)` to the residuals.
  4. Line search: `F_m = F_{m-1} + η · h_m`.
- **Why arbitrary loss**: the tree is always fit to residuals (gradients), regardless of what the loss is. Log-loss → residuals are `y - p`; regression L2 → residuals are `y - F(x)`. Just need a differentiable loss.
- **Second-order (XGBoost)**: uses both gradient (g) and Hessian (h) of the loss to compute optimal leaf values analytically: `w_j* = -Σg / (Σh + λ)`. More accurate leaf values, faster convergence.

**Company context:** Stripe, Reddit. Expected to articulate the gradient view, not just "it builds trees sequentially."

**Common wrong answer:** "Each tree is fit to the error of the previous tree." — Technically: it's fit to the gradient of the loss, which equals the residuals only for L2 loss. For log-loss, the "residuals" are `y - σ(F(x))`, which is the gradient.

---

## Gradient Boosting Internals (Deep Dive)

### The Core Analogy: Gradient Descent in Function Space

Standard gradient descent moves **model parameters** in the direction of steepest loss reduction:
```
θ_new = θ_old - η · ∂L/∂θ
```

Gradient boosting does the same thing, but instead of moving parameters, it moves the **prediction function** itself:
```
F_m(x) = F_{m-1}(x) + η · h_m(x)
```

Each new tree `h_m` is a **step in function space** — it directly adds a correction to the current predictions. You're doing gradient descent where the "parameter" is the entire prediction function.

### Why Fit to Negative Gradients?

The gradient of the loss tells you how each prediction needs to change to reduce loss:

```
r_i = -∂L/∂F(x_i)
```

If `r_i > 0`, the prediction needs to go up. If `r_i < 0`, it needs to go down.

**The tree's job**: learn a mapping `x → r` so it can compute these corrections for any input, including unseen data. That's why you fit the tree to residuals rather than directly subtracting the gradient — you need a generalizable function, not just per-sample corrections.

### Concrete: L2 Loss vs. Log-Loss

**L2 regression loss**: `L = ½(y - F(x))²`

```
∂L/∂F = -(y - F(x))
r_i   = y_i - F(x_i)   ← literal residual
```

The gradient IS the residual. This is why people say "fit to residuals" — it's only exactly true for L2.

**Log-loss (binary cross-entropy)**: `L = -y·log(p) - (1-y)·log(1-p)`, where `p = σ(F(x))`

```
∂L/∂F = p - y   =  σ(F(x)) - y
r_i   = y_i - σ(F(x_i))   ← residual in probability space
```

The gradient is `y - p` — the error between the true label and the predicted probability. The tree is correcting the **log-odds** `F(x)`, not the probability directly.

This is why gradient boosting handles arbitrary losses: **any differentiable loss gives you a gradient, and the gradient defines what the next tree should fit to.**

| Loss | Pseudo-residual `r_i` | Intuition |
|---|---|---|
| L2 | `y - F(x)` | Raw prediction error |
| Log-loss | `y - σ(F(x))` | Probability error |
| MAE (L1) | `sign(y - F(x))` | Direction only, not magnitude |
| Huber | L2 residual if small, L1 sign if large | Robust to outliers |

### Why Not Just Fit One Big Tree?

```
1 deep tree:
  → Low bias (fits training data well)
  → High variance (overfits, generalizes poorly)

100 shallow trees (depth 3):
  → Each tree: high bias, low variance
  → Ensemble: low bias (accumulated corrections), low variance (averaging effect)
```

The learning rate `η` controls step size. Small `η` = many small corrections = smoother path through function space = better generalization.

### XGBoost Second-Order: Why It Helps

Standard GBT uses first-order (gradient only). XGBoost uses a second-order Taylor expansion of the loss:

```
L(y, F + h) ≈ L(y, F) + g·h + ½·H·h²

where:
  g = ∂L/∂F       (gradient)
  H = ∂²L/∂F²     (Hessian — curvature)
```

This lets XGBoost solve for the **optimal leaf value analytically**:

```
Optimal leaf weight: w* = -Σg / (Σh + λ)
```

The Hessian `H` tells you how curved the loss is. High curvature → smaller step. Low curvature → bigger step. This is Newton's method applied to each leaf.

**Example with log-loss:**
```
g = p - y          (gradient)
H = p(1-p)         (Hessian — variance of Bernoulli)

w* = -Σ(p_i - y_i) / (Σ p_i(1-p_i) + λ)
```

For confident predictions (p ≈ 0 or 1): `H ≈ 0` → denominator large → small step (don't over-correct). For uncertain predictions (p ≈ 0.5): `H ≈ 0.25` → normal step size.

### Summary

| Concept | Intuition |
|---|---|
| Pseudo-residuals | Direction each prediction must move to reduce loss |
| Fit tree to residuals | Learn a generalizable correction function, not just per-sample fixes |
| L2 residuals = gradients | Only true for L2 — for other losses, residuals are loss-specific |
| Arbitrary loss support | Any differentiable loss → gradient → tree target |
| Shallow trees + many iterations | High-bias low-variance learners accumulate into low-bias ensemble |
| Learning rate | Step size in function space — smaller = smoother path = better generalization |
| XGBoost Hessian | Curvature-aware step size → faster convergence, analytically optimal leaf values |

---

## Hyperparameter Interactions

### Q: How do `n_estimators`, `learning_rate`, and `max_depth` interact?

**Answer (Staff level):**
- **`n_estimators` × `learning_rate`**: these are coupled. Lower `learning_rate` requires more trees to achieve the same training loss reduction. As a rule of thumb, halve `learning_rate` and double `n_estimators` for similar performance with better generalization (more regularized path).
- **`max_depth`**: controls each tree's complexity. Shallow trees (depth 3–5) have high bias individually but work well with many iterations. Deep trees (depth 7+) are "strong learners" — you need fewer but risk overfitting. For tabular fraud/churn: depth 4–6 is typical.
- **`subsample`** (row subsampling) and **`colsample_bytree`** (column subsampling): stochastic gradient boosting. Each tree sees a random subset of rows/columns. Reduces variance, speeds training. Typical: 0.7–0.9.
- **Overfitting indicators**: training loss < validation loss by a large gap; validation AUPRC plateaus while training continues improving. Fix: reduce `max_depth`, add `min_samples_leaf`, reduce `n_estimators`, or use `early_stopping_rounds`.

**Company context:** Stripe, Shopify (model tuning round).

**Common wrong answer:** "I'd use grid search over all parameters." — Grid search is exponential. Staff answer prioritizes: first n_estimators + learning_rate (coupled), then max_depth, then subsample.

---

## SHAP Values

### Q: What are SHAP values? Why are they preferred over split-gain feature importance?

**Answer (Staff level):**
- **Split-gain feature importance** (default in sklearn): sums impurity reduction at each split involving a feature. Biased toward high-cardinality features (more possible splits), doesn't reflect actual prediction contribution for individual samples.
- **SHAP (Shapley Additive Explanations)**: based on Shapley values from game theory. Feature i's SHAP value = its marginal contribution averaged over all subsets of features (all possible "coalitions").
- **Properties**:
  - **Local**: each prediction has its own SHAP vector; sum of SHAP values = prediction − baseline.
  - **Consistent**: if a feature's marginal contribution increases, its SHAP value can only increase or stay the same.
  - **Global importance**: take mean |SHAP| over training set — fair summary importance without high-cardinality bias.
- **TreeExplainer (SHAP library)**: exact Shapley values for tree models in O(T × D) per sample (vs. exponential for the naive formulation). `shap.TreeExplainer(model).shap_values(X)`.
- **Use cases**: individual prediction explanation (fraud alert reasoning), model debugging (find systematic errors), feature selection (eliminate features with near-zero mean |SHAP|).

**Company context:** Stripe, Shopify (take-home template uses SHAP), OpenAI (model audit).

**Common wrong answer:** "Feature importance shows which features matter." — Split-gain importance is biased and doesn't explain individual predictions. SHAP provides both local (per-prediction) and global (aggregate) explanations with consistency guarantees.

---

## When Trees Beat Neural Nets

### Q: When do gradient boosted trees outperform neural networks on tabular data, and why?

**Answer (Staff level):**
- **GBT typically wins for**:
  - Moderate-sized tabular datasets (< 1M rows) with mixed feature types
  - Features are heterogeneous (mix of continuous, ordinal, high-cardinality categorical)
  - Limited compute budget (faster training, no GPU required)
  - Requires explainability (SHAP is simpler for trees)
  - Label noise is low (GBT overfits but recovers with regularization)
- **Neural nets win when**:
  - Very large datasets (>10M rows) where deep learning's capacity advantage emerges
  - Features have complex spatial/temporal structure (time-series, images, text)
  - Pre-trained representations are available (transfer learning)
  - Embeddings for high-cardinality categoricals matter more than tree splits
- **Gorishniy et al. (2021) "Revisiting Deep Learning Models for Tabular Data"**: FT-Transformer and ResNet variants can match GBT, but require more tuning. GBT (specifically LightGBM) wins on standard benchmarks with default settings.
- **Practical rule**: start with LightGBM. Switch to neural net only if: you have embedding needs, temporal patterns, or very large scale.

**Company context:** Stripe, Reddit (model selection reasoning), Shopify.

**Common wrong answer:** "Neural networks are always better with enough data." — False for tabular data. The inductive biases of trees (axis-aligned splits, additive structure) are often well-matched to tabular data structure.

---

## XGBoost vs. LightGBM

### Q: Key algorithmic differences between XGBoost and LightGBM?

**Answer (Staff level):**

| | **XGBoost** | **LightGBM** |
|---|---|---|
| Tree growing | Level-wise (breadth-first) | Leaf-wise (best-first: splits the leaf with max gain) |
| Speed | Slower for large datasets | 5–10× faster; smaller memory footprint |
| Accuracy | Slightly more stable for small datasets | Can overfit more aggressively (leaf-wise is greedy) |
| Missing values | Native `NaN` handling (learns split direction) | Native `NaN` handling |
| Categorical features | Must encode before training | `categorical_feature` param — auto-discretizes |
| Histogram binning | Pre-sort algorithm (original) / histogram (XGB ≥1.0) | Histogram from the start — primary speed advantage |
| Distributed training | XGBoost has mature distributed support | Also distributed, but LightGBM's gradient compression reduces communication overhead |

- **LightGBM leaf-wise growth**: maximizes gain per split without constraint on depth. Fewer splits needed for same training loss. BUT: must constrain `max_depth` or `num_leaves` to prevent overfitting on small datasets.
- **Default choice**: LightGBM for speed and large datasets; XGBoost when you want more stable defaults or have a well-tested XGBoost pipeline.

**Company context:** Reddit (LightGBM is the ranking model standard), Stripe (XGBoost historically, LightGBM in modern stacks).

**Common wrong answer:** "XGBoost uses second-order gradients, LightGBM doesn't." — Both use second-order gradients (Hessian). The key difference is tree-growing strategy (level-wise vs. leaf-wise) and histogram binning for speed.
