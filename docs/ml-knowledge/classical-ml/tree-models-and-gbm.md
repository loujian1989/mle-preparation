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

**Setup: predicting house prices, 3 training houses**

```
F₀ = $300k (initialized to the mean — everyone gets the same prediction)

House A (large, good location):  true=$450k  prediction=$300k  → need +$150k
House B (small, bad location):   true=$200k  prediction=$300k  → need -$100k
House C (medium, good location): true=$380k  prediction=$300k  → need  +$80k
```

**Standard gradient descent (neural net)**: you have weights `[w_size, w_location]`. Gradient descent updates those weights directly. The function changes because the weights change.

**Gradient boosting**: there are no weights to update. Instead, ask: *"what correction function, added on top of $300k, would reduce the loss?"* The answer is the residuals `[+150, -100, +80]`. Fit a tree to those:

```
Tree 1 learns:  large + good location → predict +$120k correction
                small + bad location  → predict  -$90k correction
                medium + good         → predict  +$70k correction

F₁ = F₀ + 0.1 × Tree₁

House A: $300k + 0.1×$120k = $312k   (true=$450k — closer)
House B: $300k + 0.1×(-$90k) = $291k (true=$200k — closer)
House C: $300k + 0.1×$70k = $307k    (true=$380k — closer)
```

New residuals: `[+138, -91, +73]`. Fit Tree 2 to these. Repeat.

| | Standard GD | Gradient Boosting |
|---|---|---|
| What you optimize | Parameters θ ∈ ℝⁿ | Prediction function F: X → ℝ |
| One step | θ := θ - η·∇L | F := F + η·(new tree) |
| "Gradient" | ∂L/∂θ — direction in parameter space | residuals — direction in prediction space |
| Model capacity | Fixed (same network) | Grows — one tree added per step |

**Key insight**: the "gradient" is a vector of per-sample corrections `[+150, -100, +80]`. You can't subtract it directly — that would memorize training data. Instead, fit a tree that *generalizes* the correction to unseen houses. That's the step in function space.

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

## Hyperparameter Interactions (Deep Dive)

### `n_estimators` × `learning_rate`: The Coupled Pair

These two are mathematically linked. Total movement through function space after M steps:

```
ΔF = η × h₁ + η × h₂ + ... + η × hₘ = η × Σhₘ
```

Halving `η` and doubling `M` gives roughly the same total correction — but a smoother path:

```
High η, few trees:   coarse steps → overshoots loss minima → high variance
Low η, many trees:   fine steps   → finds flatter minima   → better generalization
```

`n_estimators` is not tuned independently. Tune it jointly with `learning_rate` via early stopping:

```
Step 1: Fix lr=0.1, find n_estimators via early_stopping_rounds=50
Step 2: Optionally lr=0.05, re-run early stopping → n_estimators roughly doubles
Step 3: Then tune max_depth, min_child_weight, subsample
```

### `max_depth`: Interaction Complexity Budget

Each tree of depth `d` can model interactions between at most `d` features — one feature per split level:

```
depth=1 (stumps):  main effects only — additive model
depth=3:           up to 3-way interactions
depth=6:           up to 6-way interactions
depth=∞:           memorizes training data
```

For fraud/ranking: depth 4–6. Most predictive signals are pairwise or three-way (amount × country, time × device). Deeper trees model noise.

**LightGBM `num_leaves` vs. `max_depth`:**
```
XGBoost depth=6:     up to 64 leaves, uniformly distributed
LightGBM leaves=64:  64 leaves, concentrated where gain is highest (leaf-wise)
```

LightGBM spends its leaf budget on the most informative branches — more expressive for the same budget. Always cap `num_leaves` to prevent overfitting on small datasets.

### `subsample` and `colsample`: Stochastic Gradient Boosting

Each tree sees a random subset of rows (`subsample`) and features (`colsample_bytree`):

```
subsample=0.8, colsample=0.8:
  → each tree sees ~64% of feature-row combinations
  → trees are less correlated → ensemble variance drops
  → acts as implicit regularization without reducing model capacity
```

**Interaction with `max_depth`**: deep trees (depth 6+) need lower `subsample` (0.6–0.7). Shallow trees (depth 3–4) are more robust — `subsample=0.9` is fine.

### `min_child_weight`: Leaf-Level Regularization

Controls minimum sum of Hessians required to form a leaf node:

```
XGBoost: min_child_weight = min Σ H in a leaf
         For log-loss: H = p(1-p) ≈ 0.25 per sample near p=0.5
         min_child_weight=10 → requires ~40 samples per leaf
```

Most impactful on imbalanced datasets — minority class leaves naturally have small Hessian sums and will be pruned aggressively. Increase `min_child_weight` to prevent splits on tiny minority-class nodes.

### Early Stopping

```python
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          early_stopping_rounds=50)
```

`early_stopping_rounds=50`: stop if validation metric doesn't improve for 50 consecutive trees. Implicitly finds optimal `n_estimators` for the given `learning_rate`.

**Rule**: set `early_stopping_rounds ≈ 10% of expected n_estimators`. Too small → stops at a local plateau. Too large → wastes compute.

### Tuning Priority

| Priority | Parameter | Reason |
|---|---|---|
| 1 | `learning_rate` + `n_estimators` (early stopping) | Coupled; determines overall capacity |
| 2 | `max_depth` / `num_leaves` | Controls interaction complexity |
| 3 | `min_child_weight` / `min_samples_leaf` | Leaf-level regularization |
| 4 | `subsample`, `colsample_bytree` | Variance reduction; tune after structure is fixed |
| 5 | `reg_alpha` (L1), `reg_lambda` (L2) | Fine-grained regularization; usually minor gains |

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

## SHAP Values (Deep Dive)

### The Problem With Split-Gain Importance

Split-gain importance sums impurity reduction across all splits on a feature:

```
importance(age) = Σ (impurity_before - impurity_after) at all splits on age
```

Two failure modes:

**1. High-cardinality bias**: a feature with 1000 unique values has more possible split points than a binary feature → more splits → more accumulated impurity reduction → artificially inflated importance. A random ID column can appear "important."

**2. No local explanation**: one number per feature, averaged over all samples. Tells you nothing about why a specific prediction was made.

### The Game Theory Foundation

Features are **players** in a cooperative game. The **payout** is the model's prediction minus the baseline. SHAP asks: how much did each player contribute?

The Shapley value = average marginal contribution of feature `i` across all possible orderings of features being added:

```
φᵢ = Σ_{S ⊆ F\{i}} [|S|!(|F|-|S|-1)! / |F|!] · [f(S∪{i}) - f(S)]
```

- `S` = subset of other features already in the coalition
- `f(S∪{i}) - f(S)` = marginal contribution of adding feature `i` to coalition `S`
- The weight averages over all orderings — every feature gets equal treatment

**Concrete example** (features: amount, country, hour; prediction = fraud probability):

```
Coalition {country, hour}          → f = 0.60
Coalition {country, hour, amount}  → f = 0.92
Marginal contribution of amount in this coalition = +0.32

Average across all 6 orderings of {amount, country, hour}
→ φ(amount) = weighted average ≈ +0.40
```

### The Additive Property

```
f(x) = base_value + φ₁ + φ₂ + ... + φₙ
```

The sum of all SHAP values equals the prediction minus the baseline. Every unit of prediction is fully accounted for — no residual.

**Example** (fraud model, prediction = 0.92, baseline = 0.05):

```
base_value:              0.05
+ φ(amount):            +0.40  ← large, unusual amount
+ φ(country):           +0.30  ← high-risk country
+ φ(hour):              +0.15  ← 3am transaction
+ φ(device_age):        +0.02  ← minor signal
= prediction:            0.92
```

Fully explainable to a compliance officer.

### Why TreeExplainer Is Efficient

Naive Shapley: enumerate all `2^n` feature subsets. For 100 features: infeasible.

TreeExplainer exploits the tree's split structure. Each path from root to leaf already encodes which features interacted. TreeExplainer reads these paths directly, computing exact Shapley values in `O(T × D × 2^D)` — at depth 6, `2^6 = 64` operations per tree.

### Global vs. Local Importance

| Type | How | Use case |
|---|---|---|
| Local SHAP | One SHAP vector per prediction | Explain a single fraud alert |
| Global importance | `mean(\|φᵢ\|)` over all samples | Feature selection, model audit |
| SHAP dependence plot | `φᵢ` vs. feature value | Detect nonlinear relationships |
| SHAP interaction values | `φᵢⱼ` — joint effect of feature pair | Find feature interactions |

**Global importance advantage**: `mean(|φᵢ|)` weights by actual prediction impact, not split count. A high-cardinality ID column used for many small splits → near-zero mean |SHAP| → correctly flagged as unimportant.

### Summary

| Concept | Intuition |
|---|---|
| Split-gain bias | Rewards features with many split opportunities, not actual prediction impact |
| Shapley value | Average marginal contribution across all feature orderings — fair credit |
| Additivity | SHAP values sum exactly to prediction − baseline — fully accountable |
| TreeExplainer | Exploits tree path structure; exact in O(T × D × 2^D) |
| Local explanation | Per-prediction SHAP vector — explains individual decisions |
| Global importance | mean(\|φᵢ\|) — unbiased aggregate importance |

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

## When Trees Beat Neural Nets (Deep Dive)

### Inductive Bias: What Each Model Gets for Free

Every model has built-in assumptions about data structure. The question is whether they match.

**Trees**: axis-aligned splits — each decision is "feature X > threshold."
**Neural nets**: smooth, continuous functions via weighted sums + activations.

Tabular data is full of threshold effects — exactly what trees model natively:

```
Fraud detection:
  amount > $10,000 AND country ∈ {high-risk} → flag
  Trees: one split on amount, one split on country → done
  Neural net: must approximate this sharp boundary
              with many neurons → needs more data, more tuning
```

### The Irrelevant Feature Problem

You have 200 features. 20 are predictive. 180 are noise.

**Trees**: at each node, evaluate all features and split on the best one. Irrelevant features are never split on — automatically ignored with no explicit feature selection.

**Neural nets**: every input connects to every hidden neuron. All 200 features contribute to every activation:

```
Neural net with 200 inputs, 128 hidden units:
  200 × 128 = 25,600 weights in layer 1
  180 × 128 = 23,040 of those weights should ideally → 0
  Requires heavy L1/dropout to suppress noise features
  On small data: not enough signal to reliably zero them out
```

Trees need zero regularization to ignore irrelevant features — it falls out of the split criterion naturally.

### The Rotation Invariance Problem

Neural nets learn linear combinations of features in the first layer:

```
h = σ(w₁·age + w₂·income + w₃·credit_score + ...)
```

This makes the representation **rotation-invariant** — rotating the feature space gives an equivalent model. But tabular features are NOT interchangeable. `age` and `income` have different units, distributions, and semantics. Mixing them linearly forces the network to first learn to "un-mix" them.

Trees make decisions on one feature at a time — they respect feature identity by design:

```
Tree split: income > $50,000          ← pure, interpretable threshold
Neural net: 0.3·income + 0.7·age > h  ← mixed, requires learning to separate
```

On tabular data with heterogeneous features, the axis-aligned constraint is an advantage, not a limitation.

### Feature Heterogeneity: What Mixed Types Actually Cost

A real tabular dataset might have: continuous, binary, ordinal, and high-cardinality categoricals.

**Trees**: split criterion works identically for all types. No preprocessing beyond label encoding.

**Neural nets**: each type needs different treatment:
```
Continuous:           normalize to zero mean, unit variance
Binary:               fine as-is or embed
Ordinal:              normalize or embed
High-cardinality cat: embedding layer (adds parameters, needs data to train)
```

Each preprocessing choice is a hyperparameter. Each embedding layer needs sufficient data. On a 100k-row dataset with 50 mixed-type features, this overhead hurts generalization.

### The Data Size Threshold

```
Generalization gap ≈ model_complexity / n_samples

LightGBM on 50k rows:   manageable complexity → strong generalization
Neural net on 50k rows: millions of parameters → large gap without heavy regularization
```

Empirical crossover points:
```
< 100k rows:   LightGBM almost always wins
100k–1M rows:  LightGBM wins with default tuning; NNs can win with heavy tuning
> 10M rows:    NNs start to win — capacity advantage emerges
```

Gorishniy et al. (2021): FT-Transformer and ResNet variants match LightGBM on some datasets but require 3–5× more tuning effort and still lose on datasets with many irrelevant features.

### When Neural Nets Win on Tabular

| Scenario | Why NNs win |
|---|---|
| High-cardinality embeddings | Entity embeddings (user ID, item ID) capture latent structure splits can't |
| Sequential/temporal patterns | LSTMs, Transformers model time dependencies; trees see only snapshots |
| Transfer learning available | Pre-trained tabular transformers can win with near-zero data |
| Very large scale (>10M rows) | Capacity advantage; NNs use all data efficiently via mini-batches |
| Dense cross-feature interactions | If nearly all features interact, smooth functions beat axis-aligned splits |

**The embedding case is the most common exception in production**:

```
Ranking model (ads, feeds):
  User ID → 256-dim embedding  (learned from billions of interactions)
  Item ID → 256-dim embedding

  Trees can't learn these — no split on a 256-dim vector is meaningful
  → Neural net or DLRM wins here

  But: downstream ranker that takes frozen embeddings as input?
  → LightGBM on top of frozen embeddings often beats a fine-tuned NN
```

### Summary

| Advantage | Trees | Neural Nets |
|---|---|---|
| Irrelevant features | Ignored automatically | Must regularize away |
| Threshold effects | Native (axis-aligned splits) | Must approximate with many neurons |
| Mixed feature types | No preprocessing needed | Normalization + embeddings per type |
| Small-medium data (<1M rows) | Strong generalization | Needs heavy tuning |
| Rotation invariance | Respected (one feature at a time) | Violated (linear mixing in layer 1) |
| High-cardinality embeddings | Can't learn | Native |
| Temporal/sequential patterns | No | Native (LSTM, Transformer) |
| Very large data (>10M rows) | Hits capacity ceiling | Scales better |

**Decision rule**: start with LightGBM. Switch to neural net only if: learned embeddings needed, temporal structure present, or >10M rows with compute budget to tune.

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

---

## XGBoost vs. LightGBM (Deep Dive)

### Level-wise vs. Leaf-wise: The Core Difference

**Level-wise (XGBoost)** — breadth-first. Split ALL nodes at depth `d` before going to depth `d+1`:

```
Depth 0:  [all 1000 samples]
           split on: amount (gain=100)

Depth 1:  [high amount: 400]  [low amount: 600]
           split on: country   split on: hour
           (gain=80)           (gain=12)   ← forced to split even though signal is weak

Depth 2:  [4 nodes, all split regardless of gain]
```

At depth 3 you always get exactly `2³ = 8` leaves — even if 6 of them are in a low-signal region.

**Leaf-wise (LightGBM)** — best-first. Always split the single leaf with highest gain:

```
Start:    [all 1000 samples]
Split 1:  amount (gain=100)  → 2 leaves
Split 2:  country on high-amount leaf (gain=80)  → 3 leaves
Split 3:  country on low-amount leaf  (gain=60)  → 4 leaves
Split 4:  hour on best remaining leaf (gain=45)  → 5 leaves
```

Same number of splits — but ALL of them went where the signal was highest. No wasted splits on low-gain branches.

**Same leaf budget, different shapes:**
```
XGBoost depth=3:       balanced tree, 8 leaves, uniform depth
LightGBM num_leaves=8: unbalanced tree, 8 leaves, concentrated on high-gain branches
```

LightGBM extracts more signal per leaf. But: without `num_leaves` constraints, leaf-wise keeps chasing the highest-gain leaf indefinitely → overfits on small datasets. Always set `num_leaves` explicitly.

### Histogram Binning: Why LightGBM Is 5–10× Faster

**XGBoost pre-sort**: to find the best split on feature `amount` across N samples:
1. Sort all N samples by value: `O(N log N)`
2. Scan sorted list for best threshold: `O(N)`

**LightGBM histogram**: pre-bucket values into B=255 bins once before training:
```
amount values: [1.2, 45.3, 2.1, 999.0, ...]
→ bin indices: [  0,   12,   0,   254, ...]
```

Build a histogram of gradients per bin, then scan B=255 bins instead of N=1M samples: `O(N)` build + `O(B)` scan.

**Bonus — histogram subtraction**: parent histogram − left child histogram = right child histogram. Half the histogram-building work is free for every split.

### GOSS: Smarter Row Sampling

Standard `subsample=0.8` randomly discards 20% of rows. GOSS is smarter:

```
Large gradient samples = model is wrong on them = informative
Small gradient samples = model already fits them = less informative
```

GOSS keeps all top-`a%` large-gradient samples + random `b%` of small-gradient samples (upweighted by `(1-a)/b` to correct for bias). Result: same sample count as random subsampling, but focused on hard cases → better gradient signal per sample.

### EFB: Handling Sparse/High-Cardinality Features

One-hot encoding 1000 categories → 1000 binary features, 99.9% zeros, mutually exclusive. EFB bundles them:

```
[country_US=1, country_UK=0, country_DE=0]
[country_US=0, country_UK=1, country_DE=0]  →  encoded as [0, 1, 2] in one feature
[country_US=0, country_UK=0, country_DE=1]

1000 one-hot features → ~50 bundles → histogram over 50 features instead of 1000
```

XGBoost has no equivalent — it builds a full histogram for each of the 1000 features separately.

### When to Use Each

| Scenario | Choice | Reason |
|---|---|---|
| Large dataset (>500k rows) | LightGBM | Histogram + GOSS = much faster |
| Small dataset (<50k rows) | XGBoost | Leaf-wise overfits without careful tuning |
| Sparse/high-cardinality categoricals | LightGBM | EFB + native categorical support |
| Need stable defaults | XGBoost | Level-wise is less sensitive to depth choice |

**Overfitting guard for LightGBM on small data**: always set `num_leaves < 2^(max_depth)` and `min_child_samples ≥ 20`.

### Summary

| Concept | XGBoost | LightGBM |
|---|---|---|
| Tree growth | Level-wise (BFS) — uniform depth | Leaf-wise (best-first) — concentrated splits |
| Split finding | Pre-sort O(N log N) | Histogram O(N) build + O(B) scan |
| Row sampling | Random subsampling | GOSS — prioritizes large-gradient samples |
| Sparse features | Histogram per feature | EFB bundles mutually exclusive features |
| Speed | Baseline | 5–10× faster on large datasets |
| Overfit risk | Lower | Higher on small data — needs explicit caps |
