# Regularization — ML Knowledge Q&A

P0: Universal. OpenAI, Meta, Stripe, Reddit, Uber, Shopify, Pinterest.

---

## L1 vs. L2

### Q: L1 and L2 regularization — geometric intuition, when to use each?

**Answer (Staff level):**
- **L2 (Ridge)**: penalty = λ·Σ w_i². L2 ball is a sphere. The gradient of the penalty is proportional to w (Δw ∝ w), so large weights shrink faster than small ones. Solution is always dense — all weights are small but non-zero.
- **L1 (Lasso)**: penalty = λ·Σ|w_i|. L1 ball is a diamond (in 2D). The subgradient of the penalty is ±λ regardless of weight magnitude (sign function). For small weights, the penalty pushes them exactly to zero → **sparse solutions**.
- **Geometric intuition**: the optimal solution lies where the contour of the loss function first touches the constraint ball. The L1 ball has corners at the axes — the optimal point often hits a corner where some coordinates = 0. The L2 ball has no corners — the intersection is typically off-axis, so all weights are nonzero.
- **When to use**:

| | **L1** | **L2** |
|---|---|---|
| Goal | Feature selection, sparsity | Stable, dense solutions |
| Separable data | Does NOT prevent divergence | Prevents divergence |
| High-dimensional sparse input | Good (NLP bag-of-words) | Overkill |
| Neural networks | Rarely (unstable) | Common via weight decay |

**Company context:** OpenAI (coding: logistic regression with L2 prevents divergence on separable data — see `logistic_regression_edge_cases.py`), Stripe, Shopify.

**Common wrong answer:** "L1 always outperforms L2 for feature selection." — L1 gives sparsity, not necessarily better features. In practice, Elastic Net (L1 + L2) is more robust than either alone.

---

### Q: Why does L2 (but not L1) prevent weight divergence on linearly separable data?

**Answer (Staff level):**
- On separable data, gradient descent drives weights → ∞ because larger weights push predictions closer to 0/1, always reducing loss. The gradient never reaches zero.
- **L2 gradient contribution**: `∂L/∂w = data_gradient + λ·w`. As w grows, the L2 penalty gradient `λ·w` grows proportionally, eventually balancing the data gradient. An equilibrium `w*` is reached.
- **L1 gradient contribution**: `∂L/∂w = data_gradient ± λ` (constant magnitude). As w grows large, the data gradient dominates the constant ±λ. No equilibrium — weights still diverge.
- **Practical implication**: logistic regression implementations (sklearn `LogisticRegression`) use L2 by default precisely for this reason. If you set `penalty=None` on separable data, the solver will warn of non-convergence.

**Company context:** OpenAI (direct coding + conceptual probe in same round).

**Common wrong answer:** "L1 also prevents divergence by penalizing large weights." — The penalty magnitude doesn't grow with weight magnitude for L1. This is the specific failure mode.

---

## Dropout

### Q: Explain the mechanics of dropout. Why does it fail to apply naively at inference?

**Answer (Staff level):**
- **Training**: for each forward pass, independently zero each activation with probability p (dropout rate). Scale surviving activations by 1/(1-p) to maintain expected value (inverted dropout — standard in PyTorch/TF).
- **Inference**: all activations are kept active. No scaling needed if inverted dropout was used during training.
- **Why it helps**: forces the network to learn redundant representations — no single neuron can memorize a feature because it may be absent. Equivalent to training an exponential ensemble of 2^D sub-networks (Srivastava et al., 2014).
- **Failure modes**:
  - Batch norm + dropout interact poorly: BN estimates batch statistics from the full activation distribution; dropout changes that distribution. BN after dropout sees different variance than BN at inference (where dropout is off). Fix: apply dropout before BN, or use dropout only in fully-connected layers (not conv+BN blocks).
  - Small batch size: with dropout, the effective batch for BN statistics is `(1-p)×batch_size`, which may be too small.

**Company context:** OpenAI, Meta.

**Common wrong answer:** "At inference I apply dropout with p/2." — Wrong. Inference = no dropout (or equivalently, keep probability = 1). The inverted scaling in training handles the expectation correction.

---

## Early Stopping

### Q: How is early stopping a form of regularization?

**Answer (Staff level):**
- Early stopping halts training when validation loss stops improving (patience = K epochs without improvement).
- **Equivalence to L2**: for gradient descent on quadratic loss (simplified), early stopping with T steps produces a solution equivalent to L2 regularization with λ ≈ 1/(ηT) where η is learning rate. Stopping early = implicit shrinkage.
- **In practice**: not exactly equivalent to L2, but has the same effect of keeping weights in a neighborhood of initialization (which are typically small/zero). Weights never grow as large as they would with unlimited training.
- **Implementation**: monitor validation metric (not training loss). Save checkpoint when validation improves; restore best checkpoint after patience is exceeded.
- **Caveat**: for non-convex (neural net) loss, the equivalence breaks down — early stopping is heuristic, but empirically effective.

**Company context:** Universal. Netflix (monitoring / "how do you know when to stop training?").

**Common wrong answer:** "Early stopping just prevents overfitting by stopping training." — The regularization equivalence and the weight-shrinkage interpretation are what elevate this to a Staff answer.

---

## Data Augmentation

### Q: How does data augmentation act as regularization?

**Answer (Staff level):**
- Data augmentation expands the effective training distribution by applying label-preserving transformations (crop, flip, noise injection, mixup). The model must learn invariances to these transformations.
- **Regularization view**: augmentation increases effective dataset size (reduces variance), and the invariance constraint acts like an explicit regularizer — the model cannot overfit to exact pixel values.
- **Mixup** (Zhang et al., 2018): `x_mix = λ·x_i + (1−λ)·x_j`, `y_mix = λ·y_i + (1−λ)·y_j`. Trains on convex combinations of examples and labels. Smooths the decision boundary (implicit L2-type constraint on the Lipschitz constant of the network).
- **For tabular ML (fraud, churn)**: standard image augmentations don't apply. Alternatives: feature noise injection (Gaussian noise on numeric features), SMOTE for class imbalance (synthetic oversampling in feature space), feature dropout (randomly zero out feature values during training).

**Company context:** OpenAI (image/LLM context), Meta (vision-heavy), Stripe/Shopify (tabular augmentation for imbalance).

**Common wrong answer:** "Data augmentation just gives you more training data." — The invariance constraint is the regularization effect; size is secondary.

---

## Weight Decay vs. L2 in Adam

### Q: Weight decay and L2 regularization are equivalent in SGD but not in Adam. Explain.

**Answer (Staff level):**
- **SGD**: L2 adds `λ·w` to the gradient. Update: `w ← w − η·(g + λ·w) = (1 − η·λ)·w − η·g`. This is equivalent to scaling weights before the gradient step — weight decay.
- **Adam**: the gradient `g + λ·w` is fed through Adam's adaptive scaling: `w ← w − η·(g + λ·w) / (√v̂ + ε)`. The weight decay term `λ·w` is scaled by the adaptive step size `1/√v̂`, which varies per-parameter. High-variance parameters (high v̂) get less regularization. This decouples regularization from gradient magnitude in an undesirable way — parameters with large gradients get less L2 penalty.
- **AdamW (Loshchilov & Hutter, 2018)**: apply weight decay directly to weights, outside the adaptive scaling: `w ← (1 − λ)·w − η·g/√v̂`. Regularization is uniform across parameters. Empirically superior to Adam + L2 for transformers.
- **Practical rule**: always use AdamW (not Adam) when applying weight decay. In PyTorch: `torch.optim.AdamW(params, lr=lr, weight_decay=wd)`.

**Company context:** OpenAI, Meta (transformer training).

**Common wrong answer:** "Weight decay and L2 regularization are the same thing." — True for SGD, false for Adam/AdamW. This is a Staff-level discriminator.

---

## L1 vs. L2 (Deep Dive)

### The Geometric Intuition — Why L1 Gives Sparsity

Both regularizers constrain the weights to lie within a ball. The loss contours are ellipsoids. The optimal solution is where the loss contour first touches the constraint ball:

```
L2 (sphere):            L1 (diamond in 2D):
   /---\                      /\
  / opt \   ← off-axis       /  \
  \ (dense)                 / opt \  ← hits corner
   \---/                    \ (sparse!)
                              \  /
                               \/
```

The L2 ball is smooth — the loss contour meets it at a generic curved surface point where both `w₁` and `w₂` are nonzero. The L1 ball has corners at the coordinate axes. The loss contour is likely to hit a corner, where one coordinate is exactly zero.

**Why corners cause sparsity — the subgradient argument:**

For L2: gradient of penalty at `w_i = 0` is `2λw_i = 0`. The penalty provides no force pushing `w_i` to exactly zero.

For L1: subgradient of `|w_i|` at `w_i = 0` is any value in `[-λ, +λ]`. The optimizer can choose the subgradient to exactly cancel the loss gradient, pinning `w_i = 0`. This is the mechanism that achieves exact zeros.

### L2 Divergence Prevention — The Equilibrium Argument

On linearly separable data, logistic regression loss decreases forever as weights grow:

```
As ||w|| → ∞:  σ(w·x) → 0 or 1 for all training points
               log-loss → 0 for all training points
               gradient of loss → 0 but never reaches it
```

**With L2**: total gradient = data gradient + λ·w. As `w` grows, `λ·w` grows proportionally. Eventually `λ·w ≈ -data_gradient` → equilibrium at finite `w*`.

**With L1**: total gradient = data gradient ± λ (constant). As `w` grows large:
```
data_gradient ≈ tiny (predictions saturated)
penalty gradient = ±λ (constant, doesn't grow)
→ data gradient can still dominate → no equilibrium → divergence
```

The constant magnitude of the L1 subgradient is why it fails here. L2's proportional penalty is what creates the restoring force.

### Dropout — The Ensemble Interpretation

A network with N neurons and dropout rate p can be seen as training an ensemble of 2^N sub-networks, one for each possible binary mask:

```
Full network:  h₁, h₂, h₃, h₄, h₅
Dropout mask:  [1,  0,  1,  1,  0]  → this forward pass trains sub-network #37
Next batch:    [1,  1,  0,  1,  1]  → this forward pass trains sub-network #112
```

At inference, instead of explicitly ensembling 2^N networks (infeasible), we use the full network with all weights scaled by (1-p) — this approximates the geometric mean of the 2^N sub-networks' predictions.

**Why this prevents co-adaptation:** if neuron A always fires with neuron B, the network can learn `A AND B → feature`. With dropout, A is sometimes absent, B is sometimes absent — the network can't rely on the joint signal. Each neuron must be individually useful.

**The train-test variance mismatch without inverted dropout:**
```
Training:   each neuron active with probability (1-p)
            expected activation: (1-p) × h
Inference:  all neurons active
            activation: h  ← (1/(1-p))× larger than training

Inverted dropout fix: scale by 1/(1-p) during training
  training activation: h/(1-p) × (1-p) = h  ← same expectation as inference
  inference: no scaling needed
```

### Early Stopping — The L2 Equivalence

For gradient descent on a quadratic loss (linear model):

```
Without regularization, after T steps:
  w_T = (I - (I - ηH)^T) H⁻¹ g

This is equivalent to L2 regularization with λ ≈ 1/(ηT):
  w_L2* = (H + λI)⁻¹ g

As T → ∞: w_T → H⁻¹g (exact solution, full overfit)
As T → 0:  w_T → 0 (stays near initialization)
```

Early stopping is implicit shrinkage — it keeps the solution near the (small) initialization, equivalent to penalizing distance from the origin.

**The practical mechanism for neural nets** (non-convex, not exactly equivalent):
- Weights start small at initialization (near zero)
- Without regularization: weights grow to fit training noise
- With early stopping: weights never leave the neighborhood of initialization → stay small → less overfitting

Monitor validation loss, not training loss. Save the checkpoint at validation minimum. Restore after patience is exceeded.

### Data Augmentation — Invariance as Regularization

Standard framing: augmentation increases dataset size. The deeper mechanism is **invariance injection**:

```
Horizontal flip augmentation:
  The model must predict the same class for img and flip(img)
  → The decision boundary must be symmetric under horizontal flip
  → This is a constraint on the model's function class
  → Equivalent to adding a regularization term: L(f(x)) + L(f(flip(x)))
```

Mixup makes this explicit:

```
x_mix = λ·x_i + (1−λ)·x_j
y_mix = λ·y_i + (1−λ)·y_j

Model must predict y_mix for x_mix.
This forces linear interpolation in the output space along the path between x_i and x_j.
→ Enforces a Lipschitz constraint on the decision boundary.
→ Prevents the model from placing sharp decision boundaries between training examples.
```

**For tabular data** (no natural geometric transformations):
- Feature noise: add `ε ~ N(0, σ²)` to numeric features → forces robustness to small measurement errors
- Feature dropout: randomly zero out feature values → forces the model to not rely on any single feature
- SMOTE: generate synthetic minority class examples by interpolating between existing ones → directly addresses class imbalance without the false negative risk of downsampling

### Summary

| Technique | Mechanism | What it limits |
|---|---|---|
| L2 | Proportional penalty → equilibrium at finite w | Weight magnitude |
| L1 | Constant subgradient → corners of constraint ball | Number of nonzero weights |
| Dropout | Trains exponential ensemble, prevents co-adaptation | Feature co-dependence |
| Early stopping | Keeps weights near (small) initialization | Weight magnitude implicitly |
| Data augmentation | Injects invariances as constraints on function class | Model sensitivity to specified transformations |
| AdamW weight decay | Uniform weight shrinkage independent of gradient variance | Weight magnitude, correctly decoupled from adaptive LR |
