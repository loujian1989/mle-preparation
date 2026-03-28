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
