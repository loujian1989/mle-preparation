# Optimization & Training — ML Knowledge Q&A

P0: OpenAI, Meta, Netflix, Roblox.

---

## Adam vs. SGD

### Q: Adam converges faster than SGD, so why does SGD still appear in production training?

**Answer (Staff level):**
- **Adam** uses per-parameter adaptive learning rates (first moment = gradient momentum, second moment = gradient variance). Fast convergence, robust to LR choice, handles sparse gradients well.
- **SGD + momentum** has better-documented **generalization**. Theory (Keskar et al., 2017): Adam converges to sharp minima in the loss landscape; SGD converges to flatter minima with smaller Hessian eigenvalues → wider optima generalize better to OOD data.
- In practice: for large-scale production training (Meta ranking, LLMs), Adam/AdamW is standard because training cost dominates — you can't afford the extra epochs SGD needs. For fine-tuning where generalization gap is critical (RLHF reward model on small data), SGD or AdamW with aggressive weight decay is preferred.
- **AdamW** (weight decay decoupled from gradient normalization) is strictly preferred over Adam + L2 regularization. In vanilla Adam, L2 penalty is scaled by the adaptive learning rate; AdamW applies weight decay directly to weights.

**Company context:** OpenAI (LLM training), Meta (DLRM/two-tower training at scale).

**Common wrong answer:** "I always use Adam because it's faster." — Staff answer articulates the generalization trade-off and knows when to deviate. Also: confusing L2 regularization with weight decay in Adam (they're only equivalent in SGD).

---

### Q: What is the effect of batch size on generalization, and what do you do when you scale it?

**Answer (Staff level):**
- **Large batch** → sharper minima (higher Hessian max eigenvalue), typically worse generalization. More parallel computation but converges to worse optima.
- **Small batch** → more gradient noise, acts as implicit regularization, converges to flatter minima.
- **Linear scaling rule** (Goyal et al., Facebook, 2017): when you scale batch size by k×, scale LR by k× (linear rule) with **LR warmup** for the first few epochs to avoid instability. Batch size beyond a critical threshold (~8192 for ImageNet-scale) shows diminishing returns even with scaling.
- **Gradient accumulation** achieves large effective batch size without memory cost: run M mini-batches, accumulate gradients, update once.

**Company context:** Meta (data-parallel training, 1000+ GPUs), OpenAI (GPT pretraining).

**Common wrong answer:** "I'd just increase batch size to use all available GPUs." — No LR scaling, no warmup → training instability or worse generalization.

---

## Learning Rate Schedules

### Q: Explain cosine annealing with warm restarts. When would you use it vs. step decay?

**Answer (Staff level):**
- **Cosine annealing**: LR follows a cosine curve from η_max to η_min over T epochs, then resets. The cycle creates multiple "escape from local minima" opportunities — the LR reset can help traverse between basins in the loss landscape.
- **Warm restarts** (SGDR): each restart can have a progressively larger period (T_mult > 1), allowing slower decay on later cycles for fine-grained exploration near convergence.
- **Step decay**: simpler, interpretable, common in pre-LLM era. Drop LR by a factor (e.g., 10×) at fixed milestones. Works well when you know the training duration.
- **For LLMs**: cosine decay with a warmup phase (linear warmup for first 1–4% of training steps) is now standard. Step decay is too coarse for transformer training dynamics.
- **Warmup** is critical because: at initialization, gradients are large and estimates of the second moment in Adam are poorly initialized (high variance). Ramping LR from 0 avoids early instability.

**Company context:** OpenAI (LLM pretraining/fine-tuning), Meta.

**Common wrong answer:** "I use warmup to let the model warm up." — The actual reason is that Adam's second-moment estimate is unreliable early in training (denominator in the adaptive step is based on a few noisy gradient samples). LR warmup keeps steps small until the estimate stabilizes.

---

## Gradient Issues

### Q: What causes exploding gradients and how do you fix it? What about vanishing gradients?

**Answer (Staff level):**
- **Exploding**: deep networks with large weight matrices; BPTT (backprop through time) in RNNs where gradients multiply across many timesteps. Fix: **gradient clipping** (clip by global L2 norm, not per-parameter — preserves gradient direction). Clip norm threshold: typically 1.0 for LLMs, 5.0 for RNNs.
- **Vanishing**: saturating activations (sigmoid/tanh) push gradients to near-zero; deep networks without skip connections. Fix: **ReLU** (non-saturating in positive region), **residual connections** (gradient highway bypassing layers), **careful initialization** (He for ReLU, Xavier/Glorot for tanh).
- In transformers: pre-layer normalization (LayerNorm before attention, not after) stabilizes gradient flow more than post-LN, and is now standard for deep models (GPT-2 onward).

**Company context:** OpenAI (transformer training), Meta (deep ranking networks).

**Common wrong answer:** "I use batch normalization to fix vanishing gradients." — BN helps with internal covariate shift but doesn't directly address gradient propagation across depth. Residual connections are the primary fix for vanishing gradients.

---

### Q: How does gradient clipping work, and why clip the global norm rather than per-parameter?

**Answer (Staff level):**
- **Global norm clipping**: compute `global_norm = sqrt(Σ ||∇θ_i||²)` across all parameters. If `global_norm > max_norm`, scale all gradients by `max_norm / global_norm`.
- **Per-parameter clipping** changes the direction of the overall gradient update — if some parameters are clipped and others aren't, the resulting step points in a different direction than the true gradient. Global norm clipping scales the full vector down while preserving direction.
- In PyTorch: `torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)`.

**Company context:** OpenAI, Meta (any deep network training).

**Common wrong answer:** Clipping each gradient individually (`torch.clamp`) instead of the global norm. Common mistake in custom training loops.

---

## Loss Landscapes

### Q: What is a saddle point? Is it a problem for modern deep learning optimizers?

**Answer (Staff level):**
- **Saddle point**: a critical point where gradient = 0 but curvature is positive in some directions and negative in others. Surrounded by "flat valleys" — gradient is tiny, optimizer stalls.
- Early theory (Dauphin et al., 2014): local minima in high-dimensional spaces are very rare; most critical points are saddle points. The good news: in high dimensions, escaping saddle points is easier than in low dimensions because there are many negative-curvature directions.
- **Adam/SGD + noise**: gradient noise from mini-batching helps escape saddle points. Pure gradient descent (full-batch) stalls; stochastic methods naturally perturb away.
- **In practice**: saddle points are less of a problem than originally feared for overparameterized networks. The more pressing issue is flat regions (vanishing gradients near initialization).

**Company context:** OpenAI (training dynamics deep-dive).

**Common wrong answer:** "Saddle points are a major problem that prevents convergence." — Modern understanding is that in high dimensions, well-initialized overparameterized networks rarely get stuck at saddle points under stochastic optimization.

---

## Convergence Diagnostics

### Q: Training loss is decreasing but validation loss is flat. What do you check?

**Answer (Staff level):**
1. **Overfitting**: training has memorized; add regularization (dropout, weight decay, data augmentation), reduce model capacity, or get more data.
2. **Validation set leakage**: if validation loss is suspiciously flat (not diverging), check if validation labels somehow leaked into training features.
3. **Distribution shift**: training and validation data distributions differ. Check: feature histograms, label proportions, temporal splits.
4. **Wrong evaluation metric**: training optimizes cross-entropy, validation reports accuracy — flat accuracy on imbalanced classes can mask real progress. Report same metric on both.
5. **LR too high**: model oscillates around a good region in training but doesn't settle. Reduce LR or use a schedule.

**Company context:** Netflix (observability probing — "how do you know it's working?"), Shopify.

**Common wrong answer:** Jumping to "add dropout" without first ruling out distribution shift or metric mismatch. The diagnostic sequence matters.

---

## Adam vs. SGD (Deep Dive)

### What Adaptive Learning Rates Actually Do

Standard SGD updates every parameter with the same global learning rate:
```
θ ← θ - η · ∇L
```

Adam maintains per-parameter statistics:
```
m_t = β₁·m_{t-1} + (1-β₁)·g_t          ← 1st moment: exponential moving avg of gradient
v_t = β₂·v_{t-1} + (1-β₂)·g_t²         ← 2nd moment: exponential moving avg of gradient²
θ ← θ - η · m̂_t / (√v̂_t + ε)           ← bias-corrected update
```

The `√v̂_t` term in the denominator is the key: it normalizes each parameter's update by the square root of its recent gradient variance.

**Concrete intuition:**
```
Parameter A: gradients have been consistently large (v̂ is large)
  → Adam divides by large √v̂ → small effective step
  → "This parameter is already getting large updates, be conservative"

Parameter B: gradients are usually near zero but occasionally spike (sparse)
  → v̂ is small → large effective step when gradient appears
  → "This parameter rarely gets signal, amplify it when it does"
```

This is why Adam dominates for NLP/recommendation: embedding tables have sparse gradient updates (most embeddings get zero gradient per batch) — Adam amplifies the signal for rarely-seen embeddings.

### Why SGD Generalizes Better — The Sharp vs. Flat Minima Intuition

The loss landscape has many minima. Not all are equal:

```
Sharp minimum:  high curvature, loss rises steeply when you step away
                model is very sensitive to small weight perturbations
                → poor OOD generalization (test distribution ≠ train exactly)

Flat minimum:   low curvature, loss stays low over a wide region
                model is robust to small weight perturbations
                → better OOD generalization
```

Large gradient noise (small batches, SGD) acts like a random perturbation on the parameters. Sharp minima are destabilized by this noise — the optimizer is pushed out of them. Flat minima absorb the noise and hold.

Adam's adaptive step sizes reduce the effective noise → optimizer settles into whatever minimum it finds first, including sharp ones. SGD's uniform noise is more likely to escape sharp minima and find flat ones.

**Production implication:**
```
Pretraining LLMs on 1T tokens:  AdamW — can't afford extra epochs, fast convergence critical
Fine-tuning RLHF reward model:  AdamW with aggressive weight decay — sharp minima hurt here
Small tabular model:            Either; difference is marginal
```

### AdamW vs. Adam — The Weight Decay Fix

In vanilla Adam, L2 regularization is added to the loss:
```
L_reg = L + (λ/2)||θ||²
∇L_reg = ∇L + λθ
```

The penalty gradient `λθ` is then normalized by `√v̂` just like the loss gradient. In regions where the loss gradient is small (sparse parameters), `v̂` is small → `λθ / √v̂` is large → weight decay is amplified for sparse parameters.

AdamW decouples weight decay from the gradient update:
```
θ ← θ - η · m̂_t / (√v̂_t + ε) - η · λ · θ
```

Weight decay is applied directly to the weights, uniformly, regardless of gradient variance. This is the correct way to regularize adaptive optimizers.

---

## Batch Size and Scaling (Deep Dive)

### Why Large Batches Hurt Generalization — The Noise Argument

Each gradient step with a mini-batch is a noisy estimate of the true gradient:
```
ĝ = (1/B) Σᵢ ∇L(xᵢ, yᵢ)   ← sample mean of B gradients
Var(ĝ) = σ²/B              ← variance shrinks as B grows
```

Small batch (B=32): high variance gradient → noisy steps → acts like random perturbations → escapes sharp minima.
Large batch (B=8192): low variance gradient → nearly exact gradient → converges directly to nearest minimum, sharp or flat.

The generalization gap from large batches is an empirical regularity, not a theorem — but it's consistently observed in practice.

### The Linear Scaling Rule — Why It Works and When It Breaks

When you scale batch size by k×, you want to keep the same effective training dynamics. SGD update with batch B:
```
θ_{t+1} = θ_t - η · ĝ_B
```

Simulate k steps of batch-B SGD with one step of batch-kB SGD:
```
k steps of batch B: θ updates by approximately  -k·η·ĝ_B  (if ĝ ≈ constant)
1 step of batch kB: θ updates by               -η_new·ĝ_{kB} = -η_new·ĝ_B (by LLN)
```

Setting these equal: `η_new = k·η`. Hence **multiply LR by the same factor you multiplied batch size**.

**When it breaks**: at very large batches (B > 8k for ImageNet), the approximation that k steps ≈ 1 large step breaks down because the gradient changes significantly over k steps. The linear rule overestimates the optimal LR → training instability.

Fix: LR warmup. Start from `η_small`, linearly ramp up to `k·η` over the first few epochs. Lets the model settle before taking large steps.

### Gradient Accumulation — Large Batch Without the Memory

```python
optimizer.zero_grad()
for i, (x, y) in enumerate(dataloader):
    loss = model(x, y) / accumulation_steps   # scale loss
    loss.backward()                            # accumulate gradients
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()                       # update once every N batches
        optimizer.zero_grad()
```

Effective batch size = `batch_size × accumulation_steps`. Memory usage = single `batch_size`. The math is identical to a large batch — gradients are summed before the optimizer step.

**Caveat for BatchNorm**: BN statistics (mean, variance) are computed per mini-batch, not the accumulated batch. This makes BN behave as if batch size = `batch_size`, not the accumulated size. Use SyncBN or switch to LayerNorm when using gradient accumulation.

---

## Learning Rate Schedules (Deep Dive)

### Why Warmup Is Necessary — The Adam Initialization Problem

At step t=1, Adam's second moment estimate is:
```
v_1 = (1 - β₂) · g_1²   (β₂ = 0.999 typically)
v̂_1 = v_1 / (1 - β₂¹) = g_1²   ← bias correction makes it equal to g_1²
```

After only 1 step, `v̂` is estimated from a single gradient sample. The effective learning rate per parameter is `η / (√v̂ + ε)` — if `g_1` is small or large by accident, this initial step can be wildly wrong.

Over the first ~1/(1-β₂) = 1000 steps, `v̂` converges to a reliable estimate of gradient variance. Until then, the step size is unreliable.

Warmup keeps actual steps small (low η) during this unreliable window, then raises η once the statistics are trustworthy.

### Cosine Annealing — What the Shape Buys You

```
η_t = η_min + ½(η_max - η_min)(1 + cos(πt/T))
```

The cosine shape has a specific advantage over linear decay:
- Spends most time at intermediate LR (the wide part of the cosine) — allows broad exploration
- Decays slowly early (model still escaping poor regions)
- Decays rapidly near the end (fine-grained convergence to a good minimum)

```
Linear decay: equal time at all LR values → inefficient at both ends
Step decay:   jumps → abrupt transitions can destabilize training
Cosine:       smooth transitions, more time in productive range
```

**Warm restarts**: after reaching η_min, reset to η_max and repeat with a longer period (T_mult × T). The reset can kick the model out of a shallow local minimum into a deeper basin:

```
Cycle 1 (T=100 steps):   coarse exploration
Cycle 2 (T=200 steps):   medium exploration starting from cycle 1's best
Cycle 3 (T=400 steps):   fine-grained convergence
```

In practice: warm restarts are most useful for finding ensembles (save model at each η_min — each cycle's endpoint is a diverse model). For LLM pretraining, a single long cosine decay is now standard.

---

## Gradient Issues (Deep Dive)

### Exploding Gradients — Why BPTT Is the Worst Case

In a standard feedforward network with L layers, the gradient of the loss w.r.t. layer 1 involves a product of L Jacobians:

```
∂L/∂h₁ = (∂h_L/∂h_{L-1}) · (∂h_{L-1}/∂h_{L-2}) · ... · (∂h₂/∂h₁) · ∂L/∂h_L
```

If each Jacobian has spectral radius > 1, this product grows exponentially with depth → exploding gradients.

In RNNs (BPTT), the same weight matrix W is applied at every timestep:
```
∂L/∂h₁ ∝ W^T  (T timesteps)
```

If W has eigenvalue > 1: gradient grows as `λ^T` → explodes for long sequences.
If W has eigenvalue < 1: gradient shrinks as `λ^T` → vanishes for long sequences.

This is the fundamental RNN training problem — solved by LSTMs (gating prevents the product chain) and transformers (attention is not a recurrence).

### Global Norm Clipping — The Direction Preservation Argument

Naive per-parameter clipping:
```python
for p in parameters:
    p.grad.clamp_(-max_val, max_val)   # clips each gradient independently
```

Problem: if parameter A's gradient is clipped at 1.0 but parameter B's is clipped at 0.1 due to different max_val, the resulting update direction changes. You're no longer taking a step in the gradient direction.

Global norm clipping:
```python
global_norm = torch.sqrt(sum(p.grad.norm()**2 for p in parameters))
if global_norm > max_norm:
    scale = max_norm / global_norm
    for p in parameters:
        p.grad *= scale
```

All gradients scaled by the same factor → update direction unchanged, only magnitude reduced. This preserves the relative importance of each parameter's gradient.

### Pre-LN vs. Post-LN in Transformers

Original transformer (Vaswani 2017): Post-LN — LayerNorm applied after residual connection:
```
x = LayerNorm(x + Sublayer(x))
```

Modern transformers (GPT-2+): Pre-LN — LayerNorm applied before the sublayer:
```
x = x + Sublayer(LayerNorm(x))
```

Why Pre-LN is more stable: in Post-LN, gradients must flow through the LayerNorm at every layer. For deep models (>12 layers), this creates gradient vanishing at initialization. Pre-LN keeps the residual stream unnormalized — the gradient highway through residual connections is clean.

---

## Loss Landscape (Deep Dive)

### Saddle Points — The High-Dimensional Intuition

In 2D: a saddle point has one direction going up and one going down (like a horse saddle). Easy to get stuck — gradient points away in one direction, toward the saddle in the other.

In N-dimensional space (N = millions for deep nets): a saddle point has some directions going up and some going down. The probability that ALL N directions point upward (a true local minimum) is exponentially small.

```
N=2:   ~50% of critical points are local minima
N=1M:  ~0% of critical points are local minima (nearly all are saddle points)
       but there are also exponentially more negative-curvature directions to escape
```

The escape is easy in high dimensions: gradient noise from mini-batching has components in many random directions, almost certainly including a negative-curvature direction at any saddle point.

**What actually causes stalling**: not saddle points, but **flat regions near initialization**. Before the model has learned useful representations, gradients are tiny everywhere — not because you're at a critical point, but because the loss surface is nearly flat in a large basin around initialization.

---

## Convergence Diagnostics (Deep Dive)

### Reading Training Curves — A Systematic Approach

```
Scenario 1: train loss ↓, val loss ↓ together
  → Healthy training. Both improving.

Scenario 2: train loss ↓, val loss flat then diverges
  → Classic overfitting. Gap widens over time.
  → Fix: regularization, less capacity, more data.

Scenario 3: train loss ↓, val loss flat (not diverging)
  → Distribution shift or label leakage. Not just overfitting.
  → Overfitting would show val loss eventually rising.
  → Flat val loss means the model can't learn the val distribution at all.

Scenario 4: both losses flat from step 1
  → Initialization problem or LR too low. Model not learning.
  → Check: gradient norms (should be non-zero), LR magnitude.

Scenario 5: both losses oscillate without converging
  → LR too high. Optimizer overshooting.
  → Fix: reduce LR, add warmup, clip gradients.

Scenario 6: train loss decreasing in steps, not smoothly
  → LR schedule steps are too large. Smooth with cosine or reduce step size.
```

### Gradient Norm Monitoring — The Most Underused Diagnostic

Most engineers only watch the loss. Gradient norms tell you much more:

```
Gradient norm growing over time:
  → Exploding gradients. Add clipping or reduce LR.

Gradient norm near zero from the start:
  → Vanishing gradients or dead ReLUs. Check activation statistics.

Gradient norm normal for most layers, near zero for early layers:
  → Gradient vanishing across depth. Add residual connections or use Pre-LN.

Gradient norm spikes at specific steps:
  → Outlier samples or numerical instability. Check data for corrupt samples.
```

In PyTorch, log gradient norms to your experiment tracker:
```python
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
wandb.log({"grad_norm": total_norm})
```

Monitor this alongside loss — a rising grad_norm before the loss spikes is an early warning of instability.
