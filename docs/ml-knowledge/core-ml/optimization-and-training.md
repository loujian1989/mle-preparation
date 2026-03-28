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
