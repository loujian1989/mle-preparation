# Transformers & Attention — ML Knowledge Q&A

P0: OpenAI, Meta. Expected depth: can implement from scratch, can explain design decisions.

---

## Self-Attention Mechanics

### Q: Walk through the self-attention computation. What is the O(N²) bottleneck?

**Answer (Staff level):**
- For an input sequence of length N, each token produces Q, K, V vectors of dimension d_k.
- Attention(Q, K, V) = softmax(QK^T / √d_k) · V
- **QK^T** is an N×N matrix: computing it requires O(N²·d_k) operations and storing it requires O(N²) memory. This is the bottleneck — for N=100K tokens, the attention matrix alone is 10^10 elements at fp32 ≈ 40 GB.
- **√d_k scaling**: without scaling, dot products grow in magnitude with d_k (variance of QK^T = d_k), pushing softmax into saturation (vanishing gradients). Dividing by √d_k keeps variance ≈ 1.
- **Causal masking** (decoder): set upper-triangle of QK^T to -∞ before softmax so token i cannot attend to tokens j > i.

**Company context:** OpenAI (core architecture question — you must be able to derive attention), Meta (DLRM embeds queries/keys for feature interactions).

**Common wrong answer:** "Attention is O(N) per token." — No. The pairwise scoring step is O(N²). Interviewers at OpenAI will catch this immediately.

---

### Q: What is multi-head attention and why does it help over single-head?

**Answer (Staff level):**
- **Multi-head**: project Q, K, V into H parallel lower-dimensional subspaces (d_k = d_model / H), compute attention in each head independently, concatenate, project back.
- **Why it helps**: single-head attention collapses to a single weighted average of values — it can only attend to one "type" of relationship at a time. Multi-head allows the model to simultaneously attend to:
  - Syntactic relationships (one head)
  - Semantic co-references (another head)
  - Positional patterns (another head)
- **Interpretation (Voita et al., 2019)**: pruning experiments show that most heads are redundant; a small subset are "interpretable" (attend to specific linguistic structures). But in practice, all heads contribute to robustness.
- **Compute cost**: same FLOPs as single-head (dimension reduction cancels), so it's "free" in theory. In practice, H separate softmax operations have overhead.

**Company context:** OpenAI.

**Common wrong answer:** "Multi-head adds capacity by having more parameters." — The projection matrices add parameters, but the key benefit is parallel attention over different representation subspaces, not raw parameter count.

---

## Positional Encoding

### Q: Why does the transformer need positional encoding? Absolute vs. relative — which is better?

**Answer (Staff level):**
- Attention is **permutation-invariant**: if you shuffle the tokens, the self-attention scores change but the model has no inherent notion of which token was where. Positional encoding injects order information.
- **Absolute (original Vaswani et al.)**: fixed sinusoidal functions: PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(...). Added to token embeddings. Works for fixed-length sequences; extrapolates poorly beyond training length.
- **Learned absolute**: token embedding + positional embedding (both learned). GPT-2/BERT. Fails to extrapolate beyond max training length.
- **Relative (Shaw et al., Transformer-XL, ALiBi)**: encode relative distance between token pairs, not absolute position. Attention score gets a bias: `A_{ij} = QK^T_{ij} − m·|i−j|` (ALiBi). Extrapolates better to longer sequences at inference.
- **RoPE (Rotary Position Embedding, GPT-NeoX, LLaMA)**: rotate Q and K vectors by angle proportional to position. The dot product QK^T then naturally encodes relative position. Current standard for large LLMs.

**Company context:** OpenAI (expected to know RoPE and why absolute breaks at long context).

**Common wrong answer:** "I'd use learned positional embeddings." — Standard but doesn't extrapolate. Staff-level answer mentions relative encoding and the failure mode of absolute at long context.

---

## KV Cache

### Q: What is the KV cache and how does it reduce inference cost?

**Answer (Staff level):**
- During autoregressive generation, token i attends to all previous tokens 1..i. Without caching, every new token requires recomputing K and V for all previous tokens — O(N²) total cost for N tokens.
- **KV cache**: store K and V tensors for all previously computed tokens. Each new token only computes its own Q, K, V and appends K, V to the cache. Attention for new token = softmax(q_new · K_cache^T) · V_cache. Reduces computation from O(N²) to O(N) over the full sequence.
- **Memory cost**: KV cache size = 2 × (num_layers × num_heads × d_head × seq_len × batch_size × dtype_bytes). For LLaMA-2-70B, seq_len=4096, batch=1: ≈ 16 GB. This is often the binding constraint on serving batch size.
- **Multi-Query Attention (MQA)** and **Grouped-Query Attention (GQA)**: share K, V heads across query heads (GQA: group H_q query heads to share 1 KV head). Reduces KV cache size by H_q/H_kv×. Used in LLaMA-2, Mistral.

**Company context:** OpenAI (critical for understanding LLM serving costs), Meta (LLM serving infra).

**Common wrong answer:** "KV cache reduces memory usage." — It increases memory (storing all past KV), but reduces compute. The trade-off: more memory → higher throughput (fewer recomputations per request).

---

## Pre-LN vs. Post-LN

### Q: Why did transformers shift from Post-LayerNorm to Pre-LayerNorm?

**Answer (Staff level):**
- **Post-LN (original)**: `x → Sublayer → LayerNorm`. Residual stream passes through the normalization at each layer. Gradients vanish in early layers because the normalization at the output layer clips gradient magnitude.
- **Pre-LN (GPT-2 onward)**: `x → LayerNorm → Sublayer → x + output`. The residual connection bypasses the normalization, so gradients flow unimpeded. Enables training very deep models without warmup instability.
- **Empirical**: Pre-LN models train more stably without warmup (though warmup is still used); converge to similar final performance. Post-LN can theoretically achieve slightly better final loss but is harder to train.
- **Root cause**: in Post-LN, at initialization, the layer's contribution to the residual is ≈0 (output of sublayer ≈0 with He init), so the residual is identity-like — but LayerNorm applied to the sum kills the gradient. Pre-LN avoids this by normalizing the input to the sublayer, not the output.

**Company context:** OpenAI (architectural decision-making question).

**Common wrong answer:** "Pre-LN is faster." — Speed is similar; the benefit is training stability, not throughput.

---

## Attention Variants for Long Context

### Q: Flash Attention — what problem does it solve?

**Answer (Staff level):**
- Standard attention materializes the N×N attention matrix in HBM (GPU DRAM). For N=8K, this is 8K² × 4 bytes = 256 MB per head — bandwidth-bound, not compute-bound.
- **FlashAttention (Dao et al., 2022)**: reorders computation to tile the attention matrix in SRAM (on-chip, fast). Computes softmax and output in blocks without materializing the full N×N matrix. Result: O(N²) FLOPs but O(N) HBM memory accesses (instead of O(N²)).
- **Speedup**: 2–4× wall-clock for training (memory I/O is the bottleneck at most sequence lengths), up to 20× for very long sequences.
- **FlashAttention-2**: improved parallelism (split across seq dimension, not batch), better GPU utilization.
- This is now standard in most serious LLM implementations (Triton kernel in PyTorch).

**Company context:** OpenAI, Meta. Not expected to implement, but must know what problem it solves and why IO-bound matters.

**Common wrong answer:** "FlashAttention reduces FLOPs." — FLOPs are the same. It reduces HBM memory I/O. Important distinction: modern GPUs are memory-bandwidth-bound for attention at practical sequence lengths.
