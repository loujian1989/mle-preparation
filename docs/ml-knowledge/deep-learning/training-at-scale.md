# Training at Scale — ML Knowledge Q&A

P1: Meta, OpenAI. Expected depth: understand GPU memory management, distributed training strategies.

---

## Normalization Layers

### Q: Compare BatchNorm, LayerNorm, GroupNorm, InstanceNorm. When do you use each?

**Answer (Staff level):**

| | **BatchNorm** | **LayerNorm** | **GroupNorm** | **InstanceNorm** |
|---|---|---|---|---|
| Normalize over | Batch dimension | Feature dimension | Groups of channels | Per-sample, per-channel |
| Depends on batch size | Yes (breaks at batch=1) | No | No | No |
| Best for | CNN (image, large batch) | Transformers (NLP, any batch size) | Object detection (small batch) | Style transfer |
| Training-inference difference | Uses batch stats in training, running stats in inference | Same computation (no train/test difference) | Same | Same |

- **BatchNorm fails at batch_size=1**: statistics computed over 1 sample are noisy (std=0 → division by 0 or by epsilon only). Use LayerNorm or GroupNorm when batch size is small (e.g., multi-GPU training with 1 sample/GPU, video models).
- **Why transformers use LayerNorm**: sequence lengths vary; you can't normalize across a variable-size batch dimension. LayerNorm normalizes per token (across the feature dimension) — consistent regardless of batch size or sequence length.
- **GroupNorm (He et al., 2018)**: divide channels into G groups, normalize within each group. Works at batch_size=1. Standard for object detection (Mask R-CNN) where batch=2 per GPU.

**Company context:** OpenAI, Meta (both use LayerNorm in transformers; BN in vision models).

**Common wrong answer:** "LayerNorm is just BatchNorm with a different axis." — The key distinction is *dependency on batch size*: BatchNorm needs a sufficient batch for stable statistics; LayerNorm is batch-size-independent. This matters for deployment (variable batch sizes at inference).

---

## Distributed Training

### Q: When do you use DDP vs. FSDP, and what problem does ZeRO solve?

**Answer (Staff level):**
- **DDP (Distributed Data Parallel)**: each GPU holds a full copy of the model. Gradients are averaged across GPUs via All-Reduce after each backward pass. Scales well when model fits in single-GPU memory.
- **FSDP (Fully Sharded Data Parallel)**: shards model parameters, gradients, and optimizer states across GPUs. Each GPU holds only a fraction of the model. Required when model doesn't fit on a single GPU (e.g., LLaMA-2 70B needs ~140 GB at fp32 — doesn't fit on an 80 GB A100).

**ZeRO (Zero Redundancy Optimizer)** stages:
| Stage | What's sharded | Memory savings |
|---|---|---|
| ZeRO-1 | Optimizer states | ~4× |
| ZeRO-2 | + Gradients | ~8× |
| ZeRO-3 | + Parameters | ~64× (model itself sharded) |

- **FSDP in PyTorch implements ZeRO-3**: shards parameters + gradients + optimizer states. Before each layer's forward/backward, parameters are gathered (All-Gather), used, then discarded (to save memory).
- **Communication overhead**: FSDP/ZeRO-3 trades memory for communication (additional All-Gather ops vs. DDP's single All-Reduce). Use when memory is the binding constraint; DDP when communication is the bottleneck.

**Company context:** Meta (FSDP for Llama pretraining), OpenAI.

**Common wrong answer:** "I'd use multi-GPU training to speed up training." — Speed (throughput) and memory (model size) are different concerns. DDP for throughput; FSDP/ZeRO when model doesn't fit in memory.

---

## Gradient Checkpointing

### Q: What is gradient checkpointing and what does it trade?

**Answer (Staff level):**
- **Problem**: the backward pass requires all intermediate activations from the forward pass (stored in GPU memory). For a 100-layer transformer with batch size 32, activation memory ∝ `batch × seq_len × d_model × num_layers` — can exceed available memory.
- **Gradient checkpointing**: instead of storing all activations, store only activations at checkpoint boundaries (e.g., every N layers). During backward pass, recompute the non-stored activations on-the-fly from the nearest checkpoint.
- **Trade-off**: saves memory (O(√N) vs. O(N) activations for optimal checkpointing) at the cost of ~33% more FLOPs (each activation computed twice: once forward, once during recomputation in backward).
- **PyTorch**: `torch.utils.checkpoint.checkpoint(module, *inputs)` wraps a module to use checkpointing.
- **When to use**: when increasing batch size causes OOM and you can afford longer per-step time. Also used in very long sequence training.

**Company context:** Meta, OpenAI (LLM training where activation memory dominates).

**Common wrong answer:** "Gradient checkpointing saves GPU memory for free." — It trades compute for memory. The ~33% additional FLOPs is a real cost — total training time increases.

---

## Mixed Precision

### Q: Explain mixed precision training (FP16/BF16). When does it fail?

**Answer (Staff level):**
- **Motivation**: FP32 parameters + gradients use 4 bytes/value. FP16 uses 2 bytes → 2× memory reduction, 2× bandwidth improvement, and faster matrix multiplication on Tensor Cores.
- **Mixed precision (Micikevicius et al., NVIDIA, 2018)**:
  1. Maintain **FP32 master copy** of weights.
  2. **Forward and backward** pass in FP16 (fast, small).
  3. **Gradient scaling**: FP16 has small dynamic range — gradients near zero underflow to 0. Scale loss before backward (multiply by a scale factor), unscale before weight update. PyTorch `GradScaler` does this automatically.
  4. **Update** master FP32 weights with unscaled FP16 gradients.
- **BF16 (Brain Float)**: same 2 bytes as FP16 but larger dynamic range (same exponent bits as FP32). Doesn't need gradient scaling. Preferred on A100/H100. Not supported on older hardware (V100).
- **When it fails**:
  - Very small gradients that underflow even with scaling (increase scale or use BF16)
  - Layer normalization: can cause NaN if variance is very small in FP16 (run in FP32 if needed)
  - Custom CUDA ops that don't handle FP16 correctly

**Company context:** Meta, OpenAI (both train LLMs with BF16 on H100s).

**Common wrong answer:** "I'd use FP16 everywhere for speed." — Master weights must be FP32 to avoid precision loss in the optimizer update. The "mixed" in mixed precision is the key — specific parts use lower precision.

---

## Gradient Accumulation

### Q: What is gradient accumulation and when do you use it?

**Answer (Staff level):**
- **Purpose**: simulate a large batch size without allocating memory for all samples simultaneously.
- **Mechanism**: run forward+backward for M mini-batches without calling `optimizer.step()`. Gradients accumulate in `param.grad`. After M steps, `optimizer.step()` + `optimizer.zero_grad()`.
- **Effective batch size** = `per_device_batch_size × accumulation_steps × num_gpus`.
- **When to use**: GPU memory too small for desired batch size; gradient updates are too noisy with small batches; you need to match a published large-batch training recipe.
- **Caveats**:
  - BatchNorm sees only the small mini-batch per forward pass, not the accumulated effective batch. If using BN, gradient accumulation doesn't improve batch statistics for BN.
  - Gradient accumulation increases time per optimizer step (M forward passes vs. 1), but total compute is identical to the equivalent large batch.

**Company context:** Meta, OpenAI (fine-tuning on single GPU with large effective batch).

**Common wrong answer:** "Gradient accumulation is the same as training with a larger batch size in all respects." — BatchNorm statistics are still computed on the small mini-batch, not the accumulated batch. LayerNorm is unaffected.

---

## GPU Memory Budget

### Q: A 70B parameter model. How much GPU memory does it require for inference vs. training?

**Answer (Staff level):**
- **Parameters**: 70B × 2 bytes (BF16/FP16) = 140 GB. Minimum for inference.
- **Inference** (forward only, batch=1): parameters + activations. Activations for a single forward pass with short context are small relative to parameters. **Total: ~140–160 GB**. Requires 2 × 80 GB A100s (or 1 × H100 NVL with 94 GB).
- **Training** (fine-tuning with Adam, FP32 master weights):
  - FP32 parameters (master copy): 70B × 4 = 280 GB
  - FP32 gradients: 280 GB
  - Adam states (first + second moment, FP32): 2 × 280 = 560 GB
  - BF16 working copy: 140 GB
  - Activations: depends on batch size + seq length, estimate 40–100 GB
  - **Total: ~1.3 TB** — requires ZeRO-3 / FSDP across many GPUs
- **With quantization (QLORA, 4-bit)**: 70B × 0.5 bytes = 35 GB for inference (fits in 1 × A100). Training gradient + optimizer still in FP32, but much more accessible.

**Company context:** OpenAI, Meta. Shows fluency with GPU memory planning.

**Common wrong answer:** "A 70B model needs 70 GB." — This assumes 1 byte/parameter (INT8). At BF16 (2 bytes), it's 140 GB. At FP32, 280 GB. State the precision.
