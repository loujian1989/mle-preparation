"""
Multi-Head Attention — Transformer Component
=============================================

Meta ML coding probe: "Implement multi-head attention. Why use multiple heads
instead of one large attention head?"

Architecture:
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O

    head_i = Attention(Q * W_Qi, K * W_Ki, V * W_Vi)

    W_Qi: (d_model, d_k)   where d_k = d_model // h
    W_Ki: (d_model, d_k)
    W_Vi: (d_model, d_v)   where d_v = d_model // h
    W_O:  (h * d_v, d_model)

Why multiple heads?
    Each head can attend to different aspects of the input simultaneously:
    one head may capture syntactic relationships, another semantic ones.
    It's like having h different "views" of the same input, then merging.
    Empirically outperforms a single head with the same parameter count.

Implementation detail:
    Rather than h separate weight matrices, we use a single (d_model, d_model)
    projection and reshape. This is equivalent but more GPU-efficient.

    Reshape trick for batched h-head attention:
        (N, seq, d_model) -> (N, seq, h, d_k) -> (N, h, seq, d_k)
        Then batch matmul runs h heads simultaneously.

Complexity:
    Time:  O(N * h * seq^2 * d_k) — attention; O(N * seq * d_model^2) — projections
    Space: O(N * h * seq^2) — attention weight matrices
"""

import numpy as np
from typing import Optional, Tuple

from scaled_dot_product_attention import make_causal_mask, _softmax


class MultiHeadAttention:
    """Multi-head self/cross attention using NumPy.

    Args:
        d_model: Total model dimension.
        num_heads: Number of attention heads h. Must evenly divide d_model.
        seed: RNG seed for weight initialization.

    Raises:
        ValueError: If d_model % num_heads != 0.
    """

    def __init__(self, d_model: int, num_heads: int, seed: int = 0) -> None:
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_heads={num_heads}"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head

        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / d_model)

        # Single combined projection matrices (more efficient than h separate ones)
        self.W_Q = rng.standard_normal((d_model, d_model)) * scale  # (d_model, d_model)
        self.W_K = rng.standard_normal((d_model, d_model)) * scale
        self.W_V = rng.standard_normal((d_model, d_model)) * scale
        self.W_O = rng.standard_normal((d_model, d_model)) * scale  # output projection

    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute multi-head attention.

        Args:
            query:  Shape (N, seq_q, d_model).
            key:    Shape (N, seq_k, d_model).
            value:  Shape (N, seq_k, d_model).
            mask:   Optional boolean mask broadcastable to (N, h, seq_q, seq_k).
                    True = masked out (attend weight -> 0).

        Returns:
            (output, attn_weights):
                output:       (N, seq_q, d_model)
                attn_weights: (N, num_heads, seq_q, seq_k) — averaged across heads for inspection

        Raises:
            ValueError: If input shapes are incompatible.

        Complexity:
            Time:  O(N * h * seq^2 * d_k + N * seq * d_model^2)
            Space: O(N * h * seq^2)
        """
        if query.ndim != 3:
            raise ValueError(f"query must be 3-dimensional, got shape {query.shape}")

        N, seq_q, _ = query.shape
        seq_k = key.shape[1]
        h = self.num_heads
        d_k = self.d_k

        # Linear projections: (N, seq, d_model) -> (N, seq, d_model)
        Q = query @ self.W_Q
        K = key   @ self.W_K
        V = value @ self.W_V

        # Split into h heads: (N, seq, d_model) -> (N, h, seq, d_k)
        Q = Q.reshape(N, seq_q, h, d_k).transpose(0, 2, 1, 3)  # (N, h, seq_q, d_k)
        K = K.reshape(N, seq_k, h, d_k).transpose(0, 2, 1, 3)  # (N, h, seq_k, d_k)
        V = V.reshape(N, seq_k, h, d_k).transpose(0, 2, 1, 3)  # (N, h, seq_k, d_k)

        # Scaled dot-product attention per head (batch over N * h)
        # Flatten N and h into one batch dimension for matmul
        Q_flat = Q.reshape(N * h, seq_q, d_k)
        K_flat = K.reshape(N * h, seq_k, d_k)
        V_flat = V.reshape(N * h, seq_k, d_k)

        scores = Q_flat @ K_flat.transpose(0, 2, 1) / np.sqrt(d_k)  # (N*h, seq_q, seq_k)

        if mask is not None:
            # Reshape scores to (N, h, seq_q, seq_k) for broadcast-compatible masking,
            # then flatten back. This handles any mask shape broadcastable to (N, h, seq_q, seq_k).
            scores_4d = scores.reshape(N, h, seq_q, seq_k)
            scores_4d = np.where(mask, -1e9, scores_4d)
            scores = scores_4d.reshape(N * h, seq_q, seq_k)

        attn_flat = _softmax(scores, axis=-1)                    # (N*h, seq_q, seq_k)
        context_flat = attn_flat @ V_flat                        # (N*h, seq_q, d_k)

        # Reshape back: (N*h, seq_q, d_k) -> (N, h, seq_q, d_k) -> (N, seq_q, d_model)
        context = context_flat.reshape(N, h, seq_q, d_k)
        context = context.transpose(0, 2, 1, 3).reshape(N, seq_q, h * d_k)  # (N, seq_q, d_model)

        # Output projection
        output = context @ self.W_O  # (N, seq_q, d_model)

        attn_weights = attn_flat.reshape(N, h, seq_q, seq_k)
        return output, attn_weights


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    rng = np.random.default_rng(7)
    N, seq, d_model, num_heads = 2, 5, 32, 4

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    x = rng.standard_normal((N, seq, d_model))

    # Self-attention
    out, weights = mha.forward(x, x, x)
    assert out.shape == (N, seq, d_model), f"Output shape: {out.shape}"
    assert weights.shape == (N, num_heads, seq, seq), f"Weights shape: {weights.shape}"

    # Attention weights sum to 1 along key dimension
    assert np.allclose(weights.sum(axis=-1), 1.0, atol=1e-5), "Weights should sum to 1"

    # Cross-attention (different seq lengths for query and key/value)
    seq_q, seq_kv = 3, 7
    q = rng.standard_normal((N, seq_q, d_model))
    kv = rng.standard_normal((N, seq_kv, d_model))
    out_cross, weights_cross = mha.forward(q, kv, kv)
    assert out_cross.shape == (N, seq_q, d_model)
    assert weights_cross.shape == (N, num_heads, seq_q, seq_kv)

    # Causal mask in self-attention: future positions should have ~0 weight
    mask = make_causal_mask(seq)                    # (1, seq, seq)
    mask_4d = mask[:, np.newaxis, :, :]             # (1, 1, seq, seq) -> broadcast over h
    _, w_causal = mha.forward(x, x, x, mask=mask_4d.reshape(1, 1, seq, seq))
    for i in range(seq):
        for j in range(i + 1, seq):
            assert w_causal[0, :, i, j].max() < 1e-5, (
                f"Causal mask failed at ({i},{j})"
            )

    # Invalid: d_model not divisible by num_heads
    try:
        MultiHeadAttention(d_model=10, num_heads=3)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass

    # Overfit: output should not be all zeros or constant
    assert not np.allclose(out, 0.0), "Output should not be zero"
    assert out.std() > 1e-3, f"Output has no variance: std={out.std():.4f}"

    print("  MultiHeadAttention: all tests passed")


if __name__ == "__main__":
    _test()
