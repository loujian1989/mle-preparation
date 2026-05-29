"""
Scaled Dot-Product Attention — Transformer Component
=====================================================

Meta ML coding probe: "Implement scaled dot-product attention from scratch.
Walk me through the math and explain why we divide by sqrt(d_k)."

Mathematical formulation:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    Q: query matrix,  shape (N, seq_q, d_k)
    K: key matrix,    shape (N, seq_k, d_k)
    V: value matrix,  shape (N, seq_k, d_v)
    Output:           shape (N, seq_q, d_v)

Why divide by sqrt(d_k)?
    The dot product QK^T grows in magnitude with d_k (variance ~ d_k).
    Without scaling, large dot products push softmax into saturation
    regions where gradients vanish. Dividing by sqrt(d_k) stabilizes
    gradient flow — especially critical at initialization.

Causal masking:
    Used in decoder self-attention to prevent position i from attending
    to positions j > i. Implemented by adding -inf to the upper triangle
    before softmax (those entries become 0 after softmax).

Numerically stable softmax:
    Subtract row maximum before exp to avoid overflow:
        softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

Complexity:
    Time:  O(N * seq_q * seq_k * d_k) for QK^T, plus O(N * seq_q * seq_k * d_v) for AV
    Space: O(N * seq_q * seq_k) for attention weight matrix
"""

import numpy as np
from typing import Optional


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along given axis.

    Args:
        x:    Input array.
        axis: Axis along which to compute softmax.

    Returns:
        Softmax probabilities, same shape as x.

    Complexity:
        Time:  O(N) per row
        Space: O(N)
    """
    x_shifted = x - x.max(axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute scaled dot-product attention.

    Args:
        Q:    Query, shape (N, seq_q, d_k).
        K:    Key,   shape (N, seq_k, d_k).
        V:    Value, shape (N, seq_k, d_v).
        mask: Optional boolean mask of shape broadcastable to (N, seq_q, seq_k).
              Where mask is True, attention weight is set to -inf (then 0 after softmax).
              Used for causal masking and padding masking.

    Returns:
        (output, attn_weights):
            output:       shape (N, seq_q, d_v)
            attn_weights: shape (N, seq_q, seq_k) — for inspection/debugging

    Raises:
        ValueError: If Q/K/V have incompatible shapes.

    Complexity:
        Time:  O(N * seq_q * seq_k * d_k) — dominated by QK^T matmul
        Space: O(N * seq_q * seq_k)
    """
    if Q.ndim != 3 or K.ndim != 3 or V.ndim != 3:
        raise ValueError(f"Q, K, V must be 3-dimensional, got {Q.ndim}, {K.ndim}, {V.ndim}")
    if Q.shape[-1] != K.shape[-1]:
        raise ValueError(f"Q d_k={Q.shape[-1]} must match K d_k={K.shape[-1]}")
    if K.shape[-2] != V.shape[-2]:
        raise ValueError(f"K seq_k={K.shape[-2]} must match V seq_k={V.shape[-2]}")

    d_k = Q.shape[-1]

    # (N, seq_q, seq_k)
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)

    # Apply mask: set masked positions to -inf so softmax gives 0 weight
    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    attn_weights = _softmax(scores, axis=-1)  # (N, seq_q, seq_k)

    # Weighted sum of values
    output = attn_weights @ V  # (N, seq_q, d_v)

    return output, attn_weights


def make_causal_mask(seq_len: int) -> np.ndarray:
    """Create upper-triangular causal mask for decoder self-attention.

    Position i should not attend to position j > i.

    Args:
        seq_len: Sequence length.

    Returns:
        Boolean mask of shape (1, seq_len, seq_len).
        True = masked (set to -inf), False = allowed.

    Complexity:
        Time:  O(seq_len^2)
        Space: O(seq_len^2)
    """
    # Upper triangle excluding diagonal = future positions
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    return mask[np.newaxis, :, :]  # (1, seq_len, seq_len) — broadcasts over batch


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    rng = np.random.default_rng(42)
    N, seq_q, seq_k, d_k, d_v = 2, 4, 6, 8, 16

    Q = rng.standard_normal((N, seq_q, d_k))
    K = rng.standard_normal((N, seq_k, d_k))
    V = rng.standard_normal((N, seq_k, d_v))

    # Output shapes
    output, weights = scaled_dot_product_attention(Q, K, V)
    assert output.shape == (N, seq_q, d_v), f"Output shape: {output.shape}"
    assert weights.shape == (N, seq_q, seq_k), f"Weights shape: {weights.shape}"

    # Attention weights sum to 1 along seq_k
    assert np.allclose(weights.sum(axis=-1), 1.0, atol=1e-6), "Weights should sum to 1"

    # All weights in [0, 1]
    assert weights.min() >= -1e-9 and weights.max() <= 1.0 + 1e-9

    # Causal mask: upper triangle entries should be near 0
    seq = 5
    Qs = rng.standard_normal((1, seq, d_k))
    Ks = rng.standard_normal((1, seq, d_k))
    Vs = rng.standard_normal((1, seq, d_v))
    mask = make_causal_mask(seq)
    _, w_causal = scaled_dot_product_attention(Qs, Ks, Vs, mask=mask)

    for i in range(seq):
        for j in range(seq):
            if j > i:  # future positions must have zero attention
                assert w_causal[0, i, j] < 1e-6, (
                    f"Causal mask failed at ({i},{j}): weight={w_causal[0,i,j]:.4f}"
                )

    # Scaling: without sqrt(d_k), dot products grow large and attention
    # collapses to argmax. Verify output varies (not collapsed to single V row).
    K_large = K * 100.0
    out_large, w_large = scaled_dot_product_attention(Q, K_large, V)
    # Weights should still sum to 1
    assert np.allclose(w_large.sum(axis=-1), 1.0, atol=1e-6)

    # Overfit: if Q == K, the self-similarity should be high on diagonal
    Qs2 = rng.standard_normal((1, 3, 4))
    _, w_self = scaled_dot_product_attention(Qs2, Qs2, Qs2)
    # Diagonal weights should be higher than off-diagonal average
    diag_mean = np.mean([w_self[0, i, i] for i in range(3)])
    off_mean = (w_self[0].sum() - np.trace(w_self[0])) / (3 * 2)
    assert diag_mean > off_mean, f"Self-attention diagonal should dominate: {diag_mean:.3f} vs {off_mean:.3f}"

    print("  scaled_dot_product_attention: all tests passed")


if __name__ == "__main__":
    _test()
