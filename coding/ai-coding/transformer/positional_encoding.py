"""
Positional Encoding — Transformer Component
============================================

Meta ML coding probe: "Why does a Transformer need positional encoding?
Implement sinusoidal positional encoding from Attention Is All You Need."

Motivation:
    Self-attention is permutation-invariant — it treats all positions equally.
    Positional encoding injects sequence order information before attention.

Sinusoidal formulation (Vaswani et al., 2017):
    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    where pos = token position (0-indexed), i = dimension pair index,
    d_model = model embedding dimension.

Key interview points:
    - Output shape is (max_len, d_model); added to token embeddings (not concatenated)
    - The wavelengths form a geometric progression from 2*pi to 10000*2*pi
    - Sinusoidal allows generalization to sequences longer than those seen in training
    - Learned PE (used in BERT/GPT-2) is an embedding table indexed by position
    - Modern models (RoPE, ALiBi) encode relative positions directly in attention

Complexity:
    Time:  O(max_len * d_model)
    Space: O(max_len * d_model)
"""

import numpy as np
from typing import Optional


def sinusoidal_encoding(max_len: int, d_model: int) -> np.ndarray:
    """Compute fixed sinusoidal positional encoding matrix.

    Args:
        max_len: Maximum sequence length.
        d_model: Model embedding dimension (must be even for clean pairing).

    Returns:
        PE matrix of shape (max_len, d_model).

    Raises:
        ValueError: If d_model < 2 or max_len < 1.

    Complexity:
        Time:  O(max_len * d_model)
        Space: O(max_len * d_model)
    """
    if d_model < 2:
        raise ValueError(f"d_model must be >= 2, got {d_model}")
    if max_len < 1:
        raise ValueError(f"max_len must be >= 1, got {max_len}")

    pe = np.zeros((max_len, d_model))
    positions = np.arange(max_len)[:, np.newaxis]          # (max_len, 1)
    dim_pairs = np.arange(0, d_model, 2)[np.newaxis, :]    # (1, d_model//2)

    # Denominator: 10000^(2i / d_model) for each dimension pair i
    denom = np.power(10000.0, dim_pairs / d_model)          # (1, d_model//2)
    angles = positions / denom                              # (max_len, d_model//2)

    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles[: , : d_model // 2])

    return pe


class LearnedPositionalEncoding:
    """Learned positional embedding table (BERT/GPT-style).

    Parameters are an embedding matrix indexed by absolute position.
    Unlike sinusoidal PE, this does not generalize beyond max_len.

    Args:
        max_len: Maximum sequence length.
        d_model: Embedding dimension.
        seed:    RNG seed for reproducibility.

    Raises:
        ValueError: If max_len < 1 or d_model < 1.
    """

    def __init__(self, max_len: int, d_model: int, seed: int = 0) -> None:
        if max_len < 1 or d_model < 1:
            raise ValueError("max_len and d_model must be >= 1")
        rng = np.random.default_rng(seed)
        # Initialize small — same scale as token embeddings
        self.embeddings = rng.standard_normal((max_len, d_model)) * 0.02

    def forward(self, seq_len: int) -> np.ndarray:
        """Return positional embeddings for positions 0..seq_len-1.

        Args:
            seq_len: Sequence length (must be <= max_len).

        Returns:
            Embedding matrix of shape (seq_len, d_model).

        Raises:
            ValueError: If seq_len > max_len.
        """
        if seq_len > self.embeddings.shape[0]:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_len={self.embeddings.shape[0]}"
            )
        return self.embeddings[:seq_len]


def add_positional_encoding(
    x: np.ndarray,
    pe: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Add positional encoding to token embeddings.

    Args:
        x:  Token embeddings of shape (seq_len, d_model) or (N, seq_len, d_model).
        pe: Positional encoding matrix of shape (seq_len, d_model).
            If None, sinusoidal encoding is computed automatically.

    Returns:
        x + pe, same shape as x.

    Raises:
        ValueError: If shape mismatch between x and pe.
    """
    seq_len = x.shape[-2]
    d_model = x.shape[-1]

    if pe is None:
        pe = sinusoidal_encoding(seq_len, d_model)

    if pe.shape != (seq_len, d_model):
        raise ValueError(f"PE shape {pe.shape} != (seq_len={seq_len}, d_model={d_model})")

    return x + pe  # broadcasting handles batched input (N, seq_len, d_model)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    # Shape
    pe = sinusoidal_encoding(max_len=100, d_model=64)
    assert pe.shape == (100, 64), f"Shape: {pe.shape}"

    # Even dims are sin, odd dims are cos
    # At position 0: sin(0) = 0 and cos(0) = 1
    assert np.allclose(pe[0, 0::2], 0.0, atol=1e-6), "Even dims at pos=0 should be 0"
    assert np.allclose(pe[0, 1::2], 1.0, atol=1e-6), "Odd dims at pos=0 should be 1"

    # Values are bounded in [-1, 1]
    assert pe.min() >= -1.0 - 1e-9 and pe.max() <= 1.0 + 1e-9

    # Different positions have different encodings
    assert not np.allclose(pe[0], pe[1]), "Positions 0 and 1 must differ"

    # Learned PE
    lpe = LearnedPositionalEncoding(max_len=50, d_model=16)
    emb = lpe.forward(10)
    assert emb.shape == (10, 16)

    # add_positional_encoding: 2D input
    x = np.ones((10, 16))
    out = add_positional_encoding(x)
    assert out.shape == (10, 16)
    assert not np.allclose(out, x), "PE should change values"

    # add_positional_encoding: 3D batched input
    x_batch = np.ones((4, 10, 16))
    pe16 = sinusoidal_encoding(10, 16)
    out_batch = add_positional_encoding(x_batch, pe16)
    assert out_batch.shape == (4, 10, 16)

    # Verify all batch samples got same PE added
    for i in range(1, 4):
        assert np.allclose(out_batch[0], out_batch[i]), "All batch items should get same PE"

    print("  positional_encoding: all tests passed")


if __name__ == "__main__":
    _test()
