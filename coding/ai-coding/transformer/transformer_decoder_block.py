"""
Transformer Decoder Block — Transformer Component
==================================================

Meta ML coding probe: "What are the three sublayers in a Transformer decoder block?
How does cross-attention differ from self-attention?"

Architecture (Pre-LN variant):
    x = x + MaskedMHA(LayerNorm(x))              # sublayer 1: masked self-attention
    x = x + CrossMHA(LayerNorm(x), enc_output)   # sublayer 2: cross-attention
    x = x + FFN(LayerNorm(x))                    # sublayer 3: feed-forward

Sublayer 1 — Masked Self-Attention:
    Causal mask prevents position i from attending to j > i.
    Q = K = V = decoder input.

Sublayer 2 — Cross-Attention (Encoder-Decoder Attention):
    Q comes from decoder; K and V come from encoder output.
    This is what allows the decoder to "look at" the input sequence.
    No causal mask needed here (encoder output is fully visible).

Sublayer 3 — FFN:
    Same as encoder: position-wise Linear -> ReLU -> Linear.

Key interview points:
    - Decoder is auto-regressive: generates one token at a time at inference
    - During training: teacher forcing lets all positions train in parallel
    - Cross-attention Q has shape (N, tgt_seq, d_model);
      K and V have shape (N, src_seq, d_model) — can differ from tgt_seq
    - KV cache optimization: cache K and V of past positions at inference

Complexity:
    Forward:  O(N * tgt_seq^2 * d_model) — masked self-attn
              O(N * tgt_seq * src_seq * d_model) — cross-attn
    Space:    O(N * max(tgt_seq^2, tgt_seq * src_seq))
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from layer_normalization import LayerNorm
from multi_head_attention import MultiHeadAttention
from scaled_dot_product_attention import make_causal_mask
from transformer_encoder_block import FeedForward


class TransformerDecoderBlock:
    """Single Transformer decoder block using Pre-LN architecture.

    Sublayers:
        1. Pre-LN Masked Multi-Head Self-Attention + residual
        2. Pre-LN Cross-Attention (encoder-decoder) + residual
        3. Pre-LN Feed-Forward Network + residual

    Args:
        d_model:    Model dimension.
        num_heads:  Number of attention heads.
        d_ff:       Feed-forward hidden dimension (default 4 * d_model).
        dropout:    Dropout rate.
        seed:       RNG seed.

    Raises:
        ValueError: If d_model % num_heads != 0.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = 0,
        dropout: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.dropout_rate = dropout
        d_ff = d_ff or 4 * d_model

        self.norm1 = LayerNorm(d_model)   # for masked self-attention
        self.norm2 = LayerNorm(d_model)   # for cross-attention
        self.norm3 = LayerNorm(d_model)   # for FFN

        self.self_attn = MultiHeadAttention(d_model, num_heads, seed=seed)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, seed=seed + 1)
        self.ffn = FeedForward(d_model, d_ff, seed=seed + 2)

    def forward(
        self,
        x: np.ndarray,
        encoder_output: np.ndarray,
        src_mask: "np.ndarray | None" = None,
        training: bool = False,
    ) -> np.ndarray:
        """Apply decoder block.

        Args:
            x:              Decoder input, shape (N, tgt_seq, d_model).
            encoder_output: Encoder output, shape (N, src_seq, d_model).
            src_mask:       Optional padding mask for encoder output.
                            Shape broadcastable to (N, num_heads, tgt_seq, src_seq).
            training:       If True, apply dropout.

        Returns:
            Output of shape (N, tgt_seq, d_model).

        Complexity:
            Time:  O(N * tgt_seq^2 * d_k + N * tgt_seq * src_seq * d_k)
            Space: O(N * (tgt_seq^2 + tgt_seq * src_seq))
        """
        tgt_seq = x.shape[1]

        # Sublayer 1: masked self-attention (causal)
        causal_mask = make_causal_mask(tgt_seq)                    # (1, tgt_seq, tgt_seq)
        normed = self.norm1.forward(x)
        self_attn_out, _ = self.self_attn.forward(normed, normed, normed, mask=causal_mask)
        if training and self.dropout_rate > 0:
            self_attn_out = self._dropout(self_attn_out)
        x = x + self_attn_out

        # Sublayer 2: cross-attention (query from decoder, key/value from encoder)
        normed2 = self.norm2.forward(x)
        cross_out, _ = self.cross_attn.forward(
            normed2, encoder_output, encoder_output, mask=src_mask
        )
        if training and self.dropout_rate > 0:
            cross_out = self._dropout(cross_out)
        x = x + cross_out

        # Sublayer 3: feed-forward
        normed3 = self.norm3.forward(x)
        ffn_out = self.ffn.forward(normed3)
        if training and self.dropout_rate > 0:
            ffn_out = self._dropout(ffn_out)
        x = x + ffn_out

        return x

    def _dropout(self, x: np.ndarray) -> np.ndarray:
        """Apply inverted dropout.

        Args:
            x: Input array.

        Returns:
            Dropped-out array.
        """
        mask = np.random.binomial(1, 1.0 - self.dropout_rate, x.shape)
        return x * mask / (1.0 - self.dropout_rate + 1e-9)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    rng = np.random.default_rng(42)
    N, tgt_seq, src_seq, d_model, num_heads = 2, 5, 7, 32, 4

    block = TransformerDecoderBlock(d_model=d_model, num_heads=num_heads)

    x = rng.standard_normal((N, tgt_seq, d_model))
    enc_out = rng.standard_normal((N, src_seq, d_model))

    # Shape preservation
    out = block.forward(x, enc_out)
    assert out.shape == (N, tgt_seq, d_model), f"Shape: {out.shape}"

    # Output differs from input
    assert not np.allclose(out, x), "Decoder output should differ from input"

    # Causal property: output at position i should not depend on positions > i
    # Test: perturb x at position j=4 and verify position i=0 output unchanged
    x_perturbed = x.copy()
    x_perturbed[:, 4, :] += rng.standard_normal((N, d_model)) * 10
    out_perturbed = block.forward(x_perturbed, enc_out)
    # Position 0 output must be unchanged (causal mask blocks future)
    assert np.allclose(out[:, 0, :], out_perturbed[:, 0, :], atol=1e-6), (
        "Causal property violated: position 0 changed when position 4 was perturbed"
    )
    # But later positions should change (position 4 can see positions 0-4)
    # Position 4 sees its own input so it must differ
    # (Note: position 3 is also affected by position 4 change in encoder path
    #  only if they share encoder output — here they don't change enc_out)

    # Different encoder outputs -> different decoder outputs
    enc_out2 = rng.standard_normal((N, src_seq, d_model))
    out2 = block.forward(x, enc_out2)
    assert not np.allclose(out, out2), "Different encoder outputs should produce different decoder outputs"

    # Variable src/tgt lengths
    block2 = TransformerDecoderBlock(d_model=16, num_heads=2)
    x_short = rng.standard_normal((1, 3, 16))
    enc_long = rng.standard_normal((1, 10, 16))
    out_cross = block2.forward(x_short, enc_long)
    assert out_cross.shape == (1, 3, 16), f"Cross-attention shape wrong: {out_cross.shape}"

    print("  TransformerDecoderBlock: all tests passed")


if __name__ == "__main__":
    _test()
