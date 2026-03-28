"""
Batch Normalization — Implementation & Training vs. Inference Behavior
======================================================================

OpenAI interview probe: "Implement batch normalization. Explain what changes
between training and inference. Why does BN fail with batch_size=1?"

Key interview points:
    TRAINING:
        - Normalize using batch statistics: μ_B, σ²_B (computed from current batch)
        - Update running_mean and running_var with momentum for inference use
        - γ (scale) and β (shift) are learnable; BN does NOT remove expressiveness

    INFERENCE:
        - Use running_mean and running_var (accumulated over training)
        - Never use batch stats — batch may be size 1; also need determinism
        - Frozen: no updates to running stats

    FAILURE MODES:
        - Batch size 1: σ²_B = 0 → division by zero (use LayerNorm instead)
        - Very small batch: noisy estimates of μ_B → unstable training
        - Recurrent networks: different sequence lengths break batch stats

    ALTERNATIVES:
        - LayerNorm: normalize over feature dim (not batch dim) — preferred for NLP/LLMs
        - GroupNorm: normalize over feature groups — stable for small batches
        - InstanceNorm: per-sample, per-channel — used in style transfer

    IMPLEMENTATION NOTES:
        - eps added to variance BEFORE sqrt to prevent division by zero
        - Running stats updated as: running = momentum * running + (1-momentum) * batch_stat
        - Backward: gradient flows through γ AND through the normalization (non-trivial)

Complexity:
    Forward  (train):     O(N * D)
    Forward  (inference): O(N * D)
    Backward:             O(N * D)
    Space:                O(D) for running stats + γ, β
"""

import numpy as np
from typing import Optional, Tuple


EPS = 1e-5
MOMENTUM = 0.1


class BatchNorm1d:
    """Batch normalization for 2D input (N, D).

    Args:
        num_features: D — number of features (channels).
        eps:          Added to variance for numerical stability.
        momentum:     Weight for exponential moving average of running stats.

    Raises:
        ValueError: If num_features < 1.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = EPS,
        momentum: float = MOMENTUM,
    ) -> None:
        if num_features < 1:
            raise ValueError(f"num_features must be >= 1, got {num_features}")
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = True

        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        # Running statistics (used at inference)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        # Cache for backward pass
        self._cache: Optional[dict] = None

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Apply batch normalization.

        Args:
            X: Input, shape (N, D). N must be > 1 in training mode.

        Returns:
            Normalized output, shape (N, D).

        Raises:
            ValueError: If X.ndim != 2 or N == 1 in training mode.

        Complexity:
            Time:  O(N * D)
            Space: O(N * D) for cache; O(D) running stats
        """
        if X.ndim != 2:
            raise ValueError(f"Expected 2D input (N, D), got shape {X.shape}")
        N, D = X.shape
        if D != self.num_features:
            raise ValueError(f"Expected D={self.num_features}, got {D}")

        if self.training:
            if N == 1:
                raise ValueError(
                    "Batch size = 1 in training mode: σ²_B = 0 → division by zero. "
                    "Use LayerNorm or GroupNorm instead."
                )
            # Compute batch statistics
            mu = X.mean(axis=0)                        # (D,)
            var = X.var(axis=0)                        # (D,) — population var (ddof=0)
            X_centered = X - mu
            std_inv = 1.0 / np.sqrt(var + self.eps)
            X_hat = X_centered * std_inv               # (N, D) — normalized

            # Update running stats (exponential moving average)
            self.running_mean = (
                (1 - self.momentum) * self.running_mean + self.momentum * mu
            )
            self.running_var = (
                (1 - self.momentum) * self.running_var + self.momentum * var
            )

            # Cache for backward
            self._cache = {
                "X_hat": X_hat,
                "X_centered": X_centered,
                "std_inv": std_inv,
                "N": N,
            }
        else:
            # Inference: use accumulated running statistics
            X_hat = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)
            self._cache = None  # no backward at inference

        return self.gamma * X_hat + self.beta   # scale + shift

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backpropagate through batch normalization.

        Gradient derivation (Ioffe & Szegedy, 2015):
            dL/dγ = Σ_i dout_i * X_hat_i
            dL/dβ = Σ_i dout_i
            dL/dX = (1/N) * σ_inv * [N * dX_hat - Σ dX_hat - X_hat * Σ(dX_hat * X_hat)]

        Args:
            dout: Upstream gradient, shape (N, D).

        Returns:
            Tuple (dX, dgamma, dbeta).

        Raises:
            RuntimeError: If called outside of training mode or before forward.

        Complexity:
            Time:  O(N * D)
            Space: O(N * D)
        """
        if not self.training or self._cache is None:
            raise RuntimeError("backward() requires training mode and a prior forward() call")

        X_hat = self._cache["X_hat"]
        X_centered = self._cache["X_centered"]
        std_inv = self._cache["std_inv"]
        N = self._cache["N"]

        dgamma = (dout * X_hat).sum(axis=0)         # (D,)
        dbeta = dout.sum(axis=0)                    # (D,)

        # Gradient w.r.t. X_hat
        dX_hat = dout * self.gamma                  # (N, D)

        # Gradient through normalization (chain rule — 3-term decomposition)
        dX = (
            std_inv / N
            * (
                N * dX_hat
                - dX_hat.sum(axis=0)
                - X_hat * (dX_hat * X_hat).sum(axis=0)
            )
        )

        return dX, dgamma, dbeta

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def train(self) -> "BatchNorm1d":
        """Switch to training mode (updates running stats, uses batch stats)."""
        self.training = True
        return self

    def eval(self) -> "BatchNorm1d":
        """Switch to inference mode (uses running stats, no cache)."""
        self.training = False
        return self


# ---------------------------------------------------------------------------
# LayerNorm for comparison (normalizes over feature dim, not batch)
# ---------------------------------------------------------------------------

class LayerNorm1d:
    """Layer normalization for 2D input (N, D).

    Normalizes each sample independently — safe for batch_size=1.
    Preferred for Transformers, RNNs, and small-batch settings.

    Args:
        num_features: D — number of features.
        eps:          Numerical stability.
    """

    def __init__(self, num_features: int, eps: float = EPS) -> None:
        self.num_features = num_features
        self.eps = eps
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Apply layer normalization.

        Args:
            X: Input, shape (N, D).

        Returns:
            Normalized output, shape (N, D).

        Complexity:
            Time:  O(N * D)
            Space: O(N * D)
        """
        mu = X.mean(axis=1, keepdims=True)         # per-sample mean
        var = X.var(axis=1, keepdims=True)          # per-sample variance
        X_hat = (X - mu) / np.sqrt(var + self.eps)
        return self.gamma * X_hat + self.beta


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test_batch_norm_training() -> None:
    rng = np.random.default_rng(0)
    N, D = 16, 8
    X = rng.standard_normal((N, D)) * 5 + 3   # mean=3, std=5

    bn = BatchNorm1d(num_features=D)
    out = bn.forward(X)

    # Normalized output should have ~zero mean and ~unit variance
    assert out.shape == (N, D)
    assert np.abs(out.mean(axis=0)).max() < 0.1, "BN output should have ~zero mean"
    assert np.abs(out.var(axis=0) - 1.0).max() < 0.1, "BN output should have ~unit variance"

    # Running stats should update
    assert not np.allclose(bn.running_mean, 0), "Running mean should update from zero"

    print("  BatchNorm1d training mode: passed")


def _test_batch_norm_inference() -> None:
    rng = np.random.default_rng(1)
    N, D = 16, 4
    X = rng.standard_normal((N, D))

    bn = BatchNorm1d(num_features=D)
    # Warm up running stats
    for _ in range(100):
        bn.forward(rng.standard_normal((N, D)))

    bn.eval()
    out = bn.forward(X)  # uses running stats, not batch stats
    assert out.shape == (N, D)
    print("  BatchNorm1d inference mode: passed")


def _test_batch_size_1_fails() -> None:
    bn = BatchNorm1d(num_features=4)
    X = np.ones((1, 4))
    try:
        bn.forward(X)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Batch size = 1" in str(e)
    print("  BatchNorm1d batch_size=1 guard: passed")


def _test_layer_norm() -> None:
    rng = np.random.default_rng(2)
    X = rng.standard_normal((1, 8))  # batch_size=1 — valid for LayerNorm
    ln = LayerNorm1d(num_features=8)
    out = ln.forward(X)
    assert out.shape == (1, 8)
    assert np.abs(out.mean()) < 0.1
    print("  LayerNorm1d batch_size=1: passed (vs. BatchNorm failure)")


if __name__ == "__main__":
    print("Batch Normalization")
    _test_batch_norm_training()
    _test_batch_norm_inference()
    _test_batch_size_1_fails()
    _test_layer_norm()
