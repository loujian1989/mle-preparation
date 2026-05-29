"""
Layer Normalization — Transformer Component
===========================================

Meta ML coding probe: "Implement Layer Normalization forward and backward pass
using NumPy. Validate with a gradient check."

Mathematical formulation (input x of shape (N, D)):
    mu    = mean(x, axis=-1, keepdims=True)       # (N, 1)
    var   = var(x, axis=-1, keepdims=True)         # (N, 1)
    x_hat = (x - mu) / sqrt(var + eps)             # (N, D)  normalized
    y     = gamma * x_hat + beta                   # (N, D)  rescaled

    gamma, beta: learnable parameters, shape (D,)
    eps: numerical stability constant (default 1e-5)

Difference from Batch Normalization:
    - Normalizes over feature dimension D, not the batch dimension N
    - Per-sample statistics — no running mean/var needed at inference
    - Used in Transformers; BatchNorm breaks with variable-length sequences

Backward pass key equations (chain rule):
    dgamma  = sum(dL/dy * x_hat, axis=0)
    dbeta   = sum(dL/dy, axis=0)
    dx_hat  = dL/dy * gamma
    dx = (dx_hat - mean(dx_hat) - x_hat * mean(dx_hat * x_hat)) / std
         [derived by differentiating through mean and variance]

Complexity:
    Forward:  O(N * D)
    Backward: O(N * D)
    Space:    O(N * D) for activation cache
"""

import numpy as np
from typing import Dict, Tuple


class LayerNorm:
    """Layer normalization with learnable scale (gamma) and shift (beta).

    Args:
        normalized_shape: Dimension D of the feature vector to normalize.
        eps:              Stability constant added inside sqrt.

    Raises:
        ValueError: If normalized_shape < 1.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        if normalized_shape < 1:
            raise ValueError(f"normalized_shape must be >= 1, got {normalized_shape}")
        self.eps = eps
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)
        self._cache: Dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Normalize input along last dimension, then scale and shift.

        Args:
            x: Input of shape (N, D) or (D,) for a single sample.

        Returns:
            Normalized output, same shape as x.

        Complexity:
            Time:  O(N * D)
            Space: O(N * D)
        """
        original_shape = x.shape
        x_2d = x.reshape(-1, self.gamma.shape[0])

        mu = x_2d.mean(axis=-1, keepdims=True)        # (N, 1)
        var = x_2d.var(axis=-1, keepdims=True)         # (N, 1)
        std = np.sqrt(var + self.eps)                  # (N, 1)
        x_hat = (x_2d - mu) / std                     # (N, D)
        out = self.gamma * x_hat + self.beta           # (N, D)

        self._cache = {
            "x_hat": x_hat,
            "std": std,
        }
        return out.reshape(original_shape)

    def backward(self, dy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute gradients through layer normalization.

        Derivation sketch:
            dy_hat  = dy * gamma                                  [dL/dx_hat]
            dx_hat contribution to dx involves subtracting mean terms
            to account for the fact that mu and var depend on x.

        Args:
            dy: Upstream gradient, same shape as forward output.

        Returns:
            (dx, dgamma, dbeta) gradients.

        Raises:
            RuntimeError: If forward() was not called first.

        Complexity:
            Time:  O(N * D)
            Space: O(N * D)
        """
        if not self._cache:
            raise RuntimeError("Must call forward() before backward()")

        x_hat = self._cache["x_hat"]
        std = self._cache["std"]
        dy_2d = dy.reshape(x_hat.shape)

        dgamma = (dy_2d * x_hat).sum(axis=0)   # (D,)
        dbeta = dy_2d.sum(axis=0)              # (D,)

        # Gradient through normalization (accounting for mu and var dependence on x)
        dx_hat = dy_2d * self.gamma                                          # (N, D)
        dx = (
            dx_hat
            - dx_hat.mean(axis=-1, keepdims=True)
            - x_hat * (dx_hat * x_hat).mean(axis=-1, keepdims=True)
        ) / std                                                              # (N, D)

        return dx.reshape(dy.shape), dgamma, dbeta

    def step(self, dgamma: np.ndarray, dbeta: np.ndarray, lr: float = 0.01) -> None:
        """SGD parameter update.

        Args:
            dgamma: Gradient w.r.t. gamma.
            dbeta:  Gradient w.r.t. beta.
            lr:     Learning rate.
        """
        self.gamma -= lr * dgamma
        self.beta -= lr * dbeta


# ---------------------------------------------------------------------------
# Gradient check utility
# ---------------------------------------------------------------------------

def _gradient_check(
    ln: LayerNorm,
    x: np.ndarray,
    eps: float = 1e-5,
    tol: float = 1e-4,
) -> None:
    """Verify backward() matches central finite differences on L = sum(dy * forward(x)).

    Uses random dy (not all-ones) because L = sum(x_hat) is identically 0
    (LayerNorm removes mean), which makes numeric gradients noise-dominated.

    Args:
        ln:  LayerNorm instance with current gamma/beta.
        x:   Input array.
        eps: Perturbation magnitude.
        tol: Maximum allowed relative error.

    Raises:
        AssertionError: If gradient is incorrect.
    """
    rng_gc = np.random.default_rng(123)
    dy = rng_gc.standard_normal(x.shape)  # random upstream gradient avoids degenerate L=0

    _ = ln.forward(x)
    dx_analytic, _, _ = ln.backward(dy)

    dx_numeric = np.zeros_like(x)
    gamma_copy = ln.gamma.copy()
    beta_copy = ln.beta.copy()
    eps_val = ln.eps

    for idx in np.ndindex(x.shape):
        def _fwd(inp: np.ndarray) -> float:
            mu = inp.mean(axis=-1, keepdims=True)
            var = inp.var(axis=-1, keepdims=True)
            x_hat = (inp - mu) / np.sqrt(var + eps_val)
            y = gamma_copy * x_hat + beta_copy
            return float(np.sum(dy * y))  # L = dot(dy, y)

        x_p = x.copy(); x_p[idx] += eps
        x_m = x.copy(); x_m[idx] -= eps
        dx_numeric[idx] = (_fwd(x_p) - _fwd(x_m)) / (2 * eps)

    rel_err = (
        np.abs(dx_analytic - dx_numeric)
        / (np.abs(dx_analytic) + np.abs(dx_numeric) + 1e-10)
    )
    assert rel_err.max() < tol, f"Gradient check FAILED: max error = {rel_err.max():.2e}"
    print(f"  gradient_check: passed (max relative error = {rel_err.max():.2e})")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    rng = np.random.default_rng(42)

    # Output shape preserved
    ln = LayerNorm(8)
    x = rng.standard_normal((4, 8))
    out = ln.forward(x)
    assert out.shape == (4, 8), f"Shape mismatch: {out.shape}"

    # With identity gamma/beta, normalized rows should have mean ~0 and std ~1
    row_means = out.mean(axis=-1)
    row_stds = out.std(axis=-1)
    assert np.allclose(row_means, 0, atol=1e-5), f"Row means not 0: {row_means}"
    assert np.allclose(row_stds, 1, atol=1e-5), f"Row stds not 1: {row_stds}"

    # Gradient check on small array
    ln2 = LayerNorm(4)
    x2 = rng.standard_normal((3, 4))
    _gradient_check(ln2, x2)

    # Overfit: learn to output all ones from random input
    ln3 = LayerNorm(6)
    x3 = rng.standard_normal((8, 6))
    target = np.ones((8, 6))
    loss = float("inf")
    for _ in range(1000):
        out3 = ln3.forward(x3)
        loss = float(np.mean((out3 - target) ** 2))
        dy3 = 2.0 * (out3 - target) / x3.size
        _, dgamma, dbeta = ln3.backward(dy3)
        ln3.step(dgamma, dbeta, lr=0.1)
    assert loss < 0.01, f"Failed to overfit: final loss = {loss:.4f}"

    print("  LayerNorm: all tests passed")


if __name__ == "__main__":
    _test()
