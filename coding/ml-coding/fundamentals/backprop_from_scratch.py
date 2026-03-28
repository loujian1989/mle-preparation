"""
Backpropagation from Scratch — 2-Layer MLP
==========================================

OpenAI interview probe: "Implement forward and backward pass for a 2-layer MLP.
Walk me through the gradient computation for each parameter."

Architecture:
    Input(D) → Linear(D→H) → ReLU → Linear(H→C) → Softmax → Cross-Entropy Loss

This file implements:
    1. Forward pass with cached activations (required for backward)
    2. Backward pass: analytic gradients via chain rule
    3. Gradient check vs. numerical gradients (correctness proof)
    4. Mini-batch SGD training loop on XOR / MNIST-like synthetic data

Key interview points:
    - Cache activations during forward pass — needed in backward
    - ReLU gradient = 1 if z > 0 else 0 (not differentiable at 0; convention: 0)
    - Softmax + cross-entropy gradient simplifies to (y_pred - y_true) / N
    - Vanishing gradients: ReLU > sigmoid for depth, but dying ReLU if init bad
    - Weight init: He (ReLU layers) vs. Xavier (tanh/sigmoid layers)

Complexity:
    Forward:  O(N * D * H + N * H * C)
    Backward: O(N * D * H + N * H * C)  — same as forward
    Space:    O(N * H) for activation cache
"""

import numpy as np
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def relu(z: np.ndarray) -> np.ndarray:
    """ReLU activation: max(0, z).

    Complexity:
        Time:  O(N)
        Space: O(N)
    """
    return np.maximum(0, z)


def relu_grad(z: np.ndarray) -> np.ndarray:
    """Gradient of ReLU: 1 if z > 0 else 0.

    Args:
        z: Pre-activation values (before ReLU was applied).

    Returns:
        Binary mask, same shape as z.
    """
    return (z > 0).astype(float)


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax (subtract row max before exp).

    Args:
        z: Logits, shape (N, C).

    Returns:
        Probabilities, shape (N, C), rows sum to 1.

    Complexity:
        Time:  O(N * C)
        Space: O(N * C)
    """
    z_shifted = z - z.max(axis=1, keepdims=True)  # subtract max for stability
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def cross_entropy_loss(probs: np.ndarray, y: np.ndarray) -> float:
    """Mean cross-entropy loss for multi-class classification.

    Args:
        probs: Predicted probabilities, shape (N, C).
        y:     Integer class labels, shape (N,).

    Returns:
        Scalar mean loss.

    Complexity:
        Time:  O(N)
        Space: O(N)
    """
    N = probs.shape[0]
    eps = 1e-12
    log_probs = np.log(np.clip(probs[np.arange(N), y], eps, 1.0))
    return float(-np.mean(log_probs))


# ---------------------------------------------------------------------------
# 2-Layer MLP
# ---------------------------------------------------------------------------

class MLP2Layer:
    """Two-layer MLP: Linear → ReLU → Linear → Softmax.

    Args:
        input_dim:   D — number of input features.
        hidden_dim:  H — hidden layer width.
        output_dim:  C — number of output classes.
        lr:          Learning rate for SGD.
        seed:        Random seed for reproducibility.

    Raises:
        ValueError: If any dimension < 1.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        lr: float = 0.01,
        seed: int = 42,
    ) -> None:
        if any(d < 1 for d in [input_dim, hidden_dim, output_dim]):
            raise ValueError("All dimensions must be >= 1")
        self.lr = lr
        rng = np.random.default_rng(seed)

        # He initialization for ReLU layers: scale = sqrt(2 / fan_in)
        self.params: Dict[str, np.ndarray] = {
            "W1": rng.standard_normal((input_dim, hidden_dim)) * np.sqrt(2.0 / input_dim),
            "b1": np.zeros(hidden_dim),
            "W2": rng.standard_normal((hidden_dim, output_dim)) * np.sqrt(2.0 / hidden_dim),
            "b2": np.zeros(output_dim),
        }
        self._cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute forward pass and cache activations for backward.

        Args:
            X: Input batch, shape (N, D).

        Returns:
            Class probabilities, shape (N, C).

        Complexity:
            Time:  O(N * D * H + N * H * C)
            Space: O(N * H) for cache
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]

        z1 = X @ W1 + b1          # (N, H) — pre-activation layer 1
        a1 = relu(z1)             # (N, H) — post-activation layer 1
        z2 = a1 @ W2 + b2         # (N, C) — logits
        probs = softmax(z2)       # (N, C) — output probabilities

        # Cache everything needed for backward pass
        self._cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "probs": probs}
        return probs

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute gradients via backpropagation.

        Derivation (softmax + cross-entropy combined gradient):
            dL/dz2 = (probs - one_hot(y)) / N   [softmax-CE shortcut]

        Then chain rule back through W2, ReLU, W1.

        Args:
            y: Integer class labels, shape (N,).

        Returns:
            Dict of gradients: dW1, db1, dW2, db2.

        Raises:
            RuntimeError: If forward() has not been called.

        Complexity:
            Time:  O(N * D * H + N * H * C)
            Space: O(N * H)
        """
        if not self._cache:
            raise RuntimeError("Must call forward() before backward()")

        X, z1, a1 = self._cache["X"], self._cache["z1"], self._cache["a1"]
        probs = self._cache["probs"]
        N = X.shape[0]
        C = probs.shape[1]

        # --- Layer 2 gradients ---
        # One-hot encode y
        one_hot = np.zeros((N, C))
        one_hot[np.arange(N), y] = 1.0

        # Softmax + CE combined: dL/dz2 = (probs - one_hot) / N
        dz2 = (probs - one_hot) / N            # (N, C)
        dW2 = a1.T @ dz2                       # (H, C)
        db2 = dz2.sum(axis=0)                  # (C,)

        # --- Layer 1 gradients ---
        da1 = dz2 @ self.params["W2"].T        # (N, H)
        dz1 = da1 * relu_grad(z1)              # (N, H) — ReLU gate
        dW1 = X.T @ dz1                        # (D, H)
        db1 = dz1.sum(axis=0)                  # (H,)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    # ------------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------------

    def step(self, grads: Dict[str, np.ndarray]) -> None:
        """Apply SGD update: param -= lr * grad.

        Args:
            grads: Gradient dict from backward().
        """
        for name in ["W1", "b1", "W2", "b2"]:
            self.params[name] -= self.lr * grads[f"d{name}"]

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """Forward + backward + step in one call.

        Args:
            X: Batch input, shape (N, D).
            y: Integer labels, shape (N,).

        Returns:
            Scalar loss for this batch.
        """
        probs = self.forward(X)
        loss = cross_entropy_loss(probs, y)
        grads = self.backward(y)
        self.step(grads)
        return loss


# ---------------------------------------------------------------------------
# Gradient check — numerical vs. analytic
# ---------------------------------------------------------------------------

def gradient_check(
    model: MLP2Layer,
    X: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-5,
    tol: float = 1e-4,
) -> None:
    """Verify analytic gradients match numerical finite-difference estimates.

    Uses central difference: df/dx ≈ [f(x+ε) - f(x-ε)] / (2ε)

    Args:
        model: Fitted MLP2Layer instance.
        X:     Input batch.
        y:     Labels.
        eps:   Finite difference step size.
        tol:   Maximum allowed relative error.

    Raises:
        AssertionError: If any gradient is incorrect.
    """
    probs = model.forward(X)
    _ = cross_entropy_loss(probs, y)
    analytic_grads = model.backward(y)

    for param_name in ["W1", "b1", "W2", "b2"]:
        param = model.params[param_name]
        analytic = analytic_grads[f"d{param_name}"]
        numeric = np.zeros_like(param)

        it = np.nditer(param, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            original = param[idx]

            param[idx] = original + eps
            loss_plus = cross_entropy_loss(model.forward(X), y)

            param[idx] = original - eps
            loss_minus = cross_entropy_loss(model.forward(X), y)

            numeric[idx] = (loss_plus - loss_minus) / (2 * eps)
            param[idx] = original  # restore
            it.iternext()

        rel_error = np.abs(analytic - numeric) / (np.abs(analytic) + np.abs(numeric) + 1e-10)
        max_err = float(rel_error.max())
        assert max_err < tol, (
            f"Gradient check FAILED for {param_name}: max relative error = {max_err:.2e} (tol={tol})"
        )
        print(f"  {param_name}: gradient check passed (max relative error = {max_err:.2e})")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test_mlp() -> None:
    # XOR with 4 classes (multi-class generalization)
    rng = np.random.default_rng(0)
    N, D, H, C = 32, 4, 8, 3
    X = rng.standard_normal((N, D))
    y = rng.integers(0, C, size=N)

    model = MLP2Layer(input_dim=D, hidden_dim=H, output_dim=C, lr=0.05)

    # Forward output shape
    probs = model.forward(X)
    assert probs.shape == (N, C), f"Expected ({N},{C}), got {probs.shape}"
    assert np.allclose(probs.sum(axis=1), 1.0), "Rows must sum to 1"

    # Training loop — loss should decrease
    losses = [model.train_step(X, y) for _ in range(200)]
    assert losses[-1] < losses[0], "Loss should decrease after 200 steps"

    print("  MLP2Layer: forward/backward/train tests passed")


def _test_gradient_check() -> None:
    # Small model for fast gradient check
    rng = np.random.default_rng(7)
    N, D, H, C = 8, 3, 4, 2
    X = rng.standard_normal((N, D))
    y = rng.integers(0, C, size=N)
    model = MLP2Layer(input_dim=D, hidden_dim=H, output_dim=C)
    gradient_check(model, X, y)
    print("  Gradient check: all parameters verified")


if __name__ == "__main__":
    print("Backpropagation from Scratch")
    _test_mlp()
    _test_gradient_check()
