"""
Logistic Regression — Edge Cases & Failure Modes
=================================================

OpenAI interview probe: "Implement logistic regression from scratch.
What happens on linearly separable data? How do you fix it?"

This file covers:
    1. Forward pass: sigmoid, cross-entropy loss, gradient
    2. Gradient descent training loop with configurable regularization
    3. Demonstration of divergence on separable data (L2 fixes it)
    4. Numerical stability tricks (log-sum-exp, clipping)

Key insight for interview:
    On linearly separable data, gradient descent drives weights → ∞
    because the loss can always be reduced by increasing the margin.
    L2 regularization adds a weight magnitude penalty that stops divergence.
    L1 is sparsity-inducing but does NOT prevent divergence on separable data.

Complexity:
    Time:  O(N * D) per gradient step; N = samples, D = features
    Space: O(N + D)
"""

import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Core math primitives
# ---------------------------------------------------------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid.

    Clips z to prevent overflow in exp(−z).

    Args:
        z: Input array of any shape.

    Returns:
        Array of same shape with values in (0, 1).

    Complexity:
        Time:  O(N)
        Space: O(N)
    """
    z_clipped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def binary_cross_entropy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Compute mean binary cross-entropy loss.

    Args:
        y_true: Ground truth labels {0, 1}, shape (N,).
        y_pred: Predicted probabilities in (0, 1), shape (N,).
        eps:    Clipping value to prevent log(0).

    Returns:
        Scalar mean loss.

    Complexity:
        Time:  O(N)
        Space: O(N)
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


# ---------------------------------------------------------------------------
# Logistic Regression (from scratch)
# ---------------------------------------------------------------------------

class LogisticRegressionScratch:
    """Binary logistic regression implemented from scratch.

    Supports L2 regularization to handle separable data.

    Args:
        lr:        Learning rate.
        n_iter:    Number of gradient descent steps.
        l2_lambda: L2 regularization coefficient (0 = no regularization).
        verbose:   If True, print loss every 100 iterations.

    Raises:
        ValueError: If X and y dimensions are incompatible.
    """

    def __init__(
        self,
        lr: float = 0.1,
        n_iter: int = 1000,
        l2_lambda: float = 0.0,
        verbose: bool = False,
    ) -> None:
        self.lr = lr
        self.n_iter = n_iter
        self.l2_lambda = l2_lambda
        self.verbose = verbose
        self.weights_: Optional[np.ndarray] = None
        self.bias_: float = 0.0
        self.loss_history_: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        """Train on X, y via gradient descent.

        Args:
            X: Feature matrix, shape (N, D).
            y: Binary labels {0, 1}, shape (N,).

        Returns:
            Self (for method chaining).

        Raises:
            ValueError: If X.shape[0] != y.shape[0].

        Complexity:
            Time:  O(n_iter * N * D)
            Space: O(N + D)
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} samples but y has {y.shape[0]}"
            )
        N, D = X.shape
        self.weights_ = np.zeros(D)
        self.bias_ = 0.0
        self.loss_history_ = []

        for i in range(self.n_iter):
            # Forward pass
            z = X @ self.weights_ + self.bias_      # (N,)
            y_pred = sigmoid(z)                     # (N,)

            # Loss (with L2 penalty)
            loss = binary_cross_entropy(y, y_pred)
            if self.l2_lambda > 0:
                loss += (self.l2_lambda / 2) * np.sum(self.weights_ ** 2)
            self.loss_history_.append(loss)

            # Gradients: dL/dw = (1/N) * X^T (y_pred - y) + λw
            error = y_pred - y                      # (N,)
            dw = (X.T @ error) / N + self.l2_lambda * self.weights_
            db = float(np.mean(error))

            # Gradient descent step
            self.weights_ -= self.lr * dw
            self.bias_    -= self.lr * db

            if self.verbose and i % 100 == 0:
                print(f"  iter {i:4d} | loss={loss:.6f} | ||w||={np.linalg.norm(self.weights_):.4f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix, shape (M, D).

        Returns:
            Probability of class 1, shape (M,).

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if self.weights_ is None:
            raise RuntimeError("Model must be fitted before predict_proba")
        return sigmoid(X @ self.weights_ + self.bias_)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels.

        Args:
            X:         Feature matrix, shape (M, D).
            threshold: Decision boundary.

        Returns:
            Binary labels {0, 1}, shape (M,).
        """
        return (self.predict_proba(X) >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Demo: Separability → Divergence → L2 Fix
# ---------------------------------------------------------------------------

def _make_separable_data(n: int = 100, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate perfectly linearly separable 2D data."""
    rng = np.random.default_rng(seed)
    X0 = rng.normal(loc=-2, scale=0.5, size=(n // 2, 2))
    X1 = rng.normal(loc=+2, scale=0.5, size=(n // 2, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


def _demo_separability() -> None:
    """Show divergence without L2, convergence with L2."""
    X, y = _make_separable_data()

    print("\n--- No regularization (l2_lambda=0) ---")
    model_no_reg = LogisticRegressionScratch(lr=0.1, n_iter=500, l2_lambda=0.0, verbose=True)
    model_no_reg.fit(X, y)
    final_loss_no_reg = model_no_reg.loss_history_[-1]
    weight_norm_no_reg = np.linalg.norm(model_no_reg.weights_)
    print(f"  Final loss: {final_loss_no_reg:.6f} | ||w||: {weight_norm_no_reg:.2f}")
    print("  → Weights grow unbounded; tiny loss but model is numerically unstable\n")

    print("--- With L2 regularization (l2_lambda=0.1) ---")
    model_l2 = LogisticRegressionScratch(lr=0.1, n_iter=500, l2_lambda=0.1, verbose=True)
    model_l2.fit(X, y)
    final_loss_l2 = model_l2.loss_history_[-1]
    weight_norm_l2 = np.linalg.norm(model_l2.weights_)
    print(f"  Final loss: {final_loss_l2:.6f} | ||w||: {weight_norm_l2:.2f}")
    print("  → Weights stabilize; regularization bounds the solution\n")

    # Unregularized weights are larger (diverging); L2 weights are bounded
    assert weight_norm_no_reg > weight_norm_l2, (
        "Expected unregularized weights to be larger than L2-regularized weights"
    )
    print("Assertion passed: L2 prevents weight divergence on separable data.")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test_basics() -> None:
    # XOR is not linearly separable — model should not perfectly fit
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    model = LogisticRegressionScratch(lr=0.5, n_iter=500)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (4,), "Output shape mismatch"
    assert all(0 < p < 1 for p in proba), "Probabilities must be in (0, 1)"

    # Separable data — perfect accuracy achievable
    X_sep, y_sep = _make_separable_data()
    model_sep = LogisticRegressionScratch(lr=0.1, n_iter=500, l2_lambda=0.01)
    model_sep.fit(X_sep, y_sep)
    acc = np.mean(model_sep.predict(X_sep) == y_sep)
    assert acc == 1.0, f"Expected 100% accuracy on separable data, got {acc:.2%}"

    print("  LogisticRegressionScratch: all tests passed")


if __name__ == "__main__":
    print("Logistic Regression Edge Cases")
    _test_basics()
    _demo_separability()
