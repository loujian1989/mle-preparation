"""
Adam Optimizer — Transformer Component
=======================================

Meta ML coding probe: "Implement the Adam optimizer from scratch. Why is bias
correction needed, and what goes wrong without it?"

Algorithm (Kingma & Ba, 2014):
    Initialize: m_0 = 0, v_0 = 0, t = 0
    For each step:
        t += 1
        g_t   = gradient at step t
        m_t   = beta1 * m_{t-1} + (1 - beta1) * g_t          [1st moment: momentum]
        v_t   = beta2 * v_{t-1} + (1 - beta2) * g_t^2        [2nd moment: adaptive lr]
        m_hat = m_t / (1 - beta1^t)                           [bias correction]
        v_hat = v_t / (1 - beta2^t)                           [bias correction]
        theta -= lr * m_hat / (sqrt(v_hat) + eps)

Why bias correction?
    At step 1: m_1 = (1-beta1)*g is much smaller than the true gradient if beta1~0.9.
    The estimates m and v are biased toward 0 early in training.
    Dividing by (1 - beta^t) scales them up to unbiased estimates, especially
    critical in the first ~10 steps. Without it, Adam takes very small steps early on.

Common defaults:
    lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8

Why eps?
    Prevents division by zero when v_hat is near 0 (e.g., sparse gradients).

Complexity:
    Per-step update: O(P) where P = total parameter count
    Space:           O(P) for m and v moment buffers
"""

import numpy as np
from typing import Dict, List


class AdamOptimizer:
    """Adam optimizer: adaptive learning rates per parameter with momentum.

    Args:
        params:    Dict of parameter name -> numpy array.
        lr:        Learning rate (step size). Default 1e-3.
        beta1:     Exponential decay for 1st moment (momentum). Default 0.9.
        beta2:     Exponential decay for 2nd moment (RMSProp-like). Default 0.999.
        eps:       Small constant for numerical stability. Default 1e-8.
        weight_decay: L2 regularization coefficient. Default 0 (off).

    Raises:
        ValueError: If lr, beta1, beta2, or eps are out of valid range.
    """

    def __init__(
        self,
        params: Dict[str, np.ndarray],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if not (0 < lr):
            raise ValueError(f"lr must be > 0, got {lr}")
        if not (0 <= beta1 < 1):
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if not (0 <= beta2 < 1):
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")

        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0  # step counter

        # Initialize moment buffers to zero (same shape as each param)
        self.m: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in params.items()}
        self.v: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads: Dict[str, np.ndarray]) -> None:
        """Apply one Adam update step to all parameters.

        Args:
            grads: Dict of parameter name -> gradient array.
                   Must contain all keys in self.params.

        Raises:
            KeyError: If a parameter key is missing from grads.

        Complexity:
            Time:  O(P) where P = total number of parameters
            Space: O(1) additional
        """
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t  # bias correction denominator for m
        bc2 = 1.0 - self.beta2 ** self.t  # bias correction denominator for v

        for name, param in self.params.items():
            if name not in grads:
                raise KeyError(f"Gradient missing for parameter: {name!r}")

            g = grads[name]

            # Optional L2 regularization: add weight_decay * param to gradient
            if self.weight_decay > 0:
                g = g + self.weight_decay * param

            # Update biased moment estimates
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * g ** 2

            # Bias-corrected estimates
            m_hat = self.m[name] / bc1
            v_hat = self.v[name] / bc2

            # Parameter update
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        """Reset moment buffers (call if restarting optimization)."""
        self.t = 0
        for name in self.m:
            self.m[name][:] = 0
            self.v[name][:] = 0


# ---------------------------------------------------------------------------
# Comparison: SGD vs Adam on a simple quadratic
# ---------------------------------------------------------------------------

def _compare_sgd_vs_adam(num_steps: int = 200) -> tuple[List[float], List[float]]:
    """Compare SGD and Adam convergence on f(x) = ||x||^2 (optimal at 0).

    Args:
        num_steps: Number of optimization steps.

    Returns:
        (sgd_losses, adam_losses) — loss trajectories.
    """
    # Minimize ||x||^2; gradient = 2x
    x_sgd = np.array([2.0, -3.0, 1.5])
    x_adam = x_sgd.copy()

    adam = AdamOptimizer({"x": x_adam}, lr=0.1)
    sgd_lr = 0.01

    sgd_losses, adam_losses = [], []

    for _ in range(num_steps):
        # SGD
        g_sgd = 2.0 * x_sgd
        x_sgd -= sgd_lr * g_sgd
        sgd_losses.append(float(np.sum(x_sgd ** 2)))

        # Adam
        g_adam = 2.0 * x_adam
        adam.step({"x": g_adam})
        adam_losses.append(float(np.sum(x_adam ** 2)))

    return sgd_losses, adam_losses


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    rng = np.random.default_rng(42)

    # Basic parameter update
    params = {"W": rng.standard_normal((4, 4)), "b": np.zeros(4)}
    opt = AdamOptimizer(params, lr=0.01)

    # Parameters should change after one step
    W_before = params["W"].copy()
    grads = {"W": rng.standard_normal((4, 4)), "b": rng.standard_normal(4)}
    opt.step(grads)
    assert not np.allclose(params["W"], W_before), "Parameters should change after step"

    # Step counter increments
    assert opt.t == 1
    opt.step(grads)
    assert opt.t == 2

    # Bias correction: step 1 with beta1=0.9 should give effective lr close to
    # lr / sqrt(1 - beta2) * (1 - beta1) which is larger than without correction
    params2 = {"x": np.array([1.0])}
    opt2 = AdamOptimizer(params2, lr=0.1, beta1=0.9, beta2=0.999)
    x_before = params2["x"].copy()
    opt2.step({"x": np.array([1.0])})
    update = float(abs(params2["x"][0] - x_before[0]))
    # Update should be close to lr (bias correction makes it lr * ~1)
    assert 0.05 < update < 0.2, f"Update magnitude seems off: {update:.4f}"

    # Convergence: optimize ||x||^2 to near 0
    x = rng.standard_normal(10) * 5
    params3 = {"x": x}
    opt3 = AdamOptimizer(params3, lr=0.1)
    for _ in range(500):
        g = 2.0 * params3["x"]
        opt3.step({"x": g})
    final_loss = float(np.sum(params3["x"] ** 2))
    assert final_loss < 0.01, f"Failed to converge: loss = {final_loss:.4f}"

    # Adam vs SGD comparison
    sgd_losses, adam_losses = _compare_sgd_vs_adam(200)
    # Adam should converge faster (lower final loss) with its default lr=0.1
    assert adam_losses[-1] < sgd_losses[-1], (
        f"Adam ({adam_losses[-1]:.4f}) should beat SGD ({sgd_losses[-1]:.4f})"
    )

    # ValueError on bad inputs
    try:
        AdamOptimizer({}, lr=-0.1)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass

    print("  AdamOptimizer: all tests passed")
    print(f"    After 200 steps: Adam loss={adam_losses[-1]:.6f}, SGD loss={sgd_losses[-1]:.6f}")


if __name__ == "__main__":
    _test()
