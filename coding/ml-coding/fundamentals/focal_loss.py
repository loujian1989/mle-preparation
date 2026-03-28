"""
Focal Loss — Implementation & Gradient Verification
====================================================

OpenAI / Stripe interview probe: "Implement focal loss. Why does it outperform
cross-entropy for imbalanced classification? Verify your gradient numerically."

Background:
    Standard binary cross-entropy (BCE) treats all samples equally.
    On imbalanced datasets (e.g., 1:1000 fraud rate), easy negatives
    dominate the gradient, drowning the signal from rare positives.

    Focal loss (Lin et al., RetinaNet 2017) adds a modulating factor
    (1 - p_t)^γ that down-weights easy examples:

        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    where:
        p_t = p      if y = 1
              1 - p  if y = 0
        α_t = α      if y = 1   (class-level weight for imbalance)
              1 - α  if y = 0
        γ ≥ 0        focusing parameter (γ=0 recovers BCE)

Key intuition:
    - Correctly classified easy examples (p_t → 1): (1-p_t)^γ → 0 → minimal loss contribution
    - Hard examples and misclassified examples: (1-p_t)^γ → 1 → full loss contribution
    - γ=2 is the canonical default; α=0.25 for the minority class

Complexity:
    Time:  O(N) forward + backward
    Space: O(N)
"""

import numpy as np
from typing import Optional


EPS = 1e-12


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

def focal_loss(
    y_pred_logits: np.ndarray,
    y_true: np.ndarray,
    gamma: float = 2.0,
    alpha: float = 0.25,
    reduction: str = "mean",
) -> float:
    """Compute focal loss from logits.

    Args:
        y_pred_logits: Raw model outputs (before sigmoid), shape (N,).
        y_true:        Binary labels {0, 1}, shape (N,).
        gamma:         Focusing parameter (0 = standard BCE).
        alpha:         Class weight for positive class.
        reduction:     "mean" | "sum" | "none".

    Returns:
        Scalar loss (or per-sample array if reduction="none").

    Raises:
        ValueError: If gamma < 0 or alpha not in (0, 1).

    Complexity:
        Time:  O(N)
        Space: O(N)
    """
    if gamma < 0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")

    # Sigmoid probabilities
    probs = 1.0 / (1.0 + np.exp(-np.clip(y_pred_logits, -500, 500)))  # (N,)

    # p_t: probability of the true class
    p_t = np.where(y_true == 1, probs, 1 - probs)                      # (N,)

    # α_t: per-sample class weight
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)                  # (N,)

    # Focal modulating factor
    focal_weight = (1 - p_t) ** gamma                                   # (N,)

    # Cross-entropy component (stable log via log(p_t))
    bce = -np.log(np.clip(p_t, EPS, 1.0))                              # (N,)

    per_sample_loss = alpha_t * focal_weight * bce                      # (N,)

    if reduction == "mean":
        return float(per_sample_loss.mean())
    if reduction == "sum":
        return float(per_sample_loss.sum())
    return per_sample_loss  # type: ignore[return-value]


def focal_loss_gradient(
    y_pred_logits: np.ndarray,
    y_true: np.ndarray,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> np.ndarray:
    """Analytic gradient of mean focal loss w.r.t. logits.

    Derivation via chain rule:
        p   = σ(z),  p_t = p if y=1 else 1-p,  q_t = 1-p_t

        dFL/dp_t = α_t * [γ * q_t^(γ-1) * log(p_t) - q_t^γ / p_t]
        dp_t/dz  = (2y-1) * p * (1-p)

        dLoss/dz = (1/N) * dFL/dp_t * dp_t/dz

    Args:
        y_pred_logits: Raw logits, shape (N,).
        y_true:        Binary labels {0, 1}, shape (N,).
        gamma:         Focusing parameter.
        alpha:         Positive class weight.

    Returns:
        Gradient w.r.t. logits, shape (N,), divided by N (mean reduction).

    Complexity:
        Time:  O(N)
        Space: O(N)
    """
    N = len(y_pred_logits)
    probs = 1.0 / (1.0 + np.exp(-np.clip(y_pred_logits, -500, 500)))

    p_t = np.where(y_true == 1, probs, 1 - probs)
    q_t = 1.0 - p_t                               # = 1 - p_t
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)

    log_pt = np.log(np.clip(p_t, EPS, 1.0))

    # dFL/dp_t: derivative of -α_t * q_t^γ * log(p_t) w.r.t. p_t
    #   = -α_t * [-γ * q_t^(γ-1) * log(p_t) + q_t^γ / p_t]
    #   = α_t * [γ * q_t^(γ-1) * log(p_t) - q_t^γ / p_t]
    dFL_dp_t = alpha_t * (
        gamma * (q_t ** (gamma - 1)) * log_pt
        - (q_t ** gamma) / np.clip(p_t, EPS, 1.0)
    )

    # dp_t/dz: +p(1-p) for y=1, -p(1-p) for y=0
    sign = 2.0 * y_true - 1.0
    dp_t_dz = sign * probs * (1.0 - probs)

    return dFL_dp_t * dp_t_dz / N


# ---------------------------------------------------------------------------
# Gradient verification
# ---------------------------------------------------------------------------

def verify_gradient(
    logits: np.ndarray,
    y_true: np.ndarray,
    gamma: float = 2.0,
    alpha: float = 0.25,
    eps: float = 1e-5,
    tol: float = 1e-4,
) -> None:
    """Numerical gradient check for focal_loss_gradient.

    Central difference: df/dz_i ≈ [f(z+ε*e_i) - f(z-ε*e_i)] / (2ε)

    Args:
        logits: Input logits.
        y_true: Binary labels.
        gamma:  Focusing parameter.
        alpha:  Class weight.
        eps:    Finite difference step.
        tol:    Maximum allowed relative error.

    Raises:
        AssertionError: If gradient check fails.
    """
    analytic = focal_loss_gradient(logits, y_true, gamma, alpha)
    numeric = np.zeros_like(logits, dtype=float)

    for i in range(len(logits)):
        logits_plus = logits.copy()
        logits_plus[i] += eps
        logits_minus = logits.copy()
        logits_minus[i] -= eps

        loss_plus = focal_loss(logits_plus, y_true, gamma, alpha)
        loss_minus = focal_loss(logits_minus, y_true, gamma, alpha)
        numeric[i] = (loss_plus - loss_minus) / (2 * eps)

    rel_error = np.abs(analytic - numeric) / (np.abs(analytic) + np.abs(numeric) + 1e-10)
    max_err = float(rel_error.max())
    assert max_err < tol, (
        f"Gradient check FAILED: max relative error = {max_err:.2e} (tol={tol})\n"
        f"  Analytic: {analytic}\n  Numeric:  {numeric}"
    )
    print(f"  Gradient check passed: max relative error = {max_err:.2e}")


# ---------------------------------------------------------------------------
# Comparison: Focal Loss vs. BCE on imbalanced data
# ---------------------------------------------------------------------------

def _demo_imbalanced_comparison() -> None:
    """Show focal loss down-weights easy negatives vs. BCE."""
    rng = np.random.default_rng(42)

    # Imbalanced: 990 easy negatives (logit=-3) + 10 hard positives (logit=0.1)
    neg_logits = rng.normal(-3, 0.2, 990)   # confidently predicted negative
    pos_logits = rng.normal(0.1, 0.5, 10)   # uncertain, near decision boundary

    logits = np.concatenate([neg_logits, pos_logits])
    y = np.array([0] * 990 + [1] * 10)

    bce = focal_loss(logits, y, gamma=0.0, alpha=0.5)    # gamma=0 → BCE with α
    fl2 = focal_loss(logits, y, gamma=2.0, alpha=0.25)

    print(f"\n  BCE (gamma=0):       {bce:.4f}")
    print(f"  Focal (gamma=2):    {fl2:.4f}")
    print("  Focal loss reduces the dominance of easy negatives.\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test_focal_loss() -> None:
    rng = np.random.default_rng(0)
    N = 20
    logits = rng.standard_normal(N)
    y = rng.integers(0, 2, size=N)

    # gamma=0 should approximate BCE (with alpha weighting)
    fl_gamma0 = focal_loss(logits, y, gamma=0.0, alpha=0.5)
    assert fl_gamma0 > 0, "Loss must be positive"

    # Reduction modes
    loss_mean = focal_loss(logits, y, reduction="mean")
    loss_sum = focal_loss(logits, y, reduction="sum")
    per_sample = focal_loss(logits, y, reduction="none")
    assert isinstance(per_sample, np.ndarray)
    assert np.isclose(loss_mean, per_sample.mean())
    assert np.isclose(loss_sum, per_sample.sum())

    print("  focal_loss: basic tests passed")


def _test_gradient() -> None:
    rng = np.random.default_rng(99)
    N = 10
    logits = rng.standard_normal(N)
    y = rng.integers(0, 2, size=N)
    verify_gradient(logits, y, gamma=2.0, alpha=0.25)
    verify_gradient(logits, y, gamma=0.5, alpha=0.75)


if __name__ == "__main__":
    print("Focal Loss")
    _test_focal_loss()
    _test_gradient()
    _demo_imbalanced_comparison()
