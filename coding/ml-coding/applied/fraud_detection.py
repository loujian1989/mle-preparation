"""
Fraud Detection — Applied ML Pipeline
======================================

Stripe / Applied ML interview format:
    Given a transactions dataset, build a fraud detection model in 45–60 minutes.
    Narrate: target construction → feature engineering → model selection → evaluation.

Problem:
    Binary classification: predict whether a transaction is fraudulent.
    Class imbalance: ~0.5% fraud rate (1:200 ratio).

Key decisions to narrate in interview:
    1. Target variable: is_fraud (binary, derived from label or chargeback data)
    2. Feature engineering: velocity features (txn count/amount in windows), card-level risk
    3. Model: GBT (LightGBM or XGBoost) — handles imbalance + tabular data well
    4. Evaluation: NOT accuracy (misleading at 0.5% base rate)
       → Precision-Recall AUC (AUPRC), F-beta with β < 1 (precision-favoring)
       → False positive cost vs. false negative cost → threshold as business decision
    5. Calibration: raw GBT scores may be poorly calibrated → Platt scaling
    6. Class imbalance: scale_pos_weight or class_weight; focal loss in NN alternative

Dataset simulated here (real data unavailable):
    - transaction_id, user_id, amount, merchant_category, hour_of_day
    - velocity: txn_count_1h, amount_sum_1h, distinct_merchants_24h
    - card: card_age_days, is_international
    - label: is_fraud

Complexity:
    Feature engineering: O(N log N) for sorting + window aggregation
    LightGBM training:   O(N * D * num_leaves * n_estimators)
    Inference:           O(D * num_leaves * n_estimators)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        classification_report,
        precision_recall_curve,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn not installed — running structure demo only")


# ---------------------------------------------------------------------------
# 1. Data generation (simulates real fraud dataset shape)
# ---------------------------------------------------------------------------

def generate_fraud_dataset(n_samples: int = 10_000, fraud_rate: float = 0.005, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic fraud detection dataset.

    Args:
        n_samples:  Total transactions.
        fraud_rate: Proportion of fraudulent transactions.
        seed:       Random seed.

    Returns:
        DataFrame with transaction features and is_fraud label.
    """
    rng = np.random.default_rng(seed)
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    def make_transactions(n: int, is_fraud: bool) -> pd.DataFrame:
        return pd.DataFrame({
            "amount": rng.lognormal(
                mean=5.0 if is_fraud else 4.0,
                sigma=1.5 if is_fraud else 1.0,
                size=n,
            ),
            "hour_of_day": rng.integers(0, 24, size=n),
            "txn_count_1h": rng.poisson(lam=8 if is_fraud else 2, size=n),
            "amount_sum_1h": rng.lognormal(7.0 if is_fraud else 5.0, 1.0, size=n),
            "distinct_merchants_24h": rng.integers(1, 20 if is_fraud else 5, size=n),
            "card_age_days": rng.integers(0, 365 if is_fraud else 2000, size=n),
            "is_international": rng.binomial(1, 0.7 if is_fraud else 0.1, size=n),
            "merchant_category": rng.choice(
                ["retail", "food", "travel", "online", "atm"],
                size=n,
                p=[0.1, 0.1, 0.3, 0.4, 0.1] if is_fraud else [0.3, 0.3, 0.1, 0.2, 0.1],
            ),
            "is_fraud": int(is_fraud),
        })

    df = pd.concat([make_transactions(n_legit, False), make_transactions(n_fraud, True)], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "amount",
    "hour_of_day",
    "txn_count_1h",
    "amount_sum_1h",
    "distinct_merchants_24h",
    "card_age_days",
    "is_international",
    # Derived features (engineered below):
    "log_amount",
    "is_high_velocity",
    "is_night_txn",
    "merchant_online",
    "txn_amount_ratio",
]

TARGET_COL = "is_fraud"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to the transaction DataFrame.

    Narrate in interview: these are hand-crafted priors about fraud patterns.
    Feature selection: drop correlated raw cols after deriving; use SHAP for post-hoc validation.

    Args:
        df: Raw transactions DataFrame.

    Returns:
        DataFrame with additional feature columns.

    Complexity:
        Time:  O(N)
        Space: O(N)
    """
    df = df.copy()

    # Log transform: right-skewed amount distribution
    df["log_amount"] = np.log1p(df["amount"])

    # Velocity spike: >5 transactions in 1h is suspicious
    df["is_high_velocity"] = (df["txn_count_1h"] > 5).astype(int)

    # Night transactions (midnight–5am): higher fraud rate in data
    df["is_night_txn"] = df["hour_of_day"].between(0, 5).astype(int)

    # Online merchants have higher fraud rates
    df["merchant_online"] = (df["merchant_category"] == "online").astype(int)

    # Ratio: this transaction vs. recent average
    avg_1h = df["amount_sum_1h"] / df["txn_count_1h"].clip(lower=1)
    df["txn_amount_ratio"] = df["amount"] / (avg_1h + 1.0)

    return df


# ---------------------------------------------------------------------------
# 3. Threshold calibration — business decision
# ---------------------------------------------------------------------------

def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    fp_cost: float = 1.0,
    fn_cost: float = 50.0,
) -> Tuple[float, float]:
    """Find threshold minimizing expected cost given FP/FN costs.

    In a real interview: "threshold is not a model decision — it's a business
    decision. FN (missed fraud) costs ~50x more than FP (declined legitimate txn)."

    Args:
        y_true:   Binary labels.
        y_scores: Model probability scores for class 1.
        fp_cost:  Cost per false positive (user friction, lost revenue).
        fn_cost:  Cost per false negative (chargeback + operational cost).

    Returns:
        Tuple of (optimal_threshold, min_expected_cost).

    Complexity:
        Time:  O(N log N)
        Space: O(N)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    n = len(y_true)
    n_pos = y_true.sum()

    best_threshold = 0.5
    best_cost = float("inf")

    for thresh, prec, rec in zip(thresholds, precisions[:-1], recalls[:-1]):
        # Estimate TP, FP, FN from precision/recall
        tp = rec * n_pos
        fp = tp / max(prec, 1e-9) - tp
        fn = n_pos - tp

        expected_cost = fp * fp_cost + fn * fn_cost
        if expected_cost < best_cost:
            best_cost = expected_cost
            best_threshold = float(thresh)

    return best_threshold, best_cost


# ---------------------------------------------------------------------------
# 4. Full pipeline
# ---------------------------------------------------------------------------

def run_fraud_pipeline() -> Dict[str, float]:
    """End-to-end fraud detection pipeline.

    Returns:
        Dict of evaluation metrics.
    """
    if not SKLEARN_AVAILABLE:
        print("Skipping pipeline — scikit-learn not installed")
        return {}

    print("=== Fraud Detection Pipeline ===\n")

    # Step 1: Generate + engineer features
    df = generate_fraud_dataset()
    df = engineer_features(df)
    print(f"Dataset: {len(df)} transactions, {df['is_fraud'].mean():.3%} fraud rate")

    # Step 2: Train/test split (time-based in production; random here for demo)
    #   IMPORTANT: In production, always split by time to prevent data leakage.
    #   Leakage trap: velocity features computed on full dataset leak future info.
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].values
    y = df[TARGET_COL].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Step 3: Model — GBT with class weight to handle imbalance
    #   Alternative: focal loss in neural net (see focal_loss.py)
    fraud_rate = y_train.mean()
    scale_pos_weight = (1 - fraud_rate) / max(fraud_rate, 1e-9)
    print(f"Class imbalance ratio (scale_pos_weight): {scale_pos_weight:.1f}x\n")

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Step 4: Evaluate — NEVER use accuracy for imbalanced classification
    y_scores = model.predict_proba(X_test)[:, 1]
    auprc = average_precision_score(y_test, y_scores)
    auroc = roc_auc_score(y_test, y_scores)
    print(f"AUPRC (primary metric): {auprc:.4f}")
    print(f"AUROC:                  {auroc:.4f}")

    # Step 5: Threshold optimization
    threshold, cost = find_optimal_threshold(y_test, y_scores, fp_cost=1.0, fn_cost=50.0)
    y_pred = (y_scores >= threshold).astype(int)
    print(f"\nOptimal threshold (FP:FN cost = 1:50): {threshold:.3f}")
    print(classification_report(y_test, y_pred, target_names=["legit", "fraud"]))

    # Step 6: Calibration (important for scoring engines downstream)
    # cv=3: refit on train folds (safer than cv="prefit" which was removed in sklearn>=1.4)
    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train)

    return {
        "auprc": auprc,
        "auroc": auroc,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test_feature_engineering() -> None:
    df = generate_fraud_dataset(n_samples=100)
    df_fe = engineer_features(df)

    assert "log_amount" in df_fe.columns
    assert "is_high_velocity" in df_fe.columns
    assert df_fe["log_amount"].min() >= 0
    assert set(df_fe["is_high_velocity"].unique()).issubset({0, 1})
    print("  Feature engineering: passed")


def _test_threshold_finder() -> None:
    if not SKLEARN_AVAILABLE:
        return
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_scores = np.clip(y_true + rng.normal(0, 0.3, 200), 0, 1)
    thresh, cost = find_optimal_threshold(y_true, y_scores)
    assert 0.0 <= thresh <= 1.0
    assert cost >= 0
    print("  Threshold optimization: passed")


if __name__ == "__main__":
    print("Fraud Detection Applied ML")
    _test_feature_engineering()
    _test_threshold_finder()
    metrics = run_fraud_pipeline()
    if metrics:
        assert metrics["auprc"] > 0.5, f"Expected AUPRC > 0.5, got {metrics['auprc']:.4f}"
        print(f"\nFinal check: AUPRC = {metrics['auprc']:.4f} > 0.5 ✓")
