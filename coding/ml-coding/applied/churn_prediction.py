"""
Churn Prediction — Applied ML Pipeline
=======================================

Shopify / Applied ML interview format:
    Given merchant activity data, predict churn within 30 days.
    Narrate: time-aware split → feature leakage detection → calibration.

Problem:
    Binary classification: will a merchant churn in the next 30 days?
    Key challenge: time-aware train/test split to prevent data leakage.

Key decisions to narrate in interview:
    1. Target: churned = no activity for 30+ days after observation window
    2. CRITICAL: Time-based train/test split — random split leaks future data
    3. Feature leakage traps:
       - Post-observation features (things that happen AFTER observation cutoff)
       - Target leakage (e.g., "support_ticket_count" if tickets cause churn)
       - Future-derived aggregations rolled backward
    4. Calibration: GBT probabilities ≠ true probabilities → Platt/isotonic scaling
    5. Evaluation: AUPRC + calibration curve (NOT accuracy)
    6. Business metric: at 5% churn rate, cost of intervention vs. cost of missed churn

Time-based split illustrated:
    |——— train window (Jan–Sep) ———| |— valid (Oct) —| |— test (Nov–Dec) —|
    observation_date in train → features from past 90 days → label: churned in next 30 days

Complexity:
    Feature engineering: O(N * W) where W = lookback window length
    Training:            O(N * D * leaves * estimators)
    Calibration:         O(N) for Platt scaling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
    )
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. Data generation — merchant cohort
# ---------------------------------------------------------------------------

def generate_merchant_dataset(n_merchants: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Simulate merchant activity history with churn labels.

    Each row = merchant observation at a point in time.
    Features = activity in prior 90 days. Label = churned in next 30 days.

    Args:
        n_merchants: Number of merchant snapshots.
        seed:        Random seed.

    Returns:
        DataFrame with merchant features, observation_date, and churned label.
    """
    rng = np.random.default_rng(seed)

    # Assign observation dates (2023-01 through 2023-12) for time-based split
    dates = pd.date_range("2023-01-01", "2023-12-01", freq="MS")
    obs_dates = rng.choice(dates, size=n_merchants)
    obs_dates = pd.to_datetime(obs_dates)

    # Higher churn risk if: low orders, high support tickets, declining revenue
    low_activity = rng.binomial(1, 0.3, n_merchants)  # 30% low-activity merchants

    df = pd.DataFrame({
        "merchant_id": [f"M{i:05d}" for i in range(n_merchants)],
        "observation_date": obs_dates,
        "orders_90d": rng.poisson(lam=np.where(low_activity, 5, 50), size=n_merchants),
        "revenue_90d": rng.lognormal(
            mean=np.where(low_activity, 7, 10), sigma=1.0, size=n_merchants
        ),
        "support_tickets_90d": rng.poisson(
            lam=np.where(low_activity, 3, 1), size=n_merchants
        ),
        "active_products": rng.integers(1, 20, size=n_merchants),
        "days_since_last_order": rng.integers(
            1, 90, size=n_merchants
        ) * np.where(low_activity, 2, 1),
        "account_age_days": rng.integers(30, 1825, size=n_merchants),
        "subscription_tier": rng.choice(["basic", "standard", "advanced"], size=n_merchants),
        "revenue_trend": rng.normal(
            loc=np.where(low_activity, -0.2, 0.1), scale=0.15, size=n_merchants
        ),
    })

    # Churn label: logistic function of risk factors
    churn_logit = (
        -2.5
        + 1.5 * low_activity
        - 0.02 * np.log1p(df["orders_90d"])
        + 0.3 * np.log1p(df["support_tickets_90d"])
        - 0.5 * df["revenue_trend"]
        + 0.01 * df["days_since_last_order"]
        + rng.normal(0, 0.3, n_merchants)
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    df["churned"] = rng.binomial(1, churn_prob, size=n_merchants)

    # Leakage trap: column known only AFTER the observation cutoff (strong correlation)
    df["post_obs_support_ticket"] = (df["churned"] == 1).astype(int)

    return df


# ---------------------------------------------------------------------------
# 2. Time-based train/test split (prevent leakage)
# ---------------------------------------------------------------------------

def time_based_split(
    df: pd.DataFrame,
    date_col: str = "observation_date",
    train_end: str = "2023-09-01",
    val_end: str = "2023-10-01",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame by observation date.

    IMPORTANT: Never use random split for time-series / user-cohort data.
    Random split leaks: merchant behavior after their "observation date"
    might appear in both train and test.

    Timeline:
        [train: before train_end] [val: train_end—val_end] [test: after val_end]

    Args:
        df:        Merchant dataset.
        date_col:  Column containing observation timestamp.
        train_end: Exclusive upper bound for training set.
        val_end:   Exclusive upper bound for validation set.

    Returns:
        (train_df, val_df, test_df)
    """
    train = df[df[date_col] < train_end].copy()
    val   = df[(df[date_col] >= train_end) & (df[date_col] < val_end)].copy()
    test  = df[df[date_col] >= val_end].copy()
    return train, val, test


# ---------------------------------------------------------------------------
# 3. Feature engineering (leakage-safe)
# ---------------------------------------------------------------------------

SAFE_FEATURES = [
    "log_orders_90d",
    "log_revenue_90d",
    "log_support_tickets_90d",
    "active_products",
    "days_since_last_order",
    "log_account_age",
    "revenue_trend",
    "is_basic_tier",
    "order_velocity",            # orders / account_age (bounded)
]

TARGET_COL = "churned"

# LEAKAGE COLUMN — must be dropped before training
LEAKAGE_COLS = ["post_obs_support_ticket"]


def check_for_leakage(df: pd.DataFrame, target: str, threshold: float = 0.5) -> List[str]:
    """Identify columns with suspiciously high correlation to target.

    Heuristic: any feature with |correlation| > threshold warrants manual review.
    Common leakage sources: post-observation events, derived-from-target columns.

    Args:
        df:        DataFrame with features and target.
        target:    Target column name.
        threshold: Correlation threshold for flagging.

    Returns:
        List of column names flagged as potential leakage.

    Complexity:
        Time:  O(N * D)
        Space: O(D)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(target, errors="ignore")
    flagged = []
    for col in numeric_cols:
        corr = abs(df[col].corr(df[target]))
        if corr > threshold:
            flagged.append(col)
            print(f"  ⚠️  Leakage suspect: {col} | |correlation| = {corr:.3f}")
    return flagged


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build leakage-safe features for churn model.

    Args:
        df: Merchant observation DataFrame (MUST NOT include post-observation cols).

    Returns:
        DataFrame with engineered features.
    """
    df = df.copy()

    # Drop known leakage columns FIRST
    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])

    df["log_orders_90d"] = np.log1p(df["orders_90d"])
    df["log_revenue_90d"] = np.log1p(df["revenue_90d"])
    df["log_support_tickets_90d"] = np.log1p(df["support_tickets_90d"])
    df["log_account_age"] = np.log1p(df["account_age_days"])
    df["is_basic_tier"] = (df["subscription_tier"] == "basic").astype(int)

    # Order velocity: normalize by account age to compare new vs. old merchants
    df["order_velocity"] = df["orders_90d"] / (df["account_age_days"].clip(lower=1))

    return df


# ---------------------------------------------------------------------------
# 4. Calibration check
# ---------------------------------------------------------------------------

def evaluate_calibration(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE = Σ_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Where B_b = samples in bin b, acc = true fraction, conf = mean predicted prob.

    Args:
        y_true:   Binary labels.
        y_scores: Predicted probabilities.
        n_bins:   Number of calibration bins.

    Returns:
        ECE value (lower = better calibrated).

    Complexity:
        Time:  O(N)
        Space: O(bins)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(y_true)

    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_scores >= low) & (y_scores < high)
        if mask.sum() == 0:
            continue
        bin_true_rate = y_true[mask].mean()
        bin_conf = y_scores[mask].mean()
        ece += (mask.sum() / N) * abs(bin_true_rate - bin_conf)

    return float(ece)


# ---------------------------------------------------------------------------
# 5. Full pipeline
# ---------------------------------------------------------------------------

def run_churn_pipeline() -> Dict[str, float]:
    """End-to-end churn prediction pipeline.

    Returns:
        Dict of evaluation metrics.
    """
    if not SKLEARN_AVAILABLE:
        print("Skipping pipeline — scikit-learn not installed")
        return {}

    print("=== Churn Prediction Pipeline ===\n")

    df = generate_merchant_dataset()
    df = engineer_features(df)

    print(f"Overall churn rate: {df[TARGET_COL].mean():.2%}")

    # Leakage check on training data
    print("\nLeakage check (should find none after feature engineering):")
    train_df, val_df, test_df = time_based_split(df)
    flagged = check_for_leakage(train_df, TARGET_COL, threshold=0.4)
    if not flagged:
        print("  No leakage suspects found ✓")

    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    X_train = train_df[SAFE_FEATURES].values
    y_train = train_df[TARGET_COL].values
    X_test = test_df[SAFE_FEATURES].values
    y_test = test_df[TARGET_COL].values

    # Train uncalibrated GBT
    model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    raw_scores = model.predict_proba(X_test)[:, 1]

    # Calibrate via Platt scaling (sigmoid method); cv=3 refit on folds
    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train)
    cal_scores = calibrated.predict_proba(X_test)[:, 1]

    auprc_raw = average_precision_score(y_test, raw_scores)
    auprc_cal = average_precision_score(y_test, cal_scores)
    auroc     = roc_auc_score(y_test, cal_scores)
    ece_raw   = evaluate_calibration(y_test, raw_scores)
    ece_cal   = evaluate_calibration(y_test, cal_scores)
    brier     = brier_score_loss(y_test, cal_scores)

    print(f"\nAUPRC (raw):       {auprc_raw:.4f}")
    print(f"AUPRC (calibrated): {auprc_cal:.4f}")
    print(f"AUROC:              {auroc:.4f}")
    print(f"ECE (raw):          {ece_raw:.4f}  ← calibration error before Platt")
    print(f"ECE (calibrated):   {ece_cal:.4f}  ← calibration error after Platt")
    print(f"Brier score:        {brier:.4f}  (lower = better)")

    return {
        "auprc": auprc_cal,
        "auroc": auroc,
        "ece_before": ece_raw,
        "ece_after": ece_cal,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test_time_split() -> None:
    df = generate_merchant_dataset(n_merchants=500)
    train, val, test = time_based_split(df)

    assert len(train) + len(val) + len(test) == len(df)
    assert train["observation_date"].max() < pd.Timestamp("2023-09-01")
    assert test["observation_date"].min() >= pd.Timestamp("2023-10-01")
    print("  Time-based split: no overlap ✓")


def _test_calibration_ece() -> None:
    rng = np.random.default_rng(42)
    # Perfect calibration: predicted probs match true frequencies
    y_scores = rng.uniform(0, 1, 1000)
    y_true = rng.binomial(1, y_scores, 1000)
    ece = evaluate_calibration(y_true, y_scores)
    assert ece < 0.1, f"ECE should be low for calibrated scores, got {ece:.4f}"
    print(f"  ECE test: {ece:.4f} (expected < 0.1) ✓")


def _test_leakage_detection() -> None:
    df = generate_merchant_dataset(n_merchants=1000)  # larger N for stable correlation
    # Before dropping leakage col — should be flagged (lower threshold for binary col)
    flagged = check_for_leakage(df, "churned", threshold=0.2)
    assert any("post_obs" in c for c in flagged), "Leakage column should be detected"

    # After engineering (drops leakage col) — should be clean
    df_clean = engineer_features(df)
    flagged_clean = check_for_leakage(df_clean, "churned", threshold=0.4)
    assert not any("post_obs" in c for c in flagged_clean), "Leakage should be removed"
    print("  Leakage detection: correctly identifies and removes leakage ✓")


if __name__ == "__main__":
    print("Churn Prediction Applied ML")
    _test_time_split()
    _test_calibration_ece()
    _test_leakage_detection()
    metrics = run_churn_pipeline()
    if metrics:
        assert metrics["ece_after"] < metrics["ece_before"], "Calibration should improve ECE"
        print(f"\nFinal check: calibration improved ECE {metrics['ece_before']:.4f} → {metrics['ece_after']:.4f} ✓")
