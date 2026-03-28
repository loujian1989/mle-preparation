"""
Content Feed Ranking — Applied ML Pipeline
===========================================

Reddit / Applied ML interview format:
    Given a content engagement dataset, build a ranking model in 45–60 minutes.
    Narrate: target construction → feature engineering → ranking model → NDCG evaluation.

Problem:
    Learn-to-rank: given (user, post) pairs, predict engagement probability.
    Goal: rank posts by predicted engagement to maximize feed quality.

Key decisions to narrate in interview:
    1. Target variable: pointwise binary (engaged=1/not=0); or listwise — start pointwise
    2. Feature engineering: content signals (upvotes, comments, awards), user affinity, recency
    3. Class imbalance: ~5% engagement rate → class_weight or threshold tuning
    4. Ranking metric: NDCG@K (not accuracy, not AUC) — position-weighted relevance
    5. Position bias: items ranked higher get more clicks — use IPS correction or randomized position
    6. Spam pre-filter: spam score is a hard pre-filter, NOT a feature (Reddit specific)

Ranking evaluation metrics explained:
    - NDCG@K: Normalized Discounted Cumulative Gain at position K
      DCG@K  = Σ_{i=1}^{K} (2^rel_i - 1) / log2(i+1)
      NDCG@K = DCG@K / IDCG@K  (where IDCG = ideal ranking DCG)
    - MRR: Mean Reciprocal Rank — position of first relevant item
    - Precision@K: fraction of top-K items that are relevant

Complexity:
    Feature engineering: O(N)
    LightGBM training:   O(N * D * leaves * estimators)
    NDCG@K:              O(N log N) per query
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. Data generation
# ---------------------------------------------------------------------------

def generate_ranking_dataset(n_posts: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Simulate content feed with engagement labels.

    Each row = (user, post) pair with features and binary engagement label.

    Args:
        n_posts: Number of (user, post) samples.
        seed:    Random seed.

    Returns:
        DataFrame with content features and engaged label.
    """
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "post_age_hours": rng.exponential(scale=24, size=n_posts),
        "upvote_count": rng.negative_binomial(n=5, p=0.5, size=n_posts),
        "comment_count": rng.negative_binomial(n=2, p=0.6, size=n_posts),
        "award_count": rng.poisson(lam=0.2, size=n_posts),
        "user_subreddit_affinity": rng.beta(a=2, b=5, size=n_posts),  # 0–1
        "post_position": rng.integers(1, 25, size=n_posts),           # position in feed
        "content_type": rng.choice(["text", "image", "video", "link"], size=n_posts),
        "subreddit_size": rng.lognormal(mean=10, sigma=2, size=n_posts),
        "spam_score": rng.beta(a=1, b=20, size=n_posts),              # mostly low
    })

    # Simulate engagement: function of quality signals + position bias
    engagement_logit = (
        0.3 * np.log1p(df["upvote_count"])
        + 0.2 * np.log1p(df["comment_count"])
        + 0.5 * df["user_subreddit_affinity"]
        - 0.05 * df["post_age_hours"]
        + 0.3 * (df["content_type"] == "video").astype(float)
        - 0.1 * df["post_position"]          # position bias: later = less engaged
        + rng.normal(0, 0.5, n_posts)
    )
    probs = 1 / (1 + np.exp(-engagement_logit))
    df["engaged"] = rng.binomial(1, probs, size=n_posts)

    # Hard filter: mark spam posts (these should be removed before ranking)
    df["is_spam"] = (df["spam_score"] > 0.5).astype(int)

    return df


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "log_upvotes",
    "log_comments",
    "award_count",
    "user_subreddit_affinity",
    "post_age_hours",
    "recency_score",
    "is_video",
    "log_subreddit_size",
    # NOTE: post_position intentionally excluded — it's a bias artifact,
    # not a signal we want the model to learn. Handle via IPS correction.
]

TARGET_COL = "engaged"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived ranking features.

    Position bias note:
        post_position is excluded from features — we don't want the model
        to learn "items in position 1 get clicks" (that's a feedback loop).
        In production, use Inverse Propensity Scoring (IPS) to debias:
            IPS weight = 1 / P(click | position)

    Args:
        df: Raw DataFrame (pre-filtered — spam already removed).

    Returns:
        DataFrame with added feature columns.
    """
    df = df.copy()

    # STEP 1: Remove spam before any feature engineering (hard pre-filter)
    df = df[df["is_spam"] == 0].reset_index(drop=True)
    print(f"  After spam filter: {len(df)} posts remain")

    # Log transforms for right-skewed engagement counts
    df["log_upvotes"] = np.log1p(df["upvote_count"])
    df["log_comments"] = np.log1p(df["comment_count"])
    df["log_subreddit_size"] = np.log1p(df["subreddit_size"])

    # Recency decay: exponential decay with half-life = 24 hours
    HALF_LIFE_HOURS = 24.0
    df["recency_score"] = np.exp(-df["post_age_hours"] * np.log(2) / HALF_LIFE_HOURS)

    # Content type one-hot (video is the strongest signal)
    df["is_video"] = (df["content_type"] == "video").astype(int)

    return df


# ---------------------------------------------------------------------------
# 3. NDCG implementation
# ---------------------------------------------------------------------------

def ndcg_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int = 10) -> float:
    """Compute NDCG@K for a single query.

    NDCG@K = DCG@K / IDCG@K
    DCG@K  = Σ_{i=1}^{K} (2^rel_i - 1) / log2(i+2)  [i is 0-indexed]

    Args:
        y_true:   Binary relevance labels (0 or 1).
        y_scores: Model scores (higher = more relevant).
        k:        Number of top positions to evaluate.

    Returns:
        NDCG@K score in [0, 1].

    Complexity:
        Time:  O(N log N)
        Space: O(N)
    """
    k = min(k, len(y_true))
    sorted_indices = np.argsort(-y_scores)[:k]
    ranked_labels = y_true[sorted_indices]

    # DCG
    discounts = np.log2(np.arange(2, k + 2))          # log2(2), log2(3), ...
    dcg = np.sum((2 ** ranked_labels - 1) / discounts)

    # IDCG: ideal ordering (sort true labels descending)
    ideal_labels = np.sort(y_true)[::-1][:k]
    idcg = np.sum((2 ** ideal_labels - 1) / discounts[:len(ideal_labels)])

    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


def mean_ndcg_at_k(
    y_true_groups: List[np.ndarray],
    y_score_groups: List[np.ndarray],
    k: int = 10,
) -> float:
    """Average NDCG@K over multiple queries/users.

    Args:
        y_true_groups:  List of relevance arrays, one per query.
        y_score_groups: List of score arrays, one per query.
        k:              Cutoff rank.

    Returns:
        Mean NDCG@K.
    """
    scores = [
        ndcg_at_k(y_t, y_s, k)
        for y_t, y_s in zip(y_true_groups, y_score_groups)
    ]
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# 4. Full pipeline
# ---------------------------------------------------------------------------

def run_ranking_pipeline() -> Dict[str, float]:
    """End-to-end content ranking pipeline.

    Returns:
        Dict of evaluation metrics.
    """
    if not SKLEARN_AVAILABLE:
        print("Skipping pipeline — scikit-learn not installed")
        return {}

    print("=== Content Ranking Pipeline ===\n")

    # Steps 1–2: Generate data + engineer features
    df = generate_ranking_dataset()
    df = engineer_features(df)
    print(f"Engagement rate: {df[TARGET_COL].mean():.2%}")

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Step 3: Pointwise ranking model (GBT classifier predicts engagement probability)
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_scores = model.predict_proba(X_test)[:, 1]

    # Step 4: Ranking metrics
    auprc = average_precision_score(y_test, y_scores)
    auroc = roc_auc_score(y_test, y_scores)

    # Simulate per-user NDCG (group test set into 50-post feeds)
    chunk_size = 50
    groups_true = [y_test[i:i+chunk_size] for i in range(0, len(y_test) - chunk_size + 1, chunk_size)]
    groups_scores = [y_scores[i:i+chunk_size] for i in range(0, len(y_scores) - chunk_size + 1, chunk_size)]
    ndcg_10 = mean_ndcg_at_k(groups_true, groups_scores, k=10)

    print(f"\nAUPRC:    {auprc:.4f}")
    print(f"AUROC:    {auroc:.4f}")
    print(f"NDCG@10: {ndcg_10:.4f}")

    # Feature importance (narrate top signals in interview)
    feature_importance = sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    print("\nTop features by importance:")
    for feat, imp in feature_importance[:5]:
        print(f"  {feat:<35} {imp:.4f}")

    return {"auprc": auprc, "auroc": auroc, "ndcg_10": ndcg_10}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test_ndcg() -> None:
    # Perfect ranking
    y_true = np.array([1, 1, 0, 0, 0])
    y_scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    assert ndcg_at_k(y_true, y_scores, k=5) == 1.0, "Perfect ranking should give NDCG=1.0"

    # Reversed ranking (worst case)
    y_scores_rev = np.array([0.1, 0.2, 0.9, 0.8, 0.7])
    assert ndcg_at_k(y_true, y_scores_rev, k=5) < 1.0, "Reversed ranking should give NDCG<1.0"

    # All irrelevant
    y_all_zero = np.zeros(5)
    assert ndcg_at_k(y_all_zero, y_scores, k=5) == 0.0, "All-zero relevance → NDCG=0"

    print("  NDCG@K: all tests passed")


def _test_feature_engineering() -> None:
    df = generate_ranking_dataset(n_posts=200)
    df_fe = engineer_features(df)
    assert "log_upvotes" in df_fe.columns
    assert "recency_score" in df_fe.columns
    assert (df_fe["recency_score"] >= 0).all() and (df_fe["recency_score"] <= 1).all()
    assert "is_spam" not in df_fe.columns or df_fe["is_spam"].sum() == 0, "Spam should be filtered"
    print("  Feature engineering: passed")


if __name__ == "__main__":
    print("Content Ranking Applied ML")
    _test_ndcg()
    _test_feature_engineering()
    metrics = run_ranking_pipeline()
    if metrics:
        assert metrics["ndcg_10"] > 0.5, f"Expected NDCG@10 > 0.5, got {metrics['ndcg_10']:.4f}"
        print(f"\nFinal check: NDCG@10 = {metrics['ndcg_10']:.4f} > 0.5 ✓")
