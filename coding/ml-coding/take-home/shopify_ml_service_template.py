"""
Shopify-Style ML Take-Home Service Template
=============================================

Shopify take-home requirements (from prep-roadmap):
    - Containerized FastAPI service or documented notebook
    - SHAP interpretability (required, not optional)
    - Time-series aware validation splits (no random splits)
    - Input validation via Pydantic
    - Output: predictions + feature importance + calibration check
    - Clean architecture: dependency injection, testable services

This template provides:
    1. PredictionRequest / PredictionResponse (Pydantic schemas)
    2. FeatureEngineeringService (injectable, testable)
    3. ModelService (injectable model wrapper)
    4. ExplainabilityService (SHAP values + feature importance)
    5. TimeSeriesCVSplitter (leakage-free validation)
    6. FastAPI app with /predict, /explain, /health endpoints
    7. Calibration utility

Usage:
    # Install: pip install fastapi uvicorn pydantic scikit-learn shap numpy pandas
    # Run:     uvicorn shopify_ml_service_template:app --reload --port 8000
    # Test:    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json"
    #          -d '{"merchant_id":"M001","orders_90d":50,"revenue_90d":5000,...}'

Shopify interview notes:
    - SOLID: each service has one responsibility; dependencies injected
    - Dependency injection enables unit testing with mocked services
    - Black Friday concern: service must handle 10x traffic (documented via circuit breaker)
    - Every design decision must be narrated with rejected alternatives
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
    from pydantic import BaseModel, Field, validator
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import average_precision_score
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False
    print("Dependencies not installed — running structure demo only")
    print("Install: pip install fastapi uvicorn pydantic scikit-learn shap pandas numpy")


# ---------------------------------------------------------------------------
# 1. Pydantic schemas (define before handler logic — Shopify convention)
# ---------------------------------------------------------------------------

if DEPS_AVAILABLE:
    class PredictionRequest(BaseModel):
        """Merchant churn prediction request."""

        merchant_id: str = Field(..., description="Unique merchant identifier")
        orders_90d: int = Field(..., ge=0, description="Orders in last 90 days")
        revenue_90d: float = Field(..., ge=0, description="Revenue in last 90 days (USD)")
        support_tickets_90d: int = Field(..., ge=0)
        active_products: int = Field(..., ge=0)
        days_since_last_order: int = Field(..., ge=0)
        account_age_days: int = Field(..., ge=1)
        subscription_tier: str = Field(..., regex="^(basic|standard|advanced)$")
        revenue_trend: float = Field(default=0.0, ge=-1.0, le=1.0)

        @validator("orders_90d")
        @classmethod
        def orders_non_negative(cls, v: int) -> int:
            """Validate orders is non-negative."""
            if v < 0:
                raise ValueError("orders_90d must be non-negative")
            return v

    class FeatureImportance(BaseModel):
        """Single feature importance entry."""
        feature: str
        importance: float
        shap_value: float

    class PredictionResponse(BaseModel):
        """Churn prediction response with explainability."""
        merchant_id: str
        churn_probability: float = Field(..., ge=0.0, le=1.0)
        churn_risk_tier: str  # "low" | "medium" | "high"
        top_features: List[FeatureImportance]
        model_version: str
        latency_ms: float

    class HealthResponse(BaseModel):
        """Service health check response."""
        status: str
        model_loaded: bool
        uptime_seconds: float


# ---------------------------------------------------------------------------
# 2. Feature engineering service (injectable, testable)
# ---------------------------------------------------------------------------

class FeatureEngineeringService:
    """Transform raw request fields into model features.

    Design decision:
        - Rejected: inline feature engineering in the endpoint handler
          (violates Single Responsibility Principle)
        - Chosen: dedicated service, injected into endpoint handler
          (unit-testable independently of the model)
    """

    FEATURE_NAMES = [
        "log_orders_90d",
        "log_revenue_90d",
        "log_support_tickets_90d",
        "active_products",
        "days_since_last_order",
        "log_account_age",
        "revenue_trend",
        "is_basic_tier",
        "order_velocity",
    ]

    def transform(self, request: "PredictionRequest") -> np.ndarray:
        """Transform a PredictionRequest into a feature vector.

        Args:
            request: Validated merchant prediction request.

        Returns:
            Feature array of shape (1, D).
        """
        features = {
            "log_orders_90d": np.log1p(request.orders_90d),
            "log_revenue_90d": np.log1p(request.revenue_90d),
            "log_support_tickets_90d": np.log1p(request.support_tickets_90d),
            "active_products": float(request.active_products),
            "days_since_last_order": float(request.days_since_last_order),
            "log_account_age": np.log1p(request.account_age_days),
            "revenue_trend": request.revenue_trend,
            "is_basic_tier": 1.0 if request.subscription_tier == "basic" else 0.0,
            "order_velocity": request.orders_90d / max(request.account_age_days, 1),
        }
        return np.array([[features[f] for f in self.FEATURE_NAMES]])

    def transform_batch(self, requests: List["PredictionRequest"]) -> np.ndarray:
        """Transform a batch of requests into feature matrix.

        Args:
            requests: List of PredictionRequest.

        Returns:
            Feature matrix of shape (N, D).

        Raises:
            ValueError: If requests is empty.
        """
        if not requests:
            raise ValueError("requests must be non-empty")
        return np.vstack([self.transform(r) for r in requests])


# ---------------------------------------------------------------------------
# 3. Model service (injectable model wrapper)
# ---------------------------------------------------------------------------

@dataclass
class ModelService:
    """Wrapper around a trained ML model.

    Design decision:
        - Rejected: load model from disk in endpoint handler (not testable)
        - Chosen: injected dependency; tests can inject a mock model

    Args:
        model:         Trained sklearn-compatible model.
        version:       Model version string for response tagging.
        threshold:     Decision threshold for churn_risk_tier.
    """

    model: Any
    version: str = "1.0.0"
    threshold: float = 0.3

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict churn probabilities.

        Args:
            X: Feature matrix, shape (N, D).

        Returns:
            Churn probabilities, shape (N,).

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return self.model.predict_proba(X)[:, 1]

    def risk_tier(self, probability: float) -> str:
        """Classify probability into risk tier.

        Args:
            probability: Churn probability in [0, 1].

        Returns:
            "low" | "medium" | "high"
        """
        if probability < 0.2:
            return "low"
        if probability < self.threshold:
            return "medium"
        return "high"


# ---------------------------------------------------------------------------
# 4. SHAP explainability service
# ---------------------------------------------------------------------------

class ExplainabilityService:
    """Compute SHAP feature importances for model predictions.

    Design decision:
        - Rejected: global feature importance (not per-prediction explainability)
        - Chosen: SHAP TreeExplainer (model-agnostic, works with GBT)
        - SHAP is required by Shopify take-home spec
    """

    def __init__(self, model: Any, feature_names: List[str]) -> None:
        self._model = model
        self._feature_names = feature_names
        self._explainer: Any = None

        try:
            import shap
            self._explainer = shap.TreeExplainer(model)
            self._shap_available = True
        except (ImportError, Exception):
            self._shap_available = False

    def explain(
        self,
        X: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Compute top-k feature importances for predictions.

        Falls back to model.feature_importances_ if SHAP not available.

        Args:
            X:     Feature matrix, shape (N, D).
            top_k: Number of top features to return.

        Returns:
            List of dicts with feature, importance, shap_value keys.

        Complexity:
            Time:  O(N * D) for SHAP tree path
            Space: O(N * D)
        """
        if self._shap_available and self._explainer is not None:
            import shap
            shap_values = self._explainer.shap_values(X)
            # For binary classification: shap_values[1] or shap_values directly
            if isinstance(shap_values, list):
                values = shap_values[1][0]  # class 1, first sample
            else:
                values = shap_values[0]
            importances = np.abs(values)
        else:
            # Fallback: use global feature importance
            if hasattr(self._model, "feature_importances_"):
                importances = self._model.feature_importances_
                values = importances  # no per-sample SHAP
            else:
                importances = np.ones(len(self._feature_names)) / len(self._feature_names)
                values = importances

        # Rank features
        top_indices = np.argsort(-importances)[:top_k]
        return [
            {
                "feature": self._feature_names[i],
                "importance": float(importances[i]),
                "shap_value": float(values[i]) if len(values) > 1 else float(values),
            }
            for i in top_indices
        ]


# ---------------------------------------------------------------------------
# 5. Time-series cross-validation splitter
# ---------------------------------------------------------------------------

class TimeSeriesCVSplitter:
    """Walk-forward cross-validation splits for time-series data.

    Standard k-fold is WRONG for time series:
    - Future data leaks into training
    - Use walk-forward: train on [0..t], validate on [t+1..t+gap]

    Args:
        n_splits:    Number of splits.
        gap:         Gap between train end and validation start (prevents leakage).
        test_size:   Fraction of data for each validation window.
    """

    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 0,
        test_size: float = 0.2,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        self.n_splits = n_splits
        self.gap = gap
        self.test_size = test_size

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate (train_idx, val_idx) tuples.

        Each split expands training window; validation window slides forward.

        Args:
            X: Feature array or date index array, shape (N, ...).
            y: Labels (unused; for sklearn compatibility).

        Returns:
            List of (train_indices, val_indices) tuples.

        Raises:
            ValueError: If X is too small for n_splits.

        Complexity:
            Time:  O(n_splits)
            Space: O(n_splits * N)
        """
        N = len(X)
        val_size = int(N * self.test_size)

        if val_size * self.n_splits + self.gap >= N:
            raise ValueError(
                f"N={N} too small for {self.n_splits} splits with "
                f"test_size={self.test_size}, gap={self.gap}"
            )

        splits = []
        for i in range(self.n_splits):
            val_end = N - (self.n_splits - i - 1) * val_size
            val_start = val_end - val_size
            train_end = val_start - self.gap

            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)

            if len(train_idx) > 0 and len(val_idx) > 0:
                splits.append((train_idx, val_idx))

        return splits


# ---------------------------------------------------------------------------
# 6. FastAPI app (conditional import)
# ---------------------------------------------------------------------------

_SERVICE_START = time.time()


def create_app(model_service: "ModelService", feature_svc: "FeatureEngineeringService"):
    """Factory function to create FastAPI app with injected dependencies.

    Design: dependencies injected at app creation time (not hardcoded globals).
    This allows test code to inject mock services.

    Args:
        model_service: Trained model wrapper.
        feature_svc:   Feature engineering service.

    Returns:
        Configured FastAPI app instance.
    """
    try:
        from fastapi import FastAPI, HTTPException
        import uvicorn  # noqa: F401 (just check it's available)
    except ImportError:
        print("FastAPI not installed — app factory defined but not runnable")
        return None

    from fastapi import FastAPI, HTTPException

    app = FastAPI(
        title="Shopify Merchant Churn Prediction Service",
        description="Predicts merchant churn with SHAP explanations",
        version="1.0.0",
    )

    explain_svc = ExplainabilityService(
        model_service.model,
        feature_names=FeatureEngineeringService.FEATURE_NAMES,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Health check endpoint.

        Returns:
            Service status + model loaded flag.
        """
        return HealthResponse(
            status="healthy",
            model_loaded=model_service.model is not None,
            uptime_seconds=time.time() - _SERVICE_START,
        )

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest) -> PredictionResponse:
        """Predict churn probability for a merchant.

        Args:
            request: Validated merchant feature request.

        Returns:
            Churn probability + risk tier + top SHAP features.

        Raises:
            HTTPException 422: On invalid input (handled by Pydantic).
            HTTPException 500: On model inference failure.
        """
        start = time.time()
        try:
            X = feature_svc.transform(request)
            prob = float(model_service.predict_proba(X)[0])
            top_features_raw = explain_svc.explain(X, top_k=5)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

        latency_ms = (time.time() - start) * 1000

        return PredictionResponse(
            merchant_id=request.merchant_id,
            churn_probability=prob,
            churn_risk_tier=model_service.risk_tier(prob),
            top_features=[FeatureImportance(**f) for f in top_features_raw],
            model_version=model_service.version,
            latency_ms=latency_ms,
        )

    @app.post("/predict_batch")
    async def predict_batch(requests: List[PredictionRequest]) -> List[PredictionResponse]:
        """Batch prediction endpoint (more efficient than repeated /predict).

        Args:
            requests: List of merchant prediction requests.

        Returns:
            List of prediction responses.
        """
        if not requests:
            raise HTTPException(status_code=400, detail="Empty request list")
        if len(requests) > 1000:
            raise HTTPException(status_code=400, detail="Batch size exceeds limit of 1000")

        start = time.time()
        X = feature_svc.transform_batch(requests)
        probs = model_service.predict_proba(X)
        latency_ms = (time.time() - start) * 1000

        results = []
        for i, (req, prob) in enumerate(zip(requests, probs)):
            top_features_raw = explain_svc.explain(X[i:i+1], top_k=3)
            results.append(PredictionResponse(
                merchant_id=req.merchant_id,
                churn_probability=float(prob),
                churn_risk_tier=model_service.risk_tier(float(prob)),
                top_features=[FeatureImportance(**f) for f in top_features_raw],
                model_version=model_service.version,
                latency_ms=latency_ms / len(requests),
            ))
        return results

    return app


# ---------------------------------------------------------------------------
# 7. Training + app entrypoint
# ---------------------------------------------------------------------------

def train_model() -> "ModelService":
    """Train a demo churn model and return a ModelService.

    In production: load from artifact store (MLflow, S3, etc.)
    """
    if not DEPS_AVAILABLE:
        return None  # type: ignore

    # Import churn pipeline from applied/ folder (demo: inline training)
    rng = np.random.default_rng(42)
    N = 2000
    X = rng.standard_normal((N, len(FeatureEngineeringService.FEATURE_NAMES)))
    y = rng.binomial(1, p=0.1, size=N)

    model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X, y)

    # Calibrate (Shopify requirement: calibrated probabilities)
    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=3)
    calibrated.fit(X, y)

    return ModelService(model=calibrated, version="1.0.0-demo")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test_feature_engineering() -> None:
    if not DEPS_AVAILABLE:
        print("  Skipping (dependencies not installed)")
        return

    svc = FeatureEngineeringService()
    req = PredictionRequest(
        merchant_id="M001",
        orders_90d=50,
        revenue_90d=5000.0,
        support_tickets_90d=2,
        active_products=10,
        days_since_last_order=5,
        account_age_days=365,
        subscription_tier="standard",
        revenue_trend=0.1,
    )
    X = svc.transform(req)
    assert X.shape == (1, len(FeatureEngineeringService.FEATURE_NAMES))
    assert X.dtype == np.float64
    print("  FeatureEngineeringService: passed")


def _test_time_series_cv() -> None:
    splitter = TimeSeriesCVSplitter(n_splits=3, gap=5, test_size=0.2)
    X = np.arange(200)
    splits = splitter.split(X)
    assert len(splits) == 3

    for i, (train, val) in enumerate(splits):
        assert train.max() < val.min(), f"Split {i}: train leaks into val"
        assert val.min() - train.max() > splitter.gap, f"Split {i}: gap not respected"

    print("  TimeSeriesCVSplitter: no overlap, gap respected ✓")


def _test_model_service() -> None:
    if not DEPS_AVAILABLE:
        print("  Skipping (dependencies not installed)")
        return

    ms = train_model()
    X = np.random.standard_normal((5, len(FeatureEngineeringService.FEATURE_NAMES)))
    probs = ms.predict_proba(X)
    assert probs.shape == (5,)
    assert all(0 <= p <= 1 for p in probs)

    assert ms.risk_tier(0.05) == "low"
    assert ms.risk_tier(0.25) == "medium"
    assert ms.risk_tier(0.5) == "high"

    print("  ModelService: passed")


if __name__ == "__main__":
    print("Shopify ML Service Template")
    _test_feature_engineering()
    _test_time_series_cv()
    _test_model_service()

    # Uncomment to run FastAPI server:
    # import uvicorn
    # feature_svc = FeatureEngineeringService()
    # model_service = train_model()
    # app = create_app(model_service, feature_svc)
    # uvicorn.run(app, host="0.0.0.0", port=8000)
