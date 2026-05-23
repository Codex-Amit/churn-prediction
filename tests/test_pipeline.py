"""
tests/test_pipeline.py
Run with: python -m pytest tests/ -v
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.cleaner.data_cleaner import clean
from components.encoder.data_encoder import encode_and_split
from components.evaluator.model_evaluator import ChurnMetrics, evaluate, pick_best
from components.trainer.model_trainer import get_model_catalogue
from features.engineering.feature_builder import build_features


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    return pd.DataFrame({
        "customer_id":       [f"C{i}" for i in range(n)],
        "gender":            np.random.choice(["Male", "Female"], n),
        "senior_citizen":    np.random.randint(0, 2, n),
        "partner":           np.random.choice(["Yes", "No"], n),
        "dependents":        np.random.choice(["Yes", "No"], n),
        "tenure":            np.random.randint(1, 73, n),
        "phone_service":     np.random.choice(["Yes", "No"], n),
        "multiple_lines":    np.random.choice(["Yes", "No", "No phone service"], n),
        "internet_service":  np.random.choice(["DSL", "Fiber optic", "No"], n),
        "online_security":   np.random.choice(["Yes", "No", "No internet service"], n),
        "online_backup":     np.random.choice(["Yes", "No", "No internet service"], n),
        "device_protection": np.random.choice(["Yes", "No", "No internet service"], n),
        "tech_support":      np.random.choice(["Yes", "No", "No internet service"], n),
        "streaming_tv":      np.random.choice(["Yes", "No", "No internet service"], n),
        "streaming_movies":  np.random.choice(["Yes", "No", "No internet service"], n),
        "contract":          np.random.choice(["Month-to-month", "One year", "Two year"], n),
        "paperless_billing": np.random.choice(["Yes", "No"], n),
        "payment_method":    np.random.choice(
                                 ["Electronic check", "Mailed check",
                                  "Bank transfer (automatic)", "Credit card (automatic)"], n),
        "monthly_charges":   np.round(np.random.uniform(18, 120, n), 2),
        "total_charges":     np.round(np.random.uniform(100, 8000, n), 2),
        "churn":             np.random.choice(["Yes", "No"], n, p=[0.27, 0.73]),
    })


# ── Cleaner tests ──────────────────────────────────────────────────────────────

class TestCleaner:
    def test_removes_customer_id(self, sample_df):
        assert "customer_id" not in clean(sample_df).columns

    def test_binary_target(self, sample_df):
        assert set(clean(sample_df)["churn"].unique()).issubset({0, 1})

    def test_no_nulls_after_clean(self, sample_df):
        assert not clean(sample_df).isnull().any().any()


# ── Feature engineering tests ─────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_num_services_added(self, sample_df):
        df = build_features(clean(sample_df))
        assert "num_services" in df.columns
        assert df["num_services"].between(0, 8).all()

    def test_contract_flags(self, sample_df):
        df = build_features(clean(sample_df))
        assert "is_month_to_month" in df.columns
        assert "is_long_term" in df.columns

    def test_spend_ratio(self, sample_df):
        df = build_features(clean(sample_df))
        assert "avg_monthly_spend" in df.columns
        assert (df["avg_monthly_spend"] >= 0).all()

    def test_streaming_flag(self, sample_df):
        df = build_features(clean(sample_df))
        assert "has_streaming" in df.columns
        assert df["has_streaming"].isin([0, 1]).all()


# ── Encoder tests ─────────────────────────────────────────────────────────────

class TestEncoder:
    def test_shapes_match(self, sample_df):
        df = build_features(clean(sample_df))
        X_train, X_test, y_train, y_test, _, _ = encode_and_split(df)
        assert X_train.shape[1] == X_test.shape[1]
        assert len(X_train) == len(y_train)

    def test_scaled_range(self, sample_df):
        df = build_features(clean(sample_df))
        X_train, *_ = encode_and_split(df)
        assert X_train.min() >= -0.01
        assert X_train.max() <= 1.01


# ── Evaluator tests ───────────────────────────────────────────────────────────

class TestEvaluator:
    def test_pick_best_by_roc_auc(self):
        m1 = ChurnMetrics("A", accuracy=0.80, f1=0.70, roc_auc=0.82)
        m2 = ChurnMetrics("B", accuracy=0.78, f1=0.72, roc_auc=0.91)
        assert pick_best([m1, m2]).model_name == "B"

    def test_evaluate_returns_metrics(self, sample_df):
        from sklearn.linear_model import LogisticRegression
        df = build_features(clean(sample_df))
        X_train, X_test, y_train, y_test, _, _ = encode_and_split(df)
        model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        m = evaluate("LR", model, X_test, y_test)
        assert 0 <= m.accuracy <= 1
        assert 0 <= m.f1 <= 1


# ── Trainer tests ──────────────────────────────────────────────────────────────

class TestTrainer:
    def test_catalogue_not_empty(self):
        assert len(get_model_catalogue()) > 0

    def test_all_models_have_predict(self):
        for name, model in get_model_catalogue().items():
            assert hasattr(model, "predict"), f"{name} missing predict"
