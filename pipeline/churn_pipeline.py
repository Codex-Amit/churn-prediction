"""
pipeline/churn_pipeline.py
Orchestrates the full end-to-end churn prediction workflow.
"""

from __future__ import annotations

import logging

import pandas as pd

from components.cleaner.data_cleaner import clean
from components.encoder.data_encoder import encode_and_split
from components.evaluator.model_evaluator import evaluate_all, pick_best, save_metrics
from components.loader.data_loader import load_or_generate
from components.trainer.model_trainer import get_model_catalogue, save_model, train_all
from components.visualizer.chart_builder import (
    categorical_churn_rates,
    churn_distribution,
    confusion_matrix_chart,
    correlation_heatmap,
    feature_importance_chart,
    model_comparison,
    numerical_by_churn,
    roc_curves,
)
from config.settings import (
    BEST_MODEL_FILE,
    METRICS_FILE,
    PROC_DATA_FILE,
    RAW_DATA_FILE,
    REPORTS_DIR,
)
from features.engineering.feature_builder import build_features
from features.selection.feature_selector import get_importance_df

logger = logging.getLogger(__name__)


def run() -> None:
    """Execute the complete pipeline from data loading to report generation."""

    # 1. Load ──────────────────────────────────────────────────────────────────
    logger.info("=== Step 1: Loading Data ===")
    df_raw = load_or_generate(RAW_DATA_FILE)

    # 2. Clean ─────────────────────────────────────────────────────────────────
    logger.info("=== Step 2: Cleaning ===")
    df_clean = clean(df_raw)

    # 3. Feature engineering ───────────────────────────────────────────────────
    logger.info("=== Step 3: Feature Engineering ===")
    df_features = build_features(df_clean)
    PROC_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(PROC_DATA_FILE, index=False)
    logger.info("Processed data saved → %s", PROC_DATA_FILE)

    # 4. Encode & split ────────────────────────────────────────────────────────
    logger.info("=== Step 4: Encoding & Splitting ===")
    X_train, X_test, y_train, y_test, feature_cols, scaler = encode_and_split(df_features)

    # 5. Train ─────────────────────────────────────────────────────────────────
    logger.info("=== Step 5: Training Models ===")
    catalogue     = get_model_catalogue()
    fitted_models = train_all(catalogue, X_train, y_train)

    # 6. Evaluate ──────────────────────────────────────────────────────────────
    logger.info("=== Step 6: Evaluating Models ===")
    metrics_list = evaluate_all(fitted_models, X_test, y_test)

    print("\n--- Results (sorted by ROC-AUC) ---")
    for m in sorted(metrics_list, key=lambda x: -x.roc_auc):
        print(f"  {m}")

    best = pick_best(metrics_list)
    print(f"\n  ✅ Best model: {best.model_name}  (ROC-AUC = {best.roc_auc:.4f})")
    print(f"\n{best.report}")

    # 7. Save artefacts ────────────────────────────────────────────────────────
    logger.info("=== Step 7: Saving Artefacts ===")
    save_model(fitted_models[best.model_name], BEST_MODEL_FILE)
    save_metrics(metrics_list, METRICS_FILE)

    # 8. Visualise ─────────────────────────────────────────────────────────────
    logger.info("=== Step 8: Generating Figures ===")
    figs = REPORTS_DIR
    churn_distribution(df_features,        save_path=figs / "01_churn_distribution.png")
    numerical_by_churn(df_features,        save_path=figs / "02_numerical_features.png")
    categorical_churn_rates(df_features,   save_path=figs / "03_categorical_churn.png")
    correlation_heatmap(df_features,       save_path=figs / "04_correlation_heatmap.png")
    roc_curves(fitted_models, X_test, y_test, save_path=figs / "05_roc_curves.png")
    model_comparison(metrics_list,         save_path=figs / "06_model_comparison.png")
    confusion_matrix_chart(best.conf_matrix, best.model_name,
                                           save_path=figs / "07_confusion_matrix.png")
    imp_df = get_importance_df(fitted_models[best.model_name], feature_cols)
    feature_importance_chart(imp_df, best.model_name,
                                           save_path=figs / "08_feature_importance.png")

    logger.info("Pipeline complete ✅")
