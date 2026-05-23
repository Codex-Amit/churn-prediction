"""
features/selection/feature_selector.py
Selects the most relevant features using importance scores.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config.settings import RANDOM_SEED, TARGET_COL, TOP_FEATURES

logger = logging.getLogger(__name__)


def select_top_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    top_n: int = TOP_FEATURES,
) -> tuple[list[str], np.ndarray]:
    """
    Fit a quick Random Forest to rank features.
    Returns (top_feature_names, importance_scores).
    """
    rf = RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    top_names   = [feature_names[i] for i in indices]
    top_scores  = importances[indices]

    logger.info("Top %d features selected out of %d", top_n, len(feature_names))
    for name, score in zip(top_names, top_scores):
        logger.debug("  %-35s %.4f", name, score)

    return top_names, top_scores


def get_importance_df(
    model: Any,
    feature_names: list[str],
    top_n: int = TOP_FEATURES,
) -> pd.DataFrame:
    """Extract feature importances from any fitted sklearn estimator."""
    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_
    elif hasattr(model, "coef_"):
        scores = np.abs(model.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    df = pd.DataFrame({"feature": feature_names, "importance": scores})
    return df.nlargest(top_n, "importance").reset_index(drop=True)
