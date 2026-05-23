"""
components/trainer/model_trainer.py
Defines all classifiers and handles training + persistence.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config.settings import RANDOM_SEED

logger = logging.getLogger(__name__)


def get_model_catalogue() -> dict[str, Any]:
    return {
        "LogisticRegression": LogisticRegression(
                                  max_iter=1000, random_state=RANDOM_SEED,
                                  class_weight="balanced"),
        "DecisionTree":       DecisionTreeClassifier(
                                  max_depth=6, random_state=RANDOM_SEED,
                                  class_weight="balanced"),
        "KNN":                KNeighborsClassifier(n_neighbors=7),
        "RandomForest":       RandomForestClassifier(
                                  n_estimators=200, max_depth=10,
                                  min_samples_leaf=5, random_state=RANDOM_SEED,
                                  class_weight="balanced", n_jobs=-1),
        "GradientBoosting":   GradientBoostingClassifier(
                                  n_estimators=200, learning_rate=0.05,
                                  max_depth=4, subsample=0.8,
                                  random_state=RANDOM_SEED),
        "SVM":                SVC(kernel="rbf", probability=True,
                                  random_state=RANDOM_SEED,
                                  class_weight="balanced"),
    }


def train_all(
    models:  dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, Any]:
    """Fit every model and return dict of fitted estimators."""
    fitted: dict[str, Any] = {}
    for name, model in models.items():
        logger.info("Training %s …", name)
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted


def save_model(model: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved → %s", path)


def load_model(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)
