"""
components/evaluator/model_evaluator.py
Computes classification metrics and picks the best model.
"""

from __future__ import annotations

from sklearn.metrics import classification_report

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class ChurnMetrics:
    model_name:  str
    accuracy:    float
    f1:          float
    roc_auc:     float
    report:      str = field(repr=False, default="")
    conf_matrix: Any = field(repr=False, default=None)

    def __str__(self) -> str:
        return (
            f"{self.model_name:30s} | "
            f"Accuracy={self.accuracy:.4f} | "
            f"F1={self.f1:.4f} | "
            f"ROC-AUC={self.roc_auc:.4f}"
        )


def evaluate(model_name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    roc    = roc_auc_score(y_test, y_prob) if y_prob is not None else float("nan")

    return ChurnMetrics(
        model_name=model_name,
        accuracy=accuracy_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred, zero_division=0),
        roc_auc=roc,
        report=classification_report(y_test, y_pred,
                   target_names=["No Churn", "Churn"], zero_division=0),
        conf_matrix=confusion_matrix(y_test, y_pred),
    )


def evaluate_all(
    fitted_models: dict[str, Any],
    X_test:        np.ndarray,
    y_test:        np.ndarray,
) -> list[ChurnMetrics]:
    results = []
    for name, model in fitted_models.items():
        m = evaluate(name, model, X_test, y_test)
        logger.info("  %s", m)
        results.append(m)
    return results


def pick_best(metrics_list: list[ChurnMetrics]) -> ChurnMetrics:
    return max(metrics_list, key=lambda m: m.roc_auc)


def save_metrics(metrics_list: list[ChurnMetrics], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    records = [
        {"model": m.model_name, "accuracy": round(m.accuracy, 4),
         "f1": round(m.f1, 4), "roc_auc": round(m.roc_auc, 4)}
        for m in metrics_list
    ]
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    logger.info("Metrics saved → %s", path)
