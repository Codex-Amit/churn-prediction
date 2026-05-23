"""
components/visualizer/chart_builder.py
All matplotlib/seaborn figures for EDA and model evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

PALETTE = sns.color_palette("muted")


def _save(fig: plt.Figure, path: str | Path | None) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.close(fig)


# ── EDA charts ────────────────────────────────────────────────────────────────

def churn_distribution(df: pd.DataFrame, save_path=None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Churn Distribution", fontsize=14, fontweight="bold")
    counts = df["churn"].value_counts()
    labels = ["No Churn", "Churn"]
    axes[0].bar(labels, counts.values, color=[PALETTE[0], PALETTE[3]], alpha=0.85)
    axes[0].set_ylabel("Customers")
    axes[0].set_title("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 20, str(v), ha="center", fontweight="bold")
    axes[1].pie(counts.values, labels=labels, autopct="%1.1f%%",
                colors=[PALETTE[0], PALETTE[3]], startangle=90)
    axes[1].set_title("Proportion")
    fig.tight_layout()
    _save(fig, save_path)


def numerical_by_churn(df: pd.DataFrame, save_path=None) -> None:
    cols = ["tenure", "monthly_charges", "total_charges"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Numerical Features by Churn", fontsize=14, fontweight="bold")
    for ax, col in zip(axes, cols):
        for val, label, color in [(0, "No Churn", PALETTE[0]), (1, "Churn", PALETTE[3])]:
            ax.hist(df[df["churn"] == val][col], bins=30, alpha=0.6,
                    label=label, color=color, density=True)
        ax.set_title(col.replace("_", " ").title())
        ax.legend()
        ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, save_path)


def categorical_churn_rates(df: pd.DataFrame, save_path=None) -> None:
    cols = ["contract", "internet_service", "payment_method", "gender"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Churn Rate by Category", fontsize=14, fontweight="bold")
    for ax, col in zip(axes.flat, cols):
        rate = df.groupby(col)["churn"].mean().sort_values(ascending=False)
        bars = ax.bar(range(len(rate)), rate.values * 100, color=PALETTE[2], alpha=0.85)
        ax.set_xticks(range(len(rate)))
        ax.set_xticklabels(rate.index, rotation=20, ha="right")
        ax.set_title(col.replace("_", " ").title())
        ax.set_ylabel("Churn Rate (%)")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, save_path)


def correlation_heatmap(df: pd.DataFrame, save_path=None) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.select_dtypes(include="number").corr(),
                annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)


# ── Model charts ──────────────────────────────────────────────────────────────

def roc_curves(fitted_models: dict[str, Any], X_test, y_test, save_path=None) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    for i, (name, model) in enumerate(fitted_models.items()):
        if hasattr(model, "predict_proba"):
            y_prob       = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _  = roc_curve(y_test, y_prob)
            auc = np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr)
            ax.plot(fpr, tpr, lw=1.5, color=PALETTE[i % len(PALETTE)],
                    label=f"{name} (AUC={auc:.3f})")
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, save_path)


def model_comparison(metrics_list: list, save_path=None) -> None:
    names = [m.model_name for m in metrics_list]
    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w, [m.accuracy for m in metrics_list], w, label="Accuracy", color=PALETTE[0], alpha=0.85)
    ax.bar(x,     [m.f1       for m in metrics_list], w, label="F1 Score", color=PALETTE[1], alpha=0.85)
    ax.bar(x + w, [m.roc_auc  for m in metrics_list], w, label="ROC-AUC",  color=PALETTE[2], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, save_path)


def confusion_matrix_chart(conf_matrix, model_name: str, save_path=None) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"], ax=ax)
    ax.set_title(f"{model_name} — Confusion Matrix", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    _save(fig, save_path)


def feature_importance_chart(importance_df: pd.DataFrame, model_name: str, save_path=None) -> None:
    if importance_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(importance_df["feature"][::-1], importance_df["importance"][::-1],
            color=PALETTE[1], alpha=0.85)
    ax.set_title(f"{model_name} — Top Feature Importances", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, save_path)
