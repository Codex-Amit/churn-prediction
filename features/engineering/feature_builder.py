"""
features/engineering/feature_builder.py
Creates derived features that improve model performance.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature transformations and return enriched DataFrame."""
    df = df.copy()
    df = _add_spend_ratio(df)
    df = _add_service_count(df)
    df = _add_contract_flags(df)
    df = _add_streaming_flag(df)
    df = _add_tenure_group(df)
    logger.info("Feature engineering done — columns now: %d", df.shape[1])
    return df


def _add_spend_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df["avg_monthly_spend"] = df["total_charges"] / (df["tenure"] + 1)
    return df


def _add_service_count(df: pd.DataFrame) -> pd.DataFrame:
    service_cols = [
        "phone_service", "multiple_lines", "online_security",
        "online_backup", "device_protection", "tech_support",
        "streaming_tv", "streaming_movies",
    ]
    df["num_services"] = sum(
        (df[col] == "Yes").astype(int) for col in service_cols if col in df.columns
    )
    return df


def _add_contract_flags(df: pd.DataFrame) -> pd.DataFrame:
    df["is_month_to_month"] = (df["contract"] == "Month-to-month").astype(int)
    df["is_long_term"]      = (df["contract"] == "Two year").astype(int)
    return df


def _add_streaming_flag(df: pd.DataFrame) -> pd.DataFrame:
    df["has_streaming"] = (
        (df.get("streaming_tv", "No") == "Yes") |
        (df.get("streaming_movies", "No") == "Yes")
    ).astype(int)
    return df


def _add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4+yr"],
    )
    return df
