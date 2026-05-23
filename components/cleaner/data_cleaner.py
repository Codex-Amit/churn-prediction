"""
components/cleaner/data_cleaner.py
Handles deduplication, type coercion, and outlier handling.
"""

from __future__ import annotations

import logging

import pandas as pd

from config.settings import DROP_COLS, TARGET_COL

logger = logging.getLogger(__name__)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline: dedup → drop IDs → coerce types → binary target."""
    df = _drop_duplicates(df)
    df = _drop_unused_columns(df)
    df = _coerce_types(df)
    df = _encode_target(df)
    logger.info("Clean complete: %d rows | churn rate %.1f%%",
                len(df), df[TARGET_COL].mean() * 100)
    return df


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    if removed:
        logger.info("Removed %d duplicate rows", removed)
    return df


def _drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in DROP_COLS if c in df.columns]
    return df.drop(columns=cols, errors="ignore")


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df["total_charges"] = df["total_charges"].fillna(
        df["monthly_charges"] * df["tenure"]
    )
    return df


def _encode_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TARGET_COL] = (df[TARGET_COL].astype(str).str.strip().str.lower() == "yes").astype(int)
    return df
