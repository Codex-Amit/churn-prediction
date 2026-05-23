"""
components/encoder/data_encoder.py
Label-encodes categoricals, scales numericals, and splits train/test.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from config.settings import RANDOM_SEED, TARGET_COL, TEST_SIZE

logger = logging.getLogger(__name__)


def encode_and_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], MinMaxScaler]:
    """
    1. Label-encode all object/category columns
    2. Split into train/test (stratified)
    3. MinMax-scale features

    Returns: X_train, X_test, y_train, y_test, feature_cols, scaler
    """
    df = df.copy()
    le = LabelEncoder()

    for col in df.select_dtypes(include=["object", "category"]).columns:
        if col != TARGET_COL:
            encoded = le.fit_transform(df[col].astype("object").fillna("unknown"))
            df[col] = df[col].astype(object)
            df.loc[:, col] = pd.Series(encoded, index=df.index)

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols].astype(float).to_numpy()
    y = df[TARGET_COL].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    logger.info(
        "Split — Train: %d | Test: %d | Features: %d | Churn rate train: %.1f%%",
        len(X_train), len(X_test), len(feature_cols),
        y_train.mean() * 100,
    )
    return X_train, X_test, y_train, y_test, feature_cols, scaler
