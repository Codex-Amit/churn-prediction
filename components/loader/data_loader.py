"""
components/loader/data_loader.py
Responsible for loading raw data from disk.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import N_CUSTOMERS, RAW_DATA_FILE

logger = logging.getLogger(__name__)


def load_csv(filepath: str | Path) -> pd.DataFrame:
    """Load any CSV file and return a DataFrame."""
    df = pd.read_csv(filepath)
    logger.info("Loaded %d rows, %d columns from %s", *df.shape, filepath)
    return df


def generate_synthetic_data(n: int = N_CUSTOMERS, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic customer churn data."""
    np.random.seed(seed)

    customer_id = [f"CUST_{i:05d}" for i in range(1, n + 1)]
    gender         = np.random.choice(["Male", "Female"], n)
    senior_citizen = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner        = np.random.choice(["Yes", "No"], n)
    dependents     = np.random.choice(["Yes", "No"], n, p=[0.3, 0.7])
    tenure         = np.random.randint(1, 73, n)
    phone_service  = np.random.choice(["Yes", "No"], n, p=[0.9, 0.1])
    multiple_lines = np.where(phone_service == "No", "No phone service",
                              np.random.choice(["Yes", "No"], n))
    internet       = np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])

    def svc(col): return np.where(internet == "No", "No internet service",
                                   np.random.choice(["Yes", "No"], n))

    contract       = np.random.choice(["Month-to-month", "One year", "Two year"],
                                      n, p=[0.55, 0.24, 0.21])
    paperless      = np.random.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment        = np.random.choice(
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"], n)
    monthly        = np.round(np.random.uniform(18, 120, n), 2)
    total          = np.round(monthly * tenure + np.random.normal(0, 50, n), 2)
    total          = np.clip(total, 0, None)

    churn_prob = np.clip(
        0.05
        + 0.25 * (contract == "Month-to-month")
        + 0.10 * (internet == "Fiber optic")
        + 0.08 * (svc("online_security") == "No")
        - 0.15 * (tenure > 36)
        - 0.10 * (contract == "Two year")
        + np.random.normal(0, 0.05, n),
        0.02, 0.95,
    )
    churn = np.where(np.random.rand(n) < churn_prob, "Yes", "No")

    return pd.DataFrame({
        "customer_id": customer_id, "gender": gender,
        "senior_citizen": senior_citizen, "partner": partner,
        "dependents": dependents, "tenure": tenure,
        "phone_service": phone_service, "multiple_lines": multiple_lines,
        "internet_service": internet,
        "online_security": svc("os"), "online_backup": svc("ob"),
        "device_protection": svc("dp"), "tech_support": svc("ts"),
        "streaming_tv": svc("stv"), "streaming_movies": svc("smv"),
        "contract": contract, "paperless_billing": paperless,
        "payment_method": payment, "monthly_charges": monthly,
        "total_charges": total, "churn": churn,
    })


def load_or_generate(filepath: str | Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Load CSV if it exists, otherwise generate synthetic data and save it."""
    filepath = Path(filepath)
    if filepath.exists():
        logger.info("Found existing data at %s", filepath)
        return load_csv(filepath)

    logger.info("No data found — generating synthetic dataset …")
    df = generate_synthetic_data()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info("Saved %d rows → %s  (churn rate %.1f%%)",
                len(df), filepath, (df["churn"] == "Yes").mean() * 100)
    return df
