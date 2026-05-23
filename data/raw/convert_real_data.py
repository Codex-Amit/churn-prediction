"""
data/raw/convert_real_data.py

Converts the Kaggle Telco Customer Churn dataset into the
format used by the churn-prediction pipeline.

Usage:
    python data/raw/convert_real_data.py

Place your downloaded Kaggle CSV at:
    data/raw/kaggle_churn_raw.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_FILE  = Path(__file__).parent / "kaggle_churn_raw.csv"
OUTPUT_FILE = Path(__file__).parent / "churn_data.csv"
# ───────────────────────────────────────────────────────────────────────────────

# Kaggle column → pipeline column
COLUMN_MAP = {
    "customerID":       "customer_id",
    "gender":           "gender",
    "SeniorCitizen":    "senior_citizen",
    "Partner":          "partner",
    "Dependents":       "dependents",
    "tenure":           "tenure",
    "PhoneService":     "phone_service",
    "MultipleLines":    "multiple_lines",
    "InternetService":  "internet_service",
    "OnlineSecurity":   "online_security",
    "OnlineBackup":     "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport":      "tech_support",
    "StreamingTV":      "streaming_tv",
    "StreamingMovies":  "streaming_movies",
    "Contract":         "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod":    "payment_method",
    "MonthlyCharges":   "monthly_charges",
    "TotalCharges":     "total_charges",
    "Churn":            "churn",
}


def convert(input_path: Path = INPUT_FILE,
            output_path: Path = OUTPUT_FILE) -> None:

    # ── Check file exists ──────────────────────────────────────────────────────
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        print(f"   Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        print(f"   Rename it to:  kaggle_churn_raw.csv")
        print(f"   Place it in:   data/raw/")
        return

    # ── Load ───────────────────────────────────────────────────────────────────
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows from {input_path.name}")
    print(f"Columns found: {df.columns.tolist()}")

    # ── Rename columns ─────────────────────────────────────────────────────────
    rename = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)
    missing = [v for k, v in COLUMN_MAP.items() if k not in rename]
    if missing:
        print(f"  Columns not found (will be skipped): {missing}")

    # ── Fix types ──────────────────────────────────────────────────────────────
    # TotalCharges is stored as string with spaces in Kaggle file
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df["total_charges"] = df["total_charges"].fillna(
        df["monthly_charges"] * df["tenure"]
    )

    # Strip whitespace from all string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # ── Remove duplicates ──────────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < before:
        print(f"  Removed {before - len(df)} duplicate rows")

    # ── Save ───────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n✅ Done! {len(df):,} rows saved → {output_path.name}")
    print(f"   Churn rate  : {(df['churn'].str.lower()=='yes').mean():.1%}")
    print(f"   Tenure range: {df['tenure'].min()}–{df['tenure'].max()} months")
    print(f"   Avg monthly : ${df['monthly_charges'].mean():.2f}")
    print(f"\n   Now run: python main.py")


if __name__ == "__main__":
    convert()
