"""
config/settings.py — Central configuration for the entire project.
All constants, paths, and hyperparameters live here.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).resolve().parents[1]
DATA_RAW_DIR    = ROOT_DIR / "data" / "raw"
DATA_PROC_DIR   = ROOT_DIR / "data" / "processed"
MODELS_DIR      = ROOT_DIR / "models"
REPORTS_DIR     = ROOT_DIR / "reports" / "figures"

RAW_DATA_FILE   = DATA_RAW_DIR  / "churn_data.csv"
PROC_DATA_FILE  = DATA_PROC_DIR / "churn_processed.csv"
BEST_MODEL_FILE = MODELS_DIR    / "best_model.pkl"
METRICS_FILE    = MODELS_DIR    / "metrics.json"

# ── Data ──────────────────────────────────────────────────────────────────────
TARGET_COL = "churn"
DROP_COLS  = ["customer_id"]

CATEGORICAL_COLS = [
    "gender", "partner", "dependents", "phone_service", "multiple_lines",
    "internet_service", "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies",
    "contract", "paperless_billing", "payment_method",
]
NUMERICAL_COLS = ["tenure", "monthly_charges", "total_charges", "senior_citizen"]

# ── Training ──────────────────────────────────────────────────────────────────
TEST_SIZE   = 0.2
RANDOM_SEED = 42
TOP_FEATURES = 15

# ── Synthetic data ────────────────────────────────────────────────────────────
N_CUSTOMERS = 5000
