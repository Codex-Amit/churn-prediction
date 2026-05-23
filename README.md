# 📉 Customer Churn Prediction

A modular, component-based Python project that predicts whether a customer will leave a service using classification techniques. Supports **synthetic data** (quick testing) and **real datasets** (Kaggle Telco Customer Churn).

---

## ✨ Features

| Area | What's included |
|------|----------------|
| **Data** | Synthetic generator + real Kaggle dataset support |
| **Components** | Separate loader, cleaner, encoder, trainer, evaluator, visualizer |
| **Features** | Engineering (spend ratio, service count, contract flags) + selection |
| **Pipeline** | Single orchestrator ties all components together |
| **API** | FastAPI endpoint to serve live predictions |
| **Models** | Logistic Regression, Decision Tree, KNN, Random Forest, Gradient Boosting, SVM |
| **Evaluation** | Accuracy, F1, ROC-AUC — best model auto-selected |
| **Figures** | 8 publication-ready charts saved to `reports/figures/` |
| **Tests** | `pytest` suite across all components |

---

## 📁 Project Structure

```
churn-prediction/
├── config/
│   └── settings.py               # All paths, constants & hyperparameters
│
├── components/
│   ├── loader/
│   │   └── data_loader.py        # Load CSV or generate synthetic data
│   ├── cleaner/
│   │   └── data_cleaner.py       # Dedup, type coercion, target encoding
│   ├── encoder/
│   │   └── data_encoder.py       # Label encoding, scaling, train/test split
│   ├── trainer/
│   │   └── model_trainer.py      # Model catalogue, training, persistence
│   ├── evaluator/
│   │   └── model_evaluator.py    # Metrics, best-model selection
│   └── visualizer/
│       └── chart_builder.py      # All EDA and model evaluation charts
│
├── features/
│   ├── engineering/
│   │   └── feature_builder.py    # Derived features (spend ratio, num_services …)
│   └── selection/
│       └── feature_selector.py   # Importance-based feature ranking
│
├── pipeline/
│   └── churn_pipeline.py         # End-to-end orchestrator
│
├── api/
│   └── predict.py                # FastAPI prediction endpoint
│
├── utils/
│   └── logger.py                 # Centralised logging setup
│
├── tests/
│   └── test_pipeline.py          # pytest unit tests
│
├── data/
│   ├── raw/
│   │   ├── generate_data.py          # Synthetic data generator (quick testing)
│   │   ├── convert_real_data.py      # Converts Kaggle CSV → pipeline format
│   │   ├── kaggle_churn_raw.csv      # Your downloaded Kaggle CSV (git-ignored)
│   │   └── churn_data.csv            # Final CSV used by pipeline (git-ignored)
│   └── processed/
│       └── churn_processed.csv       # Feature-engineered CSV (git-ignored)
│
├── models/                       # Saved .pkl files (git-ignored)
├── reports/
│   └── figures/                  # Auto-generated PNGs
├── docs/
│
├── main.py                       # ▶ Entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1 — Clone & install

```bash
git clone https://github.com/Codex-Amit/churn-prediction.git
cd churn-prediction
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

### 2 — Prepare your data

You have **two options** — pick one:

---

#### Option A — Synthetic Data (Quick testing, no download needed)

Data is generated automatically when you run the pipeline. Skip to step 3.

---

#### Option B — Real Kaggle Data (Recommended)

**Step 1 — Download the dataset:**

1. Go to **https://www.kaggle.com/datasets/blastchar/telco-customer-churn**
2. Click **Download**
3. Rename the downloaded file to `kaggle_churn_raw.csv`
4. Place it in `data/raw/`

> ⚠️ The Kaggle Telco dataset has **7,043 rows** which is enough for the pipeline to train properly.

**Step 2 — Convert the downloaded CSV:**

```bash
python data/raw/convert_real_data.py
```

This converts the Kaggle column format (e.g. `SeniorCitizen`, `MonthlyCharges`) into the project schema and saves it as `data/raw/churn_data.csv`. It automatically fixes type issues and prints a summary of what it found.

---

### 3 — Run the full pipeline

```bash
python main.py
```

This will:
1. Load and clean the data
2. Engineer features (service count, spend ratio, contract flags)
3. Train all six models
4. Save the best model (by ROC-AUC) to `models/best_model.pkl`
5. Generate all 8 figures in `reports/figures/`

---

### 5 — Run tests

```bash
python -m pytest tests/ -v
```

---

## 📊 Models

| Model | Notes |
|-------|-------|
| `LogisticRegression` | Fast baseline with class balancing |
| `DecisionTree` | Interpretable tree (max depth 6) |
| `KNN` | K-Nearest Neighbours (k=7) |
| `RandomForest` | 200 trees, max depth 10, class balanced |
| `GradientBoosting` | 200 estimators, lr 0.05 |
| `SVM` | RBF kernel with probability calibration, class balanced |

Best model selected by **ROC-AUC** automatically.

---

## 📈 Output Figures

After running `main.py` the following PNGs appear in `reports/figures/`:

1. `01_churn_distribution.png` — Count & proportion of churned customers
2. `02_numerical_features.png` — Tenure, charges distribution by churn
3. `03_categorical_churn.png` — Churn rate by contract, internet, payment method
4. `04_correlation_heatmap.png` — Feature correlations
5. `05_roc_curves.png` — All models on one ROC plot
6. `06_model_comparison.png` — Accuracy / F1 / ROC-AUC bar chart
7. `07_confusion_matrix.png` — Best model confusion matrix
8. `08_feature_importance.png` — Top features driving churn predictions

---

## 🔑 Key Churn Factors Identified

- **Contract type** — Month-to-month customers churn significantly more
- **Internet service** — Fiber optic users show higher churn rates
- **Tenure** — Longer-tenured customers are less likely to churn
- **Online security / tech support** — Customers without these churn more
- **Monthly charges** — Higher charges correlate with higher churn

---

## 🔧 Configuration

All settings live in one place:

```python
# config/settings.py
TARGET_COL   = "churn"
TEST_SIZE    = 0.2
RANDOM_SEED  = 42
N_CUSTOMERS  = 5000   # synthetic data size
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- pandas · NumPy · scikit-learn
- Matplotlib · Seaborn
- FastAPI · Uvicorn · Pydantic
- pytest

---

## 📜 License

MIT — free to use, adapt, and distribute.