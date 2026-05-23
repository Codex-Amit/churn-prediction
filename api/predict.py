"""
api/predict.py
Lightweight FastAPI endpoint to serve churn predictions.

Run with:
    pip install fastapi uvicorn
    uvicorn api.predict:app --reload
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from components.trainer.model_trainer import load_model
from config.settings import BEST_MODEL_FILE


# ── Request / Response schemas ────────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    tenure:           float
    monthly_charges:  float
    total_charges:    float
    senior_citizen:   int   = 0
    num_services:     int   = 3
    is_month_to_month: int  = 0
    is_long_term:     int   = 0
    has_streaming:    int   = 0
    avg_monthly_spend: float = 50.0

    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 12,
                "monthly_charges": 75.5,
                "total_charges": 906.0,
                "senior_citizen": 0,
                "num_services": 4,
                "is_month_to_month": 1,
                "is_long_term": 0,
                "has_streaming": 1,
                "avg_monthly_spend": 75.5,
            }
        }


class PredictionResponse(BaseModel):
    churn_prediction: str
    churn_probability: float
    risk_level: str


# ── App ───────────────────────────────────────────────────────────────────────

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Customer Churn Prediction API",
        description="Predicts whether a customer will churn based on their profile.",
        version="1.0.0",
    )

    _model = None

    def get_model():
        global _model
        if _model is None:
            if not BEST_MODEL_FILE.exists():
                raise FileNotFoundError(
                    f"No trained model found at {BEST_MODEL_FILE}. "
                    "Run `python main.py` first."
                )
            _model = load_model(BEST_MODEL_FILE)
        return _model

    @app.get("/")
    def root():
        return {"message": "Customer Churn Prediction API is running."}

    @app.get("/health")
    def health():
        return {"status": "ok", "model_ready": BEST_MODEL_FILE.exists()}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(customer: CustomerFeatures):
        import numpy as np
        model = get_model()

        features = [[
            customer.tenure, customer.monthly_charges, customer.total_charges,
            customer.senior_citizen, customer.num_services,
            customer.is_month_to_month, customer.is_long_term,
            customer.has_streaming, customer.avg_monthly_spend,
        ]]

        pred      = model.predict(features)[0]
        prob      = model.predict_proba(features)[0][1] if hasattr(model, "predict_proba") else 0.5
        risk      = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"

        return PredictionResponse(
            churn_prediction="Yes" if pred == 1 else "No",
            churn_probability=round(float(prob), 4),
            risk_level=risk,
        )

    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

else:
    print("FastAPI not installed. Run: pip install fastapi uvicorn")
