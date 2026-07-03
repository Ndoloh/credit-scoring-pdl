
"""
Credit Scoring API - Stage 1 (PD model) + Stage 2 (policy engine)
Run with: uvicorn api.main:app --reload --port 8000
Docs at: http://127.0.0.1:8000/docs
"""

import sys
import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from features import engineer_features, ALL_FEATURES_ENGINEERED, CATEGORICAL_FEATURES
from policy_engine import score_application

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "pd_model.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")

app = FastAPI(title="Aventus-style Credit Scoring API", version="1.0")

model = joblib.load(MODEL_PATH)
with open(METADATA_PATH) as f:
    METADATA = json.load(f)


class ScoringRequest(BaseModel):
    acquisition_channel: str = Field(..., example="digital")
    region: str = Field(..., example="Nairobi")
    employment_type: str = Field(..., example="formal_employed")
    gender: str = Field(..., example="M")
    device_type: Optional[str] = Field(None, example="android_mid")

    age: int
    monthly_income_kes: float
    requested_amount_kes: float

    crb_score: int
    num_active_loans_other_lenders: int = 0
    total_outstanding_debt_kes: float = 0
    num_defaults_last_24m: int = 0
    max_days_past_due_24m: int = 0
    crb_listed_negative: int = 0

    avg_monthly_inflow_kes: float
    avg_monthly_outflow_kes: float
    inflow_volatility_cv: float = 0.3
    avg_closing_balance_kes: float = 0
    salary_regularity_score: float = 0.5
    num_bounced_payments_6m: int = 0
    mobile_money_txn_count_monthly: int = 20
    num_distinct_income_sources: int = 1

    app_tenure_days: Optional[float] = None
    app_sessions_per_week: Optional[float] = None
    contacts_permission_granted: Optional[float] = None
    sms_permission_granted: Optional[float] = None
    num_support_tickets_90d: Optional[float] = None
    profile_completeness_pct: Optional[float] = None
    days_since_last_app_open: Optional[float] = None

    prior_completed_loans_with_us: int = 0
    prior_defaults_with_us: int = 0
    prior_avg_days_late_with_us: float = 0
    prior_repayment_rate_with_us: float = 1.0


@app.get("/")
def root():
    return {"status": "ok", "model_auc": METADATA.get("test_auc"), "model_gini": METADATA.get("test_gini")}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/score")
def score(req: ScoringRequest):
    try:
        row = pd.DataFrame([req.dict()])
        row_feat = engineer_features(row)

        for c in CATEGORICAL_FEATURES:
            row_feat[c] = row_feat[c].astype("category")

        X = row_feat[ALL_FEATURES_ENGINEERED]
        pd_score = float(model.predict(X)[0])

        decision = score_application(
            pd_score=pd_score,
            channel=req.acquisition_channel,
            requested_amount=req.requested_amount_kes,
        )
        return decision
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
