
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from features import engineer_features, ALL_FEATURES_ENGINEERED, CATEGORICAL_FEATURES
from policy_engine import score_application

st.set_page_config(page_title="Credit Scoring Dashboard", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "pd_model.pkl")
META_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")
DATA_PATH = os.path.join(BASE_DIR, "data", "master_dataset.csv")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(META_PATH) as f:
        return json.load(f)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


model = load_model()
metadata = load_metadata()

st.title("Credit Scoring Dashboard")
st.caption("PD model (Stage 1) + Policy engine (Stage 2) - unified across digital and sales-assisted channels")

tab1, tab2, tab3 = st.tabs(["Portfolio Overview", "Score a New Applicant", "Model Performance"])

with tab1:
    df = load_data()
    approval_rate = df["approved"].mean()
    digital_share = (df["acquisition_channel"] == "digital").mean()
    closed = df[df["target_default"].notna()]
    default_rate = closed["target_default"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Applications", f"{len(df):,}")
    col2.metric("Approval Rate", f"{approval_rate:.1%}")
    col3.metric("Default Rate (closed loans)", f"{default_rate:.1%}")
    col4.metric("Digital Channel Share", f"{digital_share:.1%}")

    st.subheader("Default Rate by Risk Tier")
    tier_stats = closed.groupby("internal_risk_tier_at_application")["target_default"].agg(["mean", "count"])
    st.bar_chart(tier_stats["mean"])
    st.dataframe(tier_stats)

    st.subheader("Approval Rate by Channel")
    channel_approval = df.groupby("acquisition_channel")["approved"].mean()
    st.bar_chart(channel_approval)

with tab2:
    st.subheader("Score a New Loan Applicant")
    c1, c2, c3 = st.columns(3)
    with c1:
        channel = st.selectbox("Acquisition Channel", ["digital", "sales_assisted"])
        region = st.selectbox("Region", metadata["categorical_levels"]["region"])
        employment = st.selectbox("Employment Type", metadata["categorical_levels"]["employment_type"])
        gender = st.selectbox("Gender", ["M", "F"])
        age = st.number_input("Age", 18, 65, 30)
    with c2:
        income = st.number_input("Monthly Income (KES)", 5000, 300000, 40000, step=1000)
        requested = st.number_input("Requested Amount (KES)", 2000, 150000, 20000, step=1000)
        crb_score = st.slider("CRB Score", 200, 900, 600)
        num_defaults = st.number_input("Defaults Last 24m", 0, 10, 0)
        crb_negative = st.checkbox("CRB Listed Negative")
    with c3:
        inflow = st.number_input("Avg Monthly Inflow (KES)", 5000, 500000, 45000, step=1000)
        outflow = st.number_input("Avg Monthly Outflow (KES)", 3000, 500000, 38000, step=1000)
        salary_reg = st.slider("Salary Regularity Score", 0.0, 1.0, 0.7)
        prior_loans = st.number_input("Prior Completed Loans With Us", 0, 20, 0)
        prior_repay_rate = st.slider("Prior Repayment Rate With Us", 0.0, 1.0, 1.0)

    if st.button("Score Applicant", type="primary"):
        row = pd.DataFrame([{
            "acquisition_channel": channel,
            "region": region,
            "employment_type": employment,
            "gender": gender,
            "device_type": "android_mid" if channel == "digital" else None,
            "age": age,
            "monthly_income_kes": income,
            "requested_amount_kes": requested,
            "crb_score": crb_score,
            "num_active_loans_other_lenders": 0,
            "total_outstanding_debt_kes": 0,
            "num_defaults_last_24m": num_defaults,
            "max_days_past_due_24m": 30 if num_defaults > 0 else 0,
            "crb_listed_negative": int(crb_negative),
            "avg_monthly_inflow_kes": inflow,
            "avg_monthly_outflow_kes": outflow,
            "inflow_volatility_cv": 0.3,
            "avg_closing_balance_kes": inflow - outflow,
            "salary_regularity_score": salary_reg,
            "num_bounced_payments_6m": 0,
            "mobile_money_txn_count_monthly": 25,
            "num_distinct_income_sources": 1,
            "app_tenure_days": 150 if channel == "digital" else None,
            "app_sessions_per_week": 4 if channel == "digital" else None,
            "contacts_permission_granted": 1 if channel == "digital" else None,
            "sms_permission_granted": 1 if channel == "digital" else None,
            "num_support_tickets_90d": 0 if channel == "digital" else None,
            "profile_completeness_pct": 80 if channel == "digital" else None,
            "days_since_last_app_open": 2 if channel == "digital" else None,
            "prior_completed_loans_with_us": prior_loans,
            "prior_defaults_with_us": 0,
            "prior_avg_days_late_with_us": 0,
            "prior_repayment_rate_with_us": prior_repay_rate,
        }])
        row_feat = engineer_features(row)
        for c in CATEGORICAL_FEATURES:
            row_feat[c] = row_feat[c].astype("category")
        X = row_feat[ALL_FEATURES_ENGINEERED]
        pd_score = float(model.predict(X)[0])
        decision = score_application(pd_score, channel, requested)

        st.divider()
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("PD Score", f"{decision['pd_score']:.1%}")
        r2.metric("Risk Tier", decision["risk_tier"])
        r3.metric("Decision", "APPROVED" if decision["approved"] else "DECLINED")
        if decision["approved"]:
            approved_amt = decision["approved_amount_kes"]
            r4.metric("Approved Amount", f"KES {approved_amt:,.0f}")
            rate = decision["interest_rate_monthly_pct"]
            st.success(f"Interest rate: {rate}% per month")
        else:
            st.error(f"Decline reason: {decision['decline_reason']}")

with tab3:
    st.subheader("Model Performance Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("ROC-AUC", metadata.get("test_auc"))
    m2.metric("Gini Coefficient", metadata.get("test_gini"))
    m3.metric("KS Statistic", metadata.get("test_ks"))
    st.caption(f"Trained on {metadata.get('trained_on_rows'):,} closed loans")

    fi_path = os.path.join(BASE_DIR, "models", "feature_importance.csv")
    if os.path.exists(fi_path):
        fi = pd.read_csv(fi_path).head(15)
        st.subheader("Top 15 Features by Importance")
        st.bar_chart(fi.set_index("feature")["gain"])
