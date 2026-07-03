
"""Feature engineering - shared between training and serving"""
import numpy as np
import pandas as pd

NUMERIC_FEATURES = ['age', 'monthly_income_kes', 'crb_score', 'num_active_loans_other_lenders', 'total_outstanding_debt_kes', 'num_defaults_last_24m', 'max_days_past_due_24m', 'crb_listed_negative', 'avg_monthly_inflow_kes', 'avg_monthly_outflow_kes', 'inflow_volatility_cv', 'avg_closing_balance_kes', 'salary_regularity_score', 'num_bounced_payments_6m', 'mobile_money_txn_count_monthly', 'num_distinct_income_sources', 'app_tenure_days', 'app_sessions_per_week', 'contacts_permission_granted', 'sms_permission_granted', 'num_support_tickets_90d', 'profile_completeness_pct', 'days_since_last_app_open', 'prior_completed_loans_with_us', 'prior_defaults_with_us', 'prior_avg_days_late_with_us', 'prior_repayment_rate_with_us', 'requested_amount_kes']
CATEGORICAL_FEATURES = ['acquisition_channel', 'region', 'employment_type', 'gender', 'device_type']

def engineer_features(df):
    out = df.copy()
    out["debt_to_income_ratio"] = (
        out["total_outstanding_debt_kes"] / out["monthly_income_kes"].replace(0, np.nan)
    ).fillna(0)
    out["requested_to_income_ratio"] = (
        out["requested_amount_kes"] / out["monthly_income_kes"].replace(0, np.nan)
    ).fillna(0)
    out["net_cashflow_kes"] = out["avg_monthly_inflow_kes"] - out["avg_monthly_outflow_kes"]
    out["outflow_to_inflow_ratio"] = (
        out["avg_monthly_outflow_kes"] / out["avg_monthly_inflow_kes"].replace(0, np.nan)
    ).fillna(1.0)
    out["is_digital_channel"] = (out["acquisition_channel"] == "digital").astype(int)

    app_cols = [
        "app_tenure_days", "app_sessions_per_week", "contacts_permission_granted",
        "sms_permission_granted", "num_support_tickets_90d",
        "profile_completeness_pct", "days_since_last_app_open",
    ]
    for c in app_cols:
        out[c] = out[c].fillna(-1)
    out["device_type"] = out["device_type"].fillna("no_app_no_device")
    return out

NUMERIC_FEATURES_ENGINEERED = NUMERIC_FEATURES + [
    "debt_to_income_ratio", "requested_to_income_ratio",
    "net_cashflow_kes", "outflow_to_inflow_ratio", "is_digital_channel",
]
ALL_FEATURES_ENGINEERED = NUMERIC_FEATURES_ENGINEERED + CATEGORICAL_FEATURES
