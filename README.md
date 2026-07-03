# Credit Scoring System - Digital Lending PD Model & Policy Engine

A two-stage credit scoring pipeline for a digital lending / PDL business, built on synthetic data that mirrors a real fintech's data ecosystem: CRB bureau data, bank statement analytics, app behavioral signals, and lender-side repayment history - unified across both digitally-acquired and sales-assisted (field agent) customers.

## Problem Statement

The business needs one scoring stack that decides who to lend to and on what terms, serving two acquisition channels that carry different data richness:
- Digital channel: rich app behavioral data (sessions, permissions, tenure), thinner traditional credit history
- Sales-assisted channel: agent-vetted customers, no app data, often thinner CRB files

The model must serve both flows fairly without one channel silently underperforming.

## Architecture

**Stage 1 - Probability of Default (PD) Model**
LightGBM binary classifier trained on closed loans only (rejected/active applications excluded - no definitive label). Channel is included as a categorical feature rather than training separate models, so the stack stays maintainable while still letting the model learn channel-specific split patterns.

- Test ROC-AUC: 0.825
- Gini coefficient: 0.650
- KS statistic: 0.532
- AUC holds up on both channels individually (0.83 digital / 0.81 sales-assisted)

**Stage 2 - Policy Engine**
Converts PD score into a business decision: risk tier (A-D), approve/decline, credit limit (as a ratio of requested amount), and risk-based pricing. Thresholds are configurable per channel.

## Repo Structure

data/            synthetic datasets (customers, CRB, bank statements, app behavior, applications, payment history, merged master)
src/             feature engineering + policy engine (shared between training and serving)
models/          trained model, metadata, feature importance
api/             FastAPI scoring service
dashboard/       Streamlit app (portfolio overview, live scoring, model performance)

## Running it

pip install -r requirements.txt

API: uvicorn api.main:app --reload --port 8000 (docs at http://127.0.0.1:8000/docs)

Dashboard: streamlit run dashboard/app.py

## Data Notes

This is synthetic data, generated with realistic feature correlations (CRB score, salary regularity, prior repayment behavior, and cashflow volatility drive default risk) but should not be interpreted as real lender portfolio data. The overall default rate (~49.5%) is deliberately elevated versus a real PDL book (typically 5-20%) to ensure sufficient positive-class signal for training without heavy resampling.

## Key Design Decisions

- No target leakage: features exclude post-outcome fields (loan_status, days_late, repaid_pct)
- Channel-structural missingness preserved: sales-assisted customers have no app data - filled with an explicit sentinel (-1 / "no_app_no_device") rather than imputed, letting the model learn "no app data" as its own signal
- Reject inference not yet implemented: rejected applications have no observed outcome and are excluded from training; a natural next step is reject inference or a survival-style model for censored outcomes
