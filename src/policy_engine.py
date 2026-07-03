
"""Stage 2: Policy Engine - PD score to lending decision"""

def assign_risk_tier(pd_score):
    if pd_score < 0.12:
        return "A"
    elif pd_score < 0.25:
        return "B"
    elif pd_score < 0.45:
        return "C"
    else:
        return "D"

APPROVAL_POLICY = {
    "digital":        {"A": True, "B": True, "C": True,  "D": False},
    "sales_assisted": {"A": True, "B": True, "C": True,  "D": False},
}
HARD_DECLINE_PD = 0.55

TIER_TERMS = {
    "A": {"amount_ratio": 1.00, "rate_range": (6, 9)},
    "B": {"amount_ratio": 0.90, "rate_range": (9, 13)},
    "C": {"amount_ratio": 0.70, "rate_range": (13, 18)},
    "D": {"amount_ratio": 0.00, "rate_range": (18, 24)},
}

def score_application(pd_score, channel, requested_amount):
    tier = assign_risk_tier(pd_score)
    hard_decline = pd_score >= HARD_DECLINE_PD
    policy_approve = APPROVAL_POLICY.get(channel, APPROVAL_POLICY["digital"]).get(tier, False)
    approved = policy_approve and not hard_decline

    terms = TIER_TERMS[tier]
    if approved:
        approved_amount = round(requested_amount * terms["amount_ratio"], -2)
        rate_lo, rate_hi = terms["rate_range"]
        interest_rate = round(rate_lo + (rate_hi - rate_lo) * min(pd_score / 0.45, 1.0), 2)
    else:
        approved_amount = 0.0
        interest_rate = None

    return {
        "pd_score": round(pd_score, 4),
        "risk_tier": tier,
        "approved": approved,
        "approved_amount_kes": approved_amount,
        "interest_rate_monthly_pct": interest_rate,
        "decline_reason": "hard_decline_pd_threshold" if hard_decline else
                           ("policy_tier_decline" if not policy_approve else None),
    }
