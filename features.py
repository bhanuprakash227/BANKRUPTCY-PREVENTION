import pandas as pd

RISK_COLS = ["industrial_risk", "management_risk", "operating_risk"]            # higher = worse
STRENGTH_COLS = ["financial_flexibility", "credibility", "competitiveness"]     # higher = better
BASE_COLS = RISK_COLS + STRENGTH_COLS


def base_features(data):
    """Original 6 features only."""
    return pd.DataFrame(data)[BASE_COLS].copy()


def add_features(data):
    """Original 6 features + 4 engineered features."""
    d = base_features(data)
    d["risk_score"] = d[RISK_COLS].mean(axis=1)               # average risk (0 = safe, 1 = dangerous)
    d["strength_score"] = d[STRENGTH_COLS].mean(axis=1)       # average strength (0 = weak, 1 = strong)
    d["net_score"] = d["strength_score"] - d["risk_score"]    # overall health of the company
    d["weak_count"] = (d[STRENGTH_COLS] == 0).sum(axis=1)     # how many strength areas are at the lowest level
    return d
