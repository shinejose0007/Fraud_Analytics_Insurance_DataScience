import pandas as pd


CATEGORICAL_COLS = [
    "region",
    "vehicle_type",
    "contract_type",
    "application_channel",
    "occupation",
]

NUMERIC_COLS = [
    "customer_age",
    "vehicle_age",
    "premium_amount",
    "claim_amount",
    "num_claims_12m",
    "num_claims_total",
    "payment_delay_count",
    "previous_cancellations",
    "address_changes_12m",
    "policy_changes_3m",
    "contract_age_months",
    "accident_report_delay_days",
    "application_hour",
    "claims_to_premium_ratio",
    "short_contract_large_claim",
    "high_claim_frequency_flag",
    "high_payment_delay_flag",
    "multiple_recent_changes_flag",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "claims_to_premium_ratio" not in df.columns:
        df["claims_to_premium_ratio"] = df["claim_amount"] / (df["premium_amount"] + 1)

    df["claims_to_premium_ratio"] = df["claims_to_premium_ratio"].fillna(0)

    df["short_contract_large_claim"] = (
        (df["contract_age_months"] < 6) & (df["claim_amount"] > 3500)
    ).astype(int)

    df["high_claim_frequency_flag"] = (df["num_claims_12m"] >= 2).astype(int)
    df["high_payment_delay_flag"] = (df["payment_delay_count"] >= 2).astype(int)

    df["multiple_recent_changes_flag"] = (
        (df["address_changes_12m"] >= 2) | (df["policy_changes_3m"] >= 2)
    ).astype(int)

    df["late_night_application_flag"] = (
        (df["application_hour"] >= 0) & (df["application_hour"] <= 5)
    ).astype(int)

    if "late_night_application_flag" not in NUMERIC_COLS:
        pass

    return df


def get_model_columns():
    return NUMERIC_COLS + CATEGORICAL_COLS