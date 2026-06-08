import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import roc_curve, auc

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.database import load_from_csv, load_from_mysql, test_mysql_connection, save_predictions_to_mysql
from src.feature_engineering import add_features
from src.model_utils import train_models, score_dataframe, get_feature_importance
from src.scoring import risk_level_from_score, recommended_action


st.set_page_config(
    page_title="Insurance Fraud Analytics Platform",
    page_icon="🔎",
    layout="wide",
)


@st.cache_data
def cached_load_csv():
    return load_from_csv()


@st.cache_data(ttl=60)
def cached_load_mysql():
    return load_from_mysql()


@st.cache_resource
def cached_train_models(df):
    return train_models(df)


def money(value):
    return f"€{value:,.0f}".replace(",", ".")


def load_data(source):
    if source == "MySQL database":
        try:
            return cached_load_mysql()
        except Exception as exc:
            st.warning(f"MySQL could not be loaded. Falling back to CSV. Details: {exc}")
            return cached_load_csv()
    return cached_load_csv()


def sidebar():
    st.sidebar.title("Insurance Fraud Analytics")
    st.sidebar.caption("Data Science portfolio project for fraud management and risk scoring.")

    source = st.sidebar.radio("Data source", ["CSV file", "MySQL database"])

    if source == "MySQL database":
        ok, msg = test_mysql_connection()
        if ok:
            st.sidebar.success(msg)
        else:
            st.sidebar.error(msg)
            st.sidebar.info("Run `python scripts/init_mysql.py` after setting your `.env` file.")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Executive Overview",
            "Fraud Pattern Analysis",
            "Risk Scoring",
            "Model Performance",
            "Business Case",
            "Data Browser",
        ],
    )
    return source, page


def page_overview(df, scored_df, metrics):
    st.title("Executive Overview")
    st.write("Management-oriented overview of synthetic insurance applications, fraud risk, and portfolio anomalies.")

    total_cases = len(scored_df)
    fraud_rate = scored_df["fraud_label"].mean() * 100
    high_risk_cases = (scored_df["risk_level"] == "High Risk").sum()
    avg_claim = scored_df["claim_amount"].mean()
    estimated_exposure = scored_df.loc[scored_df["risk_level"] == "High Risk", "claim_amount"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Applications", f"{total_cases:,}")
    c2.metric("Observed Fraud Rate", f"{fraud_rate:.1f}%")
    c3.metric("High-Risk Cases", f"{high_risk_cases:,}")
    c4.metric("Avg. Claim Amount", money(avg_claim))
    c5.metric("High-Risk Claim Exposure", money(estimated_exposure))

    st.subheader("Risk Distribution")
    risk_counts = scored_df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]
    fig = px.bar(risk_counts, x="Risk Level", y="Count", text="Count")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Fraud Probability by Region")
    region_df = scored_df.groupby("region", as_index=False).agg(
        avg_risk_score=("fraud_risk_score", "mean"),
        fraud_rate=("fraud_label", "mean"),
        applications=("application_id", "count"),
    )
    region_df["fraud_rate"] *= 100
    fig = px.scatter(
        region_df,
        x="applications",
        y="avg_risk_score",
        size="fraud_rate",
        color="region",
        hover_data=["fraud_rate"],
        labels={"avg_risk_score": "Average Risk Score", "applications": "Applications"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model Summary")
    model_rows = []
    for model_name, m in metrics.items():
        model_rows.append({
            "Model": model_name,
            "Accuracy": round(m["accuracy"], 3),
            "Precision": round(m["precision"], 3),
            "Recall": round(m["recall"], 3),
            "F1": round(m["f1"], 3),
            "ROC-AUC": round(m["roc_auc"], 3),
        })
    st.dataframe(pd.DataFrame(model_rows), use_container_width=True)


def page_patterns(scored_df):
    st.title("Fraud Pattern Analysis")
    st.write("Explore suspicious patterns across customer, contract, vehicle, channel, and claim features.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Fraud Rate by Application Channel")
        channel = scored_df.groupby("application_channel", as_index=False)["fraud_label"].mean()
        channel["fraud_rate"] = channel["fraud_label"] * 100
        fig = px.bar(channel, x="application_channel", y="fraud_rate", text="fraud_rate")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Fraud Rate by Contract Type")
        contract = scored_df.groupby("contract_type", as_index=False)["fraud_label"].mean()
        contract["fraud_rate"] = contract["fraud_label"] * 100
        fig = px.bar(contract, x="contract_type", y="fraud_rate", text="fraud_rate")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Risk Score vs Claims-to-Premium Ratio")
    sample = scored_df.sample(min(1200, len(scored_df)), random_state=42)
    fig = px.scatter(
        sample,
        x="claims_to_premium_ratio",
        y="fraud_risk_score",
        color="risk_level",
        hover_data=["application_id", "region", "vehicle_type", "contract_type"],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Risk Drivers in the Data")
    driver_cols = [
        "short_contract_large_claim",
        "high_claim_frequency_flag",
        "high_payment_delay_flag",
        "multiple_recent_changes_flag",
        "previous_cancellations",
    ]
    driver_summary = []
    for col in driver_cols:
        subset = scored_df[scored_df[col] == 1]
        fraud_rate = subset["fraud_label"].mean() * 100 if len(subset) else 0
        avg_score = subset["fraud_risk_score"].mean() if len(subset) else 0
        driver_summary.append({
            "Risk Driver": col,
            "Cases": len(subset),
            "Fraud Rate %": round(fraud_rate, 2),
            "Average Risk Score": round(avg_score, 2),
        })
    st.dataframe(pd.DataFrame(driver_summary), use_container_width=True)


def page_risk_scoring(scored_df, model):
    st.title("Risk Scoring")
    st.write("Select an existing application or enter a new case to calculate a fraud risk score.")

    mode = st.radio("Scoring mode", ["Score existing application", "Score new manual case"])

    if mode == "Score existing application":
        app_id = st.selectbox("Select application ID", scored_df["application_id"].head(500).tolist())
        row = scored_df[scored_df["application_id"] == app_id].copy()

        if not row.empty:
            score = row["fraud_risk_score"].iloc[0]
            risk_level = row["risk_level"].iloc[0]
            action = recommended_action(risk_level)

            c1, c2, c3 = st.columns(3)
            c1.metric("Fraud Risk Score", f"{score:.1f}/100")
            c2.metric("Risk Level", risk_level)
            c3.metric("Recommended Action", action)

            st.subheader("Application Details")
            st.dataframe(row.T.rename(columns={row.index[0]: "Value"}), use_container_width=True)

    else:
        st.subheader("Manual Input")

        col1, col2, col3 = st.columns(3)
        with col1:
            customer_age = st.number_input("Customer age", 18, 90, 42)
            vehicle_age = st.number_input("Vehicle age", 0, 30, 6)
            premium_amount = st.number_input("Premium amount", 100.0, 5000.0, 750.0)
            claim_amount = st.number_input("Claim amount", 0.0, 50000.0, 3500.0)
            num_claims_12m = st.number_input("Claims in last 12 months", 0, 10, 1)

        with col2:
            num_claims_total = st.number_input("Total claims", 0, 20, 2)
            payment_delay_count = st.number_input("Payment delays", 0, 10, 0)
            previous_cancellations = st.selectbox("Previous cancellations", [0, 1])
            address_changes_12m = st.number_input("Address changes in last 12 months", 0, 10, 0)
            policy_changes_3m = st.number_input("Policy changes in last 3 months", 0, 10, 0)

        with col3:
            contract_age_months = st.number_input("Contract age in months", 1, 180, 18)
            accident_report_delay_days = st.number_input("Accident report delay in days", 0, 60, 5)
            application_hour = st.slider("Application hour", 0, 23, 14)
            region = st.selectbox("Region", sorted(scored_df["region"].unique()))
            vehicle_type = st.selectbox("Vehicle type", sorted(scored_df["vehicle_type"].unique()))
            contract_type = st.selectbox("Contract type", sorted(scored_df["contract_type"].unique()))
            application_channel = st.selectbox("Application channel", sorted(scored_df["application_channel"].unique()))
            occupation = st.selectbox("Occupation", sorted(scored_df["occupation"].unique()))

        manual = pd.DataFrame([{
            "application_id": "MANUAL_CASE",
            "customer_age": customer_age,
            "region": region,
            "vehicle_type": vehicle_type,
            "vehicle_age": vehicle_age,
            "contract_type": contract_type,
            "application_channel": application_channel,
            "occupation": occupation,
            "premium_amount": premium_amount,
            "claim_amount": claim_amount,
            "num_claims_12m": num_claims_12m,
            "num_claims_total": num_claims_total,
            "payment_delay_count": payment_delay_count,
            "previous_cancellations": previous_cancellations,
            "address_changes_12m": address_changes_12m,
            "policy_changes_3m": policy_changes_3m,
            "contract_age_months": contract_age_months,
            "accident_report_delay_days": accident_report_delay_days,
            "application_hour": application_hour,
            "fraud_label": 0,
            "claims_to_premium_ratio": claim_amount / (premium_amount + 1),
            "short_contract_large_claim": int(contract_age_months < 6 and claim_amount > 3500),
            "high_claim_frequency_flag": int(num_claims_12m >= 2),
            "high_payment_delay_flag": int(payment_delay_count >= 2),
            "multiple_recent_changes_flag": int(address_changes_12m >= 2 or policy_changes_3m >= 2),
        }])

        scored_manual = score_dataframe(manual, model)
        score = scored_manual["fraud_risk_score"].iloc[0]
        risk_level = scored_manual["risk_level"].iloc[0]
        action = recommended_action(risk_level)

        c1, c2, c3 = st.columns(3)
        c1.metric("Fraud Risk Score", f"{score:.1f}/100")
        c2.metric("Risk Level", risk_level)
        c3.metric("Recommended Action", action)


def page_model_performance(scored_df, metrics, models):
    st.title("Model Performance")
    st.write("Technical evaluation of the fraud scoring models.")

    model_name = st.selectbox("Select model", list(models.keys()))
    m = metrics[model_name]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{m['accuracy']:.3f}")
    c2.metric("Precision", f"{m['precision']:.3f}")
    c3.metric("Recall", f"{m['recall']:.3f}")
    c4.metric("F1", f"{m['f1']:.3f}")
    c5.metric("ROC-AUC", f"{m['roc_auc']:.3f}")

    st.subheader("Confusion Matrix")
    cm = np.array(m["confusion_matrix"])
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("Feature Importance")
    imp = get_feature_importance(models[model_name]).head(20)
    if not imp.empty:
        fig = px.bar(imp.sort_values("importance"), x="importance", y="feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance is not available for this model.")


def page_business_case(scored_df):
    st.title("Business Case Calculator")
    st.write("Estimate potential economic value of investigating high-risk applications.")

    c1, c2, c3 = st.columns(3)
    with c1:
        investigation_rate = st.slider("Investigate top-risk cases (%)", 1, 30, 10)
    with c2:
        detection_success_rate = st.slider("Detection success rate (%)", 5, 90, 45)
    with c3:
        investigation_cost_per_case = st.number_input("Investigation cost per case (€)", 10, 2000, 150)

    threshold = np.percentile(scored_df["fraud_risk_score"], 100 - investigation_rate)
    selected = scored_df[scored_df["fraud_risk_score"] >= threshold]

    potential_loss = selected["claim_amount"].sum()
    prevented_loss = potential_loss * detection_success_rate / 100
    investigation_cost = len(selected) * investigation_cost_per_case
    net_impact = prevented_loss - investigation_cost

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cases Investigated", f"{len(selected):,}")
    k2.metric("Potential Loss Exposure", money(potential_loss))
    k3.metric("Estimated Prevented Loss", money(prevented_loss))
    k4.metric("Net Business Impact", money(net_impact))

    st.subheader("High-Risk Investigation List")
    display_cols = [
        "application_id",
        "region",
        "vehicle_type",
        "contract_type",
        "premium_amount",
        "claim_amount",
        "fraud_risk_score",
        "risk_level",
    ]
    st.dataframe(
        selected[display_cols].sort_values("fraud_risk_score", ascending=False).head(100),
        use_container_width=True,
    )

    st.info(
        "Business interpretation: This calculation helps translate model output into management-oriented value. "
        "It shows whether reviewing a percentage of high-risk cases can create a positive net impact."
    )


def page_data_browser(df, scored_df):
    st.title("Data Browser")
    st.write("Review raw and scored insurance application data.")

    st.subheader("Scored Data")
    st.dataframe(scored_df.head(500), use_container_width=True)

    st.download_button(
        label="Download scored data as CSV",
        data=scored_df.to_csv(index=False),
        file_name="scored_insurance_fraud_data.csv",
        mime="text/csv",
    )

    if st.button("Save scored predictions to MySQL"):
        try:
            save_predictions_to_mysql(scored_df)
            st.success("Predictions saved to MySQL table `fraud_predictions`.")
        except Exception as exc:
            st.error(f"Could not save predictions to MySQL: {exc}")


def main():
    source, page = sidebar()
    df = load_data(source)
    df = add_features(df)

    models, metrics, isolation, X_test, y_test = cached_train_models(df)

    selected_model_name = "Random Forest"
    selected_model = models[selected_model_name]
    scored_df = score_dataframe(df, selected_model)

    if page == "Executive Overview":
        page_overview(df, scored_df, metrics)
    elif page == "Fraud Pattern Analysis":
        page_patterns(scored_df)
    elif page == "Risk Scoring":
        page_risk_scoring(scored_df, selected_model)
    elif page == "Model Performance":
        page_model_performance(scored_df, metrics, models)
    elif page == "Business Case":
        page_business_case(scored_df)
    elif page == "Data Browser":
        page_data_browser(df, scored_df)


if __name__ == "__main__":
    main()