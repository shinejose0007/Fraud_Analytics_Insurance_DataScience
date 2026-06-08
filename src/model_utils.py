import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .feature_engineering import add_features, CATEGORICAL_COLS, NUMERIC_COLS


def build_preprocessor():
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ]
    )
    return preprocessor


def train_models(df: pd.DataFrame):
    df = add_features(df)
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df["fraud_label"].astype(int)

    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    preprocessor = build_preprocessor()

    logistic_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    rf_model = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            random_state=42,
            class_weight="balanced_subsample",
        )),
    ])

    models = {
        "Logistic Regression": logistic_model,
        "Random Forest": rf_model,
    }

    trained = {}
    metrics = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        trained[name] = model

    # Isolation Forest trained only on numeric engineered features
    iso_features = df[NUMERIC_COLS].fillna(0)
    isolation = IsolationForest(contamination=0.08, random_state=42)
    isolation.fit(iso_features)

    return trained, metrics, isolation, X_test, y_test


def score_dataframe(df: pd.DataFrame, model):
    df = add_features(df)
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    probabilities = model.predict_proba(X)[:, 1]
    output = df.copy()
    output["fraud_probability"] = probabilities
    output["fraud_risk_score"] = np.round(probabilities * 100, 1)
    output["risk_level"] = pd.cut(
        output["fraud_risk_score"],
        bins=[-1, 30, 70, 100],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    ).astype(str)
    return output


def get_feature_importance(model):
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]

    try:
        cat_feature_names = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(CATEGORICAL_COLS)
        feature_names = NUMERIC_COLS + list(cat_feature_names)
    except Exception:
        feature_names = NUMERIC_COLS + CATEGORICAL_COLS

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    importance_df = pd.DataFrame({
        "feature": feature_names[:len(importances)],
        "importance": importances
    }).sort_values("importance", ascending=False)

    return importance_df