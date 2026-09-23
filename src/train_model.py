from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "credit_risk_dataset.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

TARGET = "loan_status"

def load_data():
    df = pd.read_csv(DATA)
    # Remove extremos inconsistentes sem apagar o padrão real do dataset
    df = df[(df["person_age"] <= 100) & (df["person_emp_length"].fillna(0) <= 80)]
    return df

def build_preprocessor(X):
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols)
    ])

def evaluate(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    else:
        proba = pred
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1_score": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, output_dict=True)
    }, model

def main():
    df = load_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X_train)
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced", n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=42)
    }

    results = []
    fitted = {}
    for name, estimator in candidates.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])
        metrics, model = evaluate(name, pipe, X_train, X_test, y_train, y_test)
        results.append(metrics)
        fitted[name] = model

    best = max(results, key=lambda r: r["roc_auc"])
    best_model = fitted[best["model"]]
    joblib.dump(best_model, MODELS / "credit_scoring_model.joblib")

    summary = pd.DataFrame([{k:v for k,v in r.items() if k not in ["classification_report", "confusion_matrix"]} for r in results])
    summary = summary.sort_values("roc_auc", ascending=False)
    summary.to_csv(REPORTS / "metrics_summary.csv", index=False)

    with open(REPORTS / "metrics_full.json", "w", encoding="utf-8") as f:
        json.dump({"best_model": best["model"], "results": results}, f, indent=2)

    print(summary.to_string(index=False))
    print(f"\nBest model: {best['model']}")
    print("Saved to models/credit_scoring_model.joblib")

if __name__ == "__main__":
    main()
