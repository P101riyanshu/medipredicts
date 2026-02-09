import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


def build_models():
    models = {
        "LogisticRegression": LogisticRegression(max_iter=400, n_jobs=None),
        "RandomForest": RandomForestClassifier(n_estimators=250, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, C=2.0, gamma="scale", random_state=42),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            objective="multi:softprob",
            n_estimators=250,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.85,
            eval_metric="mlogloss",
            random_state=42,
        )
    return models


def get_feature_importance(model, feature_names):
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.mean(np.abs(estimator.coef_), axis=0)
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)

    normalized = importances / (np.sum(importances) + 1e-9)
    return {name: float(score) for name, score in zip(feature_names, normalized)}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "dataset.csv"
    backend_dir = root / "backend"
    backend_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data_path)
    X = df.drop(columns=["disease"])
    y = df["disease"]

    categorical = ["gender"]
    numeric = [c for c in X.columns if c != "gender"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = build_models()
    results = {}
    best_model_name = None
    best_f1 = -1
    best_pipeline = None

    for name, estimator in models.items():
        pipeline = Pipeline([
            ("prep", preprocessor),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average="weighted", zero_division=0)
        results[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
        }
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_pipeline = pipeline

    transformed_feature_names = best_pipeline.named_steps["prep"].get_feature_names_out()
    importance = get_feature_importance(best_pipeline, transformed_feature_names)

    symptom_importance = {}
    for f_name, score in importance.items():
        if "symptom_" in f_name:
            raw_symptom = f_name.split("symptom_")[-1]
            symptom_importance[raw_symptom] = symptom_importance.get(raw_symptom, 0.0) + score

    model_bundle = {
        "model": best_pipeline,
        "best_model_name": best_model_name,
        "metrics": results,
        "classes": sorted(y.unique().tolist()),
        "symptoms": [c.replace("symptom_", "") for c in X.columns if c.startswith("symptom_")],
        "symptom_importance": symptom_importance,
        "feature_columns": X.columns.tolist(),
    }

    with open(backend_dir / "model.pkl", "wb") as f:
        pickle.dump(model_bundle, f)

    report_path = Path(__file__).resolve().parent / "model_metrics.json"
    report_path.write_text(json.dumps({"best_model": best_model_name, "results": results}, indent=2))

    print(f"Best model: {best_model_name}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
