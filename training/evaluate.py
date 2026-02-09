import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    model_path = root / "backend" / "model.pkl"
    data_path = root / "data" / "dataset.csv"

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    df = pd.read_csv(data_path)
    X = df.drop(columns=["disease"])
    y = df["disease"]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred, labels=sorted(y.unique().tolist())).tolist()

    output = {
        "best_model": bundle["best_model_name"],
        "weighted_avg": report.get("weighted avg", {}),
        "accuracy": report.get("accuracy", 0),
        "confusion_matrix": matrix,
        "labels": sorted(y.unique().tolist()),
    }

    out_path = Path(__file__).resolve().parent / "evaluation_report.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved evaluation report to {out_path}")


if __name__ == "__main__":
    main()
