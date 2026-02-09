import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


RNG = np.random.default_rng(42)
DISEASES = [
    "Influenza",
    "Common Cold",
    "COVID-19",
    "Migraine",
    "Gastroenteritis",
    "Hypertension",
    "Diabetes",
    "Asthma",
]

SYMPTOMS = [
    "fever",
    "cough",
    "headache",
    "fatigue",
    "sore_throat",
    "shortness_of_breath",
    "nausea",
    "vomiting",
    "diarrhea",
    "chest_pain",
    "dizziness",
    "body_ache",
    "runny_nose",
    "loss_of_taste",
]

DISEASE_PROFILES = {
    "Influenza": {"fever": 0.85, "cough": 0.7, "headache": 0.65, "fatigue": 0.75, "body_ache": 0.8, "sore_throat": 0.55, "runny_nose": 0.45},
    "Common Cold": {"cough": 0.55, "sore_throat": 0.7, "runny_nose": 0.8, "headache": 0.35, "fever": 0.25, "fatigue": 0.3},
    "COVID-19": {"fever": 0.75, "cough": 0.75, "fatigue": 0.8, "shortness_of_breath": 0.55, "loss_of_taste": 0.65, "headache": 0.45},
    "Migraine": {"headache": 0.92, "nausea": 0.45, "vomiting": 0.2, "dizziness": 0.4, "fatigue": 0.35},
    "Gastroenteritis": {"nausea": 0.8, "vomiting": 0.72, "diarrhea": 0.88, "fever": 0.35, "fatigue": 0.45},
    "Hypertension": {"dizziness": 0.5, "headache": 0.35, "chest_pain": 0.2, "fatigue": 0.25},
    "Diabetes": {"fatigue": 0.55, "dizziness": 0.35, "nausea": 0.2, "headache": 0.25},
    "Asthma": {"shortness_of_breath": 0.9, "cough": 0.5, "chest_pain": 0.3, "fatigue": 0.35},
}


def generate_synthetic_dataset(num_samples: int = 5000) -> pd.DataFrame:
    rows = []
    for _ in range(num_samples):
        disease = RNG.choice(DISEASES)
        age = int(np.clip(RNG.normal(43, 17), 5, 90))
        gender = RNG.choice(["male", "female"])
        smoking = int(RNG.random() < 0.28)
        alcohol = int(RNG.random() < 0.42)

        duration_base = {
            "Influenza": (2, 8),
            "Common Cold": (2, 7),
            "COVID-19": (3, 12),
            "Migraine": (1, 4),
            "Gastroenteritis": (1, 5),
            "Hypertension": (5, 30),
            "Diabetes": (7, 40),
            "Asthma": (2, 14),
        }
        duration = int(RNG.integers(*duration_base[disease]))

        symptom_flags = {}
        profile = DISEASE_PROFILES[disease]
        for symptom in SYMPTOMS:
            baseline = 0.06
            prob = profile.get(symptom, baseline)
            if smoking and symptom in {"cough", "shortness_of_breath", "chest_pain"}:
                prob = min(prob + 0.08, 0.97)
            if age > 55 and symptom in {"fatigue", "dizziness", "chest_pain"}:
                prob = min(prob + 0.06, 0.97)
            symptom_flags[symptom] = int(RNG.random() < prob)

        row = {
            "age": age,
            "gender": gender,
            **{f"symptom_{s}": v for s, v in symptom_flags.items()},
            "symptom_duration": duration,
            "smoking": smoking,
            "alcohol": alcohol,
            "disease": disease,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Add small amount of duplicates/noise to mimic realistic data issues.
    duplicate_rows = df.sample(frac=0.01, random_state=42)
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    return df


def clean_encode_select(df: pd.DataFrame, top_k_features: int = 18) -> tuple[pd.DataFrame, list[str]]:
    df = df.drop_duplicates().copy()
    df["gender"] = df["gender"].str.lower().str.strip().replace({"f": "female", "m": "male"})
    df["gender"] = df["gender"].fillna("female")

    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(df["age"].median())
    df["age"] = df["age"].clip(1, 100)

    binary_cols = [c for c in df.columns if c.startswith("symptom_")] + ["smoking", "alcohol"]
    for col in binary_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(0, 1).astype(int)

    df["symptom_duration"] = pd.to_numeric(df["symptom_duration"], errors="coerce").fillna(df["symptom_duration"].median())
    df["symptom_duration"] = df["symptom_duration"].clip(1, 60)

    encoded = pd.get_dummies(df.drop(columns=["disease"]), columns=["gender"], drop_first=False)
    mi_scores = mutual_info_classif(encoded, df["disease"], random_state=42, discrete_features="auto")
    feature_scores = pd.Series(mi_scores, index=encoded.columns).sort_values(ascending=False)

    protected = ["age", "symptom_duration", "smoking", "alcohol", "gender_female", "gender_male"]
    selected = list(feature_scores.head(top_k_features).index)
    selected = list(dict.fromkeys(selected + [p for p in protected if p in encoded.columns]))

    symptom_features = [c for c in encoded.columns if c.startswith("symptom_")]
    selected += [f for f in symptom_features if f in feature_scores.head(top_k_features + 5).index and f not in selected]

    final_columns = [
        "age",
        "gender",
        *sorted([c for c in df.columns if c.startswith("symptom_")]),
        "symptom_duration",
        "smoking",
        "alcohol",
        "disease",
    ]

    return df[final_columns], selected


def main() -> None:
    data_dir = Path(__file__).resolve().parent
    dataset_path = data_dir / "dataset.csv"
    meta_path = data_dir / "selected_features.json"

    raw_df = generate_synthetic_dataset()
    final_df, selected_features = clean_encode_select(raw_df)
    final_df.to_csv(dataset_path, index=False)

    meta = {"selected_encoded_features": selected_features, "num_rows": int(final_df.shape[0]), "num_columns": int(final_df.shape[1])}
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Saved cleaned dataset to {dataset_path}")
    print(f"Saved feature selection metadata to {meta_path}")


if __name__ == "__main__":
    main()
