import pickle
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator


class Lifestyle(BaseModel):
    smoking: bool = False
    alcohol: bool = False


class PredictRequest(BaseModel):
    age: int = Field(..., ge=1, le=100)
    gender: str
    symptoms: List[str]
    duration_days: int = Field(..., ge=1, le=60)
    lifestyle: Optional[Lifestyle] = Lifestyle()

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"male", "female"}:
            raise ValueError("gender must be 'male' or 'female'")
        return normalized


app = FastAPI(title="Clinical Decision Support API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_BUNDLE = None


def load_model() -> None:
    global MODEL_BUNDLE
    model_path = Path(__file__).resolve().parent / "model.pkl"
    if not model_path.exists():
        raise RuntimeError("model.pkl not found. Run training/train.py first.")
    with open(model_path, "rb") as f:
        MODEL_BUNDLE = pickle.load(f)


@app.on_event("startup")
def startup_event() -> None:
    load_model()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "best_model": MODEL_BUNDLE["best_model_name"],
        "num_diseases": len(MODEL_BUNDLE["classes"]),
        "num_symptoms": len(MODEL_BUNDLE["symptoms"]),
    }


@app.get("/metadata")
def metadata() -> dict:
    return {
        "symptoms": MODEL_BUNDLE["symptoms"],
        "diseases": MODEL_BUNDLE["classes"],
    }


@app.get("/model-info")
def model_info() -> dict:
    symptom_rank = sorted(
        MODEL_BUNDLE.get("symptom_importance", {}).items(), key=lambda item: item[1], reverse=True
    )[:10]
    return {
        "best_model": MODEL_BUNDLE["best_model_name"],
        "metrics": MODEL_BUNDLE.get("metrics", {}),
        "top_global_symptoms": [
            {"symptom": symptom, "importance": round(float(score), 5)}
            for symptom, score in symptom_rank
        ],
    }


@app.post("/predict")
def predict(payload: PredictRequest) -> dict:
    try:
        model = MODEL_BUNDLE["model"]
        symptoms_available = set(MODEL_BUNDLE["symptoms"])

        normalized_symptoms = list(dict.fromkeys(s.strip().lower() for s in payload.symptoms if s.strip()))
        if not normalized_symptoms:
            raise HTTPException(status_code=400, detail="At least one symptom is required")

        unknown = [s for s in normalized_symptoms if s not in symptoms_available]
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unknown symptoms: {unknown}")

        row = {
            "age": payload.age,
            "gender": payload.gender,
            "symptom_duration": payload.duration_days,
            "smoking": int(payload.lifestyle.smoking if payload.lifestyle else False),
            "alcohol": int(payload.lifestyle.alcohol if payload.lifestyle else False),
        }

        for symptom in symptoms_available:
            row[f"symptom_{symptom}"] = int(symptom in normalized_symptoms)

        frame = pd.DataFrame([row]).reindex(columns=MODEL_BUNDLE["feature_columns"], fill_value=0)
        probabilities = model.predict_proba(frame)[0]
        labels = model.classes_
        ranking = sorted(zip(labels, probabilities), key=lambda x: x[1], reverse=True)[:3]

        global_imp = MODEL_BUNDLE.get("symptom_importance", {})
        important_symptoms = sorted(normalized_symptoms, key=lambda s: global_imp.get(s, 0), reverse=True)[:3]

        if len(important_symptoms) < 3:
            for symptom, _ in sorted(global_imp.items(), key=lambda x: x[1], reverse=True):
                if symptom not in important_symptoms:
                    important_symptoms.append(symptom)
                if len(important_symptoms) == 3:
                    break

        return {
            "predictions": [
                {"disease": disease, "confidence": round(float(score), 4)}
                for disease, score in ranking
            ],
            "important_symptoms": important_symptoms,
            "input_summary": {
                "age": payload.age,
                "gender": payload.gender,
                "duration_days": payload.duration_days,
                "symptom_count": len(normalized_symptoms),
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
