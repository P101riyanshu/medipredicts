import pickle
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
        value = value.strip().lower()
        if value not in {"male", "female"}:
            raise ValueError("gender must be 'male' or 'female'")
        return value


app = FastAPI(
    title="Clinical Decision Support API",
    version="2.0.0",
    description="Disease prediction API with explainable outputs for top-3 predictions.",
)
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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": "Invalid request payload", "details": exc.errors()})


@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": "Internal server error", "detail": str(exc)})


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_BUNDLE["best_model_name"]}


@app.get("/metadata")
def metadata():
    return {
        "symptoms": MODEL_BUNDLE["symptoms"],
        "diseases": MODEL_BUNDLE["classes"],
        "feature_columns": MODEL_BUNDLE["feature_columns"],
        "best_model": MODEL_BUNDLE["best_model_name"],
    }


@app.get("/metrics")
def metrics():
    return {"best_model": MODEL_BUNDLE["best_model_name"], "results": MODEL_BUNDLE.get("metrics", {})}


@app.get("/importance")
def importance(limit: int = 10):
    symptom_importance = MODEL_BUNDLE.get("symptom_importance", {})
    ranked = sorted(symptom_importance.items(), key=lambda item: item[1], reverse=True)
    ranked = ranked[: max(1, min(limit, len(ranked)))]
    return {
        "top_symptoms": [
            {"symptom": symptom, "importance": round(float(score), 6)} for symptom, score in ranked
        ]
    }


@app.get("/sample-payload")
def sample_payload():
    return {
        "age": 28,
        "gender": "female",
        "symptoms": ["fever", "cough", "headache"],
        "duration_days": 2,
        "lifestyle": {"smoking": False, "alcohol": False},
    }


@app.post("/predict")
def predict(payload: PredictRequest):
    model = MODEL_BUNDLE["model"]
    symptoms_available = set(MODEL_BUNDLE["symptoms"])
    unknown = [s for s in payload.symptoms if s not in symptoms_available]
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
        row[f"symptom_{symptom}"] = int(symptom in payload.symptoms)

    frame = pd.DataFrame([row])
    frame = frame.reindex(columns=MODEL_BUNDLE["feature_columns"], fill_value=0)

    probabilities = model.predict_proba(frame)[0]
    encoded_labels = model.classes_
    label_encoder = MODEL_BUNDLE.get("label_encoder")
    labels = label_encoder.inverse_transform(encoded_labels) if label_encoder is not None else encoded_labels
    ranking = sorted(zip(labels, probabilities), key=lambda x: x[1], reverse=True)[:3]

    global_imp = MODEL_BUNDLE.get("symptom_importance", {})
    present_symptoms = [s for s in payload.symptoms if s in global_imp]
    important_symptoms = sorted(present_symptoms, key=lambda s: global_imp.get(s, 0), reverse=True)[:5]

    if len(important_symptoms) < 5:
        fill = sorted(global_imp.keys(), key=lambda s: global_imp[s], reverse=True)
        for item in fill:
            if item not in important_symptoms:
                important_symptoms.append(item)
            if len(important_symptoms) == 5:
                break

    return {
        "predictions": [{"disease": disease, "confidence": round(float(score), 4)} for disease, score in ranking],
        "important_symptoms": important_symptoms,
        "model_used": MODEL_BUNDLE["best_model_name"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
