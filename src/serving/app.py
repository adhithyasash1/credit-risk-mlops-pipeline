"""FastAPI serving for the credit-risk model, with Prometheus metrics."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import joblib
import pandas as pd
import scipy.sparse
import shap
from feast import FeatureStore
from fastapi import FastAPI, HTTPException
from google.cloud import storage
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from src.config import FEATURE_NAMES, get_decision_threshold, parse_gcs_uri

MODEL_GCS_URI = os.environ.get(
    "MODEL_GCS_URI",
    "gs://credit-risk-mlops-0812-bucket/models/credit-risk/champion/model.joblib",
)
FEAST_REPO = os.environ.get("FEAST_REPO", "feature_repo")

# --- Custom ML metrics (scraped by Prometheus) ---
PRED_COUNTER = Counter(
    "credit_predictions_total", "Total predictions", ["decision"]
)
PROB_HIST = Histogram(
    "credit_probability_default", "Predicted probability of default",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)


@dataclass
class ModelAssets:
    model: Any
    prep: Any
    explainer: Any
    feature_names_out: list[str]
    store: FeatureStore


@dataclass
class RuntimeState:
    assets: ModelAssets | None = None
    decision_threshold: float = 0.5


state = RuntimeState()


class Application(BaseModel):
    class Config:
        extra = "forbid"

    checking_status: str
    duration: float
    credit_history: str
    purpose: str
    credit_amount: float
    savings_status: str
    employment: str
    installment_commitment: float
    personal_status: str
    other_parties: str
    residence_since: float
    property_magnitude: str
    age: float
    other_payment_plans: str
    housing: str
    existing_credits: float
    job: str
    num_dependents: float
    own_telephone: str
    foreign_worker: str


def _load_model(gcs_uri: str):
    bucket, blob = parse_gcs_uri(gcs_uri)
    with NamedTemporaryFile(suffix=".joblib", delete=False) as file:
        model_path = Path(file.name)

    try:
        storage.Client().bucket(bucket).blob(blob).download_to_filename(
            str(model_path), timeout=60
        )
        return joblib.load(model_path)
    finally:
        model_path.unlink(missing_ok=True)


def _build_assets() -> ModelAssets:
    model = _load_model(MODEL_GCS_URI)
    missing_steps = {"prep", "clf"} - set(model.named_steps)
    if missing_steps:
        raise RuntimeError(f"Model pipeline is missing required steps: {missing_steps}")

    prep = model.named_steps["prep"]
    return ModelAssets(
        model=model,
        prep=prep,
        explainer=shap.TreeExplainer(model.named_steps["clf"]),
        feature_names_out=list(prep.get_feature_names_out()),
        store=FeatureStore(repo_path=FEAST_REPO),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.decision_threshold = get_decision_threshold()
    state.assets = _build_assets()
    yield
    state.assets = None


app = FastAPI(title="Credit Risk Scoring API", lifespan=lifespan)

# Exposes /metrics with HTTP request count/latency/size, auto-instrumented.
Instrumentator().instrument(app).expose(app)


def _ensure_assets() -> ModelAssets:
    if state.assets is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return state.assets


def _model_dump(application: Application) -> dict[str, Any]:
    if hasattr(application, "model_dump"):
        return application.model_dump()
    return application.dict()


def _application_frame(application: Application) -> pd.DataFrame:
    data = _model_dump(application)
    return pd.DataFrame([{feature: data[feature] for feature in FEATURE_NAMES}])


def _score(df: pd.DataFrame) -> dict:
    assets = _ensure_assets()
    prob = float(assets.model.predict_proba(df)[:, 1][0])
    decision = "reject" if prob >= state.decision_threshold else "approve"
    PRED_COUNTER.labels(decision=decision).inc()   # business metric
    PROB_HIST.observe(prob)                          # distribution metric
    return {"probability_default": round(prob, 4), "decision": decision}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state.assets is not None}


@app.post("/predict")
def predict(application: Application):
    return _score(_application_frame(application))


@app.get("/predict/{application_id}")
def predict_by_id(application_id: int):
    assets = _ensure_assets()
    refs = [f"credit_features:{feature}" for feature in FEATURE_NAMES]
    feats = assets.store.get_online_features(
        features=refs, entity_rows=[{"application_id": application_id}]
    ).to_dict()
    row = {feature: feats.get(feature, [None])[0] for feature in FEATURE_NAMES}
    if any(v is None for v in row.values()):
        raise HTTPException(404, f"No features for application_id {application_id}")
    out = _score(pd.DataFrame([row]))
    out["application_id"] = application_id
    return out


@app.post("/explain")
def explain(application: Application):
    assets = _ensure_assets()
    df = _application_frame(application)
    Xt = assets.prep.transform(df)
    if scipy.sparse.issparse(Xt):
        Xt = Xt.toarray()
    shap_values = assets.explainer.shap_values(Xt)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_row = shap_values[0]
    ranked = sorted(
        zip(assets.feature_names_out, shap_row),
        key=lambda kv: abs(kv[1]), reverse=True,
    )[:8]
    return {
        "top_contributions": [
            {"feature": f, "shap_value": round(float(v), 4)} for f, v in ranked
        ]
    }
