"""
app.py — FastAPI serving for the credit-risk model, with Prometheus metrics.
"""
import os
from contextlib import asynccontextmanager

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

MODEL_GCS_URI = os.environ.get(
    "MODEL_GCS_URI",
    "gs://credit-risk-mlops-0812-bucket/models/credit-risk/champion/model.joblib",
)
FEAST_REPO = os.environ.get("FEAST_REPO", "feature_repo")

FEATURES = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_commitment", "personal_status",
    "other_parties", "residence_since", "property_magnitude", "age",
    "other_payment_plans", "housing", "existing_credits", "job",
    "num_dependents", "own_telephone", "foreign_worker",
]

# --- Custom ML metrics (scraped by Prometheus) ---
PRED_COUNTER = Counter(
    "credit_predictions_total", "Total predictions", ["decision"]
)
PROB_HIST = Histogram(
    "credit_probability_default", "Predicted probability of default",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

state = {}


class Application(BaseModel):
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
    bucket, blob = gcs_uri[len("gs://"):].split("/", 1)
    storage.Client().bucket(bucket).blob(blob).download_to_filename("/tmp/model.joblib")
    return joblib.load("/tmp/model.joblib")


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = _load_model(MODEL_GCS_URI)
    state["model"] = model
    state["prep"] = model.named_steps["prep"]
    state["explainer"] = shap.TreeExplainer(model.named_steps["clf"])
    state["feature_names_out"] = state["prep"].get_feature_names_out()
    state["store"] = FeatureStore(repo_path=FEAST_REPO)
    yield
    state.clear()


app = FastAPI(title="Credit Risk Scoring API", lifespan=lifespan)

# Exposes /metrics with HTTP request count/latency/size, auto-instrumented.
Instrumentator().instrument(app).expose(app)


def _score(df: pd.DataFrame) -> dict:
    prob = float(state["model"].predict_proba(df)[:, 1][0])
    decision = "reject" if prob >= 0.5 else "approve"
    PRED_COUNTER.labels(decision=decision).inc()   # business metric
    PROB_HIST.observe(prob)                          # distribution metric
    return {"probability_default": round(prob, 4), "decision": decision}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict")
def predict(application: Application):
    return _score(pd.DataFrame([application.model_dump()]))


@app.get("/predict/{application_id}")
def predict_by_id(application_id: int):
    refs = [f"credit_features:{f}" for f in FEATURES]
    feats = state["store"].get_online_features(
        features=refs, entity_rows=[{"application_id": application_id}]
    ).to_dict()
    row = {f: feats[f][0] for f in FEATURES}
    if any(v is None for v in row.values()):
        raise HTTPException(404, f"No features for application_id {application_id}")
    out = _score(pd.DataFrame([row]))
    out["application_id"] = application_id
    return out


@app.post("/explain")
def explain(application: Application):
    df = pd.DataFrame([application.model_dump()])
    Xt = state["prep"].transform(df)
    if scipy.sparse.issparse(Xt):
        Xt = Xt.toarray()
    shap_row = state["explainer"].shap_values(Xt)[0]
    ranked = sorted(
        zip(state["feature_names_out"], shap_row),
        key=lambda kv: abs(kv[1]), reverse=True,
    )[:8]
    return {
        "top_contributions": [
            {"feature": f, "shap_value": round(float(v), 4)} for f, v in ranked
        ]
    }
