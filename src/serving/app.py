"""
app.py — FastAPI serving for the credit-risk model.

Endpoints:
  GET  /health                  liveness + model status
  POST /predict                 score raw applicant JSON
  GET  /predict/{application_id} score by ID (features from Feast online store)
  POST /explain                 SHAP top feature contributions for a decision
"""
import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
import scipy.sparse
import shap
from feast import FeatureStore
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_URI = os.environ.get("MODEL_URI", "models:/credit-risk-model@champion")
TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", "https://mlflow-server-t2o3hekgva-uc.a.run.app"
)
FEAST_REPO = os.environ.get("FEAST_REPO", "feature_repo")

FEATURES = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_commitment", "personal_status",
    "other_parties", "residence_since", "property_magnitude", "age",
    "other_payment_plans", "housing", "existing_credits", "job",
    "num_dependents", "own_telephone", "foreign_worker",
]

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(TRACKING_URI)
    model = mlflow.sklearn.load_model(MODEL_URI)
    state["model"] = model
    state["prep"] = model.named_steps["prep"]
    state["explainer"] = shap.TreeExplainer(model.named_steps["clf"])
    state["feature_names_out"] = state["prep"].get_feature_names_out()
    state["store"] = FeatureStore(repo_path=FEAST_REPO)
    yield
    state.clear()


app = FastAPI(title="Credit Risk Scoring API", lifespan=lifespan)


def _score(df: pd.DataFrame) -> dict:
    prob = float(state["model"].predict_proba(df)[:, 1][0])
    return {
        "probability_default": round(prob, 4),
        "decision": "reject" if prob >= 0.5 else "approve",
    }


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
