"""Optuna HPO for an XGBoost credit-risk model, fed by Feast.

Run from project root:  python3 -m src.training.tune --trials 30
"""
import argparse
import os

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier

from src.config import DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE
from src.features.preprocess import load_config, build_preprocessor
from src.features.feast_loader import load_training_data


def _positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def ks_statistic(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--model-jobs", type=int, default=_positive_int_env("XGBOOST_N_JOBS", 1))
    parser.add_argument("--cv-jobs", type=int, default=_positive_int_env("CV_N_JOBS", 1))
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    args = parser.parse_args()

    if args.trials < 1:
        raise ValueError("--trials must be >= 1")
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2")
    if args.model_jobs < 1 or args.cv_jobs < 1:
        raise ValueError("--model-jobs and --cv-jobs must be >= 1")

    cfg = load_config()
    mlflow.set_tracking_uri(cfg["mlflow_tracking_uri"])
    mlflow.set_experiment(cfg["mlflow_experiment"])

    # Features now come THROUGH Feast, not raw BigQuery.
    X, y = load_training_data(cfg)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=DEFAULT_TEST_SIZE, stratify=y, random_state=args.random_state
    )

    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    if pos == 0:
        raise ValueError("Training split has no positive examples; cannot tune class weight")
    spw = neg / pos

    def make_model(params):
        return Pipeline([
            ("prep", build_preprocessor(X)),
            ("clf", XGBClassifier(
                **params, scale_pos_weight=spw, eval_metric="auc",
                random_state=args.random_state, n_jobs=args.model_jobs)),
        ])

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            auc = cross_val_score(make_model(params), X_train, y_train,
                                  cv=args.cv_folds, scoring="roc_auc",
                                  n_jobs=args.cv_jobs).mean()
            mlflow.log_metric("cv_auc", auc)
            return auc

    with mlflow.start_run(run_name="optuna-xgb-feast") as run:
        mlflow.log_params({
            "cv_folds": args.cv_folds,
            "model_jobs": args.model_jobs,
            "cv_jobs": args.cv_jobs,
            "random_state": args.random_state,
            "test_size": DEFAULT_TEST_SIZE,
        })
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.trials)

        best = study.best_params
        mlflow.log_params({f"best_{k}": v for k, v in best.items()})
        mlflow.log_metric("best_cv_auc", study.best_value)
        print("Best CV AUC:", round(study.best_value, 4), "| params:", best)

        model = make_model(best)
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, scores)
        mlflow.log_metrics({
            "test_auc": test_auc,
            "test_ks": ks_statistic(y_test, scores),
        })
        print(f"Test AUC: {test_auc:.3f}")

        signature = infer_signature(X_test, model.predict(X_test))
        mlflow.sklearn.log_model(
            sk_model=model, artifact_path="model",
            signature=signature, input_example=X_test.iloc[:5],
            registered_model_name=cfg["model_name"],
        )

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{cfg['model_name']}'")
    run_versions = [v for v in versions if v.run_id == run.info.run_id]
    if not run_versions:
        raise RuntimeError(f"No registered model version found for run {run.info.run_id}")
    champion_version = max(run_versions, key=lambda v: int(v.version)).version
    client.set_registered_model_alias(cfg["model_name"], "champion", champion_version)
    print(f"Set alias 'champion' -> version {champion_version}")


if __name__ == "__main__":
    main()
