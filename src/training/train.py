"""
train.py — train + evaluate the credit-risk model, log everything to MLflow.

Run from the project root:  python3 -m src.training.train
"""
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
)

from src.features.preprocess import (
    load_config, load_data_from_bq, split_xy, build_preprocessor,
)


def ks_statistic(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def main():
    cfg = load_config()

    # Point the MLflow client at our Cloud Run tracking server.
    mlflow.set_tracking_uri(cfg["mlflow_tracking_uri"])
    mlflow.set_experiment(cfg["mlflow_experiment"])

    df = load_data_from_bq(cfg)
    X, y = split_xy(df, cfg["target_column"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    params = {
        "model": "logistic_regression",
        "class_weight": "balanced",
        "max_iter": 1000,
        "test_size": 0.2,
        "random_state": 42,
    }

    with mlflow.start_run() as run:
        mlflow.log_params(params)

        model = Pipeline([
            ("prep", build_preprocessor(X)),
            ("clf", LogisticRegression(max_iter=params["max_iter"],
                                       class_weight=params["class_weight"])),
        ])
        model.fit(X_train, y_train)

        scores = model.predict_proba(X_test)[:, 1]
        preds = (scores >= 0.5).astype(int)

        metrics = {
            "auc": roc_auc_score(y_test, scores),
            "ks": ks_statistic(y_test, scores),
            "accuracy": accuracy_score(y_test, preds),
            "precision_macro": precision_score(y_test, preds, average="macro"),
            "recall_macro": recall_score(y_test, preds, average="macro"),
            "f1_macro": f1_score(y_test, preds, average="macro"),
        }
        mlflow.log_metrics(metrics)
        print("Metrics:", {k: round(v, 3) for k, v in metrics.items()})
        print(classification_report(y_test, preds))

        # Log the full Pipeline as a model + register a new version.
        signature = infer_signature(X_test, preds)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_test.iloc[:5],
            registered_model_name=cfg["model_name"],
        )
        print(f"Run logged: {run.info.run_id}")


if __name__ == "__main__":
    main()
