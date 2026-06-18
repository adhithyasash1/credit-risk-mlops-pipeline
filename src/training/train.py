"""
train.py — train + evaluate the credit-risk model, save artifact to GCS.

Run from the project root:  python3 -m src.training.train
"""
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from google.cloud import storage

from src.features.preprocess import (
    load_config, load_data_from_bq, split_xy, build_preprocessor,
)


def ks_statistic(y_true, y_score) -> float:
    # KS = max separation between cumulative good/bad distributions.
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def upload_to_gcs(local_path: str, gcs_uri: str) -> None:
    bucket_name, blob_path = gcs_uri[len("gs://"):].split("/", 1)
    storage.Client().bucket(bucket_name).blob(blob_path).upload_from_filename(local_path)
    print(f"Uploaded model -> {gcs_uri}")


def main():
    cfg = load_config()
    df = load_data_from_bq(cfg)
    X, y = split_xy(df, cfg["target_column"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Preprocessing + classifier as ONE deployable object.
    model = Pipeline([
        ("prep", build_preprocessor(X)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(X_train, y_train)

    scores = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, scores)
    ks = ks_statistic(y_test, scores)
    print(f"\nTest AUC: {auc:.3f}")
    print(f"Test KS : {ks:.3f}")
    print("\nClassification report (threshold 0.5):")
    print(classification_report(y_test, (scores >= 0.5).astype(int)))

    joblib.dump(model, "model.joblib")
    upload_to_gcs("model.joblib", f"{cfg['bucket']}/models/credit-risk/model.joblib")


if __name__ == "__main__":
    main()
