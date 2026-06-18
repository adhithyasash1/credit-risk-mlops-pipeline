"""Shared configuration helpers for the credit-risk MLOps pipeline."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config/config.yaml")
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_DECISION_THRESHOLD = 0.5

FEATURE_NAMES = (
    "checking_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment",
    "installment_commitment",
    "personal_status",
    "other_parties",
    "residence_since",
    "property_magnitude",
    "age",
    "other_payment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "own_telephone",
    "foreign_worker",
)

REQUIRED_CONFIG_KEYS = {
    "project_id",
    "region",
    "bucket",
    "bq_dataset",
    "bq_table",
    "raw_data_path",
    "target_column",
    "mlflow_tracking_uri",
    "mlflow_experiment",
    "model_name",
}

ENV_OVERRIDES = {
    "project_id": "PROJECT_ID",
    "region": "REGION",
    "bucket": "BUCKET",
    "bq_dataset": "BQ_DATASET",
    "bq_table": "BQ_TABLE",
    "raw_data_path": "RAW_DATA_PATH",
    "target_column": "TARGET_COLUMN",
    "mlflow_tracking_uri": "MLFLOW_TRACKING_URI",
    "mlflow_experiment": "MLFLOW_EXPERIMENT",
    "model_name": "MODEL_NAME",
}


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load YAML config and apply documented environment overrides."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file) or {}

    if not isinstance(cfg, dict):
        raise ValueError(f"Expected mapping in {config_path}, got {type(cfg).__name__}")

    config = dict(cfg)
    for key, env_name in ENV_OVERRIDES.items():
        value = os.getenv(env_name)
        if value:
            config[key] = value

    missing = sorted(key for key in REQUIRED_CONFIG_KEYS if not config.get(key))
    if missing:
        raise ValueError(f"Missing required config keys in {config_path}: {', '.join(missing)}")

    return config


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Return (bucket, blob) for a gs:// URI with a non-empty object path."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected a gs:// URI, got {uri!r}")

    bucket_and_blob = uri[len("gs://") :]
    if "/" not in bucket_and_blob:
        raise ValueError(f"Expected a GCS object path in {uri!r}")

    bucket, blob = bucket_and_blob.split("/", 1)
    if not bucket or not blob:
        raise ValueError(f"Expected a non-empty bucket and object path in {uri!r}")

    return bucket, blob


def get_decision_threshold(default: float = DEFAULT_DECISION_THRESHOLD) -> float:
    """Read and validate the serving decision threshold from the environment."""
    raw_value = os.getenv("DECISION_THRESHOLD")
    if raw_value is None:
        return default

    try:
        threshold = float(raw_value)
    except ValueError as exc:
        raise ValueError("DECISION_THRESHOLD must be a number") from exc

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("DECISION_THRESHOLD must be between 0.0 and 1.0")

    return threshold
