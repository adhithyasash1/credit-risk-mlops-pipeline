from pathlib import Path

import pytest

from src.config import get_decision_threshold, load_config, parse_gcs_uri


def test_parse_gcs_uri_splits_bucket_and_blob():
    assert parse_gcs_uri("gs://bucket/path/to/model.joblib") == (
        "bucket",
        "path/to/model.joblib",
    )


@pytest.mark.parametrize("uri", ["bucket/path", "gs://bucket", "gs:///blob", "gs://bucket/"])
def test_parse_gcs_uri_rejects_invalid_values(uri):
    with pytest.raises(ValueError):
        parse_gcs_uri(uri)


def test_load_config_applies_environment_overrides(tmp_path, monkeypatch):
    config_path = Path(tmp_path) / "config.yaml"
    config_path.write_text(
        """
project_id: original-project
region: us-central1
bucket: gs://bucket
bq_dataset: credit_risk
bq_table: applications
raw_data_path: gs://bucket/data/raw/german_credit.csv
target_column: target
mlflow_tracking_uri: https://mlflow.example
mlflow_experiment: credit-risk
model_name: credit-risk-model
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("PROJECT_ID", "override-project")

    cfg = load_config(config_path)

    assert cfg["project_id"] == "override-project"
    assert cfg["bq_table"] == "applications"


def test_decision_threshold_defaults_and_validates(monkeypatch):
    monkeypatch.delenv("DECISION_THRESHOLD", raising=False)
    assert get_decision_threshold() == 0.5

    monkeypatch.setenv("DECISION_THRESHOLD", "0.37")
    assert get_decision_threshold() == 0.37

    monkeypatch.setenv("DECISION_THRESHOLD", "2")
    with pytest.raises(ValueError):
        get_decision_threshold()
