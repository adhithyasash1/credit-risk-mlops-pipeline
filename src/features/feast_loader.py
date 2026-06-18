"""
feast_loader.py — build the training set via Feast get_historical_features().

Training reads features through the store (not raw BigQuery), so the SAME
feature definitions feed both training and serving — no skew.
"""
from feast import FeatureStore
from google.cloud import bigquery


def load_training_data(cfg, repo_path="feature_repo"):
    store = FeatureStore(repo_path=repo_path)
    client = bigquery.Client(project=cfg["project_id"])
    table = f"{cfg['project_id']}.{cfg['bq_dataset']}.applications_fs"

    # Spine: entity + timestamp + label.
    entity_df = client.query(
        f"SELECT application_id, event_timestamp, target FROM `{table}`"
    ).to_dataframe()

    fv = store.get_feature_view("credit_features")
    feature_names = [f.name for f in fv.features]
    refs = [f"credit_features:{n}" for n in feature_names]

    training_df = store.get_historical_features(
        entity_df=entity_df, features=refs
    ).to_df()

    X = training_df[feature_names]
    y = training_df["target"]
    return X, y
