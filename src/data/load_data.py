"""
load_data.py — Phase 1 data ingestion.

Pulls the German Credit dataset (OpenML 'credit-g'), lands the raw CSV in GCS,
and loads the structured table into BigQuery. Run once to populate the project's
two data homes.
"""
import yaml
import pandas as pd
from sklearn.datasets import fetch_openml
from google.cloud import storage, bigquery


def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def fetch_data() -> pd.DataFrame:
    # as_frame=True -> a pandas DataFrame with named columns and the target.
    ds = fetch_openml(name="credit-g", version=1, as_frame=True)
    df = ds.frame.copy()

    # The target column 'class' is 'good'/'bad'. In credit risk we model the
    # BAD outcome (a default) as the positive class = 1.
    df["target"] = (df["class"] == "bad").astype(int)
    df = df.drop(columns=["class"])

    # BigQuery can't ingest pandas 'category' dtype directly -> cast to string.
    for col in df.select_dtypes(include=["category"]).columns:
        df[col] = df[col].astype(str)

    return df


def upload_csv_to_gcs(df: pd.DataFrame, gcs_uri: str) -> None:
    assert gcs_uri.startswith("gs://"), gcs_uri
    bucket_name, blob_path = gcs_uri[len("gs://"):].split("/", 1)
    blob = storage.Client().bucket(bucket_name).blob(blob_path)
    blob.upload_from_string(df.to_csv(index=False), content_type="text/csv")
    print(f"Uploaded raw CSV -> {gcs_uri}  ({len(df)} rows)")


def load_to_bigquery(df: pd.DataFrame, project: str, dataset: str, table: str) -> None:
    client = bigquery.Client(project=project)
    table_id = f"{project}.{dataset}.{table}"
    # WRITE_TRUNCATE = replace the table each run (idempotent re-ingestion).
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()  # block until the load finishes
    print(f"Loaded {len(df)} rows -> BigQuery {table_id}")


def main():
    cfg = load_config()
    df = fetch_data()
    print("Shape:", df.shape)
    print("Target distribution (1=bad/default):")
    print(df["target"].value_counts())
    upload_csv_to_gcs(df, cfg["raw_data_path"])
    load_to_bigquery(df, cfg["project_id"], cfg["bq_dataset"], cfg["bq_table"])


if __name__ == "__main__":
    main()
