"""
build_feature_table.py — produce a Feast-ready BigQuery table.

Feast needs an entity key + event timestamp for point-in-time joins. The German
Credit data has neither, so we synthesize:
  - application_id : unique row id (the entity key)
  - event_timestamp: spread over the last 90 days (realistic point-in-time joins)

Run from project root:  python3 -m src.data.build_feature_table
"""
import numpy as np
from datetime import datetime, timedelta, timezone
from google.cloud import bigquery
from src.features.preprocess import load_config, load_data_from_bq


def main():
    cfg = load_config()
    df = load_data_from_bq(cfg).reset_index(drop=True)   # 20 features + target

    df.insert(0, "application_id", df.index.astype("int64"))

    rng = np.random.default_rng(42)
    base = datetime.now(timezone.utc).replace(microsecond=0)
    offsets = rng.integers(0, 90, size=len(df))
    df["event_timestamp"] = [base - timedelta(days=int(o)) for o in offsets]

    client = bigquery.Client(project=cfg["project_id"])
    table_id = f"{cfg['project_id']}.{cfg['bq_dataset']}.applications_fs"
    client.load_table_from_dataframe(
        df, table_id,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
    ).result()

    print(f"Wrote {len(df)} rows -> {table_id}")
    print(df[["application_id", "event_timestamp"]].head())


if __name__ == "__main__":
    main()
