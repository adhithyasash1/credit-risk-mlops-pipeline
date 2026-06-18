"""
preprocess.py — load data from BigQuery and build the feature-transform pipeline.

The preprocessor auto-detects numeric vs categorical columns so it stays correct
even if the schema changes. It is returned UNFITTED; the training Pipeline fits it.
"""
import yaml
import pandas as pd
from google.cloud import bigquery
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_data_from_bq(cfg) -> pd.DataFrame:
    client = bigquery.Client(project=cfg["project_id"])
    table = f"{cfg['project_id']}.{cfg['bq_dataset']}.{cfg['bq_table']}"
    return client.query(f"SELECT * FROM `{table}`").to_dataframe()


def split_xy(df: pd.DataFrame, target_col: str):
    return df.drop(columns=[target_col]), df[target_col]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])
