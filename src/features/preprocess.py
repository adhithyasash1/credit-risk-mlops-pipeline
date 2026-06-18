"""Data loading and feature preprocessing for model training."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import load_config

__all__ = ["build_preprocessor", "load_config", "load_data_from_bq", "split_xy"]


def load_data_from_bq(cfg: dict) -> pd.DataFrame:
    from google.cloud import bigquery   # lazy: not needed for unit tests
    client = bigquery.Client(project=cfg["project_id"])
    table = f"{cfg['project_id']}.{cfg['bq_dataset']}.{cfg['bq_table']}"
    return client.query(f"SELECT * FROM `{table}`").to_dataframe()


def split_xy(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col!r} is not present in dataframe")
    return df.drop(columns=[target_col]), df[target_col]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    if X.empty or len(X.columns) == 0:
        raise ValueError("Cannot build a preprocessor without feature columns")

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
