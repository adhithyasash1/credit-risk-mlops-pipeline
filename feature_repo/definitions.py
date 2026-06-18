"""Feast feature definitions for the credit-risk model."""
import os
from datetime import timedelta

from feast import Entity, FeatureView, Field, BigQuerySource, ValueType
from feast.types import Float64, String

PROJECT_ID = os.getenv("PROJECT_ID", "credit-risk-mlops-0812")
BQ_DATASET = os.getenv("BQ_DATASET", "credit_risk")

application = Entity(
    name="application",
    join_keys=["application_id"],
    value_type=ValueType.INT64,
    description="A single credit application",
)

applications_source = BigQuerySource(
    name="applications_source",
    table=f"{PROJECT_ID}.{BQ_DATASET}.applications_fs",
    timestamp_field="event_timestamp",
)

credit_features = FeatureView(
    name="credit_features",
    entities=[application],
    ttl=timedelta(days=3650),
    online=True,
    source=applications_source,
    schema=[
        Field(name="checking_status", dtype=String),
        Field(name="duration", dtype=Float64),
        Field(name="credit_history", dtype=String),
        Field(name="purpose", dtype=String),
        Field(name="credit_amount", dtype=Float64),
        Field(name="savings_status", dtype=String),
        Field(name="employment", dtype=String),
        Field(name="installment_commitment", dtype=Float64),
        Field(name="personal_status", dtype=String),
        Field(name="other_parties", dtype=String),
        Field(name="residence_since", dtype=Float64),
        Field(name="property_magnitude", dtype=String),
        Field(name="age", dtype=Float64),
        Field(name="other_payment_plans", dtype=String),
        Field(name="housing", dtype=String),
        Field(name="existing_credits", dtype=Float64),
        Field(name="job", dtype=String),
        Field(name="num_dependents", dtype=Float64),
        Field(name="own_telephone", dtype=String),
        Field(name="foreign_worker", dtype=String),
    ],
)
