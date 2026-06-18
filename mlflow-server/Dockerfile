FROM python:3.11-slim

RUN pip install --no-cache-dir \
    "mlflow>=2.16,<3" \
    psycopg2-binary \
    google-cloud-storage \
    gcsfs

CMD exec mlflow server \
    --host 0.0.0.0 \
    --port ${PORT:-8080} \
    --backend-store-uri "postgresql+psycopg2://${DB_USER}:${DB_PASS}@/${DB_NAME}?host=/cloudsql/${CLOUD_SQL_CONN}" \
    --artifacts-destination "${ARTIFACT_ROOT}" \
    --serve-artifacts
