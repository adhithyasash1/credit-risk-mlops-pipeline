FROM python:3.12-slim

# xgboost needs libgomp1 (OpenMP) at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Serving deps only (no MLflow -> no pyarrow conflict with Feast).
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ ./src/
COPY feature_repo/ ./feature_repo/

ENV PORT=8080
EXPOSE 8080
CMD exec uvicorn src.serving.app:app --host 0.0.0.0 --port ${PORT}
