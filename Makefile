# Uses '>' as the recipe prefix so this file is tab-free (GNU Make 3.82+).
.RECIPEPREFIX = >
.PHONY: help test lint ingest features train explain feast-apply feast-materialize build deploy

help:            ## Show this help
> @grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-18s %s\n",$$1,$$2}'

test:            ## Run unit tests
> pytest -q

lint:            ## Lint source + tests
> ruff check --ignore E501 src tests

ingest:          ## Load dataset -> GCS + BigQuery
> python3 -m src.data.load_data

features:        ## Build the Feast feature table in BigQuery
> python3 -m src.data.build_feature_table

train:           ## Hyperparameter search + register champion (Optuna + MLflow)
> python3 -m src.training.tune --trials 30

explain:         ## Log global SHAP plots for the champion
> python3 -m src.training.explain

feast-apply:     ## Register Feast definitions
> cd feature_repo && feast apply

feast-materialize: ## Sync features offline -> online (Datastore)
> cd feature_repo && feast materialize-incremental "$$(date -u +%Y-%m-%dT%H:%M:%S)"

build:           ## Build + push the serving image
> gcloud builds submit --tag us-central1-docker.pkg.dev/$(PROJECT_ID)/mlops-docker/credit-serving:latest .

deploy:          ## Apply Kubernetes manifests
> kubectl apply -f k8s/
