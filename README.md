# Credit Risk MLOps Pipeline

An end-to-end, production-shaped **MLOps platform** built on an **open-source
stack running on Google Cloud**. It trains a credit-default risk model, tracks
every experiment, serves predictions with explanations behind an autoscaling
API, monitors the live service, and ships changes through CI/CD — with GCP used
only as the substrate so the skills transfer to any cloud.

**Dataset:** German Credit (`credit-g` / Statlog) — 1,000 applicants, 20
features, binary target (`1` = bad / default).

---

## What it does

- **Ingests** the dataset into Cloud Storage (raw) + BigQuery (queryable).
- **Versions data** with DVC (pointers in git, bytes in GCS).
- **Serves features** through a Feast feature store — BigQuery offline (training)
  and Datastore online (serving), eliminating training/serving skew.
- **Trains** an XGBoost model with Optuna hyperparameter search; logs params,
  metrics, the model, and SHAP explainability to **MLflow** (model registry with
  a `@champion` alias).
- **Serves** predictions via a **FastAPI** app (`/predict`, `/predict/{id}`
  fed by Feast online, `/explain` via SHAP) packaged in Docker.
- **Runs** on **GKE** with Workload Identity, a HorizontalPodAutoscaler, and a
  public LoadBalancer.
- **Observes** itself with Prometheus + Grafana (ops metrics + custom
  prediction metrics) and PrometheusRule alerts; load-tested with `wrk`.
- **Ships** via GitHub Actions CI/CD: lint + tests gate a keyless (Workload
  Identity Federation) build + deploy to GKE.

## Architecture
                     ┌─────────────── GitHub (source of truth) ───────────────┐
                     │  push -> Actions: lint+test -> build -> deploy (WIF)    │
                     └───────────────────────────┬────────────────────────────┘
                                                  ▼
DATA FEATURES TRAIN / TRACK SERVE OBSERVE
───── ──────── ───────────── ───── ───────
GCS (raw) ─▶ Feast Optuna + XGBoost ┌▶ FastAPI on GKE ─▶ Prometheus ─▶ Grafana
BigQuery ─▶ • offline=BigQuery -> MLflow (Cloud Run │ • /predict • ops metrics • dashboards
DVC (ver) • online=Datastore + Cloud SQL + GCS)│ • /predict/{id} • pred metrics • alerts
model.joblib ─▶ GCS ─┘ • /explain (SHAP) ▲
(Workload Identity) │ ServiceMonitor
│ PrometheusRule

## ML system design (lifecycle)
1. **Data** — `load_data.py` pulls `credit-g`, lands raw CSV in GCS, loads a
   typed table into BigQuery. DVC tracks the raw file by content hash.
2. **Features** — `build_feature_table.py` adds an entity key + event timestamp;
   Feast (`feature_repo/`) defines `credit_features` over it. Training reads
   point-in-time-correct features via `get_historical_features()`; serving reads
   the same definitions via `get_online_features()` from Datastore.
3. **Train** — one scikit-learn `Pipeline` (preprocessing + model) so transforms
   travel with the model. `tune.py` runs Optuna (TPE) over XGBoost, logs every
   trial to MLflow, registers the best model, and promotes it to `@champion`.
   `explain.py` logs global SHAP plots.
4. **Package** — the champion is exported to a stable GCS path; the serving image
   loads *that* (decoupled from MLflow to avoid dependency conflicts) and pins
   exact versions to prevent train/serve skew.
5. **Serve** — FastAPI validates input with Pydantic, scores, and explains.
   Runs on GKE; reads model (GCS) + features (Datastore) via Workload Identity.
6. **Monitor** — Prometheus scrapes app metrics via a ServiceMonitor; Grafana
   dashboards + Alertmanager (PrometheusRule) watch latency, throughput, and
   prediction mix. (Drift: PSI/KS/chi² — roadmap Phase 11.)
7. **Deliver** — GitHub Actions runs lint + tests, then builds and deploys to
   GKE on push, authenticating keylessly via Workload Identity Federation.
## Tech stack
| Concern | Tool |
|---|---|
| Data versioning | DVC (remote = GCS) |
| Feature store | Feast (offline = BigQuery, online = Datastore) |
| Experiment tracking + registry | MLflow (Cloud Run + Cloud SQL + GCS) |
| Hyperparameter tuning | Optuna |
| Explainability | SHAP |
| Packaging / registry | Docker → Artifact Registry |
| Serving | FastAPI + Uvicorn |
| Orchestration / runtime | GKE (Workload Identity, HPA) |
| Observability | Prometheus + Grafana (+ Alertmanager) |
| Load testing | wrk (Lua) |
| CI/CD | GitHub Actions (Workload Identity Federation) |
## Repository structure
.
├── config/config.yaml # single source of project-wide settings
├── feature_repo/ # Feast definitions + store config
├── src/
│ ├── data/ # ingestion + feature-table build
│ ├── features/ # preprocessing + Feast loader
│ ├── training/ # tune (Optuna) + explain (SHAP)
│ └── serving/ # FastAPI app
├── k8s/ # Deployment, Service, HPA, ServiceMonitor, alerts
├── mlflow-server/Dockerfile # MLflow tracking server image
├── loadtest/ # wrk scripts + saved reports
├── scripts/cloud_shell_permissions.sh # all Owner-level IAM/RBAC
├── .github/workflows/ci.yml # CI + CD pipeline
├── Dockerfile # serving image
├── Makefile # common dev tasks
└── requirements*.txt

## Quickstart (reproduce)
> Prerequisites: a GCP project with billing, `gcloud`, and a Vertex AI Workbench
> instance. Run code on the instance; run `scripts/cloud_shell_permissions.sh`
> Owner-level steps from Cloud Shell.
```bash
0. Owner-level permissions (Cloud Shell)
bash scripts/cloud_shell_permissions.sh

1. Data + features
make ingest features
cd feature_repo && feast apply &&
feast materialize-incremental "$(date -u +%Y-%m-%dT%H:%M:%S)" && cd ..

2. Train + register champion
make train explain

3. Build + deploy serving
make build
kubectl apply -f k8s/

4. Hit the API
IP=$(kubectl get svc credit-serving -n credit -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl -s http://$IP/predict/42

CI/CD then takes over: every push to `version-1` runs lint + tests and (keylessly)
builds and deploys the new image to GKE.
## Engineering & design notes
- **Architecture & design** — clean separation by lifecycle stage; serving is
  decoupled from the training/tracking stack (loads `model.joblib` from GCS, not
  via MLflow) so the two evolve independently and avoid dependency conflicts.
- **Code readability** — small single-purpose modules, type hints, docstrings;
  one `Pipeline` object carries preprocessing + model.
- **Performance** — model + SHAP explainer built once at startup, not per
  request; load-tested with wrk (CPU-bound at ~104 rps; HPA scales 2→5).
- **Memory efficiency** — tiny dataset; BigQuery does heavy joins server-side;
  the slim serving image installs only serving deps (no MLflow/Optuna).
- **Reliability** — Kubernetes startup/readiness/liveness probes; rolling
  updates keep old pods serving until new ones are healthy; CI gates deploys.
- **Resource management** — pinned CPU/memory requests+limits; Cloud Run + HPA
  scale on demand; documented teardown to stop idle spend.
- **Maintainability** — `Makefile` targets, unit tests + CI, pinned
  dependencies, infrastructure-as-code (k8s manifests, Dockerfiles, this script).
- **Configuration management** — `config/config.yaml` + env vars (`.env.example`);
  no secrets in code (Secret Manager + Workload Identity).
- **User-experience impact** — typed request validation (clear 422s),
  explainability endpoint for adverse-action reasons, auto OpenAPI docs (`/docs`).
## Cost management
GKE is the main cost. To pause (resume-friendly):
```bash
gcloud container clusters resize credit-mlops-cluster --node-pool default-pool --num-nodes 0 --zone us-central1-a --quiet
gcloud sql instances patch mlflow-db --activation-policy=NEVER
gcloud workbench instances stop credit-risk-workbench --location=us-central1-a

Resume by reversing (`--num-nodes 2`, `--activation-policy=ALWAYS`, `start`).
## Security notes (learning project)
Public LoadBalancers (API, Grafana) and a public MLflow UI are enabled for
convenience and are **not** production-safe; lock down with IAP/Ingress + auth
before using real data. Pod→GCP auth uses Workload Identity (no keys); CI→GCP
uses Workload Identity Federation (no keys).
## Roadmap
- [x] 0–10  foundations → data → models → DVC → Feast → Docker → FastAPI → GKE → observability
- [x] 12    CI/CD (GitHub Actions + Workload Identity Federation)
- [ ] 11    ML monitoring — PSI/KS/chi² drift + performance metrics → Prometheus
- [ ] 13    Orchestration + scheduled retraining (Cloud Scheduler)
