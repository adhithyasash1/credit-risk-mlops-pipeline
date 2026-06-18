#!/usr/bin/env bash
#
# cloud_shell_permissions.sh
# ---------------------------------------------------------------------------
# Every Owner-level grant that MUST run in Cloud Shell (or as a project Owner).
# The Workbench instance's service account has the Editor role, which can CREATE
# resources but cannot modify IAM policy or cluster-scoped RBAC — those need an
# Owner, hence this script.
#
# Idempotent: safe to re-run. Assumes the underlying resources already exist
# (Cloud SQL secret, MLflow Cloud Run service, GKE cluster, the `credit`
# namespace + `credit-ksa`). Run from Cloud Shell:
#     bash scripts/cloud_shell_permissions.sh
# ---------------------------------------------------------------------------
set -uo pipefail   # NOT -e: keep going so every grant is attempted

# ---- Configuration ----
PROJECT_ID="${PROJECT_ID:-credit-risk-mlops-0812}"
REGION="us-central1"
ZONE="us-central1-a"
CLUSTER="credit-mlops-cluster"
GITHUB_REPO="adhithyasash1/credit-risk-mlops-pipeline"
BUCKET="gs://${PROJECT_ID}-bucket"

gcloud config set project "$PROJECT_ID"
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
SERVING_SA="credit-serving-sa@${PROJECT_ID}.iam.gserviceaccount.com"
DEPLOY_SA="github-deployer@${PROJECT_ID}.iam.gserviceaccount.com"

echo "==> [1/6] Enable required APIs"
gcloud services enable \
  compute.googleapis.com aiplatform.googleapis.com notebooks.googleapis.com \
  storage.googleapis.com bigquery.googleapis.com artifactregistry.googleapis.com \
  cloudbuild.googleapis.com run.googleapis.com sqladmin.googleapis.com \
  secretmanager.googleapis.com cloudresourcemanager.googleapis.com \
  firestore.googleapis.com datastore.googleapis.com container.googleapis.com \
  iam.googleapis.com iamcredentials.googleapis.com

echo "==> [2/6] MLflow stack: compute SA -> Cloud SQL + secret; public Cloud Run UI"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${COMPUTE_SA}" --role="roles/cloudsql.client" --condition=None
gcloud secrets add-iam-policy-binding mlflow-db-pass \
  --member="serviceAccount:${COMPUTE_SA}" --role="roles/secretmanager.secretAccessor"
gcloud run services add-iam-policy-binding mlflow-server --region="$REGION" \
  --member="allUsers" --role="roles/run.invoker"   # learning only — public UI

echo "==> [3/6] Serving service account (Workload Identity for the GKE pod)"
gcloud iam service-accounts create credit-serving-sa --display-name="Credit serving app" 2>/dev/null
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SERVING_SA}" --role="roles/datastore.user" --condition=None
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SERVING_SA}" --role="roles/storage.objectViewer" --condition=None
# Feast registry read needs bucket metadata (storage.buckets.get) — bucket-scoped legacy role:
gcloud storage buckets add-iam-policy-binding "$BUCKET" \
  --member="serviceAccount:${SERVING_SA}" --role="roles/storage.legacyBucketReader"
# Bind the K8s SA (credit/credit-ksa) to the GSA:
gcloud iam service-accounts add-iam-policy-binding "${SERVING_SA}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="serviceAccount:${PROJECT_ID}.svc.id.goog[credit/credit-ksa]"

echo "==> [4/6] GitHub Actions deployer (keyless CD via Workload Identity Federation)"
gcloud iam service-accounts create github-deployer --display-name="GitHub Actions deployer" 2>/dev/null
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${DEPLOY_SA}" --role="roles/artifactregistry.writer" --condition=None
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${DEPLOY_SA}" --role="roles/container.developer" --condition=None
gcloud iam workload-identity-pools create github-pool \
  --location=global --display-name="GitHub pool" 2>/dev/null
gcloud iam workload-identity-pools providers create-oidc github-provider \
  --location=global --workload-identity-pool=github-pool --display-name="GitHub provider" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='${GITHUB_REPO}'" 2>/dev/null
gcloud iam service-accounts add-iam-policy-binding "${DEPLOY_SA}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/attribute.repository/${GITHUB_REPO}"

echo "==> [5/6] Cluster-admin RBAC for the Workbench instance SA (cluster-scoped)"
gcloud container clusters get-credentials "$CLUSTER" --zone "$ZONE"
kubectl create clusterrolebinding workbench-admin \
  --clusterrole=cluster-admin --user="${COMPUTE_SA}" 2>/dev/null

echo "==> [6/6] Done. All Owner-level permissions applied."
