#!/bin/bash
# ==============================================================
# setup_dagshub.sh
# Run this script AFTER:
#   1. Creating a GitHub repo (e.g., https://github.com/YOUR_USERNAME/ml-pipeline)
#   2. Creating a DagsHub repo connected to that GitHub repo
# ==============================================================

set -e

echo "=================================================="
echo "  ML Pipeline - DagsHub + DVC + MLflow Setup"
echo "=================================================="

# ---- USER CONFIGURATION (EDIT THESE) ----
GITHUB_USERNAME="Mohanned-Mahmoud"
DAGSHUB_USERNAME="Mohanned-Mahmoud"
REPO_NAME="ml-pipeline"
DAGSHUB_TOKEN="da9b6f877d84f56a7a7e0a4af53a6992c91bdf81"   # From DagsHub Settings > Tokens
# -----------------------------------------

GITHUB_REPO="https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"
DAGSHUB_REPO="https://dagshub.com/${DAGSHUB_USERNAME}/${REPO_NAME}.git"
DAGSHUB_REMOTE="https://dagshub.com/${DAGSHUB_USERNAME}/${REPO_NAME}.dvc"
MLFLOW_TRACKING_URI="https://dagshub.com/${DAGSHUB_USERNAME}/${REPO_NAME}.mlflow"

echo ""
echo "Step 1: Adding GitHub remote..."
git remote add origin "${GITHUB_REPO}" 2>/dev/null || git remote set-url origin "${GITHUB_REPO}"

echo "Step 2: Configuring DVC remote (DagsHub)..."
dvc remote add origin "${DAGSHUB_REMOTE}" --force
dvc remote modify origin --local auth basic
dvc remote modify origin --local user "${DAGSHUB_USERNAME}"
dvc remote modify origin --local password "${DAGSHUB_TOKEN}"
dvc remote default origin

echo "Step 3: Setting MLflow tracking URI..."
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}"
export MLFLOW_TRACKING_USERNAME="${DAGSHUB_USERNAME}"
export MLFLOW_TRACKING_PASSWORD="${DAGSHUB_TOKEN}"

# Persist env vars in .env file (gitignored)
cat > .env << ENVEOF
MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
MLFLOW_TRACKING_USERNAME=${DAGSHUB_USERNAME}
MLFLOW_TRACKING_PASSWORD=${DAGSHUB_TOKEN}
ENVEOF
echo "  MLflow env vars saved to .env (gitignored)"

echo "Step 4: Running DVC pipeline..."
dvc repro

echo "Step 5: Re-running pipeline with DagsHub MLflow tracking..."
source .env
python src/preprocess.py
python src/train.py
python src/tune.py

echo "Step 6: Committing to git..."
git add .
git commit -m "Initial ML pipeline: preprocess + train + tune with DVC & MLflow" || true

echo "Step 7: Pushing to GitHub..."
git branch -M main
git push -u origin main

echo "Step 8: Pushing DVC data to DagsHub..."
dvc push

echo ""
echo "=================================================="
echo "  ALL DONE!"
echo ""
echo "  GitHub repo:    ${GITHUB_REPO}"
echo "  DagsHub repo:   https://dagshub.com/${DAGSHUB_USERNAME}/${REPO_NAME}"
echo "  MLflow UI:      ${MLFLOW_TRACKING_URI}"
echo "=================================================="
